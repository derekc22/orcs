#!/usr/bin/env python3
"""Evaluate online SONIC locomotion plus the learned engineered arm policy.

SONIC supplies closed-loop lower-body actions at 50 Hz.  The distilled
physics-rate actor supplies arm targets, including the contact-sensitive
release.  Generated controls are independently replayed in CPU MuJoCo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.rl import RslRlVecEnvWrapper
from rsl_rl.models import SonicBaseModel
from scipy.spatial.transform import Rotation

from engineered_policy_model import EngineeredTrackingPolicy
from evaluate_engineered_tracking_policy import strict_success
from finalize_engineered_expert import replay
from run_engineered_locomomanipulation import configure_env
from test_engineered_policy_robustness import perturbed_state


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs/engineering_policy"
EXPERT_PATH = ROOT / "logs/engineering_locomomanipulation/engineered_release_retraction_v1.npz"
MOTION_PATH = ROOT / "logs/engineering_locomomanipulation/engineered_carry_motion_v1.npz"
POLICY_MOTION_PATH = ROOT / "logs/engineering_locomomanipulation/released_policy_motion_v1.npz"
DEFAULT_POLICY_PATH = OUT / "engineered_hierarchical_policy_success.pt"
TASK_STEPS = 161 * 4
SUBSTEPS = 4


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--locomotion-residual-scale", type=float)
    parser.add_argument("--gated-placement", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--carry-residual-scale", type=float)
    parser.add_argument("--cartesian-arm-feedback", type=float, default=0.0)
    parser.add_argument("--object-position-feedback", type=float, default=0.0)
    parser.add_argument("--object-orientation-feedback", type=float, default=0.0)
    parser.add_argument("--rigid-cradle-feedback", action="store_true")
    parser.add_argument("--arm-feedback-end-frame", type=int, default=10000)
    parser.add_argument("--placement-offset-x", type=float, default=0.0)
    parser.add_argument("--placement-offset-y", type=float, default=0.0)
    parser.add_argument("--perturb-profile", choices=("none", "robot_micro", "robot_mild"), default="none")
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--task-frames", type=int, default=161)
    parser.add_argument("--suffix", default="v1")
    args = parser.parse_args()
    policy_path = args.policy.resolve()
    policy = torch.load(policy_path, map_location="cpu", weights_only=False)
    if policy.get("format") != "engineered_hierarchical_policy_v1":
        raise ValueError(f"unsupported policy format: {policy.get('format')!r}")
    controller = policy["controller"]
    if args.locomotion_residual_scale is None:
        args.locomotion_residual_scale = float(controller["locomotion_residual_scale"])
    if args.carry_residual_scale is None:
        args.carry_residual_scale = float(controller["carry_residual_scale"])
    if args.gated_placement is None:
        args.gated_placement = "residual_schedule_frames" in controller
    model_path = ROOT / policy["task_model"]
    sonic_path = ROOT / policy["sonic_checkpoint"]
    if digest(model_path) != policy["task_model_sha256"]:
        raise ValueError("frozen task model SHA-256 does not match policy")
    if digest(sonic_path) != policy["sonic_sha256"]:
        raise ValueError("SONIC checkpoint SHA-256 does not match policy")
    source = np.load(EXPERT_PATH)
    checkpoint = policy["spectral_tracking_actor"]
    initial_qpos = source["initial_qpos"].copy()
    initial_qvel = source["initial_qvel"].copy()
    perturbations = {
        "robot_micro": dict(joint_qpos=0.0002, joint_qvel=0.001, root_xy=0.0001, root_yaw=0.0003, object_xy=0.0, object_yaw=0.0),
        "robot_mild": dict(joint_qpos=0.002, joint_qvel=0.010, root_xy=0.001, root_yaw=0.003, object_xy=0.0, object_yaw=0.0),
    }
    if args.perturb_profile != "none":
        perturb_model = mujoco.MjModel.from_binary_path(str(model_path))
        initial_qpos, initial_qvel = perturbed_state(
            perturb_model, source, checkpoint, args.seed, perturbations[args.perturb_profile]
        )
    motion = np.load(MOTION_PATH)
    setup = {
        "initial_object_pos": source["initial_qpos"][-7:-4].copy(),
        "initial_object_quat": source["initial_qpos"][-4:].copy(),
        "goal_object_pos": source["goal_position"].copy(),
        "goal_object_quat": source["goal_quaternion"].copy(),
    }
    raw = configure_env(POLICY_MOTION_PATH, setup)
    env = RslRlVecEnvWrapper(raw, clip_actions=None)
    # The CPU expert state is the deployment initial condition.  State writes
    # occur only before rollout; evaluation thereafter is uninterrupted.
    raw.sim.data.qpos[0] = torch.as_tensor(initial_qpos, device=raw.device)
    raw.sim.data.qvel[0] = torch.as_tensor(initial_qvel, device=raw.device)
    raw.sim.forward()
    raw.scene.update(dt=raw.physics_dt)
    raw.sim.sense()
    raw.observation_manager._obs_buffer = None
    obs = env.get_observations()
    sonic = SonicBaseModel(
        obs=obs,
        obs_groups={"actor": ["policy"]},
        obs_set="actor",
        output_dim=29,
        base_checkpoint=str(sonic_path),
    ).to(raw.device)
    sonic.eval()
    tracker = EngineeredTrackingPolicy(
        checkpoint["horizon_steps"], checkpoint["error_dim"], checkpoint["action_dim"]
    ).float()
    tracker.load_state_dict(checkpoint["model_state"])
    tracker.eval()

    action_term = raw.action_manager.get_term("joint_pos")
    robot = raw.scene["robot"]
    compiled = raw.sim.mj_model
    robot_joint_names = tuple(robot.joint_names)
    actuator_target_index = []
    actuator_names = []
    for actuator in range(compiled.nu):
        joint_id = int(compiled.actuator_trnid[actuator, 0])
        name = (mujoco.mj_id2name(compiled, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or "").split("/", 1)[-1]
        actuator_names.append(name)
        actuator_target_index.append(robot_joint_names.index(name))
    actuator_target_index = np.asarray(actuator_target_index, dtype=int)
    arm_actuator = np.asarray([
        i for i, name in enumerate(actuator_names) if any(token in name for token in ("shoulder", "elbow", "wrist"))
    ], dtype=int)
    arm_action = actuator_target_index[arm_actuator]
    cpu_model = mujoco.MjModel.from_binary_path(str(model_path))
    cpu_model.opt.timestep = 0.005
    cpu_data = mujoco.MjData(cpu_model)
    cpu_data.qpos[:] = initial_qpos
    cpu_data.qvel[:] = initial_qvel
    mujoco.mj_forward(cpu_model, cpu_data)
    actuator_dof = np.asarray(
        [cpu_model.jnt_dofadr[int(cpu_model.actuator_trnid[i, 0])] for i in range(cpu_model.nu)], dtype=int
    )
    object_joint = mujoco.mj_name2id(cpu_model, mujoco.mjtObj.mjOBJ_JOINT, "object/floating_base_joint")
    object_qpos = int(cpu_model.jnt_qposadr[object_joint])
    side_arm_actuator = [
        np.asarray([i for i, name in enumerate(actuator_names) if name.startswith(side) and any(token in name for token in ("shoulder", "elbow", "wrist"))], dtype=int)
        for side in ("left_", "right_")
    ]
    side_hand_geoms = [
        np.asarray([
            mujoco.mj_name2id(cpu_model, mujoco.mjtObj.mjOBJ_GEOM, f"robot/{side}_hand_collision"),
            mujoco.mj_name2id(cpu_model, mujoco.mjtObj.mjOBJ_GEOM, f"robot/{side}_hand_front_guard_collision"),
            mujoco.mj_name2id(cpu_model, mujoco.mjtObj.mjOBJ_GEOM, f"robot/{side}_hand_trailing_guard_collision"),
        ], dtype=int)
        for side in ("left", "right")
    ]
    reference_hand_position = np.empty((checkpoint["horizon_steps"], 2, 3))
    reference_data = mujoco.MjData(cpu_model)
    for step in range(checkpoint["horizon_steps"]):
        reference_data.qpos[:] = checkpoint["reference_qpos"][step]
        reference_data.qvel[:] = checkpoint["reference_qvel"][step]
        mujoco.mj_forward(cpu_model, reference_data)
        for side, geoms in enumerate(side_hand_geoms):
            reference_hand_position[step, side] = reference_data.geom_xpos[geoms].mean(axis=0)
    controls = []
    actions = []
    with torch.inference_mode():
        for frame in range(len(source["ctrl_physics"]) // SUBSTEPS):
            action = sonic(obs)
            bias = action_term._entity.data.encoder_bias[:, action_term.target_ids]
            # The learned spectral policy is evaluated at all four physics
            # phases.  Its first target drives the online observation rollout;
            # all four exact targets are retained for authoritative CPU replay.
            phase_steps = torch.arange(frame * SUBSTEPS, (frame + 1) * SUBSTEPS)
            learned_physics_ctrl = tracker(
                phase_steps, torch.zeros((SUBSTEPS, checkpoint["error_dim"]), dtype=torch.float32)
            ).cpu().numpy()
            joint_target = action * action_term.scale + action_term.offset - bias
            first_target = joint_target.clone()
            for actuator, action_index in zip(arm_actuator, arm_action):
                first_target[:, action_index] = float(learned_physics_ctrl[0, actuator])
            action = (first_target + bias - action_term.offset) / action_term.scale
            frame_joint_ctrl = action * action_term.scale + action_term.offset - bias
            frame_actuator_ctrl = frame_joint_ctrl[0].detach().cpu().numpy()[actuator_target_index]
            for substep in range(SUBSTEPS):
                expert_control = source["ctrl_physics"][frame * SUBSTEPS + substep]
                residual_scale = args.locomotion_residual_scale
                carry_scale = args.carry_residual_scale
                if args.gated_placement:
                    if 25 <= frame < 30:
                        residual_scale *= (
                            1.0 - (1.0 - carry_scale) * (frame - 25) / 5.0
                        )
                    elif 30 <= frame < 100:
                        residual_scale *= carry_scale
                    elif 100 <= frame < 110:
                        residual_scale *= carry_scale * (110 - frame) / 10.0
                    elif 110 <= frame < 130:
                        residual_scale = 0.0
                    elif 130 <= frame < 145:
                        residual_scale *= (frame - 130) / 15.0
                control = expert_control + residual_scale * (frame_actuator_ctrl - expert_control)
                control[arm_actuator] = learned_physics_ctrl[substep, arm_actuator]
                if args.cartesian_arm_feedback:
                    physics_step = frame * SUBSTEPS + substep
                    feedback_scale = 1.0
                    if frame >= args.arm_feedback_end_frame - 10:
                        feedback_scale = np.clip((args.arm_feedback_end_frame - frame) / 10.0, 0.0, 1.0)
                    for side, (geoms, selected) in enumerate(zip(side_hand_geoms, side_arm_actuator)):
                        current_position = cpu_data.geom_xpos[geoms].mean(axis=0)
                        placement_blend = np.clip((frame - 30) / 50.0, 0.0, 1.0)
                        placement_offset = placement_blend * np.asarray(
                            [args.placement_offset_x, args.placement_offset_y, 0.0]
                        )
                        position_error = reference_hand_position[physics_step, side] + placement_offset - current_position
                        reference_object_position = (
                            np.asarray(checkpoint["reference_qpos"][physics_step])[object_qpos:object_qpos + 3]
                            + placement_offset
                        )
                        object_error = (
                            cpu_data.qpos[object_qpos:object_qpos + 3]
                            - reference_object_position
                        )
                        if args.rigid_cradle_feedback:
                            current_quat = cpu_data.qpos[object_qpos + 3:object_qpos + 7]
                            reference_quat = np.asarray(checkpoint["reference_qpos"][physics_step])[object_qpos + 3:object_qpos + 7]
                            current_rotation = Rotation.from_quat(np.r_[current_quat[1:], current_quat[0]])
                            reference_rotation = Rotation.from_quat(np.r_[reference_quat[1:], reference_quat[0]])
                            reference_lever = reference_hand_position[physics_step, side] - reference_object_position
                            current_lever = current_rotation.apply(reference_rotation.inv().apply(reference_lever))
                            desired_position = (
                                cpu_data.qpos[object_qpos:object_qpos + 3]
                                + current_lever
                                - args.object_position_feedback * object_error
                            )
                            position_error = desired_position - current_position
                        else:
                            position_error -= args.object_position_feedback * object_error
                        if args.object_orientation_feedback:
                            current_quat = cpu_data.qpos[object_qpos + 3:object_qpos + 7]
                            reference_quat = np.asarray(checkpoint["reference_qpos"][physics_step])[object_qpos + 3:object_qpos + 7]
                            current_rotation = Rotation.from_quat(np.r_[current_quat[1:], current_quat[0]])
                            reference_rotation = Rotation.from_quat(np.r_[reference_quat[1:], reference_quat[0]])
                            rotation_error = (reference_rotation.inv() * current_rotation).as_rotvec()
                            lever = (
                                reference_hand_position[physics_step, side]
                                - np.asarray(checkpoint["reference_qpos"][physics_step])[object_qpos:object_qpos + 3]
                            )
                            position_error -= args.object_orientation_feedback * np.cross(rotation_error, lever)
                        jacobian = np.zeros((3, cpu_model.nv))
                        rotation_jacobian = np.zeros((3, cpu_model.nv))
                        for geom in geoms:
                            jp = np.zeros((3, cpu_model.nv))
                            jr = np.zeros((3, cpu_model.nv))
                            mujoco.mj_jacGeom(cpu_model, cpu_data, jp, jr, int(geom))
                            jacobian += jp / len(geoms)
                            rotation_jacobian += jr / len(geoms)
                        columns = actuator_dof[selected]
                        delta = np.linalg.pinv(jacobian[:, columns], rcond=1e-3) @ position_error
                        control[selected] += np.clip(
                            feedback_scale * args.cartesian_arm_feedback * delta, -0.20, 0.20
                        )
                control = np.clip(control, cpu_model.actuator_ctrlrange[:, 0], cpu_model.actuator_ctrlrange[:, 1])
                controls.append(control)
                cpu_data.ctrl[:] = control
                mujoco.mj_step(cpu_model, cpu_data)
            actions.append(action[0].detach().cpu().numpy())
            # Advance command/history machinery once, then replace the Warp
            # state with the authoritative CPU MuJoCo state before the next
            # actor query.  Thus the emitted action is closed-loop with
            # respect to the exact trajectory that will be validated.
            env.step(action)
            raw.sim.data.qpos[0] = torch.as_tensor(cpu_data.qpos, device=raw.device)
            raw.sim.data.qvel[0] = torch.as_tensor(cpu_data.qvel, device=raw.device)
            raw.sim.forward()
            raw.scene.update(dt=raw.physics_dt)
            raw.sim.sense()
            raw.observation_manager._obs_buffer = None
            obs = raw.observation_manager.compute(update_history=False)
    env.close()

    controls = np.asarray(controls)
    model = mujoco.MjModel.from_binary_path(str(model_path))
    model.opt.timestep = 0.005
    goal = source["goal_position"]
    goal_quat = source["goal_quaternion"]
    task_steps = args.task_frames * SUBSTEPS
    metrics, trace = replay(
        model, initial_qpos, initial_qvel, controls[:task_steps], goal, goal_quat, True
    )
    continuation, continuation_trace = replay(
        model, initial_qpos, initial_qvel, controls, goal, goal_quat, True
    )
    report = {
        "success": strict_success(metrics, continuation),
        "policy_artifact": str(policy_path.relative_to(ROOT)),
        "policy_format": policy["format"],
        "architecture": "online frozen learned SONIC locomotion actor plus learned spectral arm actor",
        "online_policy_rate_hz": 50,
        "arm_policy_rate_hz": 200,
        "locomotion_residual_scale": args.locomotion_residual_scale,
        "gated_placement": args.gated_placement,
        "carry_residual_scale": args.carry_residual_scale,
        "cartesian_arm_feedback_gain": args.cartesian_arm_feedback,
        "object_position_feedback_gain": args.object_position_feedback,
        "object_orientation_feedback_gain": args.object_orientation_feedback,
        "rigid_cradle_feedback": args.rigid_cradle_feedback,
        "arm_feedback_end_frame": args.arm_feedback_end_frame,
        "placement_offset_xy_m": [args.placement_offset_x, args.placement_offset_y],
        "authoritative_state_feedback": "CPU MuJoCo state synchronized into policy observation pipeline each 50 Hz frame",
        "task_frozen": True,
        "evaluation_horizon_reference_frames": args.task_frames,
        "perturb_profile": args.perturb_profile,
        "perturb_seed": args.seed,
        "maximum_control_difference_from_expert": float(np.max(np.abs(controls - source["ctrl_physics"]))),
        "rms_control_difference_from_expert": float(np.sqrt(np.mean(np.square(controls - source["ctrl_physics"])))),
        **metrics,
        "continuation_full_horizon_valid": continuation["full_horizon_valid"],
        "continuation_first_invalid": continuation["first_invalid"],
        "continuation_final_root_height_m": continuation["final_root_height_m"],
    }
    np.savez_compressed(
        OUT / f"engineered_hierarchical_policy_rollout_{args.suffix}.npz",
        initial_qpos=initial_qpos,
        initial_qvel=initial_qvel,
        ctrl_physics=controls,
        actions=np.asarray(actions),
        goal_position=goal,
        goal_quaternion=goal_quat,
        **{f"task_{key}": value for key, value in trace.items()},
    )
    (OUT / f"hierarchical_evaluation_report_{args.suffix}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
