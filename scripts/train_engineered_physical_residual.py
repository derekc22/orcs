#!/usr/bin/env python3
"""Learn compact whole-body feedback around the immutable physical expert.

The policy is a 200 Hz expert feedforward actor plus learned joint tracking and
root-state residuals.  Parameters are fit directly against exact MuJoCo
rollouts from a small deterministic perturbation set.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from finalize_engineered_expert import cube_orientation_error
from test_engineered_policy_robustness import PROFILES, perturbed_state


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs/engineering_policy"
MODEL_PATH = ROOT / "logs/engineering_locomomanipulation/engineered_locomomanipulation_expert_start0.mjb"
EXPERT_PATH = ROOT / "logs/engineering_locomomanipulation/engineered_release_retraction_v1.npz"
REFERENCE_PATH = OUT / "engineered_tracking_policy_v1.pt"
DT = 0.005
SUBSTEPS = 4
TASK_STEPS = 161 * SUBSTEPS

GROUP_TOKENS = {
    "hip_pitch": ("left_hip_pitch", "right_hip_pitch"),
    "knee": ("left_knee", "right_knee"),
    "ankle_pitch": ("left_ankle_pitch", "right_ankle_pitch"),
    "waist_pitch": ("waist_pitch",),
    "hip_roll": ("left_hip_roll", "right_hip_roll"),
    "ankle_roll": ("left_ankle_roll", "right_ankle_roll"),
    "waist_roll": ("waist_roll",),
}
SAGITTAL = ("hip_pitch", "knee", "ankle_pitch", "waist_pitch")
LATERAL = ("hip_roll", "ankle_roll", "waist_roll")


def quaternion_error(current: np.ndarray, reference: np.ndarray) -> np.ndarray:
    cur = Rotation.from_quat(np.r_[current[1:], current[0]])
    ref = Rotation.from_quat(np.r_[reference[1:], reference[0]])
    return (ref.inv() * cur).as_rotvec()


class ResidualRollout:
    def __init__(self, model: mujoco.MjModel, source: np.lib.npyio.NpzFile, checkpoint: dict):
        self.model = model
        self.source = source
        self.reference_qpos = np.asarray(checkpoint["reference_qpos"], dtype=float)
        self.reference_qvel = np.asarray(checkpoint["reference_qvel"], dtype=float)
        self.actuator_qpos = np.asarray(checkpoint["actuator_qpos"], dtype=int)
        self.actuator_dof = np.asarray(checkpoint["actuator_dof"], dtype=int)
        robot_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot/floating_base_joint")
        object_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object/floating_base_joint")
        self.robot_qpos = int(model.jnt_qposadr[robot_joint])
        self.robot_dof = int(model.jnt_dofadr[robot_joint])
        self.object_qpos = int(model.jnt_qposadr[object_joint])
        self.object_dof = int(model.jnt_dofadr[object_joint])
        self.torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot/torso_link")
        self.object_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object/object")
        self.object_geoms = {i for i in range(model.ngeom) if int(model.geom_bodyid[i]) == self.object_body}
        self.hand_geoms = [
            {
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"robot/{side}_hand_collision"),
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"robot/{side}_hand_front_guard_collision"),
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"robot/{side}_hand_trailing_guard_collision"),
            }
            for side in ("left", "right")
        ]
        names = []
        for actuator in range(model.nu):
            joint = int(model.actuator_trnid[actuator, 0])
            names.append((mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint) or "").split("/", 1)[-1])
        self.groups = {
            key: np.asarray([i for i, name in enumerate(names) if any(token in name for token in tokens)], dtype=int)
            for key, tokens in GROUP_TOKENS.items()
        }
        self.lower = model.actuator_ctrlrange[:, 0]
        self.upper = model.actuator_ctrlrange[:, 1]
        self.goal = source["goal_position"]
        goal_quat = source["goal_quaternion"]
        self.goal_rotation = Rotation.from_quat(np.r_[goal_quat[1:], goal_quat[0]])

    def run(self, parameters: np.ndarray, initial_qpos: np.ndarray, initial_qvel: np.ndarray, keep_controls: bool = False):
        kp, kd = parameters[:2]
        gains = parameters[2:].reshape(7, 4)
        data = mujoco.MjData(self.model)
        data.qpos[:] = initial_qpos
        data.qvel[:] = initial_qvel
        mujoco.mj_forward(self.model, data)
        initial_object = data.qpos[self.object_qpos:self.object_qpos + 3].copy()
        first_invalid = None
        controls = []
        task_object = None
        task_velocity = None
        task_hand_contact = None
        max_lift = -np.inf
        supported_run = 0
        longest_supported = 0
        supported_first = None
        supported_last = None
        root_tracking_cost = 0.0
        for step, feedforward in enumerate(self.source["ctrl_physics"]):
            reference_qpos = self.reference_qpos[step]
            reference_qvel = self.reference_qvel[step]
            joint_error = data.qpos[self.actuator_qpos] - reference_qpos[self.actuator_qpos]
            joint_velocity_error = data.qvel[self.actuator_dof] - reference_qvel[self.actuator_dof]
            control = feedforward - kp * joint_error - kd * joint_velocity_error
            ref_quat = reference_qpos[self.robot_qpos + 3:self.robot_qpos + 7]
            ref_rotation = Rotation.from_quat(np.r_[ref_quat[1:], ref_quat[0]])
            local_position = ref_rotation.inv().apply(
                data.qpos[self.robot_qpos:self.robot_qpos + 3] - reference_qpos[self.robot_qpos:self.robot_qpos + 3]
            )
            local_velocity = ref_rotation.inv().apply(
                data.qvel[self.robot_dof:self.robot_dof + 3] - reference_qvel[self.robot_dof:self.robot_dof + 3]
            )
            local_rotation = quaternion_error(
                data.qpos[self.robot_qpos + 3:self.robot_qpos + 7], ref_quat
            )
            local_angular = ref_rotation.inv().apply(
                data.qvel[self.robot_dof + 3:self.robot_dof + 6] - reference_qvel[self.robot_dof + 3:self.robot_dof + 6]
            )
            sagittal = np.asarray([local_position[0], local_velocity[0], local_rotation[1], local_angular[1]])
            lateral = np.asarray([local_position[1], local_velocity[1], local_rotation[0], local_angular[0]])
            delta = np.zeros(self.model.nu)
            row = 0
            for key in SAGITTAL:
                delta[self.groups[key]] += float(gains[row] @ sagittal)
                row += 1
            for key in LATERAL:
                delta[self.groups[key]] += float(gains[row] @ lateral)
                row += 1
            control += np.clip(delta, -0.35, 0.35)
            control = np.clip(control, self.lower, self.upper)
            if keep_controls:
                controls.append(control.copy())
            data.ctrl[:] = control
            mujoco.mj_step(self.model, data)
            root_tracking_cost += float(local_position @ local_position + 0.2 * local_rotation @ local_rotation)
            if step % SUBSTEPS != SUBSTEPS - 1:
                continue
            frame = step // SUBSTEPS
            height = float(data.qpos[self.robot_qpos + 2])
            up = data.xmat[self.torso].reshape(3, 3)[:, 2]
            tilt = math.degrees(math.acos(float(np.clip(up[2], -1.0, 1.0))))
            if first_invalid is None and (height < 0.45 or tilt > 50.0):
                first_invalid = frame
            hand_contact = False
            for ci in range(data.ncon):
                pair = {int(data.contact[ci].geom1), int(data.contact[ci].geom2)}
                hand_contact |= bool(pair & self.object_geoms) and any(bool(pair & geoms) for geoms in self.hand_geoms)
            object_position = data.qpos[self.object_qpos:self.object_qpos + 3].copy()
            lift = float(object_position[2] - initial_object[2])
            max_lift = max(max_lift, lift)
            supported = hand_contact and lift > 0.05
            supported_run = supported_run + 1 if supported else 0
            longest_supported = max(longest_supported, supported_run)
            if supported:
                supported_first = object_position.copy() if supported_first is None else supported_first
                supported_last = object_position.copy()
            if step == TASK_STEPS - 1:
                task_object = data.qpos[self.object_qpos:self.object_qpos + 7].copy()
                task_velocity = data.qvel[self.object_dof:self.object_dof + 6].copy()
                task_hand_contact = hand_contact
        assert task_object is not None and task_velocity is not None
        rotation = Rotation.from_quat(np.r_[task_object[4:], task_object[3]])
        position_error = float(np.linalg.norm(task_object[:3] - self.goal))
        orientation_error = cube_orientation_error(rotation, self.goal_rotation)
        object_speed = float(np.linalg.norm(task_velocity))
        carry = 0.0 if supported_first is None else float(np.linalg.norm(supported_last[:2] - supported_first[:2]))
        task_valid = first_invalid is None or first_invalid >= TASK_STEPS // SUBSTEPS
        continuation_valid = first_invalid is None
        strict = bool(
            task_valid and continuation_valid and max_lift > 0.05 and longest_supported >= 25
            and carry > 0.5 and position_error < 0.15 and orientation_error < 0.35
            and object_speed < 0.25 and not task_hand_contact
        )
        cost = (
            1800.0 * (not task_valid) + 1200.0 * (not continuation_valid)
            + 2000.0 * max(0.0, 0.05 - max_lift) ** 2
            + 30.0 * max(0, 25 - longest_supported)
            + 400.0 * max(0.0, 0.5 - carry) ** 2
            + 150.0 * position_error + 50.0 * orientation_error + 8.0 * object_speed
            + 0.002 * root_tracking_cost
        )
        metrics = {
            "strict_success": strict,
            "task_valid": task_valid,
            "continuation_valid": continuation_valid,
            "first_invalid_frame": first_invalid,
            "maximum_lift_m": max_lift,
            "supported_lift_frames": longest_supported,
            "carried_displacement_m": carry,
            "position_error_m": position_error,
            "orientation_error_rad": orientation_error,
            "object_speed": object_speed,
            "cost": cost,
        }
        return metrics, np.asarray(controls)


def rank(results: list[dict]) -> tuple:
    return (
        -sum(item["strict_success"] for item in results),
        -sum(item["continuation_valid"] for item in results),
        -sum(item["task_valid"] for item in results),
        max(item["cost"] for item in results),
        sum(item["cost"] for item in results),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument("--population", type=int, default=40)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    import torch

    model = mujoco.MjModel.from_binary_path(str(MODEL_PATH))
    model.opt.timestep = DT
    source = np.load(EXPERT_PATH)
    checkpoint = torch.load(REFERENCE_PATH, map_location="cpu", weights_only=False)
    rollout = ResidualRollout(model, source, checkpoint)
    training_states = [(source["initial_qpos"].copy(), source["initial_qvel"].copy())]
    for trial in range(args.trials):
        training_states.append(perturbed_state(model, source, checkpoint, 20261021 + trial, PROFILES["mild"]))
    rng = np.random.default_rng(20261021)
    mean = np.r_[1.28, 0.096, np.zeros(28)]
    std = np.r_[0.35, 0.035, np.full(28, 0.65)]
    lower = np.r_[0.0, 0.0, np.full(28, -5.0)]
    upper = np.r_[2.5, 0.25, np.full(28, 5.0)]
    best_parameters = mean.copy()
    best_results = [rollout.run(best_parameters, q, v)[0] for q, v in training_states]
    best_rank = rank(best_results)
    history = []
    for generation in range(args.generations):
        population = np.clip(rng.normal(mean, std, (args.population, len(mean))), lower, upper)
        population[0] = best_parameters
        evaluated = []
        for candidate in population:
            results = [rollout.run(candidate, q, v)[0] for q, v in training_states]
            evaluated.append((rank(results), results))
        order = sorted(range(len(population)), key=lambda index: evaluated[index][0])
        winner = order[0]
        if evaluated[winner][0] < best_rank:
            best_parameters = population[winner].copy()
            best_rank, best_results = evaluated[winner]
        elite = population[order[:max(4, args.population // 5)]]
        mean = 0.75 * elite.mean(axis=0) + 0.25 * mean
        std = np.maximum(0.75 * elite.std(axis=0) + 0.25 * std, np.r_[0.02, 0.004, np.full(28, 0.04)])
        item = {
            "generation": generation,
            "training_successes": sum(x["strict_success"] for x in best_results),
            "training_trials": len(best_results),
            "continuation_valid": sum(x["continuation_valid"] for x in best_results),
            "worst_cost": max(x["cost"] for x in best_results),
            "worst_position_error_m": max(x["position_error_m"] for x in best_results),
            "worst_orientation_error_rad": max(x["orientation_error_rad"] for x in best_results),
        }
        history.append(item)
        print("GEN", json.dumps(item), flush=True)
        np.savez_compressed(
            OUT / "engineered_physical_residual_checkpoint.npz",
            parameters=best_parameters, mean=mean, std=std, generation=generation,
        )
    nominal, controls = rollout.run(
        best_parameters, source["initial_qpos"], source["initial_qvel"], keep_controls=True
    )
    artifact = OUT / "engineered_physical_residual_policy_v1.npz"
    np.savez_compressed(
        artifact,
        parameters=best_parameters,
        ctrl_physics_nominal=controls,
        initial_qpos=source["initial_qpos"],
        initial_qvel=source["initial_qvel"],
        goal_position=source["goal_position"],
        goal_quaternion=source["goal_quaternion"],
        joint_tracking_gain=best_parameters[:2],
        root_feedback_gains=best_parameters[2:].reshape(7, 4),
    )
    report = {
        "method": "exact-physics CEM learning of compact whole-body residual around frozen expert",
        "artifact": str(artifact.relative_to(ROOT)),
        "training_trials": len(training_states),
        "nominal": nominal,
        "training_results": best_results,
        "history": history,
    }
    (OUT / "physical_residual_training_report_v1.json").write_text(json.dumps(report, indent=2) + "\n")
    print("FINAL", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
