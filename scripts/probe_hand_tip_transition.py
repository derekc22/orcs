#!/usr/bin/env python3
"""Search a hand-only box-tip transition from the stable upright stance.

The box is fully dynamic.  Candidate arm targets are evaluated exclusively by
MuJoCo and any meaningful non-hand robot/object impulse is invalid.  This
probe tests whether the raised-edge support mode can be reached legally rather
than teleporting the box into a tipped IK pose.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
from probe_flat_hand_grasp import build, smoothstep
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
GROUPS = (
    "left_hand",
    "right_hand",
    "wrist",
    "feet",
    "legs",
    "torso_pelvis",
    "other_robot",
)


def quat_rotation(quaternion: np.ndarray) -> Rotation:
    return Rotation.from_quat(np.r_[quaternion[1:], quaternion[0]])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--population", type=int, default=56)
    parser.add_argument("--generations", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=150)
    parser.add_argument("--anchor-step", type=int, default=800)
    parser.add_argument(
        "--stance",
        type=Path,
        default=ROOT
        / "logs/hand_manipulation_research/stable_upright_stance_v1.npz",
    )
    parser.add_argument("--maximum-disallowed-impulse", type=float, default=2.0)
    args = parser.parse_args()

    engine, generator = build(0.60)
    model = engine.model
    stance = np.load(args.stance)
    row = stance["trace"][args.anchor_step]
    initial_qpos = row[: model.nq].copy()
    initial_qvel = row[model.nq : model.nq + model.nv].copy()
    stable_target = stance["target"].copy()
    balance = np.asarray(generator.balance_actuators, dtype=int)
    arms = np.setdiff1d(np.arange(model.nu), balance)
    actuator_dofs = np.asarray(
        [model.jnt_dofadr[joint] for joint in model.actuator_trnid[:, 0]], dtype=int
    )
    arm_dofs = actuator_dofs[arms]
    object_geoms = set(map(int, engine.object_geoms))
    initial = mujoco.MjData(model)
    initial.qpos[:] = initial_qpos
    initial.qvel[:] = initial_qvel
    mujoco.mj_forward(model, initial)
    initial_hands = np.asarray(
        [initial.geom_xpos[geom].copy() for geom in engine.hand_geoms]
    )
    initial_object = initial.qpos[engine.object_qpos : engine.object_qpos + 3].copy()
    initial_object_rotation = quat_rotation(
        initial.qpos[engine.object_qpos + 3 : engine.object_qpos + 7]
    )
    object_vertices = []
    for geom in engine.object_geoms:
        mesh = int(model.geom_dataid[geom])
        begin, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        local = model.mesh_vert[begin : begin + count]
        object_vertices.append(
            local @ initial.geom_xmat[geom].reshape(3, 3).T
            + initial.geom_xpos[geom]
        )
    vertices = np.concatenate(object_vertices)
    lower_box, upper_box = vertices.min(0), vertices.max(0)
    centre_x = 0.5 * (lower_box[0] + upper_box[0])
    balance_frames = np.asarray([0.0, 60.0, float(args.horizon - 1)])

    def balance_residual(frame: float, parameters: np.ndarray) -> np.ndarray:
        knots = np.vstack(
            [np.zeros(len(balance)), parameters[8:].reshape(2, len(balance))]
        )
        if frame >= balance_frames[-1]:
            return knots[-1]
        segment = int(np.searchsorted(balance_frames, frame, side="right") - 1)
        amount = smoothstep(
            (frame - balance_frames[segment])
            / (balance_frames[segment + 1] - balance_frames[segment])
        )
        return (1.0 - amount) * knots[segment] + amount * knots[segment + 1]

    def group(geom: int) -> str:
        if geom == int(engine.hand_geoms[0]):
            return "left_hand"
        if geom == int(engine.hand_geoms[1]):
            return "right_hand"
        body = int(model.geom_bodyid[geom])
        body_name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
        )
        if "wrist" in body_name:
            return "wrist"
        if "foot" in body_name or "ankle" in body_name:
            return "feet"
        if any(token in body_name for token in ("knee", "shin", "hip", "thigh")):
            return "legs"
        if any(token in body_name for token in ("torso", "pelvis", "waist")):
            return "torso_pelvis"
        return "other_robot"

    def arm_target(
        data: mujoco.MjData,
        desired_positions: np.ndarray,
        gain: float,
    ) -> np.ndarray:
        jacobian = np.zeros((6, len(arms)))
        error = np.zeros(6)
        for hand_index, geom in enumerate(engine.hand_geoms):
            jacp = np.zeros((3, model.nv))
            mujoco.mj_jacGeom(model, data, jacp, None, int(geom))
            begin = 3 * hand_index
            jacobian[begin : begin + 3] = jacp[:, arm_dofs]
            error[begin : begin + 3] = (
                desired_positions[hand_index] - data.geom_xpos[geom]
            )
        current = data.qpos[engine.actuator_qpos[arms]]
        posture = stable_target[arms] - current
        system = jacobian.T @ jacobian + 0.025 * np.eye(len(arms))
        rhs = jacobian.T @ error + 0.025 * posture
        delta = np.linalg.solve(system, rhs)
        return current + np.clip(gain * delta, -0.07, 0.07)

    def evaluate(parameters: np.ndarray, record: bool = False):
        # x spread, contact height, front offset, approach duration, push
        # distance, vertical push, push duration, IK gain.
        spread, height_fraction, front, approach, push, vertical, duration, gain = (
            parameters[:8]
        )
        contact = np.asarray(
            [
                [centre_x + spread, upper_box[1] + front, lower_box[2] + height_fraction * (upper_box[2] - lower_box[2])],
                [centre_x - spread, upper_box[1] + front, lower_box[2] + height_fraction * (upper_box[2] - lower_box[2])],
            ]
        )
        pushed = contact + np.asarray([0.0, -push, vertical])
        data = mujoco.MjData(model)
        data.qpos[:] = initial_qpos
        data.qvel[:] = initial_qvel
        mujoco.mj_forward(model, data)
        impulses = {key: 0.0 for key in GROUPS}
        hand_frames = bilateral_frames = 0
        maximum_tip = maximum_lift = 0.0
        first_invalid = None
        reason = None
        trace = {
            key: []
            for key in (
                "qpos",
                "qvel",
                "ctrl",
                "actuator_force",
                "hand_contact",
                "foot_contact",
                "object_position",
                "object_tip",
                "root_height",
                "torso_tilt",
                "object_impulse",
                "hand_target",
            )
        }
        held_target = stable_target.copy()
        held_desired = initial_hands.copy()
        cost = 0.0
        for step in range(args.horizon * engine.config.substeps):
            frame = step / engine.config.substeps
            if frame < approach:
                phase = smoothstep(frame / max(approach, 1.0))
                desired = (1.0 - phase) * initial_hands + phase * contact
            else:
                phase = smoothstep((frame - approach) / max(duration, 1.0))
                desired = (1.0 - phase) * contact + phase * pushed
            if step % engine.config.substeps == 0:
                held_target = stable_target.copy()
                held_target[balance] += balance_residual(frame, parameters)
                held_target[arms] = arm_target(data, desired, gain)
                held_desired = desired
            data.ctrl[:] = np.clip(
                held_target, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
            )
            mujoco.mj_step(model, data)
            physical = generator._physical(data)
            for contact_index in range(data.ncon):
                item = data.contact[contact_index]
                first, second = int(item.geom1), int(item.geom2)
                if not ({first, second} & object_geoms):
                    continue
                other = second if first in object_geoms else first
                if int(model.geom_bodyid[other]) not in engine.robot_bodies:
                    continue
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, contact_index, force)
                impulses[group(other)] += float(np.linalg.norm(force[:3])) * model.opt.timestep
            object_position = data.qpos[engine.object_qpos : engine.object_qpos + 3]
            object_rotation = quat_rotation(
                data.qpos[engine.object_qpos + 3 : engine.object_qpos + 7]
            )
            delta_rotation = object_rotation * initial_object_rotation.inv()
            tip = float(np.linalg.norm(delta_rotation.as_rotvec()[:2]))
            maximum_tip = max(maximum_tip, tip)
            lift = float(object_position[2] - initial_object[2])
            maximum_lift = max(maximum_lift, lift)
            hands = int(np.sum(physical["hand_contact"]))
            root = float(data.qpos[engine.robot_qpos + 2])
            support = int(np.sum(physical["foot_contact"]))
            disallowed = sum(
                impulses[key]
                for key in ("feet", "legs", "torso_pelvis", "other_robot")
            )
            cost += (
                6000.0 * max(0.0, 0.50 - root) ** 2
                + 3500.0 * max(0.0, physical["tilt"] - np.deg2rad(42.0)) ** 2
                + 4000.0 * float(support == 0)
                + 120.0 * float(support == 1)
                + 2500.0 * float(frame > 0.7 * approach and hands == 0)
                + 500.0 * float(frame > approach and hands == 1)
                - 15000.0 * tip
                - 8000.0 * max(0.0, lift)
            )
            invalid = generator._invalid_reason(data, physical)
            if disallowed > args.maximum_disallowed_impulse and invalid is None:
                invalid = "non_hand_object_impulse"
            if step % engine.config.substeps == engine.config.substeps - 1:
                hand_frames += int(hands > 0)
                bilateral_frames += int(hands == 2)
                if record:
                    trace["qpos"].append(data.qpos.copy())
                    trace["qvel"].append(data.qvel.copy())
                    trace["ctrl"].append(data.ctrl.copy())
                    trace["actuator_force"].append(data.actuator_force.copy())
                    trace["hand_contact"].append(physical["hand_contact"].copy())
                    trace["foot_contact"].append(physical["foot_contact"].copy())
                    trace["object_position"].append(object_position.copy())
                    trace["object_tip"].append(tip)
                    trace["root_height"].append(root)
                    trace["torso_tilt"].append(float(physical["tilt"]))
                    trace["object_impulse"].append(np.asarray(list(impulses.values())))
                    trace["hand_target"].append(held_desired.copy())
            if invalid is not None:
                first_invalid, reason = step, invalid
                break
        valid = first_invalid is None
        cost /= max(step + 1, 1)
        cost -= 600.0 * hand_frames + 800.0 * bilateral_frames
        if valid:
            rank = (0, -maximum_tip, -maximum_lift, cost)
        else:
            rank = (1, -step, -maximum_tip, cost)
        metrics = {
            "first_invalid_frame": None if valid else first_invalid / 4.0,
            "invalid_reason": reason,
            "valid_frames": step / 4.0,
            "hand_contact_frames": hand_frames,
            "bilateral_hand_contact_frames": bilateral_frames,
            "maximum_box_tip_deg": float(np.rad2deg(maximum_tip)),
            "maximum_object_com_lift_m": maximum_lift,
            "final_object_position_m": object_position.tolist(),
            "object_impulse_ns": impulses,
            "final_root_height_m": root,
            "final_torso_tilt_deg": float(np.rad2deg(physical["tilt"])),
            "cost": float(cost),
        }
        return rank, metrics, {key: np.asarray(value) for key, value in trace.items()}

    lower = np.asarray([0.045, 0.55, 0.002, 20.0, 0.05, -0.10, 20.0, 0.25])
    upper = np.asarray([0.135, 0.95, 0.045, 65.0, 0.38, 0.08, 80.0, 1.20])
    mean = np.r_[
        [0.09, 0.78, 0.015, 42.0, 0.20, -0.02, 48.0, 0.65],
        np.zeros(2 * len(balance)),
    ]
    std = np.r_[
        [0.025, 0.10, 0.012, 10.0, 0.09, 0.04, 14.0, 0.22],
        np.full(2 * len(balance), 0.09),
    ]
    lower = np.r_[lower, np.full(2 * len(balance), -0.32)]
    upper = np.r_[upper, np.full(2 * len(balance), 0.32)]
    rng = np.random.default_rng(81161)
    best_parameters = mean.copy()
    best = evaluate(best_parameters)
    history = []
    started = time.perf_counter()
    for generation in range(args.generations):
        population = np.clip(
            rng.normal(mean, std, (args.population, len(mean))), lower, upper
        )
        population[0] = best_parameters
        results = [evaluate(candidate) for candidate in population]
        order = sorted(range(len(population)), key=lambda index: results[index][0])
        if results[order[0]][0] < best[0]:
            best_parameters, best = population[order[0]].copy(), results[order[0]]
        elite = population[order[:8]]
        mean = 0.72 * elite.mean(0) + 0.28 * mean
        std = np.maximum(0.72 * elite.std(0) + 0.28 * std, (upper - lower) * 0.006)
        history.append({"generation": generation, **best[1]})
        print("GEN", generation, "PARAM", best_parameters.tolist(), json.dumps(best[1]), flush=True)
    _, metrics, trace = evaluate(best_parameters, record=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        parameters=best_parameters,
        initial_qpos=initial_qpos,
        initial_qvel=initial_qvel,
        stable_target=stable_target,
        arm_actuators=arms,
        **trace,
    )
    report = {
        "method": "hand-only dynamic box-tip transition search",
        "parameter_names": [
            "half_spread_m",
            "contact_height_fraction",
            "front_clearance_m",
            "approach_frames",
            "push_distance_m",
            "vertical_push_m",
            "push_duration_frames",
            "ik_gain",
        ],
        "parameters": best_parameters.tolist(),
        "balance_actuators": balance.tolist(),
        "balance_knot_frames": balance_frames.tolist(),
        "population": args.population,
        "generations": args.generations,
        "rollouts": args.population * args.generations,
        "runtime_s": time.perf_counter() - started,
        "maximum_disallowed_impulse_ns": args.maximum_disallowed_impulse,
        "best": metrics,
        "history": history,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
