#!/usr/bin/env python3
"""Optimize a task-relative hand impulse from a physically stable stance."""

from __future__ import annotations

import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.optimize import least_squares

from orcs.cli.refine_diffusion_reference import _build_model
from orcs.tasks.uolm.contact_dynamics_refinement import _copy_data
from orcs.tasks.uolm.dynamics_refinement import DynamicsRefinementConfig, _DynamicRollout
from orcs.tasks.uolm.generated_reference import GeneratedReferenceAdapter
from orcs.tasks.uolm.physical_reference_generator import (
    PhysicalReferenceConfig,
    PhysicalReferenceGenerator,
)


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value**3 * (10.0 + value * (-15.0 + 6.0 * value))


def main() -> None:
    source_path = Path(
        "logs/direct_contact_dynamics_knot117/direct_contact_dynamics_start_000.npz"
    )
    stance_path = Path("logs/unified_physical_reference/oriented_stance_start_000.npz")
    if not stance_path.exists():
        stance_path = Path("/tmp/extended_stance_best.npz")
    with np.load(source_path) as loaded:
        trajectory = loaded["analytical_through140_trajectory"]
    with np.load(stance_path) as loaded:
        stance_parameters = loaded["parameters"]
        arm_neutral = loaded["arm_pose"]
    reference = GeneratedReferenceAdapter(
        device=torch.device("cpu"), dt=0.02, fk_batch_size=256
    )(torch.as_tensor(trajectory))
    engine = _DynamicRollout(
        _build_model(), reference, DynamicsRefinementConfig(run_stage_b=False, seed=42)
    )
    generator = PhysicalReferenceGenerator(
        engine,
        PhysicalReferenceConfig(
            robot_clearance_m=0.48,
            desired_stance_height_m=0.62,
            stance_parameters=tuple(stance_parameters),
            use_lqr_executor=False,
        ),
    )
    model = engine.model
    arm_actuators = np.setdiff1d(
        np.arange(model.nu), generator.balance_actuators
    )
    arm_qpos = engine.actuator_qpos[arm_actuators]
    generator.initial_data.qpos[arm_qpos] = arm_neutral
    generator.nominal_control[arm_actuators] = arm_neutral
    generator.initial_data.ctrl[:] = generator.nominal_control
    mujoco.mj_forward(model, generator.initial_data)
    generator.desired_rotation = generator.initial_data.xmat[
        generator.root_body
    ].reshape(3, 3).copy()
    generator.initial_root = generator.initial_data.qpos[
        engine.robot_qpos : engine.robot_qpos + 3
    ].copy()
    generator.torso_local_up = generator.initial_data.xmat[
        engine.torso_body
    ].reshape(3, 3).T @ np.asarray([0.0, 0.0, 1.0])
    arm_joint_ids = model.actuator_trnid[arm_actuators, 0]
    arm_lower = model.jnt_range[arm_joint_ids, 0] + 0.02
    arm_upper = model.jnt_range[arm_joint_ids, 1] - 0.02
    object_start = generator.initial_data.qpos[
        engine.object_qpos : engine.object_qpos + 3
    ].copy()
    object_goal = engine.reference_object_pos[-1]
    direction = object_goal[:2] - object_start[:2]
    direction /= np.linalg.norm(direction)
    lateral = np.asarray([-direction[1], direction[0]])
    substeps = engine.config.substeps
    samples = (engine.horizon - 1) * substeps + 1
    hand_data = _copy_data(model, generator.initial_data)

    direction_3d = np.r_[direction, 0.0]

    # [left advance, right advance, left z, right z, left lateral,
    #  right lateral, peak frame, return duration]
    lower = np.asarray([0.0, 0.0, 0.45, 0.45, 0.05, 0.05, 90.0, 15.0])
    upper = np.asarray([0.65, 0.65, 0.66, 0.66, 0.20, 0.20, 250.0, 80.0])
    initial = np.asarray([0.36, 0.36, 0.55, 0.55, 0.13, 0.13, 170.0, 40.0])

    def arm_pose(parameters: np.ndarray) -> np.ndarray:
        rear_left = 0.55 - parameters[0]
        rear_right = 0.55 - parameters[1]
        targets = np.asarray(
            [
                np.r_[
                    object_start[:2]
                    - rear_left * direction
                    + parameters[4] * lateral,
                    parameters[2],
                ],
                np.r_[
                    object_start[:2]
                    - rear_right * direction
                    - parameters[5] * lateral,
                    parameters[3],
                ],
            ]
        )

        def error(position: np.ndarray) -> np.ndarray:
            hand_data.qpos[arm_qpos] = position
            mujoco.mj_forward(model, hand_data)
            hands = np.asarray(
                [hand_data.geom_xpos[int(geom)] for geom in engine.hand_geoms]
            )
            normals = np.asarray(
                [
                    hand_data.geom_xmat[int(geom)].reshape(3, 3)[:, 1]
                    for geom in engine.hand_geoms
                ]
            )
            return np.r_[
                (hands - targets).ravel() * 10.0,
                np.cross(normals, direction_3d).ravel() * 0.8,
                normals[:, 2] * 0.4,
                (position - arm_neutral) * 0.10,
            ]

        return least_squares(
            error,
            arm_neutral,
            bounds=(arm_lower, arm_upper),
            max_nfev=120,
            ftol=1.0e-7,
            xtol=1.0e-7,
            gtol=1.0e-7,
        ).x

    def evaluate(parameters: np.ndarray, record: bool = False):
        peak = float(parameters[6])
        end = min(299.0, peak + float(parameters[7]))
        target = arm_pose(parameters)
        data = _copy_data(model, generator.initial_data)
        first_invalid = None
        invalid_reason = None
        forbidden_samples = 0
        hand_samples = np.zeros(2, dtype=np.int64)
        object_contact_samples = 0
        maximum_usage = 0.0
        trace: dict[str, list[np.ndarray | float | bool]] = {
            "qpos": [],
            "qvel": [],
            "ctrl": [],
            "actuator_force": [],
            "foot_contact": [],
            "hand_contact": [],
            "torso_tilt": [],
            "forbidden": [],
        }
        for sample in range(samples):
            frame = sample / substeps
            if frame < 60.0:
                amount = 0.0
            elif frame < peak:
                amount = smoothstep((frame - 60.0) / max(peak - 60.0, 1.0))
            elif frame < end:
                amount = 1.0 - smoothstep((frame - peak) / max(end - peak, 1.0))
            else:
                amount = 0.0
            command = generator.nominal_control.copy()
            command[arm_actuators] = arm_neutral + amount * (target - arm_neutral)
            data.ctrl[:] = np.clip(
                command,
                model.actuator_ctrlrange[:, 0],
                model.actuator_ctrlrange[:, 1],
            )
            mujoco.mj_step(model, data)
            physical = generator._physical(data)
            hand_samples += physical["hand_contact"]
            object_contact_samples += int(physical["object_contact"])
            forbidden_samples += int(bool(physical["forbidden"]))
            maximum_usage = max(
                maximum_usage,
                float(np.max(np.abs(data.actuator_force) / engine.force_scale)),
            )
            reason = generator._invalid_reason(data, physical)
            if reason is not None and first_invalid is None:
                first_invalid = sample
                invalid_reason = reason
            if record:
                trace["qpos"].append(data.qpos.copy())
                trace["qvel"].append(data.qvel.copy())
                trace["ctrl"].append(data.ctrl.copy())
                trace["actuator_force"].append(data.actuator_force.copy())
                trace["foot_contact"].append(physical["foot_contact"].copy())
                trace["hand_contact"].append(physical["hand_contact"].copy())
                trace["torso_tilt"].append(physical["tilt"])
                trace["forbidden"].append(bool(physical["forbidden"]))
            if first_invalid is not None and sample > first_invalid + 4 and not record:
                break
        object_position = data.qpos[
            engine.object_qpos : engine.object_qpos + 3
        ].copy()
        object_quaternion = data.qpos[
            engine.object_qpos + 3 : engine.object_qpos + 7
        ].copy()
        position_error = float(np.linalg.norm(object_position - object_goal))
        orientation_error = float(
            2.0
            * np.arccos(
                np.clip(
                    abs(np.dot(object_quaternion, engine.reference_object_quat[-1])),
                    -1.0,
                    1.0,
                )
            )
        )
        task_cost = (
            5000.0 * position_error**2
            + 800.0 * orientation_error**2
            + 1000.0 * forbidden_samples / samples
            + 10.0 * float(np.square((parameters - initial) / (upper - lower)).mean())
        )
        rank = (
            (0, task_cost)
            if first_invalid is None
            else (1, -first_invalid, task_cost)
        )
        metrics = {
            "valid": first_invalid is None,
            "first_invalid_frame": (
                None if first_invalid is None else first_invalid / substeps
            ),
            "invalid_reason": invalid_reason,
            "final_goal_position_error_m": position_error,
            "final_goal_orientation_error_rad": orientation_error,
            "at_goal": position_error < 0.15 and orientation_error < 0.35,
            "object_displacement_m": float(
                np.linalg.norm(object_position - object_start)
            ),
            "hand_contact_duration_s": (
                hand_samples * model.opt.timestep
            ).tolist(),
            "object_contact_duration_s": object_contact_samples * model.opt.timestep,
            "forbidden_support_duration_s": forbidden_samples * model.opt.timestep,
            "maximum_actuator_usage": maximum_usage,
            "final_root_height_m": float(data.qpos[engine.robot_qpos + 2]),
            "task_cost": task_cost,
        }
        arrays = {key: np.asarray(value) for key, value in trace.items()}
        return rank, metrics, arrays, target

    rng = np.random.default_rng(1907)
    mean = initial.copy()
    std = 0.20 * (upper - lower)
    best_parameters = np.asarray([0.0, 0.0, 0.55, 0.55, 0.13, 0.13, 170.0, 40.0])
    best = evaluate(best_parameters)
    print("PROBE", json.dumps(evaluate(initial)[1]), flush=True)
    progress = []
    started = time.perf_counter()
    evaluations = 1
    for generation in range(24):
        population = np.clip(
            rng.normal(mean, std, size=(64, len(mean))), lower, upper
        )
        population[0] = best_parameters
        population[1] = initial
        evaluated = [evaluate(candidate) for candidate in population]
        evaluations += len(population)
        order = sorted(range(len(population)), key=lambda index: evaluated[index][0])
        winner = order[0]
        if evaluated[winner][0] < best[0]:
            best_parameters = population[winner].copy()
            best = evaluated[winner]
        elite = population[order[:10]]
        mean = 0.72 * elite.mean(axis=0) + 0.28 * mean
        std = np.maximum(
            0.72 * elite.std(axis=0) + 0.28 * std,
            0.015 * (upper - lower),
        )
        progress.append({"generation": generation, **best[1]})
        print("GEN", generation, json.dumps(best[1]), flush=True)
        if best[1]["valid"] and best[1]["at_goal"]:
            break
    _, metrics, arrays, arm_target = evaluate(best_parameters, record=True)
    output = Path("logs/unified_physical_reference")
    output.mkdir(parents=True, exist_ok=True)
    artifact = output / "extended_push_start_000.npz"
    np.savez_compressed(
        artifact,
        parameters=best_parameters,
        stance_parameters=stance_parameters,
        arm_neutral=arm_neutral,
        arm_target=arm_target,
        physics_dt=np.asarray([model.opt.timestep]),
        **arrays,
    )
    report = {
        "method": "MuJoCo-evaluated task-relative push from stable extended stance",
        "metrics": metrics,
        "parameters": best_parameters.tolist(),
        "evaluations": evaluations,
        "runtime_s": time.perf_counter() - started,
        "progress": progress,
        "artifact": str(artifact),
    }
    (output / "extended_push_start_000.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
