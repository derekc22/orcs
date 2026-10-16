#!/usr/bin/env python3
"""Co-optimize physical support controls around the best MuJoCo kick."""

from __future__ import annotations

import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from orcs.cli.refine_diffusion_reference import _build_model
from orcs.tasks.uolm.dynamics_refinement import DynamicsRefinementConfig, _DynamicRollout
from orcs.tasks.uolm.generated_reference import GeneratedReferenceAdapter
from orcs.tasks.uolm.physical_reference_generator import PhysicalReferenceConfig, PhysicalReferenceGenerator


KNOT_FRAMES = np.asarray([0.0, 20.0, 50.0, 90.0, 160.0, 300.0])


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def main() -> None:
    anchor = np.load("logs/unified_physical_reference/contact_kick_stance_start_000.npz")
    strike = np.load("logs/unified_physical_reference/physical_kick_search_start_000.npz")
    prior = np.load("logs/direct_contact_dynamics_knot117/direct_contact_dynamics_start_000.npz")["analytical_through140_trajectory"]
    reference = GeneratedReferenceAdapter(device=torch.device("cpu"), dt=0.02, fk_batch_size=256)(torch.as_tensor(prior))
    engine = _DynamicRollout(_build_model(), reference, DynamicsRefinementConfig(run_stage_b=False))
    generator = PhysicalReferenceGenerator(engine, PhysicalReferenceConfig(optimize_initial_stance=False))
    model = engine.model
    names = np.asarray(engine.joint_names)[engine.actuator_joint_indexes]
    right = np.asarray([index for index, name in enumerate(names) if name.startswith("right_") and any(token in name for token in ("hip_", "knee", "ankle_"))])
    right_dof = model.jnt_dofadr[model.actuator_trnid[right, 0]].astype(np.int32)
    controlled = np.asarray([index for index, name in enumerate(names) if (name.startswith("left_") and any(token in name for token in ("hip_", "knee", "ankle_"))) or name.startswith("waist_")])
    parameters_strike = strike["parameters"]
    initial_velocity = parameters_strike[:6]
    strike_delta = parameters_strike[6:12]
    duration = 2.0 + 18.0 / (1.0 + np.exp(-parameters_strike[12]))
    q0 = strike["q0"]
    joint_ids = model.actuator_trnid[right, 0]
    joint_lower = model.jnt_range[joint_ids, 0] + 0.001
    joint_upper = model.jnt_range[joint_ids, 1] - 0.001
    goal_position = engine.reference_object_pos[-1]
    goal_quaternion = engine.reference_object_quat[-1]
    total_steps = (engine.horizon - 1) * engine.config.substeps + 1

    def decode(parameters: np.ndarray) -> np.ndarray:
        values = parameters.reshape(len(KNOT_FRAMES), len(controlled))
        sample_frames = np.arange(total_steps) / engine.config.substeps
        return np.column_stack([np.interp(sample_frames, KNOT_FRAMES, values[:, column]) for column in range(len(controlled))])

    def evaluate(parameters: np.ndarray, record: bool = False):
        correction = decode(parameters)
        data = mujoco.MjData(model)
        data.qpos[:] = anchor["initial_qpos"]
        data.qvel[:] = anchor["initial_qvel"]
        data.qvel[right_dof] = initial_velocity
        data.qpos[engine.object_qpos : engine.object_qpos + 3] = engine.reference_object_pos[0]
        data.qpos[engine.object_qpos + 3 : engine.object_qpos + 7] = engine.reference_object_quat[0]
        data.qvel[engine.object_dof : engine.object_dof + 6] = 0.0
        mujoco.mj_forward(model, data)
        first_invalid = None
        reason = None
        minimum_position_error = float("inf")
        minimum_orientation_error = float("inf")
        soft = 0.0
        trace = []
        for step in range(total_steps):
            strike_amount = smoothstep(step / duration) * (1.0 - smoothstep((step - 52.0) / 28.0))
            control = anchor["control"].copy()
            control[controlled] += correction[step]
            control[right] = np.clip(q0 + strike_amount * strike_delta, joint_lower, joint_upper)
            data.ctrl[:] = np.clip(control, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
            mujoco.mj_step(model, data)
            physical = generator._physical(data)
            invalid = generator._invalid_reason(data, physical)
            if invalid is not None and first_invalid is None:
                first_invalid = step
                reason = invalid
            position = data.qpos[engine.object_qpos : engine.object_qpos + 3]
            quaternion = data.qpos[engine.object_qpos + 3 : engine.object_qpos + 7]
            minimum_position_error = min(minimum_position_error, float(np.linalg.norm(position - goal_position)))
            minimum_orientation_error = min(minimum_orientation_error, float(2.0 * np.arccos(np.clip(abs(np.dot(quaternion, goal_quaternion)), 0.0, 1.0))))
            root = data.qpos[engine.robot_qpos : engine.robot_qpos + 3]
            soft += 30.0 * physical["tilt"] ** 2 + 100.0 * max(0.0, 0.55 - root[2]) ** 2 + 1.5 * float(np.square(physical["foot_slip"]).sum())
            if record:
                trace.append(np.r_[data.qpos.copy(), data.qvel.copy(), data.ctrl.copy(), data.actuator_force.copy(), physical["foot_contact"]])
            if first_invalid is not None and step > first_invalid + 2 and not record:
                break
        task_cost = 2000.0 * minimum_position_error**2 + 120.0 * minimum_orientation_error**2
        regularization = 12.0 * float(np.square(parameters).mean()) + soft / max(1, step + 1)
        rank = (0, task_cost + regularization) if first_invalid is None else (1, -first_invalid, task_cost + regularization)
        metrics = {
            "first_invalid_frame": None if first_invalid is None else first_invalid / engine.config.substeps,
            "invalid_reason": reason,
            "minimum_position_error_m": minimum_position_error,
            "minimum_orientation_error_rad": minimum_orientation_error,
            "final_root_height_m": float(data.qpos[engine.robot_qpos + 2]),
            "task_cost": task_cost,
        }
        return rank, metrics, np.asarray(trace)

    dimensions = len(KNOT_FRAMES) * len(controlled)
    rng = np.random.default_rng(99173)
    mean = np.zeros(dimensions)
    std = np.full(dimensions, 0.11)
    bound = np.full(dimensions, 0.50)
    best_parameters = mean.copy()
    best = evaluate(mean)
    history = []
    started = time.perf_counter()
    for generation in range(30):
        population = np.clip(rng.normal(mean, std, size=(72, dimensions)), -bound, bound)
        population[0] = best_parameters
        population[1] = 0.0
        results = [evaluate(candidate) for candidate in population]
        order = sorted(range(len(population)), key=lambda index: results[index][0])
        if results[order[0]][0] < best[0]:
            best_parameters = population[order[0]].copy()
            best = results[order[0]]
        elite = population[order[:10]]
        mean = 0.72 * elite.mean(axis=0) + 0.28 * mean
        std = np.maximum(0.72 * elite.std(axis=0) + 0.28 * std, 0.012)
        history.append({"generation": generation, **best[1]})
        print("GEN", generation, json.dumps(best[1]), flush=True)
    _, metrics, trace = evaluate(best_parameters, record=True)
    output = Path("logs/unified_physical_reference")
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "recovered_physical_kick_start_000.npz", parameters=best_parameters, trace=trace, controlled=controlled, knot_frames=KNOT_FRAMES)
    report = {"method": "MuJoCo support recovery around physical strike", "dimensions": dimensions, "controlled_joints": names[controlled].tolist(), "best": metrics, "history": history, "runtime_s": time.perf_counter() - started}
    (output / "recovered_physical_kick_start_000.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
