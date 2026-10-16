#!/usr/bin/env python3
"""Search a goal-directed physical box strike with MuJoCo as the evaluator."""

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


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def main() -> None:
    anchor = np.load("logs/unified_physical_reference/contact_kick_stance_start_000.npz")
    previous = np.load("logs/unified_physical_reference/physical_kick_search_start_000.npz")
    prior = np.load("logs/direct_contact_dynamics_knot117/direct_contact_dynamics_start_000.npz")["analytical_through140_trajectory"]
    reference = GeneratedReferenceAdapter(device=torch.device("cpu"), dt=0.02, fk_batch_size=256)(torch.as_tensor(prior))
    engine = _DynamicRollout(_build_model(), reference, DynamicsRefinementConfig(run_stage_b=False))
    generator = PhysicalReferenceGenerator(engine, PhysicalReferenceConfig(optimize_initial_stance=False))
    model = engine.model
    names = np.asarray(engine.joint_names)[engine.actuator_joint_indexes]
    right = np.asarray([index for index, name in enumerate(names) if name.startswith("right_") and any(token in name for token in ("hip_", "knee", "ankle_"))])
    right_dof = model.jnt_dofadr[model.actuator_trnid[right, 0]].astype(np.int32)
    joint_ids = model.actuator_trnid[right, 0]
    lower_joint = model.jnt_range[joint_ids, 0] + 0.001
    upper_joint = model.jnt_range[joint_ids, 1] - 0.001
    q0 = previous["q0"]
    goal_position = engine.reference_object_pos[-1]
    goal_quaternion = engine.reference_object_quat[-1]
    horizon_steps = 600

    # [six right-leg qvel, root vx/vy, six strike deltas,
    #  rise logit, retract-start logit, retract-duration logit]
    def decode(parameters: np.ndarray) -> tuple[float, float, float]:
        rise = 2.0 + 24.0 / (1.0 + np.exp(-parameters[14]))
        retract_start = 35.0 + 105.0 / (1.0 + np.exp(-parameters[15]))
        retract_duration = 12.0 + 88.0 / (1.0 + np.exp(-parameters[16]))
        return rise, retract_start, retract_duration

    def evaluate(parameters: np.ndarray, record: bool = False):
        rise, retract_start, retract_duration = decode(parameters)
        data = mujoco.MjData(model)
        data.qpos[:] = anchor["initial_qpos"]
        data.qvel[:] = anchor["initial_qvel"]
        data.qvel[right_dof] = parameters[:6]
        data.qvel[engine.robot_dof : engine.robot_dof + 2] += parameters[6:8]
        data.qpos[engine.object_qpos : engine.object_qpos + 3] = engine.reference_object_pos[0]
        data.qpos[engine.object_qpos + 3 : engine.object_qpos + 7] = engine.reference_object_quat[0]
        data.qvel[engine.object_dof : engine.object_dof + 6] = 0.0
        mujoco.mj_forward(model, data)
        first_invalid = None
        reason = None
        best_task = float("inf")
        best_position = float("inf")
        best_orientation = float("inf")
        best_step = 0
        best_speed = float("inf")
        maximum_speed = 0.0
        trace = []
        for step in range(horizon_steps):
            amount = smoothstep(step / rise) * (1.0 - smoothstep((step - retract_start) / retract_duration))
            control = anchor["control"].copy()
            control[right] = np.clip(q0 + amount * parameters[8:14], lower_joint, upper_joint)
            data.ctrl[:] = np.clip(control, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
            mujoco.mj_step(model, data)
            physical = generator._physical(data)
            invalid = generator._invalid_reason(data, physical)
            if invalid is not None and first_invalid is None:
                first_invalid = step
                reason = invalid
            position = data.qpos[engine.object_qpos : engine.object_qpos + 3]
            quaternion = data.qpos[engine.object_qpos + 3 : engine.object_qpos + 7]
            position_error = float(np.linalg.norm(position - goal_position))
            orientation_error = float(2.0 * np.arccos(np.clip(abs(np.dot(quaternion, goal_quaternion)), 0.0, 1.0)))
            object_speed = float(np.linalg.norm(data.qvel[engine.object_dof : engine.object_dof + 6]))
            task = position_error + 0.16 * orientation_error + 0.16 * object_speed
            if task < best_task:
                best_task = task
                best_position = position_error
                best_orientation = orientation_error
                best_step = step
                best_speed = object_speed
            maximum_speed = max(maximum_speed, float(np.linalg.norm(data.qvel[engine.object_dof : engine.object_dof + 3])))
            if record:
                trace.append(np.r_[data.qpos.copy(), data.qvel.copy(), data.ctrl.copy(), data.actuator_force.copy()])
            if first_invalid is not None and step > first_invalid + 2:
                break
        survival = horizon_steps if first_invalid is None else first_invalid
        early_failure = max(0, 120 - survival)
        pathological = float(reason == "pathological_penetration" and survival < 40)
        cost = (
            7000.0 * best_position**2
            + 850.0 * best_orientation**2
            + 1000.0 * best_speed**2
            + 2.0 * early_failure**2
            + 100000.0 * pathological
            + 0.18 * float(np.square(parameters[:6]).sum())
            + 10.0 * float(np.square(parameters[6:8]).sum())
            + 3.0 * float(np.square(parameters[8:14] - previous["nominal_delta"]).sum())
        )
        at_goal = best_position < 0.15 and best_orientation < 0.35 and best_speed < 0.20
        rank = (0 if at_goal else 1, cost)
        metrics = {
            "first_invalid_frame": None if first_invalid is None else first_invalid / engine.config.substeps,
            "invalid_reason": reason,
            "best_frame": best_step / engine.config.substeps,
            "best_position_error_m": best_position,
            "best_orientation_error_rad": best_orientation,
            "best_object_speed": best_speed,
            "maximum_object_speed_mps": maximum_speed,
            "rise_steps": rise,
            "retract_start_steps": retract_start,
            "retract_duration_steps": retract_duration,
            "cost": cost,
        }
        return rank, metrics, np.asarray(trace)

    old = previous["parameters"]
    mean = np.r_[old[:6], np.zeros(2), old[6:12], old[12], 0.0, 0.0]
    lower = np.r_[np.full(6, -22.0), [-2.0, -2.0], lower_joint - q0, [-4.0, -4.0, -4.0]]
    upper = np.r_[np.full(6, 22.0), [2.0, 2.0], upper_joint - q0, [4.0, 4.0, 4.0]]
    std = np.r_[np.full(6, 3.5), [0.45, 0.45], np.full(6, 0.35), [1.0, 1.2, 1.0]]
    rng = np.random.default_rng(936113)
    best_parameters = np.clip(mean, lower, upper)
    best = evaluate(best_parameters)
    history = []
    started = time.perf_counter()
    for generation in range(44):
        population = np.clip(rng.normal(mean, std, size=(112, len(mean))), lower, upper)
        population[0] = best_parameters
        results = [evaluate(candidate) for candidate in population]
        order = sorted(range(len(population)), key=lambda index: results[index][0])
        if results[order[0]][0] < best[0]:
            best_parameters = population[order[0]].copy()
            best = results[order[0]]
        elite = population[order[:14]]
        mean = 0.72 * elite.mean(axis=0) + 0.28 * mean
        floor = np.r_[np.full(6, 0.18), [0.03, 0.03], np.full(6, 0.012), [0.07, 0.07, 0.07]]
        std = np.maximum(0.72 * elite.std(axis=0) + 0.28 * std, floor)
        history.append({"generation": generation, **best[1]})
        print("GEN", generation, json.dumps(best[1]), flush=True)
        if best[0][0] == 0 and generation >= 8:
            break
    _, metrics, trace = evaluate(best_parameters, record=True)
    output = Path("logs/unified_physical_reference")
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "goal_strike_start_000.npz", parameters=best_parameters, trace=trace, q0=q0, right=right)
    report = {"method": "goal-directed MuJoCo foot-strike search", "dimensions": len(mean), "parameters": best_parameters.tolist(), "best": metrics, "history": history, "runtime_s": time.perf_counter() - started}
    (output / "goal_strike_start_000.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
