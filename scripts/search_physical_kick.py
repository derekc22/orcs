#!/usr/bin/env python3
"""Search a dynamically evaluated single-leg box strike from a stable anchor."""

from __future__ import annotations

import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.optimize import least_squares

from orcs.cli.refine_diffusion_reference import _build_model
from orcs.tasks.uolm.dynamics_refinement import DynamicsRefinementConfig, _DynamicRollout
from orcs.tasks.uolm.generated_reference import GeneratedReferenceAdapter
from orcs.tasks.uolm.physical_reference_generator import PhysicalReferenceConfig, PhysicalReferenceGenerator


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def main() -> None:
    anchor = np.load("logs/unified_physical_reference/contact_kick_stance_start_000.npz")
    prior = np.load("logs/direct_contact_dynamics_knot117/direct_contact_dynamics_start_000.npz")["analytical_through140_trajectory"]
    reference = GeneratedReferenceAdapter(device=torch.device("cpu"), dt=0.02, fk_batch_size=256)(torch.as_tensor(prior))
    engine = _DynamicRollout(_build_model(), reference, DynamicsRefinementConfig(run_stage_b=False))
    generator = PhysicalReferenceGenerator(engine, PhysicalReferenceConfig(optimize_initial_stance=False))
    model = engine.model
    names = np.asarray(engine.joint_names)[engine.actuator_joint_indexes]
    right = np.asarray([index for index, name in enumerate(names) if name.startswith("right_") and any(token in name for token in ("hip_", "knee", "ankle_"))])
    right_qpos = engine.actuator_qpos[right]
    right_dof = model.jnt_dofadr[model.actuator_trnid[right, 0]].astype(np.int32)
    joint_ids = model.actuator_trnid[right, 0]
    joint_lower = model.jnt_range[joint_ids, 0] + 0.001
    joint_upper = model.jnt_range[joint_ids, 1] - 0.001
    q0 = anchor["qs"].copy()
    foot = engine.foot_bodies[1]
    inverse = mujoco.MjData(model)
    inverse.qpos[:] = anchor["initial_qpos"]

    def foot_target_error(values: np.ndarray) -> np.ndarray:
        inverse.qpos[right_qpos] = values
        mujoco.mj_forward(model, inverse)
        return np.r_[(inverse.xpos[foot] - np.asarray([0.62, 0.40, 0.18])) * 20.0, (values - q0) * 0.04]

    q_strike = least_squares(foot_target_error, np.clip(q0, joint_lower, joint_upper), bounds=(joint_lower, joint_upper), max_nfev=1000).x
    nominal_delta = q_strike - q0
    goal_position = engine.reference_object_pos[-1]
    goal_quaternion = engine.reference_object_quat[-1]
    horizon_steps = 520

    def evaluate(parameters: np.ndarray, record: bool = False):
        initial_velocity = parameters[:6]
        strike_delta = parameters[6:12]
        duration = 2.0 + 18.0 / (1.0 + np.exp(-parameters[12]))
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
        best_task = float("inf")
        best_position_error = float("inf")
        best_orientation_error = float("inf")
        best_frame = 0.0
        maximum_progress = 0.0
        initial_error = float(np.linalg.norm(engine.reference_object_pos[0] - goal_position))
        trace = []
        for step in range(horizon_steps):
            strike = smoothstep(step / duration) * (1.0 - smoothstep((step - 52.0) / 28.0))
            control = anchor["control"].copy()
            control[right] = np.clip(q0 + strike * strike_delta, joint_lower, joint_upper)
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
            task = position_error + 0.12 * orientation_error
            maximum_progress = max(maximum_progress, initial_error - position_error)
            if task < best_task:
                best_task = task
                best_position_error = position_error
                best_orientation_error = orientation_error
                best_frame = step / engine.config.substeps
            if record:
                trace.append(np.r_[data.qpos.copy(), data.qvel.copy(), data.ctrl.copy(), data.actuator_force.copy()])
            # Never allow later collapse contacts to improve the strike score.
            if first_invalid is not None and step > first_invalid + 1:
                break
        survival = horizon_steps if first_invalid is None else first_invalid
        early_failure = max(0, 120 - survival)
        cost = 3000.0 * best_position_error**2 + 180.0 * best_orientation_error**2 + 2.0 * early_failure**2 + 0.4 * float(np.square(initial_velocity).sum()) + 8.0 * float(np.square(strike_delta - nominal_delta).sum())
        rank = (0 if best_position_error < 0.15 and best_orientation_error < 0.35 else 1, cost)
        metrics = {
            "first_invalid_frame": None if first_invalid is None else first_invalid / engine.config.substeps,
            "invalid_reason": reason,
            "best_frame": best_frame,
            "best_position_error_m": best_position_error,
            "best_orientation_error_rad": best_orientation_error,
            "maximum_goal_progress_m": maximum_progress,
            "duration_steps": duration,
            "cost": cost,
        }
        return rank, metrics, np.asarray(trace)

    dimensions = 13
    mean = np.r_[8.0 * nominal_delta, nominal_delta, -0.5]
    lower = np.r_[np.full(6, -12.0), joint_lower - q0, -3.0]
    upper = np.r_[np.full(6, 12.0), joint_upper - q0, 3.0]
    std = np.r_[np.full(6, 2.5), np.full(6, 0.28), 0.8]
    rng = np.random.default_rng(73199)
    best_parameters = np.clip(mean, lower, upper)
    best = evaluate(best_parameters)
    history = []
    started = time.perf_counter()
    for generation in range(34):
        population = np.clip(rng.normal(mean, std, size=(96, dimensions)), lower, upper)
        population[0] = best_parameters
        population[1] = np.r_[np.zeros(6), nominal_delta, -0.5]
        results = [evaluate(candidate) for candidate in population]
        order = sorted(range(len(population)), key=lambda index: results[index][0])
        if results[order[0]][0] < best[0]:
            best_parameters = population[order[0]].copy()
            best = results[order[0]]
        elite = population[order[:12]]
        mean = 0.72 * elite.mean(axis=0) + 0.28 * mean
        std = np.maximum(0.72 * elite.std(axis=0) + 0.28 * std, np.r_[np.full(6, 0.15), np.full(6, 0.015), 0.08])
        history.append({"generation": generation, **best[1]})
        print("GEN", generation, json.dumps(best[1]), flush=True)
        if best[0][0] == 0:
            break
    _, metrics, trace = evaluate(best_parameters, record=True)
    output = Path("logs/unified_physical_reference")
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "physical_kick_search_start_000.npz", parameters=best_parameters, trace=trace, q0=q0, nominal_delta=nominal_delta)
    report = {"method": "direct MuJoCo right-foot strike search", "dimensions": dimensions, "parameters": best_parameters.tolist(), "best": metrics, "history": history, "runtime_s": time.perf_counter() - started}
    (output / "physical_kick_search_start_000.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
