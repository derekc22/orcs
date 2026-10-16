#!/usr/bin/env python3
"""Jointly optimize a physical strike and recovery over one 300-frame rollout."""

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


SWITCH_STEP = 81  # exact state after strike trace sample 80 (frame 20.25)


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def main() -> None:
    anchor = np.load("logs/unified_physical_reference/contact_kick_stance_start_000.npz")
    strike_prior = np.load("logs/unified_physical_reference/goal_strike_start_000.npz")
    recovery_prior = np.load("logs/unified_physical_reference/goal_strike_recovery_start_000.npz")
    diffusion = np.load("logs/direct_contact_dynamics_knot117/direct_contact_dynamics_start_000.npz")["analytical_through140_trajectory"]
    reference = GeneratedReferenceAdapter(device=torch.device("cpu"), dt=0.02, fk_batch_size=256)(torch.as_tensor(diffusion))
    engine = _DynamicRollout(_build_model(), reference, DynamicsRefinementConfig(run_stage_b=False))
    generator = PhysicalReferenceGenerator(engine, PhysicalReferenceConfig(optimize_initial_stance=False))
    model = engine.model
    names = np.asarray(engine.joint_names)[engine.actuator_joint_indexes]
    right = np.asarray([index for index, name in enumerate(names) if name.startswith("right_") and any(token in name for token in ("hip_", "knee", "ankle_"))])
    right_dof = model.jnt_dofadr[model.actuator_trnid[right, 0]].astype(np.int32)
    joint_ids = model.actuator_trnid[right, 0]
    lower_right = model.jnt_range[joint_ids, 0] + 0.001
    upper_right = model.jnt_range[joint_ids, 1] - 0.001
    balance = recovery_prior["balance"]
    balance_joint_ids = model.actuator_trnid[balance, 0]
    balance_lower = np.maximum(model.actuator_ctrlrange[balance, 0], model.jnt_range[balance_joint_ids, 0] + 0.01)
    balance_upper = np.minimum(model.actuator_ctrlrange[balance, 1], model.jnt_range[balance_joint_ids, 1] - 0.01)
    q0 = strike_prior["q0"]
    base_recovery = recovery_prior["target"]
    goal_position = engine.reference_object_pos[-1]
    goal_quaternion = engine.reference_object_quat[-1]
    total_steps = (engine.horizon - 1) * engine.config.substeps + 1

    def strike_timing(parameters: np.ndarray) -> tuple[float, float, float]:
        return (
            2.0 + 24.0 / (1.0 + np.exp(-parameters[14])),
            35.0 + 105.0 / (1.0 + np.exp(-parameters[15])),
            12.0 + 88.0 / (1.0 + np.exp(-parameters[16])),
        )

    def evaluate(parameters: np.ndarray, record: bool = False):
        strike = parameters[:17]
        recovery = np.clip(base_recovery + parameters[17:], balance_lower, balance_upper)
        rise, retract_start, retract_duration = strike_timing(strike)
        data = mujoco.MjData(model)
        data.qpos[:] = anchor["initial_qpos"]
        data.qvel[:] = anchor["initial_qvel"]
        data.qvel[right_dof] = strike[:6]
        data.qvel[engine.robot_dof : engine.robot_dof + 2] += strike[6:8]
        data.qpos[engine.object_qpos : engine.object_qpos + 3] = engine.reference_object_pos[0]
        data.qpos[engine.object_qpos + 3 : engine.object_qpos + 7] = engine.reference_object_quat[0]
        data.qvel[engine.object_dof : engine.object_dof + 6] = 0.0
        mujoco.mj_forward(model, data)
        first_invalid = None
        reason = None
        forbidden_samples = 0
        maximum_lift = 0.0
        trace = []
        for step in range(total_steps):
            if step < SWITCH_STEP:
                amount = smoothstep(step / rise) * (1.0 - smoothstep((step - retract_start) / retract_duration))
                control = anchor["control"].copy()
                control[right] = np.clip(q0 + amount * strike[8:14], lower_right, upper_right)
            else:
                control = anchor["control"].copy()
                control[balance] = recovery
            data.ctrl[:] = np.clip(control, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
            mujoco.mj_step(model, data)
            physical = generator._physical(data)
            invalid = generator._invalid_reason(data, physical)
            if invalid is not None and first_invalid is None:
                first_invalid = step
                reason = invalid
            forbidden_samples += int(bool(physical["forbidden"]))
            maximum_lift = max(maximum_lift, float(data.qpos[engine.object_qpos + 2] - engine.reference_object_pos[0, 2]))
            if record:
                trace.append(np.r_[data.qpos.copy(), data.qvel.copy(), data.ctrl.copy(), data.actuator_force.copy(), physical["foot_contact"], physical["hand_contact"], float(bool(physical["forbidden"])), physical["penetration"]])
            if first_invalid is not None and step > first_invalid + 3 and not record:
                break
        position_error = float(np.linalg.norm(data.qpos[engine.object_qpos : engine.object_qpos + 3] - goal_position))
        orientation_error = float(2.0 * np.arccos(np.clip(abs(np.dot(data.qpos[engine.object_qpos + 3 : engine.object_qpos + 7], goal_quaternion)), 0.0, 1.0)))
        object_speed = float(np.linalg.norm(data.qvel[engine.object_dof : engine.object_dof + 6]))
        at_goal = position_error < 0.15 and orientation_error < 0.35
        full_valid = first_invalid is None and forbidden_samples == 0
        survival = total_steps if first_invalid is None else first_invalid
        task_cost = 7000.0 * position_error**2 + 900.0 * orientation_error**2 + 120.0 * object_speed**2
        regularization = (
            0.12 * float(np.square(strike[:6]).sum())
            + 6.0 * float(np.square(strike[6:8]).sum())
            + 2.0 * float(np.square(strike[8:14] - strike_prior["parameters"][8:14]).sum())
            + 12.0 * float(np.square(parameters[17:]).mean())
        )
        if full_valid and at_goal:
            rank = (0, task_cost + regularization)
        elif full_valid:
            rank = (1, task_cost + regularization)
        else:
            rank = (2, -survival, task_cost + regularization)
        metrics = {
            "first_invalid_frame": None if first_invalid is None else first_invalid / engine.config.substeps,
            "invalid_reason": reason,
            "full_valid": full_valid,
            "at_goal": at_goal,
            "final_position_error_m": position_error,
            "final_orientation_error_rad": orientation_error,
            "final_object_speed": object_speed,
            "maximum_lift_m": maximum_lift,
            "forbidden_samples": forbidden_samples,
            "final_root_height_m": float(data.qpos[engine.robot_qpos + 2]),
            "final_torso_tilt_deg": float(np.rad2deg(generator._physical(data)["tilt"])),
            "task_cost": task_cost,
        }
        return rank, metrics, np.asarray(trace)

    initial_strike = strike_prior["parameters"]
    mean = np.r_[initial_strike, np.zeros(len(balance))]
    lower = np.r_[np.full(6, -22.0), [-2.0, -2.0], lower_right - q0, [-4.0, -4.0, -4.0], balance_lower - base_recovery]
    upper = np.r_[np.full(6, 22.0), [2.0, 2.0], upper_right - q0, [4.0, 4.0, 4.0], balance_upper - base_recovery]
    std = np.r_[np.full(6, 2.0), [0.25, 0.25], np.full(6, 0.22), [0.65, 0.65, 0.65], np.full(len(balance), 0.18)]
    floor = np.r_[np.full(6, 0.12), [0.015, 0.015], np.full(6, 0.008), [0.04, 0.04, 0.04], np.full(len(balance), 0.006)]
    rng = np.random.default_rng(841337)
    best_parameters = np.clip(mean, lower, upper)
    best = evaluate(best_parameters)
    history = []
    started = time.perf_counter()
    for generation in range(52):
        population = np.clip(rng.normal(mean, std, size=(128, len(mean))), lower, upper)
        population[0] = best_parameters
        population[1] = np.clip(mean, lower, upper)
        results = [evaluate(candidate) for candidate in population]
        order = sorted(range(len(population)), key=lambda index: results[index][0])
        if results[order[0]][0] < best[0]:
            best_parameters = population[order[0]].copy()
            best = results[order[0]]
        elite = population[order[:16]]
        mean = 0.74 * elite.mean(axis=0) + 0.26 * mean
        std = np.maximum(0.74 * elite.std(axis=0) + 0.26 * std, floor)
        history.append({"generation": generation, **best[1]})
        print("GEN", generation, json.dumps(best[1]), flush=True)
        if best[1]["full_valid"] and best[1]["at_goal"] and generation >= 10:
            break
    _, metrics, trace = evaluate(best_parameters, record=True)
    output = Path("logs/unified_physical_reference")
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "physical_task_rollout_start_000.npz", parameters=best_parameters, trace=trace, balance=balance, switch_step=SWITCH_STEP)
    report = {"method": "unified MuJoCo strike-and-recovery optimization", "dimensions": len(mean), "switch_step": SWITCH_STEP, "parameters": best_parameters.tolist(), "best": metrics, "history": history, "runtime_s": time.perf_counter() - started}
    (output / "physical_task_rollout_start_000.json").write_text(json.dumps(report, indent=2) + "\n")
    ledger = {"attempts": [{"method": "goal-directed physical strike", "first_invalid_frame": 43.75, "task": {"transient_position_error_m": 0.0379, "transient_orientation_error_rad": 0.2721}, "switch_reason": "transient goal crossing did not settle at goal"}, {"method": "joint physical strike and early recovery", "first_invalid_frame": metrics["first_invalid_frame"], "invalid_reason": metrics["invalid_reason"], "task": {"position_error_m": metrics["final_position_error_m"], "orientation_error_rad": metrics["final_orientation_error_rad"], "at_goal": metrics["at_goal"]}, "best_physical_horizon": 300 if metrics["full_valid"] else metrics["first_invalid_frame"]}]}
    (output / "experiment_ledger_start_000.json").write_text(json.dumps(ledger, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
