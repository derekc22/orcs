#!/usr/bin/env python3
"""Optimize the settled object goal while retaining proven physical recoveries."""

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


FIRST_SWITCH = 81
SECOND_SWITCH = 882


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def main() -> None:
    anchor = np.load("logs/unified_physical_reference/contact_kick_stance_start_000.npz")
    strike_prior = np.load("logs/unified_physical_reference/goal_strike_start_000.npz")
    recovery = np.load("logs/unified_physical_reference/goal_strike_recovery_start_000.npz")
    recovery2 = np.load("logs/unified_physical_reference/goal_strike_recovery_220_start_000.npz")
    diffusion = np.load("logs/direct_contact_dynamics_knot117/direct_contact_dynamics_start_000.npz")["analytical_through140_trajectory"]
    reference = GeneratedReferenceAdapter(device=torch.device("cpu"), dt=0.02, fk_batch_size=256)(torch.as_tensor(diffusion))
    engine = _DynamicRollout(_build_model(), reference, DynamicsRefinementConfig(run_stage_b=False))
    generator = PhysicalReferenceGenerator(engine, PhysicalReferenceConfig(optimize_initial_stance=False))
    model = engine.model
    names = np.asarray(engine.joint_names)[engine.actuator_joint_indexes]
    right = np.asarray([i for i, name in enumerate(names) if name.startswith("right_") and any(token in name for token in ("hip_", "knee", "ankle_"))])
    right_dof = model.jnt_dofadr[model.actuator_trnid[right, 0]].astype(np.int32)
    joint_ids = model.actuator_trnid[right, 0]
    lower_right = model.jnt_range[joint_ids, 0] + 0.001
    upper_right = model.jnt_range[joint_ids, 1] - 0.001
    q0 = strike_prior["q0"]
    balance = recovery["balance"]
    goal_position = engine.reference_object_pos[-1]
    goal_quaternion = engine.reference_object_quat[-1]
    total_steps = (engine.horizon - 1) * engine.config.substeps + 1

    def timing(parameters: np.ndarray) -> tuple[float, float, float]:
        return (
            2.0 + 24.0 / (1.0 + np.exp(-parameters[14])),
            35.0 + 105.0 / (1.0 + np.exp(-parameters[15])),
            12.0 + 88.0 / (1.0 + np.exp(-parameters[16])),
        )

    def evaluate(parameters: np.ndarray, record: bool = False):
        rise, retract_start, retract_duration = timing(parameters)
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
        forbidden = 0
        maximum_lift = 0.0
        contact_samples = 0
        trace = []
        for step in range(total_steps):
            if step < FIRST_SWITCH:
                amount = smoothstep(step / rise) * (1.0 - smoothstep((step - retract_start) / retract_duration))
                control = anchor["control"].copy()
                control[right] = np.clip(q0 + amount * parameters[8:14], lower_right, upper_right)
            else:
                control = anchor["control"].copy()
                control[balance] = recovery["target"] if step < SECOND_SWITCH else recovery2["target"]
            data.ctrl[:] = np.clip(control, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
            mujoco.mj_step(model, data)
            physical = generator._physical(data)
            invalid = generator._invalid_reason(data, physical)
            if invalid is not None and first_invalid is None:
                first_invalid = step
                reason = invalid
            forbidden += int(bool(physical["forbidden"]))
            contact_samples += int(physical["object_contact"])
            maximum_lift = max(maximum_lift, float(data.qpos[engine.object_qpos + 2] - engine.reference_object_pos[0, 2]))
            if record:
                trace.append(np.r_[data.qpos.copy(), data.qvel.copy(), data.ctrl.copy(), data.actuator_force.copy(), physical["foot_contact"], physical["hand_contact"], float(bool(physical["forbidden"])), physical["penetration"]])
            if first_invalid is not None and step > first_invalid + 3 and not record:
                break
        position_error = float(np.linalg.norm(data.qpos[engine.object_qpos : engine.object_qpos + 3] - goal_position))
        orientation_error = float(2.0 * np.arccos(np.clip(abs(np.dot(data.qpos[engine.object_qpos + 3 : engine.object_qpos + 7], goal_quaternion)), 0.0, 1.0)))
        object_speed = float(np.linalg.norm(data.qvel[engine.object_dof : engine.object_dof + 6]))
        full_valid = first_invalid is None and forbidden == 0
        at_goal = position_error < 0.15 and orientation_error < 0.35
        survival = total_steps if first_invalid is None else first_invalid
        task_cost = 9000.0 * position_error**2 + 1200.0 * orientation_error**2 + 150.0 * object_speed**2
        prior = 0.08 * float(np.square(parameters[:6]).sum()) + 4.0 * float(np.square(parameters[6:8]).sum()) + 1.5 * float(np.square(parameters[8:14] - strike_prior["parameters"][8:14]).sum())
        rank = (0, task_cost + prior) if full_valid and at_goal else ((1, task_cost + prior) if full_valid else (2, -survival, task_cost + prior))
        metrics = {
            "first_invalid_frame": None if first_invalid is None else first_invalid / engine.config.substeps,
            "invalid_reason": reason,
            "full_valid": full_valid,
            "at_goal": at_goal,
            "final_position_error_m": position_error,
            "final_orientation_error_rad": orientation_error,
            "final_object_speed": object_speed,
            "maximum_lift_m": maximum_lift,
            "contact_duration_s": contact_samples * model.opt.timestep,
            "forbidden_samples": forbidden,
            "final_root_height_m": float(data.qpos[engine.robot_qpos + 2]),
            "final_torso_tilt_deg": float(np.rad2deg(generator._physical(data)["tilt"])),
            "task_cost": task_cost,
        }
        return rank, metrics, np.asarray(trace)

    mean = strike_prior["parameters"].copy()
    lower = np.r_[np.full(6, -22.0), [-2.0, -2.0], lower_right - q0, [-4.0, -4.0, -4.0]]
    upper = np.r_[np.full(6, 22.0), [2.0, 2.0], upper_right - q0, [4.0, 4.0, 4.0]]
    std = np.r_[np.full(6, 0.75), [0.08, 0.08], np.full(6, 0.09), [0.25, 0.25, 0.25]]
    floor = np.r_[np.full(6, 0.06), [0.008, 0.008], np.full(6, 0.005), [0.025, 0.025, 0.025]]
    rng = np.random.default_rng(412867)
    best_parameters = mean.copy()
    best = evaluate(best_parameters)
    history = []
    started = time.perf_counter()
    for generation in range(48):
        population = np.clip(rng.normal(mean, std, size=(96, len(mean))), lower, upper)
        population[0] = best_parameters
        results = [evaluate(candidate) for candidate in population]
        order = sorted(range(len(population)), key=lambda index: results[index][0])
        if results[order[0]][0] < best[0]:
            best_parameters = population[order[0]].copy()
            best = results[order[0]]
        full = [index for index in order if results[index][1]["full_valid"]]
        selected = full[:12] if len(full) >= 4 else order[:12]
        elite = population[selected]
        mean = 0.68 * elite.mean(axis=0) + 0.32 * best_parameters
        std = np.maximum(0.72 * elite.std(axis=0) + 0.28 * std, floor)
        history.append({"generation": generation, "valid_candidates": len(full), **best[1]})
        print("GEN", generation, json.dumps(history[-1]), flush=True)
        if best[1]["full_valid"] and best[1]["at_goal"] and generation >= 8:
            break
    _, metrics, trace = evaluate(best_parameters, record=True)
    output = Path("logs/unified_physical_reference")
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "settled_goal_rollout_start_000.npz", parameters=best_parameters, trace=trace, first_switch=FIRST_SWITCH, second_switch=SECOND_SWITCH)
    report = {"method": "full-horizon MuJoCo settled-goal optimization", "dimensions": len(mean), "parameters": best_parameters.tolist(), "best": metrics, "history": history, "runtime_s": time.perf_counter() - started}
    (output / "settled_goal_rollout_start_000.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
