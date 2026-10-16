#!/usr/bin/env python3
"""Find a stable rear-face palm push with palm normals along the task path."""

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


GROUPS = ("hip_pitch", "knee", "ankle_pitch", "hip_roll", "ankle_roll", "waist_pitch", "waist_roll")
ORIENTED_ARMS = np.asarray([
    -0.69739931, -0.03505290, 0.66769302, 1.41445548, 1.95120049,
    -0.75436713, 0.13481834, -1.56351571, 1.37981234, 0.15308484,
    -0.09123293, -0.30813114, -0.01854410, -0.32148771,
])


def main() -> None:
    stable = np.load("/tmp/contact_static_balanced.npz")
    prior = np.load("logs/direct_contact_dynamics_knot117/direct_contact_dynamics_start_000.npz")["analytical_through140_trajectory"]
    reference = GeneratedReferenceAdapter(device=torch.device("cpu"), dt=0.02, fk_batch_size=256)(torch.as_tensor(prior))
    engine = _DynamicRollout(_build_model(), reference, DynamicsRefinementConfig(run_stage_b=False, seed=42))
    generator = PhysicalReferenceGenerator(engine, PhysicalReferenceConfig(optimize_initial_stance=False))
    model = engine.model
    balance = np.flatnonzero(np.isin(engine.actuator_joint_indexes, engine.balance_indexes))
    arms = np.setdiff1d(np.arange(model.nu), balance)
    direction = np.r_[engine.reference_object_pos[-1, :2] - engine.reference_object_pos[0, :2], 0.0]
    direction /= np.linalg.norm(direction)
    initial = mujoco.MjData(model)
    initial.qpos[:] = stable["base_qpos"]
    initial.qvel[:] = 0.0
    initial.qpos[engine.actuator_qpos[arms]] = ORIENTED_ARMS
    initial.qpos[engine.robot_qpos : engine.robot_qpos + 2] += 0.055 * direction[:2]
    mujoco.mj_forward(model, initial)
    nominal = np.empty(model.nu)
    nominal[balance] = stable["balance_target"]
    nominal[arms] = ORIENTED_ARMS
    names = np.asarray(engine.joint_names)[engine.actuator_joint_indexes]
    matrix = np.zeros((model.nu, len(GROUPS)))
    for column, token in enumerate(GROUPS):
        matrix[[token in name for name in names], column] = 1.0
    total_steps = (engine.horizon - 1) * engine.config.substeps + 1
    object_start = initial.qpos[engine.object_qpos : engine.object_qpos + 3].copy()
    object_goal = engine.reference_object_pos[-1].copy()

    def evaluate(parameters: np.ndarray, record: bool = False):
        data = mujoco.MjData(model)
        data.qpos[:] = initial.qpos
        data.qvel[:] = initial.qvel
        mujoco.mj_forward(model, data)
        target = nominal + matrix @ parameters
        first_invalid = None
        reason = None
        best_progress = 0.0
        useful_force = 0.0
        hand_time = 0
        trace = []
        for step in range(total_steps):
            data.ctrl[:] = np.clip(target, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
            mujoco.mj_step(model, data)
            physical = generator._physical(data)
            hand_time += int(np.any(physical["hand_contact"]))
            for contact_index in range(data.ncon):
                contact = data.contact[contact_index]
                pair = {int(contact.geom1), int(contact.geom2)}
                if pair & set(engine.hand_geoms.tolist()) and pair & engine.object_geoms:
                    force = np.zeros(6)
                    mujoco.mj_contactForce(model, data, contact_index, force)
                    useful_force += max(0.0, force[0] * abs(float(np.dot(contact.frame[:3], direction))))
            invalid = generator._invalid_reason(data, physical)
            if invalid is not None and first_invalid is None:
                first_invalid = step
                reason = invalid
            position = data.qpos[engine.object_qpos : engine.object_qpos + 3]
            best_progress = max(best_progress, np.linalg.norm(object_start - object_goal) - np.linalg.norm(position - object_goal))
            if record:
                trace.append(np.r_[data.qpos.copy(), data.qvel.copy(), data.ctrl.copy(), data.actuator_force.copy()])
            if first_invalid is not None and step > first_invalid + 4 and not record:
                break
        final_position = data.qpos[engine.object_qpos : engine.object_qpos + 3]
        final_error = float(np.linalg.norm(final_position - object_goal))
        task_cost = 2000.0 * final_error**2 - 500.0 * best_progress - 0.002 * useful_force + 10.0 * float(np.square(parameters).sum())
        rank = (0, task_cost) if first_invalid is None else (1, -first_invalid, task_cost)
        metrics = {
            "first_invalid_frame": None if first_invalid is None else first_invalid / engine.config.substeps,
            "invalid_reason": reason,
            "maximum_goal_progress_m": float(best_progress),
            "final_goal_position_error_m": final_error,
            "hand_contact_duration_s": hand_time * model.opt.timestep,
            "useful_contact_impulse_ns": useful_force * model.opt.timestep,
            "final_root_height_m": float(data.qpos[engine.robot_qpos + 2]),
        }
        return rank, metrics, np.asarray(trace)

    rng = np.random.default_rng(29291)
    mean = np.zeros(len(GROUPS))
    std = np.full(len(GROUPS), 0.14)
    bounds = np.asarray([0.50, 0.50, 0.40, 0.35, 0.30, 0.40, 0.30])
    best_parameters = mean.copy()
    best = evaluate(mean)
    history = []
    started = time.perf_counter()
    for generation in range(28):
        population = np.clip(rng.normal(mean, std, size=(72, len(mean))), -bounds, bounds)
        population[0] = best_parameters
        population[1] = 0.0
        results = [evaluate(candidate) for candidate in population]
        order = sorted(range(len(population)), key=lambda index: results[index][0])
        if results[order[0]][0] < best[0]:
            best_parameters = population[order[0]].copy()
            best = results[order[0]]
        elite = population[order[:10]]
        mean = 0.75 * elite.mean(axis=0) + 0.25 * mean
        std = np.maximum(0.75 * elite.std(axis=0) + 0.25 * std, 0.008)
        history.append({"generation": generation, **best[1]})
        print("GEN", generation, json.dumps(best[1]), flush=True)
    _, metrics, trace = evaluate(best_parameters, record=True)
    output = Path("logs/unified_physical_reference")
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "oriented_push_start_000.npz", parameters=best_parameters, trace=trace, initial_qpos=initial.qpos, initial_qvel=initial.qvel, nominal=nominal)
    report = {"method": "rear-face oriented-palm push", "parameters": best_parameters.tolist(), "best": metrics, "history": history, "runtime_s": time.perf_counter() - started}
    (output / "oriented_push_start_000.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
