#!/usr/bin/env python3
"""Search a supported two-palm launch, then save its physical rollout.

This is an experimental phase used by the unified reference generator.  The
box state is initialized once and is never written during a candidate rollout.
"""

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


GROUPS = ("hip_pitch", "knee", "ankle_pitch", "hip_roll", "ankle_roll", "waist_pitch", "waist_roll")


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def main() -> None:
    stable = np.load("/tmp/contact_static_balanced.npz")
    prior = np.load("logs/direct_contact_dynamics_knot117/direct_contact_dynamics_start_000.npz")["analytical_through140_trajectory"]
    reference = GeneratedReferenceAdapter(device=torch.device("cpu"), dt=0.02, fk_batch_size=256)(torch.as_tensor(prior))
    engine = _DynamicRollout(_build_model(), reference, DynamicsRefinementConfig(run_stage_b=False, seed=42))
    generator = PhysicalReferenceGenerator(engine, PhysicalReferenceConfig(optimize_initial_stance=False))
    model = engine.model
    balance = np.flatnonzero(np.isin(engine.actuator_joint_indexes, engine.balance_indexes))
    arms = np.setdiff1d(np.arange(model.nu), balance)
    arm_qpos = engine.actuator_qpos[arms]
    arm_neutral = stable["arm_pose"].copy()
    direction = np.r_[reference.object_pos_local[0, -1, :2].numpy() - reference.object_pos_local[0, 0, :2].numpy(), 0.0]
    direction /= np.linalg.norm(direction)

    initial = mujoco.MjData(model)
    initial.qpos[:] = stable["base_qpos"]
    initial.qvel[:] = stable["base_qvel"]
    initial.qpos[engine.robot_qpos : engine.robot_qpos + 2] += 0.08 * direction[:2]
    mujoco.mj_forward(model, initial)

    ik = mujoco.MjData(model)
    ik.qpos[:] = initial.qpos
    ik.qvel[:] = initial.qvel
    mujoco.mj_forward(model, ik)
    hand_initial = np.asarray([ik.geom_xpos[int(geom)].copy() for geom in engine.hand_geoms])
    hand_goal = hand_initial + 0.14 * direction
    joint_ids = model.actuator_trnid[arms, 0]
    lower = model.jnt_range[joint_ids, 0] + 0.01
    upper = model.jnt_range[joint_ids, 1] - 0.01

    def ik_error(values: np.ndarray) -> np.ndarray:
        ik.qpos[arm_qpos] = values
        mujoco.mj_forward(model, ik)
        hands = np.asarray([ik.geom_xpos[int(geom)] for geom in engine.hand_geoms])
        return np.r_[(hands - hand_goal).ravel() * 25.0, (values - arm_neutral) * 0.1]

    arm_extended = least_squares(ik_error, arm_neutral, bounds=(lower, upper), max_nfev=800).x
    nominal = np.empty(model.nu)
    nominal[balance] = stable["balance_target"]
    nominal[arms] = arm_neutral
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
        correction = matrix @ parameters
        first_invalid = None
        reason = None
        maximum_progress = 0.0
        hand_samples = 0
        trace = []
        for step in range(total_steps):
            # A brief deliberate contact impulse; support corrections remain
            # active afterward so candidates must recover rather than collapse.
            amplitude = 0.70 * smoothstep(step / 40.0) * (1.0 - smoothstep((step - 160.0) / 80.0))
            target = nominal + correction
            target[arms] = np.clip(arm_neutral + amplitude * (arm_extended - arm_neutral), lower, upper)
            data.ctrl[:] = np.clip(target, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
            mujoco.mj_step(model, data)
            physical = generator._physical(data)
            hand_samples += int(np.any(physical["hand_contact"]))
            invalid = generator._invalid_reason(data, physical)
            if invalid is not None and first_invalid is None:
                first_invalid = step
                reason = invalid
            position = data.qpos[engine.object_qpos : engine.object_qpos + 3]
            maximum_progress = max(maximum_progress, np.linalg.norm(object_start - object_goal) - np.linalg.norm(position - object_goal))
            if record:
                trace.append(np.r_[data.qpos.copy(), data.qvel.copy(), data.ctrl.copy(), data.actuator_force.copy()])
            if first_invalid is not None and step > first_invalid + 4 and not record:
                break
        position = data.qpos[engine.object_qpos : engine.object_qpos + 3]
        goal_error = float(np.linalg.norm(position - object_goal))
        soft = (
            2000.0 * goal_error * goal_error
            - 1000.0 * maximum_progress
            + 20.0 * float(np.square(parameters).sum())
        )
        rank = (0, goal_error, soft) if first_invalid is None else (1, -first_invalid, -maximum_progress, soft)
        metrics = {
            "first_invalid_frame": None if first_invalid is None else first_invalid / engine.config.substeps,
            "invalid_reason": reason,
            "maximum_goal_progress_m": float(maximum_progress),
            "final_goal_position_error_m": goal_error,
            "hand_contact_duration_s": hand_samples * model.opt.timestep,
            "final_root_height_m": float(data.qpos[engine.robot_qpos + 2]),
        }
        return rank, metrics, np.asarray(trace)

    rng = np.random.default_rng(8675309)
    mean = np.zeros(len(GROUPS))
    std = np.full(len(GROUPS), 0.12)
    bound = np.asarray([0.45, 0.45, 0.35, 0.30, 0.25, 0.35, 0.25])
    best_parameters = mean.copy()
    best = evaluate(mean)
    progress = []
    started = time.perf_counter()
    for generation in range(25):
        population = np.clip(rng.normal(mean, std, size=(72, len(mean))), -bound, bound)
        population[0] = best_parameters
        population[1] = 0.0
        evaluated = [evaluate(candidate) for candidate in population]
        order = sorted(range(len(population)), key=lambda index: evaluated[index][0])
        if evaluated[order[0]][0] < best[0]:
            best_parameters = population[order[0]].copy()
            best = evaluated[order[0]]
        elite = population[order[:10]]
        mean = 0.75 * elite.mean(axis=0) + 0.25 * mean
        std = np.maximum(0.75 * elite.std(axis=0) + 0.25 * std, 0.008)
        progress.append({"generation": generation, **best[1]})
        print("GEN", generation, json.dumps(best[1]), flush=True)

    _, metrics, trace = evaluate(best_parameters, record=True)
    output = Path("logs/unified_physical_reference")
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "supported_contact_launch_start_000.npz", parameters=best_parameters, trace=trace, initial_qpos=initial.qpos, initial_qvel=initial.qvel, nominal=nominal, arm_extended=arm_extended)
    report = {"method": "supported contact launch CEM", "groups": GROUPS, "parameters": best_parameters.tolist(), "best": metrics, "progress": progress, "runtime_s": time.perf_counter() - started}
    (output / "supported_contact_launch_start_000.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
