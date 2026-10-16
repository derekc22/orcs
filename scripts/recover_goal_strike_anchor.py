#!/usr/bin/env python3
"""Recover balance from the actual post-contact state of the goal strike."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from orcs.cli.refine_diffusion_reference import _build_model
from orcs.tasks.uolm.dynamics_refinement import (
    DynamicsRefinementConfig,
    _DynamicRollout,
)
from orcs.tasks.uolm.generated_reference import GeneratedReferenceAdapter
from orcs.tasks.uolm.physical_reference_generator import (
    PhysicalReferenceConfig,
    PhysicalReferenceGenerator,
)

ANCHOR_FRAME = int(os.environ.get("GOAL_RECOVERY_ANCHOR", "35"))
SOURCE_STEM = os.environ.get("GOAL_STRIKE_SOURCE", "goal_strike_start_000")
OUTPUT_STEM = os.environ.get("GOAL_RECOVERY_OUTPUT", "goal_strike_recovery")
DIFFUSION_START = int(os.environ.get("PHYSICAL_START", "0"))


def main() -> None:
    source = np.load(f"logs/unified_physical_reference/{SOURCE_STEM}.npz")
    prior = (
        np.load(
            "logs/direct_contact_dynamics_knot117/direct_contact_dynamics_start_000.npz"
        )["analytical_through140_trajectory"]
        if DIFFUSION_START == 0
        else np.load(
            f"logs/unified_physical_reference/diffusion_start_{DIFFUSION_START:03d}.npz"
        )["trajectory"]
    )
    reference = GeneratedReferenceAdapter(
        device=torch.device("cpu"), dt=0.02, fk_batch_size=256
    )(torch.as_tensor(prior))
    engine = _DynamicRollout(
        _build_model(), reference, DynamicsRefinementConfig(run_stage_b=False)
    )
    generator = PhysicalReferenceGenerator(
        engine, PhysicalReferenceConfig(optimize_initial_stance=False)
    )
    model = engine.model
    if ANCHOR_FRAME <= 20:
        anchor_index = ANCHOR_FRAME * engine.config.substeps
        row = source["trace"][anchor_index]
    else:
        previous = np.load(
            f"logs/unified_physical_reference/{OUTPUT_STEM}_start_{DIFFUSION_START:03d}.npz"
        )
        anchor_index = (
            ANCHOR_FRAME - int(previous["anchor_frame"])
        ) * engine.config.substeps
        row = previous["local_trace"][anchor_index]
    data0 = mujoco.MjData(model)
    data0.qpos[:] = row[: model.nq]
    data0.qvel[:] = row[model.nq : model.nq + model.nv]
    data0.ctrl[:] = row[model.nq + model.nv : model.nq + model.nv + model.nu]
    mujoco.mj_forward(model, data0)
    balance = np.flatnonzero(
        np.isin(engine.actuator_joint_indexes, engine.balance_indexes)
    )
    arms = np.setdiff1d(np.arange(model.nu), balance)
    arm_target = data0.ctrl[arms].copy()
    remaining_steps = (engine.horizon - 1 - ANCHOR_FRAME) * engine.config.substeps
    goal_position = engine.reference_object_pos[-1]
    goal_quaternion = engine.reference_object_quat[-1]

    def evaluate(targets: np.ndarray, record: bool = False):
        data = mujoco.MjData(model)
        data.qpos[:] = data0.qpos
        data.qvel[:] = data0.qvel
        mujoco.mj_forward(model, data)
        control = np.empty(model.nu)
        control[balance] = targets
        control[arms] = arm_target
        first_invalid = None
        reason = None
        cost = 0.0
        trace = []
        for step in range(remaining_steps):
            data.ctrl[:] = control
            mujoco.mj_step(model, data)
            physical = generator._physical(data)
            invalid = generator._invalid_reason(data, physical)
            velocity = data.qvel[engine.robot_dof : engine.robot_dof + 6]
            support = int(np.sum(physical["foot_contact"]))
            cost += (
                1500.0 * max(0.0, 0.52 - data.qpos[engine.robot_qpos + 2]) ** 2
                + 700.0 * physical["tilt"] ** 2
                + 35.0 * float(np.square(velocity).sum())
                + 700.0 * float(support == 0)
                + 10.0 * float(support == 1)
                + 4.0 * float(np.square(physical["foot_slip"]).sum())
            )
            if invalid is not None and first_invalid is None:
                first_invalid = step
                reason = invalid
            if record:
                trace.append(
                    np.r_[
                        data.qpos.copy(),
                        data.qvel.copy(),
                        data.ctrl.copy(),
                        data.actuator_force.copy(),
                        physical["foot_contact"],
                    ]
                )
            if first_invalid is not None and step > first_invalid + 3 and not record:
                break
        position_error = float(
            np.linalg.norm(
                data.qpos[engine.object_qpos : engine.object_qpos + 3] - goal_position
            )
        )
        orientation_error = float(
            2.0
            * np.arccos(
                np.clip(
                    abs(
                        np.dot(
                            data.qpos[engine.object_qpos + 3 : engine.object_qpos + 7],
                            goal_quaternion,
                        )
                    ),
                    0.0,
                    1.0,
                )
            )
        )
        regularization = 2.0 * float(np.square(targets - data0.ctrl[balance]).mean())
        rank = (
            (
                0,
                position_error >= 0.15 or orientation_error >= 0.35,
                cost + regularization,
            )
            if first_invalid is None
            else (1, -first_invalid, cost + regularization)
        )
        metrics = {
            "first_invalid_frame": None
            if first_invalid is None
            else ANCHOR_FRAME + first_invalid / engine.config.substeps,
            "invalid_reason": reason,
            "final_root_height_m": float(data.qpos[engine.robot_qpos + 2]),
            "final_torso_tilt_deg": float(
                np.rad2deg(generator._physical(data)["tilt"])
            ),
            "final_position_error_m": position_error,
            "final_orientation_error_rad": orientation_error,
            "at_goal": bool(position_error < 0.15 and orientation_error < 0.35),
            "cost": float(cost),
        }
        return rank, metrics, np.asarray(trace)

    joint_ids = model.actuator_trnid[balance, 0]
    lower = np.maximum(
        model.actuator_ctrlrange[balance, 0], model.jnt_range[joint_ids, 0] + 0.01
    )
    upper = np.minimum(
        model.actuator_ctrlrange[balance, 1], model.jnt_range[joint_ids, 1] - 0.01
    )
    actual = data0.qpos[engine.actuator_qpos[balance]]
    mean = np.clip(0.35 * data0.ctrl[balance] + 0.65 * actual, lower, upper)
    rng = np.random.default_rng(210937 + DIFFUSION_START + ANCHOR_FRAME)
    std = np.minimum(0.28 * (upper - lower), 0.48)
    best_target = mean.copy()
    best = evaluate(best_target)
    history = []
    started = time.perf_counter()
    for generation in range(44):
        population = np.clip(
            rng.normal(mean, std, size=(112, len(balance))), lower, upper
        )
        population[0] = best_target
        population[1] = np.clip(data0.ctrl[balance], lower, upper)
        results = [evaluate(candidate) for candidate in population]
        order = sorted(range(len(population)), key=lambda index: results[index][0])
        if results[order[0]][0] < best[0]:
            best_target = population[order[0]].copy()
            best = results[order[0]]
        elite = population[order[:14]]
        mean = 0.72 * elite.mean(axis=0) + 0.28 * mean
        std = np.maximum(0.72 * elite.std(axis=0) + 0.28 * std, 0.008)
        history.append({"generation": generation, **best[1]})
        print("GEN", generation, json.dumps(best[1]), flush=True)
        if best[0][0] == 0 and best[1]["at_goal"] and generation >= 8:
            break
    _, metrics, trace = evaluate(best_target, record=True)
    output = Path("logs/unified_physical_reference")
    output.mkdir(parents=True, exist_ok=True)
    artifact = (
        f"{OUTPUT_STEM}_start_{DIFFUSION_START:03d}"
        if ANCHOR_FRAME <= 20
        else f"{OUTPUT_STEM}_{ANCHOR_FRAME:03d}_start_{DIFFUSION_START:03d}"
    )
    np.savez_compressed(
        output / f"{artifact}.npz",
        anchor_frame=ANCHOR_FRAME,
        anchor_index=anchor_index,
        anchor_qpos=data0.qpos,
        anchor_qvel=data0.qvel,
        target=best_target,
        balance=balance,
        local_trace=trace,
    )
    report = {
        "method": "actual-state recovery after goal-directed physical strike",
        "anchor_frame": ANCHOR_FRAME,
        "dimensions": len(balance),
        "best": metrics,
        "history": history,
        "runtime_s": time.perf_counter() - started,
    }
    (output / f"{artifact}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
