#!/usr/bin/env python3
"""Find a static physical recovery target from the valid post-strike prefix."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from orcs.cli.refine_diffusion_reference import _build_model
from orcs.tasks.uolm.dynamics_refinement import DynamicsRefinementConfig, _DynamicRollout
from orcs.tasks.uolm.generated_reference import GeneratedReferenceAdapter
from orcs.tasks.uolm.physical_reference_generator import PhysicalReferenceConfig, PhysicalReferenceGenerator


ANCHOR_FRAME = int(os.environ.get("RECOVERY_ANCHOR_FRAME", "100"))


def main() -> None:
    source = np.load("logs/unified_physical_reference/recovered_physical_kick_start_000.npz")
    prior = np.load("logs/direct_contact_dynamics_knot117/direct_contact_dynamics_start_000.npz")["analytical_through140_trajectory"]
    reference = GeneratedReferenceAdapter(device=torch.device("cpu"), dt=0.02, fk_batch_size=256)(torch.as_tensor(prior))
    engine = _DynamicRollout(_build_model(), reference, DynamicsRefinementConfig(run_stage_b=False))
    generator = PhysicalReferenceGenerator(engine, PhysicalReferenceConfig(optimize_initial_stance=False))
    model = engine.model
    if ANCHOR_FRAME == 100:
        trace = source["trace"]
        anchor_index = ANCHOR_FRAME * engine.config.substeps
    else:
        previous = np.load("logs/unified_physical_reference/kick_recovery_anchor_start_000.npz")
        trace = previous["local_trace"]
        anchor_index = (ANCHOR_FRAME - int(previous["anchor_frame"])) * engine.config.substeps
    row = trace[anchor_index]
    data0 = mujoco.MjData(model)
    data0.qpos[:] = row[: model.nq]
    data0.qvel[:] = row[model.nq : model.nq + model.nv]
    data0.ctrl[:] = row[model.nq + model.nv : model.nq + model.nv + model.nu]
    mujoco.mj_forward(model, data0)
    balance = np.flatnonzero(np.isin(engine.actuator_joint_indexes, engine.balance_indexes))
    arms = np.setdiff1d(np.arange(model.nu), balance)
    arm_target = data0.ctrl[arms].copy()
    remaining_steps = (engine.horizon - 1 - ANCHOR_FRAME) * engine.config.substeps

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
        trace_local = []
        for step in range(remaining_steps):
            data.ctrl[:] = control
            mujoco.mj_step(model, data)
            physical = generator._physical(data)
            invalid = generator._invalid_reason(data, physical)
            root_velocity = data.qvel[engine.robot_dof : engine.robot_dof + 6]
            support = int(np.sum(physical["foot_contact"]))
            cost += (
                1200.0 * max(0.0, 0.58 - data.qpos[engine.robot_qpos + 2]) ** 2
                + 500.0 * physical["tilt"] ** 2
                + 30.0 * float(np.square(root_velocity).sum())
                + 500.0 * float(support == 0)
                + 8.0 * float(support == 1)
                + 3.0 * float(np.square(physical["foot_slip"]).sum())
            )
            if invalid is not None and first_invalid is None:
                first_invalid = step
                reason = invalid
            if record:
                trace_local.append(np.r_[data.qpos.copy(), data.qvel.copy(), data.ctrl.copy(), data.actuator_force.copy(), physical["foot_contact"]])
            if first_invalid is not None and step > first_invalid + 3 and not record:
                break
        regularization = 3.0 * float(np.square(targets - data0.ctrl[balance]).mean())
        rank = (0, cost + regularization) if first_invalid is None else (1, -first_invalid, cost + regularization)
        metrics = {
            "first_invalid_frame": None if first_invalid is None else ANCHOR_FRAME + first_invalid / engine.config.substeps,
            "invalid_reason": reason,
            "final_root_height_m": float(data.qpos[engine.robot_qpos + 2]),
            "final_torso_tilt_deg": float(np.rad2deg(generator._physical(data)["tilt"])),
            "cost": float(cost),
        }
        return rank, metrics, np.asarray(trace_local)

    joint_ids = model.actuator_trnid[balance, 0]
    lower = np.maximum(model.actuator_ctrlrange[balance, 0], model.jnt_range[joint_ids, 0] + 0.01)
    upper = np.minimum(model.actuator_ctrlrange[balance, 1], model.jnt_range[joint_ids, 1] - 0.01)
    actual = data0.qpos[engine.actuator_qpos[balance]]
    mean = np.clip(0.5 * data0.ctrl[balance] + 0.5 * actual, lower, upper)
    rng = np.random.default_rng(44091)
    std = np.minimum(0.22 * (upper - lower), 0.35)
    best_target = mean.copy()
    best = evaluate(best_target)
    history = []
    started = time.perf_counter()
    for generation in range(34):
        population = np.clip(rng.normal(mean, std, size=(96, len(balance))), lower, upper)
        population[0] = best_target
        population[1] = np.clip(data0.ctrl[balance], lower, upper)
        results = [evaluate(candidate) for candidate in population]
        order = sorted(range(len(population)), key=lambda index: results[index][0])
        if results[order[0]][0] < best[0]:
            best_target = population[order[0]].copy()
            best = results[order[0]]
        elite = population[order[:12]]
        mean = 0.72 * elite.mean(axis=0) + 0.28 * mean
        std = np.maximum(0.72 * elite.std(axis=0) + 0.28 * std, 0.008)
        history.append({"generation": generation, **best[1]})
        print("GEN", generation, json.dumps(best[1]), flush=True)
        if best[0][0] == 0 and generation >= 8:
            break
    _, metrics, local_trace = evaluate(best_target, record=True)
    output = Path("logs/unified_physical_reference")
    artifact = "kick_recovery_anchor_start_000" if ANCHOR_FRAME == 100 else f"kick_recovery_anchor_{ANCHOR_FRAME:03d}_start_000"
    np.savez_compressed(output / f"{artifact}.npz", anchor_frame=ANCHOR_FRAME, anchor_qpos=data0.qpos, anchor_qvel=data0.qvel, target=best_target, balance=balance, local_trace=local_trace)
    report = {"method": "automatic actual-state recovery anchor", "anchor_frame": ANCHOR_FRAME, "dimensions": len(balance), "best": metrics, "history": history, "runtime_s": time.perf_counter() - started}
    (output / f"{artifact}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
