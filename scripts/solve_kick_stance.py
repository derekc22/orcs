#!/usr/bin/env python3
"""Search a physically stable raised-right-leg stance for a box strike."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from orcs.cli.refine_diffusion_reference import _build_model
from orcs.tasks.uolm.dynamics_refinement import DynamicsRefinementConfig, _DynamicRollout
from orcs.tasks.uolm.generated_reference import GeneratedReferenceAdapter
from orcs.tasks.uolm.physical_reference_generator import PhysicalReferenceConfig, PhysicalReferenceGenerator


def main() -> None:
    stance = np.load("logs/unified_physical_reference/oriented_stance_start_000.npz")
    kick = np.load("/tmp/upright_kick.npz")
    prior = np.load("logs/direct_contact_dynamics_knot117/direct_contact_dynamics_start_000.npz")["analytical_through140_trajectory"]
    reference = GeneratedReferenceAdapter(device=torch.device("cpu"), dt=0.02, fk_batch_size=256)(torch.as_tensor(prior))
    engine = _DynamicRollout(_build_model(), reference, DynamicsRefinementConfig(run_stage_b=False))
    generator = PhysicalReferenceGenerator(engine, PhysicalReferenceConfig(stance_parameters=tuple(stance["parameters"]), desired_stance_height_m=0.75, lqr_settle_frames=60))
    model = engine.model
    balance = generator.balance_actuators
    arms = np.setdiff1d(np.arange(model.nu), balance)
    generator.nominal_control[arms] = stance["arm_pose"]
    generator.initial_data.ctrl[:] = generator.nominal_control
    equilibrium, _, _ = generator._lqr_state()
    names = np.asarray(engine.joint_names)[engine.actuator_joint_indexes]
    right = np.asarray([index for index, name in enumerate(names) if name.startswith("right_") and any(token in name for token in ("hip_", "knee", "ankle_"))])
    adjustable = np.asarray([index for index, name in enumerate(names) if (name.startswith("left_") and any(token in name for token in ("hip_", "knee", "ankle_"))) or name.startswith("waist_")])
    right_qpos = engine.actuator_qpos[right]
    contact_ready = os.environ.get("KICK_CONTACT_READY") == "1"
    qs = (
        np.asarray([-0.17676479, -1.31215351, 0.01628475, 1.03555880, -0.86815567, 0.00401289])
        if contact_ready
        else kick["qs"]
    )
    base_quat = equilibrium.qpos[engine.robot_qpos + 3 : engine.robot_qpos + 7].copy()
    base_rot = Rotation.from_quat(np.r_[base_quat[1:], base_quat[0]])
    base_control = generator.nominal_control.copy()
    base_control[right] = qs
    total_steps = 360  # 90 reference frames: enough to establish a kick anchor.

    def decode(parameters: np.ndarray):
        data = mujoco.MjData(model)
        data.qpos[:] = equilibrium.qpos
        data.qvel[:] = 0.0
        data.qpos[right_qpos] = qs
        data.qpos[engine.robot_qpos : engine.robot_qpos + 3] += parameters[:3]
        delta = Rotation.from_rotvec(np.r_[parameters[3:5], 0.0])
        quat = (delta * base_rot).as_quat()
        data.qpos[engine.robot_qpos + 3 : engine.robot_qpos + 7] = np.r_[quat[3], quat[:3]]
        control = base_control.copy()
        control[adjustable] += parameters[5:]
        control = np.clip(control, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
        mujoco.mj_forward(model, data)
        return data, control

    def evaluate(parameters: np.ndarray, record: bool = False):
        data, control = decode(parameters)
        initial_root = data.qpos[engine.robot_qpos : engine.robot_qpos + 3].copy()
        first_invalid = None
        reason = None
        cost = 0.0
        right_contacts = 0
        trace = []
        for step in range(total_steps):
            data.ctrl[:] = control
            mujoco.mj_step(model, data)
            physical = generator._physical(data)
            right_contacts += int(physical["foot_contact"][1])
            invalid = generator._invalid_reason(data, physical)
            root = data.qpos[engine.robot_qpos : engine.robot_qpos + 3]
            velocity = data.qvel[engine.robot_dof : engine.robot_dof + 6]
            cost += (
                2500.0 * (root[2] - initial_root[2]) ** 2
                + 300.0 * float(np.square(root[:2] - initial_root[:2]).sum())
                + 1000.0 * physical["tilt"] ** 2
                + 80.0 * float(np.square(velocity).sum())
                + 2000.0 * float(not physical["foot_contact"][0])
                + 50.0 * float(physical["foot_contact"][1])
                + 30.0 * float(np.square(parameters).mean())
            )
            if invalid is not None and first_invalid is None:
                first_invalid = step
                reason = invalid
            if record:
                trace.append(np.r_[data.qpos.copy(), data.qvel.copy(), data.ctrl.copy(), data.actuator_force.copy()])
            if first_invalid is not None and step > first_invalid + 4 and not record:
                break
        rank = (0, cost) if first_invalid is None else (1, -first_invalid, cost)
        metrics = {
            "first_invalid_frame": None if first_invalid is None else first_invalid / engine.config.substeps,
            "invalid_reason": reason,
            "final_root_height_m": float(data.qpos[engine.robot_qpos + 2]),
            "final_torso_tilt_deg": float(np.rad2deg(generator._physical(data)["tilt"])),
            "left_support_fraction": float(1.0 - right_contacts / total_steps),
            "cost": float(cost),
        }
        return rank, metrics, np.asarray(trace), data, control

    dimensions = 5 + len(adjustable)
    lower = np.r_[[-0.18, -0.18, -0.06, -0.25, -0.25], np.full(len(adjustable), -0.55)]
    upper = np.r_[[0.18, 0.18, 0.06, 0.25, 0.25], np.full(len(adjustable), 0.55)]
    rng = np.random.default_rng(51023)
    if contact_ready:
        with np.load("logs/unified_physical_reference/kick_stance_start_000.npz") as previous:
            mean = previous["parameters"].copy()
    else:
        mean = np.zeros(dimensions)
        # Shift the floating base over the left support foot as the seed.
        mean[:2] = equilibrium.xpos[engine.foot_bodies[0], :2] - equilibrium.qpos[engine.robot_qpos : engine.robot_qpos + 2]
    std = 0.20 * (upper - lower)
    best_parameters = mean.copy()
    best = evaluate(mean)
    history = []
    started = time.perf_counter()
    for generation in range(24 if contact_ready else 32):
        population = np.clip(rng.normal(mean, std, size=(80, dimensions)), lower, upper)
        population[0] = best_parameters
        population[1] = np.zeros(dimensions)
        results = [evaluate(candidate) for candidate in population]
        order = sorted(range(len(population)), key=lambda index: results[index][0])
        if results[order[0]][0] < best[0]:
            best_parameters = population[order[0]].copy()
            best = results[order[0]]
        elite = population[order[:10]]
        mean = 0.72 * elite.mean(axis=0) + 0.28 * mean
        std = np.maximum(0.72 * elite.std(axis=0) + 0.28 * std, 0.008)
        history.append({"generation": generation, **best[1]})
        print("GEN", generation, json.dumps(best[1]), flush=True)
    _, metrics, trace, final, control = evaluate(best_parameters, record=True)
    output = Path("logs/unified_physical_reference")
    output.mkdir(parents=True, exist_ok=True)
    artifact_name = "contact_kick_stance_start_000" if contact_ready else "kick_stance_start_000"
    np.savez_compressed(output / f"{artifact_name}.npz", parameters=best_parameters, trace=trace, initial_qpos=decode(best_parameters)[0].qpos, initial_qvel=decode(best_parameters)[0].qvel, final_qpos=final.qpos, final_qvel=final.qvel, control=control, qs=qs, qe=kick["qe"])
    report = {"method": "single-left-support raised-leg stance", "dimensions": dimensions, "adjustable_joints": names[adjustable].tolist(), "parameters": best_parameters.tolist(), "best": metrics, "history": history, "runtime_s": time.perf_counter() - started}
    (output / f"{artifact_name}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
