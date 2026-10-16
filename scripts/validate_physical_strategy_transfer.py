#!/usr/bin/env python3
"""Validate the start-0 physical strategy on deterministic starts 10 and 20."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from orcs.cli.refine_diffusion_reference import _build_model
from orcs.tasks.uolm.dynamics_refinement import DynamicsRefinementConfig, _DynamicRollout
from orcs.tasks.uolm.generated_reference import GeneratedReferenceAdapter
from orcs.tasks.uolm.physical_reference_generator import PhysicalReferenceConfig, PhysicalReferenceGenerator


def main() -> None:
    start0_diffusion = np.load("logs/direct_contact_dynamics_knot117/direct_contact_dynamics_start_000.npz")["analytical_through140_trajectory"]
    strike = np.load("logs/unified_physical_reference/settled_physical_strike_start_000.npz")
    recovery = np.load("logs/unified_physical_reference/settled_goal_recovery_start_000.npz")
    recovery2 = np.load("logs/unified_physical_reference/settled_goal_recovery_260_start_000.npz")
    anchor = np.load("logs/unified_physical_reference/contact_kick_stance_start_000.npz")
    source_trace = strike["trace"]
    recovery_trace = recovery["local_trace"]
    recovery2_trace = recovery2["local_trace"]
    output_root = Path("logs/unified_physical_reference")
    for start in (10, 20):
        diffusion = np.load(output_root / f"diffusion_start_{start:03d}.npz")["trajectory"]
        reference = GeneratedReferenceAdapter(device=torch.device("cpu"), dt=0.02, fk_batch_size=256)(torch.as_tensor(diffusion))
        engine = _DynamicRollout(_build_model(), reference, DynamicsRefinementConfig(run_stage_b=False))
        generator = PhysicalReferenceGenerator(engine, PhysicalReferenceConfig(optimize_initial_stance=False))
        model = engine.model
        controls = np.concatenate(
            [
                source_trace[:81, model.nq + model.nv : model.nq + model.nv + model.nu],
                recovery_trace[:961, model.nq + model.nv : model.nq + model.nv + model.nu],
                recovery2_trace[:, model.nq + model.nv : model.nq + model.nv + model.nu],
            ],
            axis=0,
        )[:1200]
        names = np.asarray(engine.joint_names)[engine.actuator_joint_indexes]
        right = np.asarray([i for i, name in enumerate(names) if name.startswith("right_") and any(token in name for token in ("hip_", "knee", "ankle_"))])
        right_dof = model.jnt_dofadr[model.actuator_trnid[right, 0]].astype(np.int32)
        data = mujoco.MjData(model)
        data.qpos[:] = anchor["initial_qpos"]
        # Preserve the task-relative horizontal strike geometry.
        data.qpos[engine.robot_qpos : engine.robot_qpos + 2] += engine.reference_object_pos[0, :2] - start0_diffusion[0, 0, 38:40]
        data.qvel[:] = anchor["initial_qvel"]
        data.qvel[right_dof] = strike["parameters"][:6]
        data.qvel[engine.robot_dof : engine.robot_dof + 2] += strike["parameters"][6:8]
        data.qpos[engine.object_qpos : engine.object_qpos + 3] = engine.reference_object_pos[0]
        data.qpos[engine.object_qpos + 3 : engine.object_qpos + 7] = engine.reference_object_quat[0]
        data.qvel[engine.object_dof : engine.object_dof + 6] = 0.0
        mujoco.mj_forward(model, data)
        initial_qpos = data.qpos.copy()
        initial_qvel = data.qvel.copy()
        first_invalid = None
        reason = None
        forbidden = 0
        contact_samples = 0
        maximum_lift = 0.0
        trace = []
        frames_qpos = [data.qpos.copy()]
        frames_qvel = [data.qvel.copy()]
        for step, control in enumerate(controls):
            data.ctrl[:] = control
            mujoco.mj_step(model, data)
            physical = generator._physical(data)
            invalid = generator._invalid_reason(data, physical)
            if invalid is not None and first_invalid is None:
                first_invalid = step
                reason = invalid
            forbidden += int(bool(physical["forbidden"]))
            contact_samples += int(physical["object_contact"])
            maximum_lift = max(maximum_lift, float(data.qpos[engine.object_qpos + 2] - engine.reference_object_pos[0, 2]))
            trace.append(np.r_[data.qpos.copy(), data.qvel.copy(), data.ctrl.copy(), data.actuator_force.copy(), physical["foot_contact"], physical["hand_contact"], float(bool(physical["forbidden"])), physical["penetration"]])
            if step % engine.config.substeps == engine.config.substeps - 1 and len(frames_qpos) < engine.horizon:
                frames_qpos.append(data.qpos.copy())
                frames_qvel.append(data.qvel.copy())
        position_error = float(np.linalg.norm(data.qpos[engine.object_qpos : engine.object_qpos + 3] - engine.reference_object_pos[-1]))
        orientation_error = float(2.0 * np.arccos(np.clip(abs(np.dot(data.qpos[engine.object_qpos + 3 : engine.object_qpos + 7], engine.reference_object_quat[-1])), 0.0, 1.0)))
        report = {
            "start": start,
            "full_valid": first_invalid is None and forbidden == 0,
            "first_invalid_frame": None if first_invalid is None else first_invalid / engine.config.substeps,
            "invalid_reason": reason,
            "at_goal": position_error < 0.15 and orientation_error < 0.35,
            "final_position_error_m": position_error,
            "final_orientation_error_rad": orientation_error,
            "maximum_lift_m": maximum_lift,
            "object_contact_duration_s": contact_samples * model.opt.timestep,
            "forbidden_support_duration_s": forbidden * model.opt.timestep,
            "final_root_height_m": float(data.qpos[engine.robot_qpos + 2]),
            "final_torso_tilt_deg": float(np.rad2deg(generator._physical(data)["tilt"])),
            "object_pose_writes_after_initialization": 0,
            "external_object_forces": 0,
        }
        output = output_root / f"final_start_{start:03d}"
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output / "physical_reference.npz", initial_qpos=initial_qpos, initial_qvel=initial_qvel, qpos=np.asarray(frames_qpos), qvel=np.asarray(frames_qvel), simulation_trace=np.asarray(trace), diffusion_prior=diffusion, goal_position=engine.reference_object_pos[-1], goal_quaternion=engine.reference_object_quat[-1])
        (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
