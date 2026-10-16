#!/usr/bin/env python3
"""Uninterrupted validation and extraction of the physical start-0 reference."""

from __future__ import annotations

import json
import os
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


def main() -> None:
    start = int(os.environ.get("PHYSICAL_START", "0"))
    second_anchor = int(os.environ.get("RECOVERY_ANCHOR", "260"))
    strike = np.load(
        f"logs/unified_physical_reference/settled_physical_strike_start_{start:03d}.npz"
    )
    recovery = np.load(
        f"logs/unified_physical_reference/settled_goal_recovery_start_{start:03d}.npz"
    )
    recovery2 = np.load(
        f"logs/unified_physical_reference/settled_goal_recovery_{second_anchor:03d}_start_{start:03d}.npz"
    )
    anchor = np.load(
        "logs/unified_physical_reference/contact_kick_stance_start_000.npz"
    )
    start0_diffusion = np.load(
        "logs/direct_contact_dynamics_knot117/direct_contact_dynamics_start_000.npz"
    )["analytical_through140_trajectory"]
    diffusion = (
        start0_diffusion
        if start == 0
        else np.load(
            f"logs/unified_physical_reference/diffusion_start_{start:03d}.npz"
        )["trajectory"]
    )
    reference = GeneratedReferenceAdapter(
        device=torch.device("cpu"), dt=0.02, fk_batch_size=256
    )(torch.as_tensor(diffusion))
    engine = _DynamicRollout(
        _build_model(), reference, DynamicsRefinementConfig(run_stage_b=False)
    )
    generator = PhysicalReferenceGenerator(
        engine, PhysicalReferenceConfig(optimize_initial_stance=False)
    )
    model = engine.model
    source_trace = strike["trace"]
    recovery_trace = recovery["local_trace"]
    recovery2_trace = recovery2["local_trace"]
    # The first anchor is source sample 80; the second is recovery sample 960.
    controls = np.concatenate(
        [
            source_trace[:81, model.nq + model.nv : model.nq + model.nv + model.nu],
            recovery_trace[
                : (second_anchor - 20) * engine.config.substeps + 1,
                model.nq + model.nv : model.nq + model.nv + model.nu,
            ],
            recovery2_trace[:, model.nq + model.nv : model.nq + model.nv + model.nu],
        ],
        axis=0,
    )[:1200]
    names = np.asarray(engine.joint_names)[engine.actuator_joint_indexes]
    right = np.asarray(
        [
            i
            for i, name in enumerate(names)
            if name.startswith("right_")
            and any(token in name for token in ("hip_", "knee", "ankle_"))
        ]
    )
    right_dof = model.jnt_dofadr[model.actuator_trnid[right, 0]].astype(np.int32)
    data = mujoco.MjData(model)
    data.qpos[:] = anchor["initial_qpos"]
    data.qpos[engine.robot_qpos : engine.robot_qpos + 2] += (
        engine.reference_object_pos[0, :2] - start0_diffusion[0, 0, 38:40]
    )
    data.qvel[:] = anchor["initial_qvel"]
    data.qvel[right_dof] = strike["parameters"][:6]
    data.qvel[engine.robot_dof : engine.robot_dof + 2] += strike["parameters"][6:8]
    data.qpos[engine.object_qpos : engine.object_qpos + 3] = (
        engine.reference_object_pos[0]
    )
    data.qpos[engine.object_qpos + 3 : engine.object_qpos + 7] = (
        engine.reference_object_quat[0]
    )
    data.qvel[engine.object_dof : engine.object_dof + 6] = 0.0
    mujoco.mj_forward(model, data)
    initial_qpos = data.qpos.copy()
    initial_qvel = data.qvel.copy()
    full_trace = []
    frame_qpos = [initial_qpos.copy()]
    frame_qvel = [initial_qvel.copy()]
    frame_ctrl = [controls[0].copy()]
    first_invalid = None
    reason = None
    forbidden_samples = 0
    maximum_lift = 0.0
    object_contact_samples = 0
    maximum_penetration = 0.0
    for step, control in enumerate(controls):
        data.ctrl[:] = control
        mujoco.mj_step(model, data)
        physical = generator._physical(data)
        invalid = generator._invalid_reason(data, physical)
        if invalid is not None and first_invalid is None:
            first_invalid = step
            reason = invalid
        forbidden_samples += int(bool(physical["forbidden"]))
        object_contact_samples += int(physical["object_contact"])
        maximum_lift = max(
            maximum_lift,
            float(
                data.qpos[engine.object_qpos + 2] - engine.reference_object_pos[0, 2]
            ),
        )
        maximum_penetration = max(maximum_penetration, physical["penetration"])
        full_trace.append(
            np.r_[
                data.qpos.copy(),
                data.qvel.copy(),
                data.ctrl.copy(),
                data.actuator_force.copy(),
                physical["foot_contact"],
                physical["hand_contact"],
                float(bool(physical["forbidden"])),
                physical["penetration"],
            ]
        )
        if (
            step % engine.config.substeps == engine.config.substeps - 1
            and len(frame_qpos) < engine.horizon
        ):
            frame_qpos.append(data.qpos.copy())
            frame_qvel.append(data.qvel.copy())
            frame_ctrl.append(data.ctrl.copy())
    goal_position = engine.reference_object_pos[-1]
    goal_quaternion = engine.reference_object_quat[-1]
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
    at_goal = position_error < 0.15 and orientation_error < 0.35
    anchor20_qpos_error = float(
        np.max(np.abs(full_trace[80][: model.nq] - recovery["anchor_qpos"]))
    )
    anchor20_qvel_error = float(
        np.max(
            np.abs(
                full_trace[80][model.nq : model.nq + model.nv] - recovery["anchor_qvel"]
            )
        )
    )
    second_trace_index = 80 + (second_anchor - 20) * engine.config.substeps + 1
    anchor260_qpos_error = float(
        np.max(
            np.abs(
                full_trace[second_trace_index][: model.nq] - recovery2["anchor_qpos"]
            )
        )
    )
    anchor260_qvel_error = float(
        np.max(
            np.abs(
                full_trace[second_trace_index][model.nq : model.nq + model.nv]
                - recovery2["anchor_qvel"]
            )
        )
    )
    report = {
        "method": "uninterrupted physical validation",
        "start": start,
        "frames": engine.horizon,
        "first_invalid_frame": None
        if first_invalid is None
        else first_invalid / engine.config.substeps,
        "invalid_reason": reason,
        "full_valid": first_invalid is None and forbidden_samples == 0,
        "at_goal": at_goal,
        "final_position_error_m": position_error,
        "final_orientation_error_rad": orientation_error,
        "maximum_lift_m": maximum_lift,
        "object_contact_duration_s": object_contact_samples * model.opt.timestep,
        "forbidden_support_duration_s": forbidden_samples * model.opt.timestep,
        "maximum_penetration_m": maximum_penetration,
        "final_root_height_m": float(data.qpos[engine.robot_qpos + 2]),
        "final_torso_tilt_deg": float(np.rad2deg(generator._physical(data)["tilt"])),
        "anchor_state_errors": {
            "frame20_qpos": anchor20_qpos_error,
            "frame20_qvel": anchor20_qvel_error,
            f"frame{second_anchor}_qpos": anchor260_qpos_error,
            f"frame{second_anchor}_qvel": anchor260_qvel_error,
        },
        "object_pose_writes_after_initialization": 0,
        "external_object_forces": 0,
    }
    output = Path(f"logs/unified_physical_reference/final_start_{start:03d}")
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "physical_reference.npz",
        initial_qpos=initial_qpos,
        initial_qvel=initial_qvel,
        qpos=np.asarray(frame_qpos),
        qvel=np.asarray(frame_qvel),
        ctrl=np.asarray(frame_ctrl),
        simulation_trace=np.asarray(full_trace),
        diffusion_prior=diffusion,
        goal_position=goal_position,
        goal_quaternion=goal_quaternion,
    )
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
