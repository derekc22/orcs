#!/usr/bin/env python3
"""Search coupled support targets around a MuJoCo-realized two-hand push."""

from __future__ import annotations

import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.optimize import least_squares

from orcs.cli.refine_diffusion_reference import _build_model
from orcs.tasks.uolm.contact_dynamics_refinement import _copy_data
from orcs.tasks.uolm.dynamics_refinement import DynamicsRefinementConfig, _DynamicRollout
from orcs.tasks.uolm.generated_reference import GeneratedReferenceAdapter
from orcs.tasks.uolm.physical_reference_generator import (
    PhysicalReferenceConfig,
    PhysicalReferenceGenerator,
)


STANCE = (
    0.00933302576398513,
    0.3614698263450311,
    -0.042715565203923236,
    -0.05487674268634473,
    -0.045007645761688156,
    -0.028931038623671353,
    -0.22125262463558085,
    0.29931574802542654,
    -0.05556809081674143,
    0.24863855110278815,
)
TOKENS = (
    "hip_pitch",
    "knee",
    "ankle_pitch",
    "waist_pitch",
    "hip_roll",
    "ankle_roll",
    "waist_roll",
)
KNOT_FRAMES = np.asarray([60, 100, 140, 180, 220, 260, 299])


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value**3 * (10.0 + value * (-15.0 + 6.0 * value))


def main() -> None:
    with np.load(
        "logs/direct_contact_dynamics_knot117/direct_contact_dynamics_start_000.npz"
    ) as loaded:
        trajectory = loaded["analytical_through140_trajectory"]
    reference = GeneratedReferenceAdapter(
        device=torch.device("cpu"), dt=0.02, fk_batch_size=256
    )(torch.as_tensor(trajectory))
    engine = _DynamicRollout(
        _build_model(), reference, DynamicsRefinementConfig(run_stage_b=False, seed=42)
    )
    generator = PhysicalReferenceGenerator(
        engine,
        PhysicalReferenceConfig(
            robot_clearance_m=0.48,
            desired_stance_height_m=0.58,
            stance_parameters=STANCE,
            lqr_settle_frames=60,
        ),
    )
    equilibrium, gain, _ = generator._lqr_state()
    model = engine.model
    arm_actuators = np.setdiff1d(
        np.arange(model.nu), generator.balance_actuators
    )
    arm_qpos = engine.actuator_qpos[arm_actuators]
    arm_neutral = equilibrium.qpos[arm_qpos].copy()
    arm_joint_ids = model.actuator_trnid[arm_actuators, 0]
    arm_lower = model.jnt_range[arm_joint_ids, 0] + 0.02
    arm_upper = model.jnt_range[arm_joint_ids, 1] - 0.02
    object_start = equilibrium.qpos[
        engine.object_qpos : engine.object_qpos + 3
    ].copy()
    object_goal = engine.reference_object_pos[-1]
    direction = object_goal[:2] - object_start[:2]
    direction /= np.linalg.norm(direction)
    lateral = np.asarray([-direction[1], direction[0]])
    rear = object_start[:2] - 0.26 * direction
    hand_targets = np.asarray(
        [
            np.r_[rear + 0.13 * lateral, 0.38],
            np.r_[rear - 0.13 * lateral, 0.38],
        ]
    )
    ik_data = _copy_data(model, equilibrium)

    def ik_error(position: np.ndarray) -> np.ndarray:
        ik_data.qpos[arm_qpos] = position
        mujoco.mj_forward(model, ik_data)
        hands = np.asarray(
            [ik_data.geom_xpos[int(geom)] for geom in engine.hand_geoms]
        )
        return np.r_[(hands - hand_targets).ravel() * 10.0, (position - arm_neutral) * 0.1]

    arm_contact = least_squares(
        ik_error,
        arm_neutral,
        bounds=(arm_lower, arm_upper),
        max_nfev=300,
    ).x
    actuator_names = np.asarray(engine.joint_names)[engine.actuator_joint_indexes]
    group_matrix = np.zeros((model.nu, len(TOKENS)))
    for group, token in enumerate(TOKENS):
        for actuator, name in enumerate(actuator_names):
            if token in name:
                group_matrix[actuator, group] = 1.0
    substeps = engine.config.substeps
    sample_count = (engine.horizon - 1) * substeps + 1
    dofs = model.jnt_dofadr[model.actuator_trnid[:, 0]].astype(np.int32)

    def decode(parameters: np.ndarray) -> np.ndarray:
        knots = np.zeros((len(KNOT_FRAMES), len(TOKENS)))
        knots[1:-1] = parameters.reshape(len(KNOT_FRAMES) - 2, len(TOKENS))
        frames = np.arange(sample_count) / substeps
        return np.column_stack(
            [np.interp(frames, KNOT_FRAMES, knots[:, index]) for index in range(len(TOKENS))]
        )

    def evaluate(parameters: np.ndarray, record: bool = False):
        correction = decode(parameters)
        data = _copy_data(model, generator.initial_data)
        first_invalid = None
        reason = None
        hand_samples = 0
        forbidden_samples = 0
        maximum_usage = 0.0
        soft_cost = 0.0
        trace = []
        for sample in range(sample_count):
            frame = sample / substeps
            if frame < 60:
                amount = 0.0
            elif frame < 160:
                amount = 0.5 * smoothstep((frame - 60.0) / 100.0)
            elif frame < 200:
                amount = 0.5 * (1.0 - smoothstep((frame - 160.0) / 40.0))
            else:
                amount = 0.0
            arm_target = arm_neutral + amount * (arm_contact - arm_neutral)
            command = (
                generator.nominal_control.copy()
                if frame < 60
                else generator._lqr_control(
                    data, equilibrium, gain, arm_target=arm_target
                )
            )
            command += group_matrix @ correction[sample]
            data.ctrl[:] = np.clip(
                command,
                model.actuator_ctrlrange[:, 0],
                model.actuator_ctrlrange[:, 1],
            )
            mujoco.mj_step(model, data)
            physical = generator._physical(data)
            hand_samples += int(np.any(physical["hand_contact"]))
            forbidden_samples += int(bool(physical["forbidden"]))
            usage = float(
                np.max(np.abs(data.actuator_force) / engine.force_scale)
            )
            maximum_usage = max(maximum_usage, usage)
            soft_cost += (
                20.0 * physical["tilt"] ** 2
                + 10.0
                * max(0.0, 0.58 - data.qpos[engine.robot_qpos + 2]) ** 2
                + 2.0 * float(np.square(physical["foot_slip"]).sum())
                + 10.0 * max(0.0, usage - 0.8) ** 2
            )
            invalid = generator._invalid_reason(data, physical)
            if invalid is not None and first_invalid is None:
                first_invalid = sample
                reason = invalid
            if record:
                trace.append(
                    np.r_[
                        data.qpos.copy(),
                        data.qvel.copy(),
                        data.ctrl.copy(),
                        data.actuator_force.copy(),
                        physical["foot_contact"],
                        physical["hand_contact"],
                    ]
                )
            if first_invalid is not None and sample > first_invalid + 4 and not record:
                break
        object_position = data.qpos[
            engine.object_qpos : engine.object_qpos + 3
        ]
        object_quaternion = data.qpos[
            engine.object_qpos + 3 : engine.object_qpos + 7
        ]
        position_error = float(np.linalg.norm(object_position - object_goal))
        orientation_error = float(
            2.0
            * np.arccos(
                np.clip(
                    abs(
                        np.dot(
                            object_quaternion, engine.reference_object_quat[-1]
                        )
                    ),
                    -1.0,
                    1.0,
                )
            )
        )
        task_cost = (
            3000.0 * position_error**2
            + 300.0 * orientation_error**2
            + soft_cost / sample_count
            + 5.0 * float(np.square(parameters).mean())
            + 1000.0 * forbidden_samples / sample_count
        )
        rank = (
            (0, 0, task_cost)
            if first_invalid is None
            else (1, -first_invalid, task_cost)
        )
        metrics = {
            "first_invalid_frame": (
                None if first_invalid is None else first_invalid / substeps
            ),
            "invalid_reason": reason,
            "final_goal_position_error_m": position_error,
            "final_goal_orientation_error_rad": orientation_error,
            "object_displacement_m": float(
                np.linalg.norm(object_position - object_start)
            ),
            "hand_contact_duration_s": hand_samples * model.opt.timestep,
            "forbidden_duration_s": forbidden_samples * model.opt.timestep,
            "maximum_actuator_usage": maximum_usage,
            "final_root_height_m": float(data.qpos[engine.robot_qpos + 2]),
            "task_cost": task_cost,
        }
        return rank, metrics, np.asarray(trace)

    seed_path = Path("/tmp/balance_push_cem_best.npz")
    if seed_path.exists():
        with np.load(seed_path) as loaded:
            mean = loaded["parameters"].copy()
    else:
        mean = np.zeros((len(KNOT_FRAMES) - 2) * len(TOKENS))
    rng = np.random.default_rng(421)
    std = np.full_like(mean, 0.055)
    bounds = np.full_like(mean, 0.40)
    best_parameters = mean.copy()
    best = evaluate(best_parameters)
    progress = []
    started = time.perf_counter()
    for generation in range(20):
        population = np.clip(
            rng.normal(mean, std, size=(64, len(mean))), -bounds, bounds
        )
        population[0] = best_parameters
        population[1] = 0.0
        evaluated = [evaluate(candidate) for candidate in population]
        order = sorted(range(len(population)), key=lambda index: evaluated[index][0])
        winner = order[0]
        if evaluated[winner][0] < best[0]:
            best_parameters = population[winner].copy()
            best = evaluated[winner]
        elite = population[order[:10]]
        mean = 0.75 * elite.mean(axis=0) + 0.25 * mean
        std = np.maximum(0.75 * elite.std(axis=0) + 0.25 * std, 0.01)
        progress.append({"generation": generation, **best[1]})
        print("GEN", generation, json.dumps(best[1]), flush=True)
    _, final_metrics, trace = evaluate(best_parameters, record=True)
    output = Path("logs/unified_physical_reference")
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "balanced_push_search_start_000.npz",
        parameters=best_parameters,
        trace=trace,
        arm_contact=arm_contact,
        hand_targets=hand_targets,
    )
    report = {
        "method": "actual-MuJoCo coupled support/push control search",
        "best": final_metrics,
        "parameters": best_parameters.reshape(
            len(KNOT_FRAMES) - 2, len(TOKENS)
        ).tolist(),
        "tokens": TOKENS,
        "knot_frames": KNOT_FRAMES.tolist(),
        "runtime_s": time.perf_counter() - started,
        "progress": progress,
    }
    (output / "balanced_push_search_start_000.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
