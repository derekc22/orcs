#!/usr/bin/env python3
"""Cheap exact-physics gate for a state-convergence prelude."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from evaluate_engineered_tracking_policy import strict_success
from finalize_engineered_expert import replay
from test_engineered_policy_robustness import perturbed_state


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs/engineering_policy"
MODEL_PATH = ROOT / "logs/engineering_locomomanipulation/engineered_locomomanipulation_expert_start0.mjb"
BASELINE_PATH = OUT / "engineered_hierarchical_policy_success.npz"
EXPERT_PATH = ROOT / "logs/engineering_locomomanipulation/engineered_release_retraction_v1.npz"
CHECKPOINT_PATH = OUT / "engineered_tracking_policy_v1.pt"
MICRO = dict(joint_qpos=0.0002, joint_qvel=0.001, root_xy=0.0001, root_yaw=0.0003, object_xy=0.0, object_yaw=0.0)


def controls_with_prelude(model, baseline, qpos, qvel, frames, kp, kd):
    actuator_qpos = np.asarray([model.jnt_qposadr[int(model.actuator_trnid[i, 0])] for i in range(model.nu)], dtype=int)
    actuator_dof = np.asarray([model.jnt_dofadr[int(model.actuator_trnid[i, 0])] for i in range(model.nu)], dtype=int)
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_forward(model, data)
    initial_qpos = baseline["initial_qpos"]
    initial_qvel = baseline["initial_qvel"]
    controls = []
    for step in range(frames * 4):
        feedforward = baseline["ctrl_physics"][step % 4]
        control = (
            feedforward
            - kp * (data.qpos[actuator_qpos] - initial_qpos[actuator_qpos])
            - kd * (data.qvel[actuator_dof] - initial_qvel[actuator_dof])
        )
        control = np.clip(control, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
        controls.append(control)
        data.ctrl[:] = control
        mujoco.mj_step(model, data)
    controls.extend(baseline["ctrl_physics"])
    return np.asarray(controls)


def main():
    model = mujoco.MjModel.from_binary_path(str(MODEL_PATH))
    model.opt.timestep = 0.005
    baseline = np.load(BASELINE_PATH)
    expert = np.load(EXPERT_PATH)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    states = [("nominal", baseline["initial_qpos"].copy(), baseline["initial_qvel"].copy())]
    for seed in (20261031, 20261032, 20261033):
        qpos, qvel = perturbed_state(model, expert, checkpoint, seed, MICRO)
        states.append((str(seed), qpos, qvel))
    results = []
    for frames in (2, 4, 6, 8):
        for kp, kd in ((0.5, 0.02), (1.0, 0.04), (2.0, 0.08)):
            trial_results = []
            for name, qpos, qvel in states:
                controls = controls_with_prelude(model, baseline, qpos, qvel, frames, kp, kd)
                task_steps = (161 + frames) * 4
                metrics, _ = replay(
                    model, qpos, qvel, controls[:task_steps], baseline["goal_position"], baseline["goal_quaternion"], True
                )
                continuation, _ = replay(
                    model, qpos, qvel, controls, baseline["goal_position"], baseline["goal_quaternion"], False
                )
                trial_results.append({
                    "state": name,
                    "success": strict_success(metrics, continuation),
                    "position_error_m": metrics["final_position_error_m"],
                    "orientation_error_rad": metrics["final_orientation_error_rad"],
                    "valid": metrics["full_horizon_valid"],
                    "continuation_valid": continuation["full_horizon_valid"],
                })
            item = {
                "prelude_frames": frames,
                "kp": kp,
                "kd": kd,
                "successes": sum(x["success"] for x in trial_results),
                "trials": trial_results,
            }
            results.append(item)
            print(json.dumps(item), flush=True)
    report = {
        "method": "state-convergence prelude followed by frozen accepted controls",
        "task_physics_unchanged": True,
        "results": results,
    }
    (OUT / "settling_prelude_gate.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
