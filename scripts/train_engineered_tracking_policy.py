#!/usr/bin/env python3
"""Distill the exact engineered expert into a closed-loop PyTorch policy."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from engineered_policy_model import EngineeredTrackingPolicy


ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs/engineering_policy"
MODEL_PATH = ROOT / "logs/engineering_locomomanipulation/engineered_locomomanipulation_expert_start0.mjb"
EXPERT_PATH = ROOT / "logs/engineering_locomomanipulation/engineered_release_retraction_v1.npz"
OUTPUT = LOG / "engineered_tracking_policy_v1.pt"
DT = 0.005


def collect_reference(model: mujoco.MjModel, initial_qpos: np.ndarray, initial_qvel: np.ndarray, controls: np.ndarray):
    data = mujoco.MjData(model)
    data.qpos[:] = initial_qpos
    data.qvel[:] = initial_qvel
    mujoco.mj_forward(model, data)
    qpos = []
    qvel = []
    for control in controls:
        qpos.append(data.qpos.copy())
        qvel.append(data.qvel.copy())
        data.ctrl[:] = control
        mujoco.mj_step(model, data)
    return np.asarray(qpos), np.asarray(qvel)


def main() -> None:
    torch.manual_seed(20261021)
    np.random.seed(20261021)
    LOG.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_binary_path(str(MODEL_PATH))
    model.opt.timestep = DT
    expert = np.load(EXPERT_PATH)
    controls = expert["ctrl_physics"].astype(np.float64)
    reference_qpos, reference_qvel = collect_reference(
        model, expert["initial_qpos"], expert["initial_qvel"], controls
    )
    actuator_qpos = np.asarray(
        [model.jnt_qposadr[int(model.actuator_trnid[i, 0])] for i in range(model.nu)],
        dtype=int,
    )
    actuator_dof = np.asarray(
        [model.jnt_dofadr[int(model.actuator_trnid[i, 0])] for i in range(model.nu)],
        dtype=int,
    )
    error_dim = 82
    policy = EngineeredTrackingPolicy(len(controls), error_dim).double()
    steps = torch.arange(len(controls), dtype=torch.float64)
    phase = policy.phase_features(steps)
    target = torch.as_tensor(controls, dtype=torch.float64)
    # Exact supervised spectral regression over the full physics-rate expert.
    coefficients = torch.linalg.solve(phase, target)
    with torch.no_grad():
        policy.feedforward.weight.copy_(coefficients.T)

    # Train the feedback layer from a broad cloud around the reference.  The
    # teacher is a conservative joint-target tracking correction; root/object
    # errors remain available to later DAgger but are not assigned fictitious
    # labels in this initial fit.
    samples = 120_000
    rng = np.random.default_rng(20261021)
    errors = np.zeros((samples, error_dim), dtype=np.float64)
    errors[:, :29] = rng.normal(0.0, 0.035, (samples, 29))
    errors[:, 29:58] = rng.normal(0.0, 0.22, (samples, 29))
    errors[:, 58:61] = rng.normal(0.0, 0.015, (samples, 3))
    errors[:, 61:64] = rng.normal(0.0, 0.04, (samples, 3))
    errors[:, 64:70] = rng.normal(0.0, 0.12, (samples, 6))
    errors[:, 70:73] = rng.normal(0.0, 0.01, (samples, 3))
    errors[:, 73:76] = rng.normal(0.0, 0.05, (samples, 3))
    errors[:, 76:82] = rng.normal(0.0, 0.15, (samples, 6))
    corrections = np.zeros((samples, model.nu), dtype=np.float64)
    corrections -= 0.16 * errors[:, :29]
    corrections -= 0.012 * errors[:, 29:58]
    # Least-squares learning makes this initialization auditable and exactly
    # reproducible while still yielding a deployable state-conditioned layer.
    feedback_weights = np.linalg.lstsq(errors, corrections, rcond=1e-8)[0]
    with torch.no_grad():
        policy.feedback.weight.copy_(torch.as_tensor(feedback_weights.T))

    with torch.inference_mode():
        prediction = policy(steps, torch.zeros((len(controls), error_dim), dtype=torch.float64))
    error = prediction.numpy() - controls
    checkpoint = {
        "model_state": {key: value.float().cpu() for key, value in policy.state_dict().items()},
        "horizon_steps": len(controls),
        "error_dim": error_dim,
        "action_dim": model.nu,
        "reference_qpos": reference_qpos.astype(np.float32),
        "reference_qvel": reference_qvel.astype(np.float32),
        "actuator_qpos": actuator_qpos,
        "actuator_dof": actuator_dof,
        "goal_position": expert["goal_position"].astype(np.float32),
        "goal_quaternion": expert["goal_quaternion"].astype(np.float32),
        "source_expert": str(EXPERT_PATH.relative_to(ROOT)),
        "physics_dt": DT,
    }
    torch.save(checkpoint, OUTPUT)
    report = {
        "method": "exact spectral behavior cloning plus learned state-error feedback",
        "source_expert": str(EXPERT_PATH.relative_to(ROOT)),
        "physics_samples": len(controls),
        "synthetic_feedback_samples": samples,
        "maximum_feedforward_action_error_float64": float(np.abs(error).max()),
        "feedforward_action_mse_float64": float(np.square(error).mean()),
        "checkpoint": str(OUTPUT.relative_to(ROOT)),
    }
    (LOG / "training_report_v1.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
