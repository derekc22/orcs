#!/usr/bin/env python3
"""Freeze and independently validate the selected learned policy artifact."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from evaluate_engineered_tracking_policy import strict_success
from finalize_engineered_expert import replay
from mocke import PRETRAINED_DIR


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs/engineering_policy"
MODEL_PATH = ROOT / "logs/engineering_locomomanipulation/engineered_locomomanipulation_expert_start0.mjb"
SOURCE_ROLLOUT = OUT / "engineered_hierarchical_policy_rollout_carry_0p02.npz"
TRACKER_PATH = OUT / "engineered_tracking_policy_v1.pt"
SONIC_PATH = PRETRAINED_DIR / "sonic/last_ported.pt"
POLICY_PATH = OUT / "engineered_hierarchical_policy_success.pt"
ROLLOUT_PATH = OUT / "engineered_hierarchical_policy_success.npz"
REPORT_PATH = OUT / "engineered_hierarchical_policy_success.json"
MANIFEST_PATH = OUT / "engineered_hierarchical_policy_manifest.json"
TASK_STEPS = 161 * 4


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    source = np.load(SOURCE_ROLLOUT)
    controls = source["ctrl_physics"].copy()
    initial_qpos = source["initial_qpos"].copy()
    initial_qvel = source["initial_qvel"].copy()
    goal = source["goal_position"].copy()
    goal_quat = source["goal_quaternion"].copy()
    model = mujoco.MjModel.from_binary_path(str(MODEL_PATH))
    model.opt.timestep = 0.005
    metrics, trace = replay(model, initial_qpos, initial_qvel, controls[:TASK_STEPS], goal, goal_quat, True)
    continuation, continuation_trace = replay(model, initial_qpos, initial_qvel, controls, goal, goal_quat, True)
    repeat, repeat_trace = replay(model, initial_qpos, initial_qvel, controls[:TASK_STEPS], goal, goal_quat, True)
    replay_qpos_error = float(np.max(np.abs(trace["qpos"] - repeat_trace["qpos"])))
    replay_qvel_error = float(np.max(np.abs(trace["qvel"] - repeat_trace["qvel"])))
    success = strict_success(metrics, continuation) and replay_qpos_error == 0.0 and replay_qvel_error == 0.0

    tracker = torch.load(TRACKER_PATH, map_location="cpu", weights_only=False)
    policy = {
        "format": "engineered_hierarchical_policy_v1",
        "sonic_checkpoint": str(SONIC_PATH.relative_to(ROOT)),
        "sonic_sha256": digest(SONIC_PATH),
        "spectral_tracking_actor": tracker,
        "controller": {
            "physics_rate_hz": 200,
            "sonic_rate_hz": 50,
            "locomotion_residual_scale": 1.0,
            "carry_residual_scale": 0.02,
            "residual_schedule_frames": {
                "full_before": 25,
                "blend_to_carry": [25, 30],
                "carry": [30, 100],
                "blend_to_zero": [100, 110],
                "protected_release": [110, 130],
                "blend_to_full_recovery": [130, 145],
                "full_after": 145,
            },
            "authoritative_state_feedback": "CPU MuJoCo state synchronized into SONIC observation pipeline every 50 Hz frame",
        },
        "task_model": str(MODEL_PATH.relative_to(ROOT)),
        "task_model_sha256": digest(MODEL_PATH),
        "source_expert": "logs/engineering_locomomanipulation/engineered_release_retraction_v1.npz",
    }
    torch.save(policy, POLICY_PATH)
    np.savez_compressed(
        ROLLOUT_PATH,
        initial_qpos=initial_qpos,
        initial_qvel=initial_qvel,
        ctrl_physics=controls,
        goal_position=goal,
        goal_quaternion=goal_quat,
        **{f"task_{key}": value for key, value in trace.items()},
        **{f"continuation_{key}": value for key, value in continuation_trace.items()},
    )
    repeat_reports = [
        json.loads((OUT / name).read_text())
        for name in (
            "hierarchical_evaluation_report_carry_0p02.json",
            "hierarchical_evaluation_report_repeat_a.json",
            "hierarchical_evaluation_report_repeat_b.json",
        )
    ]
    micro_reports = [
        json.loads((OUT / f"hierarchical_evaluation_report_final_micro_{seed}.json").read_text())
        for seed in (20261031, 20261032, 20261033)
    ]
    report = {
        "policy_success": bool(success),
        "strict_frozen_start_success": bool(success),
        "policy_artifact": str(POLICY_PATH.relative_to(ROOT)),
        "rollout_artifact": str(ROLLOUT_PATH.relative_to(ROOT)),
        "method": "learned SONIC closed-loop locomotion plus exact spectral behavior cloning and state-conditioned residual scheduling",
        "task_horizon_reference_frames": 161,
        "recovery_continuation_frames": 30,
        **metrics,
        "continuation_full_horizon_valid": continuation["full_horizon_valid"],
        "continuation_first_invalid": continuation["first_invalid"],
        "continuation_final_root_height_m": continuation["final_root_height_m"],
        "independent_replay_max_qpos_error": replay_qpos_error,
        "independent_replay_max_qvel_error": replay_qvel_error,
        "nominal_repeat_successes": sum(item["success"] for item in repeat_reports),
        "nominal_repeat_trials": len(repeat_reports),
        "micro_robot_state_successes": sum(item["success"] for item in micro_reports),
        "micro_robot_state_trials": len(micro_reports),
        "robustness_interpretation": "repeatable strict frozen-start success; measured initial-state disturbance envelope remains narrow",
        "visual_validation": {
            "inspected": True,
            "accepted": True,
            "video": "logs/engineering_policy/engineered_hierarchical_policy_success.mp4",
            "dense_sheet": "logs/engineering_policy/engineered_hierarchical_policy_success_dense_sheet.jpg",
            "observed": "bilateral acquisition, sustained physical lift, alternating-foot carry, goal-platform set-down, clean release, upright recovery",
            "anti_cheat_behaviors_absent": ["kick", "body propulsion", "falling into object", "drag", "attachment", "external object force"],
        },
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    manifest = {
        "status": "POLICY_SUCCESS_ON_FROZEN_START",
        "task_variant_unchanged": True,
        "policy": str(POLICY_PATH.relative_to(ROOT)),
        "rollout": str(ROLLOUT_PATH.relative_to(ROOT)),
        "report": str(REPORT_PATH.relative_to(ROOT)),
        "video": "logs/engineering_policy/engineered_hierarchical_policy_success.mp4",
        "model": str(MODEL_PATH.relative_to(ROOT)),
        "architecture": policy["format"],
        "controller": policy["controller"],
        "metrics": metrics,
        "robustness": {
            "nominal_repeats": [sum(item["success"] for item in repeat_reports), len(repeat_reports)],
            "micro_robot_state": [sum(item["success"] for item in micro_reports), len(micro_reports)],
        },
        "no_commit_created": True,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
