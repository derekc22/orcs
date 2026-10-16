#!/usr/bin/env python3
"""Automated experimental diffusion -> MuJoCo-realized reference pipeline.

The expensive optimizers remain separate, inspectable programs.  This runner
owns their sequencing, detects the first recovery failure, tries progressively
earlier actual-state recovery anchors, and only accepts an uninterrupted
300-frame MuJoCo rollout that is valid and satisfies the task's ``at_goal``
thresholds.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from orcs.tasks.uolm.realized_reference import load_realized_reference

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs" / "unified_physical_reference"


def _run(script: str, env: dict[str, str]) -> None:
    process_env = os.environ.copy()
    process_env.update(env)
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script)],
        cwd=ROOT,
        env=process_env,
        check=True,
    )


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _deviation(final_dir: Path) -> dict[str, float]:
    artifact = load_realized_reference(final_dir)
    prior = artifact.diffusion_prior
    if prior.ndim == 3:
        prior = prior[0]
    joint_error = artifact.qpos[:, 7:36] - prior[:, 9:38]
    root_error = artifact.qpos[:, 0:3] - prior[:, 0:3]
    object_motion = artifact.qpos[-1, 36:39] - artifact.qpos[0, 36:39]
    return {
        "joint_rms_correction_rad": float(np.sqrt(np.mean(np.square(joint_error)))),
        "joint_max_correction_rad": float(np.max(np.abs(joint_error))),
        "root_position_rms_correction_m": float(
            np.sqrt(np.mean(np.square(root_error)))
        ),
        "root_position_max_correction_m": float(np.max(np.abs(root_error))),
        "physical_object_displacement_m": float(np.linalg.norm(object_motion)),
    }


def _ledger(start: int, validation: dict[str, object]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    strike_path = LOG / f"settled_physical_strike_start_{start:03d}.json"
    base_path = LOG / f"settled_goal_recovery_start_{start:03d}.json"
    for method, path in (
        ("settled physical strike", strike_path),
        ("actual-state recovery", base_path),
    ):
        if path.is_file():
            report = _json(path)
            best = report["best"]
            records.append(
                {
                    "start": start,
                    "method": method,
                    "key_parameterization": {
                        "dimensions": report.get("dimensions"),
                        "anchor_frame": report.get("anchor_frame"),
                    },
                    "first_invalid_frame": best.get("first_invalid_frame"),
                    "invalid_reason": best.get("invalid_reason"),
                    "object_at_goal": best.get("at_goal"),
                    "final_position_error_m": best.get("final_position_error_m"),
                    "reason_for_switching": "recover the actual state"
                    if method.startswith("settled")
                    else "continue from the first impending failure",
                }
            )
    anchor_keys = [
        key
        for key in validation.get("anchor_state_errors", {})
        if key.startswith("frame") and key.endswith("_qpos") and key != "frame20_qpos"
    ]
    if anchor_keys:
        anchor = int(anchor_keys[0][5:-5])
        path = LOG / f"settled_goal_recovery_{anchor:03d}_start_{start:03d}.json"
        if path.is_file():
            report = _json(path)
            best = report["best"]
            records.append(
                {
                    "start": start,
                    "method": "rolling actual-state recovery",
                    "key_parameterization": {
                        "dimensions": report.get("dimensions"),
                        "anchor_frame": anchor,
                    },
                    "first_invalid_frame": best.get("first_invalid_frame"),
                    "invalid_reason": best.get("invalid_reason"),
                    "object_at_goal": best.get("at_goal"),
                    "final_position_error_m": best.get("final_position_error_m"),
                    "reason_for_switching": "uninterrupted full-horizon validation",
                }
            )
    records.append(
        {
            "start": start,
            "method": "uninterrupted physical validation",
            "key_parameterization": {"frames": 300, "physics_hz": 200},
            "first_invalid_frame": validation.get("first_invalid_frame"),
            "invalid_reason": validation.get("invalid_reason"),
            "object_at_goal": validation.get("at_goal"),
            "final_position_error_m": validation.get("final_position_error_m"),
            "best_physical_horizon": 300
            if validation.get("full_valid")
            else validation.get("first_invalid_frame"),
            "reason_for_switching": "success",
        }
    )
    return records


def _generate_one(start: int, *, reuse: bool) -> dict[str, object]:
    final_dir = LOG / f"final_start_{start:03d}"
    if reuse and (final_dir / "validation.json").is_file():
        artifact = load_realized_reference(final_dir)
        if artifact.is_task_success:
            return artifact.validation

    if start and not (LOG / f"diffusion_start_{start:03d}.npz").is_file():
        _run("generate_validation_diffusion_starts.py", {})
    strike = LOG / f"settled_physical_strike_start_{start:03d}.npz"
    if not (reuse and strike.is_file()):
        _run("search_settled_physical_strike.py", {"PHYSICAL_START": str(start)})

    base = LOG / f"settled_goal_recovery_start_{start:03d}.json"
    if not (reuse and base.is_file()):
        _run(
            "recover_goal_strike_anchor.py",
            {
                "PHYSICAL_START": str(start),
                "GOAL_RECOVERY_ANCHOR": "20",
                "GOAL_STRIKE_SOURCE": f"settled_physical_strike_start_{start:03d}",
                "GOAL_RECOVERY_OUTPUT": "settled_goal_recovery",
            },
        )
    failure = _json(base)["best"]["first_invalid_frame"]
    if failure is None:
        anchors = [240]
    else:
        failure = float(failure)
        anchors = sorted(
            {
                max(40, int(failure - lead))
                for lead in (35, 60, 90, 120)
                if failure - lead > 20
            },
            reverse=True,
        )

    selected_anchor = None
    for anchor in anchors:
        report = LOG / f"settled_goal_recovery_{anchor:03d}_start_{start:03d}.json"
        if not (reuse and report.is_file()):
            _run(
                "recover_goal_strike_anchor.py",
                {
                    "PHYSICAL_START": str(start),
                    "GOAL_RECOVERY_ANCHOR": str(anchor),
                    "GOAL_STRIKE_SOURCE": f"settled_physical_strike_start_{start:03d}",
                    "GOAL_RECOVERY_OUTPUT": "settled_goal_recovery",
                },
            )
        result = _json(report)["best"]
        if result["first_invalid_frame"] is None and result["at_goal"]:
            selected_anchor = anchor
            break
    if selected_anchor is None:
        raise RuntimeError(f"no full-horizon physical recovery found for start {start}")

    _run(
        "validate_settled_goal_rollout.py",
        {"PHYSICAL_START": str(start), "RECOVERY_ANCHOR": str(selected_anchor)},
    )
    artifact = load_realized_reference(final_dir)
    if not artifact.is_task_success:
        raise RuntimeError(
            f"uninterrupted physical validation failed for start {start}"
        )
    return artifact.validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--starts", type=int, nargs="+", default=(0, 10, 20))
    parser.add_argument("--no-reuse", action="store_true")
    args = parser.parse_args()
    LOG.mkdir(parents=True, exist_ok=True)
    results = {
        str(start): _generate_one(start, reuse=not args.no_reuse)
        for start in args.starts
    }
    manifest = {
        "method": "automatic MuJoCo strike, actual-state recovery, and uninterrupted validation",
        "diffusion": "HF-BPS epoch-999, DDIM-50, eta=0, FP32",
        "starts": {
            key: {
                "validation": report,
                "deviation_from_diffusion": _deviation(
                    LOG / f"final_start_{int(key):03d}"
                ),
            }
            for key, report in results.items()
        },
    }
    (LOG / "pipeline_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    ledger = [
        record
        for key, report in results.items()
        for record in _ledger(int(key), report)
    ]
    (LOG / "experiment_ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
