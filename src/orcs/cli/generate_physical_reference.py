"""Generate an uninterrupted MuJoCo-realized largebox reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from orcs.cli.refine_contact_dynamics import _serializable
from orcs.cli.refine_diffusion_reference import _build_model
from orcs.tasks.uolm.dynamics_refinement import (
    DynamicsRefinementConfig,
    _DynamicRollout,
)
from orcs.tasks.uolm.generated_reference import GeneratedReferenceAdapter
from orcs.tasks.uolm.physical_reference_generator import (
    PhysicalReferenceConfig,
    generate_dynamically_feasible_reference,
)


def run(args: argparse.Namespace) -> dict:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with np.load(args.artifact.resolve()) as loaded:
        source = {key: loaded[key].copy() for key in loaded.files}
    trajectory = source["analytical_through140_trajectory"]
    device = torch.device(args.device)
    reference = GeneratedReferenceAdapter(device=device, dt=0.02, fk_batch_size=256)(
        torch.as_tensor(trajectory, device=device)
    )
    dynamics = DynamicsRefinementConfig(run_stage_b=False, seed=42)
    engine = _DynamicRollout(_build_model(), reference, dynamics)
    config = PhysicalReferenceConfig(
        population=args.population,
        elites=args.elites,
        generations=args.generations,
        horizon_frames=args.horizon,
        replan_frames=args.replan,
        stance_population=args.stance_population,
        stance_generations=args.stance_generations,
        stance_horizon_frames=args.stance_horizon,
        desired_stance_height_m=args.stance_height,
        robot_clearance_m=args.robot_clearance,
        stance_parameters=(
            None
            if args.stance_parameters_json is None
            else tuple(json.loads(args.stance_parameters_json))
        ),
        seed=args.seed,
    )
    result = generate_dynamically_feasible_reference(engine, config)
    artifact = output / "physical_reference_start_000.npz"
    np.savez_compressed(
        artifact,
        **result.data,
        controls=result.controls,
        physics_dt=np.asarray([dynamics.physics_dt]),
    )
    report = {
        "valid": result.valid,
        "first_invalid_frame": result.first_invalid_frame,
        "invalid_reason": result.invalid_reason,
        "summary": result.summary,
        "config": {name: getattr(config, name) for name in config.__dataclass_fields__},
        "replans": result.replans,
        "artifact": str(artifact),
    }
    report_path = output / "report_start_000.json"
    report["report"] = str(report_path)
    report_path.write_text(json.dumps(_serializable(report), indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path(
            "logs/direct_contact_dynamics_knot117/direct_contact_dynamics_start_000.npz"
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("logs/physical_reference_generation")
    )
    parser.add_argument("--population", type=int, default=40)
    parser.add_argument("--elites", type=int, default=6)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--replan", type=int, default=5)
    parser.add_argument("--stance-population", type=int, default=64)
    parser.add_argument("--stance-generations", type=int, default=20)
    parser.add_argument("--stance-horizon", type=int, default=60)
    parser.add_argument("--stance-height", type=float, default=0.75)
    parser.add_argument("--robot-clearance", type=float, default=0.60)
    parser.add_argument("--stance-parameters-json")
    parser.add_argument("--seed", type=int, default=1291)
    args = parser.parse_args()
    print(json.dumps(_serializable(run(args)), indent=2))


if __name__ == "__main__":
    main()
