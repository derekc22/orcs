#!/usr/bin/env python3
"""Create single-network actor checkpoints along a learned-policy segment."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    first = torch.load(args.first, map_location="cpu", weights_only=False)
    second = torch.load(args.second, map_location="cpu", weights_only=False)
    result = deepcopy(first)
    for name, first_value in first["actor_state_dict"].items():
        second_value = second["actor_state_dict"][name]
        if name.startswith("mlp.") and torch.is_floating_point(first_value):
            result["actor_state_dict"][name] = torch.lerp(
                first_value, second_value, args.alpha
            )
    result["iter"] = 0
    result["optimizer_state_dict"] = {}
    result["infos"] = {
        "training": "single_actor_weight_interpolation",
        "first": str(args.first.resolve()),
        "second": str(args.second.resolve()),
        "alpha": args.alpha,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)


if __name__ == "__main__":
    main()
