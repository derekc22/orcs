#!/usr/bin/env python3
"""Pad a TaRa actor checkpoint for appended deployable context features.

The added input columns are initialized to zero so the converted actor is
bit-equivalent to the source policy until imitation training chooses to use
the new phase/start context.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extra-dims", type=int, default=4)
    args = parser.parse_args()

    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    state = checkpoint["actor_state_dict"]
    for name, fill in (
        ("obs_normalizer._mean", 0.0),
        ("obs_normalizer._var", 1.0),
        ("obs_normalizer._std", 1.0),
    ):
        value = state[name]
        state[name] = torch.cat(
            [
                value,
                torch.full(
                    (*value.shape[:-1], args.extra_dims),
                    fill,
                    dtype=value.dtype,
                ),
            ],
            dim=-1,
        )
    weight = state["mlp.0.weight"]
    state["mlp.0.weight"] = torch.cat(
        [weight, torch.zeros(weight.shape[0], args.extra_dims, dtype=weight.dtype)],
        dim=1,
    )
    checkpoint["optimizer_state_dict"] = {}
    checkpoint["infos"] = {
        **checkpoint.get("infos", {}),
        "context_expanded_from": str(args.input.resolve()),
        "context_extra_dims": args.extra_dims,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)


if __name__ == "__main__":
    main()
