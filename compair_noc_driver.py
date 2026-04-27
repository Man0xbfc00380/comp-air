#!/usr/bin/env python3
"""
Run a single booksim2 microprogram and report COMPAIR_NOC_STEPS (greenlet step count proxy).

Requires: greenlet, built booksim2/pybind (see README).
"""

from __future__ import annotations

import argparse
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
EXAMPLE = os.path.join(REPO, "booksim2", "example")


def _run_with_counter(sim_cls, run_micro) -> int:
    """Run one microprogram and return BookSim sync step count."""
    sim = sim_cls()
    steps = 0
    orig = sim.run_step

    def wrapped(*a, **k):
        nonlocal steps
        steps += 1
        return orig(*a, **k)

    sim.run_step = wrapped  # type: ignore
    run_micro(sim)
    sim.run_step(end=True)  # type: ignore
    return steps


def run_rmsnorm(num_per_bank: int) -> int:
    sys.path.insert(0, EXAMPLE)
    import _booksim_path  # noqa: F401, type: ignore  # adds booksim2/api to sys.path
    from booksim_sync import BookSim2Sync  # type: ignore
    from rmsnorm import rms_norm_comp_air  # type: ignore

    return _run_with_counter(BookSim2Sync, lambda sim: rms_norm_comp_air(sim, num_per_bank))


def run_softmax(num_per_bank: int) -> int:
    sys.path.insert(0, EXAMPLE)
    import _booksim_path  # noqa: F401, type: ignore  # adds booksim2/api to sys.path
    from booksim_sync import BookSim2Sync  # type: ignore
    from softmax import softmax_comp_air  # type: ignore

    return _run_with_counter(BookSim2Sync, lambda sim: softmax_comp_air(sim, num_per_bank))


def run_rope(_: int) -> int:
    sys.path.insert(0, EXAMPLE)
    import _booksim_path  # noqa: F401, type: ignore  # adds booksim2/api to sys.path
    from booksim_sync import BookSim2Sync  # type: ignore
    from rope import rope_rearrange_comp_air  # type: ignore

    return _run_with_counter(BookSim2Sync, lambda sim: rope_rearrange_comp_air(sim))


def run_exp(num_per_bank: int) -> int:
    sys.path.insert(0, EXAMPLE)
    import _booksim_path  # noqa: F401, type: ignore  # adds booksim2/api to sys.path
    from booksim_sync import BookSim2Sync  # type: ignore
    from exp import exp_comp_air  # type: ignore

    # Match softmax micro's per-bank workload convention: 2 elements processed per call.
    loops = max(1, num_per_bank // 2)

    def _micro(sim):
        for i in range(loops):
            exp_comp_air(sim, 2, 2, 4, i == 0)

    return _run_with_counter(BookSim2Sync, _micro)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--micro", choices=["rmsnorm", "softmax", "rope", "exp"], required=True)
    ap.add_argument("--num-per-bank", type=int, default=52)
    args = ap.parse_args()
    if args.micro == "rmsnorm":
        n = run_rmsnorm(args.num_per_bank)
    elif args.micro == "softmax":
        n = run_softmax(args.num_per_bank)
    elif args.micro == "rope":
        n = run_rope(args.num_per_bank)
    else:
        n = run_exp(args.num_per_bank)
    print("COMPAIR_NOC_STEPS", n)


if __name__ == "__main__":
    main()
