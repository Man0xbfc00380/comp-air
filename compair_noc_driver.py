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


def run_rmsnorm(num_per_bank: int) -> int:
    sys.path.insert(0, EXAMPLE)
    import _booksim_path  # noqa: F401, type: ignore  # adds booksim2/api to sys.path
    from booksim_sync import BookSim2Sync  # type: ignore
    from rmsnorm import rms_norm_comp_air  # type: ignore

    sim = BookSim2Sync()
    steps = 0
    orig = sim.run_step

    def wrapped(*a, **k):
        nonlocal steps
        steps += 1
        return orig(*a, **k)

    sim.run_step = wrapped  # type: ignore
    rms_norm_comp_air(sim, num_per_bank)
    sim.run_step(end=True)  # type: ignore
    return steps


def run_softmax(num_per_bank: int) -> int:
    sys.path.insert(0, EXAMPLE)
    import _booksim_path  # noqa: F401, type: ignore  # adds booksim2/api to sys.path
    from booksim_sync import BookSim2Sync  # type: ignore
    from softmax import softmax_comp_air  # type: ignore

    sim = BookSim2Sync()
    steps = 0
    orig = sim.run_step

    def wrapped(*a, **k):
        nonlocal steps
        steps += 1
        return orig(*a, **k)

    sim.run_step = wrapped  # type: ignore
    softmax_comp_air(sim, num_per_bank)
    sim.run_step(end=True)  # type: ignore
    return steps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--micro", choices=["rmsnorm", "softmax"], required=True)
    ap.add_argument("--num-per-bank", type=int, default=52)
    args = ap.parse_args()
    if args.micro == "rmsnorm":
        n = run_rmsnorm(args.num_per_bank)
    else:
        n = run_softmax(args.num_per_bank)
    print("COMPAIR_NOC_STEPS", n)


if __name__ == "__main__":
    main()
