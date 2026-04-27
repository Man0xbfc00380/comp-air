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


def run_reduce(sources: int, step: int, para_num: int, data: float) -> int:
    sys.path.insert(0, EXAMPLE)
    import _booksim_path  # noqa: F401, type: ignore
    from booksim_sync import BookSim2Sync  # type: ignore
    from reduce import reduce_comp_air  # type: ignore

    return _run_with_counter(
        BookSim2Sync,
        lambda sim: reduce_comp_air(sim, sources=sources, step=step, data=data, para_num=para_num),
    )


def run_broadcast(src: int, targs: int, step: int, data: float) -> int:
    sys.path.insert(0, EXAMPLE)
    import _booksim_path  # noqa: F401, type: ignore
    from booksim_sync import BookSim2Sync  # type: ignore
    from broadcast import broadcast_comp_air  # type: ignore

    return _run_with_counter(
        BookSim2Sync,
        lambda sim: broadcast_comp_air(sim, src=src, targs=targs, step=step, data=data),
    )


def run_sqrt(num_per_bank: int, iter_num: int, data: float) -> int:
    sys.path.insert(0, EXAMPLE)
    import _booksim_path  # noqa: F401, type: ignore
    from booksim_sync import BookSim2Sync  # type: ignore
    from sqrt import sqrt_comp_air  # type: ignore

    loops = max(1, num_per_bank // 2)

    def _micro(sim):
        for i in range(loops):
            sqrt_comp_air(sim, data, iter_num, i == 0)

    return _run_with_counter(BookSim2Sync, _micro)


def run_scalar_r(scalar: float, op: int) -> int:
    sys.path.insert(0, EXAMPLE)
    import _booksim_path  # noqa: F401, type: ignore
    from booksim_sync import BookSim2Sync  # type: ignore
    from scalar import scalar_r_comp_air  # type: ignore

    return _run_with_counter(BookSim2Sync, lambda sim: scalar_r_comp_air(sim, scalar, op))


def run_scalar_exp_approx(num_per_bank: int, iter_num: int, x: float) -> int:
    """
    Approximate exp path using NoC_Scalar-only micro-ops.
    This keeps ISA-level primitive as SCALAR (no explicit EXP ISA op).
    """
    sys.path.insert(0, EXAMPLE)
    import _booksim_path  # noqa: F401, type: ignore
    from booksim_sync import BookSim2Sync  # type: ignore
    from scalar import scalar_r_comp_air  # type: ignore

    loops = max(1, num_per_bank // 2)

    def _micro(sim):
        for _ in range(loops):
            # Taylor-style iterative chain:
            #   Res = 1; for i=iter..1: Res = 1 + Res * x / i
            for i in range(max(1, iter_num), 0, -1):
                scalar_r_comp_air(sim, x, 2)          # Res *= x
                scalar_r_comp_air(sim, float(i), 3)   # Res /= i
                scalar_r_comp_air(sim, 1.0, 0)        # Res += 1

    return _run_with_counter(BookSim2Sync, _micro)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--micro",
        choices=["rmsnorm", "softmax", "rope", "exp", "reduce", "broadcast", "sqrt", "scalar_r", "scalar_exp"],
        required=True,
    )
    ap.add_argument("--num-per-bank", type=int, default=52)
    ap.add_argument("--sources", type=int, default=16)
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--para-num", type=int, default=4)
    ap.add_argument("--data", type=float, default=2.5)
    ap.add_argument("--src", type=int, default=0)
    ap.add_argument("--targs", type=int, default=8)
    ap.add_argument("--iter-num", type=int, default=4)
    ap.add_argument("--scalar", type=float, default=1.0)
    ap.add_argument("--op", type=int, default=2)
    ap.add_argument("--x", type=float, default=2.0)
    ap.add_argument("--y", type=float, default=2.0)
    args = ap.parse_args()
    if args.micro == "rmsnorm":
        n = run_rmsnorm(args.num_per_bank)
    elif args.micro == "softmax":
        n = run_softmax(args.num_per_bank)
    elif args.micro == "rope":
        n = run_rope(args.num_per_bank)
    elif args.micro == "exp":
        # Keep exp basic-op callable with explicit x/y/iter by emulating existing helper style.
        sys.path.insert(0, EXAMPLE)
        import _booksim_path  # noqa: F401, type: ignore
        from booksim_sync import BookSim2Sync  # type: ignore
        from exp import exp_comp_air  # type: ignore

        loops = max(1, args.num_per_bank // 2)

        def _micro(sim):
            for i in range(loops):
                exp_comp_air(sim, args.x, args.y, args.iter_num, i == 0)

        n = _run_with_counter(BookSim2Sync, _micro)
    elif args.micro == "reduce":
        n = run_reduce(args.sources, args.step, args.para_num, args.data)
    elif args.micro == "broadcast":
        n = run_broadcast(args.src, args.targs, args.step, args.data)
    elif args.micro == "sqrt":
        n = run_sqrt(args.num_per_bank, args.iter_num, args.data)
    elif args.micro == "scalar_exp":
        n = run_scalar_exp_approx(args.num_per_bank, args.iter_num, args.x)
    else:  # scalar_r
        n = run_scalar_r(args.scalar, args.op)
    print("COMPAIR_NOC_STEPS", n)


if __name__ == "__main__":
    main()
