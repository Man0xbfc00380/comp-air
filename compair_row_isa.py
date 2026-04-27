#!/usr/bin/env python3
"""
Emit a row-level CompAir ISA text file from offload manifests (JSON).

Rows are human-readable and map to microprograms under booksim2/example/ and
latency hooks in sram_pim/api.py (see compair_perf_pipeline.py).

SRAM-PIM rows are bank-tile based:
- FULL_* is the logical GEMV shape (e.g., 4096 x 4096 for Wo).
- BANK_* is the per-DRAM-bank tile shape actually offloaded from CENT trace
  (e.g., 1024 x 8 in current Llama2-7B config; Figure-8-style small tiles).
- PIPELINE_CHUNKS counts serialized bank-tile rounds for that role.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import math
from collections import Counter
from typing import Any, Dict, List, Tuple

# Consistent ordering for paper-style tables (Llama2-style projection names).
SRAM_ROLE_ORDER: Tuple[str, ...] = (
    "Wq",
    "Wk",
    "Wv",
    "Wo",
    "W1",
    "W2",
    "W3",
    "Emb_in",
    "Emb_out",
)


def load_manifests(paths: List[str]) -> List[Dict[str, Any]]:
    out = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def _role_dims(role: str, dim: int, ffn: int) -> tuple[int, int]:
    if role in ("Wq", "Wk", "Wv", "Wo"):
        return dim, dim
    if role in ("W1", "W3"):
        return dim, ffn
    if role == "W2":
        return ffn, dim
    if role == "Emb_in":
        return dim, dim
    if role == "Emb_out":
        return dim, dim
    return dim, dim


def _split_legacy_sram_events(events: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Legacy manifests (without `role`) still contain event order.
    Infer roles for Llama/GPT trace-only path:
      - sa_weight/reuse-GB: Wq, Wk, Wv, Wo (4 contiguous chunks)
      - ffn_weight/*-af: W1
      - ffn_weight/reuse-GB: W3 then W2 (2 contiguous chunks)
    """
    by_role: Dict[str, List[Dict[str, Any]]] = {}
    sa = [e for e in events if e.get("timing") == "breakdown_sa_weight" and "af" not in str(e.get("gemv", ""))]
    if sa:
        q = max(1, len(sa) // 4)
        by_role["Wq"] = sa[0:q]
        by_role["Wk"] = sa[q : 2 * q]
        by_role["Wv"] = sa[2 * q : 3 * q]
        by_role["Wo"] = sa[3 * q :]

    w1 = [e for e in events if e.get("timing") == "breakdown_ffn_weight" and "af" in str(e.get("gemv", ""))]
    if w1:
        by_role["W1"] = w1

    ffn_plain = [e for e in events if e.get("timing") == "breakdown_ffn_weight" and "af" not in str(e.get("gemv", ""))]
    if ffn_plain:
        h = max(1, len(ffn_plain) // 2)
        by_role["W3"] = ffn_plain[:h]
        by_role["W2"] = ffn_plain[h:]
    return by_role


def aggregate(manifests: List[Dict[str, Any]], dim_hint: int, ffn_hint: int) -> Dict[str, Any]:
    noc_ops: Counter = Counter()
    sram_events: List[Dict[str, Any]] = []

    for m in manifests:
        for e in m.get("noc", []):
            noc_ops[e.get("op", "?")] += 1
        for e in m.get("sram", []):
            if e.get("op") == "MAC_ABK":
                sram_events.append(e)

    has_role = any((ev.get("role") or "").strip() for ev in sram_events)
    by_role_events: Dict[str, List[Dict[str, Any]]] = {}
    if has_role:
        for ev in sram_events:
            role = (ev.get("role") or "").strip() or "UNKNOWN"
            by_role_events.setdefault(role, []).append(ev)
    else:
        by_role_events = _split_legacy_sram_events(sram_events)

    sram_by_role: Dict[str, Dict[str, Any]] = {}
    for role, evs in by_role_events.items():
        if not evs:
            continue
        cnt = sum(int(e.get("count", 1)) for e in evs)
        full_v, full_m = _role_dims(role, dim_hint, ffn_hint)
        if any(e.get("vector_dim") for e in evs):
            full_v = int(next(e.get("vector_dim") for e in evs if e.get("vector_dim")))
        if any(e.get("matrix_col") for e in evs):
            full_m = int(next(e.get("matrix_col") for e in evs if e.get("matrix_col")))
        burst = int(next((e.get("burst_length") for e in evs if e.get("burst_length")), 16))
        max_op = max(int(e.get("op_size", 1) or 1) for e in evs)
        bank_vec_dim = max(1, max_op * burst)
        rows = max(1, math.ceil(full_v / bank_vec_dim))
        bank_mat_col = max(1, round(cnt / rows))
        pipe_chunks = max(1, round(cnt / bank_mat_col))
        sram_by_role[role] = {
            "count": cnt,
            "full_vector_dim": full_v,
            "full_matrix_col": full_m,
            "bank_vector_dim": bank_vec_dim,
            "bank_matrix_col": bank_mat_col,
            "pipeline_chunks": pipe_chunks,
            "burst_length": burst,
            "max_op_size": max_op,
            "legacy_inferred": not has_role,
        }

    return {
        "noc_ops": dict(noc_ops),
        "sram_by_role": sram_by_role,
        "sram_untagged_macs": 0 if has_role else sum(int(e.get("count", 1)) for e in sram_events),
    }


def emit_rows(agg: Dict[str, Any], seqlen_hint: int, dim_hint: int) -> List[str]:
    """Map high-level offload ops to microprogram rows (paper-style row ISA)."""
    lines = [
        "# CompAir row-level ISA (generated). Each ROW is one logical offload step.",
        "# TARGET=NoC rows map to booksim2/example/*.py microprograms.",
        "# TARGET=SRAM-PIM: one ROW per role (Wq..), with FULL_* and BANK_* GEMV shapes.",
        "#   compair_perf_pipeline evaluates SRAM latency on BANK_* shape and sums PIPELINE_CHUNKS rounds.",
    ]
    noc = agg["noc_ops"]
    if noc.get("rmsnorm_sa_input_ewmul_store", 0) or noc.get("rmsnorm_ffn_input_ewmul_store", 0):
        n = noc.get("rmsnorm_sa_input_ewmul_store", 0) + noc.get("rmsnorm_ffn_input_ewmul_store", 0)
        nb = max(8, min(512, dim_hint // 64))
        lines.append(f"ROW\tTARGET=NoC\tMICRO=rmsnorm.py\tFUNC=rms_norm_comp_air\tREPEAT={n}\tnum_per_bank={nb}\t# RMSNorm offload phases")
    if noc.get("softmax_score_store", 0) or noc.get("softmax_reduce_load", 0):
        nb = max(8, min(4096, seqlen_hint // 2))
        lines.append(f"ROW\tTARGET=NoC\tMICRO=softmax.py\tFUNC=softmax_comp_air\tREPEAT=1\tnum_per_bank={nb}\t# fused softmax collective")
    if noc.get("rotary_prefill_store", 0) or noc.get("rotary_post_gather", 0):
        lines.append("ROW\tTARGET=NoC\tMICRO=rope.py\tFUNC=rope_rearrange_comp_air\tREPEAT=1\t# rotary layout / gather")
    if noc.get("ffn_silu_store", 0) or noc.get("ffn_silu_split_store", 0):
        lines.append("ROW\tTARGET=NoC\tMICRO=exp.py\tFUNC=exp_comp_air\tREPEAT=1\t# SiLU nonlinearity traffic (approximate micro)")

    sbr = agg.get("sram_by_role") or {}

    if sbr:
        def role_key(r: str) -> int:
            try:
                return SRAM_ROLE_ORDER.index(r)
            except ValueError:
                return 99

        for role in sorted(sbr.keys(), key=role_key):
            info = sbr[role]
            cnt = int(info.get("count", 0))
            if cnt <= 0:
                continue
            full_vd = int(info.get("full_vector_dim") or dim_hint)
            full_mc = int(info.get("full_matrix_col") or dim_hint)
            bank_vd = int(info.get("bank_vector_dim") or full_vd)
            bank_mc = int(info.get("bank_matrix_col") or full_mc)
            pipe_chunks = int(info.get("pipeline_chunks") or 1)
            legacy_mark = " legacy-inferred" if info.get("legacy_inferred") else ""
            lines.append(
                f"ROW\tTARGET=SRAM-PIM\tAPI=SRAM_PIM_Compute_API\tROLE={role}\t"
                f"FULL_VEC_DIM={full_vd}\tFULL_MAT_COL={full_mc}\t"
                f"BANK_VEC_DIM={bank_vd}\tBANK_MAT_COL={bank_mc}\t"
                f"GEMV_MAC_EVENTS={cnt}\tPIPELINE_CHUNKS={pipe_chunks}\t"
                f"# per-role bank-tile GEMV from CENT trace; sum rows for total SRAM{legacy_mark}"
            )

    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description="Build CompAir row ISA from *.offload.json manifests")
    ap.add_argument("manifests", nargs="*", help="Explicit manifest paths (default: glob)")
    ap.add_argument("--glob", dest="glob_pat", default="", help="Glob for manifests, e.g. 'cent_pim/trace/**/*.offload.json'")
    ap.add_argument("-o", "--output", default="compair_results/row_isa.txt")
    ap.add_argument("--seqlen", type=int, default=1024)
    ap.add_argument("--dim", type=int, default=4096)
    ap.add_argument("--ffn", type=int, default=11008)
    args = ap.parse_args()
    paths = list(args.manifests)
    if args.glob_pat:
        paths.extend(sorted(glob.glob(args.glob_pat, recursive=True)))
    if not paths:
        raise SystemExit("No manifests: pass paths or --glob")
    manifests = load_manifests(paths)
    agg = aggregate(manifests, args.dim, args.ffn)
    rows = emit_rows(agg, args.seqlen, args.dim)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")
    with open(args.output + ".aggregate.json", "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)
    print("Wrote", args.output, "and", args.output + ".aggregate.json")


if __name__ == "__main__":
    main()
