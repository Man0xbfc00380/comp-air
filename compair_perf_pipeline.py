#!/usr/bin/env python3
"""
End-to-end CompAir performance path:
  1) CENT / Ramulator: trace + DRAM-side timing (cent_pim).
  2) Optional offload manifests -> row ISA.
  3) booksim2 microprograms (NoC) + sram_pim.api (SRAM-PIM) -> additive latency budget.
  4) Summary JSON in compair_results/.

Example:
  python compair_perf_pipeline.py --model Llama2-7B --num_channels 32 --num_devices 32 \\
    --seqlen 1024 --use-noc --use-sram-pim --run-cent --run-ramulator --run-subsims
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import subprocess
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.abspath(__file__))
CENT_PIM = os.path.join(REPO, "cent_pim")
CENT_SIM = os.path.join(CENT_PIM, "cent_simulation")
ROW_TO_PACKET_COMPILER = os.path.join(REPO, "translate", "row_to_packet.py")


def _run(cmd: list, cwd: str, env=None) -> None:
    print("+", " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, cwd=cwd, check=True, env=env)


def _write_delay_summary_table(
    repo: str,
    case_label: str,
    model: str,
    seqlen: int,
    result_dir: str,
    summary: dict,
) -> None:
    """Append/replace one row in compair_results/delay_summary_by_case.{csv,json}."""
    sim_csv = os.path.join(result_dir, "simulation_results.csv")
    dram_pim_ms = None
    token_latency_ms = None
    if os.path.isfile(sim_csv):
        try:
            import pandas as pd

            df = pd.read_csv(sim_csv)
            if len(df) > 0:
                r0 = df.iloc[0]
                dram_pim_ms = float(r0.get("PIM latency", 0) or 0)
                token_latency_ms = float(r0.get("Token latency (ms)", 0) or 0)
        except Exception:
            pass

    noc_ms = float(summary.get("noc_offload_ms_est") or 0)
    sram_p = float(summary.get("sram_offload_ms_est") or 0)
    sram_s = float(summary.get("sram_offload_ms_est_serial") or 0)
    # End-to-end totals: token_latency already includes DRAM-PIM/CXL/acc path from run_sim.
    # Add SRAM/NoC offload budgets on top of token latency, do not add dram_pim_ms again.
    e2e_pipe_ms = (token_latency_ms or 0) + sram_p + noc_ms
    e2e_serial_ms = (token_latency_ms or 0) + sram_s + noc_ms

    os.makedirs(os.path.join(repo, "compair_results"), exist_ok=True)
    json_path = os.path.join(repo, "compair_results", "delay_summary_by_case.json")
    rows: list = []
    if os.path.isfile(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                rows = json.load(f)
        except Exception:
            rows = []
    if not isinstance(rows, list):
        rows = []
    key = (case_label, model, int(seqlen))
    rows = [x for x in rows if (x.get("case"), x.get("model"), x.get("seqlen")) != key]
    rows.append(
        {
            "case": case_label,
            "model": model,
            "seqlen": int(seqlen),
            "dram_pim_ms": dram_pim_ms,
            "token_latency_ms": token_latency_ms,
            "sram_pim_ms_pipeline": sram_p,
            "sram_pim_ms_serial": sram_s,
            "noc_ms": noc_ms,
            "sum_dram_pim_plus_sram_pipe_plus_noc_ms": e2e_pipe_ms,
            "end_to_end_token_plus_sram_pipe_plus_noc_ms": e2e_pipe_ms,
            "end_to_end_token_plus_sram_serial_plus_noc_ms": e2e_serial_ms,
            "result_dir": os.path.abspath(result_dir),
        }
    )
    rows.sort(key=lambda x: (x.get("case", ""), x.get("model", ""), x.get("seqlen", 0)))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    csv_path = os.path.join(repo, "compair_results", "delay_summary_by_case.csv")
    fieldnames = [
        "case",
        "model",
        "seqlen",
        "dram_pim_ms",
        "token_latency_ms",
        "sram_pim_ms_pipeline",
        "sram_pim_ms_serial",
        "noc_ms",
        "sum_dram_pim_plus_sram_pipe_plus_noc_ms",
        "end_to_end_token_plus_sram_pipe_plus_noc_ms",
        "end_to_end_token_plus_sram_serial_plus_noc_ms",
        "result_dir",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fieldnames})
    print("Wrote", json_path, "and", csv_path)


def parse_row_isa(path: str) -> list:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("ROW"):
                rows.append(line)
    return rows


def parse_packet_isa(path: str) -> list[dict[str, str]]:
    packets: list[dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if not line.startswith("PKT"):
                continue
            packets.append(dict(tok.split("=", 1) for tok in line.split("\t") if "=" in tok))
    return packets


def noc_steps_from_row(line: str, py: str, collective_split: int = 1, exp_micro: str = "exp") -> int:
    parts = dict(tok.split("=", 1) for tok in line.split("\t") if "=" in tok)
    op = (parts.get("OP") or "").strip().upper()
    micro = parts.get("MICRO", "rmsnorm.py")
    repeat = int(parts.get("REPEAT", "1"))
    num = int(parts.get("num_per_bank", "52"))
    driver = os.path.join(REPO, "compair_noc_driver.py")
    if op == "REDUCE":
        name = "reduce"
    elif op == "BROADCAST":
        name = "broadcast"
    elif op == "SQRT":
        name = "sqrt"
    elif op == "SCALAR":
        mode = (parts.get("MODE") or "R").strip().upper()
        # Taylor exp loop is represented by 3 SIMD rows (MUL/DIV/ADD).
        # Runtime subsim keeps one fused execution for MUL row; helper rows are skipped.
        if mode == "EXP_MUL":
            # Per user requirement: always dispatch exp semantics to exp.py backend.
            name = "exp"
        elif mode in {"EXP_DIV", "EXP_ADD"}:
            return 0
        else:
            name = "scalar_r"
    elif op == "ROPE":
        name = "rope"
    else:
        # Backward compatibility with legacy fused rows.
        micro_l = micro.lower()
        if "rmsnorm" in micro_l:
            name = "rmsnorm"
        elif "softmax" in micro_l:
            name = "softmax"
        elif "rope" in micro_l:
            name = "rope"
        elif "exp" in micro_l:
            name = "exp"
        else:
            raise ValueError(f"Unsupported NoC microprogram in row ISA: {micro}")
    # Model collective parallel split (reduction-heavy collectives):
    # split groups reduce per-group work from num_per_bank -> ceil(num_per_bank / split).
    if name in {"rmsnorm", "softmax"} and collective_split > 1:
        num = max(1, (num + collective_split - 1) // collective_split)

    total = 0
    for _ in range(repeat):
        cmd = [py, driver, "--micro", name, "--num-per-bank", str(num)]
        if name == "reduce":
            cmd.extend(
                [
                    "--sources",
                    str(int(parts.get("sources", "16"))),
                    "--step",
                    str(int(parts.get("step", "2"))),
                    "--para-num",
                    str(int(parts.get("para_num", "4"))),
                    "--data",
                    str(float(parts.get("data", "2.5"))),
                ]
            )
        elif name == "broadcast":
            cmd.extend(
                [
                    "--src",
                    str(int(parts.get("src", "0"))),
                    "--targs",
                    str(int(parts.get("targs", "8"))),
                    "--step",
                    str(int(parts.get("step", "4"))),
                    "--data",
                    str(float(parts.get("data", "2.5"))),
                ]
            )
        elif name == "sqrt":
            cmd.extend(
                [
                    "--iter-num",
                    str(int(parts.get("iter_num", "4"))),
                    "--data",
                    str(float(parts.get("data", "2.5"))),
                ]
            )
        elif name == "scalar_r":
            cmd.extend(
                [
                    "--scalar",
                    str(float(parts.get("scalar", "1.0"))),
                    "--op",
                    str(int(parts.get("op", "2"))),
                ]
            )
        elif name == "scalar_exp":
            cmd.extend(
                [
                    "--x",
                    str(float(parts.get("x", "2.0"))),
                    "--iter-num",
                    str(int(parts.get("iter_start", parts.get("iter_num", "6")))),
                    "--num-per-bank",
                    str(int(parts.get("num_per_bank", "52"))),
                ]
            )
        p = subprocess.run(
            cmd,
            cwd=REPO,
            capture_output=True,
            text=True,
        )
        if p.returncode != 0:
            print(p.stderr, file=sys.stderr)
            p.check_returncode()
        for ln in p.stdout.splitlines():
            if ln.startswith("COMPAIR_NOC_STEPS"):
                total += int(ln.split()[-1])
    return total


# Batch heuristic for MAC_ABK counts -> SRAM_PIM_Compute_API tiling (same as former single-line ISA).
_SRAM_MAC_CHUNK = 500


def _parse_isa_fields(line: str) -> dict[str, str]:
    return dict(tok.split("=", 1) for tok in line.split("\t") if "=" in tok)


def sram_ms_from_row(line: str, emb: int, ffn: int) -> tuple[float, float]:
    """Return (pipeline_ms, serial_ms) for one SRAM-PIM ROW using per-bank tile from ISA."""
    sys.path.insert(0, os.path.join(REPO, "sram_pim"))
    from api import SRAM_PIM_Compute_API  # type: ignore

    parts = _parse_isa_fields(line)
    events = int(parts.get("GEMV_MAC_EVENTS", "0"))
    if events <= 0:
        return 0.0, 0.0
    vec = int(parts.get("BANK_VEC_DIM", parts.get("VEC_DIM", str(emb))))
    mat = int(parts.get("BANK_MAT_COL", parts.get("MAT_COL", str(max(1, ffn)))))
    # Chunks now reflect serialized bank-tile rounds from CENT (rows/partials), not raw event/500 heuristic.
    chunks = int(parts.get("PIPELINE_CHUNKS", "0")) or max(1, min(events, 500_000) // _SRAM_MAC_CHUNK)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        lat_pipe_ns, lat_serial_ns = SRAM_PIM_Compute_API(vec, mat, macro_num=8, batch_num=1)
    return (float(lat_pipe_ns) * chunks / 1e6, float(lat_serial_ns) * chunks / 1e6)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Llama2-7B")
    ap.add_argument("--num_channels", type=int, default=32)
    ap.add_argument("--num_devices", type=int, default=32)
    ap.add_argument("--prefill", type=int, default=1024)
    ap.add_argument("--decoding", type=int, default=2048)
    ap.add_argument("--seqlen_gap", type=int, default=1024)
    ap.add_argument("--seqlen", type=int, nargs="*", default=None, help="If set, only these lengths (api mode)")
    ap.add_argument("--use-noc", action="store_true")
    ap.add_argument("--use-sram-pim", action="store_true")
    ap.add_argument("--model_parallel", action="store_true")
    ap.add_argument("--run-cent", action="store_true", help="Run CENT trace generation (+ Ramulator if --run-ramulator)")
    ap.add_argument(
        "--run-ramulator",
        action="store_true",
        help="Run run_sim.py --simulate_trace --process_results --update_csv (after traces exist)",
    )
    ap.add_argument("--run-subsims", action="store_true", help="Parse row ISA and run booksim2 + SRAM models")
    ap.add_argument("--noc-freq-ghz", type=float, default=1.0, help="Convert NoC steps -> ms (rough)")
    ap.add_argument(
        "--noc-collective-split",
        type=int,
        default=1,
        help="Parallel split factor for reduction-style NoC collectives (rmsnorm/softmax).",
    )
    ap.add_argument(
        "--noc-exp-micro",
        choices=["scalar_exp", "exp"],
        default="exp",
        help="Execution backend for NoC_Scalar exp loop (default: exp.py).",
    )
    ap.add_argument("--embedding", type=int, default=4096)
    ap.add_argument("--ffn", type=int, default=11008)
    ap.add_argument("--result-dir", default=os.path.join(REPO, "compair_results"))
    ap.add_argument(
        "--case-label",
        default="",
        help="Label for delay_summary_by_case table (default: basename of --result-dir)",
    )
    args = ap.parse_args()

    py = sys.executable
    os.makedirs(args.result_dir, exist_ok=True)
    summary = {
        "model": args.model,
        "use_noc": args.use_noc,
        "use_sram_pim": args.use_sram_pim,
        "noc_collective_split": max(1, int(args.noc_collective_split)),
        "noc_exp_micro": args.noc_exp_micro,
    }

    if args.run_cent:
        cmd = [
            py,
            os.path.join(CENT_PIM, "api.py"),
            "--generate_trace",
            "--model",
            args.model,
            "--num_channels",
            str(args.num_channels),
            "--num_devices",
            str(args.num_devices),
            "--prefill",
            str(args.prefill),
            "--decoding",
            str(args.decoding),
            "--seqlen_gap",
            str(args.seqlen_gap),
        ]
        if args.seqlen:
            cmd.append("--seqlen")
            cmd.extend(str(s) for s in args.seqlen)
        if args.use_noc:
            cmd.append("--use-noc")
        if args.use_sram_pim:
            cmd.append("--use-sram-pim")
        if args.model_parallel:
            cmd.append("--model_parallel")
        _run(cmd, cwd=CENT_PIM)

    if args.run_ramulator:
        # Absolute paths: cwd is cent_simulation/, so repo-relative result-dir must be absolute.
        sim_csv = os.path.abspath(os.path.join(args.result_dir, "simulation_results.csv"))
        proc_csv = os.path.abspath(os.path.join(args.result_dir, "processed_results.csv"))
        cmd = [
            py,
            "run_sim.py",
            "--model",
            args.model,
            "--simulate_trace",
            "--process_results",
            "--update_csv",
            "--simulation_result_path",
            sim_csv,
            "--processed_result_path",
            proc_csv,
            "--num_channels",
            str(args.num_channels),
            "--num_devices",
            str(args.num_devices),
            "--prefill",
            str(args.prefill),
            "--decoding",
            str(args.decoding),
            "--seqlen_gap",
            str(args.seqlen_gap),
        ]
        if args.seqlen:
            cmd.append("--seqlen")
            cmd.extend(str(s) for s in args.seqlen)
        if args.use_noc:
            cmd.append("--use-noc")
        if args.use_sram_pim:
            cmd.append("--use-sram-pim")
        if args.model_parallel:
            cmd.append("--model_parallel")
        _run(cmd, cwd=CENT_SIM)
        summary["cent_simulation_results_csv"] = sim_csv
        summary["cent_processed_results_csv"] = proc_csv

    def channels_device_folder(nc: int, use_noc: bool, use_sram_pim: bool) -> str:
        name = f"{nc}_channels_per_device"
        if use_noc:
            name += "_noc"
        if use_sram_pim:
            name += "_sram_pim"
        return name

    tfolder = channels_device_folder(args.num_channels, args.use_noc, args.use_sram_pim)
    trace_root = os.path.join(CENT_PIM, "trace", tfolder)
    manifests = []
    for root, _, files in os.walk(trace_root):
        for fn in files:
            if fn.endswith(".offload.json"):
                manifests.append(os.path.join(root, fn))

    isa_path = os.path.join(args.result_dir, "row_isa.txt")
    packet_isa_path = os.path.join(args.result_dir, "packet_isa.txt")
    packet_isa_json = packet_isa_path + ".json"
    if manifests:
        _run(
            [
                py,
                os.path.join(REPO, "compair_row_isa.py"),
                *manifests,
                "-o",
                isa_path,
                "--seqlen",
                str(args.seqlen[0] if args.seqlen else args.prefill),
                "--dim",
                str(args.embedding),
                "--ffn",
                str(args.ffn),
            ],
            cwd=REPO,
        )
        if os.path.isfile(ROW_TO_PACKET_COMPILER):
            _run(
                [
                    py,
                    ROW_TO_PACKET_COMPILER,
                    "--row-isa",
                    isa_path,
                    "--packet-isa",
                    packet_isa_path,
                    "--packet-json",
                    packet_isa_json,
                    "--exp-backend",
                    args.noc_exp_micro,
                ],
                cwd=REPO,
            )
    else:
        summary["note"] = "No .offload.json manifests found (run with --use-noc and/or --use-sram-pim)."
    summary["manifests"] = manifests

    noc_ms = 0.0
    sram_ms_pipe = 0.0
    sram_ms_serial = 0.0
    if args.run_subsims and manifests and os.path.isfile(isa_path):
        rows = parse_row_isa(isa_path)
        row_noc_ms: dict[int, float] = {}
        row_sram_pipe_ms: dict[int, float] = {}
        row_sram_serial_ms: dict[int, float] = {}
        for ridx, row in enumerate(rows):
            if "TARGET=NoC" in row:
                steps = noc_steps_from_row(
                    row,
                    py,
                    collective_split=max(1, int(args.noc_collective_split)),
                    exp_micro=args.noc_exp_micro,
                )
                row_noc_ms[ridx] = steps / (args.noc_freq_ghz * 1e6)
            if "TARGET=SRAM-PIM" in row:
                p, s = sram_ms_from_row(row, args.embedding, args.ffn)
                row_sram_pipe_ms[ridx] = p
                row_sram_serial_ms[ridx] = s

        packet_entries: list[dict[str, str]] = []
        if os.path.isfile(packet_isa_path):
            packet_entries = parse_packet_isa(packet_isa_path)

        if packet_entries:
            row_bytes_sum = defaultdict(int)
            for pkt in packet_entries:
                row_id = int(pkt.get("ROW", "-1"))
                pkt_bytes = int(pkt.get("BYTES", "0"))
                if row_id >= 0 and pkt_bytes > 0:
                    row_bytes_sum[row_id] += pkt_bytes

            for pkt in packet_entries:
                row_id = int(pkt.get("ROW", "-1"))
                pkt_bytes = int(pkt.get("BYTES", "0"))
                target = pkt.get("TARGET", "")
                total_b = row_bytes_sum.get(row_id, 0)
                if row_id < 0 or pkt_bytes <= 0 or total_b <= 0:
                    continue
                weight = float(pkt_bytes) / float(total_b)
                if target == "NoC" and row_id in row_noc_ms:
                    noc_ms += row_noc_ms[row_id] * weight
                elif target == "SRAM-PIM" and row_id in row_sram_pipe_ms:
                    sram_ms_pipe += row_sram_pipe_ms[row_id] * weight
                    sram_ms_serial += row_sram_serial_ms[row_id] * weight

            summary["subsim_isa_mode"] = "packet"
            summary["packet_instruction_count"] = len(packet_entries)
        else:
            # Backward-compatible fallback when packet ISA is unavailable.
            for ridx, _ in enumerate(rows):
                noc_ms += row_noc_ms.get(ridx, 0.0)
                sram_ms_pipe += row_sram_pipe_ms.get(ridx, 0.0)
                sram_ms_serial += row_sram_serial_ms.get(ridx, 0.0)
            summary["subsim_isa_mode"] = "row"
            summary["packet_instruction_count"] = 0
    summary["noc_offload_ms_est"] = noc_ms
    summary["sram_offload_ms_est"] = sram_ms_pipe
    summary["sram_offload_ms_est_serial"] = sram_ms_serial
    summary["row_isa"] = isa_path if os.path.isfile(isa_path) else None
    summary["packet_isa"] = packet_isa_path if os.path.isfile(packet_isa_path) else None
    summary["packet_isa_json"] = packet_isa_json if os.path.isfile(packet_isa_json) else None

    out_json = os.path.join(args.result_dir, "compair_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Wrote", out_json)

    case_label = args.case_label or os.path.basename(os.path.abspath(args.result_dir))
    _write_delay_summary_table(
        REPO,
        case_label,
        args.model,
        args.seqlen[0] if args.seqlen else args.prefill,
        args.result_dir,
        summary,
    )


if __name__ == "__main__":
    main()
