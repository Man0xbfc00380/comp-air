#!/usr/bin/env python3
"""
Compile CompAir row-level ISA into packet-level ISA using explicit address mapping.

This script models the "hierarchical ISA + autonomous translation" path:
  row instruction (logical offload op)
    -> address assignment (region + base + stride)
    -> packet instruction stream (addressed transfers/compute packets)
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple


@dataclass
class PacketInstruction:
    packet_id: int
    row_id: int
    target: str
    role: str
    op: str
    addr_start: int
    addr_end: int
    bytes: int
    chunk_idx: int
    chunk_total: int
    metadata: Dict[str, int | str]


def parse_row_lines(path: str) -> List[Tuple[int, str, Dict[str, str]]]:
    rows: List[Tuple[int, str, Dict[str, str]]] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if not line.startswith("ROW"):
                continue
            parts = dict(tok.split("=", 1) for tok in line.split("\t") if "=" in tok)
            rows.append((len(rows), line, parts))
    return rows


def _region_base(target: str, role: str) -> int:
    # Use separate virtual memory regions for NoC / SRAM-PIM packet spaces.
    if target == "NoC":
        return 0x1000_0000
    # Per-role disjoint spaces make address-based mapping explicit and debuggable.
    order = {
        "Wq": 0,
        "Wk": 1,
        "Wv": 2,
        "Wo": 3,
        "W1": 4,
        "W2": 5,
        "W3": 6,
        "Emb_in": 7,
        "Emb_out": 8,
    }
    role_idx = order.get(role, 31)
    return 0x8000_0000 + role_idx * 0x0100_0000


def _estimate_row_bytes(parts: Dict[str, str]) -> int:
    target = parts.get("TARGET", "")
    if target == "SRAM-PIM":
        vec = int(parts.get("BANK_VEC_DIM", "1"))
        mat = int(parts.get("BANK_MAT_COL", "1"))
        chunks = int(parts.get("PIPELINE_CHUNKS", "1"))
        # INT8/INT16-friendly estimate: 2 bytes per matrix element.
        return max(1, vec * mat * 2 * chunks)

    # NoC collectives: use num_per_bank as data scale hint.
    n = int(parts.get("num_per_bank", "1"))
    rpt = int(parts.get("REPEAT", "1"))
    return max(1, n * 16 * rpt)


def _estimate_runtime_chunks(parts: Dict[str, str]) -> int:
    """
    Runtime-only splitting hint (not ISA-level): modeled from micro-op loop structure.
    """
    target = parts.get("TARGET", "")
    if target != "NoC":
        return 1
    op = (parts.get("OP") or "").strip().upper()
    if op in {"SQRT", "SCALAR"}:
        n = int(parts.get("num_per_bank", "1"))
        return max(1, n // 2)
    return 1


def compile_packets(
    rows: List[Tuple[int, str, Dict[str, str]]],
    packet_bytes: int,
    row_addr_stride: int,
    exp_backend: str,
) -> Tuple[List[PacketInstruction], Dict[str, Dict[str, int]]]:
    packets: List[PacketInstruction] = []
    row_address_map: Dict[str, Dict[str, int]] = {}
    next_packet_id = 0

    i = 0
    while i < len(rows):
        row_id, _, parts = rows[i]
        target = parts.get("TARGET", "UNKNOWN")
        role = parts.get("ROLE", parts.get("MICRO", "GENERIC"))
        op = parts.get("API", parts.get("FUNC", "op"))
        row_bytes = _estimate_row_bytes(parts)
        base = _region_base(target, role) + row_id * row_addr_stride
        row_address_map[str(row_id)] = {
            "row_id": row_id,
            "base_addr": base,
            "row_bytes": row_bytes,
            "target": target,
            "role": role,
        }

        op_type = (parts.get("OP") or "").strip().upper()
        mode = (parts.get("MODE") or "").strip().upper()
        fuse_group = (parts.get("FUSE_GROUP") or "").strip()
        # Figure-16 style: fuse three dependent NoC_Scalar SIMD rows into one packet path.
        if target == "NoC" and op_type == "SCALAR" and mode == "EXP_MUL" and i + 2 < len(rows):
            _, _, p2 = rows[i + 1]
            _, _, p3 = rows[i + 2]
            if (
                (p2.get("OP") or "").strip().upper() == "SCALAR"
                and (p3.get("OP") or "").strip().upper() == "SCALAR"
                and (p2.get("MODE") or "").strip().upper() == "EXP_DIV"
                and (p3.get("MODE") or "").strip().upper() == "EXP_ADD"
                and (p2.get("FUSE_GROUP") or "").strip() == fuse_group
                and (p3.get("FUSE_GROUP") or "").strip() == fuse_group
            ):
                iter_num = max(1, int(parts.get("REPEAT", "6")))
                res_addr = base
                path = "[(0,0,F,F,*=),(1,0,F,T,/=),(-1,-1,F,F,+=)]"
                packets.append(
                    PacketInstruction(
                        packet_id=next_packet_id,
                        row_id=row_id,
                        target=target,
                        role=role,
                        op="SCALAR_PATH",
                        addr_start=res_addr,
                        addr_end=res_addr,
                        bytes=max(1, row_bytes),
                        chunk_idx=0,
                        chunk_total=1,
                        metadata={
                            "repeat": iter_num,
                            "iter_num": iter_num,
                            "pipeline_chunks": int(parts.get("PIPELINE_CHUNKS", "1")),
                            "runtime_chunks": _estimate_runtime_chunks(parts),
                            "read_addr": res_addr,
                            "rhs_addr": res_addr + 8,
                            "write_addr": res_addr,
                            "path": path,
                            "fused_from_modes": "EXP_MUL,EXP_DIV,EXP_ADD",
                            "backend": "EXP_MICRO" if exp_backend == "exp" else "SCALAR_CHAIN",
                        },
                    )
                )
                next_packet_id += 1
                i += 3
                continue
        else:
            # Row-level packet ISA keeps one packet-instruction per row-instruction.
            total_chunks = 1
            chunk_idx = 0
            start = base
            sz = max(1, row_bytes)
            end = start + max(0, sz - 1)
            packets.append(
                PacketInstruction(
                    packet_id=next_packet_id,
                    row_id=row_id,
                    target=target,
                    role=role,
                    op=op,
                    addr_start=start,
                    addr_end=end,
                    bytes=sz,
                    chunk_idx=chunk_idx,
                    chunk_total=total_chunks,
                    metadata={
                        "repeat": int(parts.get("REPEAT", "1")),
                        "pipeline_chunks": int(parts.get("PIPELINE_CHUNKS", "1")),
                        "runtime_chunks": _estimate_runtime_chunks(parts),
                        "backend": "EXP_MICRO" if exp_backend == "exp" else "SCALAR_CHAIN",
                    },
                )
            )
            next_packet_id += 1
        i += 1
    return packets, row_address_map


def write_packet_isa(out_path: str, packets: List[PacketInstruction]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# CompAir packet-level ISA (generated from row-level ISA)\n")
        f.write("# PKT_ID\tROW_ID\tTARGET\tROLE\tOP\tADDR_START\tADDR_END\tBYTES\tCHUNK\tREAD_ADDR\tRHS_ADDR\tWRITE_ADDR\t[ITER_NUM]\t[PATH]\n")
        for pkt in packets:
            read_addr = pkt.metadata.get("read_addr", pkt.addr_start)
            rhs_addr = pkt.metadata.get("rhs_addr", pkt.addr_start)
            write_addr = pkt.metadata.get("write_addr", pkt.addr_end)
            iter_num = pkt.metadata.get("iter_num", 0)
            path = pkt.metadata.get("path", "")
            line = (
                "PKT\t"
                f"ID={pkt.packet_id}\tROW={pkt.row_id}\tTARGET={pkt.target}\tROLE={pkt.role}\t"
                f"OP={pkt.op}\tADDR_START=0x{pkt.addr_start:016X}\tADDR_END=0x{pkt.addr_end:016X}\t"
                f"BYTES={pkt.bytes}\tCHUNK={pkt.chunk_idx + 1}/{pkt.chunk_total}\t"
                f"READ_ADDR=0x{int(read_addr):016X}\tRHS_ADDR=0x{int(rhs_addr):016X}\tWRITE_ADDR=0x{int(write_addr):016X}"
            )
            if path:
                line += f"\tITER_NUM={int(iter_num)}\tPATH={path}"
            line += "\n"
            f.write(line)


def write_debug_json(path: str, packets: List[PacketInstruction], row_address_map: Dict[str, Dict[str, int]]) -> None:
    obj = {
        "row_address_map": row_address_map,
        "packets": [asdict(p) for p in packets],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compile row ISA to packet ISA via explicit address mapping")
    ap.add_argument("--row-isa", required=True, help="Input row ISA file")
    ap.add_argument("--packet-isa", required=True, help="Output packet ISA file")
    ap.add_argument(
        "--packet-json",
        default="",
        help="Optional JSON debug output (defaults to <packet-isa>.json)",
    )
    ap.add_argument("--packet-bytes", type=int, default=256, help="Payload bytes per packet")
    ap.add_argument("--row-addr-stride", type=lambda x: int(x, 0), default=0x0010_0000, help="Address stride between rows")
    ap.add_argument("--exp-backend", choices=["scalar_exp", "exp"], default="scalar_exp", help="Backend label for exp-like scalar paths")
    args = ap.parse_args()

    rows = parse_row_lines(args.row_isa)
    packets, row_map = compile_packets(
        rows,
        packet_bytes=max(1, args.packet_bytes),
        row_addr_stride=max(1, args.row_addr_stride),
        exp_backend=args.exp_backend,
    )
    write_packet_isa(args.packet_isa, packets)
    json_path = args.packet_json or (args.packet_isa + ".json")
    write_debug_json(json_path, packets, row_map)
    print("Wrote", args.packet_isa, "and", json_path)


if __name__ == "__main__":
    main()
