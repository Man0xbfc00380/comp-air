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


def compile_packets(
    rows: List[Tuple[int, str, Dict[str, str]]],
    packet_bytes: int,
    row_addr_stride: int,
) -> Tuple[List[PacketInstruction], Dict[str, Dict[str, int]]]:
    packets: List[PacketInstruction] = []
    row_address_map: Dict[str, Dict[str, int]] = {}
    next_packet_id = 0

    for row_id, _, parts in rows:
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

        total_chunks = int(math.ceil(row_bytes / packet_bytes))
        for chunk_idx in range(total_chunks):
            start = base + chunk_idx * packet_bytes
            remaining = row_bytes - chunk_idx * packet_bytes
            sz = min(packet_bytes, max(0, remaining))
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
                    },
                )
            )
            next_packet_id += 1
    return packets, row_address_map


def write_packet_isa(out_path: str, packets: List[PacketInstruction]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# CompAir packet-level ISA (generated from row-level ISA)\n")
        f.write("# PKT_ID\tROW_ID\tTARGET\tROLE\tOP\tADDR_START\tADDR_END\tBYTES\tCHUNK\n")
        for pkt in packets:
            f.write(
                "PKT\t"
                f"ID={pkt.packet_id}\tROW={pkt.row_id}\tTARGET={pkt.target}\tROLE={pkt.role}\t"
                f"OP={pkt.op}\tADDR_START=0x{pkt.addr_start:016X}\tADDR_END=0x{pkt.addr_end:016X}\t"
                f"BYTES={pkt.bytes}\tCHUNK={pkt.chunk_idx + 1}/{pkt.chunk_total}\n"
            )


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
    args = ap.parse_args()

    rows = parse_row_lines(args.row_isa)
    packets, row_map = compile_packets(rows, packet_bytes=max(1, args.packet_bytes), row_addr_stride=max(1, args.row_addr_stride))
    write_packet_isa(args.packet_isa, packets)
    json_path = args.packet_json or (args.packet_isa + ".json")
    write_debug_json(json_path, packets, row_map)
    print("Wrote", args.packet_isa, "and", json_path)


if __name__ == "__main__":
    main()
