"""
Record operations offloaded to CompAir NoC (booksim2) or SRAM-PIM when generating CENT traces.

Enable via environment COMPAIR_RECORD_OFFLOAD=1 or call set_enabled(True) in-process
before trace generation (function_sim_llama does this when any offload flag is set).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

_events: Dict[str, List[Dict[str, Any]]] = {"noc": [], "sram": []}
_enabled = os.environ.get("COMPAIR_RECORD_OFFLOAD", "").lower() in ("1", "true", "yes")


def set_enabled(v: bool) -> None:
    global _enabled
    _enabled = bool(v)


def is_enabled() -> bool:
    return _enabled


def reset() -> None:
    global _events
    _events = {"noc": [], "sram": []}


def record_noc(op: str, **meta: Any) -> None:
    if not _enabled:
        return
    row = {"op": op, **meta}
    _events["noc"].append(row)


def record_sram(op: str, count: int = 1, **meta: Any) -> None:
    if not _enabled:
        return
    _events["sram"].append({"op": op, "count": int(count), **meta})


def snapshot() -> Dict[str, List[Dict[str, Any]]]:
    return {"noc": list(_events["noc"]), "sram": list(_events["sram"])}


def save(path: str, extra: Optional[Dict[str, Any]] = None) -> None:
    payload: Dict[str, Any] = {"noc": _events["noc"], "sram": _events["sram"]}
    if extra:
        payload["meta"] = extra
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def clear() -> None:
    reset()
