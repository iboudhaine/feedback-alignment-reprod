"""CSV writer + filename builder shared by all tasks."""
from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from pathlib import Path

RESULTS_DIR = Path("results")


def write_rows(path: Path, rows: list[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def stream_rows(path: Path, fieldnames: list[str]):
    """Context-manager-friendly helper for sweeps that write incrementally."""
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    return f, writer


def build_slug(parts: Iterable[tuple[str, object]]) -> str:
    """Build a filename slug from (key, value) pairs, skipping None/empty values."""
    bits: list[str] = []
    for k, v in parts:
        if v is None or v == "":
            continue
        if isinstance(v, bool):
            if v:
                bits.append(k)
            continue
        bits.append(f"{k}{v}")
    return "_".join(bits)
