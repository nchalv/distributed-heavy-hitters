#!/usr/bin/env python3
"""Summarize the hybrid exact-approximate promotion-policy ablation."""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path
from statistics import mean, stdev


ROOT = Path("experiments/calibration/hybrid_ablation/csv")

FIELDS = [
    "dataset",
    "n",
    "head_policy",
    "method",
    "windows",
    "N_global",
    "aae",
    "threshold_normalized_error",
    "hh_f1",
    "hh_recall",
    "candidate_hh_recall",
    "q_mean",
    "q_next_mean",
    "q_volatility",
    "q_head_mean",
    "q_tail_mean",
    "q_head_next_mean",
    "q_tail_next_mean",
    "mem_worker_total_kib",
    "candidate_count",
    "ambiguous_count",
    "ambiguous_mass",
    "interval_width_avg",
    "interval_width_amb_avg",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize hybrid ablation CSV files.")
    parser.add_argument("--root", default=str(ROOT), help="Directory containing per-run CSV files.")
    parser.add_argument("--default-n", type=int, default=200, help="Fallback n for legacy dataset names.")
    return parser.parse_args()


def parse_name(path: Path) -> tuple[str, str] | None:
    match = re.match(r"(.+)_(candidate)$", path.stem)
    if match:
        return match.groups()
    return None


def infer_n(dataset: str, default_n: int) -> int:
    match = re.search(r"(?:^|_)n([0-9]+)(?:_|$)", dataset)
    if match:
        return int(match.group(1))
    return default_n


def fnum(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def avg(rows: list[dict[str, str]], key: str) -> float:
    vals = [fnum(row, key) for row in rows]
    vals = [value for value in vals if math.isfinite(value)]
    return mean(vals) if vals else math.nan


def summarize_rows(
    dataset: str,
    n: int,
    head_policy: str,
    method: str,
    rows: list[dict[str, str]],
) -> dict[str, object]:
    q_vals = [fnum(row, "q_current") for row in rows]
    q_vals = [value for value in q_vals if math.isfinite(value)]
    q_next_vals = [fnum(row, "q_next") for row in rows]
    q_next_vals = [value for value in q_next_vals if math.isfinite(value)]
    n_global = avg(rows, "N_global")
    aae = avg(rows, "aae")
    threshold_error = (n * aae / n_global) if math.isfinite(aae) and n_global > 0 else math.nan

    return {
        "dataset": dataset,
        "n": n,
        "head_policy": head_policy,
        "method": method,
        "windows": len(rows),
        "N_global": n_global,
        "aae": aae,
        "threshold_normalized_error": threshold_error,
        "hh_f1": avg(rows, "hh_f1"),
        "hh_recall": avg(rows, "hh_recall"),
        "candidate_hh_recall": avg(rows, "candidate_hh_recall"),
        "q_mean": mean(q_vals) if q_vals else math.nan,
        "q_next_mean": mean(q_next_vals) if q_next_vals else math.nan,
        "q_volatility": stdev(q_vals) if len(q_vals) > 1 else 0.0,
        "q_head_mean": avg(rows, "q_head_current"),
        "q_tail_mean": avg(rows, "q_tail_current"),
        "q_head_next_mean": avg(rows, "q_head_next"),
        "q_tail_next_mean": avg(rows, "q_tail_next"),
        "mem_worker_total_kib": avg(rows, "mem_worker_total_kib"),
        "candidate_count": avg(rows, "candidate_count"),
        "ambiguous_count": avg(rows, "ambiguous_count"),
        "ambiguous_mass": avg(rows, "ambiguous_mass"),
        "interval_width_avg": avg(rows, "interval_width_avg"),
        "interval_width_amb_avg": avg(rows, "interval_width_amb_avg"),
    }


def summarize_file(path: Path, default_n: int) -> list[dict[str, object]]:
    parsed = parse_name(path)
    if parsed is None:
        raise ValueError(f"unexpected hybrid ablation CSV name: {path.name}")
    dataset, head_policy = parsed
    n = infer_n(dataset, default_n)
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    by_method = {
        "ss": [
            row
            for row in rows
            if row.get("method_type") == "ss" and "policy=difficulty" in row.get("method", "")
        ],
        "hybrid": [
            row
            for row in rows
            if row.get("method_type") == "hybrid"
        ],
    }
    summaries = []
    for method, method_rows in by_method.items():
        if not method_rows:
            raise ValueError(f"no {method} rows found in {path}")
        summaries.append(summarize_rows(dataset, n, head_policy, method, method_rows))
    return summaries


def format_cell(value: object) -> object:
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.10g}"
    return value


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    paths = sorted(root.glob("*.csv"))
    if not paths:
        print(f"error: no hybrid ablation CSVs found under {root}", file=sys.stderr)
        return 1

    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDS)
    writer.writeheader()
    for path in paths:
        parsed = parse_name(path)
        if parsed is None:
            print(f"warning: skipping stale hybrid CSV with old name: {path.name}", file=sys.stderr)
            continue
        for row in summarize_file(path, args.default_n):
            writer.writerow({field: format_cell(row.get(field, "")) for field in FIELDS})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
