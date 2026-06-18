#!/usr/bin/env python3
"""Summarize the ambiguity-adjustment ablation emitted by hh_bench."""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path
from statistics import mean, stdev


ROOT = Path("experiments/calibration/ambiguity_guard_ablation/csv")

FIELDS = [
    "dataset",
    "n",
    "epsilon_m",
    "mode",
    "windows",
    "N_global",
    "miss_eff",
    "miss_req",
    "aae",
    "over_eff",
    "over_req",
    "calib_bias",
    "mem_worker_total_kib",
    "q_mean",
    "q_next_mean",
    "q_volatility",
    "hh_f1",
    "hh_recall",
    "candidate_hh_recall",
    "ambiguous_count",
    "ambiguous_mass",
    "interval_width_avg",
    "interval_width_amb_avg",
    "ambiguity_actuation_rate",
    "ambiguity_lift_mean",
    "ambiguity_lift_active_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize ambiguity calibration CSVs.")
    parser.add_argument("--root", default=str(ROOT), help="Directory containing per-run CSV files.")
    parser.add_argument("--default-n", type=int, default=200, help="Fallback n for legacy dataset names.")
    return parser.parse_args()


def parse_value(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def avg(rows: list[dict[str, str]], key: str) -> float:
    vals = [parse_value(row, key) for row in rows]
    vals = [value for value in vals if math.isfinite(value)]
    return mean(vals) if vals else math.nan


def parse_name(path: Path) -> tuple[str, str, float] | None:
    match = re.match(r"(.+)_eps([0-9_]+)_(baseline|ambiguity)$", path.stem)
    if match:
        dataset, epsilon_tag, mode = match.groups()
        epsilon_m = float(epsilon_tag.replace("_", ".").rstrip("."))
        return dataset, mode, epsilon_m
    match = re.match(r"(.+)_(baseline|ambiguity)$", path.stem)
    if match:
        dataset, mode = match.groups()
        return dataset, mode, math.nan
    return None


def infer_n(dataset: str, default_n: int) -> int:
    match = re.search(r"(?:^|_)n([0-9]+)(?:_|$)", dataset)
    if match:
        return int(match.group(1))
    return default_n


def summarize_file(path: Path, default_n: int) -> dict[str, object]:
    parsed = parse_name(path)
    if parsed is None:
        raise ValueError(f"unexpected ambiguity calibration CSV name: {path.name}")
    dataset, mode, epsilon_m = parsed
    n = infer_n(dataset, default_n)
    with path.open(newline="", encoding="utf-8") as fh:
        rows = [
            row
            for row in csv.DictReader(fh)
            if row.get("method_type") == "ss" and "policy=difficulty" in row.get("method", "")
        ]

    if not rows:
        raise ValueError(f"no difficulty-policy Space-Saving rows found in {path}")
    if not math.isfinite(epsilon_m):
        method = rows[0].get("method", "")
        match = re.search(r"r-m=([0-9.eE+-]+)", method)
        epsilon_m = float(match.group(1)) * n if match else math.nan

    q_vals = [parse_value(row, "q_current") for row in rows]
    q_vals = [value for value in q_vals if math.isfinite(value)]
    q_next_vals = [parse_value(row, "q_next") for row in rows]
    q_next_vals = [value for value in q_next_vals if math.isfinite(value)]
    lifts = []
    for row in rows:
        baseline = parse_value(row, "q_baseline")
        adjusted = parse_value(row, "q_eff_pred_tilde")
        if math.isfinite(baseline) and math.isfinite(adjusted):
            lifts.append(max(0.0, adjusted - baseline))
    active_lifts = [value for value in lifts if value > 0.0]

    return {
        "dataset": dataset,
        "n": n,
        "epsilon_m": epsilon_m,
        "mode": mode,
        "windows": len(rows),
        "N_global": avg(rows, "N_global"),
        "miss_eff": avg(rows, "miss_eff"),
        "miss_req": avg(rows, "miss_req"),
        "aae": avg(rows, "aae"),
        "over_eff": avg(rows, "over_eff"),
        "over_req": avg(rows, "over_req"),
        "calib_bias": avg(rows, "calib_bias"),
        "mem_worker_total_kib": avg(rows, "mem_worker_total_kib"),
        "q_mean": mean(q_vals) if q_vals else math.nan,
        "q_next_mean": mean(q_next_vals) if q_next_vals else math.nan,
        "q_volatility": stdev(q_vals) if len(q_vals) > 1 else 0.0,
        "hh_f1": avg(rows, "hh_f1"),
        "hh_recall": avg(rows, "hh_recall"),
        "candidate_hh_recall": avg(rows, "candidate_hh_recall"),
        "ambiguous_count": avg(rows, "ambiguous_count"),
        "ambiguous_mass": avg(rows, "ambiguous_mass"),
        "interval_width_avg": avg(rows, "interval_width_avg"),
        "interval_width_amb_avg": avg(rows, "interval_width_amb_avg"),
        "ambiguity_actuation_rate": (
            sum(value > 0.0 for value in lifts) / len(lifts) if lifts else math.nan
        ),
        "ambiguity_lift_mean": mean(lifts) if lifts else math.nan,
        "ambiguity_lift_active_mean": (
            mean(active_lifts) if active_lifts else 0.0
        ),
    }


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
        print(f"error: no ambiguity calibration CSVs found under {root}", file=sys.stderr)
        return 1

    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDS)
    writer.writeheader()
    for path in paths:
        if parse_name(path) is None:
            print(f"warning: skipping stale ambiguity CSV with old name: {path.name}", file=sys.stderr)
            continue
        row = summarize_file(path, args.default_n)
        writer.writerow({field: format_cell(row.get(field, "")) for field in FIELDS})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
