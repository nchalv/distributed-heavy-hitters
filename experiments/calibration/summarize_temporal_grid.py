#!/usr/bin/env python3
"""Summarize the Section 8.2.2 temporal-controller ablation."""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path
from statistics import mean, stdev


ROOT = Path("experiments/calibration/temporal_controller_ablation/csv")

FIELDS = [
    "dataset",
    "n",
    "mode",
    "windows",
    "N_global",
    "service_violation_rate",
    "probe_count",
    "probe_failure_count",
    "probe_failure_rate",
    "aae",
    "are",
    "mem_worker_total_kib",
    "q_mean",
    "q_next_mean",
    "q_volatility",
    "hh_f1",
    "hh_recall",
    "candidate_hh_recall",
    "ambiguous_mass",
    "interval_width_avg",
    "interval_width_amb_avg",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize temporal-controller ablation CSVs."
    )
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


def parse_name(path: Path) -> tuple[str, str] | None:
    # Check the longest suffix first: ``probing`` is itself a suffix of
    # ``residual_guarded_probing``.
    for mode in (
        "pressure_gated_comfort_probing",
        "comfort_guided_probing",
        "residual_guarded_probing",
        "probing",
    ):
        suffix = f"_{mode}"
        if path.stem.endswith(suffix):
            dataset = path.stem[: -len(suffix)]
            return (dataset, mode) if dataset else None
    return None


def infer_n(dataset: str, default_n: int) -> int:
    match = re.search(r"(?:^|_)n([0-9]+)(?:_|$)", dataset)
    if match:
        return int(match.group(1))
    return default_n


def summarize_file(path: Path, default_n: int) -> dict[str, object]:
    parsed = parse_name(path)
    if parsed is None:
        raise ValueError(f"unexpected temporal ablation CSV name: {path.name}")
    dataset, mode = parsed
    n = infer_n(dataset, default_n)
    with path.open(newline="", encoding="utf-8") as fh:
        rows = [
            row
            for row in csv.DictReader(fh)
            if row.get("method_type") == "ss" and "policy=difficulty" in row.get("method", "")
        ]

    if not rows:
        raise ValueError(f"no difficulty-policy Space-Saving rows found in {path}")

    q_vals = [parse_value(row, "q_current") for row in rows]
    q_vals = [value for value in q_vals if math.isfinite(value)]
    q_next_vals = [parse_value(row, "q_next") for row in rows]
    q_next_vals = [value for value in q_next_vals if math.isfinite(value)]
    service_violations = [
        parse_value(row, "service_violation")
        for row in rows
        if math.isfinite(parse_value(row, "service_violation"))
    ]
    probes = [
        parse_value(row, "probe_issued")
        for row in rows
        if math.isfinite(parse_value(row, "probe_issued"))
    ]
    probe_failures = [
        parse_value(row, "probe_failed")
        for row in rows
        if math.isfinite(parse_value(row, "probe_failed"))
    ]
    probe_count = sum(probes)
    probe_failure_count = sum(probe_failures)

    return {
        "dataset": dataset,
        "n": n,
        "mode": mode,
        "windows": len(rows),
        "N_global": avg(rows, "N_global"),
        "service_violation_rate": (
            mean(service_violations) if service_violations else math.nan
        ),
        "probe_count": probe_count,
        "probe_failure_count": probe_failure_count,
        "probe_failure_rate": (
            probe_failure_count / probe_count if probe_count > 0 else 0.0
        ),
        "aae": avg(rows, "aae"),
        "are": avg(rows, "are"),
        "mem_worker_total_kib": avg(rows, "mem_worker_total_kib"),
        "q_mean": mean(q_vals) if q_vals else math.nan,
        "q_next_mean": mean(q_next_vals) if q_next_vals else math.nan,
        "q_volatility": stdev(q_vals) if len(q_vals) > 1 else 0.0,
        "hh_f1": avg(rows, "hh_f1"),
        "hh_recall": avg(rows, "hh_recall"),
        "candidate_hh_recall": avg(rows, "candidate_hh_recall"),
        "ambiguous_mass": avg(rows, "ambiguous_mass"),
        "interval_width_avg": avg(rows, "interval_width_avg"),
        "interval_width_amb_avg": avg(rows, "interval_width_amb_avg"),
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
        print(
            f"error: no temporal-controller ablation CSVs found under {root}",
            file=sys.stderr,
        )
        return 1

    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDS)
    writer.writeheader()
    for path in paths:
        if parse_name(path) is None:
            print(f"warning: skipping stale temporal CSV with old name: {path.name}", file=sys.stderr)
            continue
        row = summarize_file(path, args.default_n)
        writer.writerow({field: format_cell(row.get(field, "")) for field in FIELDS})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
