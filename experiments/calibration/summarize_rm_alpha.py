#!/usr/bin/env python3
"""Summarize the epsilon_M/alpha calibration grid emitted by hh_bench.

Expected CSV names:
  experiments/calibration/margin_service_calibration/csv/
  <dataset>_rm<r_m>_alpha<alpha>.csv

The script writes service, probing, memory, and quality summaries to stdout.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path
from statistics import mean, stdev


ROOT = Path("experiments/calibration/margin_service_calibration/csv")


FIELDS = [
    "dataset",
    "n",
    "epsilon_m",
    "r_m",
    "alpha",
    "windows",
    "N_global",
    "service_violation_rate",
    "margin_alpha_mean",
    "margin_budget_ratio_mean",
    "q_up_active_mean",
    "q_up_lift_mean",
    "probe_count",
    "probe_failure_count",
    "probe_failure_rate",
    "aae",
    "are",
    "mem_worker_total_kib",
    "q_mean",
    "q_volatility",
    "hh_f1",
    "hh_recall",
    "candidate_hh_recall",
    "ambiguous_mass",
    "interval_width_avg",
    "interval_width_amb_avg",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize epsilon_M/alpha calibration CSVs.")
    parser.add_argument(
        "--root",
        default=str(ROOT),
        help="Directory containing per-run calibration CSV files.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset name to include. May be passed multiple times. Defaults to all datasets.",
    )
    parser.add_argument(
        "--default-n",
        type=int,
        default=200,
        help="Fallback n for legacy dataset names that do not include an _n<value> suffix.",
    )
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


def finite_values(rows: list[dict[str, str]], key: str) -> list[float]:
    return [
        value
        for row in rows
        if math.isfinite(value := parse_value(row, key))
    ]


def parse_name(path: Path) -> tuple[str, str, str]:
    if "_rm" not in path.stem or "_alpha" not in path.stem:
        raise ValueError(f"unexpected calibration CSV name: {path.name}")
    dataset, rest = path.stem.split("_rm", 1)
    r_m, alpha = rest.split("_alpha", 1)
    return dataset, r_m, alpha


def infer_n(dataset: str, default_n: int) -> int:
    match = re.search(r"(?:^|_)n([0-9]+)(?:_|$)", dataset)
    if match:
        return int(match.group(1))
    return default_n


def summarize_file(path: Path, default_n: int) -> dict[str, object]:
    dataset, r_m, alpha = parse_name(path)
    n = infer_n(dataset, default_n)
    r_m_value = float(r_m)
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
    violations = finite_values(rows, "service_violation")
    margins = finite_values(rows, "margin_alpha")
    probes = finite_values(rows, "probe_issued")
    probe_failures = finite_values(rows, "probe_failed")
    q_up_active = [
        parse_value(row, "q_up")
        for row in rows
        if parse_value(row, "q_up") > 0
    ]
    q_up_lifts = [
        parse_value(row, "q_up") / parse_value(row, "q_current")
        for row in rows
        if parse_value(row, "q_up") > 0
        and parse_value(row, "q_current") > 0
    ]
    probe_count = sum(probes)
    probe_failure_count = sum(probe_failures)

    return {
        "dataset": dataset,
        "n": n,
        "epsilon_m": n * r_m_value,
        "r_m": r_m,
        "alpha": alpha,
        "windows": len(rows),
        "N_global": avg(rows, "N_global"),
        "service_violation_rate": mean(violations) if violations else math.nan,
        "margin_alpha_mean": mean(margins) if margins else math.nan,
        "margin_budget_ratio_mean": (
            mean(margins) / r_m_value if margins and r_m_value > 0 else math.nan
        ),
        "q_up_active_mean": mean(q_up_active) if q_up_active else 0.0,
        "q_up_lift_mean": mean(q_up_lifts) if q_up_lifts else 1.0,
        "probe_count": probe_count,
        "probe_failure_count": probe_failure_count,
        "probe_failure_rate": (
            probe_failure_count / probe_count if probe_count > 0 else 0.0
        ),
        "aae": avg(rows, "aae"),
        "are": avg(rows, "are"),
        "mem_worker_total_kib": avg(rows, "mem_worker_total_kib"),
        "q_mean": mean(q_vals) if q_vals else math.nan,
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
    selected_datasets = set(args.dataset)
    paths = sorted(root.glob("*.csv"))
    if selected_datasets:
        paths = [path for path in paths if parse_name(path)[0] in selected_datasets]
    if not paths:
        print(f"error: no calibration CSVs found under {root}", file=sys.stderr)
        if selected_datasets:
            print(f"selected datasets: {', '.join(sorted(selected_datasets))}", file=sys.stderr)
        print("run the epsilon_M/alpha hh_bench grid first, then rerun this summarizer", file=sys.stderr)
        return 1

    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDS)
    writer.writeheader()
    for path in paths:
        row = summarize_file(path, args.default_n)
        writer.writerow({field: format_cell(row.get(field, "")) for field in FIELDS})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
