#!/usr/bin/env python3
"""Plot the ambiguity-adjustment ablation."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path


DEFAULT_SUMMARY = Path("experiments/calibration/ambiguity_guard_ablation/summary.csv")
DEFAULT_OUT = Path("experiments/calibration/ambiguity_guard_ablation/plots")
DEFAULT_WINDOW_SIZE = 60000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot ambiguity adjustment ablation results.")
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY), help="Summary CSV produced by summarize_ambiguity_grid.py.")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output plot directory.")
    parser.add_argument("--window-size", type=float, default=DEFAULT_WINDOW_SIZE, help="Fallback N for legacy summaries.")
    return parser.parse_args()


def fnum(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def import_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/hh_matplotlib")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def threshold_normalized_error(row: dict[str, str], fallback_window_size: float) -> float:
    n = fnum(row, "n")
    aae = fnum(row, "aae")
    window_size = fnum(row, "N_global")
    if not math.isfinite(window_size):
        window_size = fallback_window_size
    if not (math.isfinite(n) and math.isfinite(aae) and window_size > 0):
        return math.nan
    return n * aae / window_size


def epsilon_key(row: dict[str, str]) -> float | None:
    epsilon_m = fnum(row, "epsilon_m")
    return epsilon_m if math.isfinite(epsilon_m) else None


def dataset_label(dataset: str) -> str:
    labels = {
        "persistent_ambiguity_adversary_n100": r"Persistent ambiguity, $n=100$",
        "persistent_ambiguity_adversary_n200": r"Persistent ambiguity, $n=200$",
        "persistent_ambiguity_adversary_n400": r"Persistent ambiguity, $n=400$",
        "round_robin_n200": r"Round robin, $n=200$",
        "temporal_pressure_step_locality_n200": "Step",
        "temporal_pressure_ramp_locality_n200": "Ramp",
        "temporal_pressure_burst_locality_n200": "Burst",
    }
    if dataset in labels:
        return labels[dataset]
    if dataset.startswith("milp_certificate_adversary_n"):
        return fr"$n={dataset.rsplit('n', 1)[-1]}$"
    return dataset.replace("_", " ")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        row
        for row in load_rows(Path(args.summary))
        if all(math.isfinite(fnum(row, key)) for key in ("n", "q_mean", "aae"))
        and math.isfinite(threshold_normalized_error(row, args.window_size))
        and row.get("mode") in {"baseline", "ambiguity"}
    ]
    if not rows:
        raise RuntimeError(f"no usable ambiguity ablation rows found in {args.summary}")

    plt = import_matplotlib()
    datasets = sorted({row.get("dataset", "") for row in rows})
    eps_values = sorted({
        epsilon_key(row)
        for row in rows
        if epsilon_key(row) is not None
    })
    if not eps_values:
        eps_values = [None]
    modes = ["baseline", "ambiguity"]
    configs = [(eps, mode) for eps in eps_values for mode in modes]
    x_by_config = {config: i for i, config in enumerate(configs)}
    colors = plt.cm.tab10([i for i in range(len(datasets))])
    color_by_dataset = {dataset: colors[i] for i, dataset in enumerate(datasets)}

    fig, axes = plt.subplots(
        2, 2, figsize=(8.6, 5.5), sharex=True
    )
    ax_err, ax_mem, ax_amb, ax_act = axes.flat

    for dataset in datasets:
        group_rows = sorted(
            [row for row in rows if row.get("dataset", "") == dataset],
            key=lambda row: x_by_config[(epsilon_key(row), row.get("mode", ""))],
        )
        xs = [
            x_by_config[(epsilon_key(row), row.get("mode", ""))]
            for row in group_rows
        ]
        ys_err = [threshold_normalized_error(row, args.window_size) for row in group_rows]
        ys_mem = [fnum(row, "q_mean") / fnum(row, "n") for row in group_rows]
        ys_amb = [fnum(row, "ambiguous_count") for row in group_rows]
        ys_act = [fnum(row, "ambiguity_actuation_rate") for row in group_rows]
        label = dataset_label(dataset)
        color = color_by_dataset[dataset]
        ax_err.plot(xs, ys_err, marker="o", markersize=3.8, linewidth=1.3, color=color, label=label)
        ax_mem.plot(xs, ys_mem, marker="D", markersize=3.4, linewidth=1.2, color=color, label=label)
        ax_amb.plot(xs, ys_amb, marker="s", markersize=3.4, linewidth=1.2, color=color)
        ax_act.plot(xs, ys_act, marker="^", markersize=3.6, linewidth=1.2, color=color)

    ax_err.set_title("(a) Threshold-normalized error", fontsize=9.5)
    ax_mem.set_title(r"(b) Mean deployed capacity $q/n$", fontsize=9.5)
    ax_amb.set_title("(c) Mean ambiguous-candidate count", fontsize=9.5)
    ax_act.set_title("(d) Ambiguity actuation rate", fontsize=9.5)
    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(len(configs)))
        labels = []
        for eps, mode in configs:
            prefix = rf"$\epsilon_M={eps:.2f}$" if eps is not None else ""
            suffix = "Off" if mode == "baseline" else "On"
            labels.append(f"{prefix}\n{suffix}")
        ax.set_xticklabels(labels)
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax_act.set_ylim(bottom=0)

    handles, labels = ax_err.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7.5, frameon=False)
    fig.suptitle("Ambiguity-Guard Ablation", fontsize=11)
    fig.tight_layout(rect=(0, 0.12, 1, 0.96))
    fig.savefig(out_dir / "ambiguity_guard_ablation.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "ambiguity_guard_ablation.png", dpi=220, bbox_inches="tight")
    print(f"plots written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
