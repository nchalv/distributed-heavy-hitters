#!/usr/bin/env python3
"""Plot the hybrid exact-approximate promotion-policy ablation."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path


DEFAULT_SUMMARY = Path("experiments/calibration/hybrid_ablation/summary.csv")
DEFAULT_OUT = Path("experiments/calibration/hybrid_ablation/plots")


SERIES = [
    ("ss", "candidate", "Space-Saving", "#6C757D", "///"),
    ("hybrid", "candidate", "Hybrid: top/head + frontier", "#287271", ""),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot hybrid exact-approximate tracking ablation results."
    )
    parser.add_argument(
        "--summary",
        default=str(DEFAULT_SUMMARY),
        help="Summary CSV produced by summarize_hybrid_ablation.py.",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output plot directory.")
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
    from matplotlib.patches import Patch
    from matplotlib.ticker import FormatStrFormatter

    return plt, Patch, FormatStrFormatter


def family_key(dataset: str) -> tuple[int, str]:
    if dataset.startswith("milp_certificate_adversary"):
        return (0, "MILP stress")
    if dataset.startswith("persistent_ambiguity_adversary"):
        return (1, "Persistent locality")
    return (2, dataset.replace("_", " "))


def n_label(row: dict[str, str]) -> str:
    n = fnum(row, "n")
    if math.isfinite(n):
        return f"$n={int(round(n))}$"
    return row.get("dataset", "").replace("_", " ")


def index_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, str]]:
    indexed: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        dataset = row.get("dataset", "")
        method = row.get("method", "")
        policy = row.get("head_policy", "")
        if method == "ss":
            # The baseline is duplicated once per head policy because each
            # source CSV contains both Space-Saving and hybrid rows. Keep the
            # candidate copy as the canonical baseline for plotting.
            if policy != "candidate":
                continue
            policy = "candidate"
        indexed[(dataset, method, policy)] = row
    return indexed


def panel_data(
    rows: list[dict[str, str]],
) -> dict[str, list[str]]:
    families: dict[str, list[str]] = {}
    for row in rows:
        dataset = row.get("dataset", "")
        order, family = family_key(dataset)
        if order >= 2:
            continue
        families.setdefault(family, [])
        if dataset not in families[family]:
            families[family].append(dataset)
    for family, datasets in families.items():
        datasets.sort(key=lambda dataset: min(fnum(row, "n") for row in rows if row.get("dataset") == dataset))
    return families


def metric_value(row: dict[str, str], metric: str) -> float:
    if metric == "error":
        return 1_000.0 * fnum(row, "threshold_normalized_error")
    if metric == "memory":
        return fnum(row, "mem_worker_total_kib")
    if metric == "f1":
        return 100.0 * fnum(row, "hh_f1")
    raise ValueError(metric)


def add_grouped_bars(
    ax,
    *,
    datasets: list[str],
    indexed: dict[tuple[str, str, str], dict[str, str]],
    metric: str,
    ylabel: str,
    title: str,
    yformatter=None,
) -> None:
    width = 0.30
    offsets = [-width / 2.0, width / 2.0]
    xs = list(range(len(datasets)))

    for (method, policy, label, color, hatch), offset in zip(SERIES, offsets):
        values = []
        for dataset in datasets:
            row = indexed.get((dataset, method, policy))
            values.append(metric_value(row, metric) if row is not None else math.nan)
        ax.bar(
            [x + offset for x in xs],
            values,
            width=width,
            label=label,
            color=color,
            edgecolor="#222222",
            linewidth=0.45,
            hatch=hatch,
            alpha=0.92,
        )

    tick_labels = []
    for dataset in datasets:
        row = next(
            (
                indexed[key]
                for key in indexed
                if key[0] == dataset and key[1] == "ss"
            ),
            None,
        )
        tick_labels.append(n_label(row) if row else dataset.replace("_", " "))
    ax.set_xticks(xs)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel(ylabel)
    if yformatter is not None:
        ax.yaxis.set_major_formatter(yformatter)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> int:
    args = parse_args()
    summary = Path(args.summary)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        row
        for row in load_rows(summary)
        if row.get("method") in {"ss", "hybrid"}
        and math.isfinite(fnum(row, "threshold_normalized_error"))
        and math.isfinite(fnum(row, "mem_worker_total_kib"))
    ]
    if not rows:
        raise RuntimeError(f"no usable hybrid ablation rows found in {summary}")

    plt, Patch, FormatStrFormatter = import_matplotlib()
    indexed = index_rows(rows)
    families = panel_data(rows)
    ordered_families = [family for _, family in sorted({family_key(row.get("dataset", "")) for row in rows}) if family in families]

    fig, axes = plt.subplots(
        len(ordered_families),
        3,
        figsize=(10.6, 2.85 * len(ordered_families)),
        sharex=False,
    )
    if len(ordered_families) == 1:
        axes = [axes]

    panel_letters = iter(["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"])
    for row_axes, family in zip(axes, ordered_families):
        datasets = families[family]
        add_grouped_bars(
            row_axes[0],
            datasets=datasets,
            indexed=indexed,
            metric="error",
            ylabel=r"Error ($10^3 n\cdot\mathrm{AAE}/N$)",
            title=f"{next(panel_letters)} {family}: error",
            yformatter=FormatStrFormatter("%.3f"),
        )
        add_grouped_bars(
            row_axes[1],
            datasets=datasets,
            indexed=indexed,
            metric="memory",
            ylabel="Worker memory (KiB)",
            title=f"{next(panel_letters)} {family}: memory",
        )
        add_grouped_bars(
            row_axes[2],
            datasets=datasets,
            indexed=indexed,
            metric="f1",
            ylabel="HH F1 (%)",
            title=f"{next(panel_letters)} {family}: HH F1",
        )
        row_axes[2].set_ylim(bottom=90.0, top=100.5)

    handles = [
        Patch(
            facecolor=color,
            edgecolor="#222222",
            linewidth=0.45,
            hatch=hatch,
            label=label,
        )
        for _, _, label, color, hatch in SERIES
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        fontsize=8,
        frameon=False,
        columnspacing=1.4,
    )
    fig.suptitle("Hybrid Exact-Approximate Tracking Ablation", fontsize=11)
    fig.tight_layout(rect=(0, 0.075, 1, 0.94))

    pdf_path = out_dir / "hybrid_ablation_quality_memory.pdf"
    png_path = out_dir / "hybrid_ablation_quality_memory.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    print(f"plots written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
