#!/usr/bin/env python3
"""Plot controller diagnostics for the epsilon_M/alpha calibration summary."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path


DEFAULT_SUMMARY = Path("experiments/calibration/margin_service_calibration/summary.csv")
DEFAULT_OUT = Path("experiments/calibration/margin_service_calibration/plots")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot epsilon_M/alpha controller diagnostics."
    )
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY), help="Summary CSV produced by summarize_rm_alpha.py.")
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

    return plt


def marker_for_alpha(alpha: float) -> str:
    markers = {
        0.90: "o",
        0.95: "s",
        0.98: "^",
        0.995: "D",
    }
    return markers.get(round(alpha, 3), "o")


def margin_fraction(row: dict[str, str]) -> float:
    epsilon = fnum(row, "epsilon_m")
    if math.isfinite(epsilon):
        return epsilon
    legacy_eta = fnum(row, "eta_m")
    if math.isfinite(legacy_eta):
        return legacy_eta
    n = fnum(row, "n")
    r_m = fnum(row, "r_m")
    if math.isfinite(n) and math.isfinite(r_m):
        return n * r_m
    return math.nan


def plot_controller_diagnostics(
    plt,
    rows: list[dict[str, str]],
    out_dir: Path,
) -> None:
    n_values = sorted(
        {
            int(round(fnum(row, "n")))
            for row in rows
            if math.isfinite(fnum(row, "n"))
        }
    )
    alpha_values = sorted(
        {
            fnum(row, "alpha")
            for row in rows
            if math.isfinite(fnum(row, "alpha"))
        }
    )
    colors = plt.cm.tab10(
        [i / max(1, len(alpha_values) - 1) for i in range(len(alpha_values))]
    )
    color_by_alpha = {
        alpha: colors[index] for index, alpha in enumerate(alpha_values)
    }
    metrics = [
        (
            "service_violation_rate",
            "Observed service-violation rate",
            lambda row: fnum(row, "service_violation_rate"),
        ),
        (
            "q_mean",
            "Mean deployed capacity $q/n$",
            lambda row: fnum(row, "q_mean") / fnum(row, "n"),
        ),
        (
            "probe_failure_rate",
            "Failed probes / issued probes",
            lambda row: fnum(row, "probe_failure_rate"),
        ),
    ]

    fig, axes = plt.subplots(
        len(n_values),
        len(metrics),
        figsize=(10.8, 2.55 * len(n_values) + 0.8),
        sharex=True,
        squeeze=False,
    )
    panel_index = 0
    for row_index, n_value in enumerate(n_values):
        for column_index, (_, title, value_fn) in enumerate(metrics):
            ax = axes[row_index][column_index]
            for alpha in alpha_values:
                subset = sorted(
                    [
                        row
                        for row in rows
                        if int(round(fnum(row, "n"))) == n_value
                        and math.isclose(fnum(row, "alpha"), alpha)
                        and math.isfinite(margin_fraction(row))
                        and math.isfinite(value_fn(row))
                    ],
                    key=margin_fraction,
                )
                if not subset:
                    continue
                ax.plot(
                    [margin_fraction(row) for row in subset],
                    [value_fn(row) for row in subset],
                    marker=marker_for_alpha(alpha),
                    markersize=4,
                    linewidth=1.2,
                    color=color_by_alpha[alpha],
                    label=fr"$\alpha={alpha:g}$",
                )
            panel_label = chr(ord("a") + panel_index)
            ax.set_title(f"({panel_label}) {title}", fontsize=9)
            panel_index += 1
            ax.grid(True, alpha=0.25)
            ax.tick_params(labelsize=8)
            if column_index == 0:
                ax.set_ylabel(fr"$n={n_value}$", fontsize=9)
            if row_index == len(n_values) - 1:
                ax.set_xlabel(r"Margin fraction $\varepsilon_M$", fontsize=9)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(alpha_values),
        frameon=False,
        fontsize=8,
    )
    fig.suptitle(
        "Margin and Service-Fraction Controller Diagnostics",
        fontsize=11,
        y=1.035,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(
        out_dir / "margin_service_diagnostics.pdf",
        bbox_inches="tight",
        pad_inches=0.05,
    )
    fig.savefig(
        out_dir / "margin_service_diagnostics.png",
        dpi=220,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close(fig)


def main() -> int:
    args = parse_args()
    summary = Path(args.summary)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(summary)
    if not rows:
        raise RuntimeError(f"no rows found in {summary}")

    plt = import_matplotlib()
    plot_controller_diagnostics(plt, rows, out_dir)

    print(f"plots written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
