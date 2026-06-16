#!/usr/bin/env python3
"""Plot the epsilon_M/alpha calibration frontier across topology settings."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path


DEFAULT_SUMMARIES = [
    Path("experiments/calibration/margin_service_calibration/summary.csv"),
]
DEFAULT_OUT = Path("experiments/calibration/margin_service_calibration/plots")
DEFAULT_WINDOW_SIZE = 60000.0
Y_LABEL = "Threshold-normalized error"
QUALITY_PAIR = (0.10, 0.90)
MEMORY_PAIR = (0.15, 0.95)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot topology sensitivity for epsilon_M/alpha calibration.")
    parser.add_argument(
        "--summary",
        action="append",
        default=[],
        help="Summary CSV to include. May be passed multiple times. Defaults to n=100,200,400 summaries.",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output directory.")
    parser.add_argument(
        "--window-size",
        type=float,
        default=DEFAULT_WINDOW_SIZE,
        help="Fallback window size N for legacy summaries without an N_global column.",
    )
    return parser.parse_args()


def fnum(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def load_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def threshold_normalized_error(row: dict[str, str], fallback_window_size: float) -> float:
    n = fnum(row, "n")
    aae = fnum(row, "aae")
    window_size = fnum(row, "N_global")
    if not math.isfinite(window_size):
        window_size = fallback_window_size
    if not (math.isfinite(n) and math.isfinite(aae) and window_size > 0):
        return math.nan
    return n * aae / window_size


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


def fmt_epsilon(epsilon: float) -> str:
    return f"{epsilon:.2f}"


def promoted_pair(epsilon: float, alpha: float) -> str:
    if math.isclose(epsilon, QUALITY_PAIR[0]) and math.isclose(
        alpha, QUALITY_PAIR[1]
    ):
        return "quality"
    if math.isclose(epsilon, MEMORY_PAIR[0]) and math.isclose(
        alpha, MEMORY_PAIR[1]
    ):
        return "memory"
    return ""


def duplicate_offsets(points: list[tuple[float, float]], x_span: float, y_span: float) -> list[tuple[float, float]]:
    groups: dict[tuple[float, float], list[int]] = {}
    for idx, point in enumerate(points):
        groups.setdefault((round(point[0], 10), round(point[1], 10)), []).append(idx)

    offsets = [(0.0, 0.0) for _ in points]
    for group in groups.values():
        if len(group) == 1:
            continue
        radius_x = 0.006 * x_span
        radius_y = 0.010 * y_span
        for pos, idx in enumerate(group):
            angle = 2.0 * math.pi * pos / len(group)
            offsets[idx] = (radius_x * math.cos(angle), radius_y * math.sin(angle))
    return offsets


def main() -> int:
    args = parse_args()
    summary_paths = [Path(p) for p in args.summary] if args.summary else DEFAULT_SUMMARIES
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(summary_paths)
    rows = [
        row
        for row in rows
        if all(math.isfinite(fnum(row, key)) for key in ("n", "alpha", "q_mean"))
        and math.isfinite(margin_fraction(row))
        and math.isfinite(threshold_normalized_error(row, args.window_size))
    ]
    if not rows:
        raise RuntimeError("no usable calibration rows found")

    plt = import_matplotlib()
    n_values = sorted({int(round(fnum(row, "n"))) for row in rows})
    epsilon_values = sorted({margin_fraction(row) for row in rows})
    colors = plt.cm.viridis([i / max(1, len(epsilon_values) - 1) for i in range(len(epsilon_values))])
    color_by_epsilon = {epsilon: colors[i] for i, epsilon in enumerate(epsilon_values)}

    fig, axes = plt.subplots(
        1,
        len(n_values),
        figsize=(3.6 * len(n_values), 3.5),
        sharey=True,
    )
    if len(n_values) == 1:
        axes = [axes]

    panel_labels = ["(a)", "(b)", "(c)", "(d)"]
    for idx, (ax, n_value) in enumerate(zip(axes, n_values)):
        subset = [row for row in rows if int(round(fnum(row, "n"))) == n_value]
        points = [(fnum(row, "q_mean") / n_value, threshold_normalized_error(row, args.window_size)) for row in subset]
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        x_span = max(x_values) - min(x_values) or max(abs(x_values[0]), 1.0)
        y_span = max(y_values) - min(y_values) or max(abs(y_values[0]), 1.0)
        offsets = duplicate_offsets(points, x_span, y_span)

        for row, (dx, dy) in zip(subset, offsets):
            epsilon = margin_fraction(row)
            alpha = fnum(row, "alpha")
            x = fnum(row, "q_mean") / n_value + dx
            y = threshold_normalized_error(row, args.window_size) + dy
            ax.scatter(
                x,
                y,
                marker=marker_for_alpha(alpha),
                s=18,
                color=color_by_epsilon[epsilon],
                edgecolor="black",
                linewidth=0.35,
                alpha=0.86,
                zorder=3,
            )
            promoted = promoted_pair(epsilon, alpha)
            if promoted:
                ax.scatter(
                    x,
                    y,
                    marker="o",
                    s=66,
                    facecolors="none",
                    edgecolor="#111111" if promoted == "quality" else "#777777",
                    linewidth=1.15,
                    zorder=4,
                )
        ax.set_title(f"{panel_labels[idx]} $n={n_value}$", fontsize=10)
        ax.set_xlabel("Mean deployed capacity $q/n$")
        ax.grid(True, alpha=0.3)
        if idx > 0:
            ax.tick_params(axis="y", labelleft=False)
    axes[0].set_ylabel(Y_LABEL)
    axes[0].yaxis.set_label_coords(-0.22, 0.46)

    epsilon_handles = [
        axes[0].scatter([], [], s=18, color=color_by_epsilon[epsilon], edgecolor="black", linewidth=0.35, label=fr"$\varepsilon_M={fmt_epsilon(epsilon)}$")
        for epsilon in epsilon_values
    ]
    alpha_values = sorted({fnum(row, "alpha") for row in rows})
    alpha_handles = [
        axes[0].scatter([], [], s=18, marker=marker_for_alpha(alpha), color="white", edgecolor="black", linewidth=0.35, label=fr"$\alpha={alpha:g}$")
        for alpha in alpha_values
    ]
    promoted_handles = [
        axes[0].scatter(
            [],
            [],
            s=66,
            marker="o",
            facecolors="none",
            edgecolor="#111111",
            linewidth=1.15,
            label=r"quality-oriented $(0.10,0.9)$",
        ),
        axes[0].scatter(
            [],
            [],
            s=66,
            marker="o",
            facecolors="none",
            edgecolor="#777777",
            linewidth=1.15,
            label=r"memory-oriented $(0.15,0.95)$",
        ),
    ]
    fig.legend(
        handles=epsilon_handles + alpha_handles + promoted_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=5,
        fontsize=7.5,
        frameon=False,
        borderaxespad=0.0,
        columnspacing=0.9,
        handletextpad=0.35,
    )
    fig.subplots_adjust(
        left=0.125,
        right=0.985,
        top=0.94,
        bottom=0.30,
        wspace=0.24,
    )
    fig.savefig(out_dir / "margin_service_topology_frontier.pdf", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out_dir / "margin_service_topology_frontier.png", dpi=220, bbox_inches="tight", pad_inches=0.08)
    print(f"plots written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
