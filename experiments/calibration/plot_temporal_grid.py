#!/usr/bin/env python3
"""Plot aggregate and stepwise temporal-controller ablation results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re


DEFAULT_SUMMARY = Path("experiments/calibration/temporal_controller_ablation/summary.csv")
DEFAULT_OUT = Path("experiments/calibration/temporal_controller_ablation/plots")
DEFAULT_CSV_ROOT = Path("experiments/calibration/temporal_controller_ablation/csv")
DEFAULT_CONFIG_ROOT = Path("generator/config/partitioning")
DEFAULT_WINDOW_SIZE = 60000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot temporal-controller ablation results."
    )
    parser.add_argument(
        "--summary",
        action="append",
        help="Summary CSV produced by summarize_temporal_grid.py. Repeat to combine experiments.",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output plot directory.")
    parser.add_argument(
        "--csv-root",
        action="append",
        help="Directory containing per-window temporal CSVs. Repeat to combine experiments.",
    )
    parser.add_argument("--config-root", default=str(DEFAULT_CONFIG_ROOT), help="Directory containing partitioning schedules.")
    parser.add_argument("--window-size", type=float, default=DEFAULT_WINDOW_SIZE, help="Fallback N for legacy summaries.")
    parser.add_argument(
        "--epsilon-m",
        type=float,
        default=0.10,
        help="Threshold-normalized margin budget shown on the error plot.",
    )
    parser.add_argument(
        "--dataset-group",
        choices=("all", "scheduled", "stationary"),
        default="all",
        help="Restrict plots to scheduled temporal profiles or stationary controls.",
    )
    parser.add_argument(
        "--output-stem",
        default="",
        help="Filename stem for the aggregate plot. Defaults to a group-specific name.",
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

    return plt, Patch


def threshold_normalized_error(row: dict[str, str], fallback_window_size: float) -> float:
    n = fnum(row, "n")
    aae = fnum(row, "aae")
    window_size = fnum(row, "N_global")
    if not math.isfinite(window_size):
        window_size = fallback_window_size
    if not (math.isfinite(n) and math.isfinite(aae) and window_size > 0):
        return math.nan
    return n * aae / window_size


def dataset_label(dataset: str) -> str:
    labels = {
        "round_robin_n200": "Stationary round robin",
        "milp_certificate_adversary_n200": "Stationary MILP stress",
        "temporal_guard_step_schedule_n200": "Sharp step",
        "temporal_guard_ramp_schedule_n200": "Smooth ramp",
        "temporal_guard_burst_schedule_n200": "Short bursts",
        "temporal_guard_oscillation_schedule_n200": "Block oscillation",
        "temporal_pressure_step_locality_n200": "Step",
        "temporal_pressure_ramp_locality_n200": "Ramp",
        "temporal_pressure_burst_locality_n200": "Burst",
        "temporal_calibration_step_locality_n200": "Step",
        "temporal_calibration_ramp_locality_n200": "Ramp",
        "temporal_calibration_burst_locality_n200": "Burst",
    }
    return labels.get(dataset, dataset.replace("_", " "))


def infer_n(dataset: str, default_n: int = 200) -> int:
    match = re.search(r"(?:^|_)n([0-9]+)(?:_|$)", dataset)
    return int(match.group(1)) if match else default_n


def load_schedule(config_root: Path, dataset: str) -> list[dict[str, object]]:
    path = config_root / f"{dataset}.json"
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as fh:
        config = json.load(fh)
    return config.get("policy_schedule", [])


def stationary_partitioning_policy(dataset: str) -> str | None:
    policies = {
        "round_robin_n200": "round_robin",
        "milp_certificate_adversary_n200": "milp_certificate_adversary",
    }
    return policies.get(dataset)


def plot_stepwise(
    plt,
    Patch,
    *,
    csv_roots: list[Path],
    config_root: Path,
    out_dir: Path,
    datasets: list[str],
    fallback_window_size: float,
    epsilon_m: float,
    output_suffix: str,
) -> None:
    mode_specs = {
        "probing": (
            "Bracket probing",
            "#B85C38",
        ),
        "residual_guarded_probing": (
            "Residual-guarded bracket probing",
            "#6F4E7C",
        ),
        "comfort_guided_probing": (
            "Guarded margin-comfort probing",
            "#287271",
        ),
        "pressure_gated_comfort_probing": (
            r"Pressure gate $\rho\leq0.5$",
            "#3264A8",
        ),
    }
    phase_specs = {
        "round_robin": ("Round robin", "#E8ECEF"),
        "milp_certificate_adversary_intermediate": ("Intermediate MILP", "#F4D58D"),
        "milp_certificate_adversary": ("MILP", "#E89B8E"),
    }

    dataset_rows = {}
    for dataset in datasets:
        n_value = infer_n(dataset)
        mode_rows = {}
        for mode in mode_specs:
            filename = f"{dataset}_{mode}.csv"
            path = next(
                (root / filename for root in csv_roots if (root / filename).is_file()),
                None,
            )
            if path is None:
                continue
            if not path.is_file():
                continue
            rows = [
                row
                for row in load_rows(path)
                if row.get("method_type") == "ss"
                and "policy=difficulty" in row.get("method", "")
            ]
            if rows:
                mode_rows[mode] = rows
        if mode_rows:
            dataset_rows[dataset] = (n_value, mode_rows)

    if not dataset_rows:
        return

    phase_handles = list(
        Patch(facecolor=color, alpha=0.42, edgecolor="none", label=label)
        for label, color in phase_specs.values()
    )

    metric_specs = [
        (
            "error",
            "Window-by-Window Threshold-Normalized Error",
            "Threshold-normalized error",
            lambda row, n_value: n_value
            * fnum(row, "aae")
            / (
                fnum(row, "N_global")
                if math.isfinite(fnum(row, "N_global"))
                else fallback_window_size
            ),
        ),
        (
            "capacity",
            r"Window-by-Window Deployed Capacity $q/n$",
            r"Deployed capacity $q/n$",
            lambda row, n_value: fnum(row, "q_current") / n_value,
        ),
    ]

    for stem, figure_title, y_label, value_fn in metric_specs:
        compact_header = output_suffix == "_scheduled"
        header_extra = 0.8 if compact_header else 1.4
        legend_y = 0.925 if compact_header else 0.91
        axes_top = 0.84 if compact_header else 0.76
        fig, axes = plt.subplots(
            len(datasets),
            len(mode_specs),
            figsize=(19.2, 2.0 * len(datasets) + header_extra),
            sharey="row",
            squeeze=False,
        )

        for panel_idx, dataset in enumerate(datasets):
            if dataset not in dataset_rows:
                for column_idx in range(len(mode_specs)):
                    axes[panel_idx, column_idx].set_visible(False)
                continue
            n_value, mode_rows = dataset_rows[dataset]

            for column_idx, mode in enumerate(mode_specs):
                ax = axes[panel_idx, column_idx]
                rows = mode_rows.get(mode, [])
                phase_start = 0
                schedule = load_schedule(config_root, dataset)
                stationary_policy = stationary_partitioning_policy(dataset)
                if not schedule and stationary_policy and rows:
                    last_window = max(fnum(row, "window") for row in rows)
                    schedule = [
                        {
                            "duration": int(last_window) + 1,
                            "policy": stationary_policy,
                        }
                    ]
                for phase in schedule:
                    duration = int(phase["duration"])
                    policy = str(phase["policy"])
                    _, phase_color = phase_specs.get(policy, (policy, "#F2F2F2"))
                    ax.axvspan(
                        phase_start - 0.5,
                        phase_start + duration - 0.5,
                        color=phase_color,
                        alpha=0.42,
                        linewidth=0,
                        zorder=0,
                    )
                    phase_start += duration

                _, line_color = mode_specs[mode]
                ax.plot(
                    [fnum(row, "window") for row in rows],
                    [value_fn(row, n_value) for row in rows],
                    color=line_color,
                    linewidth=1.45,
                    drawstyle="steps-post",
                    zorder=3,
                )
                if stem == "capacity":
                    probe_rows = [
                        row for row in rows if fnum(row, "probe_issued") == 1.0
                    ]
                    failed_rows = [
                        row for row in rows if fnum(row, "probe_failed") == 1.0
                    ]
                    violation_rows = [
                        row
                        for row in rows
                        if fnum(row, "service_violation") == 1.0
                    ]
                    ax.scatter(
                        [fnum(row, "window") + 1 for row in probe_rows],
                        [fnum(row, "q_next") / n_value for row in probe_rows],
                        marker="v",
                        s=18,
                        facecolor="white",
                        edgecolor=line_color,
                        linewidth=0.8,
                        zorder=4,
                    )
                    ax.scatter(
                        [fnum(row, "window") for row in failed_rows],
                        [value_fn(row, n_value) for row in failed_rows],
                        marker="x",
                        s=22,
                        color="#B22222",
                        linewidth=0.9,
                        zorder=5,
                    )
                    ax.scatter(
                        [fnum(row, "window") for row in violation_rows],
                        [value_fn(row, n_value) for row in violation_rows],
                        marker="o",
                        s=12,
                        facecolor="#B22222",
                        edgecolor="none",
                        zorder=4,
                    )
                if stem == "capacity":
                    ax.axhline(1.0, color="#555555", linewidth=0.7, linestyle=":", zorder=1)
                elif stem == "error":
                    ax.axhline(
                        epsilon_m,
                        color="#B22222",
                        linewidth=1.0,
                        linestyle="--",
                        zorder=2,
                    )
                ax.grid(axis="y", alpha=0.25, linewidth=0.7)
                ax.set_axisbelow(True)
                ax.tick_params(axis="both", labelsize=8)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.margins(x=0)

            panel_label = chr(ord("a") + panel_idx)
            axes[panel_idx, 0].set_ylabel(
                f"({panel_label}) {dataset_label(dataset)}\n{y_label}",
                fontsize=8.5,
            )

        for column_idx, (_, (mode_label, _)) in enumerate(mode_specs.items()):
            axes[0, column_idx].set_title(mode_label, fontsize=10, pad=7)
            axes[-1, column_idx].set_xlabel("Window", fontsize=9)
        legend_handles = list(phase_handles)
        if stem == "error":
            legend_handles.append(
                plt.Line2D(
                    [],
                    [],
                    color="#B22222",
                    linewidth=1.0,
                    linestyle="--",
                    label=rf"$\varepsilon_M={epsilon_m:g}$",
                )
            )
        elif stem == "capacity":
            legend_handles.extend(
                [
                    plt.Line2D(
                        [],
                        [],
                        marker="v",
                        linestyle="none",
                        markerfacecolor="white",
                        markeredgecolor="#555555",
                        markersize=5,
                        label="Downward probe",
                    ),
                    plt.Line2D(
                        [],
                        [],
                        marker="x",
                        linestyle="none",
                        color="#B22222",
                        markersize=5,
                        label="Failed probe",
                    ),
                    plt.Line2D(
                        [],
                        [],
                        marker="o",
                        linestyle="none",
                        color="#B22222",
                        markersize=4,
                        label="Service violation",
                    ),
                ]
            )
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, legend_y),
            ncol=len(legend_handles),
            fontsize=8,
            frameon=False,
            columnspacing=1.4,
            handlelength=2.3,
        )
        fig.suptitle(figure_title, fontsize=11, y=0.995)
        fig.subplots_adjust(
            left=0.13,
            right=0.985,
            bottom=0.07,
            top=axes_top,
            hspace=0.30,
            wspace=0.08,
        )
        output_stem = f"temporal_controller_stepwise_{stem}{output_suffix}"
        fig.savefig(out_dir / f"{output_stem}.pdf", bbox_inches="tight", pad_inches=0.04)
        fig.savefig(out_dir / f"{output_stem}.png", dpi=220, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)

    for suffix in ("pdf", "png"):
        for stem in (
            "temporal_residual_guard_stepwise",
            "temporal_residual_guard_stepwise_error",
            "temporal_residual_guard_stepwise_capacity",
            "temporal_residual_guard",
        ):
            legacy_path = out_dir / f"{stem}.{suffix}"
            if legacy_path.exists():
                legacy_path.unlink()


def plot_memory_error_frontier(
    plt,
    *,
    out_dir: Path,
    datasets: list[str],
    by_dataset: dict[str, dict[str, dict[str, str]]],
    mode_specs: list[tuple[str, str, str, float]],
    fallback_window_size: float,
    output_suffix: str,
) -> None:
    columns = 2
    rows = math.ceil(len(datasets) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(8.8, 3.3 * rows + 1.1),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    flat_axes = list(axes.flat)

    for panel_idx, dataset in enumerate(datasets):
        ax = flat_axes[panel_idx]
        for mode, label, color, _ in mode_specs:
            row = by_dataset[dataset][mode]
            capacity = fnum(row, "q_mean") / fnum(row, "n")
            error = threshold_normalized_error(row, fallback_window_size)
            ax.scatter(
                capacity,
                error,
                s=42,
                color=color,
                edgecolor="white",
                linewidth=0.7,
                zorder=3,
                label=label,
            )

        panel_label = chr(ord("a") + panel_idx)
        ax.set_title(
            f"({panel_label}) {dataset_label(dataset)}",
            fontsize=9.5,
            pad=7,
        )
        ax.grid(alpha=0.25, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in flat_axes[len(datasets):]:
        ax.set_visible(False)

    for ax in axes[-1, :]:
        if ax.get_visible():
            ax.set_xlabel(r"Mean deployed capacity $q/n$", fontsize=9)
    for ax in axes[:, 0]:
        if ax.get_visible():
            ax.set_ylabel("Threshold-normalized error", fontsize=9)

    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor=color,
            markeredgecolor="white",
            markersize=6,
            label=label,
        )
        for _, label, color, _ in mode_specs
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=len(handles),
        fontsize=8,
        frameon=False,
        columnspacing=1.2,
        handlelength=2.0,
    )
    fig.suptitle(
        "Temporal Controller Memory-Accuracy Frontier",
        fontsize=11,
        y=0.995,
    )
    fig.subplots_adjust(
        left=0.10,
        right=0.985,
        bottom=0.10,
        top=0.82,
        hspace=0.30,
        wspace=0.16,
    )
    output_stem = f"temporal_controller_memory_error_frontier{output_suffix}"
    fig.savefig(
        out_dir / f"{output_stem}.pdf",
        bbox_inches="tight",
        pad_inches=0.04,
    )
    fig.savefig(
        out_dir / f"{output_stem}.png",
        dpi=220,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(fig)


def main() -> int:
    args = parse_args()
    summaries = [Path(path) for path in (args.summary or [DEFAULT_SUMMARY])]
    csv_roots = [Path(path) for path in (args.csv_root or [DEFAULT_CSV_ROOT])]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        row
        for summary in summaries
        for row in load_rows(summary)
        if row.get("mode", "")
        in {
            "probing",
            "residual_guarded_probing",
            "comfort_guided_probing",
            "pressure_gated_comfort_probing",
        }
        and all(
            math.isfinite(fnum(row, key))
            for key in ("n", "q_mean", "aae", "service_violation_rate")
        )
        and math.isfinite(threshold_normalized_error(row, args.window_size))
    ]
    if not rows:
        raise RuntimeError(
            "no usable temporal-controller ablation rows found in "
            + ", ".join(str(summary) for summary in summaries)
        )

    plt, Patch = import_matplotlib()
    preferred_order = [
        "round_robin_n200",
        "milp_certificate_adversary_n200",
        "temporal_guard_step_schedule_n200",
        "temporal_guard_ramp_schedule_n200",
        "temporal_guard_burst_schedule_n200",
        "temporal_guard_oscillation_schedule_n200",
    ]
    available = {row.get("dataset", "") for row in rows}
    scheduled_datasets = {
        "temporal_guard_step_schedule_n200",
        "temporal_guard_ramp_schedule_n200",
        "temporal_guard_burst_schedule_n200",
        "temporal_guard_oscillation_schedule_n200",
    }
    stationary_datasets = {
        "round_robin_n200",
        "milp_certificate_adversary_n200",
    }
    if args.dataset_group == "scheduled":
        available &= scheduled_datasets
    elif args.dataset_group == "stationary":
        available &= stationary_datasets

    datasets = [dataset for dataset in preferred_order if dataset in available]
    datasets.extend(sorted(available.difference(datasets)))

    by_dataset = {
        dataset: {
            row["mode"]: row
            for row in rows
            if row.get("dataset", "") == dataset
        }
        for dataset in datasets
    }
    datasets = [
        dataset
        for dataset in datasets
        if {
            "probing",
            "residual_guarded_probing",
            "comfort_guided_probing",
            "pressure_gated_comfort_probing",
        }.issubset(by_dataset[dataset])
    ]
    if not datasets:
        raise RuntimeError(
            "no complete four-mode temporal comparison found in "
            + ", ".join(str(summary) for summary in summaries)
        )

    metrics = [
        (
            "(a) Threshold-normalized error",
            lambda row: threshold_normalized_error(row, args.window_size),
            "{:.3f}",
        ),
        (
            "(b) Observed service-violation rate",
            lambda row: fnum(row, "service_violation_rate"),
            "{:.2f}",
        ),
        (
            r"(c) Mean deployed capacity $q/n$",
            lambda row: fnum(row, "q_mean") / fnum(row, "n"),
            "{:.2f}",
        ),
        (
            r"(d) Memory--error product",
            lambda row: (
                fnum(row, "q_mean")
                / fnum(row, "n")
                * threshold_normalized_error(row, args.window_size)
            ),
            "{:.3f}",
        ),
    ]
    mode_specs = [
        (
            "probing",
            "Bracket probing",
            "#B85C38",
            -0.27,
        ),
        (
            "residual_guarded_probing",
            "Guarded bracket probing",
            "#6F4E7C",
            -0.09,
        ),
        (
            "comfort_guided_probing",
            "Guarded margin comfort",
            "#287271",
            0.09,
        ),
        (
            "pressure_gated_comfort_probing",
            r"Pressure gate $\rho\leq0.5$",
            "#3264A8",
            0.27,
        ),
    ]

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(14.4, 4.0),
        sharey=True,
        gridspec_kw={"wspace": 0.18},
    )
    y_positions = list(range(len(datasets)))
    bar_height = 0.15

    for ax, (title, value_fn, value_format) in zip(axes, metrics):
        all_values = []
        for mode, _, color, offset in mode_specs:
            values = [value_fn(by_dataset[dataset][mode]) for dataset in datasets]
            all_values.extend(values)
            bars = ax.barh(
                [y + offset for y in y_positions],
                values,
                height=bar_height,
                color=color,
                edgecolor="white",
                linewidth=0.6,
            )
            for bar, value in zip(bars, values):
                ax.annotate(
                    value_format.format(value),
                    xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                    xytext=(4, 0),
                    textcoords="offset points",
                    va="center",
                    ha="left",
                    fontsize=7.5,
                    color="#222222",
                )

        upper = max(all_values) * 1.20 if all_values else 1.0
        ax.set_xlim(0, upper)
        ax.set_title(title, fontsize=9.5, pad=8)
        ax.grid(axis="x", alpha=0.25, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_yticks(y_positions, [dataset_label(dataset) for dataset in datasets])
    axes[0].invert_yaxis()

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color, label=label)
        for _, label, color, _ in mode_specs
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=4,
        fontsize=8.5,
        frameon=False,
    )
    group_titles = {
        "all": "Downward-Probe Selection and Residual-Guard Ablation",
        "scheduled": "Temporal-Controller Ablation under Scheduled Difficulty Changes",
        "stationary": "Temporal-Controller Ablation under Stationary Partitioning",
    }
    fig.suptitle(group_titles[args.dataset_group], fontsize=11, y=0.99)
    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.12, top=0.76, wspace=0.18)
    aggregate_stem = args.output_stem or (
        "temporal_controller_ablation"
        if args.dataset_group == "all"
        else f"temporal_controller_ablation_{args.dataset_group}"
    )
    fig.savefig(out_dir / f"{aggregate_stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_dir / f"{aggregate_stem}.png", dpi=220, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    output_suffix = "" if args.dataset_group == "all" else f"_{args.dataset_group}"
    plot_memory_error_frontier(
        plt,
        out_dir=out_dir,
        datasets=datasets,
        by_dataset=by_dataset,
        mode_specs=mode_specs,
        fallback_window_size=args.window_size,
        output_suffix=output_suffix,
    )
    plot_stepwise(
        plt,
        Patch,
        csv_roots=csv_roots,
        config_root=Path(args.config_root),
        out_dir=out_dir,
        datasets=datasets,
        fallback_window_size=args.window_size,
        epsilon_m=args.epsilon_m,
        output_suffix=output_suffix,
    )
    print(f"plots written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
