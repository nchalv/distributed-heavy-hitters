#!/usr/bin/env python3
"""Run HeavyHeads experiment sweeps and generate plots.

The script expects one or more already-partitioned input streams in the
hh_bench JSON/JSON.GZ format. Pass datasets as PATH or LABEL:PATH.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_ROOT = PROJECT_ROOT / "evaluator"
DEFAULT_BIN = EVALUATOR_ROOT / "build" / "hh_bench"

DEFAULTS = {
    "r_m": "0.03",
    "alpha_req": "0.98",
    "rho": "0.50",
    "rho_a": "0.50",
    "calib_window": "5",
    "delta_m": "0.10",
    "trend_h": "3",
    "lambda_a": "0.50",
    "lambda_g": "0.50",
    "ss_eps": "per-item",
}

SWEEPS = {
    "r_m": ["0.01", "0.02", "0.03", "0.05", "0.08"],
    "alpha_req": ["0.90", "0.95", "0.98", "0.995"],
    "rho": ["0.00", "0.25", "0.50", "0.75", "0.90"],
    "rho_a": ["0.00", "0.25", "0.50", "0.75", "0.90"],
    "calib_window": ["2", "3", "5", "8", "12"],
    "delta_m": ["0.05", "0.10", "0.20", "0.30"],
    "trend_h": ["2", "3", "5", "8"],
    "lambda_a": ["0.00", "0.25", "0.50", "1.00", "1.50"],
    "lambda_g": ["0.00", "0.25", "0.50", "1.00", "1.50"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run hh_bench sweeps, collect CSVs, and create plots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", action="append", required=True,
                   help="Dataset as PATH or LABEL:PATH. Repeat for hash/stress/etc.")
    p.add_argument("--out", default=str(PROJECT_ROOT / "experiments" / "runs"),
                   help="Output root directory.")
    p.add_argument("--bin", default=str(DEFAULT_BIN),
                   help="Path to hh_bench executable.")
    p.add_argument("--build", action="store_true",
                   help="Build hh_bench before running experiments.")
    p.add_argument("--m", type=int, required=True, help="Number of partitions.")
    p.add_argument("--n-param", type=int, required=True, help="Heavy-hitter denominator n.")
    p.add_argument("--mem-kib", default="32,64,128,256",
                   help="Comma-separated per-partition memory budgets.")
    p.add_argument("--topk", type=int, default=0, help="Optional top-k overlap.")
    p.add_argument("--profile", choices=["quick", "full"], default="quick",
                   help="quick runs one-at-a-time sweeps; full also runs pairwise lambda grid.")
    p.add_argument("--skip-run", action="store_true",
                   help="Only plot existing CSV files in --run-dir.")
    p.add_argument("--run-dir", default="",
                   help="Existing run directory for --skip-run, or explicit output run directory.")
    p.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return p.parse_args()


def sanitize(s: str) -> str:
    out = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in s).strip("_")
    return out or "run"


def dataset_label(spec: str) -> Tuple[str, Path]:
    if ":" in spec and not Path(spec).exists():
        label, path = spec.split(":", 1)
        return sanitize(label), Path(path).expanduser().resolve()
    path = Path(spec).expanduser().resolve()
    return sanitize(path.stem), path


def run_cmd(cmd: Sequence[str], log_path: Path, dry_run: bool = False) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = " ".join(shlex.quote(x) for x in cmd)
    print(printable)
    if dry_run:
        log_path.write_text(printable + "\n", encoding="utf-8")
        return
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + printable + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with exit code {proc.returncode}: {printable}")


def method_string(kind: str, params: Dict[str, str]) -> str:
    diff = (
        "ss[policy=difficulty "
        f"alpha-req={params['alpha_req']} "
        f"r-m={params['r_m']} "
        f"rho={params['rho']} "
        f"rho-a={params['rho_a']} "
        f"calib-window={params['calib_window']} "
        f"delta-m={params['delta_m']} "
        f"trend-h={params['trend_h']} "
        f"lambda-a={params['lambda_a']} "
        f"lambda-g={params['lambda_g']} "
        "diff-mode=predictive "
        f"ss-eps={params['ss_eps']}]"
    )
    if kind == "difficulty":
        return (
            "oracle,ss[policy=static q=n],ss[policy=static q=2n],"
            f"{diff},"
            "ss[policy=difficulty diff-mode=reactive-req "
            f"alpha-req={params['alpha_req']} r-m={params['r_m']} ss-eps={params['ss_eps']}],"
            "ss[policy=difficulty diff-mode=reactive-eff "
            f"alpha-req={params['alpha_req']} r-m={params['r_m']} ss-eps={params['ss_eps']}]"
        )
    if kind == "certification":
        return f"oracle,ss[policy=static q=n],ss[policy=static q=2n],ss[policy=static q=4n],{diff}"
    if kind == "heavylocker":
        return f"oracle,ss[policy=static q=n],ss[policy=static q=2n],hl,{diff}"
    if kind == "hybrid":
        hyb = (
            "hybrid[hyb-head=candidate hyb-tail=difficulty "
            f"alpha-req={params['alpha_req']} r-m={params['r_m']} "
            f"rho={params['rho']} rho-a={params['rho_a']} "
            f"calib-window={params['calib_window']} delta-m={params['delta_m']} "
            f"trend-h={params['trend_h']} lambda-a={params['lambda_a']} "
            f"lambda-g={params['lambda_g']} diff-mode=predictive ss-eps={params['ss_eps']}]"
        )
        return f"oracle,{diff},{hyb}"
    raise ValueError(kind)


def planned_runs(datasets: List[Tuple[str, Path]], mems: List[int], profile: str) -> List[dict]:
    runs = []
    for dlabel, dpath in datasets:
        for mem in mems:
            base = dict(DEFAULTS)
            for group, kind in [
                ("default", "difficulty"),
                ("certification", "certification"),
                ("heavylocker", "heavylocker"),
                ("hybrid", "hybrid"),
            ]:
                runs.append({
                    "dataset": dlabel, "path": dpath, "mem": mem,
                    "group": group, "name": f"{group}_mem{mem}", "params": base,
                    "methods": method_string(kind, base),
                })
            for pname, values in SWEEPS.items():
                for value in values:
                    params = dict(DEFAULTS)
                    params[pname] = value
                    runs.append({
                        "dataset": dlabel, "path": dpath, "mem": mem,
                        "group": f"sweep_{pname}",
                        "name": f"{pname}_{sanitize(value)}_mem{mem}",
                        "sweep_param": pname,
                        "sweep_value": value,
                        "params": params,
                        "methods": method_string("difficulty", params),
                    })
            if profile == "full":
                for la in SWEEPS["lambda_a"]:
                    for lg in SWEEPS["lambda_g"]:
                        params = dict(DEFAULTS)
                        params["lambda_a"] = la
                        params["lambda_g"] = lg
                        runs.append({
                            "dataset": dlabel, "path": dpath, "mem": mem,
                            "group": "grid_lambda",
                            "name": f"lambdaA_{sanitize(la)}_lambdaG_{sanitize(lg)}_mem{mem}",
                            "sweep_param": "lambda_grid",
                            "sweep_value": f"{la}:{lg}",
                            "params": params,
                            "methods": method_string("difficulty", params),
                        })
    return runs


def execute_runs(args: argparse.Namespace, run_dir: Path, runs: List[dict]) -> None:
    if args.build:
        run_cmd(["cmake", "--build", str(EVALUATOR_ROOT / "build"), "--target", "hh_bench"],
                run_dir / "logs" / "build.log", args.dry_run)
    bin_path = Path(args.bin).expanduser().resolve()
    for r in runs:
        csv_path = run_dir / "csv" / r["dataset"] / r["group"] / f"{r['name']}.csv"
        log_path = run_dir / "logs" / r["dataset"] / r["group"] / f"{r['name']}.log"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(bin_path), str(r["path"]), str(args.m), str(args.n_param),
            str(r["mem"]), r["methods"], "--csv-out", str(csv_path),
        ]
        if args.topk > 0:
            cmd += ["--topk", str(args.topk)]
        run_cmd(cmd, log_path, args.dry_run)

    with (run_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "dataset", "group", "name", "mem", "sweep_param", "sweep_value", "csv", "methods",
        ])
        w.writeheader()
        for r in runs:
            rel_csv = Path("csv") / r["dataset"] / r["group"] / f"{r['name']}.csv"
            w.writerow({
                "dataset": r["dataset"], "group": r["group"], "name": r["name"], "mem": r["mem"],
                "sweep_param": r.get("sweep_param", ""), "sweep_value": r.get("sweep_value", ""),
                "csv": str(rel_csv), "methods": r["methods"],
            })


def read_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(row: dict, key: str, default: float = math.nan) -> float:
    try:
        value = row.get(key, "")
        return default if value == "" else float(value)
    except ValueError:
        return default


def mean(xs: Iterable[float]) -> float:
    vals = [x for x in xs if math.isfinite(x)]
    return sum(vals) / len(vals) if vals else math.nan


def stdev(xs: Iterable[float]) -> float:
    vals = [x for x in xs if math.isfinite(x)]
    if len(vals) < 2:
        return 0.0 if vals else math.nan
    m = sum(vals) / len(vals)
    return math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1))


def summarize_csv(path: Path, method_contains: Optional[str] = None) -> Dict[str, float]:
    rows = read_rows(path)
    if method_contains:
        rows = [r for r in rows if method_contains in r.get("method", "")]
    if not rows:
        return {}
    q_vals = [fnum(r, "q_current") for r in rows]
    return {
        "hh_f1": mean(fnum(r, "hh_f1") for r in rows),
        "hh_recall": mean(fnum(r, "hh_recall") for r in rows),
        "candidate_hh_recall": mean(fnum(r, "candidate_hh_recall") for r in rows),
        "cert_pos_precision": mean(fnum(r, "cert_pos_precision") for r in rows),
        "ambiguous_mass": mean(fnum(r, "ambiguous_mass") for r in rows),
        "ambiguous_count": mean(fnum(r, "ambiguous_count") for r in rows),
        "interval_width_avg": mean(fnum(r, "interval_width_avg") for r in rows),
        "interval_width_amb_avg": mean(fnum(r, "interval_width_amb_avg") for r in rows),
        "miss_req": mean(fnum(r, "miss_req") for r in rows),
        "miss_eff": mean(fnum(r, "miss_eff") for r in rows),
        "over_req": mean(fnum(r, "over_req") for r in rows),
        "over_eff": mean(fnum(r, "over_eff") for r in rows),
        "mem_worker_total_kib": mean(fnum(r, "mem_worker_total_kib") for r in rows),
        "mem_volatility": stdev(q_vals),
        "update_mops": mean(fnum(r, "update_mops") for r in rows),
        "reduce_ms": mean(fnum(r, "reduce_ms") for r in rows),
        "control_ms": mean(fnum(r, "control_ms") for r in rows),
    }


def import_matplotlib():
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/hh_matplotlib")
        Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as exc:
        print(f"plotting skipped: matplotlib unavailable ({exc})", file=sys.stderr)
        return None


def load_manifest(run_dir: Path) -> List[dict]:
    with (run_dir / "manifest.csv").open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_experiments(run_dir: Path) -> None:
    plt = import_matplotlib()
    if plt is None:
        return
    manifest = load_manifest(run_dir)
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    make_sweep_plots(plt, run_dir, manifest, plot_dir)
    make_memory_quality_plot(plt, run_dir, manifest, plot_dir)
    make_trace_plots(plt, run_dir, manifest, plot_dir)
    make_certification_plots(plt, run_dir, manifest, plot_dir)
    make_timing_plot(plt, run_dir, manifest, plot_dir)


def make_sweep_plots(plt, run_dir: Path, manifest: List[dict], plot_dir: Path) -> None:
    metrics = [
        ("miss_eff", "Effective-target miss rate"),
        ("over_eff", "Overprovisioning q/qEff"),
        ("mem_worker_total_kib", "Worker memory KiB"),
        ("ambiguous_mass", "Ambiguous mass"),
        ("interval_width_amb_avg", "Ambiguous interval width"),
        ("hh_f1", "HH F1"),
    ]
    for param in SWEEPS:
        entries = [m for m in manifest if m.get("group") == f"sweep_{param}"]
        points_by_series: Dict[Tuple[str, str], List[Tuple[float, Dict[str, float]]]] = {}
        for e in entries:
            stats = summarize_csv(run_dir / e["csv"], "mode=predictive")
            if not stats:
                continue
            try:
                x = float(e["sweep_value"])
            except ValueError:
                continue
            points_by_series.setdefault((e["dataset"], e["mem"]), []).append((x, stats))
        for metric, ylabel in metrics:
            fig, ax = plt.subplots(figsize=(7, 4))
            for (dataset, mem), points in sorted(points_by_series.items()):
                points.sort(key=lambda p: p[0])
                ax.plot([p[0] for p in points], [p[1].get(metric, math.nan) for p in points],
                        marker="o", linewidth=1.8, label=f"{dataset}, {mem} KiB")
            ax.set_xlabel(param)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(plot_dir / f"sweep_{param}_{metric}.pdf")
            fig.savefig(plot_dir / f"sweep_{param}_{metric}.png", dpi=180)
            plt.close(fig)


def make_memory_quality_plot(plt, run_dir: Path, manifest: List[dict], plot_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for e in [m for m in manifest if m.get("group") in ("default", "heavylocker", "hybrid")]:
        rows = read_rows(run_dir / e["csv"])
        methods = sorted({r["method"] for r in rows if r.get("method_type") != "oracle"})
        for method in methods:
            stats = summarize_csv(run_dir / e["csv"], method)
            if not stats:
                continue
            ax.scatter(stats["mem_worker_total_kib"], stats["hh_f1"], s=40)
            ax.annotate(f"{e['dataset']} {method} {e['mem']}K",
                        (stats["mem_worker_total_kib"], stats["hh_f1"]),
                        fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Mean worker memory per partition (KiB)")
    ax.set_ylabel("HH F1")
    ax.set_title("Memory-quality tradeoff")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "memory_quality_tradeoff.pdf")
    fig.savefig(plot_dir / "memory_quality_tradeoff.png", dpi=180)
    plt.close(fig)


def make_trace_plots(plt, run_dir: Path, manifest: List[dict], plot_dir: Path) -> None:
    for e in [m for m in manifest if m.get("group") == "default"]:
        rows = [r for r in read_rows(run_dir / e["csv"]) if "mode=predictive" in r.get("method", "")]
        if not rows:
            continue
        xs = [int(r["window"]) for r in rows]
        fig, ax1 = plt.subplots(figsize=(8, 4))
        for key, label in [
            ("q_current", "q deployed"), ("q_req", "qReq"),
            ("q_eff_replay", "qEff"), ("q_eff_pred", "qEff pred"),
        ]:
            ax1.plot(xs, [fnum(r, key) for r in rows], linewidth=1.5, label=label)
        ax1.set_xlabel("Window ID")
        ax1.set_ylabel("Capacity")
        ax2 = ax1.twinx()
        ax2.plot(xs, [fnum(r, "ambiguous_mass") for r in rows],
                 color="black", linestyle="--", linewidth=1.2, label="ambiguous mass")
        ax2.set_ylabel("Ambiguous mass")
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, fontsize=8)
        ax1.grid(True, alpha=0.3)
        fig.tight_layout()
        stem = f"trace_{sanitize(e['dataset'])}_mem{e['mem']}"
        fig.savefig(plot_dir / f"{stem}.pdf")
        fig.savefig(plot_dir / f"{stem}.png", dpi=180)
        plt.close(fig)


def make_certification_plots(plt, run_dir: Path, manifest: List[dict], plot_dir: Path) -> None:
    labels, cert_pos, amb, cert_neg, widths = [], [], [], [], []
    for e in [m for m in manifest if m.get("group") == "certification"]:
        all_rows = [r for r in read_rows(run_dir / e["csv"]) if r.get("method_type") == "ss"]
        for method in sorted({r.get("method", "") for r in all_rows}):
            rows = [r for r in all_rows if r.get("method") == method]
            if not rows:
                continue
            short_method = method.replace("ss[", "").replace("]", "")
            labels.append(f"{e['dataset']} {e['mem']}K {short_method}")
            cert_pos.append(mean(fnum(r, "cert_pos_mass") for r in rows))
            amb.append(mean(fnum(r, "ambiguous_mass") for r in rows))
            cert_neg.append(mean(fnum(r, "cert_neg_mass") for r in rows))
            widths.append([fnum(r, "interval_width_amb_avg") for r in rows
                           if math.isfinite(fnum(r, "interval_width_amb_avg"))])
    if labels:
        fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.5), 4))
        xs = list(range(len(labels)))
        ax.bar(xs, cert_pos, label="certified positive mass")
        ax.bar(xs, amb, bottom=cert_pos, label="ambiguous mass")
        ax.bar(xs, cert_neg, bottom=[cert_pos[i] + amb[i] for i in xs],
               label="certified negative mass")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Reported candidate mass")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_dir / "certification_mass_stack.pdf")
        fig.savefig(plot_dir / "certification_mass_stack.png", dpi=180)
        plt.close(fig)
    if widths:
        fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.5), 4))
        ax.boxplot(widths, labels=labels, showfliers=False)
        ax.set_ylabel("Ambiguous interval width")
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / "certification_interval_boxplot.pdf")
        fig.savefig(plot_dir / "certification_interval_boxplot.png", dpi=180)
        plt.close(fig)


def make_timing_plot(plt, run_dir: Path, manifest: List[dict], plot_dir: Path) -> None:
    labels, update, reduce, control = [], [], [], []
    for e in [m for m in manifest if m.get("group") in ("default", "heavylocker", "hybrid")]:
        rows = [r for r in read_rows(run_dir / e["csv"]) if r.get("method_type") != "oracle"]
        if not rows:
            continue
        labels.append(f"{e['dataset']} {e['group']} {e['mem']}K")
        update.append(mean(fnum(r, "update_mops") for r in rows))
        reduce.append(mean(fnum(r, "reduce_ms") for r in rows))
        control.append(mean(fnum(r, "control_ms") for r in rows))
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.55), 4))
    xs = list(range(len(labels)))
    ax.plot(xs, update, marker="o", label="update Mops/s")
    ax.set_ylabel("Update Mops/s")
    ax2 = ax.twinx()
    ax2.plot(xs, reduce, marker="s", color="tab:orange", label="reduce ms")
    ax2.plot(xs, control, marker="^", color="tab:green", label="control ms")
    ax2.set_ylabel("End-of-window ms")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    lines, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels1 + labels2, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "timing_overheads.pdf")
    fig.savefig(plot_dir / "timing_overheads.png", dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    datasets = [dataset_label(s) for s in args.dataset]
    for label, path in datasets:
        if not path.exists():
            raise FileNotFoundError(f"dataset {label} not found: {path}")
    mems = [int(x) for x in args.mem_kib.split(",") if x.strip()]
    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
    else:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(args.out).expanduser().resolve() / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_run:
        execute_runs(args, run_dir, planned_runs(datasets, mems, args.profile))
    if not args.dry_run:
        plot_experiments(run_dir)
    print(f"results: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
