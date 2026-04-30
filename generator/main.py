import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil

from data.generate_data import prepare_and_store_data


CONFIG_DIR = Path(__file__).resolve().parent / "config"
DEFAULT_RUN_CONFIG = CONFIG_DIR / "runs" / "default.json"


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _slug(value):
    text = str(value)
    out = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in text).strip("_")
    return out or "run"


def _unique_run_dir(output_root, run_name):
    base = output_root / "runs" / run_name
    path = base
    suffix = 1
    while path.exists():
        path = output_root / "runs" / f"{run_name}_{suffix:02d}"
        suffix += 1
    return path


def _resolve_path(path, base_dir):
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


def _load_component(value, base_dir):
    if isinstance(value, dict):
        return value, None
    if value is None:
        return {}, None
    path = _resolve_path(value, base_dir)
    return _load_json(path), path


def _load_run_configuration(args):
    """Load either the new separated config set or a legacy combined params file."""
    if args.params:
        params_path = Path(args.params).resolve()
        params = _load_json(params_path)
        generation_config = {k: v for k, v in params.items() if k != "partitioning"}
        partitioning_config = params.get("partitioning", {})
        scenario_path = (
            Path(args.scenario).resolve()
            if args.scenario
            else (CONFIG_DIR / "scenarios" / "default.json").resolve()
        )
        scenario = [] if args.real_counts else _load_json(scenario_path)
        return {
            "mode": "legacy_params",
            "run_config_path": None,
            "generation_config": generation_config,
            "generation_path": params_path,
            "partitioning_config": partitioning_config,
            "partitioning_path": None,
            "scenario": scenario,
            "scenario_path": None if args.real_counts else scenario_path,
            "legacy_params_path": params_path,
        }

    run_config_path = Path(args.run_config).resolve()
    run_config = _load_json(run_config_path)
    run_base = run_config_path.parent

    generation_ref = args.generation or run_config.get("generation")
    partitioning_ref = args.partitioning or run_config.get("partitioning")
    scenario_ref = args.scenario or run_config.get("scenario")

    generation_base = Path.cwd() if args.generation else run_base
    partitioning_base = Path.cwd() if args.partitioning else run_base
    scenario_base = Path.cwd() if args.scenario else run_base

    generation_config, generation_path = _load_component(generation_ref, generation_base)
    partitioning_config, partitioning_path = _load_component(partitioning_ref, partitioning_base)
    scenario, scenario_path = ([], None)
    if not args.real_counts:
        scenario, scenario_path = _load_component(scenario_ref, scenario_base)

    return {
        "mode": "composed",
        "run_config_path": run_config_path,
        "generation_config": generation_config,
        "generation_path": generation_path,
        "partitioning_config": partitioning_config,
        "partitioning_path": partitioning_path,
        "scenario": scenario,
        "scenario_path": scenario_path,
        "legacy_params_path": None,
    }


def _copy_or_write_config(config_dir, name, data, source_path):
    """Snapshot the exact config used for a run, including inline overrides."""
    dest = config_dir / name
    if source_path:
        shutil.copy2(source_path, dest)
    else:
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


parser = argparse.ArgumentParser(description="Generate streaming data from separated generator configs.")
parser.add_argument(
    "--run-config",
    default=str(DEFAULT_RUN_CONFIG),
    help="Run preset that composes generation, scenario, and partitioning configs.",
)
parser.add_argument(
    "--generation",
    default=None,
    help="Override generation/output config path for a composed run.",
)
parser.add_argument(
    "--partitioning",
    default=None,
    help="Override partitioning config path for a composed run.",
)
parser.add_argument(
    "--scenario",
    default=None,
    help="Override scenario config path for generated global counts.",
)
parser.add_argument(
    "--params",
    default=None,
    help="Legacy combined params JSON. Prefer --run-config for new runs.",
)
parser.add_argument(
    "--real-counts",
    default=None,
    help="Path to real per-window global counts (.jsonl or .csv).",
)
parser.add_argument(
    "--save-windows-separately",
    action="store_true",
    help="When expanding streams, write one compressed JSON file per window.",
)
parser.add_argument(
    "--run-name",
    default=None,
    help="Optional output run name. Defaults to a timestamp plus config/scenario/policy name.",
)
args = parser.parse_args()

resolved = _load_run_configuration(args)
generation = resolved["generation_config"]
partitioning_config = resolved["partitioning_config"]
scenario = resolved["scenario"]

seed = generation["seed"]
window_size = generation["window_size"]
num_keys = generation["num_keys"]
n = generation["n"]
m = generation["m"]

output_root = Path(generation["output_dir"])
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
generation_stem = _slug(resolved["generation_path"].stem if resolved["generation_path"] else "generation")
scenario_source = resolved["scenario_path"] if args.real_counts is None else Path(args.real_counts)
scenario_stem = _slug(scenario_source.stem if scenario_source else "real_counts")
policy_stem = _slug(partitioning_config.get("policy", "partitioning"))
run_name = _slug(args.run_name) if args.run_name else f"{timestamp}_{generation_stem}_{scenario_stem}_{policy_stem}"
run_dir = _unique_run_dir(output_root, run_name)
streams_dir = run_dir / "streams"
plots_dir = run_dir / "plots"
config_dir = run_dir / "config"
streams_dir.mkdir(parents=True, exist_ok=False)
plots_dir.mkdir(parents=True, exist_ok=True)
config_dir.mkdir(parents=True, exist_ok=True)

stream_file = str(streams_dir / generation["stream_file"])
summary_file = str(streams_dir / generation["summary_file"])
stream_json_file = str(streams_dir / generation["stream_json_file"])

plot_distr = generation["plot_distr"]
plot_data_config = (str(plots_dir), generation["plot_format"])
hot_key_drift_pct = generation.get("hot_key_drift_pct", 0)
max_rel_freq_delta_pct = generation.get("max_rel_freq_delta_pct", 20)
new_key_pct = generation.get("new_key_pct", 10)
newborn_hot_birth_prob = generation.get("newborn_hot_birth_prob", 0.05)
max_hot_newborn_frac = generation.get("max_hot_newborn_frac", 0.10)
expand_streams = generation.get("expand_streams")
save_windows_separately = generation.get("save_windows_separately", False) or args.save_windows_separately

if resolved["run_config_path"]:
    shutil.copy2(resolved["run_config_path"], config_dir / "run.json")
_copy_or_write_config(config_dir, "generation.json", generation, resolved["generation_path"])
_copy_or_write_config(config_dir, "partitioning.json", partitioning_config, resolved["partitioning_path"])
if args.real_counts is None:
    _copy_or_write_config(config_dir, "scenario.json", scenario, resolved["scenario_path"])

with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "run_name": run_dir.name,
            "config_mode": resolved["mode"],
            "run_config": None if resolved["run_config_path"] is None else str(resolved["run_config_path"]),
            "generation": None if resolved["generation_path"] is None else str(resolved["generation_path"]),
            "partitioning": None if resolved["partitioning_path"] is None else str(resolved["partitioning_path"]),
            "scenario": None if resolved["scenario_path"] is None else str(resolved["scenario_path"]),
            "legacy_params": None if resolved["legacy_params_path"] is None else str(resolved["legacy_params_path"]),
            "real_counts": None if args.real_counts is None else str(Path(args.real_counts).resolve()),
            "streams_dir": str(streams_dir),
            "plots_dir": str(plots_dir),
            "stream_file": stream_file,
            "summary_file": summary_file,
            "stream_json_file": stream_json_file,
        },
        f,
        indent=2,
    )

print(f"Generator run directory: {run_dir}")

prepare_and_store_data(
    seed,
    window_size,
    num_keys,
    n,
    m,
    scenario,
    args.real_counts,
    stream_file,
    summary_file,
    stream_json_file,
    plot_distr,
    plot_data_config,
    partitioning_config,
    hot_key_drift_pct,
    max_rel_freq_delta_pct,
    new_key_pct,
    newborn_hot_birth_prob,
    max_hot_newborn_frac,
    expand_streams,
    save_windows_separately,
)
