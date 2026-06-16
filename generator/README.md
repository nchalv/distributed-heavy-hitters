# Generator

Generates synthetic or trace-derived per-window key-frequency distributions,
partitions them across workers, and writes the stream format consumed by the
evaluator.

Window IDs are zero-based throughout the repository. Generated streams use
top-level window keys `0..N-1` or per-window files named
`window_000000.json.gz`, `window_000001.json.gz`, and so on. Evaluator output,
CSV rows, plotting labels, and helper-tool arguments use the same zero-based
IDs.

## Output layout

Each invocation writes to a fresh run directory under `output_dir`:

```text
generated/runs/<run-name>/
  manifest.json
  config/
    run.json
    generation.json
    partitioning.json
    scenario.json
  streams/
    stream.json.gz
    ...
  plots/
    ...
```

`--run-name` can be used to choose the run directory name. If the name already
exists, the generator appends a numeric suffix instead of overwriting it.

## Config layout

Generator configs are separated by concern:

- `config/scenarios/`: global per-window frequency distributions and transitions.
- `config/partitioning/`: key-to-partition policies and policy-specific knobs.
- `config/generation/`: run scale, output filenames, plotting, stream expansion,
  and temporal key-smoothing settings.
- `config/runs/`: small presets that compose one generation config, one scenario,
  and one partitioning config.

The default command uses `config/runs/default.json`:

```bash
python3 main.py
```

Use a different preset:

```bash
python3 main.py --run-config config/runs/round_robin.json
```

Or override one concern while keeping the rest of the preset:

```bash
python3 main.py \
  --run-config config/runs/default.json \
  --partitioning config/partitioning/round_robin.json
```

## Real Counts

The generator can also start from normalized real-world counts instead of a
synthetic scenario:

```bash
python3 main.py \
  --run-config config/runs/round_robin.json \
  --real-counts /path/to/counts.jsonl \
  --save-windows-separately
```

Supported normalized count formats:

- JSONL: one row per source window, shaped as
  `{"window": 0, "counts": {"key": 123}}`.
- CSV: long format with columns `window,src_ip,packet_count`.

When `--real-counts` is supplied, the scenario config is ignored. The source
window labels are sorted and then re-indexed to contiguous zero-based IDs in the
generated stream. By default real-count runs write partitioned counts instead of
fully expanded packet streams; use `expand_streams: true` in the generation
config if packet-expanded output is required.

## Partitioning policies

- `round_robin`: simple even split of each key across partitions, with
  remainders assigned cyclically from a stable per-key offset.
- `synthetic_mixture`: stochastic synthetic partitioning; each key uses a
  uniform, normal-shaped, or skewed allocation shape.
- `imbalance`: rarity-aware skewing; rarer keys are made more concentrated.
- `hh_hidden_optimal`: stress-test construction that tries to hide global heavy
  hitters below local thresholds while adding near-threshold distractors.
- `hh_hidden_exact`: same hidden-heavy-hitter construction without distractor
  amplification.
- `milp_certificate_adversary`: MILP stress construction that maximizes hidden
  HH mass and locally salient non-HH distractors.
- `milp_certificate_adversary_intermediate`: the same MILP construction with
  its target-key budget scaled by `milp_intermediate_scale`.
- `policy_schedule`: optional per-window phases, each with a `duration` and
  `policy`, for workloads that transition between partitioning regimes.
