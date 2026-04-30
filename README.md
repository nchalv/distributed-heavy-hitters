# Top-N Heavy Hitters Research Repository

This repository contains the paper sources and experimental code for the
distributed heavy-hitter research project. The code is organized around a
simple pipeline: prepare or synthesize per-window key frequencies, partition
those frequencies into worker streams, and evaluate heavy-hitter algorithms on
the resulting distributed stream.

## Repository Layout

- `paper/`: LaTeX sources for the research paper.
- `data_preparation/`: dataset-specific converters from raw external datasets
  into normalized per-window key-count files.
- `generator/`: synthetic and trace-derived stream generation. It consumes a
  scenario or normalized real counts, applies a partitioning policy, and writes
  evaluator-ready streams.
- `evaluator/`: C++ heavy-hitter evaluation harness, sketch implementations,
  policies, and tests.
- `experiments/`: Python orchestration for evaluator sweeps and generated run
  artifacts.
- `docs/`: supplementary notes and reference material.
- `third_party/`: external code kept for comparison, provenance, or reference.

The component READMEs contain lower-level details:

- [data_preparation/README.md](data_preparation/README.md)
- [generator/README.md](generator/README.md)
- [evaluator/README.md](evaluator/README.md)

## Data Model

The repository has one normalized interchange format for real-world datasets:

```json
{"window": 0, "counts": {"key": 123, "other_key": 45}}
```

Each JSONL row represents one logical source window before worker partitioning.
The generator sorts source windows and emits contiguous zero-based window IDs in
the final stream. This keeps downstream evaluator behavior independent of how a
raw dataset labels its windows.

Evaluator-ready streams are gzip-compressed JSON, either as one nested file or
as one file per window:

```text
generated/runs/<run-name>/streams/stream.json.gz
generated/runs/<run-name>/streams/window_000000.json.gz
generated/runs/<run-name>/streams/window_000001.json.gz
```

## Pipeline

Synthetic data starts directly at the generator:

```text
generator scenario
  -> global per-window key counts
  -> partitioning policy
  -> evaluator-ready stream
  -> evaluator / experiments
```

Real-world data starts in `data_preparation/`:

```text
raw external dataset
  -> dataset-specific converter
  -> normalized real counts JSONL
  -> generator partitioning policy
  -> evaluator-ready stream
  -> evaluator / experiments
```

This separation is intentional. Dataset-specific parsing stays outside the
generator, while the generator and evaluator only need to understand normalized
counts and partitioned streams.

## Generator Configs

Generator configs are split by concern:

```text
generator/config/
  generation/      # run scale, output names, plotting, stream expansion
  partitioning/    # partitioning policy and policy-specific knobs
  scenarios/       # synthetic global frequency distributions
  runs/            # small presets composing the three above
```

A run preset points to one generation config, one scenario, and one
partitioning config:

```json
{
  "generation": "../generation/default.json",
  "scenario": "../scenarios/default.json",
  "partitioning": "../partitioning/synthetic_mixture.json"
}
```

When `--real-counts` is supplied, the scenario config is ignored because the
global per-window counts come from the real-count file.

## Common Workflows

Generate a default synthetic stream:

```bash
cd generator
python3 main.py
```

Generate a synthetic stream with round-robin partitioning:

```bash
cd generator
python3 main.py --run-config config/runs/round_robin.json
```

Convert CAIDA pcaps to normalized real counts:

```bash
python3 data_preparation/caida_analyzer.py /path/to/caida \
  --output /tmp/caida_counts \
  --window-seconds 10
```

Partition normalized real counts into per-window evaluator input:

```bash
cd generator
python3 main.py \
  --run-config config/runs/round_robin.json \
  --real-counts /tmp/caida_counts.jsonl \
  --save-windows-separately
```

Build the evaluator:

```bash
cmake -S evaluator -B evaluator/build -DCMAKE_BUILD_TYPE=Release
cmake --build evaluator/build
```

Run the batch evaluator directly:

```bash
./evaluator/build/hh_bench \
  generator/generated/runs/<run-name>/streams \
  20 200 64 oracle,ss,hybrid \
  --csv-out /tmp/results.csv
```

Run the experiment orchestration script:

```bash
python3 experiments/run_experiments.py \
  --dataset caida:generator/generated/runs/<run-name>/streams \
  --m 20 --n-param 200 --mem-kib 32,64,128 --topk 200 --build
```

## Partitioning Policies

The generator currently supports:

- `round_robin`: simple even split of each key across partitions.
- `synthetic_mixture`: stochastic synthetic partitioning using uniform,
  normal-shaped, and skewed allocation shapes.
- `imbalance`: rarity-aware skewing of low-frequency keys.
- `hh_hidden_optimal`: stress-test construction that tries to hide global heavy
  hitters below local thresholds while adding distractors.
- `hh_hidden_exact`: hidden-heavy-hitter construction without distractor
  amplification.

See [generator/README.md](generator/README.md) for details and config examples.

## Window IDs

Window IDs are zero-based throughout the active pipeline. The generator emits
zero-based IDs for synthetic and real-count sources. The evaluator preserves
numeric IDs in single-file streams and infers IDs from per-window filenames such
as `window_000000.json.gz`.

For chunked real-world preprocessing, every chunk should use the same source
origin and window duration so the normalized window IDs remain comparable before
the generator re-indexes them.

## Dependencies

Python is used for data preparation, generation, and experiment orchestration.
The CAIDA converter requires `dpkt` when processing pcap files:

```bash
pip install dpkt
```

The evaluator requires a C++17 compiler, CMake, zlib development headers, and a
standard POSIX environment. See [evaluator/README.md](evaluator/README.md) for
build and runtime details.

## Generated Artifacts

Generator outputs are written under:

```text
generator/generated/runs/<run-name>/
```

Each run snapshots the configs used under its local `config/` directory and
writes a `manifest.json` with the input paths and output locations. This makes
generated streams reproducible even when top-level config files later change.
