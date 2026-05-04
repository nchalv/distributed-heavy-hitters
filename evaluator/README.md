# Heavyheads Experimental Harness

This repo contains the C++ harness used to reproduce the experiments from **“Adaptive Frequency Estimation for Load-Balanced Distributed Stream Processing.”** It simulates a load-balanced distributed pipeline, instantiates several heavy-hitter sketches, and applies the sizing policies described in the paper.

## Layout
- `apps/hh_run.cpp` – single-method driver for interactive runs and debugging.
- `apps/hh_bench.cpp` – batch benchmark runner that evaluates multiple methods against the oracle and reports window-aggregated metrics.
- `src/` and `include/` – sketch implementations (SpaceSaving, HeavyLocker, CHK), hybrid exact/approx head–tail sketch, and policy logic for adaptive sizing.
- `tests/` – light sanity checks for the sketch primitives.

## Input format
Streams can be provided in either of these forms:

- Single gzip-compressed JSON file (original format).
- Directory containing one gzip-compressed JSON file per window (new format), e.g. `stream_window/window_000000.json.gz`, `stream_window/window_000001.json.gz`, ...

Window IDs are zero-based. The evaluator preserves numeric window IDs from
single-file streams and infers zero-based IDs from per-window filenames such as
`window_000000.json.gz`.

### Single-file format
The original format is a gzip-compressed JSON with a nested structure per window and partition:

```json
{
  "0": {                          // window id (string, numeric preferred)
    "0": ["a", ["b", 3]],         // partition id -> list of keys or [key, weight]
    "1": [["a", 2], "c"]
  },
  "1": { "0": ["d"] }
}
```

`JsonGzNestedReader` preserves the natural numeric ordering of windows and partitions; any non-numeric keys fall back to lexical order and default to index 0.

### Directory-per-window format
If the input path is a directory, the reader loads all `*.json.gz` files in lexical filename order.

Each file may be either:
- The same nested format as above (`{window:{partition:[...]}}`), or
- A single-window partition object (`{partition:[...]}`), in which case the window id is inferred from the filename `window_<num>.json.gz` when possible (otherwise lexical file order is used).

## Building
```sh
cmake -S evaluator -B evaluator/build -DCMAKE_BUILD_TYPE=Release
cmake --build evaluator/build
```

For quick local rebuilds there is also an evaluator-local `makefile`:
- `make -C evaluator all` builds the binaries into `evaluator/build`.
- `make -C evaluator clean` drops `evaluator/build` for a fresh out-of-tree build.

## Requirements
- C++17 compiler (gcc/clang).
- CMake >= 3.16.
- zlib development headers (for gzip via `zstr`).
- Standard POSIX environment (pthread); no external runtime services required.

Executables land in `evaluator/build/` (e.g., `hh_run`, `hh_bench`, `test_basic`).

## Running `hh_run`
Single-method exploration with per-window telemetry:
```sh
./hh_run <stream.json.gz|stream_dir> <m> <n_param> <memKiB_each> <method:oracle|ss|hl|chk|hybrid> \
  [--policy difficulty|static] [--alpha-req A] [--r-m RM] \
  [--hyb-tail n|2n|difficulty] [--q kN]
```
- `m`: number of partitions; `n_param`: global HH denominator (strict threshold `floor(N/n_param)+1`, i.e., `p(k) > 1/n`).
- `memKiB_each`: memory budget per partition; `--q kN` sets static SS capacity to `k * n_param`.
- Policies (SpaceSaving only): `difficulty` applies the requirement-based predictive controller; `static` holds `q` fixed.
- Space-Saving exports per-item error certificates by default. Pass `--ss-eps global` only for the coarser global-maximum error mode.
- `hybrid` uses an exact head (`q_e`) plus SS tail (`q_a`); the tail can be fixed or controlled with the same difficulty policy via `--hyb-tail`.

The program prints per-window global counts, certified bounds, and the next `q` suggestion when an adaptive policy is active.

## Running `hh_bench`
Batch evaluation against the oracle (exact counts):
```sh
./hh_bench <stream.json.gz|stream_dir> <m> <n_param> <memKiB_each> <methods_csv> \
  [--csv-out results.csv] [--topk K] [policy flags...]
```
`methods_csv` can include multiple entries (e.g., `oracle,ss,hybrid`). The first `oracle` run is treated as ground truth; subsequent methods reuse the same stream ordering. Metrics reported per window and as averages:
- Heavy-hitter precision/recall at the paper’s HH threshold 𝑝(𝑘) > 1/𝑛 (implemented as `f(k) ≥ floor(N/n_param)+1` with your supplied `n_param` = 𝑛).
- Heavy-hitter F1 at the same threshold.
- Average absolute/relative error on oracle heavy hitters.
- Optional top-`K` overlap if `--topk` is supplied.

## Running experiment sweeps
The top-level `experiments/run_experiments.py` script automates the
paper-oriented experiment grid. It runs `hh_bench`, stores one CSV per run under
a timestamped output directory, writes a manifest, and creates PDF/PNG plots for
parameter sweeps, memory-quality frontiers, controller traces, certification
mass/interval summaries, and timing overheads.

```sh
python3 experiments/run_experiments.py \
  --dataset hash:/path/to/hash_partitioned_stream \
  --dataset stress:/path/to/stress_partitioned_stream \
  --m 16 --n-param 200 --mem-kib 32,64,128,256 --topk 200 --build
```

Datasets must already be partitioned in the input format above. The script uses
one-at-a-time sweeps around the recommended default controller configuration by
default; pass `--profile full` to add the pairwise ambiguity-gain grid. Use
`--dry-run` to inspect the generated `hh_bench` commands without executing them.

## Relation to the paper
- The implemented adaptive sizing policy is the paper’s requirement-based controller: it estimates the current per-key capacity requirement over the certified/ambiguous candidate set, smooths the effective requirement in log space, and applies one-sided residual calibration before choosing the next-window capacity.
- The hybrid sketch mirrors the paper’s head/tail decomposition: an exact head seeded from the previous window’s certified set, plus an adaptive SS tail sized for the remaining mass.
