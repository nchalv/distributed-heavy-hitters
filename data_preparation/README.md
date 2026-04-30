# Data Preparation

Utilities in this directory convert raw or external datasets into the
per-window count formats consumed by the generator.

The normalized interchange format is JSONL with one row per logical window:

```json
{"window": 0, "counts": {"key": 123, "other_key": 45}}
```

Dataset-specific converters should emit that shape so the generator and
evaluator remain dataset-agnostic.

## CAIDA

`caida_analyzer.py` converts CAIDA `.pcap.gz` traces into JSONL rows:

```json
{"window": 0, "counts": {"192.0.2.1": 42}}
```

Example:

```bash
python3 data_preparation/caida_analyzer.py /path/to/caida \
  --output /tmp/caida_counts \
  --window-seconds 10
```

The generated `/tmp/caida_counts.jsonl` can then be passed to the generator:

```bash
cd generator
python3 main.py \
  --run-config config/runs/round_robin.json \
  --real-counts /tmp/caida_counts.jsonl \
  --save-windows-separately
```

For chunked CAIDA processing, pass the same `--origin-timestamp` and
`--window-seconds` to every chunk so window IDs remain comparable.

`--window-seconds` is the canonical window-size argument and supports
sub-minute windows. The older `--window` option is still accepted as minutes for
compatibility.
