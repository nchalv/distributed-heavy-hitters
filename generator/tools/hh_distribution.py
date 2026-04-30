import argparse, gzip, json
from pathlib import Path
from typing import Dict, List, Tuple
import ijson  # required

def _window_file_path(stream_path: Path, target_win: int) -> Path:
    return stream_path / f"window_{target_win:06d}.json.gz"


def _load_window_from_single_file(path: Path, target_win: int):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for win_key, win_val in ijson.kvitems(f, ""):  # top-level object keys/values
            if win_key == str(target_win):
                for part_key, items in win_val.items():
                    yield part_key, items
                return
    raise KeyError(f"window {target_win} not found")


def _load_window_from_directory(stream_dir: Path, target_win: int):
    win_file = _window_file_path(stream_dir, target_win)
    if not win_file.exists():
        raise KeyError(f"window file not found: {win_file}")
    with gzip.open(win_file, "rt", encoding="utf-8") as f:
        win_val = json.load(f)
    for part_key, items in win_val.items():
        yield part_key, items


def _load_window_streaming(stream_path: Path, target_win: int):
    """
    Supports either:
    - monolithic file: stream.json.gz ({window: {partition: [keys]}})
    - window directory: stream_windows/window_000123.json.gz ({partition: [keys]})
    """
    if stream_path.is_dir():
        yield from _load_window_from_directory(stream_path, target_win)
    else:
        yield from _load_window_from_single_file(stream_path, target_win)

def count_key(stream_path: Path, window: int, key: str) -> Tuple[Dict[int, int], int]:
    per_part: Dict[int, int] = {}
    total = 0
    for part_key, items in _load_window_streaming(stream_path, window):
        try:
            pid = int(part_key)
        except ValueError:
            pid = part_key
        cnt = sum(1 for x in items if x == key)
        per_part[pid] = cnt
        total += cnt
    return per_part, total

def top_keys(stream_path: Path, window: int, k: int) -> List[Tuple[str, int]]:
    from collections import Counter
    c = Counter()
    for _, items in _load_window_streaming(stream_path, window):
        c.update(items)
    return c.most_common(k)

def top_keys_partition(stream_path: Path, window: int, partition, k: int) -> List[Tuple[str, int]]:
    from collections import Counter
    target = str(partition)
    c = Counter()
    found = False
    for part_key, items in _load_window_streaming(stream_path, window):
        if part_key == target:
            c.update(items)
            found = True
            break
    if not found:
        raise KeyError(f"partition {partition} not found in window {window}")
    return c.most_common(k)

def main():
    ap = argparse.ArgumentParser(description="Inspect per-partition distribution of a key in a window.")
    ap.add_argument(
        "--stream",
        required=True,
        type=Path,
        help="Path to stream.json.gz or stream_windows/ directory produced by main.py",
    )
    ap.add_argument("--window", required=True, type=int, help="Window index (0-based)")
    ap.add_argument("--key", required=True, help="Key to inspect (e.g., key_947)")
    ap.add_argument("--top", type=int, default=0, help="Optional: also show top-K keys in the window")
    ap.add_argument("--partition", type=str, help="Partition id to inspect (use with --part-top)")
    ap.add_argument("--part-top", type=int, default=0, help="Top-K inside a specific partition (requires --partition)")
    args = ap.parse_args()

    per_part, total = count_key(args.stream, args.window, args.key)
    print(f"Key '{args.key}' in window {args.window}: total={total}")
    for pid in sorted(per_part, key=lambda x: (isinstance(x, str), x)):
        print(f"  partition {pid}: {per_part[pid]}")

    if args.top and args.top > 0:
        print(f"\nTop {args.top} keys in window {args.window}:")
        for k, v in top_keys(args.stream, args.window, args.top):
            print(f"  {k}: {v}")

    if args.part_top and args.part_top > 0:
        if args.partition is None:
            ap.error("--part-top requires --partition")
        top_p = top_keys_partition(args.stream, args.window, args.partition, args.part_top)
        print(f"\nTop {args.part_top} keys in window {args.window}, partition {args.partition}:")
        for k, v in top_p:
            print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
