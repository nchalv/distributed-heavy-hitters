#!/usr/bin/env python3
"""
CAIDA Dataset Batch Processor - Processes trace directories and produces
consolidated JSONL output with zero-based window IDs.
Handles both:
  - Parent directory with multiple date subdirectories
  - Single date directory directly
"""

import gzip
import socket
import argparse
import json
import os
import sys
import glob
from collections import defaultdict, Counter
from datetime import datetime
import time

try:
    import dpkt
except ImportError:
    dpkt = None


def require_dpkt():
    if dpkt is None:
        raise RuntimeError("dpkt is required to process CAIDA pcap files. Install it with: pip install dpkt")


def process_single_file(
    pcap_file_path,
    window_seconds=60.0,
    origin_timestamp=0.0,
    sample_mode=False,
    sample_packets=1000000,
):
    """
    Process a single CAIDA pcap file and return per-window source-IP counts.

    Window IDs are computed relative to origin_timestamp so independently
    processed chunks can be merged if they use the same origin and window size.
    """
    window_counts = {}
    first_timestamp = None
    last_timestamp = None
    packet_count = 0
    require_dpkt()
    
    open_func = gzip.open if pcap_file_path.endswith('.gz') else open
    
    try:
        with open_func(pcap_file_path, 'rb') as f:
            pcap_reader = dpkt.pcap.Reader(f)
            
            for i, (timestamp, buf) in enumerate(pcap_reader):
                if sample_mode and i >= sample_packets:
                    break
                    
                packet_count += 1
                
                if first_timestamp is None:
                    first_timestamp = timestamp
                last_timestamp = timestamp
                
                try:
                    # Try IPv4 first
                    try:
                        ip = dpkt.ip.IP(buf)
                        src_ip = socket.inet_ntoa(ip.src)
                    except dpkt.dpkt.UnpackError:
                        # Try IPv6
                        try:
                            ip6 = dpkt.ip6.IP6(buf)
                            src_ip = socket.inet_ntop(socket.AF_INET6, ip6.src)
                        except dpkt.dpkt.UnpackError:
                            continue
                    
                    # Calculate window ID relative to the shared dataset origin.
                    window_id = int((timestamp - origin_timestamp) // window_seconds)
                    if window_id < 0:
                        continue
                    
                    if window_id not in window_counts:
                        window_counts[window_id] = Counter()
                    
                    window_counts[window_id][src_ip] += 1
                    
                except Exception:
                    continue
                    
    except Exception as e:
        print(f"  Warning: Error processing {os.path.basename(pcap_file_path)}: {e}")
        return None, None, None, 0
    
    return window_counts, first_timestamp, last_timestamp, packet_count


def find_pcap_files(input_path, direction='both'):
    """
    Find all pcap files in the given path.
    
    Args:
        input_path: Can be a directory or a file pattern
        direction: 'A', 'B', or 'both'
    
    Returns:
        List of (group_name, file_path) tuples
    """
    all_files = []
    
    # Check if input_path is a directory
    if os.path.isdir(input_path):
        # Look for all pcap.gz files in the directory
        pcap_files = sorted(glob.glob(os.path.join(input_path, "*.pcap.gz")))
        
        if pcap_files:
            # It's a directory with pcap files directly
            dir_name = os.path.basename(input_path)
            for f in pcap_files:
                all_files.append((dir_name, f))
        else:
            # Maybe it's a parent directory with date subdirectories
            subdirs = sorted(glob.glob(os.path.join(input_path, "20*")))
            for subdir in subdirs:
                subdir_name = os.path.basename(subdir)
                subdir_files = sorted(glob.glob(os.path.join(subdir, "*.pcap.gz")))
                for f in subdir_files:
                    all_files.append((subdir_name, f))
    else:
        # Input might be a pattern (e.g., with wildcards)
        matching_files = sorted(glob.glob(input_path))
        for f in matching_files:
            # Try to extract group name from parent directory
            parent_dir = os.path.basename(os.path.dirname(f))
            all_files.append((parent_dir, f))
    
    # Filter by direction if specified
    if direction == 'A':
        all_files = [(g, f) for g, f in all_files if 'dirA' in os.path.basename(f)]
    elif direction == 'B':
        all_files = [(g, f) for g, f in all_files if 'dirB' in os.path.basename(f)]
    
    return sorted(all_files, key=lambda x: x[1])  # Sort by file path


def process_dataset(
    input_path,
    window_seconds=60.0,
    origin_timestamp=None,
    direction='both',
    sample_mode=False,
    sample_packets=1000000,
):
    """
    Process pcap files and produce normalized per-window count records.
    
    Args:
        input_path: Path to directory or file pattern
        window_seconds: Size of time windows in seconds
        origin_timestamp: Timestamp that maps to window 0. If omitted, use the
            earliest readable packet timestamp found in the selected files.
        direction: 'A', 'B', or 'both'
        sample_mode: If True, only process first N packets per file
    
    Returns:
        Consolidated window counts and metadata
    """
    # Find all files
    all_files = find_pcap_files(input_path, direction)
    
    if not all_files:
        print(f"No pcap files found in {input_path}")
        return None, None, None

    require_dpkt()
    
    print(f"\n{'='*70}")
    print(f"CAIDA DATASET PROCESSOR")
    print(f"{'='*70}")
    print(f"Input: {input_path}")
    print(f"Direction: {direction}")
    print(f"Found {len(all_files)} files:")
    
    # Group by directory for display
    dir_counts = defaultdict(int)
    for group, _ in all_files:
        dir_counts[group] += 1
    
    for group, count in sorted(dir_counts.items()):
        print(f"  {group}: {count} files")
    print(f"{'='*70}\n")
    
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")

    # First pass: find origin timestamp unless the caller provided one.
    if origin_timestamp is None:
        print("First pass: Finding dataset origin timestamp...")
        global_start_time = None

        for group, file_path in all_files:
            open_func = gzip.open if file_path.endswith('.gz') else open
            try:
                with open_func(file_path, 'rb') as f:
                    pcap_reader = dpkt.pcap.Reader(f)
                    try:
                        timestamp, _ = next(pcap_reader)
                        if global_start_time is None or timestamp < global_start_time:
                            global_start_time = timestamp
                    except StopIteration:
                        pass
            except Exception:
                continue

        if global_start_time is None:
            print("Warning: Could not determine origin timestamp, using 0 as base")
            global_start_time = 0.0
    else:
        global_start_time = float(origin_timestamp)

    global_start_dt = datetime.fromtimestamp(global_start_time)
    print(f"Origin timestamp: {global_start_dt} ({global_start_time:.6f})")
    print(f"Window size: {window_seconds:g} seconds")
    print(f"{'='*70}\n")
    
    # Second pass: process all files
    consolidated_counts = defaultdict(Counter)
    file_stats = []
    total_packets = 0
    
    for idx, (group, file_path) in enumerate(all_files, 1):
        file_base = os.path.basename(file_path)
        print(f"[{idx}/{len(all_files)}] Processing {group}/{file_base}")
        
        # Check that the file is readable and non-empty.
        open_func = gzip.open if file_path.endswith('.gz') else open
        
        try:
            with open_func(file_path, 'rb') as f:
                pcap_reader = dpkt.pcap.Reader(f)
                try:
                    next(pcap_reader)
                except StopIteration:
                    print("  File is empty, skipping")
                    continue
        except Exception as e:
            print(f"  Error reading file: {e}")
            continue

        # Process the file
        file_counts, first_ts, last_ts, pkt_count = process_single_file(
            file_path,
            window_seconds,
            global_start_time,
            sample_mode,
            sample_packets
        )
        
        if file_counts and pkt_count > 0:
            # Merge counts
            for window_id, counter in file_counts.items():
                consolidated_counts[window_id].update(counter)
            
            total_packets += pkt_count
            file_stats.append({
                'file': f"{group}/{file_base}",
                'packets': pkt_count,
                'windows': len(file_counts),
                'first_timestamp': first_ts,
                'last_timestamp': last_ts,
                'window_range': f"{min(file_counts.keys())}-{max(file_counts.keys())}"
            })
            
            print(f"  Processed {pkt_count:,} packets across {len(file_counts)} windows")
        else:
            print(f"  No packets processed")
        
        if idx % 10 == 0:
            time.sleep(0.1)
    
    # Print summary
    print(f"\n{'='*70}")
    print("PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Total packets processed: {total_packets:,}")
    print(f"Total unique windows: {len(consolidated_counts):,}")
    
    if consolidated_counts:
        print(f"Window range: {min(consolidated_counts.keys())} to {max(consolidated_counts.keys())}")
    
    return consolidated_counts, file_stats, global_start_time


def save_jsonl(consolidated_counts, output_prefix):
    """Save results in JSONL format."""
    output_file = f'{output_prefix}.jsonl'
    
    print(f"\nSaving JSONL to: {output_file}")
    
    sorted_windows = sorted(consolidated_counts.items())
    
    with open(output_file, 'w') as f:
        for window_id, counter in sorted_windows:
            counts_dict = dict(counter)
            json_line = json.dumps({
                "window": window_id,
                "counts": counts_dict
            })
            f.write(json_line + '\n')
    
    file_size = os.path.getsize(output_file) / (1024*1024)
    print(f"Saved: {output_file} ({file_size:.1f} MB)")
    print(f"  Total windows: {len(sorted_windows):,}")


def save_metadata(file_stats, global_start_time, output_prefix, window_seconds, direction):
    """Save processing metadata."""
    metadata_file = f'{output_prefix}_metadata.json'
    
    metadata = {
        'dataset_start_time': {
            'timestamp': int(global_start_time) if global_start_time else 0,
            'datetime': datetime.fromtimestamp(global_start_time).isoformat() if global_start_time else None
        },
        'window_seconds': window_seconds,
        'window_minutes': window_seconds / 60.0,
        'direction': direction,
        'total_files_processed': len(file_stats),
        'total_packets': sum(f['packets'] for f in file_stats),
        'files': file_stats
    }
    
    for f in metadata['files']:
        if f.get('first_timestamp'):
            f['first_datetime'] = datetime.fromtimestamp(f['first_timestamp']).isoformat()
        if f.get('last_timestamp'):
            f['last_datetime'] = datetime.fromtimestamp(f['last_timestamp']).isoformat()
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved metadata: {metadata_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Process CAIDA pcap files and produce JSONL output with window IDs and IP counts',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('input_path', 
                       help='Path to directory or file pattern (e.g., "/path/to/equinix-nyc" or "/path/to/20180315-130000.UTC")')
    parser.add_argument('--output', '-o', default='caida_output',
                       help='Output file prefix (default: caida_output)')
    parser.add_argument('--window-seconds', type=float, default=None,
                       help='Window size in seconds. Supports sub-minute windows.')
    parser.add_argument('--window', '-w', type=float, default=None,
                       help='Deprecated: window size in minutes. Use --window-seconds.')
    parser.add_argument('--origin-timestamp', type=float, default=None,
                       help='Timestamp that maps to window 0. Use this for chunked processing.')
    parser.add_argument('--direction', '-d', choices=['A', 'B', 'both'], default='both',
                       help='Process only dirA, dirB, or both (default: both)')
    parser.add_argument('--sample', '-s', action='store_true',
                       help='Sample mode - only process first 1M packets per file')
    parser.add_argument('--sample-size', type=int, default=1000000,
                       help='Number of packets to process per file in sample mode')
    
    args = parser.parse_args()
    
    window_seconds = args.window_seconds
    if window_seconds is None:
        window_seconds = 60.0 if args.window is None else args.window * 60.0

    # Process the dataset
    consolidated_counts, file_stats, global_start_time = process_dataset(
        args.input_path,
        window_seconds=window_seconds,
        origin_timestamp=args.origin_timestamp,
        direction=args.direction,
        sample_mode=args.sample,
        sample_packets=args.sample_size
    )
    
    if not consolidated_counts:
        print("No data processed. Exiting.")
        sys.exit(1)
    
    # Save outputs
    save_jsonl(consolidated_counts, args.output)
    save_metadata(file_stats, global_start_time, args.output, window_seconds, args.direction)
    
    # Show sample
    print(f"\nSample output (first 3 windows):")
    print("-" * 50)
    sorted_windows = sorted(consolidated_counts.items())[:3]
    for window_id, counter in sorted_windows:
        top_ips = dict(counter.most_common(5))
        sample_item = {
            "window": window_id,
            "counts": top_ips,
            "total_ips": len(counter),
            "total_packets": sum(counter.values())
        }
        print(json.dumps(sample_item, indent=2))
    
    print(f"\nProcessing complete!")


if __name__ == "__main__":
    main()
