from typing import List, Dict, Any, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment
import random
import csv
import json
import os
import gzip
from collections import defaultdict


from visualisation.data_visualiser import (
    plot_frequency_distribution_with_hh,
    plot_partition_skew
)

from data.distributions import (
    ZipfianDistributionGenerator,
    UniformDistributionGenerator,
    NormalDistributionGenerator,
    FlattenedHHDistributionGenerator
)
from data.partitioning import assign_partitions
from data.data_utils import save_compressed_json


# === Scenario Runner ===

def _apply_profile(dist_type, profile, params, n):
    if not profile:
        return params

    if dist_type == "zipfian":
        profiles = {
            "low_skew": {"s": 1.05},
            "medium_skew": {"s": 1.2},
            "high_skew": {"s": 1.5},
        }
    elif dist_type == "flattened":
        profiles = {
            "low_skew": {"num_hh": max(1, n // 20), "flatness": 0.2},
            "medium_skew": {"num_hh": max(1, n // 10), "flatness": 0.5},
            "high_skew": {"num_hh": max(1, n // 5), "flatness": 0.8},
        }
    elif dist_type == "normal":
        profiles = {
            "narrow": {"std_frac": 0.02},
            "medium": {"std_frac": 0.031},
            "wide": {"std_frac": 0.06},
        }
    else:
        profiles = {}

    prof_params = profiles.get(profile, {})
    merged = dict(prof_params)
    merged.update(params)
    return merged


def _resolve_step_config(step, default_total_items, default_num_keys, default_n):
    step_total_items = step.get("total_items", default_total_items)
    step_num_keys = step.get("num_keys", default_num_keys)
    step_n = step.get("n", default_n)
    step_params = dict(step.get("params", {}))
    step_profile = step.get("profile")
    step_params = _apply_profile(step["type"], step_profile, step_params, step_n)
    if step["type"] in {"normal", "flattened"} and "n" not in step_params:
        step_params["n"] = step_n
    return step_total_items, step_num_keys, step_n, step_params


def run_scenario(scenario, total_items, num_keys, default_n):
    gens = {
        'uniform': UniformDistributionGenerator,
        'normal': NormalDistributionGenerator,
        'flattened': FlattenedHHDistributionGenerator,
        'zipfian': ZipfianDistributionGenerator
    }

    def interpolate_distributions_smoothly(dist_a, dist_b, steps):
        keys = set(dist_a).union(dist_b)
        windows = []
        for t in range(1, steps + 1):
            alpha = t / (steps + 1)
            blended = {k: int(round((1 - alpha) * dist_a.get(k, 0) + alpha * dist_b.get(k, 0)))
                    for k in keys}
            windows.append({k: v for k, v in blended.items() if v > 0})
        return windows

    output = []
    num_windows = 0
    for step in scenario:
        step_total_items, step_num_keys, win_n, step_params = _resolve_step_config(
            step, total_items, num_keys, default_n
        )
        G = gens[step['type']]
        gen = G(**step_params) if step_params else G()
        dur = step['duration']
        trans = step.get('transition', {})
        if trans:
            from_step = {
                "type": trans["from"],
                "params": trans.get("from_params", {}),
                "n": trans.get("n", win_n),
                "profile": trans.get("from_profile"),
                "total_items": trans.get("from_total_items", step_total_items),
                "num_keys": trans.get("from_num_keys", step_num_keys),
            }
            from_total_items, from_num_keys, _, from_params = _resolve_step_config(
                from_step, total_items, num_keys, default_n
            )
            G_from = gens[trans['from']]
            gen_from = G_from(**from_params) if from_params else G_from()
            dist_a = gen_from.generate(from_total_items, from_num_keys)
            dist_b = gen.generate(step_total_items, step_num_keys)
            intermediate = interpolate_distributions_smoothly(dist_a, dist_b, trans['transition_windows'])
            for t_dist in intermediate:
                output.append(('transition', t_dist, win_n))
                num_windows += 1

        for _ in range(dur):
            output.append((step['type'], gen.generate(step_total_items, step_num_keys), win_n))
            num_windows += 1

    return output, num_windows

def smooth_key_transitions(
    windows: List[Tuple[str, Dict[str, int], int]],
    *,
    hot_key_drift_pct: float = 0.0,
    seed: int = 0,
    max_rel_freq_delta_pct: float = 20.0,
    new_key_pct: float = 10.0,
    newborn_hot_birth_prob: float = 0.05,
    max_hot_newborn_frac: float = 0.10,
):
    result = []
    # Birth/death smoothing constants (kept internal for now).
    # New IDs start cold; only cold keys are eligible for extinction.
    newborn_hot_cap_frac = 0.6
    extinction_prev_hot_cap_frac = 0.6
    drift_pct = max(0.0, min(100.0, float(hot_key_drift_pct)))
    max_rel_delta = max(0.0, min(100.0, float(max_rel_freq_delta_pct))) / 100.0
    new_key_ratio = max(0.0, min(100.0, float(new_key_pct))) / 100.0
    newborn_hot_birth_prob = max(0.0, min(1.0, float(newborn_hot_birth_prob)))
    max_hot_newborn_frac = max(0.0, min(1.0, float(max_hot_newborn_frac)))
    rng = random.Random(seed)
    prev_keys = None
    prev_freqs = None
    prev_n = None
    next_global_id = 1

    def _key_id(key: str) -> int:
        if isinstance(key, str) and key.startswith("key_"):
            try:
                return int(key.split("_", 1)[1])
            except ValueError:
                return 0
        return 0

    def _hot_keys_from_map(freq_map: Dict[str, int], n_workers: int) -> set[str]:
        total = sum(int(v) for v in freq_map.values() if int(v) > 0)
        if total <= 0 or int(n_workers) <= 0:
            return set()
        threshold = (total // int(n_workers)) + 1
        return {k for k, v in freq_map.items() if int(v) >= threshold}

    def _apply_relative_delta_cap(curr_map: Dict[str, int], prev_map: Dict[str, int]) -> Dict[str, int]:
        if max_rel_delta >= 1.0:
            return curr_map
        keys = list(curr_map.keys())
        if not keys:
            return {}

        v = np.array([float(max(0, int(curr_map[k]))) for k in keys], dtype=float)
        lo = np.zeros(len(keys), dtype=float)
        hi = np.zeros(len(keys), dtype=float)

        for i, k in enumerate(keys):
            pv = int(prev_map.get(k, 0))
            if pv > 0:
                lo_i = int(np.floor(pv * (1.0 - max_rel_delta)))
                hi_i = int(np.ceil(pv * (1.0 + max_rel_delta)))
            else:
                lo_i = 0
                hi_i = max(int(v[i]), 0)
            lo_i = max(0, lo_i)
            hi_i = max(lo_i, hi_i)
            lo[i] = float(lo_i)
            hi[i] = float(hi_i)

        target_sum = float(np.sum(v))
        lo_sum = float(np.sum(lo))
        hi_sum = float(np.sum(hi))

        # If bounds are infeasible for requested mass, relax uniformly so a bounded solution exists.
        if target_sum > hi_sum:
            hi += (target_sum - hi_sum) / float(len(keys))
        elif target_sum < lo_sum:
            lo = np.maximum(0.0, lo - (lo_sum - target_sum) / float(len(keys)))

        # Recompute feasibility envelope after relaxation.
        lo_sum = float(np.sum(lo))
        hi_sum = float(np.sum(hi))
        target_eff = min(max(target_sum, lo_sum), hi_sum)

        def total_with_shift(shift: float) -> float:
            return float(np.sum(np.clip(v + shift, lo, hi)))

        low_s, high_s = -1.0, 1.0
        for _ in range(80):
            if total_with_shift(low_s) <= target_eff:
                break
            low_s *= 2.0
        for _ in range(80):
            if total_with_shift(high_s) >= target_eff:
                break
            high_s *= 2.0
        for _ in range(70):
            mid = 0.5 * (low_s + high_s)
            if total_with_shift(mid) < target_eff:
                low_s = mid
            else:
                high_s = mid
        x = np.clip(v + high_s, lo, hi)

        # Integerize while preserving exact total mass and bounds.
        x_floor = np.floor(x).astype(int)
        lo_i = np.floor(lo).astype(int)
        hi_i = np.ceil(hi).astype(int)
        x_floor = np.maximum(lo_i, np.minimum(hi_i, x_floor))
        residual = int(round(target_eff)) - int(np.sum(x_floor))

        if residual > 0:
            room = hi_i - x_floor
            order = np.argsort(-(x - x_floor))
            for idx in order:
                if residual <= 0:
                    break
                if room[idx] <= 0:
                    continue
                add = min(room[idx], residual)
                x_floor[idx] += add
                residual -= add
        elif residual < 0:
            need = -residual
            room = x_floor - lo_i
            order = np.argsort(x - x_floor)
            for idx in order:
                if need <= 0:
                    break
                if room[idx] <= 0:
                    continue
                take = min(room[idx], need)
                x_floor[idx] -= take
                need -= take

        return {k: int(max(0, x_floor[i])) for i, k in enumerate(keys)}

    for idx, (dtype, freq_dict, n) in enumerate(windows):
        curr_keys = list(freq_dict)
        curr_freqs = np.array([freq_dict[k] for k in curr_keys])
        if idx == 0:
            result.append((dtype, {f'key_{i+1}': curr_freqs[i] for i in range(len(curr_keys))}, n))
            prev_keys = [f'key_{i+1}' for i in range(len(curr_keys))]
            prev_freqs = curr_freqs
            prev_n = n
            next_global_id = max((_key_id(k) for k in prev_keys), default=0) + 1
            continue
        size = max(len(prev_freqs), len(curr_freqs))
        A = np.pad(prev_freqs, (0, size - len(prev_freqs)))
        B = np.pad(curr_freqs, (0, size - len(curr_freqs)))
        cost = np.abs(A[:, None] - B[None, :])
        r, c = linear_sum_assignment(cost)
        matched_i_to_j = {}
        matched_j_to_i = {}
        for i, j in zip(r, c):
            if i < len(prev_keys) and j < len(curr_keys):
                matched_i_to_j[i] = j
                matched_j_to_i[j] = i

        curr_total = int(curr_freqs.sum())
        curr_threshold = (curr_total // int(n)) + 1 if curr_total > 0 and int(n) > 0 else 1
        prev_total = int(prev_freqs.sum()) if prev_freqs is not None else 0
        prev_threshold = (prev_total // int(prev_n)) + 1 if prev_total > 0 and int(prev_n) > 0 else 1
        prev_hot_i = {i for i in range(len(prev_freqs)) if int(prev_freqs[i]) >= prev_threshold}
        curr_hot_j = {j for j in range(len(curr_freqs)) if int(curr_freqs[j]) >= curr_threshold}

        # Keys assigned to these slots will be replaced by brand-new IDs.
        force_new_j = set()
        if new_key_ratio > 0:
            target_new_ids = int(round(len(curr_keys) * new_key_ratio))
            target_new_ids = max(0, min(len(curr_keys), target_new_ids))
            cold_matched_slots = [j for j in matched_j_to_i if int(curr_freqs[j]) < curr_threshold]
            # Smooth extinction: only keys already well below hot threshold can die.
            cold_matched_slots = [
                j
                for j in cold_matched_slots
                if int(prev_freqs[matched_j_to_i[j]]) <= int(prev_threshold * extinction_prev_hot_cap_frac)
            ]
            cold_matched_slots.sort(key=lambda j: int(curr_freqs[j]))
            force_new_j.update(cold_matched_slots[:target_new_ids])

        # Build slot->existing-key mapping for non-new slots.
        slot_to_key: Dict[int, str] = {}
        for i, j in zip(r, c):
            if i < len(prev_keys) and j < len(curr_keys) and j not in force_new_j:
                slot_to_key[j] = prev_keys[i]

        # Hot-key churn: demote some previously-hot keys and promote some previously-cold keys.
        if drift_pct > 0 and curr_hot_j:
            prev_hot_keys = {prev_keys[i] for i in prev_hot_i}
            hot_slots = [j for j in curr_hot_j if j in slot_to_key]
            target_new_hot = int(round(len(curr_hot_j) * (drift_pct / 100.0)))
            target_new_hot = max(0, min(len(hot_slots), target_new_hot))
            current_new_hot = sum(1 for j in hot_slots if slot_to_key[j] not in prev_hot_keys)

            if current_new_hot < target_new_hot:
                need = target_new_hot - current_new_hot
                demote_slots = [j for j in hot_slots if slot_to_key[j] in prev_hot_keys]
                promote_slots = [j for j in slot_to_key if j not in curr_hot_j and slot_to_key[j] not in prev_hot_keys]
                demote_slots.sort(key=lambda j: int(curr_freqs[j]), reverse=True)
                promote_slots.sort(key=lambda j: int(curr_freqs[j]), reverse=True)
                for hot_slot, cold_slot in zip(demote_slots[:need], promote_slots[:need]):
                    slot_to_key[hot_slot], slot_to_key[cold_slot] = slot_to_key[cold_slot], slot_to_key[hot_slot]
            elif current_new_hot > target_new_hot:
                excess = current_new_hot - target_new_hot
                demote_slots = [j for j in hot_slots if slot_to_key[j] not in prev_hot_keys]
                promote_slots = [j for j in slot_to_key if j not in curr_hot_j and slot_to_key[j] in prev_hot_keys]
                demote_slots.sort(key=lambda j: int(curr_freqs[j]), reverse=True)
                promote_slots.sort(key=lambda j: int(curr_freqs[j]))
                for hot_slot, old_slot in zip(demote_slots[:excess], promote_slots[:excess]):
                    slot_to_key[hot_slot], slot_to_key[old_slot] = slot_to_key[old_slot], slot_to_key[hot_slot]

        remap: Dict[str, int] = {}
        used_slots = set()
        for j, key in slot_to_key.items():
            remap[key] = int(curr_freqs[j])
            used_slots.add(j)
        for j in range(len(curr_keys)):
            if j not in used_slots:
                remap[f'key_{next_global_id}'] = int(curr_freqs[j])
                next_global_id += 1

        # Smooth birth: mostly cold newborns, with occasional allowed hot births.
        born_keys = [k for k in remap if k not in prev_keys]
        if born_keys:
            cold_cap = max(1, int(curr_threshold * newborn_hot_cap_frac))
            hot_cap = max(curr_threshold + 1, int(curr_threshold * 1.25))
            hot_budget = int(np.floor(len(born_keys) * max_hot_newborn_frac))
            hot_budget = max(0, min(len(born_keys), hot_budget))
            candidate_hot = [k for k in born_keys if remap[k] >= curr_threshold]
            rng.shuffle(candidate_hot)
            hot_birth_set = set()
            for k in candidate_hot:
                if len(hot_birth_set) >= hot_budget:
                    break
                if rng.random() < newborn_hot_birth_prob:
                    hot_birth_set.add(k)
            excess = 0
            for k in born_keys:
                cap_k = hot_cap if k in hot_birth_set else cold_cap
                if remap[k] > cap_k:
                    excess += remap[k] - cap_k
                    remap[k] = cap_k
            if excess > 0:
                recipients = [k for k in remap if k not in born_keys]
                recipients.sort(key=lambda kk: remap[kk], reverse=True)
                if not recipients:
                    recipients = born_keys
                ridx = 0
                while excess > 0 and recipients:
                    key = recipients[ridx % len(recipients)]
                    remap[key] += 1
                    excess -= 1
                    ridx += 1

        prev_map = {k: int(v) for k, v in zip(prev_keys, prev_freqs)}
        remap = _apply_relative_delta_cap(remap, prev_map)
        prev_keys = list(remap)
        prev_freqs = np.array([remap[k] for k in prev_keys])
        prev_n = n
        result.append((dtype, remap, n))
    return result


def enforce_hot_key_drift(
    windows: List[Tuple[str, Dict[str, int], int]],
    seed: int,
    hot_key_drift_pct: float = 0.0,
) -> List[Tuple[str, Dict[str, int], int]]:
    """
    Enforce a target share of hot keys in each window that are new
    relative to the previous window.
    """
    pct = max(0.0, min(100.0, float(hot_key_drift_pct)))
    if pct <= 0.0 or len(windows) <= 1:
        return windows

    rng = random.Random(seed)

    def _hot_keys(freq_dist: Dict[str, int], n: int) -> set[str]:
        total = sum(int(v) for v in freq_dist.values() if int(v) > 0)
        if total <= 0 or n <= 0:
            return set()
        threshold = (total // int(n)) + 1
        return {k for k, v in freq_dist.items() if int(v) >= threshold}

    output: List[Tuple[str, Dict[str, int], int]] = [windows[0]]
    prev_hot = _hot_keys(windows[0][1], windows[0][2])

    for idx in range(1, len(windows)):
        dtype, freq_dist, win_n = windows[idx]
        fd = dict(freq_dist)
        curr_hot = _hot_keys(fd, win_n)
        if not curr_hot:
            output.append((dtype, fd, win_n))
            prev_hot = curr_hot
            continue

        total = sum(int(v) for v in fd.values() if int(v) > 0)
        threshold = (total // int(win_n)) + 1 if win_n > 0 else 1
        target_new = int(round(len(curr_hot) * (pct / 100.0)))
        target_new = max(0, min(len(curr_hot), target_new))
        curr_new = curr_hot - prev_hot

        if len(curr_new) < target_new:
            need = target_new - len(curr_new)
            demote_pool = [k for k in (curr_hot & prev_hot) if fd.get(k, 0) >= threshold]
            promote_pool = [
                k
                for k, v in fd.items()
                if k not in prev_hot and k not in curr_hot and int(v) < threshold
            ]
            demote_pool.sort(key=lambda k: fd[k], reverse=True)
            promote_pool.sort(key=lambda k: fd[k])
            for demote_key, promote_key in zip(demote_pool[:need], promote_pool[:need]):
                fd[demote_key], fd[promote_key] = fd[promote_key], fd[demote_key]
        elif len(curr_new) > target_new:
            excess = len(curr_new) - target_new
            demote_new_pool = [k for k in curr_new if fd.get(k, 0) >= threshold]
            promote_old_pool = [k for k in (prev_hot - curr_hot) if fd.get(k, 0) < threshold]
            demote_new_pool.sort(key=lambda k: fd[k], reverse=True)
            promote_old_pool.sort(key=lambda k: fd[k])
            for demote_key, promote_key in zip(demote_new_pool[:excess], promote_old_pool[:excess]):
                fd[demote_key], fd[promote_key] = fd[promote_key], fd[demote_key]

        curr_hot_after = _hot_keys(fd, win_n)
        output.append((dtype, fd, win_n))
        prev_hot = curr_hot_after

    return output


def inject_violent_drifts(
    windows: List[Tuple[str, Dict[str, int], int]],
    seed: int,
    drift_probability: float = 0.1,
    replace_fraction: float = 0.25,
    new_key_mass_frac: float = 0.7,
) -> List[Tuple[str, Dict[str, int], int]]:
    """
    Randomly pick windows and create aggressive context drift by:
      - gutting a fraction of the current hot keys,
      - dropping some cold keys,
      - redistributing most freed mass to brand‑new keys (violent churn).
    """
    if drift_probability <= 0 or replace_fraction <= 0:
        return windows

    rng = random.Random(seed)

    # track the largest numeric suffix so we can mint fresh key IDs
    def _max_key_index(freq_dist: Dict[str, int]) -> int:
        mx = 0
        for k in freq_dist:
            if isinstance(k, str) and k.startswith("key_"):
                try:
                    mx = max(mx, int(k.split("_", 1)[1]))
                except ValueError:
                    continue
        return mx

    next_key_id = max((_max_key_index(fd) for _, fd, _ in windows), default=0) + 1
    drifted: List[Tuple[str, Dict[str, int], int]] = []

    for idx, (dtype, freq_dist, win_n) in enumerate(windows):
        # leave the very first window unchanged for a stable baseline
        if idx == 0 or rng.random() >= drift_probability:
            drifted.append((dtype, freq_dist, win_n))
            continue

        fd = dict(freq_dist)
        total_before = sum(fd.values())
        if total_before <= 0 or len(fd) < 2:
            drifted.append((dtype, fd, win_n))
            continue

        sorted_items = sorted(fd.items(), key=lambda kv: kv[1], reverse=True)
        replace_count = max(1, int(len(sorted_items) * replace_fraction))
        hot_cut = max(1, replace_count // 2)

        hot_to_drop = [k for k, _ in sorted_items[:hot_cut]]
        cold_to_drop = [k for k, _ in sorted(fd.items(), key=lambda kv: kv[1])[: replace_count - len(hot_to_drop)]]

        mass_pool = 0
        # cripple current hot keys: keep a tiny residue so they linger as "cold"
        for key in hot_to_drop:
            orig = fd.get(key, 0)
            new_freq = max(1, int(orig * 0.05))
            mass_pool += max(0, orig - new_freq)
            fd[key] = new_freq

        # drop some cold keys outright
        for key in cold_to_drop:
            if key in fd:
                mass_pool += fd.pop(key)

        if mass_pool <= 0:
            drifted.append((dtype, fd, win_n))
            continue

        # Split mass pool between entirely new keys and a few existing cold ones we want to heat up
        new_mass = int(round(mass_pool * max(0.0, min(1.0, new_key_mass_frac))))
        existing_mass = mass_pool - new_mass

        # Warm up a handful of previously cold keys
        if existing_mass > 0 and len(fd) > 0:
            candidates = [k for k in fd if k not in hot_to_drop]
            if candidates:
                pick_count = min(len(candidates), max(1, len(hot_to_drop)))
                chosen = rng.sample(candidates, pick_count)
                weights = [rng.random() for _ in chosen]
                wsum = sum(weights) or 1.0
                remaining = existing_mass
                for i, key in enumerate(chosen):
                    share = remaining if i == len(chosen) - 1 else int(round(existing_mass * (weights[i] / wsum)))
                    share = max(0, min(remaining, share))
                    fd[key] += share
                    remaining -= share
                    if remaining <= 0:
                        break

        # Inject brand-new keys that suddenly become hot
        if new_mass > 0:
            new_keys = max(1, len(hot_to_drop))
            weights = [rng.random() for _ in range(new_keys)]
            wsum = sum(weights) or 1.0
            remaining = new_mass
            for i in range(new_keys):
                share = remaining if i == new_keys - 1 else int(round(new_mass * (weights[i] / wsum)))
                share = max(0, min(remaining, share))
                if share > 0:
                    key = f"key_{next_key_id}"
                    next_key_id += 1
                    fd[key] = share
                    remaining -= share
                if remaining <= 0:
                    break

        # fix any rounding drift to keep totals aligned
        total_after = sum(fd.values())
        delta = total_before - total_after
        if delta != 0:
            target_key = rng.choice(list(fd))
            fd[target_key] = max(1, fd[target_key] + delta)

        drifted.append((dtype, fd, win_n))

    return drifted


def reconstruct_streams(
    windowed_data: Dict[int, Dict[int, Dict[str, int]]],
    seed: int = 42,
    preserve_window_order: bool = True
) -> Dict[int, Dict[int, List[str]]]:
    """
    Reconstructs streams for multiple windows and partitions with reproducible ordering.

    Args:
        windowed_data: Nested dictionary {window: {partition: {key: frequency}}}
        seed: Random seed for reproducibility
        preserve_window_order: If True, maintains window sequence in shuffling

    Returns:
        Dictionary {window: {partition: [ordered_keys]}}
    """
    random.seed(seed)
    reconstructed = defaultdict(dict)

    # Determine window processing order
    windows = sorted(windowed_data.keys()) if preserve_window_order else windowed_data.keys()

    for window in windows:
        partitioned_data = windowed_data[window]

        for partition, key_counts in partitioned_data.items():
            # Create list with keys repeated by their frequencies
            stream = []
            for key, count in key_counts.items():
                stream.extend([key] * count)

            # Shuffle while maintaining reproducibility
            random.shuffle(stream)
            reconstructed[window][partition] = stream

    return dict(reconstructed)


def reconstruct_streams_to_window_files(
    windowed_data: Dict[int, Dict[int, Dict[str, int]]],
    output_base_path: str,
    seed: int = 42,
) -> str:
    """
    Reconstruct and save each window independently to avoid large in-memory output.

    Returns the directory where window files were written.
    """
    base = output_base_path
    if base.endswith(".json.gz"):
        base = base[:-8]
    out_dir = f"{base}_windows"
    os.makedirs(out_dir, exist_ok=True)

    windows = sorted(windowed_data.keys())
    for window in windows:
        random.seed(seed + int(window))
        partitioned_data = windowed_data[window]
        reconstructed_window: Dict[int, List[str]] = {}
        for partition, key_counts in partitioned_data.items():
            stream = []
            for key, count in key_counts.items():
                stream.extend([key] * count)
            random.shuffle(stream)
            reconstructed_window[int(partition)] = stream

        out_file = os.path.join(out_dir, f"window_{int(window):06d}.json.gz")
        with gzip.open(out_file, "wt", encoding="utf-8") as f:
            json.dump(reconstructed_window, f)

    return out_dir


def generate_global_distributions(
    scenario: List[Dict[str, Any]],
    total_items: int,
    num_keys: int,
    n: int,
    seed: int,
    hot_key_drift_pct: float,
    max_rel_freq_delta_pct: float,
    new_key_pct: float,
    newborn_hot_birth_prob: float,
    max_hot_newborn_frac: float,
) -> List[Tuple[str, Dict[str, int], int]]:
    """
    Phase 1: build global per-window key-frequency distributions from scenario.
    """
    print("Generating global distributions for each window...")
    raw_windows, _ = run_scenario(scenario, total_items, num_keys, n)
    print("Smoothing key identities across windows...")
    if hot_key_drift_pct > 0:
        print(f"Applying smoothing with hot-key drift target: {hot_key_drift_pct:.1f}%")
    if new_key_pct > 0:
        print(f"Applying new-key introduction target: {new_key_pct:.1f}% per window")
    if newborn_hot_birth_prob > 0:
        print(
            f"Allowing occasional hot newborns: prob={newborn_hot_birth_prob:.3f}, "
            f"max_frac={max_hot_newborn_frac:.3f}"
        )
    return smooth_key_transitions(
        raw_windows,
        hot_key_drift_pct=hot_key_drift_pct,
        seed=seed,
        max_rel_freq_delta_pct=max_rel_freq_delta_pct,
        new_key_pct=new_key_pct,
        newborn_hot_birth_prob=newborn_hot_birth_prob,
        max_hot_newborn_frac=max_hot_newborn_frac,
    )


def load_real_window_counts(
    counts_path: str,
    n: int,
) -> List[Tuple[str, Dict[str, int], int]]:
    """
    Load normalized real per-window global key counts and return window tuples:
    [(dist_type, freq_dist, n), ...]

    Supported formats:
    - JSONL: {"window": <id>, "counts": {<key>: <count>, ...}}
    - CSV (long): window, src_ip, packet_count

    The original window labels are used only for sorting. Downstream generator
    output is re-indexed to zero-based contiguous IDs to match evaluator input.
    """
    print(f"Loading real window counts: {counts_path}")
    windows: List[Tuple[Any, Dict[str, int]]] = []

    if counts_path.lower().endswith(".jsonl"):
        with open(counts_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "counts" not in row:
                    raise ValueError(f"Missing 'counts' in JSONL line {line_no}")
                win = row.get("window", line_no - 1)
                counts = {str(k): int(v) for k, v in row["counts"].items() if int(v) > 0}
                windows.append((win, counts))
    elif counts_path.lower().endswith(".csv"):
        with open(counts_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            required = {"window", "src_ip", "packet_count"}
            if not required.issubset(set(reader.fieldnames or [])):
                raise ValueError("CSV must contain columns: window, src_ip, packet_count")
            by_window: Dict[str, Dict[str, int]] = defaultdict(dict)
            for row in reader:
                w = row["window"]
                k = row["src_ip"]
                c = int(row["packet_count"])
                if c <= 0:
                    continue
                by_window[w][k] = by_window[w].get(k, 0) + c
            windows = [(w, by_window[w]) for w in by_window]
    else:
        raise ValueError("Unsupported real counts format. Use .jsonl or .csv")

    if not windows:
        return []

    def _window_sort_key(w: Any):
        try:
            return (0, int(w))
        except (TypeError, ValueError):
            return (1, str(w))

    windows.sort(key=lambda x: _window_sort_key(x[0]))
    loaded = [("real", counts, n) for _, counts in windows]
    print(f"Loaded {len(loaded)} windows from real counts.")
    return loaded


def assign_windows_to_partitions(
    windows: List[Tuple[str, Dict[str, int], int]],
    m: int,
    seed: int,
    partitioning_config: dict | None,
    plot_distr: bool,
    save_info: tuple[str | None, str | None],
) -> Dict[int, Dict[int, Dict[str, int]]]:
    """
    Phase 2: assign per-window global frequencies to partitions.

    The returned stream uses contiguous zero-based window IDs regardless of
    whether the source windows came from a synthetic scenario or real counts.
    """

    def _assert_count_preservation(
        window_idx: int,
        freq_dist: Dict[str, int],
        partitioned_window: Dict[int, Dict[str, int]],
    ) -> None:
        expected = {k: int(v) for k, v in freq_dist.items() if int(v) > 0}
        observed: Dict[str, int] = defaultdict(int)
        for part_counts in partitioned_window.values():
            for key, count in part_counts.items():
                observed[key] += int(count)

        expected_total = sum(expected.values())
        observed_total = sum(observed.values())
        bad_keys = []
        for key in set(expected) | set(observed):
            exp = expected.get(key, 0)
            obs = observed.get(key, 0)
            if exp != obs:
                bad_keys.append((key, exp, obs))
                if len(bad_keys) >= 10:
                    break

        if expected_total != observed_total or bad_keys:
            sample = ", ".join(
                f"{key}: expected={exp}, observed={obs}" for key, exp, obs in bad_keys
            )
            raise ValueError(
                f"Partitioning count preservation failed for window {window_idx}: "
                f"global_total={expected_total}, partitioned_total={observed_total}. "
                f"Mismatched keys: {sample or 'none'}"
            )

    partitioning_config = partitioning_config or {}
    policy = partitioning_config.get("policy", "synthetic_mixture")
    validate_partitioning = bool(partitioning_config.get("validate_count_preservation", True))
    mode_probs = tuple(partitioning_config.get("mode_probs", (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)))
    normal_subset_frac_range = tuple(partitioning_config.get("normal_subset_frac_range", (0.5, 1.0)))
    normal_std_frac_range = tuple(partitioning_config.get("normal_std_frac_range", (0.2, 0.6)))
    skew_mass_frac_range = tuple(partitioning_config.get("skew_mass_frac_range", (0.4, 0.8)))
    skew_subset_frac_range = tuple(partitioning_config.get("skew_subset_frac_range", (0.1, 0.3)))
    rest_subset_frac_range = tuple(partitioning_config.get("rest_subset_frac_range", (0.2, 1.0)))

    num_windows = len(windows)
    stream = {w: {} for w in range(num_windows)}
    print(f"Starting generation: windows={num_windows}, partitions={m}, seed={seed}")
    for idx, (dist_type, freq_dist, win_n) in enumerate(windows):
        window_id = idx
        progress = f"{idx + 1}/{num_windows}"
        print(
            f"[window_id={window_id} progress={progress}] "
            f"dist={dist_type}, keys={len(freq_dist)}, n={win_n}"
        )
        if plot_distr:
            plot_frequency_distribution_with_hh(
                freq_dist,
                win_n,
                save_info,
                f"Window {window_id}: {dist_type.capitalize()}",
            )

        print(f"[window_id={window_id} progress={progress}] partitioning stream window...")
        partitioned_window = assign_partitions(
            freq_dist,
            num_partitions=m,
            policy=policy,
            seed=seed,
            window_id=window_id,
            mode_probs=mode_probs,
            normal_subset_frac_range=normal_subset_frac_range,
            normal_std_frac_range=normal_std_frac_range,
            skew_mass_frac_range=skew_mass_frac_range,
            skew_subset_frac_range=skew_subset_frac_range,
            rest_subset_frac_range=rest_subset_frac_range,
            top_n=partitioning_config.get("top_n", win_n),
            skewed_fraction=partitioning_config.get("skewed_fraction", 1.0),
            skew_ratio=partitioning_config.get("skew_ratio", 0.6),
            skew_jitter=partitioning_config.get("skew_jitter", 0.15),
            skew_sigma_frac=partitioning_config.get("skew_sigma_frac", 0.14),
            rest_sigma_frac=partitioning_config.get("rest_sigma_frac", 0.32),
            head_band_frac=partitioning_config.get("head_band_frac", 0.12),
            tail_band_frac=partitioning_config.get("tail_band_frac", 0.45),
            floor_ratio=partitioning_config.get("floor_ratio", 0.03),
            skew_dilute_fraction=partitioning_config.get("skew_dilute_fraction", 0.04),
            rest_dilute_fraction=partitioning_config.get("rest_dilute_fraction", 0.14),
            imbalance=partitioning_config.get("imbalance", 0.0),
            rarity_gamma=partitioning_config.get("rarity_gamma", 1.0),
            rarity_jitter=partitioning_config.get("rarity_jitter", 0.0),
            rarity_midpoint=partitioning_config.get("rarity_midpoint", 0.7),
            rarity_steepness=partitioning_config.get("rarity_steepness", 0.0),
            rare_spread_fraction=partitioning_config.get("rare_spread_fraction", 0.0),
            rare_spread_min_partitions=partitioning_config.get("rare_spread_min_partitions", 2),
            rare_spread_max_partitions=partitioning_config.get("rare_spread_max_partitions", None),
            rarity_mode=partitioning_config.get("rarity_mode", "threshold"),
            distractor_min_ratio=partitioning_config.get("distractor_min_ratio", 0.45),
            distractor_max_ratio=partitioning_config.get("distractor_max_ratio", 0.98),
            distractor_count_ratio=partitioning_config.get("distractor_count_ratio", 1.0),
            distractor_subset_frac_range=tuple(partitioning_config.get("distractor_subset_frac_range", (0.05, 0.15))),
            partition_mass_imbalance=partitioning_config.get("partition_mass_imbalance", 0.25),
            distractor_home_mass_frac=partitioning_config.get("distractor_home_mass_frac", 0.90),
            trap_partition_count=partitioning_config.get("trap_partition_count", 0),
            distractor_trap_prob=partitioning_config.get("distractor_trap_prob", 0.0),
            max_partition_mass_frac=partitioning_config.get("max_partition_mass_frac", None),
            local_detection_threshold_scale=partitioning_config.get("local_detection_threshold_scale", 1.0),
            hh_overflow_spread_power=partitioning_config.get("hh_overflow_spread_power", 1.0),
        )
        if validate_partitioning:
            _assert_count_preservation(window_id, freq_dist, partitioned_window)
        stream[window_id] = partitioned_window
        if plot_distr:
            plot_partition_skew(partitioned_window, f"Window {window_id}", save_info)
    return stream



def prepare_and_store_data(seed=42, total_items=10_000, num_keys = 1_000, n = 100, m=10,
        scenario= [],
        real_counts_path: str | None = None,
        path = 'stream_data.pkl.gz',
        summ_path = 'summ_data.pkl.gz',
        json_gz_path='stream_data.json.gz',
        plot_distr = True,
        save_info: tuple[str | None, str | None] = ('./plots/data', 'png'),
        partitioning_config: dict | None = None,
        hot_key_drift_pct: float = 0.0,
        max_rel_freq_delta_pct: float = 20.0,
        new_key_pct: float = 10.0,
        newborn_hot_birth_prob: float = 0.05,
        max_hot_newborn_frac: float = 0.10,
        expand_streams: bool | None = None,
        save_windows_separately: bool = False,
    ):

    print("Initializing data generator...")
    if real_counts_path:
        global_windows = load_real_window_counts(real_counts_path, n=n)
    else:
        if scenario == []:
            scenario = {'type': 'uniform', 'duration': 2, 'params': {}, 'n': n}
        global_windows = generate_global_distributions(
            scenario=scenario,
            total_items=total_items,
            num_keys=num_keys,
            n=n,
            seed=seed,
            hot_key_drift_pct=hot_key_drift_pct,
            max_rel_freq_delta_pct=max_rel_freq_delta_pct,
            new_key_pct=new_key_pct,
            newborn_hot_birth_prob=newborn_hot_birth_prob,
            max_hot_newborn_frac=max_hot_newborn_frac,
        )
    stream = assign_windows_to_partitions(
        windows=global_windows,
        m=m,
        seed=seed,
        partitioning_config=partitioning_config,
        plot_distr=plot_distr,
        save_info=save_info,
    )

    if expand_streams is None:
        # Real traces can be very large; default to counts-only output to avoid OOM.
        expand_streams = (real_counts_path is None)

    if expand_streams:
        if save_windows_separately and json_gz_path:
            print("Reconstructing per-partition streams (window-wise)...")
            out_dir = reconstruct_streams_to_window_files(stream, json_gz_path, seed=seed)
            print(f"Saved expanded stream windows to: {out_dir}")
        else:
            print("Reconstructing per-partition streams...")
            reconstructed = reconstruct_streams(stream, seed)
            if json_gz_path:
                save_compressed_json(reconstructed, json_gz_path)
                print(f"Saved expanded stream JSON: {json_gz_path}")
    else:
        if json_gz_path:
            save_compressed_json(stream, json_gz_path)
            print(f"Saved partitioned counts JSON: {json_gz_path}")

