"""
Partitioning helpers for generating stress-test key-to-partition layouts.

The main entry point, `assign_partitions`, splits key mass across partitions with
controlled imbalance. Keys inside `top_n` are treated as “common”; keys outside
are “rare” and can be made sharply skewed via the `imbalance` knob.
"""
#     seed: int | None = 0,

#     # Base (flattest) defaults:
#     skew_sigma_frac: float = 0.13,
#     rest_sigma_frac: float = 0.28,
#     head_band_frac: float = 0.12,
#     tail_band_frac: float = 0.38,
#     floor_ratio: float = 0.025,
#     skew_dilute_fraction: float = 0.035,
#     rest_dilute_fraction: float = 0.11,


#    # 0..1 (higher => more imbalanced). With top_n set: applied to keys outside top_n.
#     imbalance: float = 0.0,

#     # Optional shaping knobs (mostly relevant when top_n is None and log-based rarity is used):
#     rarity_gamma: float = 1.0,
#     rarity_jitter: float = 0.0,
#     rarity_midpoint: float = 0.7,
#     rarity_steepness: float = 0.0,
# ) -> Dict[int, Dict[Any, int]]:
#     if num_partitions <= 0:
#         raise ValueError("num_partitions must be > 0")

#     partitioned: Dict[int, Dict[Any, int]] = {p: {} for p in range(num_partitions)}

#     # Precompute "common" set if top_n is provided (rare := outside top_n)
#     top_keys_set = None
#     if top_n is not None:
#         tn = max(0, int(top_n))
#         if tn > 0:
#             keys_sorted = sorted(freq_dist.items(), key=lambda kv: int(kv[1]), reverse=True)
#             top_keys_set = {k for k, _ in keys_sorted[: min(tn, len(keys_sorted))]}
#         else:
#             top_keys_set = set()  # top_n=0 => everyone is "rare"

#     # Fallback log rarity precompute (only used when top_n is None)
#     freqs = [int(v) for v in freq_dist.values() if int(v) > 0]
#     f_min = min(freqs) if freqs else 0
#     f_max = max(freqs) if freqs else 0
#     log_min = math.log(f_min) if f_min > 0 else 0.0
#     log_max = math.log(f_max) if f_max > 0 else 0.0
#     denom = (log_max - log_min)

#     imb_max = max(0.0, min(1.0, float(imbalance)))
#     rarity_gamma = max(1e-9, float(rarity_gamma))
#     rarity_jitter = max(0.0, min(1.0, float(rarity_jitter)))
#     rarity_midpoint = max(0.0, min(1.0, float(rarity_midpoint)))
#     rarity_steepness = max(0.0, float(rarity_steepness))

#     for k, freq in freq_dist.items():
#         freq = int(freq)
#         if freq <= 0:
#             continue

#         rng = random if seed is None else random.Random(_stable_u64(f"{seed}|{k}"))

#         if rng.random() >= skewed_fraction:
#             # Optional uniform mode if skewed_fraction < 1
#             base = freq // num_partitions
#             rem = freq % num_partitions
#             for i in range(num_partitions):
#                 share = base + (1 if i < rem else 0)
#                 if share > 0:
#                     partitioned[i][k] = partitioned[i].get(k, 0) + share
#             continue

#         # --- RARITY (sharp) ---
#         if top_keys_set is not None:
#             # Hard step: rare := outside top_n
#             rarity = 0.0 if k in top_keys_set else 1.0
#         else:
#             # Fallback: log-based rarity (0 = most common, 1 = rarest)
#             if denom <= 0:
#                 rarity = 0.0
#             else:
#                 rarity = (log_max - math.log(freq)) / denom
#                 rarity = max(0.0, min(1.0, rarity))

#             rarity = rarity ** rarity_gamma
#             if rarity_steepness > 0.0:
#                 rarity = _sigmoid(rarity_steepness * (rarity - rarity_midpoint))
#             if rarity_jitter > 0.0:
#                 rarity = max(0.0, min(1.0, rarity + (rng.random() - 0.5) * rarity_jitter))
#         # ----------------------

#         key_imb = imb_max * rarity

#         # Imbalance-derived coefficients
#         shape_scale = 1.0 - 0.75 * key_imb          # 1.0 -> 0.25
#         dilute_scale = (1.0 - key_imb) ** 2         # 1.0 -> 0.0
#         weight_power = 1.0 + 3.0 * key_imb          # 1.0 -> 4.0

#         # Effective knobs (more imbalance => smaller/less tail support)
#         skew_sigma_eff = max(0.02, skew_sigma_frac * shape_scale)
#         rest_sigma_eff = max(0.04, rest_sigma_frac * shape_scale)
#         head_band_eff = max(0.0, head_band_frac * shape_scale)
#         tail_band_eff = max(0.0, tail_band_frac * shape_scale)

#         floor_eff = floor_ratio * dilute_scale
#         skew_dilute_eff = skew_dilute_fraction * dilute_scale
#         rest_dilute_eff = rest_dilute_fraction * dilute_scale

#         skew_ratio_eff = min(0.95, skew_ratio + 0.25 * key_imb)

#         # choose skew subset
#         min_count = max(1, num_partitions // 5)
#         max_count = min(num_partitions, max(min_count, math.ceil(num_partitions / 3)))
#         skew_part_count = rng.randint(min_count, max_count)

#         skew_parts = rng.sample(range(num_partitions), skew_part_count)
#         skew_set = set(skew_parts)
#         rest_parts = [p for p in range(num_partitions) if p not in skew_set]

#         rng.shuffle(skew_parts)
#         rng.shuffle(rest_parts)

#         theta = rng.uniform(skew_ratio_eff - skew_jitter, skew_ratio_eff + skew_jitter)
#         theta = max(0.0, min(1.0, theta))

#         skew_mass = int(round(freq * theta))
#         rest_mass = freq - skew_mass

#         sw = _banded_gaussian_weights(
#             n=len(skew_parts),
#             sigma_frac=skew_sigma_eff,
#             head_band_frac=head_band_eff,
#             tail_band_frac=tail_band_eff,
#             floor_ratio=floor_eff,
#             mu_index=0.0,
#             weight_power=weight_power,
#         )
#         alloc_skew = _allocate_with_dilution(skew_mass, skew_parts, sw, skew_dilute_eff)
#         for p, share in alloc_skew.items():
#             partitioned[p][k] = partitioned[p].get(k, 0) + share

#         targets = rest_parts if rest_parts else skew_parts
#         rw = _banded_gaussian_weights(
#             n=len(targets),
#             sigma_frac=(rest_sigma_eff if rest_parts else skew_sigma_eff),
#             head_band_frac=head_band_eff,
#             tail_band_frac=tail_band_eff,
#             floor_ratio=floor_eff,
#             mu_index=0.0,
#             weight_power=weight_power,
#         )
#         alloc_rest = _allocate_with_dilution(rest_mass, targets, rw, rest_dilute_eff)
#         for p, share in alloc_rest.items():
#             partitioned[p][k] = partitioned[p].get(k, 0) + share

#     return partitioned



# import random
# import math
# import hashlib
# from typing import Dict, Any, List, Tuple


# def _stable_u64(x: str) -> int:
#     return int.from_bytes(hashlib.blake2b(x.encode("utf-8"), digest_size=8).digest(), "big")


# def _zipf_weights(n: int, alpha: float) -> List[float]:
#     """Zipf weights over ranks 1..n: w_r ∝ 1/r^alpha (alpha>0 => more imbalance)."""
#     if n <= 0:
#         return []
#     a = max(1e-9, float(alpha))
#     w = [1.0 / ((i + 1) ** a) for i in range(n)]
#     s = sum(w)
#     return [wi / s for wi in w] if s > 0 else [1.0 / n] * n


# def _allocate_integer_mass(total: int, parts: List[int], weights: List[float]) -> Dict[int, int]:
#     """Deterministic integer allocation proportional to weights (largest-fraction remainder)."""
#     if total <= 0 or not parts:
#         return {}
#     if len(parts) != len(weights):
#         raise ValueError("parts and weights must have the same length")

#     wsum = sum(weights)
#     if wsum <= 0:
#         weights = [1.0] * len(parts)
#         wsum = float(len(parts))

#     raw = [total * (w / wsum) for w in weights]
#     base = [int(math.floor(x)) for x in raw]
#     rem = total - sum(base)

#     fracs = [(raw[i] - base[i], i) for i in range(len(parts))]
#     fracs.sort(reverse=True)
#     for _, i in fracs[:rem]:
#         base[i] += 1

#     return {parts[i]: base[i] for i in range(len(parts)) if base[i] > 0}


# def _banded_gaussian_weights(
#     n: int,
#     sigma_frac: float,
#     head_band_frac: float,
#     tail_band_frac: float,
#     floor_ratio: float,
#     mu_index: float = 0.0,      # 0.0 => peak at the "head" (index 0)
#     weight_power: float = 1.0,  # >1 => sharper / more imbalanced
# ) -> List[float]:
#     """Gaussian-ish weights with flat head/tail bands + floor + optional sharpening."""
#     if n <= 0:
#         return []
#     if n == 1:
#         return [1.0]

#     sigma = max(1e-9, float(sigma_frac) * (n - 1))
#     mu = float(mu_index)

#     pdf = [math.exp(-0.5 * ((i - mu) / sigma) ** 2) for i in range(n)]
#     mx = max(pdf) if pdf else 1.0
#     floor = max(0.0, float(floor_ratio)) * mx
#     w = [p + floor for p in pdf]

#     hb = max(0, min(n, int(round(max(0.0, float(head_band_frac)) * n))))
#     tb = max(0, min(n - hb, int(round(max(0.0, float(tail_band_frac)) * n))))

#     if hb > 0:
#         head_mean = sum(w[:hb]) / hb
#         for i in range(hb):
#             w[i] = head_mean

#     if tb > 0:
#         tail_mean = sum(w[-tb:]) / tb
#         for i in range(n - tb, n):
#             w[i] = tail_mean

#     if weight_power and weight_power != 1.0:
#         wp = max(1.0, float(weight_power))
#         w = [wi ** wp for wi in w]

#     return w


# def _allocate_with_dilution(total: int, parts: List[int], weights: List[float], dilute_fraction: float) -> Dict[int, int]:
#     """
#     Allocate most mass by weights, reserve dilute_fraction to spread uniformly
#     (helps keep low-concentration presence across more partitions when possible).
#     """
#     if total <= 0 or not parts:
#         return {}
#     df = max(0.0, min(1.0, float(dilute_fraction)))
#     dilute = int(round(total * df))
#     main = total - dilute

#     out: Dict[int, int] = {}
#     if main > 0:
#         a = _allocate_integer_mass(main, parts, weights)
#         for p, s in a.items():
#             out[p] = out.get(p, 0) + s

#     if dilute > 0:
#         a = _allocate_integer_mass(dilute, parts, [1.0] * len(parts))
#         for p, s in a.items():
#             out[p] = out.get(p, 0) + s

#     return out


# def _sorted_keys_by_freq(freq_dist: Dict[Any, int]) -> List[Tuple[Any, int]]:
#     """Deterministic ordering: by descending freq, then stable hash of key string."""
#     items = [(k, int(v)) for k, v in freq_dist.items() if int(v) > 0]
#     items.sort(key=lambda kv: (-kv[1], _stable_u64(str(kv[0]))))
#     return items


# def assign_partitions(
#     freq_dist: Dict[Any, int],
#     num_partitions: int,
#     top_n: int | None = None,          # rare := outside top_n (if provided)
#     skewed_fraction: float = 1.0,      # if <1, some keys go uniform
#     seed: int | None = 0,

#     # --- Common-keys (top_n) shaping (kept fairly flat by default) ---
#     skew_sigma_frac: float = 0.14,
#     head_band_frac: float = 0.12,
#     tail_band_frac: float = 0.45,
#     floor_ratio: float = 0.03,
#     common_dilute_fraction: float = 0.10,

#     # --- Make “random partitioning” assumptions weak (global + rare-key controls) ---
#     stress_test: bool = True,

#     # Global partition size skew: higher => a few partitions dominate window mass
#     partition_size_alpha: float = 1.4,       # Zipf alpha over partitions
#     common_popularity_power: float = 1.6,    # >1 pushes common mass to popular partitions

#     # Rare keys (outside top_n): home-shard + haze
#     rare_home_k: int = 1,                   # 1–2 recommended
#     rare_home_mass_frac: float = 0.98,      # 0.95–0.995
#     rare_home_sigma_frac: float = 0.06,     # smaller => more concentrated within home set
#     rare_home_weight_power: float = 4.0,    # >1 => very concentrated in first home partition

#     rare_haze_partitions_frac: float = 0.4, # fraction of partitions to receive haze (excluding home)
#     rare_haze_min_partitions: int = 2,
#     rare_haze_dilute_fraction: float = 0.0, # extra uniform dilution inside haze targets (optional)

#     # Prefer putting rare reporters in small/unpopular partitions (reduces “coverage”)
#     prefer_small_for_rare: bool = True,
#     home_popularity_beta: float = 2.0,      # >0 penalizes popular partitions when choosing rare homes
#     haze_popularity_beta: float = 1.0,      # >0 penalizes popular partitions when choosing haze targets

#     # Temporal churn: rotate rare homes for some fraction (needs window_id)
#     rare_churn_fraction: float = 0.0,       # e.g. 0.3
#     window_id: int | None = 0,              # set per window if using churn
# ) -> Dict[int, Dict[Any, int]]:
#     """
#     Stress-test partitioning generator to make simple partitioning assumptions very weak.

#     - If top_n is provided: keys outside top_n are treated as “rare” and get home-sharded.
#     - Common keys are biased toward “popular” partitions to create highly imbalanced partition sizes.
#     - Rare keys are biased toward small/unpopular partitions to minimize cross-partition coverage.
#     """
#     if num_partitions <= 0:
#         raise ValueError("num_partitions must be > 0")

#     # Output and live load tracking (used to steer rare keys into smaller partitions)
#     partitioned: Dict[int, Dict[Any, int]] = {p: {} for p in range(num_partitions)}
#     load = [0] * num_partitions

#     items = _sorted_keys_by_freq(freq_dist)

#     # Define “common” (top_n) set if requested
#     top_keys_set = None
#     if top_n is not None:
#         tn = max(0, int(top_n))
#         top_keys_set = {k for k, _ in items[: min(tn, len(items))]}

#     # Partition popularity weights (Zipf), mapped to partition IDs with a stable per-window permutation
#     global_rng = random if seed is None else random.Random(_stable_u64(f"{seed}|GLOBAL|{window_id}"))
#     perm = list(range(num_partitions))
#     global_rng.shuffle(perm)

#     # popularity_by_rank[0] is most popular
#     popularity_by_rank = _zipf_weights(num_partitions, partition_size_alpha)

#     # popularity[p] assigned via permuted ranks
#     popularity = [0.0] * num_partitions
#     for rank, p in enumerate(perm):
#         popularity[p] = popularity_by_rank[rank]

#     def add_mass(p: int, k: Any, c: int) -> None:
#         if c <= 0:
#             return
#         partitioned[p][k] = partitioned[p].get(k, 0) + c
#         load[p] += c

#     def pick_k_smallest(candidates: List[int], k: int, beta: float, rng: random.Random) -> List[int]:
#         """
#         Pick k partitions with smallest score:
#           score = (load+1) * (popularity ** beta)
#         Smaller => less loaded AND less popular.
#         """
#         if k <= 0 or not candidates:
#             return []
#         beta = max(0.0, float(beta))
#         scored = []
#         for p in candidates:
#             score = (load[p] + 1) * ((popularity[p] ** beta) if beta > 0 else 1.0)
#             scored.append((score, _stable_u64(f"{p}|{rng.random()}"), p))  # tie-break w/ deterministic-ish jitter
#         scored.sort()
#         return [p for _, _, p in scored[: min(k, len(scored))]]

#     for k, freq in items:
#         if freq <= 0:
#             continue

#         rng = random if seed is None else random.Random(_stable_u64(f"{seed}|{k}"))

#         if rng.random() >= float(skewed_fraction):
#             # Uniform fallback
#             base = freq // num_partitions
#             rem = freq % num_partitions
#             for i in range(num_partitions):
#                 add_mass(i, k, base + (1 if i < rem else 0))
#             continue

#         is_rare = (top_keys_set is not None and k not in top_keys_set)

#         # ---- RARE KEYS: home-shard + haze ----
#         if stress_test and is_rare:
#             # Optional churn decision (stable per key)
#             churn = False
#             if rare_churn_fraction > 0.0 and window_id is not None:
#                 churn_rng = random if seed is None else random.Random(_stable_u64(f"{seed}|CHURN|{k}"))
#                 churn = (churn_rng.random() < float(rare_churn_fraction))

#             # Per-key home RNG (window-specific if churned)
#             home_tag = window_id if churn else 0
#             home_rng = random if seed is None else random.Random(_stable_u64(f"{seed}|HOME|{k}|{home_tag}"))

#             # Choose home partitions (prefer small/unpopular by default)
#             hk = max(1, min(int(rare_home_k), num_partitions))
#             candidates = list(range(num_partitions))
#             if prefer_small_for_rare:
#                 homes = pick_k_smallest(candidates, hk, home_popularity_beta, home_rng)
#             else:
#                 homes = home_rng.sample(candidates, hk)
#             home_set = set(homes)

#             # Home mass
#             home_mass = int(round(freq * max(0.0, min(1.0, float(rare_home_mass_frac)))))
#             home_mass = max(0, min(freq, home_mass))
#             haze_mass = freq - home_mass

#             # Allocate home mass (non-even, very concentrated)
#             # Order homes deterministically (so “head” is consistent) by increasing (load, popularity)
#             homes_sorted = sorted(homes, key=lambda p: (load[p], popularity[p], p))
#             hw = _banded_gaussian_weights(
#                 n=len(homes_sorted),
#                 sigma_frac=rare_home_sigma_frac,
#                 head_band_frac=0.0,
#                 tail_band_frac=0.0,
#                 floor_ratio=0.0,
#                 mu_index=0.0,
#                 weight_power=rare_home_weight_power,
#             )
#             alloc_home = _allocate_integer_mass(home_mass, homes_sorted, hw)
#             for p, c in alloc_home.items():
#                 add_mass(p, k, c)

#             if haze_mass > 0:
#                 # Choose haze targets (exclude homes by default)
#                 remaining = [p for p in range(num_partitions) if p not in home_set]
#                 if not remaining:
#                     remaining = homes_sorted

#                 haze_count = int(round(num_partitions * float(rare_haze_partitions_frac)))
#                 haze_count = max(int(rare_haze_min_partitions), haze_count)
#                 haze_count = min(len(remaining), haze_count)

#                 if prefer_small_for_rare:
#                     haze_targets = pick_k_smallest(remaining, haze_count, haze_popularity_beta, home_rng)
#                 else:
#                     haze_targets = home_rng.sample(remaining, haze_count)

#                 # Allocate haze fairly flat across haze_targets (optionally with a tiny extra dilution)
#                 # Flat weights keep it “low-dose” across many targets.
#                 hz_w = [1.0] * len(haze_targets)
#                 alloc_haze = _allocate_with_dilution(haze_mass, haze_targets, hz_w, rare_haze_dilute_fraction)
#                 for p, c in alloc_haze.items():
#                     add_mass(p, k, c)

#             continue

#         # ---- COMMON KEYS (top_n) OR non-stress-test mode ----
#         # Make partition sizes imbalanced by biasing common mass toward “popular” partitions.
#         # We do this by ordering partitions by popularity and applying a banded gaussian profile,
#         # then multiplying by popularity^power to strengthen “rich get richer”.
#         parts_by_pop = sorted(range(num_partitions), key=lambda p: popularity[p], reverse=True)

#         gw = _banded_gaussian_weights(
#             n=num_partitions,
#             sigma_frac=skew_sigma_frac,
#             head_band_frac=head_band_frac,
#             tail_band_frac=tail_band_frac,
#             floor_ratio=floor_ratio,
#             mu_index=0.0,
#             weight_power=1.0,
#         )

#         # strengthen bias toward popular partitions
#         pow_ = max(0.0, float(common_popularity_power))
#         blended = []
#         for i, p in enumerate(parts_by_pop):
#             blended.append(gw[i] * (popularity[p] ** pow_ if pow_ > 0 else 1.0))

#         alloc = _allocate_with_dilution(freq, parts_by_pop, blended, common_dilute_fraction)
#         for p, c in alloc.items():
#             add_mass(p, k, c)

#     return partitioned

import random
import math
import hashlib
from typing import Dict, Any, List


def _stable_u64(x: str) -> int:
    return int.from_bytes(hashlib.blake2b(x.encode("utf-8"), digest_size=8).digest(), "big")


def _sigmoid(z: float) -> float:
    if z >= 60.0:
        return 1.0
    if z <= -60.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def _allocate_integer_mass(total: int, parts: List[int], weights: List[float]) -> Dict[int, int]:
    if total <= 0 or not parts:
        return {}
    if len(parts) != len(weights):
        raise ValueError("parts and weights must have the same length")

    wsum = sum(weights)
    if wsum <= 0:
        weights = [1.0] * len(parts)
        wsum = float(len(parts))

    raw = [total * (w / wsum) for w in weights]
    base = [int(math.floor(x)) for x in raw]
    rem = total - sum(base)

    fracs = [(raw[i] - base[i], i) for i in range(len(parts))]
    fracs.sort(reverse=True)
    for _, i in fracs[:rem]:
        base[i] += 1

    return {parts[i]: base[i] for i in range(len(parts)) if base[i] > 0}


def _allocate_integer_mass_with_caps(
    total: int,
    parts: List[int],
    weights: List[float],
    caps: List[int],
    relax_caps: bool = True,
) -> Dict[int, int]:
    """
    Integer allocation with per-part upper caps.
    If caps are feasible (sum(caps) >= total), allocation respects them.
    """
    if total <= 0 or not parts:
        return {}
    if len(parts) != len(weights) or len(parts) != len(caps):
        raise ValueError("parts, weights and caps must have the same length")

    caps = [max(0, int(c)) for c in caps]
    if sum(caps) < total and relax_caps:
        # Keep function total-preserving by relaxing caps minimally.
        deficit = total - sum(caps)
        order = sorted(range(len(parts)), key=lambda i: weights[i], reverse=True)
        idx = 0
        while deficit > 0 and order:
            j = order[idx % len(order)]
            caps[j] += 1
            deficit -= 1
            idx += 1

    alloc = [0] * len(parts)
    remaining = total
    active = [i for i in range(len(parts)) if caps[i] > 0]
    while remaining > 0 and active:
        p_sub = [parts[i] for i in active]
        w_sub = [weights[i] for i in active]
        step = _allocate_integer_mass(remaining, p_sub, w_sub)
        gave_any = False
        overflow = 0
        for j, idx in enumerate(active):
            share = int(step.get(p_sub[j], 0))
            room = caps[idx] - alloc[idx]
            if room <= 0 or share <= 0:
                continue
            take = min(room, share)
            alloc[idx] += take
            gave_any = gave_any or (take > 0)
            overflow += share - take
        remaining = overflow
        active = [i for i in active if alloc[i] < caps[i]]
        if remaining > 0 and not gave_any:
            # Defensive fallback for degenerate weighting/cap combinations.
            for idx in active:
                if remaining <= 0:
                    break
                room = caps[idx] - alloc[idx]
                if room <= 0:
                    continue
                take = min(room, remaining)
                alloc[idx] += take
                remaining -= take
            if remaining > 0:
                break

    return {parts[i]: alloc[i] for i in range(len(parts)) if alloc[i] > 0}


def _solve_hidden_maxflow(
    heavy_items: List[tuple[Any, int]],
    partition_sizes: List[int],
    hh_n: int,
    local_threshold_scale: float = 1.0,
) -> Dict[Any, Dict[int, int]]:
    """
    Stage-1 hidden allocation via exact max-flow on integer capacities:
      source -> item_i (cap c_i)
      item_i -> part_p (cap floor(s_p / n))
      part_p -> sink (cap s_p)
    """
    if not heavy_items:
        return {}
    m = len(partition_sizes)
    if m <= 0:
        return {k: {} for k, _ in heavy_items}

    item_caps = [max(0, int(c)) for _, c in heavy_items]
    scale = max(0.0, float(local_threshold_scale))
    up = [max(0, int(math.floor(scale * partition_sizes[p] / float(max(1, hh_n))))) for p in range(m)]
    part_caps = [max(0, int(partition_sizes[p])) for p in range(m)]

    n_items = len(heavy_items)
    src = 0
    item_off = 1
    part_off = item_off + n_items
    sink = part_off + m
    n_nodes = sink + 1

    # Dinic graph
    to: List[List[int]] = [[] for _ in range(n_nodes)]
    cap: List[Dict[int, int]] = [dict() for _ in range(n_nodes)]
    init_item_to_part: Dict[tuple[int, int], int] = {}

    def add_edge(u: int, v: int, c: int) -> None:
        if c <= 0:
            return
        if v not in cap[u]:
            to[u].append(v)
            cap[u][v] = 0
        if u not in cap[v]:
            to[v].append(u)
            cap[v][u] = 0
        cap[u][v] += int(c)

    # source -> items
    for i, c in enumerate(item_caps):
        add_edge(src, item_off + i, c)

    # items -> partitions (uniform per-item local hidden cap)
    for i in range(n_items):
        u = item_off + i
        for p in range(m):
            v = part_off + p
            c = up[p]
            if c > 0:
                add_edge(u, v, c)
                init_item_to_part[(u, v)] = c

    # partitions -> sink
    for p, c in enumerate(part_caps):
        add_edge(part_off + p, sink, c)

    level = [-1] * n_nodes
    it = [0] * n_nodes

    def bfs() -> bool:
        for i in range(n_nodes):
            level[i] = -1
        q = [src]
        level[src] = 0
        qi = 0
        while qi < len(q):
            u = q[qi]
            qi += 1
            for v in to[u]:
                if level[v] < 0 and cap[u].get(v, 0) > 0:
                    level[v] = level[u] + 1
                    q.append(v)
        return level[sink] >= 0

    def dfs(u: int, f: int) -> int:
        if u == sink:
            return f
        while it[u] < len(to[u]):
            v = to[u][it[u]]
            if level[v] == level[u] + 1 and cap[u].get(v, 0) > 0:
                pushed = dfs(v, min(f, cap[u][v]))
                if pushed > 0:
                    cap[u][v] -= pushed
                    cap[v][u] += pushed
                    return pushed
            it[u] += 1
        return 0

    while bfs():
        for i in range(n_nodes):
            it[i] = 0
        while True:
            pushed = dfs(src, 1 << 60)
            if pushed <= 0:
                break

    # Recover flow on item->partition edges.
    out: Dict[Any, Dict[int, int]] = {k: {} for k, _ in heavy_items}
    for i, (k, _) in enumerate(heavy_items):
        u = item_off + i
        for p in range(m):
            v = part_off + p
            init_c = init_item_to_part.get((u, v), 0)
            if init_c <= 0:
                continue
            resid = cap[u].get(v, 0)
            flow_uv = init_c - resid
            if flow_uv > 0:
                out[k][p] = int(flow_uv)
    return out


def _banded_gaussian_weights(
    n: int,
    sigma_frac: float,
    head_band_frac: float,
    tail_band_frac: float,
    floor_ratio: float,
    mu_index: float = 0.0,      # 0.0 => peak at the "head" (index 0)
    weight_power: float = 1.0,  # >1 => sharper / more imbalanced
) -> List[float]:
    if n <= 0:
        return []
    if n == 1:
        return [1.0]

    sigma = max(1e-9, sigma_frac * (n - 1))
    mu = mu_index

    pdf = [math.exp(-0.5 * ((i - mu) / sigma) ** 2) for i in range(n)]
    mx = max(pdf) if pdf else 1.0
    floor = max(0.0, floor_ratio) * mx
    w = [p + floor for p in pdf]

    hb = max(0, min(n, int(round(max(0.0, head_band_frac) * n))))
    tb = max(0, min(n - hb, int(round(max(0.0, tail_band_frac) * n))))

    if hb > 0:
        head_mean = sum(w[:hb]) / hb
        for i in range(hb):
            w[i] = head_mean

    if tb > 0:
        tail_mean = sum(w[-tb:]) / tb
        for i in range(n - tb, n):
            w[i] = tail_mean

    if weight_power and weight_power != 1.0:
        wp = max(1.0, float(weight_power))
        w = [wi ** wp for wi in w]

    return w


def _allocate_with_dilution(
    total: int,
    parts: List[int],
    weights: List[float],
    dilute_fraction: float,
) -> Dict[int, int]:
    if total <= 0 or not parts:
        return {}
    dilute_fraction = max(0.0, min(1.0, float(dilute_fraction)))

    dilute = int(round(total * dilute_fraction))
    main = total - dilute

    out: Dict[int, int] = {}

    if main > 0:
        a = _allocate_integer_mass(main, parts, weights)
        for p, s in a.items():
            out[p] = out.get(p, 0) + s

    if dilute > 0:
        uniform_w = [1.0] * len(parts)
        a = _allocate_integer_mass(dilute, parts, uniform_w)
        for p, s in a.items():
            out[p] = out.get(p, 0) + s

    return out


def _normal_weights(
    n: int,
    rng: random.Random,
    std_frac_range: tuple[float, float],
) -> List[float]:
    if n <= 0:
        return []
    if n == 1:
        return [1.0]
    mu = rng.uniform(0, n - 1)
    std = rng.uniform(std_frac_range[0], std_frac_range[1]) * max(1, n - 1)
    std = max(0.5, std)
    return [math.exp(-0.5 * ((i - mu) / std) ** 2) for i in range(n)]


def _choose_partitions(
    rng: random.Random,
    parts: List[int],
    min_frac: float,
    max_frac: float,
) -> List[int]:
    if not parts:
        return []
    n = len(parts)
    min_count = max(1, int(math.ceil(min_frac * n)))
    max_count = max(min_count, int(math.ceil(max_frac * n)))
    count = rng.randint(min_count, min(max_count, n))
    return rng.sample(parts, count)


def assign_partitions(
    freq_dist: Dict[Any, int],
    num_partitions: int,
    *,
    policy: str = "synthetic_mixture",
    seed: int | None = 0,
    window_id: int | None = None,
    # synthetic_mixture policy params
    mode_probs: tuple[float, float, float] = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    normal_subset_frac_range: tuple[float, float] = (0.5, 1.0),
    normal_std_frac_range: tuple[float, float] = (0.2, 0.6),
    skew_mass_frac_range: tuple[float, float] = (0.4, 0.8),
    skew_subset_frac_range: tuple[float, float] = (0.1, 0.3),
    rest_subset_frac_range: tuple[float, float] = (0.2, 1.0),
    # imbalance policy params
    top_n: int | None = None,
    skewed_fraction: float = 1.0,
    skew_ratio: float = 0.6,
    skew_jitter: float = 0.15,
    skew_sigma_frac: float = 0.14,
    rest_sigma_frac: float = 0.32,
    head_band_frac: float = 0.12,
    tail_band_frac: float = 0.45,
    floor_ratio: float = 0.03,
    skew_dilute_fraction: float = 0.04,
    rest_dilute_fraction: float = 0.14,
    imbalance: float = 0.0,
    rarity_gamma: float = 1.0,
    rarity_jitter: float = 0.0,
    rarity_midpoint: float = 0.7,
    rarity_steepness: float = 0.0,
    rare_spread_fraction: float = 0.0,
    rare_spread_min_partitions: int = 2,
    rare_spread_max_partitions: int | None = None,
    rarity_mode: str = "threshold",
    distractor_min_ratio: float = 0.45,
    distractor_max_ratio: float = 0.98,
    distractor_count_ratio: float = 1.0,
    distractor_subset_frac_range: tuple[float, float] = (0.05, 0.15),
    partition_mass_imbalance: float = 0.25,
    distractor_home_mass_frac: float = 0.90,
    trap_partition_count: int = 0,
    distractor_trap_prob: float = 0.0,
    max_partition_mass_frac: float | None = None,
    local_detection_threshold_scale: float = 1.0,
    hh_overflow_spread_power: float = 1.0,
) -> Dict[int, Dict[Any, int]]:
    """
    Partition keys across partitions using the selected policy.

    policy="round_robin": simple even split of each key across partitions.
    policy="synthetic_mixture": per-key stochastic mix (uniform / normal / skewed).
    policy="imbalance": rarity-aware skewing based on top_n threshold and imbalance knob.
    """
    if policy == "round_robin":
        return _assign_partitions_round_robin(
            freq_dist=freq_dist,
            num_partitions=num_partitions,
            seed=seed,
            window_id=window_id,
        )
    if policy == "synthetic_mixture":
        return _assign_partitions_synthetic_mixture(
            freq_dist=freq_dist,
            num_partitions=num_partitions,
            seed=seed,
            mode_probs=mode_probs,
            normal_subset_frac_range=normal_subset_frac_range,
            normal_std_frac_range=normal_std_frac_range,
            skew_mass_frac_range=skew_mass_frac_range,
            skew_subset_frac_range=skew_subset_frac_range,
            rest_subset_frac_range=rest_subset_frac_range,
        )
    if policy == "imbalance":
        return _assign_partitions_imbalance(
            freq_dist=freq_dist,
            num_partitions=num_partitions,
            top_n=top_n,
            skewed_fraction=skewed_fraction,
            skew_ratio=skew_ratio,
            skew_jitter=skew_jitter,
            seed=seed,
            skew_sigma_frac=skew_sigma_frac,
            rest_sigma_frac=rest_sigma_frac,
            head_band_frac=head_band_frac,
            tail_band_frac=tail_band_frac,
            floor_ratio=floor_ratio,
            skew_dilute_fraction=skew_dilute_fraction,
            rest_dilute_fraction=rest_dilute_fraction,
            imbalance=imbalance,
            rarity_gamma=rarity_gamma,
            rarity_jitter=rarity_jitter,
            rarity_midpoint=rarity_midpoint,
            rarity_steepness=rarity_steepness,
            rare_spread_fraction=rare_spread_fraction,
            rare_spread_min_partitions=rare_spread_min_partitions,
            rare_spread_max_partitions=rare_spread_max_partitions,
            rarity_mode=rarity_mode,
        )
    if policy in {"hh_hidden_optimal", "flow_hidden", "mass_hidden"}:
        return _assign_partitions_hh_hidden_optimal(
            freq_dist=freq_dist,
            num_partitions=num_partitions,
            seed=seed,
            window_id=window_id,
            top_n=top_n,
            distractor_min_ratio=distractor_min_ratio,
            distractor_max_ratio=distractor_max_ratio,
            distractor_count_ratio=distractor_count_ratio,
            distractor_subset_frac_range=distractor_subset_frac_range,
            distractor_home_mass_frac=distractor_home_mass_frac,
            partition_mass_imbalance=partition_mass_imbalance,
            trap_partition_count=trap_partition_count,
            distractor_trap_prob=distractor_trap_prob,
            max_partition_mass_frac=max_partition_mass_frac,
            local_detection_threshold_scale=local_detection_threshold_scale,
            hh_overflow_spread_power=hh_overflow_spread_power,
        )
    if policy in {"hh_hidden_exact", "flow_hidden_exact"}:
        return _assign_partitions_hh_hidden_optimal(
            freq_dist=freq_dist,
            num_partitions=num_partitions,
            seed=seed,
            window_id=window_id,
            top_n=top_n,
            distractor_min_ratio=distractor_min_ratio,
            distractor_max_ratio=distractor_max_ratio,
            distractor_count_ratio=0.0,
            distractor_subset_frac_range=distractor_subset_frac_range,
            distractor_home_mass_frac=distractor_home_mass_frac,
            partition_mass_imbalance=partition_mass_imbalance,
            trap_partition_count=trap_partition_count,
            distractor_trap_prob=0.0,
            max_partition_mass_frac=max_partition_mass_frac,
            local_detection_threshold_scale=local_detection_threshold_scale,
            hh_overflow_spread_power=hh_overflow_spread_power,
        )
    raise ValueError(f"Unknown partitioning policy: {policy}")


def _assign_partitions_hh_hidden_optimal(
    freq_dist: Dict[Any, int],
    num_partitions: int,
    *,
    seed: int | None = 0,
    window_id: int | None = None,
    top_n: int | None = None,
    distractor_min_ratio: float = 0.55,
    distractor_max_ratio: float = 0.995,
    distractor_count_ratio: float = 3.0,
    distractor_subset_frac_range: tuple[float, float] = (0.03, 0.08),
    distractor_home_mass_frac: float = 0.9,
    partition_mass_imbalance: float = 0.25,
    trap_partition_count: int = 6,
    distractor_trap_prob: float = 0.85,
    max_partition_mass_frac: float | None = 0.04,
    local_detection_threshold_scale: float = 1.0,
    hh_overflow_spread_power: float = 1.0,
) -> Dict[int, Dict[Any, int]]:
    """
    Fixed-global-count stress-test partitioning in three stages:
      1) maximize hidden HH mass below local 1/n thresholds
      2) spread unavoidable HH overflow across available partitions
      3) fill remaining capacity with non-HHs (with correlated distractors)
    """
    if num_partitions <= 0:
        raise ValueError("num_partitions must be > 0")

    items = [(k, int(v)) for k, v in freq_dist.items() if int(v) > 0]
    partitioned: Dict[int, Dict[Any, int]] = {p: {} for p in range(num_partitions)}
    if not items:
        return partitioned

    items.sort(key=lambda kv: kv[1], reverse=True)
    total_mass = sum(v for _, v in items)
    if total_mass <= 0:
        return partitioned

    window_tag = 0 if window_id is None else int(window_id)
    hh_n = int(top_n) if top_n is not None and int(top_n) > 0 else max(1, num_partitions)
    threshold_count = total_mass / float(hh_n)
    parts = list(range(num_partitions))

    # --- Build fixed partition sizes s_p for the window (sum s_p = T) ---
    g_rng = random if seed is None else random.Random(_stable_u64(f"{seed}|HHO|GLOBAL|{window_tag}"))
    imb = max(0.0, float(partition_mass_imbalance))
    raw_w = [math.exp(g_rng.uniform(-imb, imb)) for _ in range(num_partitions)]
    s_map = _allocate_integer_mass(total_mass, parts, raw_w)
    s = [int(s_map.get(p, 0)) for p in parts]

    # Hard anti-collapse on target partition sizes.
    if max_partition_mass_frac is not None:
        cap = int(math.ceil(total_mass * max(0.0, min(1.0, float(max_partition_mass_frac)))))
        if cap > 0:
            over = [p for p in parts if s[p] > cap]
            under = [p for p in parts if s[p] < cap]
            while over and under:
                src = max(over, key=lambda p: s[p])
                dst = min(under, key=lambda p: s[p])
                move = min(s[src] - cap, cap - s[dst])
                if move <= 0:
                    break
                s[src] -= move
                s[dst] += move
                over = [p for p in parts if s[p] > cap]
                under = [p for p in parts if s[p] < cap]

    rem_cap = {p: int(s[p]) for p in parts}

    # Identify heavy hitters with fixed global counts.
    heavy_items = [(k, c) for k, c in items if float(c) > threshold_count]
    non_heavy_items = [(k, c) for k, c in items if float(c) <= threshold_count]

    # --- Stage 1: hidden allocation for HHs (maximize mass <= local thresholds) ---
    # Exact max-flow stage from the formal construction.
    hidden_flow = _solve_hidden_maxflow(
        heavy_items,
        s,
        hh_n,
        local_threshold_scale=local_detection_threshold_scale,
    )
    hh_overflow: Dict[Any, int] = {}
    for k, c in heavy_items:
        hidden_assigned = 0
        for p, v in hidden_flow.get(k, {}).items():
            if v <= 0:
                continue
            partitioned[p][k] = partitioned[p].get(k, 0) + v
            rem_cap[p] -= v
            hidden_assigned += v
        hh_overflow[k] = c - hidden_assigned

    # Trap partitions (for correlated non-random concentration).
    trap_k = max(1, min(int(trap_partition_count), num_partitions))
    trap_parts = sorted(parts, key=lambda p: s[p])[:trap_k]

    # --- Stage 2: spread unavoidable HH overflow ---
    spread_power = max(0.0, float(hh_overflow_spread_power))
    for k, ov in sorted(hh_overflow.items(), key=lambda kv: kv[1], reverse=True):
        rem = int(ov)
        if rem <= 0:
            continue
        candidate_parts = [p for p in parts if rem_cap[p] > 0]
        weights = [(float(rem_cap[p]) ** spread_power if spread_power > 0 else 1.0) for p in candidate_parts]
        caps = [rem_cap[p] for p in candidate_parts]
        alloc = _allocate_integer_mass_with_caps(rem, candidate_parts, weights, caps, relax_caps=False)
        for p, take in alloc.items():
            if take <= 0:
                continue
            partitioned[p][k] = partitioned[p].get(k, 0) + take
            rem_cap[p] -= take

    # Build distractor set from near-threshold non-HHs.
    d_min = max(0.0, float(distractor_min_ratio)) * threshold_count
    d_max = max(d_min, float(distractor_max_ratio) * threshold_count)
    d_pool = [(k, c) for k, c in non_heavy_items if d_min <= float(c) < d_max]
    d_pool.sort(key=lambda kv: kv[1], reverse=True)
    target_d = int(round(len(heavy_items) * max(0.0, float(distractor_count_ratio))))
    target_d = max(0, min(target_d, len(d_pool)))
    distractor_keys = {k for k, _ in d_pool[:target_d]}

    # --- Stage 3: fill remaining capacity with non-HHs ---
    n = len(parts)
    sub_lo = max(1, int(math.ceil(max(0.0, distractor_subset_frac_range[0]) * n)))
    sub_hi = max(sub_lo, int(math.ceil(max(distractor_subset_frac_range[0], distractor_subset_frac_range[1]) * n)))
    home_frac = max(0.0, min(1.0, float(distractor_home_mass_frac)))

    for k, c in non_heavy_items:
        rem = int(c)
        if rem <= 0:
            continue
        krng = random if seed is None else random.Random(_stable_u64(f"{seed}|HHO|NH|{window_tag}|{k}"))

        if k in distractor_keys:
            use_trap = bool(trap_parts) and (krng.random() < max(0.0, min(1.0, float(distractor_trap_prob))))
            home = krng.choice(trap_parts if use_trap else parts)
            home_take = min(rem, rem_cap.get(home, 0), int(round(rem * home_frac)))
            if home_take > 0:
                partitioned[home][k] = partitioned[home].get(k, 0) + home_take
                rem_cap[home] -= home_take
                rem -= home_take
            if rem > 0:
                cnt = krng.randint(sub_lo, min(sub_hi, n))
                rest_parts = [p for p in parts if p != home]
                krng.shuffle(rest_parts)
                chosen = rest_parts[: max(1, min(cnt - 1, len(rest_parts)))]
                for p in chosen:
                    if rem <= 0:
                        break
                    room = rem_cap[p]
                    if room <= 0:
                        continue
                    take = min(rem, room)
                    partitioned[p][k] = partitioned[p].get(k, 0) + take
                    rem_cap[p] -= take
                    rem -= take

        # Any leftover (or regular non-HH) is spread as a local distractor carpet:
        # many partitions receive competitive chunks, but no single partition gets
        # the whole key mass.
        if rem > 0:
            target_chunk = max(1, int(math.ceil((threshold_count / float(max(1, num_partitions))) * 1.3)))
            while rem > 0:
                candidate_parts = [p for p in parts if rem_cap[p] > 0]
                if not candidate_parts:
                    break
                candidate_parts.sort(
                    key=lambda p: (
                        -rem_cap[p],
                        _stable_u64(f"{seed}|HHO|NHFILL|{window_tag}|{k}|{p}|{krng.random()}"),
                    )
                )
                placed = False
                for p in candidate_parts:
                    if rem <= 0:
                        break
                    room = rem_cap[p]
                    if room <= 0:
                        continue
                    take = min(rem, room, target_chunk)
                    if take <= 0:
                        continue
                    partitioned[p][k] = partitioned[p].get(k, 0) + take
                    rem_cap[p] -= take
                    rem -= take
                    placed = True
                if not placed:
                    break

    # Final cleanup: drop zeros.
    for p in parts:
        partitioned[p] = {k: v for k, v in partitioned[p].items() if v > 0}

    return partitioned


def _assign_partitions_round_robin(
    freq_dist: Dict[Any, int],
    num_partitions: int,
    seed: int | None = 0,
    window_id: int | None = None,
) -> Dict[int, Dict[Any, int]]:
    if num_partitions <= 0:
        raise ValueError("num_partitions must be > 0")

    partitioned: Dict[int, Dict[Any, int]] = {p: {} for p in range(num_partitions)}
    window_tag = 0 if window_id is None else int(window_id)

    for k, freq in freq_dist.items():
        freq = int(freq)
        if freq <= 0:
            continue

        base = freq // num_partitions
        rem = freq % num_partitions
        start = _stable_u64(f"{seed}|RR|{window_tag}|{k}") % num_partitions

        for offset in range(num_partitions):
            p = (start + offset) % num_partitions
            share = base + (1 if offset < rem else 0)
            if share > 0:
                partitioned[p][k] = partitioned[p].get(k, 0) + share

    return partitioned


def _assign_partitions_synthetic_mixture(
    freq_dist: Dict[Any, int],
    num_partitions: int,
    seed: int | None = 0,
    mode_probs: tuple[float, float, float] = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    normal_subset_frac_range: tuple[float, float] = (0.5, 1.0),
    normal_std_frac_range: tuple[float, float] = (0.2, 0.6),
    skew_mass_frac_range: tuple[float, float] = (0.4, 0.8),
    skew_subset_frac_range: tuple[float, float] = (0.1, 0.3),
    rest_subset_frac_range: tuple[float, float] = (0.2, 1.0),
) -> Dict[int, Dict[Any, int]]:
    if num_partitions <= 0:
        raise ValueError("num_partitions must be > 0")

    partitioned: Dict[int, Dict[Any, int]] = {p: {} for p in range(num_partitions)}
    parts_all = list(range(num_partitions))
    uniform_p, normal_p, skew_p = mode_probs
    total_p = uniform_p + normal_p + skew_p
    if total_p <= 0:
        raise ValueError("mode_probs must sum to a positive value")
    uniform_p /= total_p
    normal_p /= total_p
    skew_p /= total_p
    normal_min_frac, normal_max_frac = normal_subset_frac_range
    skew_min_frac, skew_max_frac = skew_subset_frac_range
    rest_min_frac, rest_max_frac = rest_subset_frac_range
    normal_std_min, normal_std_max = normal_std_frac_range

    for k, freq in freq_dist.items():
        freq = int(freq)
        if freq <= 0:
            continue

        rng = random if seed is None else random.Random(_stable_u64(f"{seed}|{k}"))

        mode_roll = rng.random()
        if mode_roll < uniform_p:
            # Uniform split across all partitions
            base = freq // num_partitions
            rem = freq % num_partitions
            for i in range(num_partitions):
                share = base + (1 if i < rem else 0)
                if share > 0:
                    partitioned[i][k] = partitioned[i].get(k, 0) + share
            continue
        if mode_roll < (uniform_p + normal_p):
            # Normal-shaped allocation over a random subset
            parts = _choose_partitions(rng, parts_all, normal_min_frac, normal_max_frac)
            rng.shuffle(parts)
            weights = _normal_weights(len(parts), rng, (normal_std_min, normal_std_max))
            alloc = _allocate_integer_mass(freq, parts, weights)
            for p, share in alloc.items():
                partitioned[p][k] = partitioned[p].get(k, 0) + share
            continue

        # Skewed split with normal-distributed remainder
        skew_parts = _choose_partitions(rng, parts_all, skew_min_frac, skew_max_frac)
        skew_mass_frac = rng.uniform(skew_mass_frac_range[0], skew_mass_frac_range[1])
        skew_mass = int(round(freq * skew_mass_frac))
        rest_mass = freq - skew_mass

        if skew_mass > 0:
            weights = [rng.random() for _ in range(len(skew_parts))]
            alloc_skew = _allocate_integer_mass(skew_mass, skew_parts, weights)
            for p, share in alloc_skew.items():
                partitioned[p][k] = partitioned[p].get(k, 0) + share

        if rest_mass > 0:
            skew_set = set(skew_parts)
            rest_parts_all = [p for p in parts_all if p not in skew_set]
            if not rest_parts_all:
                rest_parts_all = list(skew_parts)
            rest_parts = _choose_partitions(rng, rest_parts_all, rest_min_frac, rest_max_frac)
            rng.shuffle(rest_parts)
            weights = _normal_weights(len(rest_parts), rng, (normal_std_min, normal_std_max))
            alloc_rest = _allocate_integer_mass(rest_mass, rest_parts, weights)
            for p, share in alloc_rest.items():
                partitioned[p][k] = partitioned[p].get(k, 0) + share

    return partitioned


def _assign_partitions_imbalance(
    freq_dist: Dict[Any, int],
    num_partitions: int,
    top_n: int | None = None,          # rare := freq < total_items/top_n
    skewed_fraction: float = 1.0,
    skew_ratio: float = 0.6,
    skew_jitter: float = 0.15,
    seed: int | None = 0,
    skew_sigma_frac: float = 0.14,
    rest_sigma_frac: float = 0.32,
    head_band_frac: float = 0.12,
    tail_band_frac: float = 0.45,
    floor_ratio: float = 0.03,
    skew_dilute_fraction: float = 0.04,
    rest_dilute_fraction: float = 0.14,
    imbalance: float = 0.0,
    rarity_gamma: float = 1.0,
    rarity_jitter: float = 0.0,
    rarity_midpoint: float = 0.7,
    rarity_steepness: float = 0.0,
    rare_spread_fraction: float = 0.0,
    rare_spread_min_partitions: int = 2,
    rare_spread_max_partitions: int | None = None,
    rarity_mode: str = "threshold",
) -> Dict[int, Dict[Any, int]]:
    if num_partitions <= 0:
        raise ValueError("num_partitions must be > 0")

    partitioned: Dict[int, Dict[Any, int]] = {p: {} for p in range(num_partitions)}

    total_mass = sum(int(v) for v in freq_dist.values() if int(v) > 0)
    freq_threshold = None
    top_keys_set = None
    if top_n is not None and top_n > 0:
        if rarity_mode == "rank":
            keys_sorted = sorted(freq_dist.items(), key=lambda kv: int(kv[1]), reverse=True)
            top_keys_set = {k for k, _ in keys_sorted[: min(int(top_n), len(keys_sorted))]}
        else:
            freq_threshold = total_mass / float(top_n)

    freqs = [int(v) for v in freq_dist.values() if int(v) > 0]
    f_min = min(freqs) if freqs else 0
    f_max = max(freqs) if freqs else 0
    log_min = math.log(f_min) if f_min > 0 else 0.0
    log_max = math.log(f_max) if f_max > 0 else 0.0
    denom = (log_max - log_min)

    imb_max = max(0.0, min(1.0, float(imbalance)))
    rarity_gamma = max(1e-9, float(rarity_gamma))
    rarity_jitter = max(0.0, min(1.0, float(rarity_jitter)))
    rarity_midpoint = max(0.0, min(1.0, float(rarity_midpoint)))
    rarity_steepness = max(0.0, float(rarity_steepness))

    for k, freq in freq_dist.items():
        freq = int(freq)
        if freq <= 0:
            continue

        rng = random if seed is None else random.Random(_stable_u64(f"{seed}|{k}"))

        if rng.random() >= skewed_fraction:
            base = freq // num_partitions
            rem = freq % num_partitions
            for i in range(num_partitions):
                share = base + (1 if i < rem else 0)
                if share > 0:
                    partitioned[i][k] = partitioned[i].get(k, 0) + share
            continue

        if top_keys_set is not None:
            rarity = 0.0 if k in top_keys_set else 1.0
        elif freq_threshold is not None:
            rarity = 0.0 if freq >= freq_threshold else 1.0
        else:
            if denom <= 0:
                rarity = 0.0
            else:
                rarity = (log_max - math.log(freq)) / denom
                rarity = max(0.0, min(1.0, rarity))
            rarity = rarity ** rarity_gamma
            if rarity_steepness > 0.0:
                rarity = _sigmoid(rarity_steepness * (rarity - rarity_midpoint))
            if rarity_jitter > 0.0:
                rarity = max(0.0, min(1.0, rarity + (rng.random() - 0.5) * rarity_jitter))

        key_imb = imb_max * rarity

        if rarity == 1.0 and float(rare_spread_fraction) > 0.0:
            if rng.random() < float(rare_spread_fraction):
                mn = max(1, int(rare_spread_min_partitions))
                mx = num_partitions if rare_spread_max_partitions is None else max(mn, int(rare_spread_max_partitions))
                k_parts = max(mn, min(mx, num_partitions))
                chosen = rng.sample(range(num_partitions), k_parts)
                alloc = _allocate_integer_mass(freq, chosen, [1.0] * k_parts)
                for p, share in alloc.items():
                    partitioned[p][k] = partitioned[p].get(k, 0) + share
                continue

        shape_scale = 1.0 - 0.75 * key_imb
        dilute_scale = (1.0 - key_imb) ** 2
        weight_power = 1.0 + 3.0 * key_imb

        skew_sigma_eff = max(0.02, skew_sigma_frac * shape_scale)
        rest_sigma_eff = max(0.04, rest_sigma_frac * shape_scale)
        head_band_eff = max(0.0, head_band_frac * shape_scale)
        tail_band_eff = max(0.0, tail_band_frac * shape_scale)

        floor_eff = floor_ratio * dilute_scale
        skew_dilute_eff = skew_dilute_fraction * dilute_scale
        rest_dilute_eff = rest_dilute_fraction * dilute_scale

        skew_ratio_eff = min(0.95, skew_ratio + 0.25 * key_imb)

        min_count = max(1, num_partitions // 5)
        max_count = min(num_partitions, max(min_count, math.ceil(num_partitions / 3)))
        skew_part_count = rng.randint(min_count, max_count)

        skew_parts = rng.sample(range(num_partitions), skew_part_count)
        rest_parts = [p for p in range(num_partitions) if p not in skew_parts]

        rng.shuffle(skew_parts)
        rng.shuffle(rest_parts)

        theta = rng.uniform(skew_ratio_eff - skew_jitter, skew_ratio_eff + skew_jitter)
        theta = max(0.0, min(1.0, theta))

        skew_mass = int(round(freq * theta))
        rest_mass = freq - skew_mass

        sw = _banded_gaussian_weights(
            n=len(skew_parts),
            sigma_frac=skew_sigma_eff,
            head_band_frac=head_band_eff,
            tail_band_frac=tail_band_eff,
            floor_ratio=floor_eff,
            mu_index=0.0,
            weight_power=weight_power,
        )
        alloc_skew = _allocate_with_dilution(skew_mass, skew_parts, sw, skew_dilute_eff)
        for p, share in alloc_skew.items():
            partitioned[p][k] = partitioned[p].get(k, 0) + share

        targets = rest_parts if rest_parts else skew_parts
        rw = _banded_gaussian_weights(
            n=len(targets),
            sigma_frac=(rest_sigma_eff if rest_parts else skew_sigma_eff),
            head_band_frac=head_band_eff,
            tail_band_frac=tail_band_eff,
            floor_ratio=floor_eff,
            mu_index=0.0,
            weight_power=weight_power,
        )
        alloc_rest = _allocate_with_dilution(rest_mass, targets, rw, rest_dilute_eff)
        for p, share in alloc_rest.items():
            partitioned[p][k] = partitioned[p].get(k, 0) + share

    return partitioned
