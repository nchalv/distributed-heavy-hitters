"""
Partitioning helpers for generating stress-test key-to-partition layouts.

The main entry point, `assign_partitions`, splits key mass across partitions with
controlled imbalance. Keys inside `top_n` are treated as “common”; keys outside
are “rare” and can be made sharply skewed via the `imbalance` knob.
"""

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
    frontier_min_ratio: float = 0.80,
    frontier_max_ratio: float = 1.20,
    frontier_reporter_frac: float = 0.08,
    frontier_reporter_min: int = 4,
    frontier_reporter_max: int = 12,
    frontier_reporter_weight_jitter: float = 0.20,
    adv_hh_local_cap_scale: float = 0.8,
    adv_hh_overflow_cap_scale: float = 2.5,
    adv_nonhh_min_ratio: float = 0.65,
    adv_nonhh_max_ratio: float = 0.995,
    adv_nonhh_count_ratio: float = 4.0,
    adv_nonhh_min_reporters: int = 4,
    adv_nonhh_max_reporters: int = 16,
    adv_key_max_fraction: float = 0.25,
    milp_target_max_keys: int = 80,
    milp_nonhh_min_ratio: float = 0.65,
    milp_nonhh_max_ratio: float = 0.995,
    milp_nonhh_count_ratio: float = 2.0,
    milp_key_max_fraction: float = 0.25,
    milp_hidden_weight: float = 1.0,
    milp_mislead_weight: float = 1.0,
    milp_time_limit_sec: float = 120.0,
    milp_mip_rel_gap: float = 0.0,
    milp_target_key_fraction: float = 0.0,
    milp_target_min_keys: int = 1,
    milp_hh_target_fraction: float = 0.5,
    milp_intermediate_scale: float = 0.5,
    milp_print_diagnostics: bool = False,
    locality_global_fraction: float = 0.15,
    locality_regional_fraction: float = 0.35,
    locality_local_fraction: float = 0.35,
    locality_regional_home_fraction: float = 0.20,
    locality_local_home_fraction: float = 0.04,
    locality_global_strength: float = 0.10,
    locality_regional_strength: float = 0.75,
    locality_local_strength: float = 0.90,
    locality_background_strength: float = 0.00,
    locality_uniform_floor_fraction: float = 0.0,
    locality_weight_jitter: float = 0.05,
    pa_hh_spread_fraction: float = 0.75,
    pa_hh_local_cap_scale: float = 0.85,
    pa_nonhh_min_ratio: float = 0.65,
    pa_nonhh_max_ratio: float = 0.995,
    pa_nonhh_home_fraction: float = 0.05,
    pa_nonhh_home_mass_frac: float = 0.90,
    pa_background_policy: str = "persistent_locality",
) -> Dict[int, Dict[Any, int]]:
    """
    Partition keys across partitions using the selected policy.

    policy="round_robin": simple even split of each key across partitions.
    policy="synthetic_mixture": per-key stochastic mix (uniform / normal / skewed).
    policy="persistent_locality": stable per-key affinity profiles over partitions.
    policy="persistent_ambiguity_adversary": stable profiles chosen to produce
      temporally coherent certificate ambiguity.
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
    if policy in {"persistent_locality", "persistent_affinity", "locality"}:
        return _assign_partitions_persistent_locality(
            freq_dist=freq_dist,
            num_partitions=num_partitions,
            seed=seed,
            global_fraction=locality_global_fraction,
            regional_fraction=locality_regional_fraction,
            local_fraction=locality_local_fraction,
            regional_home_fraction=locality_regional_home_fraction,
            local_home_fraction=locality_local_home_fraction,
            global_strength=locality_global_strength,
            regional_strength=locality_regional_strength,
            local_strength=locality_local_strength,
            background_strength=locality_background_strength,
            uniform_floor_fraction=locality_uniform_floor_fraction,
            weight_jitter=locality_weight_jitter,
        )
    if policy in {
        "persistent_ambiguity_adversary",
        "persistent_locality_adversary",
        "persistent_ambiguity",
    }:
        return _assign_partitions_persistent_ambiguity_adversary(
            freq_dist=freq_dist,
            num_partitions=num_partitions,
            seed=seed,
            top_n=top_n,
            hh_spread_fraction=pa_hh_spread_fraction,
            hh_local_cap_scale=pa_hh_local_cap_scale,
            nonhh_min_ratio=pa_nonhh_min_ratio,
            nonhh_max_ratio=pa_nonhh_max_ratio,
            nonhh_home_fraction=pa_nonhh_home_fraction,
            nonhh_home_mass_frac=pa_nonhh_home_mass_frac,
            background_policy=pa_background_policy,
            locality_global_fraction=locality_global_fraction,
            locality_regional_fraction=locality_regional_fraction,
            locality_local_fraction=locality_local_fraction,
            locality_regional_home_fraction=locality_regional_home_fraction,
            locality_local_home_fraction=locality_local_home_fraction,
            locality_global_strength=locality_global_strength,
            locality_regional_strength=locality_regional_strength,
            locality_local_strength=locality_local_strength,
            locality_background_strength=locality_background_strength,
            locality_uniform_floor_fraction=locality_uniform_floor_fraction,
            locality_weight_jitter=locality_weight_jitter,
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
    if policy in {"certificate_stress", "cert_ambiguity", "ambiguity_stress"}:
        return _assign_partitions_certificate_stress(
            freq_dist=freq_dist,
            num_partitions=num_partitions,
            seed=seed,
            window_id=window_id,
            top_n=top_n,
            frontier_min_ratio=frontier_min_ratio,
            frontier_max_ratio=frontier_max_ratio,
            frontier_reporter_frac=frontier_reporter_frac,
            frontier_reporter_min=frontier_reporter_min,
            frontier_reporter_max=frontier_reporter_max,
            frontier_reporter_weight_jitter=frontier_reporter_weight_jitter,
        )
    if policy in {"constrained_certificate_adversary", "certificate_adversary", "cert_adversary"}:
        return _assign_partitions_constrained_certificate_adversary(
            freq_dist=freq_dist,
            num_partitions=num_partitions,
            seed=seed,
            window_id=window_id,
            top_n=top_n,
            hh_local_cap_scale=adv_hh_local_cap_scale,
            hh_overflow_cap_scale=adv_hh_overflow_cap_scale,
            nonhh_min_ratio=adv_nonhh_min_ratio,
            nonhh_max_ratio=adv_nonhh_max_ratio,
            nonhh_count_ratio=adv_nonhh_count_ratio,
            nonhh_min_reporters=adv_nonhh_min_reporters,
            nonhh_max_reporters=adv_nonhh_max_reporters,
            key_max_fraction=adv_key_max_fraction,
        )
    if policy in {
        "milp_certificate_adversary",
        "certificate_milp_adversary",
        "milp_adversary",
        "milp_certificate_adversary_intermediate",
    }:
        difficulty_scale = (
            max(0.0, min(1.0, float(milp_intermediate_scale)))
            if policy == "milp_certificate_adversary_intermediate"
            else 1.0
        )
        return _assign_partitions_milp_certificate_adversary(
            freq_dist=freq_dist,
            num_partitions=num_partitions,
            seed=seed,
            window_id=window_id,
            top_n=top_n,
            target_max_keys=max(1, int(round(milp_target_max_keys * difficulty_scale))),
            nonhh_min_ratio=milp_nonhh_min_ratio,
            nonhh_max_ratio=milp_nonhh_max_ratio,
            nonhh_count_ratio=milp_nonhh_count_ratio,
            key_max_fraction=milp_key_max_fraction,
            hidden_weight=milp_hidden_weight,
            mislead_weight=milp_mislead_weight,
            time_limit_sec=milp_time_limit_sec,
            mip_rel_gap=milp_mip_rel_gap,
            target_key_fraction=milp_target_key_fraction * difficulty_scale,
            target_min_keys=max(1, int(round(milp_target_min_keys * difficulty_scale))),
            hh_target_fraction=milp_hh_target_fraction,
            print_diagnostics=milp_print_diagnostics,
        )
    raise ValueError(f"Unknown partitioning policy: {policy}")


def _assign_partitions_persistent_locality(
    freq_dist: Dict[Any, int],
    num_partitions: int,
    *,
    seed: int | None = 0,
    global_fraction: float = 0.15,
    regional_fraction: float = 0.35,
    local_fraction: float = 0.35,
    regional_home_fraction: float = 0.20,
    local_home_fraction: float = 0.04,
    global_strength: float = 0.10,
    regional_strength: float = 0.75,
    local_strength: float = 0.90,
    background_strength: float = 0.00,
    uniform_floor_fraction: float = 0.0,
    weight_jitter: float = 0.05,
) -> Dict[int, Dict[Any, int]]:
    """
    Stable locality model for persistent substream communities.

    Each key receives a deterministic locality class and home-partition set
    based only on the seed and key. Every window then splits the key's current
    global count through that same affinity profile, so local hot spots persist
    over time while global key frequencies may evolve. A configurable uniform
    floor keeps local substreams diverse enough for q=n not to become exact.
    """
    if num_partitions <= 0:
        raise ValueError("num_partitions must be > 0")

    partitioned: Dict[int, Dict[Any, int]] = {p: {} for p in range(num_partitions)}
    parts = list(range(num_partitions))
    seed_tag = 0 if seed is None else int(seed)

    g_frac = max(0.0, float(global_fraction))
    r_frac = max(0.0, float(regional_fraction))
    l_frac = max(0.0, float(local_fraction))
    total = g_frac + r_frac + l_frac
    if total > 1.0:
        g_frac /= total
        r_frac /= total
        l_frac /= total

    def clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    uniform_floor = clamp01(uniform_floor_fraction)

    def choose_profile(k: Any) -> tuple[str, List[int], float]:
        rng = random.Random(_stable_u64(f"{seed_tag}|PERSIST|{k}"))
        roll = rng.random()
        if roll < g_frac:
            kind = "global"
            home_count = num_partitions
            strength = global_strength
        elif roll < g_frac + r_frac:
            kind = "regional"
            home_count = max(1, min(num_partitions, int(round(regional_home_fraction * num_partitions))))
            strength = regional_strength
        elif roll < g_frac + r_frac + l_frac:
            kind = "local"
            home_count = max(1, min(num_partitions, int(round(local_home_fraction * num_partitions))))
            strength = local_strength
        else:
            kind = "background"
            home_count = num_partitions
            strength = background_strength

        if home_count >= num_partitions:
            homes = parts
        else:
            homes = sorted(rng.sample(parts, home_count))
        return kind, homes, clamp01(strength)

    jitter = max(0.0, float(weight_jitter))
    for k, count_raw in freq_dist.items():
        count = int(count_raw)
        if count <= 0:
            continue
        _, homes, strength = choose_profile(k)
        home_set = set(homes)
        home_count = max(1, len(homes))
        rng = random.Random(_stable_u64(f"{seed_tag}|PERSIST|WEIGHTS|{k}"))
        floor_count = int(round(count * uniform_floor))
        floor_count = max(0, min(count, floor_count))
        if floor_count > 0:
            floor_parts = parts[:]
            rng.shuffle(floor_parts)
            floor_alloc = _allocate_integer_mass(floor_count, floor_parts, [1.0] * len(floor_parts))
            for p, take in floor_alloc.items():
                partitioned[p][k] = partitioned[p].get(k, 0) + int(take)

        residual_count = count - floor_count
        if residual_count <= 0:
            continue

        weights: List[float] = []
        for p in parts:
            base = (1.0 - strength) / float(num_partitions)
            if p in home_set:
                base += strength / float(home_count)
            if jitter > 0.0:
                base *= rng.uniform(max(0.0, 1.0 - jitter), 1.0 + jitter)
            weights.append(max(0.0, base))
        alloc = _allocate_integer_mass(residual_count, parts, weights)
        for p, take in alloc.items():
            partitioned[p][k] = partitioned[p].get(k, 0) + int(take)

    return partitioned


def _assign_partitions_persistent_ambiguity_adversary(
    freq_dist: Dict[Any, int],
    num_partitions: int,
    *,
    seed: int | None = 0,
    top_n: int | None = None,
    hh_spread_fraction: float = 0.75,
    hh_local_cap_scale: float = 0.85,
    nonhh_min_ratio: float = 0.65,
    nonhh_max_ratio: float = 0.995,
    nonhh_home_fraction: float = 0.05,
    nonhh_home_mass_frac: float = 0.90,
    background_policy: str = "persistent_locality",
    locality_global_fraction: float = 0.15,
    locality_regional_fraction: float = 0.35,
    locality_local_fraction: float = 0.35,
    locality_regional_home_fraction: float = 0.20,
    locality_local_home_fraction: float = 0.04,
    locality_global_strength: float = 0.10,
    locality_regional_strength: float = 0.75,
    locality_local_strength: float = 0.90,
    locality_background_strength: float = 0.00,
    locality_uniform_floor_fraction: float = 0.0,
    locality_weight_jitter: float = 0.05,
) -> Dict[int, Dict[Any, int]]:
    """
    Persistent adversarial locality for ambiguity-aware sizing experiments.

    The policy keeps a deterministic per-key partition signature across windows.
    Near-threshold non-HHs are concentrated into stable local homes, creating
    persistent local distractors and inflated local substream masses. Global HHs
    are then spread over stable broad homes, preferring the inflated partitions
    and capping each local contribution below a fraction of that partition's
    current HH threshold whenever possible. Remaining background keys use the
    benign persistent-locality model.
    """
    if num_partitions <= 0:
        raise ValueError("num_partitions must be > 0")

    seed_tag = 0 if seed is None else int(seed)
    hh_n = max(1, int(top_n or 1))
    total_mass = sum(max(0, int(c)) for c in freq_dist.values())
    threshold = float(total_mass) / float(hh_n) if hh_n > 0 else float("inf")
    parts = list(range(num_partitions))

    partitioned: Dict[int, Dict[Any, int]] = {p: {} for p in parts}
    partition_sizes = [0 for _ in parts]

    items = [(k, int(c)) for k, c in freq_dist.items() if int(c) > 0]
    heavy = [(k, c) for k, c in items if c > threshold]
    near_nonhh = [
        (k, c)
        for k, c in items
        if c <= threshold and c >= nonhh_min_ratio * threshold and c <= nonhh_max_ratio * threshold
    ]
    special = {k for k, _ in heavy} | {k for k, _ in near_nonhh}
    background = {k: c for k, c in items if k not in special}

    def stable_parts(k: Any, tag: str, count: int) -> List[int]:
        count = max(1, min(num_partitions, int(count)))
        rng = random.Random(_stable_u64(f"{seed_tag}|PERSIST_ADV|{tag}|{k}"))
        if count >= num_partitions:
            return parts[:]
        return sorted(rng.sample(parts, count))

    def add_alloc(k: Any, alloc: Dict[int, int]) -> None:
        for p, take in alloc.items():
            if take <= 0:
                continue
            partitioned[p][k] = partitioned[p].get(k, 0) + int(take)
            partition_sizes[p] += int(take)

    # 1. Background keys preserve stable locality, so the partition geography is
    # not rebuilt independently in each window.
    if background:
        bg = _assign_partitions_persistent_locality(
            background,
            num_partitions,
            seed=seed,
            global_fraction=locality_global_fraction,
            regional_fraction=locality_regional_fraction,
            local_fraction=locality_local_fraction,
            regional_home_fraction=locality_regional_home_fraction,
            local_home_fraction=locality_local_home_fraction,
            global_strength=locality_global_strength,
            regional_strength=locality_regional_strength,
            local_strength=locality_local_strength,
            background_strength=locality_background_strength,
            uniform_floor_fraction=locality_uniform_floor_fraction,
            weight_jitter=locality_weight_jitter,
        )
        for p, counts in bg.items():
            for k, take in counts.items():
                add_alloc(k, {p: int(take)})

    # 2. Stable local distractors: near-HH non-HHs become local hot keys in the
    # same home partitions across windows.
    home_count_nonhh = max(1, int(round(max(0.0, nonhh_home_fraction) * num_partitions)))
    home_mass = max(0.0, min(1.0, float(nonhh_home_mass_frac)))
    for k, c in sorted(near_nonhh, key=lambda x: (-x[1], str(x[0]))):
        homes = stable_parts(k, "NONHH_HOME", home_count_nonhh)
        home_total = int(round(c * home_mass))
        home_total = max(0, min(c, home_total))
        rest_total = c - home_total
        alloc: Dict[int, int] = {}
        rng = random.Random(_stable_u64(f"{seed_tag}|PERSIST_ADV|NONHH_W|{k}"))
        home_weights = [rng.uniform(0.8, 1.2) for _ in homes]
        for p, take in _allocate_integer_mass(home_total, homes, home_weights).items():
            alloc[p] = alloc.get(p, 0) + take
        if rest_total > 0:
            rest_weights = [1.0 for _ in parts]
            for p, take in _allocate_integer_mass(rest_total, parts, rest_weights).items():
                alloc[p] = alloc.get(p, 0) + take
        add_alloc(k, alloc)

    # 3. Stable broad HH hiding: put HHs preferentially on persistent broad homes
    # whose local masses were inflated by background/distractor traffic.
    spread_count = max(1, int(round(max(0.0, min(1.0, hh_spread_fraction)) * num_partitions)))
    cap_scale = max(0.0, float(hh_local_cap_scale))
    for k, c in sorted(heavy, key=lambda x: (-x[1], str(x[0]))):
        homes = stable_parts(k, "HH_SPREAD", spread_count)
        # Prefer partitions with persistent background/distractor mass. The
        # stable jitter prevents perfect ties from producing identical profiles.
        rng = random.Random(_stable_u64(f"{seed_tag}|PERSIST_ADV|HH_W|{k}"))
        weights = [
            max(1.0, float(partition_sizes[p])) * rng.uniform(0.9, 1.1)
            for p in homes
        ]
        caps = [
            max(1, int(math.floor(cap_scale * max(1.0, float(partition_sizes[p])) / float(hh_n))))
            for p in homes
        ]
        alloc = _allocate_integer_mass_with_caps(c, homes, weights, caps, relax_caps=True)
        add_alloc(k, alloc)

    return partitioned


def _assign_partitions_milp_certificate_adversary(
    freq_dist: Dict[Any, int],
    num_partitions: int,
    *,
    seed: int | None = 0,
    window_id: int | None = None,
    top_n: int | None = None,
    target_max_keys: int = 80,
    nonhh_min_ratio: float = 0.65,
    nonhh_max_ratio: float = 0.995,
    nonhh_count_ratio: float = 2.0,
    key_max_fraction: float = 0.25,
    hidden_weight: float = 1.0,
    mislead_weight: float = 1.0,
    time_limit_sec: float = 120.0,
    mip_rel_gap: float = 0.0,
    target_key_fraction: float = 0.0,
    target_min_keys: int = 1,
    hh_target_fraction: float = 0.5,
    print_diagnostics: bool = False,
) -> Dict[int, Dict[Any, int]]:
    """
    MILP-backed adversarial placement under a certificate proxy.

    The MILP optimizes target keys only: true HHs and near-threshold non-HHs.
    Remaining keys are used as deterministic filler to meet exact partition
    masses. The proxy rewards hidden HH mass below local detection thresholds
    and locally salient non-HH distractors.
    """
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix
    except Exception as exc:  # pragma: no cover - depends on runtime deps
        raise RuntimeError("milp_certificate_adversary requires scipy.optimize.milp") from exc

    if num_partitions <= 0:
        raise ValueError("num_partitions must be > 0")

    partitioned: Dict[int, Dict[Any, int]] = {p: {} for p in range(num_partitions)}
    items = [(k, int(v)) for k, v in freq_dist.items() if int(v) > 0]
    if not items:
        return partitioned

    total_mass = sum(v for _, v in items)
    if total_mass <= 0:
        return partitioned

    window_tag = 0 if window_id is None else int(window_id)
    hh_n = int(top_n) if top_n is not None and int(top_n) > 0 else max(1, num_partitions)
    threshold_count = total_mass / float(hh_n)
    parts = list(range(num_partitions))
    target_map = _allocate_integer_mass(total_mass, parts, [1.0] * num_partitions)
    target_mass = [int(target_map.get(p, 0)) for p in parts]
    local_threshold = [int(math.floor(target_mass[p] / float(hh_n)) + 1) for p in parts]

    items_sorted = sorted(items, key=lambda kv: (-kv[1], str(kv[0])))
    heavy = [(k, c) for k, c in items_sorted if float(c) > threshold_count]
    non_heavy = [(k, c) for k, c in items_sorted if float(c) <= threshold_count]

    configured_max_keys = max(1, int(target_max_keys))
    adaptive_target_keys = int(math.ceil(max(0.0, float(target_key_fraction)) * float(hh_n)))
    max_keys = max(configured_max_keys, max(1, int(target_min_keys)), adaptive_target_keys)
    max_keys = min(max_keys, len(items_sorted))
    heavy_sorted = sorted(heavy, key=lambda kv: (abs(float(kv[1]) - threshold_count), -kv[1], str(kv[0])))
    d_min = max(0.0, float(nonhh_min_ratio)) * threshold_count
    d_max = max(d_min, float(nonhh_max_ratio) * threshold_count)
    nonhh_pool = [(k, c) for k, c in non_heavy if d_min <= float(c) <= d_max]
    nonhh_pool.sort(key=lambda kv: (abs(float(kv[1]) - threshold_count), -kv[1], str(kv[0])))

    hh_budget = int(round(max_keys * max(0.0, min(1.0, float(hh_target_fraction)))))
    if heavy_sorted and hh_budget <= 0:
        hh_budget = 1
    hh_target_count = min(len(heavy_sorted), hh_budget)
    selected_hh = heavy_sorted[:hh_target_count]
    nonhh_budget = max(0, max_keys - len(selected_hh))
    if nonhh_count_ratio > 0:
        ratio_base = max(len(selected_hh), nonhh_budget)
        nonhh_budget = min(nonhh_budget, int(round(float(nonhh_count_ratio) * max(1, ratio_base))))
    nonhh_target_count = min(len(nonhh_pool), nonhh_budget)
    selected_nonhh = nonhh_pool[:nonhh_target_count]
    target_items = selected_hh + selected_nonhh
    if print_diagnostics:
        warnings = []
        if not heavy_sorted:
            warnings.append("no_global_hh")
        if not nonhh_pool:
            warnings.append("no_near_nonhh")
        if max(local_threshold) <= 2:
            warnings.append("local_threshold_le_2")
        print(
            f"[window_id={window_tag}] MILP targets: "
            f"n={hh_n}, N={total_mass}, threshold={threshold_count:.3f}, "
            f"global_hh={len(heavy_sorted)}, near_nonhh={len(nonhh_pool)}, "
            f"target_budget={max_keys}, selected_hh={len(selected_hh)}, "
            f"selected_nonhh={len(selected_nonhh)}, local_threshold_min={min(local_threshold)}, "
            f"local_threshold_max={max(local_threshold)}, "
            f"warnings={','.join(warnings) if warnings else 'none'}"
        )
    if not target_items:
        if print_diagnostics:
            print(f"[window_id={window_tag}] MILP targets: no eligible targets; falling back to round_robin")
        return _assign_partitions_round_robin(freq_dist, num_partitions, seed=seed, window_id=window_id)

    target_keys = [k for k, _ in target_items]
    target_counts = [int(c) for _, c in target_items]
    target_key_set = set(target_keys)
    is_heavy = [k in {hk for hk, _ in selected_hh} for k in target_keys]
    K = len(target_items)
    P = num_partitions

    x_off = 0
    z_off = x_off + K * P
    u_off = z_off + K * P
    n_vars = u_off + K * P

    def x_idx(t: int, p: int) -> int:
        return x_off + t * P + p

    def z_idx(t: int, p: int) -> int:
        return z_off + t * P + p

    def u_idx(t: int, p: int) -> int:
        return u_off + t * P + p

    c_vec = np.zeros(n_vars)
    lb = np.zeros(n_vars)
    ub = np.full(n_vars, np.inf)
    integrality = np.ones(n_vars)

    key_frac = max(1e-9, min(1.0, float(key_max_fraction)))
    for t, count in enumerate(target_counts):
        per_key_cap = max(1, int(math.ceil(key_frac * count)))
        for p in parts:
            ub[x_idx(t, p)] = min(per_key_cap, count, target_mass[p])
            ub[z_idx(t, p)] = 1
            ub[u_idx(t, p)] = max(0, local_threshold[p] - 1)
            if is_heavy[t]:
                c_vec[u_idx(t, p)] = -float(hidden_weight)
            else:
                c_vec[z_idx(t, p)] = -float(mislead_weight)

    rows = []
    lows = []
    highs = []

    # Exact target-key mass preservation.
    for t, count in enumerate(target_counts):
        row = {}
        for p in parts:
            row[x_idx(t, p)] = 1.0
        rows.append(row)
        lows.append(float(count))
        highs.append(float(count))

    # Target keys may not exceed partition target masses; filler uses the rest.
    for p in parts:
        row = {}
        for t in range(K):
            row[x_idx(t, p)] = 1.0
        rows.append(row)
        lows.append(0.0)
        highs.append(float(target_mass[p]))

    for t, count in enumerate(target_counts):
        for p in parts:
            T = local_threshold[p]
            # z=1 -> x >= T
            rows.append({x_idx(t, p): 1.0, z_idx(t, p): -float(T)})
            lows.append(0.0)
            highs.append(np.inf)
            # z=0 -> x <= T-1
            rows.append({x_idx(t, p): 1.0, z_idx(t, p): -float(count)})
            lows.append(-np.inf)
            highs.append(float(T - 1))
            # u <= x
            rows.append({u_idx(t, p): 1.0, x_idx(t, p): -1.0})
            lows.append(-np.inf)
            highs.append(0.0)
            # u <= (T-1)(1-z)
            rows.append({u_idx(t, p): 1.0, z_idx(t, p): float(T - 1)})
            lows.append(-np.inf)
            highs.append(float(T - 1))

    A = lil_matrix((len(rows), n_vars), dtype=float)
    for r, row in enumerate(rows):
        for col, val in row.items():
            A[r, col] = val

    constraints = LinearConstraint(A.tocsr(), np.array(lows), np.array(highs))
    options = {
        "time_limit": max(1.0, float(time_limit_sec)),
        "mip_rel_gap": max(0.0, float(mip_rel_gap)),
        "disp": False,
    }
    result = milp(
        c=c_vec,
        integrality=integrality,
        bounds=Bounds(lb, ub),
        constraints=constraints,
        options=options,
    )
    if result.x is None:
        raise RuntimeError(f"MILP adversarial partitioning failed: {result.message}")

    sol = result.x
    rem_cap = {p: target_mass[p] for p in parts}

    def place(k: Any, p: int, take: int) -> None:
        if take <= 0:
            return
        if rem_cap[p] < take:
            raise ValueError(f"MILP solution exceeded partition capacity {p}")
        partitioned[p][k] = partitioned[p].get(k, 0) + take
        rem_cap[p] -= take

    for t, k in enumerate(target_keys):
        assigned = 0
        shares = []
        for p in parts:
            val = int(round(float(sol[x_idx(t, p)])))
            if val > 0:
                shares.append((p, val))
                assigned += val
        diff = target_counts[t] - assigned
        if diff != 0:
            shares.sort(key=lambda pv: (rem_cap[pv[0]], -_stable_u64(f"{seed}|MILP|FIX|{window_tag}|{k}|{pv[0]}")), reverse=True)
            idx = 0
            while diff != 0 and shares:
                p, val = shares[idx % len(shares)]
                if diff > 0 and rem_cap[p] > 0:
                    shares[idx % len(shares)] = (p, val + 1)
                    diff -= 1
                elif diff < 0 and val > 0:
                    shares[idx % len(shares)] = (p, val - 1)
                    diff += 1
                idx += 1
                if idx > 10 * (len(shares) + abs(diff) + 1):
                    break
        if sum(v for _, v in shares) != target_counts[t]:
            raise ValueError(f"failed to round MILP solution for key {k}")
        for p, take in shares:
            place(k, p, take)

    def stable_fill_order(k: Any) -> List[int]:
        return sorted(
            parts,
            key=lambda p: (rem_cap[p], -_stable_u64(f"{seed}|MILP|FILL|{window_tag}|{k}|{p}")),
            reverse=True,
        )

    for k, count in items_sorted:
        if k in target_key_set:
            continue
        rem = int(count)
        while rem > 0:
            candidates = [p for p in stable_fill_order(k) if rem_cap[p] > 0]
            if not candidates:
                break
            total = min(rem, sum(rem_cap[p] for p in candidates))
            alloc = _allocate_integer_mass_with_caps(
                total,
                candidates,
                [float(rem_cap[p] + 1) for p in candidates],
                [rem_cap[p] for p in candidates],
                relax_caps=False,
            )
            if not alloc:
                break
            for p, take in alloc.items():
                place(k, p, take)
                rem -= take
        if rem != 0:
            raise ValueError(f"failed to place filler key {k}")

    if any(v != 0 for v in rem_cap.values()):
        raise ValueError("MILP adversarial partitioning left unfilled capacity")

    if print_diagnostics:
        selected_hh_set = {k for k, _ in selected_hh}
        selected_nonhh_set = {k for k, _ in selected_nonhh}
        hidden_hh_keys = 0
        selected_hh_local_reporters = 0
        promoted_nonhh_keys = 0
        selected_nonhh_local_reporters = 0
        all_nonhh_local_keys = set()

        for k in selected_hh_set:
            reporters = sum(1 for p in parts if partitioned[p].get(k, 0) >= local_threshold[p])
            selected_hh_local_reporters += reporters
            if reporters == 0:
                hidden_hh_keys += 1

        for k in selected_nonhh_set:
            reporters = sum(1 for p in parts if partitioned[p].get(k, 0) >= local_threshold[p])
            selected_nonhh_local_reporters += reporters
            if reporters > 0:
                promoted_nonhh_keys += 1

        for k, c in non_heavy:
            if any(partitioned[p].get(k, 0) >= local_threshold[p] for p in parts):
                all_nonhh_local_keys.add(k)

        print(
            f"[window_id={window_tag}] MILP achieved: "
            f"hidden_selected_hh_keys={hidden_hh_keys}/{len(selected_hh_set)}, "
            f"selected_hh_local_reporters={selected_hh_local_reporters}, "
            f"promoted_selected_nonhh_keys={promoted_nonhh_keys}/{len(selected_nonhh_set)}, "
            f"selected_nonhh_local_reporters={selected_nonhh_local_reporters}, "
            f"all_nonhh_local_hh_keys={len(all_nonhh_local_keys)}"
        )

    return partitioned



def _assign_partitions_constrained_certificate_adversary(
    freq_dist: Dict[Any, int],
    num_partitions: int,
    *,
    seed: int | None = 0,
    window_id: int | None = None,
    top_n: int | None = None,
    hh_local_cap_scale: float = 0.8,
    hh_overflow_cap_scale: float = 2.5,
    nonhh_min_ratio: float = 0.65,
    nonhh_max_ratio: float = 0.995,
    nonhh_count_ratio: float = 4.0,
    nonhh_min_reporters: int = 4,
    nonhh_max_reporters: int = 16,
    key_max_fraction: float = 0.25,
) -> Dict[int, Dict[Any, int]]:
    """
    Deterministic constrained adversary for certificate stress.

    The policy combines hidden-HH spreading, concentrated non-HH distractors,
    and balanced background saturation while preserving exact global key counts
    and exact target partition masses.
    """
    if num_partitions <= 0:
        raise ValueError("num_partitions must be > 0")

    partitioned: Dict[int, Dict[Any, int]] = {p: {} for p in range(num_partitions)}
    items = [(k, int(v)) for k, v in freq_dist.items() if int(v) > 0]
    if not items:
        return partitioned

    total_mass = sum(v for _, v in items)
    if total_mass <= 0:
        return partitioned

    window_tag = 0 if window_id is None else int(window_id)
    hh_n = int(top_n) if top_n is not None and int(top_n) > 0 else max(1, num_partitions)
    threshold_count = total_mass / float(hh_n)
    parts = list(range(num_partitions))

    target_map = _allocate_integer_mass(total_mass, parts, [1.0] * num_partitions)
    target = [int(target_map.get(p, 0)) for p in parts]
    rem_cap = {p: target[p] for p in parts}

    def place(k: Any, alloc: Dict[int, int]) -> None:
        for p, take in alloc.items():
            take = int(take)
            if take <= 0:
                continue
            if rem_cap[p] < take:
                raise ValueError(f"partition capacity exceeded for partition {p}")
            partitioned[p][k] = partitioned[p].get(k, 0) + take
            rem_cap[p] -= take

    def stable_part_order(label: str, key: Any, candidates: List[int]) -> List[int]:
        return sorted(
            candidates,
            key=lambda p: (
                rem_cap[p],
                -_stable_u64(f"{seed}|CCA|{label}|{window_tag}|{key}|{p}"),
            ),
            reverse=True,
        )

    def allocate_on_parts(
        k: Any,
        amount: int,
        candidates: List[int],
        *,
        label: str,
        per_part_cap: int | None = None,
    ) -> int:
        if amount <= 0:
            return 0
        candidates = [p for p in stable_part_order(label, k, candidates) if rem_cap[p] > 0]
        if not candidates:
            return 0
        caps = [rem_cap[p] for p in candidates]
        if per_part_cap is not None:
            cap = max(0, int(per_part_cap))
            existing = [partitioned[p].get(k, 0) for p in candidates]
            caps = [min(caps[i], max(0, cap - existing[i])) for i in range(len(candidates))]
        active = [(p, c) for p, c in zip(candidates, caps) if c > 0]
        if not active:
            return 0
        candidates = [p for p, _ in active]
        caps = [c for _, c in active]
        total = min(int(amount), sum(caps))
        weights = [float(rem_cap[p] + 1) for p in candidates]
        alloc = _allocate_integer_mass_with_caps(total, candidates, weights, caps, relax_caps=False)
        place(k, alloc)
        return sum(alloc.values())

    items_sorted = sorted(items, key=lambda kv: (-kv[1], str(kv[0])))
    heavy_items = [(k, c) for k, c in items_sorted if float(c) > threshold_count]
    non_heavy_items = [(k, c) for k, c in items_sorted if float(c) <= threshold_count]
    placed_keys: set[Any] = set()

    # 1. True HHs: place as much mass as possible under local detection caps.
    hidden_scale = max(0.0, float(hh_local_cap_scale))
    overflow_scale = max(hidden_scale, float(hh_overflow_cap_scale))
    key_frac = max(1e-9, min(1.0, float(key_max_fraction)))
    for k, c in heavy_items:
        hidden_cap = [max(0, int(math.floor(hidden_scale * target[p] / float(hh_n)))) for p in parts]
        hidden_cap = [min(hidden_cap[p], int(math.ceil(key_frac * c))) for p in parts]
        hidden = _allocate_integer_mass_with_caps(
            c,
            parts,
            [float(rem_cap[p] + 1) for p in parts],
            [min(rem_cap[p], hidden_cap[p]) for p in parts],
            relax_caps=False,
        )
        place(k, hidden)
        rem = c - sum(hidden.values())
        if rem > 0:
            overflow_cap = max(1, int(math.ceil(overflow_scale * (total_mass / num_partitions) / float(hh_n))))
            rem -= allocate_on_parts(k, rem, parts, label="HH_OVERFLOW", per_part_cap=overflow_cap)
        if rem > 0:
            rem -= allocate_on_parts(k, rem, parts, label="HH_REMAINDER")
        if rem != 0:
            raise ValueError(f"failed to place all mass for heavy key {k}")
        placed_keys.add(k)

    # 2. Near-threshold non-HHs: concentrate into constrained local distractors.
    d_min = max(0.0, float(nonhh_min_ratio)) * threshold_count
    d_max = max(d_min, float(nonhh_max_ratio) * threshold_count)
    d_pool = [(k, c) for k, c in non_heavy_items if d_min <= float(c) <= d_max]
    d_pool.sort(key=lambda kv: (-kv[1], abs(float(kv[1]) - threshold_count), str(kv[0])))
    target_d = int(round(max(0.0, float(nonhh_count_ratio)) * max(1, len(heavy_items))))
    if target_d <= 0:
        target_d = len(d_pool)
    distractors = d_pool[: min(len(d_pool), target_d)]
    min_reporters = max(1, min(num_partitions, int(nonhh_min_reporters), int(nonhh_max_reporters)))
    max_reporters = max(min_reporters, min(num_partitions, int(nonhh_max_reporters)))

    for k, c in distractors:
        min_by_fraction = int(math.ceil(1.0 / key_frac))
        r = max(min_reporters, min_by_fraction)
        r = min(max_reporters, max(1, r))
        chosen = stable_part_order("DISTR", k, [p for p in parts if rem_cap[p] > 0])[:r]
        if not chosen:
            chosen = [p for p in parts if rem_cap[p] > 0]
        per_key_cap = max(1, int(math.ceil(key_frac * c)))
        placed = allocate_on_parts(k, c, chosen, label="DISTR", per_part_cap=per_key_cap)
        rem = c - placed
        if rem > 0:
            rem -= allocate_on_parts(k, rem, parts, label="DISTR_REMAINDER", per_part_cap=per_key_cap)
        if rem > 0:
            rem -= allocate_on_parts(k, rem, parts, label="DISTR_FALLBACK")
        if rem != 0:
            raise ValueError(f"failed to place all mass for distractor key {k}")
        placed_keys.add(k)

    # 3. Remaining keys: fill exact target capacities as a summary-saturating carpet.
    for k, c in items_sorted:
        if k in placed_keys:
            continue
        rem = int(c)
        if rem <= 0:
            continue
        rem -= allocate_on_parts(k, rem, parts, label="BACKGROUND")
        if rem != 0:
            raise ValueError(f"failed to place all mass for background key {k}")

    if any(v != 0 for v in rem_cap.values()):
        raise ValueError("constrained certificate adversary left unfilled partition capacity")

    return partitioned


def _assign_partitions_certificate_stress(
    freq_dist: Dict[Any, int],
    num_partitions: int,
    *,
    seed: int | None = 0,
    window_id: int | None = None,
    top_n: int | None = None,
    frontier_min_ratio: float = 0.80,
    frontier_max_ratio: float = 1.20,
    frontier_reporter_frac: float = 0.08,
    frontier_reporter_min: int = 4,
    frontier_reporter_max: int = 12,
    frontier_reporter_weight_jitter: float = 0.20,
) -> Dict[int, Dict[Any, int]]:
    """
    Stress the coordinator certificate, not just local heavy-hitter visibility.

    Near-threshold keys are concentrated on a small, deterministic reporter set.
    All other keys are spread as a background carpet. With bounded Space-Saving
    summaries, the carpet raises non-reporter min counters while the frontier
    keys still appear in a few workers, widening cert_lb/cert_ub around the
    global threshold.
    """
    if num_partitions <= 0:
        raise ValueError("num_partitions must be > 0")

    partitioned: Dict[int, Dict[Any, int]] = {p: {} for p in range(num_partitions)}
    items = [(k, int(v)) for k, v in freq_dist.items() if int(v) > 0]
    if not items:
        return partitioned

    total_mass = sum(v for _, v in items)
    if total_mass <= 0:
        return partitioned

    window_tag = 0 if window_id is None else int(window_id)
    hh_n = int(top_n) if top_n is not None and int(top_n) > 0 else max(1, num_partitions)
    threshold_count = total_mass / float(hh_n)
    lo = max(0.0, float(frontier_min_ratio)) * threshold_count
    hi = max(lo, float(frontier_max_ratio) * threshold_count)

    frontier = [(k, c) for k, c in items if lo <= float(c) <= hi]
    frontier_keys = {k for k, _ in frontier}
    carpet = [(k, c) for k, c in items if k not in frontier_keys]

    # Build the background first. This intentionally resembles round-robin
    # because round-robin already generated larger min-counter ambiguity than
    # the old hidden-HH construction.
    for k, freq in carpet:
        base = freq // num_partitions
        rem = freq % num_partitions
        start = _stable_u64(f"{seed}|CERT|CARPET|{window_tag}|{k}") % num_partitions
        for offset in range(num_partitions):
            share = base + (1 if offset < rem else 0)
            if share <= 0:
                continue
            p = (start + offset) % num_partitions
            partitioned[p][k] = partitioned[p].get(k, 0) + share

    reporter_frac = max(0.0, min(1.0, float(frontier_reporter_frac)))
    base_reporter_count = int(math.ceil(reporter_frac * num_partitions))
    min_reporters = max(1, min(num_partitions, int(frontier_reporter_min)))
    max_reporters = max(min_reporters, min(num_partitions, int(frontier_reporter_max)))
    fixed_reporter_count = max(min_reporters, base_reporter_count)
    fixed_reporter_count = min(max_reporters, fixed_reporter_count)
    jitter = max(0.0, float(frontier_reporter_weight_jitter))
    parts = list(range(num_partitions))

    for k, freq in frontier:
        krng = random if seed is None else random.Random(_stable_u64(f"{seed}|CERT|FRONTIER|{window_tag}|{k}"))
        reporter_count = fixed_reporter_count
        reporters = krng.sample(parts, reporter_count)
        weights = [math.exp(krng.uniform(-jitter, jitter)) for _ in reporters]
        alloc = _allocate_integer_mass(freq, reporters, weights)
        for p, share in alloc.items():
            partitioned[p][k] = partitioned[p].get(k, 0) + share

    return partitioned


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
