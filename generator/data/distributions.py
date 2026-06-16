import numpy as np
import math
import itertools
from abc import ABC, abstractmethod
import random
from typing import Dict


# === Generator Interface ===

class FrequencyDistributionGenerator(ABC):
    @abstractmethod
    def generate(self, total_items: int, num_keys: int):
        pass

# === Distribution Generators ===

class UniformDistributionGenerator(FrequencyDistributionGenerator):
    def generate(self, total_items: int, num_keys: int):
        base = total_items // num_keys
        remainder = total_items % num_keys
        freqs = [base + 1 if i < remainder else base for i in range(num_keys)]
        return {f'key_{i+1}': freqs[i] for i in range(num_keys)}

class NormalDistributionGenerator(FrequencyDistributionGenerator):
    def __init__(
        self,
        n,
        center_frac: float = 0.05,
        std_frac: float = 0.031,
        ensure_hh: bool = True,
        max_sharpen_steps: int = 30,
    ):
        self.n = n
        self.center_frac = center_frac
        self.std_frac = std_frac
        self.ensure_hh = ensure_hh
        self.max_sharpen_steps = max(0, int(max_sharpen_steps))

    def generate(self, total_items: int, num_keys: int):
        center = self.center_frac * num_keys
        base_std = max(1e-6, self.std_frac * num_keys)
        x = np.linspace(0, num_keys - 1, num_keys)
        required = int(math.floor(total_items / self.n) + 1) if self.n > 0 else 1
        std = base_std
        freqs = None

        attempts = self.max_sharpen_steps if self.ensure_hh else 0
        for _ in range(attempts + 1):
            freqs = np.exp(-0.5 * ((x - center) / std) ** 2)
            freqs = freqs / freqs.sum() * total_items
            freqs = np.floor(freqs).astype(int)
            if not self.ensure_hh or freqs.max(initial=0) >= required:
                break
            std = max(std * 0.85, 1e-6)
        for i in range(total_items - freqs.sum()):
            freqs[i % num_keys] += 1
        if self.ensure_hh and freqs.max(initial=0) < required:
            top_k = min(3, num_keys)
            top_idx = np.argsort(freqs)[-top_k:][::-1]
            donors = [i for i in np.argsort(freqs) if i not in set(top_idx)]
            for idx in top_idx:
                if freqs[idx] >= required:
                    continue
                need = required - freqs[idx]
                for d in donors:
                    if need <= 0:
                        break
                    if freqs[d] <= 0:
                        continue
                    take = min(freqs[d], need)
                    freqs[d] -= take
                    freqs[idx] += take
                    need -= take
        return {f'key_{i+1}': freq for i, freq in enumerate(freqs)}

class ZipfianDistributionGenerator(FrequencyDistributionGenerator):
    def __init__(self, s=1.2): self.s = s
    def generate(self, total_items: int, num_keys: int):
        ranks = np.arange(1, num_keys + 1)
        weights = 1 / ranks**self.s
        weights /= weights.sum()
        freqs = weights * total_items
        freqs = np.floor(freqs).astype(int)
        for i in range(total_items - freqs.sum()):
            freqs[i % num_keys] += 1
        return {f'key_{i+1}': freq for i, freq in enumerate(freqs)}

class HHBackgroundDistributionGenerator(FrequencyDistributionGenerator):
    def __init__(
        self,
        n=200,
        num_hh=20,
        hh_min_ratio=1.05,
        hh_max_ratio=2.0,
        background_keys=None,
        background_shape="uniform",
    ):
        self.n = n
        self.num_hh = num_hh
        self.hh_min_ratio = hh_min_ratio
        self.hh_max_ratio = hh_max_ratio
        self.background_keys = background_keys
        self.background_shape = background_shape

    def generate(self, total_items: int, num_keys: int):
        num_keys = max(1, int(num_keys))
        threshold = total_items / max(1, self.n)
        num_hh = max(0, min(int(self.num_hh), num_keys))
        background_keys = self.background_keys
        if background_keys is None:
            background_keys = num_keys - num_hh
        background_keys = max(0, min(int(background_keys), num_keys - num_hh))

        counts = [0] * num_keys
        if num_hh > 0:
            if num_hh == 1:
                ratios = [self.hh_max_ratio]
            else:
                ratios = np.linspace(self.hh_max_ratio, self.hh_min_ratio, num_hh)
            for i, ratio in enumerate(ratios):
                counts[i] = max(int(math.floor(threshold)) + 1, int(round(threshold * ratio)))

        hh_total = sum(counts[:num_hh])
        if hh_total > total_items:
            scale = total_items / float(hh_total)
            for i in range(num_hh):
                counts[i] = max(int(math.floor(threshold)) + 1, int(math.floor(counts[i] * scale)))
            hh_total = sum(counts[:num_hh])

        remaining = max(0, total_items - hh_total)
        if background_keys > 0 and remaining > 0:
            max_non_hh = max(1, int(math.floor(threshold)) - 1)
            if self.background_shape == "linear":
                weights = np.linspace(2.0, 1.0, background_keys)
            else:
                weights = np.ones(background_keys)
            weights = weights / weights.sum()
            raw = weights * remaining
            bg = np.floor(raw).astype(int)
            bg = np.minimum(bg, max_non_hh)
            rem = remaining - int(bg.sum())
            order = np.argsort(raw - bg)[::-1]
            idx = 0
            while rem > 0 and len(order) > 0:
                j = int(order[idx % len(order)])
                if bg[j] < max_non_hh:
                    bg[j] += 1
                    rem -= 1
                idx += 1
                if idx > len(order) * (max_non_hh + 1):
                    break
            for j, value in enumerate(bg):
                counts[num_hh + j] = int(value)

        # If the configured background cannot absorb all mass under the
        # non-HH cap, add the residue to the strongest heavy hitters.
        residue = total_items - sum(counts)
        i = 0
        while residue > 0 and num_hh > 0:
            counts[i % num_hh] += 1
            residue -= 1
            i += 1

        return {f'key_{i+1}': freq for i, freq in enumerate(counts) if freq > 0}


class PersistentBoundaryDistributionGenerator(FrequencyDistributionGenerator):
    def __init__(
        self,
        n=200,
        num_hh=20,
        num_near_nonhh=60,
        hh_min_ratio=1.03,
        hh_max_ratio=1.15,
        nonhh_min_ratio=0.85,
        nonhh_max_ratio=0.99,
        drift_amplitude=0.025,
        drift_period=48,
        noise_frac=0.008,
        background_shape="uniform",
        seed=0,
    ):
        self.n = max(1, int(n))
        self.num_hh = max(0, int(num_hh))
        self.num_near_nonhh = max(0, int(num_near_nonhh))
        self.hh_min_ratio = float(hh_min_ratio)
        self.hh_max_ratio = float(hh_max_ratio)
        self.nonhh_min_ratio = float(nonhh_min_ratio)
        self.nonhh_max_ratio = float(nonhh_max_ratio)
        self.drift_amplitude = max(0.0, float(drift_amplitude))
        self.drift_period = max(1, int(drift_period))
        self.noise_frac = max(0.0, float(noise_frac))
        self.background_shape = background_shape
        self.seed = int(seed)
        self._window = 0
        self._profiles = {}

    def _profile(self, total_items: int, num_keys: int):
        key = (int(total_items), int(num_keys))
        if key in self._profiles:
            return self._profiles[key]

        total_items = max(1, int(total_items))
        num_keys = max(1, int(num_keys))
        boundary_count = min(num_keys, self.num_hh + self.num_near_nonhh)
        num_hh = min(self.num_hh, boundary_count)
        num_near = min(self.num_near_nonhh, boundary_count - num_hh)
        rng = random.Random(self.seed + 7919 * num_keys + 104729 * total_items)

        hh_ratios = []
        if num_hh > 0:
            if num_hh == 1:
                hh_ratios = [(self.hh_min_ratio + self.hh_max_ratio) / 2.0]
            else:
                hh_ratios = np.linspace(self.hh_max_ratio, self.hh_min_ratio, num_hh).tolist()
            rng.shuffle(hh_ratios)

        nonhh_ratios = []
        if num_near > 0:
            if num_near == 1:
                nonhh_ratios = [(self.nonhh_min_ratio + self.nonhh_max_ratio) / 2.0]
            else:
                nonhh_ratios = np.linspace(self.nonhh_max_ratio, self.nonhh_min_ratio, num_near).tolist()
            rng.shuffle(nonhh_ratios)

        phases = [rng.random() for _ in range(boundary_count)]
        noise_seeds = [rng.randrange(1 << 30) for _ in range(boundary_count)]
        self._profiles[key] = (num_hh, num_near, hh_ratios, nonhh_ratios, phases, noise_seeds)
        return self._profiles[key]

    def generate(self, total_items: int, num_keys: int):
        total_items = max(1, int(total_items))
        num_keys = max(1, int(num_keys))
        threshold = total_items / float(self.n)
        counts = [0] * num_keys
        num_hh, num_near, hh_ratios, nonhh_ratios, phases, noise_seeds = self._profile(total_items, num_keys)

        def drifted_ratio(base_ratio: float, idx: int, *, above_threshold: bool) -> float:
            phase = phases[idx]
            periodic = self.drift_amplitude * math.sin(
                2.0 * math.pi * (float(self._window) / float(self.drift_period) + phase)
            )
            rng = random.Random(noise_seeds[idx] + self._window * 1009)
            noise = rng.uniform(-self.noise_frac, self.noise_frac)
            ratio = base_ratio * (1.0 + periodic + noise)
            if above_threshold:
                ratio = max(1.001, ratio)
            else:
                ratio = min(0.999, ratio)
            return max(0.0, ratio)

        for i, ratio in enumerate(hh_ratios):
            counts[i] = max(int(math.floor(threshold)) + 1, int(round(threshold * drifted_ratio(ratio, i, above_threshold=True))))

        for j, ratio in enumerate(nonhh_ratios):
            idx = num_hh + j
            cap = max(0, int(math.floor(threshold)) - 1)
            counts[idx] = min(cap, int(round(threshold * drifted_ratio(ratio, idx, above_threshold=False))))

        boundary_total = sum(counts)
        if boundary_total > total_items:
            excess = boundary_total - total_items
            order = sorted(range(num_hh + num_near), key=lambda i: counts[i], reverse=True)
            oi = 0
            while excess > 0 and order:
                idx = order[oi % len(order)]
                floor_count = int(math.floor(threshold)) + 1 if idx < num_hh else 0
                if counts[idx] > floor_count:
                    counts[idx] -= 1
                    excess -= 1
                oi += 1
                if oi > total_items * max(1, len(order)):
                    break

        remaining = max(0, total_items - sum(counts))
        bg_start = num_hh + num_near
        bg_count = max(0, num_keys - bg_start)
        if remaining > 0 and bg_count > 0:
            if self.background_shape == "linear":
                weights = np.linspace(2.0, 1.0, bg_count)
            else:
                weights = np.ones(bg_count)
            weights = weights / weights.sum()
            raw = weights * remaining
            bg = np.floor(raw).astype(int)
            residue = remaining - int(bg.sum())
            order = np.argsort(raw - bg)[::-1]
            for k in range(residue):
                bg[int(order[k % len(order)])] += 1
            for j, value in enumerate(bg):
                counts[bg_start + j] = int(value)
        elif remaining > 0 and num_hh > 0:
            for k in range(remaining):
                counts[k % num_hh] += 1

        self._window += 1
        return {f'key_{i+1}': freq for i, freq in enumerate(counts) if freq > 0}

class FlattenedHHDistributionGenerator(FrequencyDistributionGenerator):
    def __init__(self, n=10, num_hh=5, flatness=0.5):
        self.n = n; self.num_hh = num_hh; self.flatness = flatness

    def _generate_flattened_hh_distribution(self, n, total_items, num_hh, flatness):
        if num_hh > n - 1: num_hh = n - 1
        min_hh = total_items // n + 1
        max_possible = 2 * (total_items - 1) / num_hh - min_hh
        max_hh = int(min_hh + flatness * (max_possible - min_hh))
        hh_items = [max(max_hh - i * (max_hh - min_hh) // max(num_hh - 1, 1), min_hh) for i in range(num_hh)]
        total_hh = sum(hh_items)
        non_hh_items = total_items - total_hh
        return hh_items, non_hh_items

    def _generate_smooth_linear_non_hh(self, non_hh, min_hh, min_freq=1):
        if non_hh <= 0:
            return []
        top = min_hh - 1
        if top < min_freq:
            return []
        k = 0
        while True:
            k += 1
            seq = np.floor(np.linspace(top, min_freq, k)).astype(int)
            seq = seq[seq > 0]
            if seq.sum() > non_hh:
                k -= 1
                break
        seq = np.floor(np.linspace(top, min_freq, k)).astype(int)
        seq = seq[seq > 0]
        if len(seq) == 0:
            return []
        seq[0] = min(seq[0], top)
        diff = non_hh - seq.sum()
        idx = 1
        while diff != 0:
            if diff > 0:
                seq[idx] += 1
                diff -= 1
            elif diff < 0 and seq[idx] > min_freq:
                seq[idx] -= 1
                diff += 1
            idx = (idx + 1) % len(seq)
        return seq.tolist()

    def generate(self, total_items: int, num_keys: int):
        num_keys = max(1, int(num_keys))
        hh_count = max(1, min(self.num_hh, max(1, num_keys - 1)))
        tail_count = num_keys - hh_count
        min_hh = total_items // self.n + 1

        def _linear_vals(count: int, top: float, bottom: float) -> list[int]:
            if count <= 0:
                return []
            if count == 1:
                return [int(round(top))]
            arr = np.linspace(top, bottom, count)
            return np.round(arr).astype(int).tolist()

        # Single straight-line across all keys.
        if num_keys == 1:
            vals = [min_hh]
        else:
            N = num_keys
            hh_idx = hh_count - 1
            denom = N * (hh_idx - (N - 1) / 2.0)
            if denom == 0:
                slope = 0.0
            else:
                slope = (total_items - N * min_hh) / denom
            if slope <= 0:
                slope = 0.0
            top = min_hh + slope * hh_idx
            vals = [int(round(top - slope * i)) for i in range(N)]
            vals = [max(1, v) for v in vals]

        # If the straight-line sum exceeds total_items, crop from the tail.
        excess = sum(vals) - total_items
        if excess > 0:
            i = len(vals) - 1
            while excess > 0 and i >= 0:
                if vals[i] > 0:
                    take = min(vals[i], excess)
                    vals[i] -= take
                    excess -= take
                i -= 1

        np.random.shuffle(vals)
        return {f'key_{i+1}': f for i, f in enumerate(vals)}
