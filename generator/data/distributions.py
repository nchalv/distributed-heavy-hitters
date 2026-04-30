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
    def __init__(self, n, center_frac: float = 0.05, std_frac: float = 0.031):
        self.n = n
        self.center_frac = center_frac
        self.std_frac = std_frac
    def generate(self, total_items: int, num_keys: int):
        center = self.center_frac * num_keys
        base_std = max(1e-6, self.std_frac * num_keys)
        x = np.linspace(0, num_keys - 1, num_keys)
        required = int(math.floor(total_items / self.n) + 1) if self.n > 0 else 1
        std = base_std
        freqs = None
        for _ in range(30):
            freqs = np.exp(-0.5 * ((x - center) / std) ** 2)
            freqs = freqs / freqs.sum() * total_items
            freqs = np.floor(freqs).astype(int)
            if freqs.max(initial=0) >= required:
                break
            std = max(std * 0.85, 1e-6)
        for i in range(total_items - freqs.sum()):
            freqs[i % num_keys] += 1
        if freqs.max(initial=0) < required:
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
