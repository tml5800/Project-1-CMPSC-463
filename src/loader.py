from __future__ import annotations
import glob
import csv
import numpy as np

def load_csv_glob(glob_pat: str, length: int | None = None):
    """
    Load multiple CSV files matching a glob pattern.
    Assumes each CSV has a header 'value' and one column of floats.
    Optionally crops/pads each series to the given length.
    """
    paths = sorted(glob.glob(glob_pat))
    series = []
    for p in paths:
        vals = []
        with open(p, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)  # skip header (e.g., 'value')
            for row in reader:
                if not row:
                    continue
                try:
                    vals.append(float(row[0]))
                except ValueError:
                    continue  # skip bad rows if any
        x = np.asarray(vals, dtype=float)
        if length and len(x) != length:
            # crop or pad
            if len(x) > length:
                x = x[:length]
            else:
                x = np.pad(x, (0, length - len(x)))
        series.append(x)
    return series


def make_synthetic_mixture(n_series=1000, length=512, n_groups=3, seed=7):
    """
    Generate synthetic sine/noisy signals for toy experiments.
    """
    rng = np.random.default_rng(seed)
    series = []
    freqs = rng.uniform(0.8, 2.0, size=n_groups)
    for i in range(n_series):
        g = i % n_groups
        t = np.linspace(0, 10, length)
        base = np.sin(2 * np.pi * freqs[g] * t + rng.uniform(0, 2 * np.pi))
        noise = 0.15 * rng.standard_normal(length)
        x = base + noise
        # add bursts to one group
        if g == 1 and rng.random() < 0.7:
            s = rng.integers(length // 4, length // 2)
            e = min(length - 1, s + rng.integers(20, 60))
            x[s:e] += rng.uniform(1.0, 2.0)
        series.append(x.astype(float))
    return series
