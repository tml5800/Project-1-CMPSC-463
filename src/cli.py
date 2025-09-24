import os, time, argparse
import numpy as np
from .loader import load_csv_glob, make_synthetic_mixture
from .similarity import euclid, corr_dist, dtw_banded
from .dnc_cluster import DnCClusterer
from .closest_pair import closest_pair
from .kadane import max_activity_window
from .reporting import (
    summarize_clusters,
    write_closest_pairs,
    write_kadane,
    plot_cluster_examples,
    plot_kadane_bands,
)

METRICS = {
    'euclid': euclid,
    'corr': corr_dist,
    'dtw': dtw_banded,
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['toy','data'], required=True)
    ap.add_argument('--glob', default=None)
    ap.add_argument('--signal', default='ABP')
    ap.add_argument('--n_series', type=int, default=1000)
    ap.add_argument('--length', type=int, default=512)
    ap.add_argument('--metric', choices=METRICS.keys(), default='corr')
    ap.add_argument('--band', type=float, default=0.05)
    ap.add_argument('--max-depth', type=int, default=6)
    ap.add_argument('--min-size', type=int, default=20)
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()

    # ---- Load ----
    if args.mode == 'toy':
        series = make_synthetic_mixture(
            args.n_series, args.length, n_groups=3, seed=args.seed
        )
    else:
        assert args.glob, 'Provide --glob for data mode'
        series = load_csv_glob(args.glob, length=args.length)
    print(f'Loaded {len(series)} series of length {len(series[0])}')

    # ---- Cluster ----
    dist_fn = (
        (lambda a, b: dtw_banded(a, b, args.band))
        if args.metric == 'dtw'
        else METRICS[args.metric]
    )
    clus = DnCClusterer(
        dist_fn, max_depth=args.max_depth, min_size=args.min_size
    ).fit(series)
    leaves = clus.leaves()

    # ---- Outputs dir ----
    stamp = time.strftime('%Y%m%d_%H%M%S')
    out = os.path.join('outputs', stamp)
    os.makedirs(out, exist_ok=True)

    summarize_clusters(out, leaves)

    # ---- Closest pairs ----
    cp_rows = []
    for cid, idxs in enumerate(leaves):
        if len(idxs) >= 2:
            i, j, d = closest_pair(idxs, series, dist_fn)
            cp_rows.append((cid, i, j, d))
    write_closest_pairs(out, cp_rows)

    # ---- Kadane windows ----
    kad_rows = []
    windows = []
    for i, x in enumerate(series):
        s, e, val = max_activity_window(x)
        windows.append((s, e, val))
        kad_rows.append((i, s, e, val))
    write_kadane(out, kad_rows)

    # ---- Figures ----
    plot_cluster_examples(out, series, leaves)
    plot_kadane_bands(out, series, windows)

    print('Done. Outputs in', out)


if __name__ == '__main__':
    main()
