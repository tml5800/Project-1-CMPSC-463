import os, json, csv
import matplotlib.pyplot as plt

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def summarize_clusters(outdir, leaves):
    ensure_dir(outdir)
    with open(os.path.join(outdir, 'clusters.json'), 'w') as f:
        json.dump(leaves, f)

def write_closest_pairs(outdir, rows):
    ensure_dir(outdir)
    with open(os.path.join(outdir, 'closest_pairs.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['cluster_id', 'i', 'j', 'distance'])
        w.writerows(rows)

def write_kadane(outdir, rows):
    ensure_dir(outdir)
    with open(os.path.join(outdir, 'kadane_summary.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['series_id', 'start', 'end', 'sum'])
        w.writerows(rows)

def plot_cluster_examples(outdir, series, leaves, per_cluster=6):
    ensure_dir(outdir)
    for cid, idxs in enumerate(leaves):
        pick = idxs[:per_cluster]
        if not pick:
            continue
        plt.figure(figsize=(10, 6))
        for k in pick:
            plt.plot(series[k], alpha=0.8)
        plt.title(f'Cluster {cid} – {len(idxs)} series (first {len(pick)})')
        plt.xlabel('t')
        plt.ylabel('value')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'fig_cluster_{cid}.png'))
        plt.close()

def plot_kadane_bands(outdir, series, windows, k=6):
    ensure_dir(outdir)
    for pid, (i, (s, e, _)) in enumerate(list(enumerate(windows))[:k]):
        plt.figure(figsize=(8, 3))
        x = series[i]
        plt.plot(x)
        plt.axvspan(s, e, alpha=0.2)
        plt.title(f'Series {i}: max-activity window [{s},{e}]')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'fig_kadane_{i}.png'))
        plt.close()
