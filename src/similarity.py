import numpy as np
from .preprocess import z_norm

def euclid(a, b):
    a, b = z_norm(a), z_norm(b)
    return np.linalg.norm(a - b)

def corr_dist(a, b):
    a, b = z_norm(a), z_norm(b)
    c = np.corrcoef(a, b)[0, 1]
    return 1.0 - float(c)

def dtw_banded(a, b, band=0.05):
    a, b = z_norm(a), z_norm(b)
    n, m = len(a), len(b)
    w = int(max(band * max(n, m), abs(n - m)))
    INF = 1e18
    dp = np.full((n + 1, m + 1), INF)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        j_start = max(1, i - w)
        j_end = min(m, i + w)
        for j in range(j_start, j_end + 1):
            cost = (a[i - 1] - b[j - 1]) ** 2
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(np.sqrt(dp[n, m]))
