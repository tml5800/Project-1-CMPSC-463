import math

def closest_pair(indices, series, dist_fn):
    best = (None, None, math.inf)
    n = len(indices)
    for a in range(n):
        ia = indices[a]
        for b in range(a + 1, n):
            ib = indices[b]
            d = dist_fn(series[ia], series[ib])
            if d < best[2]:
                best = (ia, ib, d)
    return best
