from __future__ import annotations
import random

class DnCClusterer:
    def __init__(self, dist_fn, max_depth=6, min_size=20, impurity_eps=None, rand_seed=7):
        self.dist_fn = dist_fn
        self.max_depth = max_depth
        self.min_size = min_size
        self.impurity_eps = impurity_eps
        self.rand = random.Random(rand_seed)
        self.tree = None  # (indices, left, right, e1, e2)

    def _approx_medoid(self, indices, series, probes=8):
        cand = self.rand.sample(indices, min(probes, len(indices)))
        best_i, best_val = None, 1e18
        for i in cand:
            s = 0.0
            for j in cand:
                if i == j:
                    continue
                s += self.dist_fn(series[i], series[j])
            if s < best_val:
                best_val, best_i = s, i
        return best_i if best_i is not None else indices[0]

    def _farthest_from(self, anchor, indices, series):
        best_j, best_d = None, -1.0
        a = series[anchor]
        for j in indices:
            if j == anchor:
                continue
            d = self.dist_fn(a, series[j])
            if d > best_d:
                best_d, best_j = d, j
        return best_j

    def _impurity(self, indices, series):
        if len(indices) < 2:
            return 0.0
        m = self._approx_medoid(indices, series, probes=min(8, len(indices)))
        s = 0.0
        for j in indices:
            if j == m:
                continue
            s += self.dist_fn(series[m], series[j])
        return s / max(1, len(indices) - 1)

    def _build(self, indices, series, depth):
        if depth >= self.max_depth or len(indices) <= self.min_size:
            return (indices, None, None, None, None)
        if self.impurity_eps is not None and self._impurity(indices, series) <= self.impurity_eps:
            return (indices, None, None, None, None)
        e1 = self._approx_medoid(indices, series)
        e2 = self._farthest_from(e1, indices, series)
        if e2 is None:
            return (indices, None, None, e1, None)
        L, R = [], []
        for k in indices:
            d1 = self.dist_fn(series[k], series[e1])
            d2 = self.dist_fn(series[k], series[e2])
            (L if d1 <= d2 else R).append(k)
        if not L or not R:
            return (indices, None, None, e1, e2)
        left = self._build(L, series, depth + 1)
        right = self._build(R, series, depth + 1)
        return (indices, left, right, e1, e2)

    def fit(self, series_list):
        indices = list(range(len(series_list)))
        self.tree = self._build(indices, series_list, 0)
        return self

    def _collect_leaves(self, node, acc):
        indices, left, right, e1, e2 = node
        if left is None and right is None:
            acc.append(indices)
            return
        if left:
            self._collect_leaves(left, acc)
        if right:
            self._collect_leaves(right, acc)

    def leaves(self):
        out = []
        self._collect_leaves(self.tree, out)
        return out
