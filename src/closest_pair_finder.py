"""
closest_pair_finder.py
----------------------
Finds the closest pair of time-series segments using correlation distance.
"""

import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

class ClosestPairFinder:
    def __init__(self, distance_type='dtw'):
        self.distance_type = distance_type

    def compute_distance(self, a, b):
        """
        Compute DTW or Euclidean distance between two signals.
        """
        if self.distance_type == 'dtw':
            dist, _ = fastdtw(a, b, dist=euclidean)
            return dist
        return np.linalg.norm(a - b)

    def find_closest_pair(self, cluster):
        """
        Identify the most similar pair within a cluster.
        Returns:
            tuple: (index_pair, min_distance)
        """
        min_dist = float('inf')
        best_pair = (None, None)

        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                dist = self.compute_distance(cluster[i], cluster[j])
                if dist < min_dist:
                    min_dist, best_pair = dist, (i, j)

        return best_pair, min_dist
