"""
divide_conquer_cluster.py
-------------------------
Performs recursive divide-and-conquer clustering on time-series data.
"""

import numpy as np
from scipy.spatial.distance import correlation

class DivideConquerCluster:
    def __init__(self, threshold=0.8):
        self.threshold = threshold

    def similarity(self, a, b):
        """
        Computes correlation-based similarity between two time-series.
        """
        try:
            return 1 - correlation(a, b)
        except Exception:
            return 0.0

    def recursive_cluster(self, segments):
        """
        Recursively splits dataset into clusters based on average similarity.
        """
        if len(segments) <= 2:
            return [segments]

        mid = len(segments) // 2
        left, right = segments[:mid], segments[mid:]

        # Calculate average similarity between left and right halves
        sim_score = np.mean([self.similarity(l, r) for l in left for r in right])

        if sim_score > self.threshold:
            return [segments]  # Cohesive cluster
        else:
            return self.recursive_cluster(left) + self.recursive_cluster(right)
