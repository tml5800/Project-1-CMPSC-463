"""
visualizer.py
--------------
Generates plots of time-series clusters and subarray intervals.
"""

import matplotlib.pyplot as plt
import numpy as np

class Visualizer:
    def plot_cluster(self, cluster, title="Cluster Visualization"):
        """
        Plots multiple time-series in a single figure.
        """
        plt.figure(figsize=(8, 4))
        for ts in cluster:
            plt.plot(ts)
        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel("Signal Value")
        plt.show()

    def plot_subarray(self, signal, start, end, title="Max Subarray Region"):
        """
        Highlights the most active region found by Kadane’s algorithm.
        """
        plt.figure(figsize=(8, 3))
        plt.plot(signal, color='gray')
        plt.plot(range(start, end + 1), signal[start:end + 1], color='red', linewidth=2)
        plt.title(title)
        plt.show()
