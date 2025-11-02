"""
main.py
--------
Main entry point for Time-Series Clustering and Segment Analysis on PulseDB.

Author: Tommy Lu
Course: CMPSC 488 – Advanced Algorithms
Instructor: Dr. Janghoon Yang
GitHub: https://github.com/tml5800/Project-1-CMPSC-463
"""

from src.data_loader import DataLoader
from src.divide_conquer_cluster import DivideConquerCluster
from src.closest_pair_finder import ClosestPairFinder
from src.max_subarray_analyzer import MaxSubarrayAnalyzer
from src.visualizer import Visualizer
import numpy as np

def main():
    print("=== PulseDB Time-Series Clustering System ===\n")

    # Step 1: Load Data
    data_loader = DataLoader(data_dir="Segment_Files")  # adjust path if needed
    segments = data_loader.load_segments(limit=10)  # demo: load 10 segments
    print(f"Loaded {len(segments)} segments.\n")

    # Step 2: Divide-and-Conquer Clustering
    clusterer = DivideConquerCluster(threshold=0.8)
    clusters = clusterer.recursive_cluster(segments)
    print(f"Formed {len(clusters)} clusters.\n")

    # Step 3: Closest Pair Analysis
    cp_finder = ClosestPairFinder(distance_type='dtw')
    for idx, cluster in enumerate(clusters):
        if len(cluster) > 1:
            pair, dist = cp_finder.find_closest_pair(cluster)
            print(f"Cluster {idx+1}: Closest pair {pair} (Distance: {dist:.3f})")

    # Step 4: Kadane’s Algorithm for Each Segment
    kadane = MaxSubarrayAnalyzer()
    for i, seg in enumerate(segments[:3]):  # demo on first 3
        max_sum, start, end = kadane.kadane(seg)
        print(f"Segment {i+1}: Max subarray sum={max_sum:.2f}, range=({start}-{end})")

    # Step 5: Visualization
    vis = Visualizer()
    if clusters and len(clusters[0]) > 0:
        vis.plot_cluster(clusters[0], title="Sample Cluster Visualization")

    print("\n=== Analysis Complete ===")

if __name__ == "__main__":
    main()
