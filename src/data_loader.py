"""
data_loader.py
---------------
Loads PulseDB time-series segments into NumPy arrays.
Author: Tommy Lu
Course: CMPSC 488 – Advanced Algorithms
Instructor: Dr. Janghoon Yang
"""

import os
import numpy as np

class DataLoader:
    def __init__(self, data_dir):
        """
        Initialize the data loader with the dataset directory.
        """
        self.data_dir = data_dir

    def load_segments(self, limit=None):
        """
        Loads time-series segments from text or CSV files.
        Args:
            limit (int): optional limit of number of segments to load
        Returns:
            list of np.ndarray
        """
        segments = []
        files = [f for f in os.listdir(self.data_dir) if f.endswith(('.txt', '.csv'))]

        if limit:
            files = files[:limit]

        for file in files:
            try:
                data = np.loadtxt(os.path.join(self.data_dir, file), delimiter=',')
                segments.append(data)
            except Exception as e:
                print(f"Error loading {file}: {e}")
        return segments
