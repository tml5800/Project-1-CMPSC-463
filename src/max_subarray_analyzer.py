"""
max_subarray_analyzer.py
------------------------
Implements Kadane’s algorithm for detecting peak activity intervals.
"""

class MaxSubarrayAnalyzer:
    @staticmethod
    def kadane(arr):
        """
        Returns the maximum subarray sum using Kadane's Algorithm.
        """
        max_so_far = max_ending_here = arr[0]
        start = end = s = 0

        for i in range(1, len(arr)):
            if arr[i] > max_ending_here + arr[i]:
                max_ending_here = arr[i]
                s = i
            else:
                max_ending_here += arr[i]

            if max_ending_here > max_so_far:
                max_so_far = max_ending_here
                start, end = s, i

        return max_so_far, start, end
