import numpy as np

def kadane_max_subarray(arr):
    max_sum = -1e18
    cur_sum = 0.0
    start = best_l = best_r = 0
    for i, v in enumerate(arr):
        if cur_sum <= 0:
            cur_sum = v
            start = i
        else:
            cur_sum += v
        if cur_sum > max_sum:
            max_sum = cur_sum
            best_l, best_r = start, i
    return best_l, best_r, float(max_sum)

def max_activity_window(x):
    dx = np.diff(x)
    l, r, s = kadane_max_subarray(dx)
    # convert from diff indices to original indices (window is l..r on diffs -> l..r+1 on x)
    return l, r + 1, s
