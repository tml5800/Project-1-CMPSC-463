import numpy as np

def z_norm(x: np.ndarray) -> np.ndarray:
    """
    Normalize a time series to zero mean and unit variance.
    If the standard deviation is extremely small, returns the centered signal.
    """
    mu = x.mean()
    sd = x.std()
    return (x - mu) / (sd if sd > 1e-8 else 1.0)

def detrend(x: np.ndarray) -> np.ndarray:
    """
    Remove a simple linear trend from the series.
    """
    t = np.arange(len(x))
    coeffs = np.polyfit(t, x, 1)  # slope, intercept
    trend = np.polyval(coeffs, t)
    return x - trend

def resample(x: np.ndarray, new_len: int) -> np.ndarray:
    """
    Resample the series to a new length using simple interpolation.
    """
    old_t = np.linspace(0, 1, len(x))
    new_t = np.linspace(0, 1, new_len)
    return np.interp(new_t, old_t, x)
