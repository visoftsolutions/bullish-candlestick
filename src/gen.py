import numpy as np


def gen_values():
    # Synthetic timeseries data
    time = np.arange(365)
    base_values = np.sin(time / 30.0) * 50 + 100
    noise = np.random.normal(0, 8, size=time.shape)
    values = base_values + noise
    return values
