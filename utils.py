from __future__ import annotations

import numpy as np
from typing import Callable

def sigmoid_function(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z)) 

def ecdf(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, counts = np.unique(a, return_counts=True)
    cusum = np.cumsum(counts)
    return x, cusum / cusum[-1]

def ks_distance(
        values: np.ndarray, 
        cprobs: np.ndarray, 
        cdf: Callable[[np.ndarray], np.ndarray],
    ) -> np.float64:
    true_cprobs = cdf(values)
    cprobs = np.insert(cprobs, 0, 0.0)[:-1]
    return np.abs(true_cprobs - cprobs).max()

def kl_divergence(
        values: np.ndarray, 
        cprobs: np.ndarray, 
        cdf: Callable[[np.ndarray], np.ndarray],
    ) -> np.float64:
    P_dx = np.diff(cprobs)
    Q_dx = cdf(values[1:]) - cdf(values[:-1])
    return np.sum(P_dx * np.log(P_dx / Q_dx))