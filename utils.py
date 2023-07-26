from __future__ import annotations

import numpy as np
from typing import Callable

def euler_maruyama_method(X, f_x, g_x, dt, U1, *args, **kwargs):
    U1 = U1 if U1 is not None else np.random.normal(loc=0.0, scale=1.0)
    dW = np.sqrt(dt) * U1
    return X + f_x * dt + g_x * dW

def strong_order_taylor_method(X, f_x, df_x, ddf_x, g_x, dt, U1, U2, *args, **kwargs):
    
    U1 = U1 if U1 is not None else np.random.normal(loc=0.0, scale=1.0)
    U2 = U2 if U2 is not None else np.random.normal(loc=0.0, scale=1.0)
    dW = np.sqrt(dt) * U1
    dZ = 0.5 * dt**1.5 * (U1 + U2 / np.sqrt(3))

    to_sum = [X]
    to_sum.append(f_x * dt + g_x * dW)
    to_sum.append(df_x * g_x * dZ)
    to_sum.append(0.5 * (f_x * df_x + 0.5 * g_x**2 * ddf_x) * dt**2)
    return np.sum(to_sum)

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