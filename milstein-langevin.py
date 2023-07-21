from __future__ import annotations

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Callable

from scipy import stats
from torch.distributions.beta import Beta
from torch.distributions.gamma import Gamma
from torch.distributions.normal import Normal
from torch.distributions.laplace import Laplace
from torch.distributions.studentT import StudentT
from torch.distributions.log_normal import LogNormal
from torch.distributions.transforms import ExpTransform, SigmoidTransform
from torch.distributions.transformed_distribution import TransformedDistribution

plt.rcParams['text.usetex'] = True

np.random.seed(0)
torch.manual_seed(0)

def sigmoid_function(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z)) 

def euler_maruyama_method(X, f_x, g_x, dt):
    return X + f_x * dt + g_x * np.sqrt(dt) * torch.normal(mean=torch.tensor([0.0]), std=torch.tensor([1.0]))

def milstein_method(X, f_x, g_x, gg_x, dt):
    dW = np.sqrt(dt) * torch.normal(mean=torch.tensor([0.0]), std=torch.tensor([1.0]))
    return X + f_x * dt + g_x * dW + gg_x * (dW ** 2 + - dt) / 2

# def runge_kutta_method(X, f_x, g_x, dt):
#     X_hat = X + f_x * dt + g_x * dt**(1/2)
#     return euler_maruyama_method(X, f_x, g_x, dt) + X_hat * () / 2

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

class LangevinAlgoSampler(object):
    def __init__(
            self,
            X_zero,
            target_log_func,
            method=euler_maruyama_method,
            step_size=1e-1,
            max_itr=1e4,
        ) -> None:
        self.target_log_func = target_log_func
        self.step_size = step_size
        self.max_itr = max_itr
        self.method = method

        self.X = torch.zeros(X_zero.shape, requires_grad=True)
        self.X.data = X_zero.data
        
    def step(self):
        target_log_prob = self.target_log_func(self.X)
        grad = torch.zeros(self.X.shape)
        grad.data = torch.autograd.grad(
            outputs=target_log_prob, 
            inputs=self.X, 
            create_graph=False
        )[0].data
        X_prime = self.method(self.X, 0.5*grad*self.X.data**2, self.X.data, self.step_size)
        self.X.data = X_prime
        return self.X.clone().detach(), grad.clone()

    def run(self):
        samples = []
        grads = []
        for _ in tqdm(range(int(self.max_itr))):
            sample, grad = self.step()
            samples.append(sample.numpy().item())
            grads.append(grad.numpy().item())
        return np.asarray(samples), np.asarray(grads)

if __name__ == '__main__':

    file_path = os.getcwd()

    step_size = 1e-1
    max_itr = 1e4
    t = np.arange(step_size, (max_itr + 1.0) * step_size, step_size)

    alpha1, beta1 = 2.0, 6.0 # Gamma dist
    
    gamma_dist = Gamma(concentration=torch.tensor([alpha1]), rate=torch.tensor([beta1]))
    target_dist = TransformedDistribution(base_distribution=gamma_dist, transforms=ExpTransform().inv)

    sampler = LangevinAlgoSampler(
        X_zero=torch.tensor(1.0),
        target_log_func=target_dist.log_prob,
        step_size=step_size,
        max_itr=max_itr,
    )

    sample, grad = sampler.run()