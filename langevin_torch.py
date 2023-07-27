from __future__ import annotations

import torch
import numpy as np
from tqdm import tqdm

def euler_maruyama_method(X, f_x, g_x, dt):
    return X + f_x * dt + g_x * np.sqrt(dt) * np.random.normal()

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
        X_prime = self.method(self.X, 0.5*grad, 1, self.step_size)
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