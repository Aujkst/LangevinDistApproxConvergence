from __future__ import annotations
from typing import Callable

import numpy as np
from tqdm import tqdm
from scipy import stats
from distributions import Dist

class LangevinAlgoSampler(object):
    def __init__(
            self,
            X_zero: float,
            target_dist: Dist,
            method: Callable[..., float],
            step_size: float = 0.1,
            max_itr: float = 1e4,
            U: np.ndarray = None,
        ) -> None:
        self.target_dist = target_dist
        self.step_size = step_size
        self.max_itr = max_itr
        self.method = method
        self.X = X_zero
        self._U = U
        
    def step(self, U1=None, U2=None):
        grad = self.target_dist.grad_log_pdf(self.X)
        ggrad = self.target_dist.ggrad_log_pdf(self.X)
        gggrad = self.target_dist.gggrad_log_pdf(self.X)
        X_prime = self.method(
            X=self.X, f_x=0.5*grad, 
            df_x=0.5*ggrad, ddf_x=0.5*gggrad,
            g_x=1.0, dt=self.step_size, U1=U1, U2=U2)
        self.X = X_prime
        return self.X, grad

    def run(self):
        samples = []
        grads = []
        for i in tqdm(range(int(self.max_itr))):
            if self._U is None:
                sample, grad = self.step()
            else:
                sample, grad = self.step(U1=self._U[0, i], U2=self._U[1, i])
            samples.append(sample)
            grads.append(grad)
        return np.asarray(samples), np.asarray(grads)
    
class MetropolisAdjLangevinAlgoSampler(LangevinAlgoSampler):
    def __init__(
            self, 
            X_zero: float, 
            target_dist: Dist, 
            method: Callable[..., float], 
            step_size: float = 0.1,
            max_itr: float = 10000, 
            U: np.ndarray = None,
        ) -> None:
        super().__init__(X_zero, target_dist, method, step_size, max_itr, U)
        pass

    def run(self):
        return super().run()

    def step(self, U1=None, U2=None):
        grad = self.target_dist.grad_log_pdf(self.X)
        ggrad = self.target_dist.ggrad_log_pdf(self.X)
        gggrad = self.target_dist.gggrad_log_pdf(self.X)
        self.X_prime = self.method(
            X=self.X, f_x=0.5*grad, 
            df_x=0.5*ggrad, ddf_x=0.5*gggrad,
            g_x=1.0, dt=self.step_size, U1=U1, U2=U2)
        if self.accept():
            self.X = self.X_prime
        return self.X, grad
    
    def proposal_log_density(self, X1, X2):
        grad = self.target_dist.grad_log_pdf(X2)
        density = stats.norm.pdf(
            x=X1, 
            loc=X2 + 0.5 * self.step_size * grad,
            scale=np.sqrt(self.step_size)
        )
        return np.log(density)

    def accept(self):
        to_sum = [
            self.target_dist.log_pdf(self.X_prime),
            - self.target_dist.log_pdf(self.X),
            self.proposal_log_density(self.X, self.X_prime),
            - self.proposal_log_density(self.X_prime, self.X),
        ]
        alpha = min(1.0, np.exp(np.sum(to_sum)))
        return np.random.rand() <= alpha