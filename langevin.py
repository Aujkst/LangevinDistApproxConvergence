from __future__ import annotations
from typing import Callable

import numpy as np
from tqdm import tqdm
from scipy import stats
from distributions import Dist

def _log_norm_density(x: float, mean: float, var: float) -> float:
    return - 0.5 * (x - mean)**2 / var

def euler_maruyama_method(X, f_x, g_x, dt, U1=None, *args, **kwargs):
    U1 = U1 if U1 is not None else np.random.normal(loc=0.0, scale=1.0)
    dW = np.sqrt(dt) * U1
    return X + f_x * dt + g_x * dW

def strong_order_taylor_method(X, f_x, df_x, ddf_x, g_x, dt, U1=None, U2=None, *args, **kwargs):
    U1 = U1 if U1 is not None else np.random.normal(loc=0.0, scale=1.0)
    U2 = U2 if U2 is not None else np.random.normal(loc=0.0, scale=1.0)
    dW = np.sqrt(dt) * U1
    dZ = 0.5 * dt**1.5 * (U1 + U2 / np.sqrt(3))

    to_sum = [X]
    to_sum.append(f_x * dt + g_x * dW)
    to_sum.append(df_x * g_x * dZ)
    to_sum.append(0.5 * (f_x * df_x + 0.5 * g_x**2 * ddf_x) * dt**2)
    return np.sum(to_sum)

class LangevinAlgoSampler(object):
    def __init__(
            self,
            X_zero: float,
            target_dist: Dist,
            step_method: str,
            step_size: float = 0.1,
            max_itr: float = 1e4,
            U: np.ndarray = None,
        ) -> None:
        self.target_dist = target_dist
        self.step_size = step_size
        self.max_itr = max_itr
        self.step_method = step_method
        if step_method == 'euler_maruyama_method':
            self.method = euler_maruyama_method
        elif step_method == 'strong_order_taylor_method':
            self.method = strong_order_taylor_method
        else:
            raise
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
            step_method: str, 
            step_size: float = 0.1,
            max_itr: float = 10000, 
            U: np.ndarray = None,
        ) -> None:
        super().__init__(X_zero, target_dist, step_method, step_size, max_itr, U)
        self.accept_res = []
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
            self.accept_res.append(self.accept())
            self.X = self.X_prime
        return self.X, grad
    
    def proposal_log_density(self, X1, X2):
        if self.step_method == 'euler_maruyama_method':
            grad = self.target_dist.grad_log_pdf(X2)
            mean = X2 + 0.5 * self.step_size * grad
            var = self.step_size
            return _log_norm_density(x=X1, mean=mean, var=var)
        if self.step_method == 'strong_order_taylor_method':
            grad = self.target_dist.grad_log_pdf(X2)
            ggrad = self.target_dist.ggrad_log_pdf(X2)
            gggrad = self.target_dist.gggrad_log_pdf(X2)
            coef_u1 = np.sqrt(self.step_size) + 0.25 * ggrad * self.step_size**1.5
            coef_u2 = 0.25 * ggrad * self.step_size**1.5 / np.sqrt(3)
            var = coef_u1**2 + coef_u2**2
            mean = np.sum([
                X2, 0.5 * self.step_size * grad,
                0.125 * self.step_size**2 * (grad * ggrad + gggrad)])
            return _log_norm_density(x=X1, mean=mean, var=var)

    def accept(self):
        to_sum = [
            self.target_dist.log_pdf(self.X_prime),
            - self.target_dist.log_pdf(self.X),
            self.proposal_log_density(self.X, self.X_prime),
            - self.proposal_log_density(self.X_prime, self.X),
        ]
        alpha = min(1.0, np.exp(np.sum(to_sum)))
        return np.random.rand() <= alpha