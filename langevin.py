from __future__ import annotations

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
# from scipy.optimize import approx_fprime
from tqdm import tqdm

from torch.distributions.normal import Normal
from torch.distributions.gamma import Gamma
from torch.distributions.log_normal import LogNormal
from torch.distributions.studentT import StudentT

np.random.seed(0)
torch.manual_seed(0)

def euler_maruyama_method(X, f_x, g_x, dt):
    return X + f_x * dt + g_x * np.sqrt(dt) * torch.normal(mean=torch.tensor([0.0]), std=torch.tensor([1.0]))

def milstein_method(X, f_x, g_x, dt):
    dW = np.sqrt(dt) * torch.normal(mean=torch.tensor([0.0]), std=torch.tensor([1.0]))
    pass

class LangevinAlgoSampler(object):
    def __init__(
            self,
            X_zero,
            target_log_func,
            method=euler_maruyama_method,
            step_size = 1e-1,
            max_itr = 1e4,
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
        while not self.test_proposal_value(X_prime=X_prime):
            X_prime = self.method(self.X, 0.5*grad, 1, self.step_size)
        self.X.data = X_prime
        return self.X.clone().detach()

    def run(self):
        samples = []
        for _ in tqdm(range(int(self.max_itr))):
            samples.append(self.step().numpy())
        return np.array(samples[500:])
    
    def test_proposal_value(self, X_prime):
        try:
            self.target_log_func(X_prime)
            return True
        except:
            return False
        
class MetropolisAdjLangevinAlgoSampler(object):
    def __init__(
            self,
            X_zero,
            target_log_func,
            method=euler_maruyama_method,
            step_size = 1e-1,
            max_itr = 1e4,
        ) -> None:
        self.target_log_func = target_log_func
        self.step_size = step_size
        self.max_itr = max_itr
        self.method = method

        self.X = torch.zeros(X_zero.shape, requires_grad=True)
        # self.X_prime = torch.zeros(X_zero.shape, requires_grad=True)
        
        self.X.data = X_zero.data
        # self.X_prime.data = X_zero.data
        
        # self.target_log_prob = torch.zeros([1])
        # self.target_log_prob.data = target_log_func(self.X)

        # self.grad = torch.zeros(initial_values.shape)
        # self.grad.data = torch.autograd.grad(outputs=self.target_log_prob, 
        #                                      inputs=[self.X], 
        #                                      create_graph=False)

    def run(self):
        samples = []
        for _ in tqdm(range(int(self.max_itr))):
            samples.append(self.step().numpy())
        return np.array(samples[500:])
        
    def step(self):
        target_log_prob = self.target_log_func(self.X)
        self.grad = torch.zeros(self.X.shape)
        self.grad.data = torch.autograd.grad(
            outputs=target_log_prob, 
            inputs=self.X, 
            create_graph=False
        )[0].data
        self.X_prime = torch.zeros(self.X.shape, requires_grad=True)
        self.X_prime.data = self.method(self.X, 0.5*self.grad, 1, self.step_size)
        # while (not self.test_proposal_value()):
            # self.X_prime.data = self.method(self.X, 0.5*self.grad, 1, self.step_size)
        # if self.accept():
            # self.X.data = self.X_prime
            
        if self.test_proposal_value() and self.accept():
            self.X.data = self.X_prime
        return self.X.clone().detach()
    
    def test_proposal_value(self):
        try:
            self.target_log_func(self.X_prime)
            return True
        except:
            return False
        
    def accept(self):
        to_sum = [
            self.target_log_func(self.X_prime),
            - self.target_log_func(self.X),
            self.proposal_log_density(self.X, self.X_prime),
            - self.proposal_log_density(self.X_prime, self.X),
        ]
        alpha = min(1.0, torch.exp(torch.sum(torch.stack(to_sum))))
        return torch.rand([1]) <= alpha
        
    def proposal_log_density(self, X1, X2):
        log_prob = self.target_log_func(X2)
        grad = torch.zeros(X2.shape)
        grad.data = torch.autograd.grad(
            outputs=log_prob,
            inputs=X2, 
            create_graph=False
        )[0].data
        return Normal(loc=X2 + self.step_size*0.5*grad.data, scale=np.sqrt(self.step_size)).log_prob(X1)

if __name__ == '__main__':
    mu, std = 1.0, 2.0
    alpha, beta = 20.0, 50.0
    df = 4.0

    target_dists = {
        # 'Normal': Normal(loc=torch.tensor([mu]), scale=torch.tensor([std])),
        'Gamma': Gamma(concentration=torch.tensor([alpha]), rate=torch.tensor([beta])),
        'Log-Normal': LogNormal(loc=torch.tensor([mu]), scale=torch.tensor([std])),
        # 'StudentT': StudentT(df=torch.tensor([df]))
    }

    samples = {}
    for name_dist, target_dist in target_dists.items():
        # sampler = MetropolisAdjLangevinAlgoSampler(
        sampler = LangevinAlgoSampler(
            X_zero=torch.tensor(1.0),
            target_log_func=target_dist.log_prob,
        )
        samples[name_dist] = sampler.run().T.squeeze()
    samples = pd.DataFrame(samples)

    sample_result = pd.concat([samples.mean(), samples.std()], axis=1).T
    sample_result.index = 'Mean (Sample)', 'Std (Sample)'
    true_result = pd.DataFrame({name: [dist.mean.numpy().item(), dist.stddev.numpy().item()] for name, dist in target_dists.items()})
    true_result.index = 'Mean (True)', 'Std (True)'

    result = pd.concat([sample_result, true_result]).sort_index()
    print(result)







    # samples = {}
    # for name_dist, target_dist in target_dists.items():
    #     sampler = LangevinAlgoSampler(
    #         X_zero=torch.tensor(1.0),
    #         target_log_func=target_dist.log_prob,
    #         max_itr=1e5,
    #         step_size=0.1,
    #     )
    #     samples[name_dist] = sampler.run().T.squeeze()
    # samples = pd.DataFrame(samples)

    # sample_result = pd.concat([samples.mean(), samples.std()], axis=1).T
    # sample_result.index = 'Mean (Sample)', 'Std (Sample)'
    # true_result = pd.DataFrame({name: [dist.mean.numpy().item(), dist.stddev.numpy().item()] for name, dist in target_dists.items()})
    # true_result.index = 'Mean (True)', 'Std (True)'

    # result = pd.concat([sample_result, true_result]).sort_index()
    # print(result)

    # samples = {}
    # for name_dist, target_dist in target_dists.items():
    #     sampler = MetropolisAdjLangevinAlgoSampler(
    #         X_zero=torch.tensor(1.0),
    #         target_log_func=target_dist.log_prob,
    #         max_itr=1e5,
    #         step_size=0.1,
    #     )
    #     samples[name_dist] = sampler.run().T.squeeze()
    # samples = pd.DataFrame(samples)

    # sample_result = pd.concat([samples.mean(), samples.std()], axis=1).T
    # sample_result.index = 'Mean (Sample)', 'Std (Sample)'
    # true_result = pd.DataFrame({name: [dist.mean.numpy().item(), dist.stddev.numpy().item()] for name, dist in target_dists.items()})
    # true_result.index = 'Mean (True)', 'Std (True)'

    # result = pd.concat([sample_result, true_result]).sort_index()
    # print(result)