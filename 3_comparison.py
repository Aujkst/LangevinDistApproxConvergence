from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats

from distributions import NormalDist, StudentTDist, LaplaceDist, GeneralisedLogisticDist
from langevin import LangevinAlgoSampler, MetropolisAdjLangevinAlgoSampler
from utils import (
    ecdf, 
    ks_distance, 
    kl_divergence
)

np.random.seed(1)
plt.rcParams['text.usetex'] = True

if __name__ == '__main__':

    file_path = os.getcwd()

    X_zero = 1.0
    step_size = 1.
    max_itr = 1e5
    t = np.arange(step_size, (max_itr + 1.0) * step_size, step_size)
    U = np.random.normal(loc=0.0, scale=1.0, size=(2, int(max_itr)))
    
    all_dist = {}

    mu, sigma = 1.0, 2.0
    normal_dist = NormalDist(mu=mu, sigma=sigma)
    all_dist[normal_dist.name] = normal_dist
    
    df = 4
    t_dist = StudentTDist(df=df)
    all_dist[t_dist.name] = t_dist
    
    loc, scale = 1.0, 4.0
    lap_dist = LaplaceDist(loc=loc, scale=scale)
    all_dist[lap_dist.name] = lap_dist

    alpha, beta = 3.0, 5.0
    gl_dist = GeneralisedLogisticDist(alpha=alpha, beta=beta)
    beta_cdf = lambda x: stats.beta.cdf(x=x, a=alpha, b=beta)
    all_dist['Beta distribution'] = gl_dist

    samples, grads = {}, {}
    
    for name, _dist in all_dist.items():
        sampler = LangevinAlgoSampler(
            X_zero=X_zero,
            target_dist=_dist,
            step_size=step_size,
            max_itr=max_itr,
            step_method='euler_maruyama_method',
            U=U,
        )
        if name == 'Beta distribution':
            _samples, grads[name] = sampler.run()
            samples[name]  = np.exp(_samples) / (1.0 + np.exp(_samples))
        else:
            samples[name], grads[name] = sampler.run()


    # KS Distance & KL Divergence
    
    point_num = 100
    idx_list = (max_itr * (np.arange(point_num) + 1) / point_num - 1).astype(int)

    results = {}
    for name, _samples in samples.items():
        distances, divergences = [], []
        for idx in tqdm(idx_list):
            values, cprobs = ecdf(_samples[:int(idx)])
            distances.append(ks_distance(
                values=values, 
                cprobs=cprobs, 
                cdf=beta_cdf,
            ))
            divergences.append(kl_divergence(
                values=np.append(-np.inf, values), 
                cprobs=np.append(0.0, cprobs), 
                cdf=beta_cdf,
            ))
        results[name] = {
            'Kolmogorov-Smirnov distance': np.asarray(distances),
            'Kullback-Leibler divergence': np.asarray(divergences),
        }

    fig = plt.figure(figsize=(8, 8))
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)

    for name, result in results.items():
        for (distance_name, distances), ax in zip(result.items(), (ax1, ax2)):
            ax.plot(np.log10(idx_list), np.log10(distances), label=name)
            # ax.plot(idx_list, np.log10(distances), label=name)
            ax.set_title(distance_name)
            ax.set_xlabel(r'$\log_{10}t$')
            ax.set_ylabel(r'$\log_{10}D_t$')

    ax1.grid()
    ax2.grid()
    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    plt.show()