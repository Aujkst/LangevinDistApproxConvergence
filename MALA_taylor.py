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

np.random.seed(0)
plt.rcParams['text.usetex'] = True

if __name__ == '__main__':

    save_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    X_zero = 1.0
    step_size = 1
    max_itr = 1e6
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
    
    model_name = 'MALA_taylor'
    for name, _dist in all_dist.items():
        sampler = MetropolisAdjLangevinAlgoSampler(
            X_zero=X_zero,
            target_dist=_dist,
            step_size=step_size,
            max_itr=max_itr,
            step_method='strong_order_taylor_method',
            U=U,
        )
        if name == 'Beta distribution':
            _samples, grads[name] = sampler.run()
            samples[name]  = np.exp(_samples) / (1.0 + np.exp(_samples))
        else:
            samples[name], grads[name] = sampler.run()
            
    pd.DataFrame(samples).agg(['mean', 'std']).T.to_csv(
        os.path.join(save_path, f'{model_name}-stats.csv')
    )

    # Sample path and gradients

    fig = plt.figure(figsize=(8, 8))
    ax1 = plt.subplot(4, 2, 1)
    ax2 = plt.subplot(4, 2, 2)
    ax3 = plt.subplot(4, 2, 3)
    ax4 = plt.subplot(4, 2, 4)
    ax5 = plt.subplot(4, 2, 5)
    ax6 = plt.subplot(4, 2, 6)
    ax5 = plt.subplot(4, 2, 5)
    ax6 = plt.subplot(4, 2, 6)
    ax7 = plt.subplot(4, 2, 7)
    ax8 = plt.subplot(4, 2, 8)
    axes = ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8))

    num_to_show = 500
    for (name, _samples), (_ax1, _ax2) in zip(samples.items(), axes):
        
        _ax1.plot(t[-num_to_show:], _samples[-num_to_show:])
        _ax1.set_title(f'{name}')
        _ax1.set_xlabel(r'$t$')
        _ax1.set_ylabel(r'$X_t$')
        _ax1.grid()

        _ax2.plot(t[-num_to_show:], grads[name][-num_to_show:], 'g')
        _ax2.set_title(f'{name}')
        _ax2.set_xlabel(r'$t$')
        _ax2.set_ylabel(r'$\nabla \log \pi (X_t)$')
        _ax2.grid()

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{model_name}-samples.pdf'))
    # plt.show()

    # KS Distance & KL Divergence
    
    point_num = 500
    # start_point = 1
    # idx_list = (max_itr * (np.arange(point_num) + 1) / point_num - 1).astype(int)
    # idx_list = (10 ** (np.log10(t).max() * (np.arange(point_num) + 1) / point_num)).astype(int)[start_point:]

    idx_list = np.logspace(np.log10(t[100]), np.log10(t[-1]), num=point_num)

    results = {}
    for name, _samples in samples.items():
        distances, divergences = [], []
        if name == 'Beta distribution':
            _cdf = beta_cdf
        else:
            _cdf = all_dist[name].cdf
        for idx in tqdm(idx_list):
            values, cprobs = ecdf(_samples[:int(idx)])
            distances.append(ks_distance(
                values=values, 
                cprobs=cprobs, 
                cdf=_cdf,
            ))
            divergences.append(kl_divergence(
                values=np.append(-np.inf, values), 
                cprobs=np.append(0.0, cprobs), 
                cdf=_cdf,
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
    plt.savefig(os.path.join(save_path, f'{model_name}-distances.pdf'))
    plt.show()