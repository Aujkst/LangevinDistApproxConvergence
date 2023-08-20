from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats

from distributions import GeneralisedLogisticDist
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
    step_size = .5
    max_itr = 1e5
    t = np.arange(step_size, (max_itr + 1.0) * step_size, step_size)
    U = np.random.normal(loc=0.0, scale=1.0, size=(2, int(max_itr)))
    # U = None

    alpha, beta = 3.0, 5.0
    gl_dist = GeneralisedLogisticDist(alpha=alpha, beta=beta)
    beta_cdf = lambda x: stats.beta.cdf(x=x, a=alpha, b=beta)

    samples, grads = {}, {}
    
    sampler = LangevinAlgoSampler(
        X_zero=X_zero,
        target_dist=gl_dist,
        step_size=step_size,
        max_itr=max_itr,
        step_method='euler_maruyama_method',
        U=U,
    )
    samples['EulerMaruyama-Langevin'], grads['EulerMaruyama-Langevin'] = sampler.run()

    sampler = LangevinAlgoSampler(
        X_zero=X_zero,
        target_dist=gl_dist,
        step_size=step_size,
        max_itr=max_itr,
        step_method='strong_order_taylor_method',
        U=U,
    )
    samples['StrongOrderTaylor-Langevin'], grads['StrongOrderTaylor-Langevin'] = sampler.run()

    sampler = MetropolisAdjLangevinAlgoSampler(
        X_zero=X_zero,
        target_dist=gl_dist,
        step_size=step_size,
        max_itr=max_itr,
        step_method='euler_maruyama_method',
        U=U,
    )
    samples['EulerMaruyama-MALA'], grads['EulerMaruyama-MALA'] = sampler.run()

    sampler = MetropolisAdjLangevinAlgoSampler(
        X_zero=X_zero,
        target_dist=gl_dist,
        step_size=step_size,
        max_itr=max_itr,
        step_method='strong_order_taylor_method',
        U=U,
    )
    samples['StrongOrderTaylor-MALA'], grads['StrongOrderTaylor-MALA'] = sampler.run()

    for name, _samples in samples.items():
        samples[name] = np.exp(_samples) / (1.0 + np.exp(_samples))

    print(pd.DataFrame(samples).agg(['mean', 'std']).T)
    print(alpha / (alpha + beta))
    print(np.sqrt(alpha * beta / ((alpha + beta)**2 * (alpha + beta + 1))))

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

    for (name, _samples), (_ax1, _ax2) in zip(samples.items(), axes):
        _ax1.plot(t[-100:], _samples[-100:])
        _ax1.set_title(f'Beta distribution ({name})')
        _ax1.set_xlabel(r'$t$')
        _ax1.set_ylabel(r'$X_t$')

        _ax2.plot(t[-100:], grads[name][-100:], 'g')
        _ax2.set_title(f'Beta distribution ({name})')
        _ax2.set_xlabel(r'$t$')
        _ax2.set_ylabel(r'$\nabla \log \pi (X_t)$')

    plt.tight_layout()
    plt.show()

    # KS Distance & KL Divergence    
    
    idx_list = (max_itr * (np.arange(500) + 1) / 100 - 1).astype(int)

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
