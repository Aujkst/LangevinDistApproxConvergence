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
    all_dist['Beta distribution'] = gl_dist
    beta_cdf = lambda x: stats.beta.cdf(x=x, a=alpha, b=beta)

    for dist_name, target_dist in all_dist.items():

        samples, grads = {}, {}
        
        sampler = LangevinAlgoSampler(
            X_zero=X_zero,
            target_dist=target_dist,
            step_size=step_size,
            max_itr=max_itr,
            step_method='euler_maruyama_method',
            U=U,
        )
        samples['EulerMaruyama-Langevin'], grads['EulerMaruyama-Langevin'] = sampler.run()

        sampler = LangevinAlgoSampler(
            X_zero=X_zero,
            target_dist=target_dist,
            step_size=step_size,
            max_itr=max_itr,
            step_method='strong_order_taylor_method',
            U=U,
        )
        samples['StrongOrderTaylor-Langevin'], grads['StrongOrderTaylor-Langevin'] = sampler.run()

        sampler = MetropolisAdjLangevinAlgoSampler(
            X_zero=X_zero,
            target_dist=target_dist,
            step_size=step_size,
            max_itr=max_itr,
            step_method='euler_maruyama_method',
            U=U,
        )
        samples['EulerMaruyama-MALA'], grads['EulerMaruyama-MALA'] = sampler.run()

        if dist_name == 'Beta distribution':
            for method_name, _samples in samples.items():
                samples[method_name]  = np.exp(_samples) / (1.0 + np.exp(_samples))
                
        # KS Distance & KL Divergence
        
        point_num = 500
        idx_list = np.logspace(np.log10(t[100]), np.log10(t[-1]), num=point_num)

        results = {}
        for method_name, _samples in samples.items():
            distances, divergences = [], []
            if dist_name == 'Beta distribution':
                _cdf = beta_cdf
            else:
                _cdf = all_dist[dist_name].cdf
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
            results[method_name] = {
                'Kolmogorov-Smirnov distance': np.asarray(distances),
                'Kullback-Leibler divergence': np.asarray(divergences),
            }

        fig = plt.figure(figsize=(8, 8))
        ax1 = plt.subplot(2, 1, 1)
        ax2 = plt.subplot(2, 1, 2)

        for method_name, result in results.items():
            for (distance_name, distances), ax in zip(result.items(), (ax1, ax2)):
                ax.plot(np.log10(idx_list), np.log10(distances), label=method_name)
                # ax.plot(idx_list, np.log10(distances), label=method_name)
                ax.set_title(distance_name)
                ax.set_xlabel(r'$\log_{10}t$')
                ax.set_ylabel(r'$\log_{10}D_t$')

        ax1.grid()
        ax2.grid()
        ax1.legend()
        ax2.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'{dist_name}-distances.pdf'))
        plt.show()