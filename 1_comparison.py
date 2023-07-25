from __future__ import annotations

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from scipy import stats
from torch.distributions.beta import Beta
from torch.distributions.gamma import Gamma
from torch.distributions.normal import Normal
from torch.distributions.laplace import Laplace
from torch.distributions.studentT import StudentT
from torch.distributions.log_normal import LogNormal
from torch.distributions.transforms import ExpTransform, SigmoidTransform
from torch.distributions.transformed_distribution import TransformedDistribution

from langevin import LangevinAlgoSampler
from utils import ecdf, sigmoid_function, ks_distance, kl_divergence

np.random.seed(0)
torch.manual_seed(0)
plt.rcParams['text.usetex'] = True

if __name__ == '__main__':

    file_path = os.getcwd()

    step_size = 1e-1
    max_itr = 1e4
    t = np.arange(step_size, (max_itr + 1.0) * step_size, step_size)

    mu1, sigma1 = 1.0, 2.0 # Normal dist
    loc, scale = 1.0, 0.5 # Laplace dist
    df = 4.0 # StudentT dist
    alpha1, beta1 = 2.0, 6.0 # Gamma dist
    mu2, sigma2 = 0.5, 0.7 # LogNormal dist
    alpha2, beta2 = 3.0, 5.0 # Beta dist
    
    gamma_dist = Gamma(concentration=torch.tensor([alpha1]), rate=torch.tensor([beta1]))
    lognorm_dist = LogNormal(loc=torch.tensor([mu2]), scale=torch.tensor([sigma2]))
    beta_dist = Beta(torch.tensor([alpha2]), torch.tensor([beta2]))

    params_dists = {
        'Normal': [
            Normal(loc=torch.tensor([mu1]), scale=torch.tensor([sigma1])),
            None,
            None,
        ],
        'Laplace': [
            Laplace(loc=torch.tensor([loc]), scale=torch.tensor([scale])),
            None,
            None,
        ],
        'Student-t': [
            StudentT(df=torch.tensor([df])),
            None,
            None,
        ],
        'Gamma': [
            TransformedDistribution(base_distribution=gamma_dist, transforms=ExpTransform().inv),
            np.exp,
            gamma_dist,
        ],
        'Log-Normal': [
            TransformedDistribution(base_distribution=lognorm_dist, transforms=ExpTransform().inv),
            np.exp,
            lognorm_dist,
        ],
        'Beta': [
            TransformedDistribution(base_distribution=beta_dist, transforms=SigmoidTransform().inv),
            sigmoid_function,
            beta_dist,
        ],
    }

    samples = {}
    grads = {}
    true_result = {}
    for name, (target_dist, sample_trans_func, _dist) in params_dists.items():
        sampler = LangevinAlgoSampler(
            X_zero=torch.tensor(1.0),
            target_log_func=target_dist.log_prob,
            step_size=step_size,
            max_itr=max_itr,
        )
        samples[name], grads[name] = sampler.run()
        if sample_trans_func is not None:
            samples[name] = sample_trans_func(samples[name].T.squeeze())

        if _dist is None:
            true_result[name] = [target_dist.mean.numpy().item(), target_dist.stddev.numpy().item()]
        else:
            true_result[name] = [_dist.mean.numpy().item(), _dist.stddev.numpy().item()]
    
    samples, grads = pd.DataFrame(samples), pd.DataFrame(grads)
    sample_result = pd.concat([samples[200:].mean(), samples[200:].std()], axis=1).T
    sample_result.index = 'Mean (Sample)', 'Std (Sample)'
    true_result = pd.DataFrame(true_result)
    true_result.index = 'Mean (True)', 'Std (True)'

    result = pd.concat([sample_result, true_result]).sort_index()
    print(result)


#     # Sample paths

#     fig = plt.figure(figsize=(8, 6))
#     ax1 = plt.subplot(3, 2, 1)
#     ax2 = plt.subplot(3, 2, 3)
#     ax3 = plt.subplot(3, 2, 5)    
#     ax4 = plt.subplot(3, 2, 2)
#     ax5 = plt.subplot(3, 2, 4)
#     ax6 = plt.subplot(3, 2, 6)

#     for col, ax in zip(samples, [ax1, ax2, ax3, ax4, ax5, ax6]):
#         ax.plot(t[-1000:], samples[col][-1000:])
#         ax.set_title(f'{col} distribution')
#         ax.set_xlabel(r'$t$')
#         ax.set_ylabel(r'$X_t$')

#     plt.tight_layout()
#     plt.savefig(os.path.join(file_path, 'langevin-samples.pdf'))

#     # Gradient paths

#     fig = plt.figure(figsize=(8, 6))
#     ax1 = plt.subplot(3, 2, 1)
#     ax2 = plt.subplot(3, 2, 3)
#     ax3 = plt.subplot(3, 2, 5)    
#     ax4 = plt.subplot(3, 2, 2)
#     ax5 = plt.subplot(3, 2, 4)
#     ax6 = plt.subplot(3, 2, 6)

#     for col, ax in zip(grads, [ax1, ax2, ax3, ax4, ax5, ax6]):
#         ax.plot(t[-1000:], grads[col][-1000:], 'g')
#         ax.set_title(f'{col} distribution')
#         ax.set_xlabel(r'$t$')
#         ax.set_ylabel(r'$\nabla \log \pi (X_t)$')

#     plt.tight_layout()
#     plt.savefig(os.path.join(file_path, 'langevin-samplegrads.pdf'))


#     fig = plt.figure(figsize=(8, 6))
#     ax1 = plt.subplot(2, 1, 1)
#     ax2 = plt.subplot(2, 1, 2)

#     # Kolmogorov-Smirnov Distance

#     cdf_dict = {
#         'Normal': lambda x: params_dists['Normal'][0].cdf(torch.tensor(x)).numpy(),
#         'Laplace': lambda x: params_dists['Laplace'][0].cdf(torch.tensor(x)).numpy(),
#         'Student-t': lambda x: stats.t.cdf(x, df=df),
#         'Gamma': lambda x: gamma_dist.cdf(torch.tensor(x)).numpy(),
#         'Log-Normal': lambda x: lognorm_dist.cdf(torch.tensor(x)).numpy(),
#         'Beta': lambda x: stats.beta.cdf(x, a=alpha2, b=beta2),
#     }

#     for col in samples:
#         cdf_func = cdf_dict[col]
#         distances = []
#         for idx in tqdm(samples.index):
#             values, cprobs = ecdf(samples.loc[:idx, col])
#             distances.append(ks_distance(values, cprobs, cdf_func))
#         distances = np.asarray(distances)
#         ax1.plot(np.log10(t), np.log10(distances), label=f'{col} distribution')

#     ax1.grid()
#     ax1.legend()

#     bound_dict = {
#         'Normal': (-np.inf, np.inf),
#         'Laplace': (-np.inf, np.inf),
#         'Student-t': (-np.inf, np.inf),
#         'Gamma': (0.0, np.inf),
#         'Log-Normal': (0, np.inf),
#         'Beta': (0.0, 1.0),
#     }


#     for col in samples:
#         cdf_func = cdf_dict[col]
#         divergences = []
#         for idx in tqdm(samples.index):
#             lower, _ = bound_dict[col]
#             values, cprobs = ecdf(samples.loc[:idx, col])
#             values = np.append(lower, values)
#             cprobs = np.append(0.0, cprobs)
#             divergences.append(kl_divergence(values, cprobs, cdf_func))
#         divergences = np.asarray(divergences)
#         ax2.plot(np.log10(t), np.log10(divergences), label=f'{col} distribution')

#     ax2.grid()
#     ax2.legend()
#     plt.tight_layout()
#     plt.show()


# # ---------------------------- #
#     fig = plt.figure(figsize=(8, 6))
#     ax1 = plt.subplot(2, 1, 1)
#     ax2 = plt.subplot(2, 1, 2)

#     for col in samples:
#         cdf_func = cdf_dict[col]
#         distances = []
#         for idx in tqdm(samples.index):
#             values, cprobs = ecdf(samples.loc[:idx, col])
#             distances.append(ks_distance(values, cprobs, cdf_func))
#         distances = np.asarray(distances)
#         ax1.plot(t, np.log2(distances), label=f'{col} distribution')

#     ax1.grid()
#     ax1.legend()


#     for col in samples:
#         cdf_func = cdf_dict[col]
#         divergences = []
#         for idx in tqdm(samples.index):
#             lower, _ = bound_dict[col]
#             values, cprobs = ecdf(samples.loc[:idx, col])
#             values = np.append(lower, values)
#             cprobs = np.append(0.0, cprobs)
#             divergences.append(kl_divergence(values, cprobs, cdf_func))
#         divergences = np.asarray(divergences)
#         ax2.plot(np.log2(t), divergences, label=f'{col} distribution')

#     ax2.grid()
#     ax2.legend()
#     plt.tight_layout()
#     plt.show()

