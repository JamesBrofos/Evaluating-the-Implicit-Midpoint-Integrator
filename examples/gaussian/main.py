import argparse
import os
import time
from typing import Callable

import arviz
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as spst
import scipy.special as spsp
import tqdm

import hmc

# np.random.seed(0)

parser = argparse.ArgumentParser(description='Bias in Hamiltonian Monte Carlo when using the implicit midpoint integrator')
parser.add_argument('--step-size', type=float, default=0.1, help='Integration step-size')
parser.add_argument('--num-steps', type=int, default=10, help='Number of integration steps')
parser.add_argument('--num-samples', type=int, default=100000, help='Number of samples to generate')
parser.add_argument('--thresh', type=float, default=1e-6, help='Convergence tolerance for fixed-point iterations')
parser.add_argument('--method', type=str, default='imp', help='Which integrator to use in proposal operator')
args = parser.parse_args()

def experiment(method: str, step_size: float, num_steps: int, num_samples: int, proposal: Callable) -> np.ndarray:
    """Experiment to examine the use of different integrators for sampling from a
    Gaussian distribution. Given a proposal operator, attempts to draw samples
    and computes performance metrics for the sampler.

    Args:
        method: String identifier for the proposal method.
        step_size: Integration step-size.
        num_steps: Number of integration steps.
        num_samples: Number of samples to generate.
        proposal: Proposal function that will yield the next state of the Markov
            chain.

    Returns:
        samples: Samples from the Markov chain generated using the proposal
            operator.

    """
    sampler = hmc.sample(mu, step_size, num_steps, hamiltonian, proposal, sample_momentum, check_prob=0.0001)
    samples = np.zeros((num_samples, 2))
    acc = 0
    pbar = tqdm.tqdm(total=num_samples, position=0, leave=True)
    _ = next(sampler)
    start = time.time()
    for i in range(num_samples):
        samples[i], isacc = next(sampler)
        acc += isacc
        pbar.set_postfix({'accprob': acc / (i + 1)})
        pbar.update(1)

    elapsed = time.time() - start
    accprob = acc / num_samples
    print('{} - time elapsed: {:.5f} - acceptance prob.: {:.5f}'.format(method, elapsed, accprob))
    metrics = hmc.summarize(samples, ('theta-1', 'theta-2'))
    mean_ess = metrics['ess'].mean()
    mean_ess_sec = mean_ess / elapsed
    min_ess = metrics['ess'].min()
    min_ess_sec = min_ess / elapsed
    print('mean ess: {:.3f} - mean ess / sec: {:.3f} - min ess: {:.3f} - min ess / sec: {:.3f}'.format(mean_ess, mean_ess_sec, min_ess, min_ess_sec))
    return samples


mu = np.array([0.0, 0.5])
Sigma = np.array([[1.0, 0.5], [0.5, 2.0]])

distr = hmc.applications.gaussian
log_posterior, grad_log_posterior, metric = distr.posterior_factory(mu, Sigma)
hamiltonian, grad_pos_hamiltonian, grad_mom_hamiltonian, vector_field, sample_momentum = hmc.integrators.vector_fields.euclidean_hamiltonian_vector_field(log_posterior, grad_log_posterior, metric)


if args.method == 'imp':
    proposal = hmc.proposals.implicit_midpoint_proposal_factory(vector_field, args.thresh)
    name = 'implicit midpoint'
elif args.method == 'lf':
    proposal = hmc.proposals.leapfrog_proposal_factory(grad_pos_hamiltonian, grad_mom_hamiltonian)
    name = 'l.f.'
    args.thresh = 0.0
else:
    raise ValueError('Unknown method specification.')

samples = experiment(name, args.step_size, args.num_steps, args.num_samples, proposal)
print('covariance')
print(np.cov(samples.T))
