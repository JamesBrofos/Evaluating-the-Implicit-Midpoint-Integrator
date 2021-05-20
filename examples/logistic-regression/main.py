import argparse
import os
import time
from typing import Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as spla
import scipy.stats as spst
import tqdm

import hmc

parser = argparse.ArgumentParser(description='Comparison of implicit midpoint and generalized leapfrog on Bayesian logistic regression')
parser.add_argument('--dataset', type=str, default='banana', help='Logistic regression dataset')
parser.add_argument('--step-size', type=float, default=0.1, help='Integration step-size')
parser.add_argument('--num-steps', type=int, default=10, help='Number of integration steps')
parser.add_argument('--num-samples', type=int, default=1000, help='Number of samples to generate')
parser.add_argument('--thresh', type=float, default=1e-6, help='Convergence tolerance for fixed-point iterations')
parser.add_argument('--integrator', type=str, default='glf', help='Select which integrator to use; can be `glf`, `sglf`, or `imp`')
parser.add_argument('--randomize-steps', default=True, action='store_true', help='Randomize the number of integration steps')
parser.add_argument('--no-randomize-steps', action='store_false', dest='randomize_steps')
args = parser.parse_args()

# Construct posterior distribution.
distr = hmc.applications.logistic_regression
(x, y), (_, _) = distr.load_dataset(args.dataset)
alpha = 1.0
(
    log_posterior, grad_log_posterior, _,
    metric, grad_metric, _,
    grad_log_posterior_and_metric_and_grad_metric
) = distr.posterior_factory(x, y, alpha)
(
    hamiltonian, grad_pos_hamiltonian, grad_mom_hamiltonian,
    vector_field, sample_momentum
) = hmc.integrators.riemannian_hamiltonian_vector_field(
    log_posterior, grad_log_posterior, metric, grad_metric,
    grad_log_posterior_and_metric_and_grad_metric)

qmap = hmc.applications.newton_raphson(np.zeros(x.shape[-1]), grad_log_posterior, metric)
num_dims = len(qmap)
coef = ['beta-{}'.format(i+1) for i in range(num_dims)]

def experiment(method: str, step_size: float, num_steps: int, num_samples: int, proposal: Callable, randomize_steps: bool):
    """Experiment to examine the use of different integrators for sampling from the
    posterior distribution of a Bayesian logistic regression model. Given a
    proposal operator, attempts to draw samples and computes performance
    metrics for the sampler.

    Args:
        step_size: Integration step-size.
        num_steps: Number of integration steps.
        num_samples: Number of samples to generate.
        proposal: Proposal function that will yield the next state of the Markov
            chain.
        randomize_steps: Randomize the number of integration steps.

    Returns:
        samples: Samples from the Markov chain generated using the proposal
            operator.

    """
    sampler = hmc.sample(qmap, step_size, num_steps, hamiltonian, proposal, sample_momentum, randomize_sign=False, randomize_steps=randomize_steps)
    _ = next(sampler)
    samples = np.zeros((num_samples, num_dims))
    acc = 0
    pbar = tqdm.tqdm(total=num_samples, position=0, leave=True)
    start = time.time()
    for i in range(num_samples):
        samples[i], isacc = next(sampler)
        acc += isacc
        pbar.set_postfix({'accprob': acc / (i + 1)})
        pbar.update(1)

    elapsed = time.time() - start
    accprob = acc / num_samples
    print('{} - time elapsed: {:.5f} - acceptance prob.: {:.5f}'.format(method, elapsed, accprob))
    metrics = hmc.summarize(samples, coef)
    mean_ess = metrics['ess'].mean()
    mean_ess_sec = mean_ess / elapsed
    min_ess = metrics['ess'].min()
    min_ess_sec = min_ess / elapsed
    print('mean ess: {:.3f} - mean ess / sec: {:.3f} - min ess: {:.3f} - min ess / sec: {:.3f}'.format(mean_ess, mean_ess_sec, min_ess, min_ess_sec))
    return samples

def display_results(res: np.ndarray, check: str, method: str):
    """Compare the volume-preservation and reversibility properties of the two
    integrators.

    """
    res = res[~np.isnan(res) & ~np.isinf(res)]
    print('{} - {} - min.: {:.2e} - max.: {:.2e} - median: {:.2e} - 10%: {:.2e} - 90%: {:.2e}'.format(
        check, method, np.min(res), np.max(res), np.median(res), np.percentile(res, 10), np.percentile(res, 90)))

print('dataset: {} - step-size: {} - num. steps: {} - randomize steps: {}'.format(args.dataset, args.step_size, args.num_steps, args.randomize_steps))

proposal_generalized_leapfrog = hmc.proposals.generalized_leapfrog_proposal_factory(grad_pos_hamiltonian, grad_mom_hamiltonian, args.thresh)
proposal_smart_generalized_leapfrog = hmc.proposals.smart_generalized_leapfrog_proposal_factory(grad_log_posterior, metric, grad_metric, grad_log_posterior_and_metric_and_grad_metric, args.thresh)
proposal_implicit_midpoint = hmc.proposals.implicit_midpoint_proposal_factory(vector_field, args.thresh)
proposal_smart_implicit_midpoint = hmc.proposals.smart_implicit_midpoint_proposal_factory(vector_field, args.thresh)

name, proposal = {
    'imp': ('implicit midpoint', proposal_implicit_midpoint),
    'simp': ('smart implicit midpoint', proposal_smart_implicit_midpoint),
    'sglf': ('smart generalized leapfrog', proposal_smart_generalized_leapfrog),
    'glf': ('naive generalized leapfrog', proposal_generalized_leapfrog),
}[args.integrator]

samples = experiment(name, args.step_size, args.num_steps, args.num_samples, proposal, args.randomize_steps)
vp = hmc.checks.jacobian_determinant(samples, args.step_size, args.num_steps, proposal, sample_momentum)
rev = hmc.checks.reversibility(samples, args.step_size, args.num_steps, proposal, sample_momentum)
display_results(vp, 'volume', name)
display_results(rev, 'reverse', name)
