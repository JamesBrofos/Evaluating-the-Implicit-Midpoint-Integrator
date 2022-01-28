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

parser = argparse.ArgumentParser(description='Comparison of implicit midpoint and generalized leapfrog on the Fitzhugh-Nagumo differential equation model')
parser.add_argument('--step-size', type=float, default=0.01, help='Integration step-size')
parser.add_argument('--num-steps', type=int, default=10, help='Number of integration steps')
parser.add_argument('--num-samples', type=int, default=1000, help='Number of samples to generate')
parser.add_argument('--thresh', type=float, default=1e-6, help='Convergence tolerance for fixed-point iterations')
parser.add_argument('--integrator', type=str, default='glf', help='Select which integrator to use; can be `glf`, `sglf`, `imp`, or `simp`')
parser.add_argument('--hmax', type=float, default=0.0, help='Maximum integration step-size')
parser.add_argument('--mxstep', type=int, default=0, help='Maximum number of internal integration steps')
parser.add_argument('--randomize-steps', default=True, action='store_true', help='Randomize the number of integration steps')
parser.add_argument('--no-randomize-steps', action='store_false', dest='randomize_steps')
args = parser.parse_args()

# Integrator parameters.
rtol = 1e-12
atol = 1e-12

# Construct posterior distribution.
distr = hmc.applications.fitzhugh_nagumo
state = np.array([-1.0, 1.0])
t = np.linspace(0.0, 10.0, 200)
sigma = 0.5
a, b, c = 0.2, 0.2, 3.0

y = distr.generate_data(state, t, sigma, a, b, c, rtol, atol, args.hmax, args.mxstep)
(
    log_posterior, grad_log_posterior, metric, grad_metric,
    grad_log_posterior_and_metric_and_grad_metric
) = distr.posterior_factory(state, y, t, sigma, rtol, atol, args.hmax, args.mxstep)
(
    hamiltonian, grad_pos_hamiltonian, grad_mom_hamiltonian,
    vector_field, sample_momentum
) = hmc.integrators.riemannian_hamiltonian_vector_field(
    log_posterior, grad_log_posterior, metric, grad_metric,
    grad_log_posterior_and_metric_and_grad_metric)

qmap = hmc.applications.newton_raphson(np.array([0.1, 0.1, 1.0]), grad_log_posterior, metric)
num_dims = len(qmap)
coef = ['a', 'b', 'c']

def experiment(method: str, step_size: float, num_steps: int, num_samples: int, proposal: Callable, randomize_steps: bool) -> np.ndarray:
    """Experiment to examine the use of different integrators for sampling from the
    Fitzhugh-Nagumo posterior distribution. Given a proposal operator, attempts
    to draw samples and computes performance metrics for the sampler.

    Args:
        method: String identifier for the proposal method.
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
    samples = np.zeros((num_samples, num_dims))
    _ = next(sampler)
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


proposal_generalized_leapfrog = hmc.proposals.generalized_leapfrog_proposal_factory(grad_pos_hamiltonian, grad_mom_hamiltonian, args.thresh)
proposal_smart_generalized_leapfrog = hmc.proposals.smart_generalized_leapfrog_proposal_factory(grad_log_posterior, metric, grad_metric, grad_log_posterior_and_metric_and_grad_metric, args.thresh)
proposal_implicit_midpoint = hmc.proposals.implicit_midpoint_proposal_factory(vector_field, args.thresh)
proposal_smart_implicit_midpoint = hmc.proposals.smart_implicit_midpoint_proposal_factory(vector_field, args.thresh)
proposal_lagrange_implicit_midpoint = hmc.proposals.lagrange_implicit_midpoint_proposal_factory(grad_log_posterior_and_metric_and_grad_metric, args.thresh)

print('step-size: {} - num. steps: {} - threshold: {} - hmax: {} - randomize steps: {}'.format(args.step_size, args.num_steps, args.thresh, args.hmax, args.randomize_steps))

name, proposal = {
    'imp': ('implicit midpoint', proposal_implicit_midpoint),
    'simp': ('smart implicit midpoint', proposal_smart_implicit_midpoint),
    'sglf': ('smart generalized leapfrog', proposal_smart_generalized_leapfrog),
    'glf': ('naive generalized leapfrog', proposal_generalized_leapfrog),
    'limp': ('lagrange implicit midpoint', proposal_lagrange_implicit_midpoint)
}[args.integrator]
samples = experiment(name, args.step_size, args.num_steps, args.num_samples, proposal, args.randomize_steps)
vp = hmc.checks.jacobian_determinant(samples, args.step_size, args.num_steps, proposal, sample_momentum, max_trials=10, delta=1e-3)
rev = hmc.checks.reversibility(samples, args.step_size, args.num_steps, proposal, sample_momentum, max_trials=10)
display_results(vp, 'volume', name)
display_results(rev, 'reverse', name)
