import argparse
import time
from typing import Callable, Tuple

import numpy as np
import tqdm

import hmc
from hmc.applications.stochastic_volatility import (
    generate_data,
    volatility_posterior_factory,
    latent_posterior_factory,
    forward_transform,
    inverse_transform)

parser = argparse.ArgumentParser(description='Comparison of implicit midpoint and generalized leapfrog on the stochastic volatility model')
parser.add_argument('--num-samples', type=int, default=20000, help='Number of samples to generate')
parser.add_argument('--num-burn', type=int, default=10000, help='Number of burn-in samples')
parser.add_argument('--integrator', type=str, default='sglf', help='Select which integrator to use; can be `sglf`, `glf`, `imp`, or `simp`')
parser.add_argument('--randomize-steps', default=True, action='store_true', help='Randomize the number of integration steps')
parser.add_argument('--no-randomize-steps', action='store_false', dest='randomize_steps')
args = parser.parse_args()

num_steps_latent = 6
step_size_latent = 0.5
num_steps_volatility = 50
step_size_volatility = 0.1

def volatility_posterior(sigma: float, phi: float, beta: float, y: np.ndarray) -> Tuple[Callable]:
    """Construct the Hamiltonian, the proposal operator, and momentum sampling
    distribution for the stochastic volatilities.

    """
    log_posterior, grad_log_posterior, metric = volatility_posterior_factory(sigma, phi, beta, y)
    hamiltonian, grad_pos_hamiltonian, grad_mom_hamiltonian, vector_field, sample_momentum = hmc.integrators.vector_fields.euclidean_hamiltonian_vector_field(log_posterior, grad_log_posterior, metric, hmc.linalg.solve_tridiagonal, hmc.linalg.cholesky_tridiagonal)
    proposal = hmc.proposals.leapfrog_proposal_factory(grad_pos_hamiltonian, grad_mom_hamiltonian)
    return hamiltonian, proposal, sample_momentum

def latent_posterior(x: np.ndarray, y: np.ndarray, integrator: str) -> Tuple[Callable]:
    """Construct the Hamiltonian, the proposal operator, and the momentum sampling
    distribution for the latent parameters. Also determines whether or not to
    use the implicit midpoint or generalized leapfrog integrator.

    """
    (
        log_posterior, grad_log_posterior, metric, grad_metric,
        grad_log_posterior_and_metric_and_grad_metric) = latent_posterior_factory(x, y)
    (
        hamiltonian, grad_pos_hamiltonian, grad_mom_hamiltonian, vector_field, sample_momentum
    ) = hmc.integrators.vector_fields.riemannian_hamiltonian_vector_field(
        log_posterior, grad_log_posterior, metric, grad_metric, grad_log_posterior_and_metric_and_grad_metric)
    if integrator == 'sglf':
        proposal = hmc.proposals.smart_generalized_leapfrog_proposal_factory(
            grad_log_posterior, metric, grad_metric,
            grad_log_posterior_and_metric_and_grad_metric)
    elif integrator == 'imp':
        proposal = hmc.proposals.implicit_midpoint_proposal_factory(vector_field)
    elif integrator == 'glf':
        proposal = hmc.proposals.generalized_leapfrog_proposal_factory(grad_pos_hamiltonian, grad_mom_hamiltonian)
    elif integrator == 'simp':
        proposal = hmc.proposals.smart_implicit_midpoint_proposal_factory(vector_field)
    elif integrator == 'limp':
        proposal = hmc.proposals.lagrange_implicit_midpoint_proposal_factory(grad_log_posterior_and_metric_and_grad_metric)
    else:
        raise ValueError()
    return hamiltonian, proposal, sample_momentum

def experiment(method: str, num_burn: int, num_samples: int, q: np.ndarray, x: np.ndarray, y: np.ndarray, randomize_steps: bool) -> np.ndarray:
    """Experiment to examine the use of different integrators for sampling from a
    stochastic volatility model. Given a proposal operator, attempts to draw
    samples and computes performance metrics for the sampler.

    Args:
        method: String identifier for the proposal method.
        num_samples: Number of samples to generate.
        q: Initial latent parameters.
        x: Initial stochastic volatilities.
        y: Stochastic volatility observations.
        randomize_steps: Randomize the number of integration steps.

    Returns:
        samples: Samples from the Markov chain generated using the proposal
            operator.

    """
    num_total = num_burn + num_samples
    pbar = tqdm.tqdm(total=num_total, position=0, leave=True)
    qacc, xacc = 0, 0
    samples = np.zeros((num_total, len(y) + 3))
    for i in range(num_total):
        if i == num_burn:
            start = time.time()
        hamiltonian, proposal, sample_momentum = latent_posterior(x, y, method)
        sampler = hmc.sample(q, step_size_latent, num_steps_latent, hamiltonian, proposal, sample_momentum, forward_transform, inverse_transform, randomize_steps=randomize_steps)
        q, qisacc = next(sampler)
        qacc += qisacc
        hamiltonian, proposal, sample_momentum = volatility_posterior(*q, y)
        sampler = hmc.sample(x, step_size_volatility, num_steps_volatility, hamiltonian, proposal, sample_momentum, randomize_steps=randomize_steps)
        x, xisacc = next(sampler)
        xacc += xisacc
        pbar.set_postfix({'lat. accprob': qacc / (i + 1), 'vol. accprob': xacc / (i + 1)})
        pbar.update(1)
        samples[i] = np.hstack((q, x))

    elapsed = time.time() - start

    # Check the assumptions of reversibility and volume preservation.
    num_trials = min(100, num_samples)
    jacdet = np.zeros(num_trials)
    rev = np.zeros(num_trials)
    permuted = samples[num_burn:][np.random.permutation(num_samples)]
    for it in range(num_trials):
        q, x = permuted[it, :3], permuted[it, 3:]
        hamiltonian, proposal, sample_momentum = latent_posterior(x, y, method)
        rev[it] = hmc.checks.reversibility(q[np.newaxis], step_size_latent, num_steps_latent, proposal, sample_momentum, 100, forward_transform)
        jacdet[it] = hmc.checks.jacobian_determinant(q[np.newaxis], step_size_latent, num_steps_latent, proposal, sample_momentum, 100, 1e-5, forward_transform)

    display_results(jacdet, 'volume', method)
    display_results(rev, 'reverse', method)

    samples = samples[num_burn:, :3]
    print('num. samples: {}'.format(len(samples)))
    print('{} - time elapsed: {:.5f} - latent acceptance prob.: {:.5f} - vol. acceptance prob.: {:.5f}'.format(method, elapsed, qacc / num_total, xacc / num_total))
    metrics = hmc.summarize(samples, ('sigma', 'phi', 'beta'))
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


print('randomize steps: {}'.format(args.randomize_steps))
sigma = 0.15
phi = 0.98
beta = 0.65
T = 1000
x, y = generate_data(T, sigma, phi, beta)
q = np.array([sigma, phi, beta])
samples = experiment(args.integrator, args.num_burn, args.num_samples, q, x, y, args.randomize_steps)
