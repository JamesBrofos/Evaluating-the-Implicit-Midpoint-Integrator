from typing import Callable

import numpy as np
import tqdm

import hmc.errors
from hmc.transforms import identity


def jacobian(func: Callable, delta: float):
    """Finite differences approximation to the Jacobian."""
    # Finite differences perturbation size.
    def jacfn(z):
        num_dims = len(z)
        Jac = np.zeros((num_dims, num_dims))
        for j in range(num_dims):
            pert = np.zeros(num_dims)
            pert[j] = 0.5 * delta
            zh = func(z + pert)
            zl = func(z - pert)
            Jac[j] = (np.hstack(zh) - np.hstack(zl)) / delta
        return Jac
    return jacfn

def compute_jacobian(q: np.ndarray, p: np.ndarray, step_size: float, num_steps: int, proposal: Callable, delta: float):
    """Compute the Jacobian of the transformation for a single sample consisting of
    a position and momentum.

    Args:
        q: Position variable.
        p: Momentum variable.
        step_size: Integration step-size.
        num_steps: Number of integration steps.
        proposal: Proposal operator for the Markov chain; operates on both
            position and momentum.
        delta: Perturbation size for numerical Jacobian.

    Returns:
        det: Jacobian determinant of the transformation computed using finite
            differences.

    """
    # Redefine the proposal operator as a map purely to phase-space to
    # phase-space with no additional inputs or outputs.
    def _proposal(z):
        return np.hstack(proposal(*np.split(z, 2), step_size, num_steps)[0])

    try:
        z = np.hstack((q, p))
        Jac = jacobian(_proposal, delta)(z)
        det = np.linalg.det(Jac)
        return np.abs(det - 1.0)
    except hmc.errors.errors:
        return np.nan

def jacobian_determinant(
        samples: np.ndarray,
        step_size: float,
        num_steps: int,
        proposal: Callable,
        sample_momentum: Callable,
        max_trials: int=100,
        delta: float=1e-5,
        forward_transform: Callable=identity
) -> np.ndarray:
    """Compute the Jacobian determinant of the proposal operator using finite
    differences over a range of samples.

    Args:
        samples: An array representing the positions of the Markov chain
            (excluding momenta) which will be used to compute an average Jacobian
            determinant of the proposal operator.
        step_size: Integration step-size.
        num_steps: Number of integration steps.
        proposal: Proposal operator for the Markov chain; operates on both
            position and momentum.
        max_trials: Maximum number of samples to use in the average.
        delta: Perturbation size for numerical Jacobian.

    Returns:
        det: The Jacobian determinant of the transformation.

    """
    num_trials = min(samples.shape[0], max_trials)
    pbar = tqdm.tqdm(total=num_trials, position=0, leave=True)
    det = np.zeros(num_trials)
    perm = np.random.permutation(len(samples))[:num_trials]
    for i in range(num_trials):
        q = samples[perm[i]]
        qt, _ = forward_transform(q)
        p = sample_momentum(qt)
        det[i] = compute_jacobian(qt, p, step_size, num_steps, proposal, delta)
        pbar.update(1)
    return det
