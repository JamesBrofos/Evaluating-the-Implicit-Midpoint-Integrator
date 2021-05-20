from typing import Callable

import numpy as np
import tqdm

import hmc.errors
from hmc.transforms import identity


def reverse(q: np.ndarray, p: np.ndarray, step_size: float, num_steps: int, proposal: Callable):
    """Compute the reversibility of the proposal operator by first integrating
    forward, then flipping the sign of the momentum, integrating again, and
    flipping the sign of the momentum a final time in order to compute the
    distance between the original position and the terminal position. If the
    operator is symmetric (reversible) then this distance should be very
    small.

    Args:
        q: Position variable.
        p: Momentum variable.
        step_size: Integration step-size.
        num_steps: Number of integration steps.
        proposal: Proposal operator for the Markov chain; operates on both
            position and momentum.

    Returns:
        rev: The distance between the original position in phase space and the
            terminal position of the proposal operator.

    """
    try:
        (qn, pn), sf = proposal(q, p, step_size, num_steps)
        (qr, pr), sr = proposal(qn, -pn, step_size, num_steps)
        rev = np.sqrt(np.square(np.linalg.norm(q - qr)) + np.square(np.linalg.norm(p + pr)))
        return rev
    except hmc.errors.errors:
        return np.nan

def reversibility(
        samples: np.ndarray,
        step_size: float,
        num_steps: int,
        proposal: Callable,
        sample_momentum: Callable,
        max_trials: int=100,
        forward_transform: Callable=identity
) -> np.ndarray:
    """Compute the reversibility of the proposal operator using finite
    differences over a range of samples.

    Args:
        samples: An array representing the positions of the Markov chain
            (excluding momenta) which will be used to compute an average
            reversibility of the proposal operator.
        step_size: Integration step-size.
        num_steps: Number of integration steps.
        proposal: Proposal operator for the Markov chain; operates on both
            position and momentum.
        max_trials: Maximum number of samples to use in the average.

    Returns:
        rev: The reversibility of the proposal operator for each sample.

    """
    num_trials = min(samples.shape[0], max_trials)
    pbar = tqdm.tqdm(total=num_trials, position=0, leave=True)
    rev = np.zeros(num_trials)
    perm = np.random.permutation(len(samples))[:num_trials]
    for i in range(num_trials):
        q = samples[perm[i]]
        qt, _ = forward_transform(q)
        p = sample_momentum(qt)
        rev[i] = reverse(qt, p, step_size, num_steps, proposal)
        pbar.update(1)
    return rev
