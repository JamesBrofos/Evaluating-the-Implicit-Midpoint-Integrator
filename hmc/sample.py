import itertools
from typing import Callable

import numpy as np

import hmc.checks
import hmc.errors
from hmc.transforms import identity


def sample(
        q: np.ndarray,
        step_size: float,
        num_steps: int,
        hamiltonian: Callable,
        proposal: Callable,
        sample_momentum: Callable,
        forward_transform: Callable=identity,
        inverse_transform: Callable=identity,
        randomize_sign: bool=True,
        randomize_steps: bool=True,
        check_prob: float=0.0
) -> np.ndarray:
    """Draw samples from the target density using Hamiltonian Monte Carlo. This
    function requires that one specify a Hamiltonian energy, a proposal
    operator, and a function to sample momenta. This function is implemented as
    a generator so as to yield samples from the target distribution when
    requested.

    Args:
        q: The position variable.
        step_size: The integration step-size.
        num_steps: The number of integration steps.
        hamiltonian: The Hamiltonian energy function.
        proposal: A function to compute the next proposed state of the Markov
            chain.
        sample_momentum: A function to sample the momentum variable.
        randomize_sign: Whether or not to randomize the sign of the integration
            step-size.
        check_prob: Probability to compute reversibility and volume preservation
            statistics for the proposal.

    Returns:
        q: The next position variable.
        accept: Whether or not the sample was accepted.

    """
    # Sample momentum from conditional distribution and compute the associated
    # Hamiltonian energy.
    for it in itertools.count():
        qt, ildj = forward_transform(q)
        p = sample_momentum(qt)
        currH = hamiltonian(qt, p)
        sign = np.sign(np.random.normal()) if randomize_sign else 1.0
        if randomize_steps:
            # ns = 1 if np.random.uniform() < 1 / num_steps else num_steps
            ns = int(np.ceil(np.random.uniform() * num_steps)) if randomize_steps else num_steps
        else:
            ns = num_steps

        try:
            (propqt, propp), success = proposal(qt, p, sign*step_size, ns)
            propq, propfldj = inverse_transform(propqt)
            propH = hamiltonian(propqt, propp)
        except hmc.errors.errors:
            # If the integration fails, then make sure that no transition can
            # occur.
            propq, propp, success = q, p, False
            propH, propfldj = currH, -ildj

        # Notice the relevant choice of sign when the Jacobian determinant of
        # the forward or inverse transform is used.
        #
        # Write this expression as,
        # (exp(-propH) / exp(propfldj)) / (exp(-currH) * exp(ildj))
        #
        # See the following resource for understanding the Metropolis-Hastings
        # correction with a Jacobian determinant correction [1].
        #
        # [1] https://wiki.helsinki.fi/download/attachments/48865399/ch7-rev.pdf
        logu = np.log(np.random.uniform())
        metropolis = logu < currH - propH - propfldj - ildj
        accept = np.logical_and(metropolis, success)

        # Compute reversibility and volume preservation statistics of the
        # transformation. Is there any reason to check these properties for
        # samples that will be rejected?
        random_check = np.random.uniform() < check_prob
        if random_check and accept:
            delta = 1e-5
            det = hmc.checks.compute_jacobian(qt, p, step_size, ns, proposal, delta)
            rev = hmc.checks.reverse(qt, p, step_size, ns, proposal)
            print('iter.: {} - jacobian error: {:.5e} - reversal error: {:.5e}'.format(it, det, rev))

        if accept:
            q = propq
        yield q, accept
