from functools import partial
from typing import Callable, Tuple

import numpy as np

from hmc.integrators.terminal import cond
from hmc.core import for_loop, while_loop


def _single_step_leapfrog(
        grad_pos_hamiltonian: Callable,
        grad_mom_hamiltonian: Callable,
        zo: Tuple[np.ndarray],
        step_size: float,
        cache: dict
) -> Tuple[np.ndarray]:
    """Implements the leapfrog integrator, which is symmetric, symplectic, and
    second-order accurate for separable Hamiltonian systems.

    Args:
        grad_pos_hamiltonian: Gradient of the Hamiltonian with respect to the
            position variables.
        grad_mom_hamiltonian: Gradient of the Hamiltonian with respect to the
            momentum variables.
        zo: Tuple containing the position and momentum variables in the original
            phase space.
        step_size: Integration step_size.

    Returns:
        qn: The terminal position variable.
        pn: The terminal momentum variable.
        success: Boolean flag indicating successful integration. Always true for
            the leapfrog integrator.

    """
    half_step = 0.5 * step_size
    qo, po = zo
    if 'dp' in cache:
        dpo = cache['dp']
    else:
        dpo = grad_pos_hamiltonian(qo)
    pm = po - half_step * dpo
    qn = qo + step_size * grad_mom_hamiltonian(pm)
    dpn = grad_pos_hamiltonian(qn)
    pn = pm - half_step * dpn
    cache['dp'] = dpn
    return (qn, pn), cache

def leapfrog(
        grad_pos_hamiltonian: Callable,
        grad_mom_hamiltonian: Callable,
        zo: Tuple[np.ndarray],
        step_size: float,
        num_steps: int,
) -> Tuple:
    def step(it: int, val: Tuple):
        zo, cache = val
        zn, cache = _single_step_leapfrog(grad_pos_hamiltonian, grad_mom_hamiltonian, zo, step_size, cache)
        return zn, cache
    (qn, pn), _ = for_loop(0, num_steps, step, (zo, {}))
    success = True
    return (qn, pn), success

def momentum_step(val: Tuple, zo: Tuple[np.ndarray], half_step: float, grad_pos_hamiltonian: Callable) -> Tuple:
    """Function to find the fixed point of the momentum variable."""
    qo, po = zo
    pmcand, _, num_iters = val
    pm = po - half_step * grad_pos_hamiltonian(qo, pmcand)
    delta = pm - pmcand
    return pm, delta, num_iters + 1

def position_step(val: Tuple, zo: Tuple[np.ndarray], half_step: float, grad_mom_hamiltonian: Callable) -> Tuple:
    """Function to find the fixed point of the position variable."""
    qo, po = zo
    qncand, _, num_iters = val
    qn = qo + half_step * (grad_mom_hamiltonian(qo, po) + grad_mom_hamiltonian(qncand, po))
    delta = qn - qncand
    return qn, delta, num_iters + 1

def _single_step_generalized_leapfrog(
        grad_pos_hamiltonian: Callable,
        grad_mom_hamiltonian: Callable,
        zo: Tuple[np.ndarray],
        step_size: float,
        thresh: float,
        max_iters: int) -> Tuple:
    """Uses the generalized leapfrog integrator to compute a trajectory of a
    non-separable Hamiltonian. Unlike the implicit midpoint integrator, the
    generalized leapfrog method is not compatible with an arbitrary symplectic
    structure.

    Args:
        grad_pos_hamiltonian: Gradient of the Hamiltonian with respect to the
            position variables.
        grad_mom_hamiltonian: Gradient of the Hamiltonian with respect to the
            momentum variables.
        zo: Tuple containing the position and momentum variables in the original
            phase space.
        step_size: Integration step_size.
        thresh: Convergence tolerance for fixed point iterations.
        max_iters: Maximum number of fixed point iterations.

    Returns:
        qn: The terminal position variable.
        pn: The terminal momentum variable.
        num_iters: The number of fixed point iterations to find the fixed points
            defining the generalized leapfrog algorithm.
        success: Boolean flag indicating successful integration.

    """
    termcond = partial(cond, thresh=thresh, max_iters=max_iters)
    half_step = 0.5 * step_size
    qo, po = zo
    delta = np.ones_like(po) * np.inf
    # The first step of the integrator is to find a fixed point of the momentum
    # variable.
    val = (po, delta, 0)
    pm, delta_mom, num_iters_mom = while_loop(
        termcond,
        partial(momentum_step, zo=zo, half_step=half_step, grad_pos_hamiltonian=grad_pos_hamiltonian),
        val)
    success_mom = np.all(delta_mom < thresh)

    # The second step of the integrator is to find a fixed point of the
    # position variable. The first momentum gradient could be conceivably
    # cached and saved.
    val = (qo, delta, 0)
    qn, delta_pos, num_iters_pos = while_loop(
        termcond,
        partial(position_step, zo=(qo, pm), half_step=half_step, grad_mom_hamiltonian=grad_mom_hamiltonian),
        val)
    success_pos = np.all(delta_pos < thresh)
    success = np.logical_and(success_mom, success_pos)

    # Increment the number of iterations by one since the final momentum update
    # requires computing the gradient of the potential function again.
    num_iters = num_iters_pos + num_iters_mom + 1

    # Last step is to do an explicit half-step of the momentum variable.
    pn = pm - half_step * grad_pos_hamiltonian(qn, pm)
    return (qn, pn), num_iters, success

def generalized_leapfrog(
        grad_pos_hamiltonian: Callable,
        grad_mom_hamiltonian: Callable,
        zo: Tuple[np.ndarray],
        step_size: float,
        num_steps: int,
        thresh: float=1e-6,
        max_iters: int=1000
) -> Tuple:
    def step(it: int, val: Tuple):
        zo, so = val
        zn, _, sn = _single_step_generalized_leapfrog(grad_pos_hamiltonian, grad_mom_hamiltonian, zo, step_size, thresh, max_iters)
        success = np.logical_and(so, sn)
        return zn, success
    (qn, pn), success = for_loop(0, num_steps, step, (zo, True))
    return (qn, pn), success
