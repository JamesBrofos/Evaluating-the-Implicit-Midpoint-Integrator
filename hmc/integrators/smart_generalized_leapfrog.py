from functools import partial
from typing import Callable, Tuple

import numpy as np

from hmc.integrators.terminal import cond
from hmc.core import for_loop, while_loop
from hmc.linalg import solve_psd


def momentum_step(val: Tuple, half_step: float, fixed: Tuple) -> Tuple:
    """Function to find the fixed point of the momentum variable."""
    po, nglp, iG, dG, dld = fixed
    pmcand, _, num_iters = val
    # Compute the gradient of the Hamiltonian with respect to position.
    o = iG@pmcand
    a = nglp
    b = dld
    c = -0.5*o@dG@o
    gph = a + b + c
    # Check the computation like this:
    # >>> assert np.allclose(gph, grad_pos_hamiltonian(qo, pmcand))
    pm = po - half_step * gph
    delta = pm - pmcand
    return pm, delta, num_iters + 1

def position_step(val: Tuple, half_step: float, metric: Callable, fixed: Tuple) -> Tuple:
    """Function to find the fixed point of the position variable."""
    qo, pm, iGp = fixed
    qncand, _, num_iters = val
    m = iGp + solve_psd(metric(qncand), pm)
    # Check the computation like this:
    # >>> mp = (grad_mom_hamiltonian(qo, pm) + grad_mom_hamiltonian(qncand, pm))
    # >>> assert np.allclose(m, mp)
    qn = qo + half_step * m
    delta = qn - qncand
    return qn, delta, num_iters + 1

def _single_step_smart_generalized_leapfrog(
        grad_log_posterior: Callable,
        metric: Callable,
        grad_metric: Callable,
        grad_log_posterior_and_metric_and_grad_metric: Callable,
        zo: Tuple[np.ndarray],
        step_size: float,
        thresh: float,
        max_iters: int,
        cache: dict
) -> Tuple:
    """Implements the 'smart' generalized leapfrog integrator which avoids
    recomputing redundant quantities at each iteration.

    Args:
        log_posterior: Function to compute the log-posterior.
        grad_log_posterior: Function to compute the gradient of the log-posterior.
        metric: Function to compute the Fisher information metric.
        grad_metric: Function to compute the gradient of the Fisher information
            metric.
        grad_log_posterior_and_metric_and_grad_metric: A function that returns
            all of the gradient of the log-posterior, the Riemannian metric, and
            the derivatives of the metric.
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
    # Precompute the half step-size and the number of dimensions of the
    # position variable.
    half_step = 0.5 * step_size
    num_dims = len(zo[0])
    # Create partial functions to eliminate dependencies on the terminal
    # condition.
    termcond = partial(cond, thresh=thresh, max_iters=max_iters)

    # Precompute necessary quantities.
    qo, po = zo
    delta = np.ones_like(po) * np.inf
    Id = np.eye(num_dims)

    if 'aux' in cache:
        glp, iG, dG = cache['aux']
    else:
        glp, G, dG = grad_log_posterior_and_metric_and_grad_metric(qo)
        iG = solve_psd(G, Id)
    nglp = -glp
    dG = dG.swapaxes(0, -1)
    dld = 0.5*np.trace(np.asarray(np.hsplit(iG@np.hstack(dG), num_dims)), axis1=1, axis2=2)
    fixed = (po, nglp, iG, dG, dld)

    # The first step of the integrator is to find a fixed point of the momentum
    # variable.
    val = (po, delta, 0)
    momstep = partial(momentum_step, half_step=half_step, fixed=fixed)
    pm, delta_mom, num_iters_mom = while_loop(termcond, momstep, val)
    success_mom = np.all(delta_mom < thresh)

    # The second step of the integrator is to find a fixed point of the
    # position variable. The first momentum gradient could be conceivably
    # cached and saved.
    val = (qo, delta, 0)
    iGp = iG@pm
    fixed = (qo, pm, iGp)
    posstep = partial(position_step, half_step=half_step, metric=metric, fixed=fixed)
    qn, delta_pos, num_iters_pos = while_loop(termcond, posstep, val)
    success_pos = np.all(delta_pos < thresh)
    success = np.logical_and(success_mom, success_pos)

    # Increment the number of iterations by one since the final momentum update
    # requires computing the gradient of the potential function again.
    num_iters = num_iters_pos + num_iters_mom + 1

    # Last step is to do an explicit half-step of the momentum variable.
    glp, G, dG = grad_log_posterior_and_metric_and_grad_metric(qn)
    iG = solve_psd(G, Id)
    cache['aux'] = (glp, iG, dG)
    dG = dG.swapaxes(0, -1)
    o = iG@pm
    a = -glp
    b = 0.5*np.trace(np.asarray(np.hsplit(iG@np.hstack(dG), num_dims)), axis1=1, axis2=2)
    c = -0.5*o@dG@o
    gph = a + b + c
    pn = pm - half_step * gph
    return (qn, pn), cache, num_iters, success

def smart_generalized_leapfrog(
        grad_log_posterior: Callable,
        metric: Callable,
        grad_metric: Callable,
        grad_log_posterior_and_metric_and_grad_metric: Callable,
        zo: Tuple[np.ndarray],
        step_size: float,
        num_steps: int,
        thresh: float=1e-6,
        max_iters: int=1000
) -> Tuple:
    def step(it: int, val: Tuple):
        zo, cache, so = val
        zn, cache, _, sn = _single_step_smart_generalized_leapfrog(
            grad_log_posterior, metric, grad_metric,
            grad_log_posterior_and_metric_and_grad_metric,
            zo, step_size, thresh, max_iters, cache)
        success = np.logical_and(so, sn)
        return zn, cache, success
    (qn, pn), _, success = for_loop(0, num_steps, step, (zo, {}, True))
    return (qn, pn), success
