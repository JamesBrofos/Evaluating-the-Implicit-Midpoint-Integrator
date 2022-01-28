from functools import partial
from typing import Callable, Tuple

import numpy as np

from hmc.core import for_loop, while_loop
from hmc.integrators.terminal import cond
from hmc.linalg import solve_psd


def lagrange_step(val: Tuple, Id: np.ndarray, qo: np.ndarray, po: np.ndarray, step_size: float, grad_log_posterior_and_metric_and_grad_metric: Callable) -> Tuple:
    (qncand, pmcand, _), _, num_iters = val
    qm = 0.5*(qncand + qo)
    ndims = len(qo)
    a, G, dG = grad_log_posterior_and_metric_and_grad_metric(qm)
    iG = solve_psd(G, Id)
    dG = dG.swapaxes(0, -1)

    vm = iG@pmcand
    qn = qo + step_size * vm

    A = np.array(np.hsplit(iG@np.hstack(dG), ndims))
    b = -0.5*np.trace(A, axis1=1, axis2=2)
    c = 0.5*vm@dG@vm

    delta_mom = 0.5*step_size*(a + b + c)
    pm = po + delta_mom
    pn = pm + delta_mom

    delta = np.hstack((qn - qncand, pm - pmcand))
    return (qn, pm, pn), delta, num_iters + 1

def _single_step_lagrange_implicit_midpoint(
        grad_log_posterior_and_metric_and_grad_metric: Callable,
        zo: Tuple[np.ndarray],
        step_size: float,
        thresh: float,
        max_iters: int
) -> Tuple:
    termcond = partial(cond, thresh=thresh, max_iters=max_iters)
    qo, po = zo
    ndims = len(qo)
    delta = np.ones(2*ndims) * np.inf
    Id = np.eye(ndims)
    val = ((qo, po, po), delta, 0)
    (qn, _, pn), delta, num_iters = while_loop(
        termcond,
        partial(lagrange_step, Id=Id, qo=qo, po=po, step_size=step_size,
                grad_log_posterior_and_metric_and_grad_metric=grad_log_posterior_and_metric_and_grad_metric),
        val
    )
    success = np.all(delta < thresh)
    return (qn, pn), num_iters, success

def _lagrange_implicit_midpoint(
        grad_log_posterior_and_metric_and_grad_metric: Callable,
        zo: Tuple[np.ndarray],
        step_size: float,
        num_steps: int,
        thresh: float,
        max_iters: int
) -> Tuple:
    def step(it: int, val: Tuple):
        zo, so = val
        zn, _, sn = _single_step_lagrange_implicit_midpoint(
            grad_log_posterior_and_metric_and_grad_metric, zo, step_size, thresh, max_iters)
        success = np.logical_and(so, sn)
        return zn, success
    (qn, pn), success = for_loop(0, num_steps, step, (zo, True))
    return (qn, pn), success

def lagrange_implicit_midpoint(
        grad_log_posterior_and_metric_and_grad_metric: Callable,
        zo: Tuple[np.ndarray],
        step_size: float,
        num_steps: int,
        thresh: float=1e-6,
        max_iters: int=1000
):
    return _lagrange_implicit_midpoint(
        grad_log_posterior_and_metric_and_grad_metric,
        zo,
        step_size,
        num_steps,
        thresh,
        max_iters
    )
