from functools import partial
from typing import Callable, Tuple

import numpy as np

from hmc.core import for_loop, while_loop
from hmc.integrators.terminal import cond


def step(val: Tuple, zo: np.ndarray, step_size: float, vector_field: Callable) -> Tuple:
    """Single step of the implicit midpoint integrator. Computes the midpoint,
    evaluates the gradient at the midpoint, takes a step from the initial
    position in the direction of the gradient at the midpoint, and measures the
    difference between the resulting point and the candidate stationary point.

    """
    zmcand, _, _, num_iters = val
    dz = np.hstack(vector_field(*np.split(zmcand, 2)))
    zm = zo + 0.5 * step_size * dz
    delta = zm - zmcand
    return zm, dz, delta, num_iters + 1

def _single_step_smart_implicit_midpoint(
        vector_field: Callable,
        zo: Tuple[np.ndarray],
        step_size: float,
        thresh: float,
        max_iters: int) -> Tuple:
    """Implements the implicit midpoint integrator. The implicit midpoint
    integrator is symmetric, symplectic, and second-order accurate (third-order
    local error).

    Args:
        vector_field: The Hamiltonian vector field.
        zo: Tuple containing the position and momentum variables in the original
            phase space.
        step_size: Integration step_size.
        thresh: Convergence tolerance for fixed point iterations.
        max_iters: Maximum number of fixed point iterations.

    Returns:
        qn: The terminal position variable.
        pn: The terminal momentum variable.
        num_iters: The number of fixed point iterations to find the midpoint.
        success: Boolean flag indicating successful integration.

    """
    # Initial candidate.
    qo, po = zo
    zo = np.hstack((qo, po))
    # Fixed point iteration.
    delta = np.ones_like(zo) * np.inf
    dz = np.hstack(vector_field(*np.split(zo, 2)))
    zopred = zo + 0.5 * step_size * dz
    val = (zopred, np.zeros_like(zo), delta, 0)
    zm, dzm, delta, num_iters = while_loop(
        partial(cond, thresh=thresh, max_iters=max_iters),
        partial(step, zo=zo, step_size=step_size, vector_field=vector_field),
        val)

    # Determine whether or not the integration was successful.
    success = np.all(delta < thresh)
    # Final explicit Euler step.
    zn = zm + 0.5 * step_size * dzm
    qn, pn = np.split(zn, 2)
    return (qn, pn), num_iters, success

def smart_implicit_midpoint(
        vector_field: Callable,
        zo: Tuple[np.ndarray],
        step_size: float,
        num_steps: int,
        thresh: float=1e-6,
        max_iters: int=1000
) -> Tuple:
    def step(it: int, val: Tuple):
        zo, so = val
        zn, _, sn = _single_step_smart_implicit_midpoint(vector_field, zo, step_size, thresh, max_iters)
        success = np.logical_and(so, sn)
        return zn, success
    (qn, pn), success = for_loop(0, num_steps, step, (zo, True))
    return (qn, pn), success
