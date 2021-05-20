from typing import Callable

import numpy as np
import scipy.linalg as spla


def newton_raphson(q: np.ndarray, grad_log_posterior: Callable, metric: Callable, tol: float=1e-10) -> np.ndarray:
    """Implements the Newton-Raphson algorithm to find the maximum a posteriori of
    the posterior.

    Args:
        q: Initial guess for the location of the maximum of the posterior.
        grad_log_posterior: The gradient of the log-posterior. We will seek the
            zero of the gradient of the log-posterior, identifying a maximum.
        metric: The Fisher information metric to adapt the ascent direction to
            the local geometry.

    Returns:
        q: The maximizer of the posterior density.

    """
    delta = np.inf
    while delta > tol:
        g = grad_log_posterior(q)
        G = metric(q)
        q += spla.solve(G, g)
        delta = np.abs(g).max()
    return q
