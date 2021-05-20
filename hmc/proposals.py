from typing import Callable, Tuple

import numpy as np

from hmc.integrators import (
    leapfrog, generalized_leapfrog, smart_generalized_leapfrog,
    implicit_midpoint, smart_implicit_midpoint)


def leapfrog_proposal_factory(grad_pos_hamiltonian: Callable, grad_mom_hamiltonian: Callable) -> Callable:
    """Euclidean proposal distribution for a separable Hamiltonian. Integration is
    performed using the standard leapfrog integrator.

    Args:
        grad_pos_hamiltonian: Gradient of the Hamiltonian with respect to the
            position variable.
        grad_mom_hamiltonian: Gradient of the Hamiltonian with respect to the
            momentum variable.

    Returns:
        proposal: Proposal operator for Hamiltonian Monte Carlo.

    """
    def proposal(q: np.ndarray, p: np.ndarray, step_size: float, num_steps: int) -> Tuple:
        return leapfrog(grad_pos_hamiltonian, grad_mom_hamiltonian, (q, p), step_size, num_steps)
    return proposal

def generalized_leapfrog_proposal_factory(grad_pos_hamiltonian: Callable, grad_mom_hamiltonian: Callable, thresh: float=1e-6, max_iters: int=1000) -> Callable:
    """This proposal distribution for Hamiltonian Monte Carlo uses the generalized
    leapfrog integrator.

    Args:
        grad_pos_hamiltonian: Gradient of the Hamiltonian with respect to the
            position variable.
        grad_mom_hamiltonian: Gradient of the Hamiltonian with respect to the
            momentum variable.
        thresh: Convergence threshold.
        max_iters: Maximum number of internal iterations for integrator.

    Returns:
        proposal: Proposal operator for Hamiltonian Monte Carlo.

    """
    def proposal(q: np.ndarray, p: np.ndarray, step_size: float, num_steps: int) -> Tuple:
        (q, p), success = generalized_leapfrog(grad_pos_hamiltonian, grad_mom_hamiltonian, (q, p), step_size, num_steps, thresh, max_iters)
        return (q, p), success
    return proposal

def smart_generalized_leapfrog_proposal_factory(
        grad_log_posterior: Callable, metric: Callable, grad_metric: Callable,
        grad_log_posterior_and_metric_and_grad_metric: Callable,
        thresh: float=1e-6, max_iters: int=1000
) -> Callable:
    """This proposal distribution for Hamiltonian Monte Carlo uses the
    'smart' generalized leapfrog integrator.

    Args:
        grad_log_posterior: Function to compute the gradient of the
            log-posterior.
        metric: Function to compute the Riemannian metric.
        grad_metric: Function to compute the gradient of the Riemannian metric.

    Returns:
        proposal: Proposal operator for Hamiltonian Monte Carlo.

    """
    def proposal(q: np.ndarray, p: np.ndarray, step_size: float, num_steps: int) -> Tuple:
        (q, p), success = smart_generalized_leapfrog(grad_log_posterior, metric, grad_metric, grad_log_posterior_and_metric_and_grad_metric, (q, p), step_size, num_steps, thresh, max_iters)
        return (q, p), success
    return proposal

def implicit_midpoint_proposal_factory(vector_field: Callable, thresh: float=1e-6, max_iters: int=1000) -> Callable:
    """This proposal distribution for Hamiltonian Monte Carlo uses the implicit
    midpoint integrator.

    Args:
        vector_field: Hamiltonian vector field.
        thresh: Convergence threshold.
        max_iters: Maximum number of internal iterations for integrator.

    Returns:
        proposal: Proposal operator for Hamiltonian Monte Carlo.

    """
    def proposal(q: np.ndarray, p: np.ndarray, step_size: float, num_steps: int) -> Tuple:
        (q, p), success = implicit_midpoint(vector_field, (q, p), step_size, num_steps, thresh, max_iters)
        return (q, p), success
    return proposal

def smart_implicit_midpoint_proposal_factory(vector_field: Callable, thresh: float=1e-6, max_iters: int=1000) -> Callable:
    """This proposal distribution for Hamiltonian Monte Carlo uses the smart
    implicit midpoint integrator.

    Args:
        vector_field: Hamiltonian vector field.
        thresh: Convergence threshold.
        max_iters: Maximum number of internal iterations for integrator.

    Returns:
        proposal: Proposal operator for Hamiltonian Monte Carlo.

    """
    def proposal(q: np.ndarray, p: np.ndarray, step_size: float, num_steps: int) -> Tuple:
        (q, p), success = smart_implicit_midpoint(vector_field, (q, p), step_size, num_steps, thresh, max_iters)
        return (q, p), success
    return proposal
