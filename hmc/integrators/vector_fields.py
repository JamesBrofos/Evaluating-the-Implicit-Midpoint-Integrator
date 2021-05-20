from typing import Callable, Tuple, Union

import numpy as np
import scipy.linalg as spla
import scipy.stats as spst

from hmc.linalg import solve_psd


def basic_hamiltonian_vector_field(
        log_posterior: Callable,
        grad_log_posterior: Callable,
        sigmasq: float=1.0
) -> Tuple[Callable]:
    """The basic Hamiltonian vector field has a diagonal, constant, isometric mass
    preconditioner. The Hamiltonian energy is therefore separable.

    Args:
        log_posterior: Function to compute the log-posterior.
        grad_log_posterior: Function to compute the gradient of the log-posterior.
        sigmasq: The value of the constant, diagonal, isometric mass matrix.

    Returns:
        hamiltonian: Function to compute the Hamiltonian energy.
        grad_pos_hamiltonian: Function to compute the gradient of the Hamiltonian
            with respect to position.
        grad_mom_hamiltonian: Function to compute the gradient of the Hamiltonian
            with respect to momentum.
        vector_field: The Hamiltonian vector field.
        sample_momentum: Function to sample from the condition distribution of the
            momentum given the position.

    """
    sigma = np.sqrt(sigmasq)

    def hamiltonian(q: np.ndarray, p: np.ndarray) -> float:
        """Hamiltonian for sampling from the distribution."""
        U = -log_posterior(q)
        K = 0.5 * p@p / sigmasq
        return U + K

    def sample_momentum(q: np.ndarray) -> np.ndarray:
        """Sample momentum from the conditional distribution of the momentum."""
        return sigma * np.random.normal(size=(len(q), ))

    grad_pos_hamiltonian = lambda q: -grad_log_posterior(q)
    grad_mom_hamiltonian = lambda p: p / sigmasq
    vector_field = lambda q, p: (grad_mom_hamiltonian(p), -grad_pos_hamiltonian(q))
    return hamiltonian, grad_pos_hamiltonian, grad_mom_hamiltonian, vector_field, sample_momentum

_chol = lambda G: spla.cholesky(G, lower=True)

def euclidean_hamiltonian_vector_field(
        log_posterior: Callable,
        grad_log_posterior: Callable,
        metric: Callable,
        solver: Callable=solve_psd,
        chol: Callable=_chol
) -> Tuple[Callable]:
    """Construct the Euclidean equations of motion. These Hamiltonian dynamics are
    Euclidean in the sense that the metric is fixed; therefore, only the
    log-posterior, its gradient, and the fixed metric are required.

    Args:
        log_posterior: Function to compute the log-posterior.
        grad_log_posterior: Function to compute the gradient of the log-posterior.
        metric: Function to compute the Fisher information metric.
        solver: Function to compute the inverse of the constant metric; can be
            provided to exploit special structure.
        chol: Function to compute the lower Cholesky factor of the constant
            metric; can be provided to exploit special structure.

    Returns:
        hamiltonian: Function to compute the Hamiltonian energy.
        grad_pos_hamiltonian: Function to compute the gradient of the Hamiltonian
            with respect to position.
        grad_mom_hamiltonian: Function to compute the gradient of the Hamiltonian
            with respect to momentum.
        vector_field: The Hamiltonian vector field.
        sample_momentum: Function to sample from the condition distribution of the
            momentum given the position.

    """
    G = metric()
    num_dims = len(G)
    # Caution: SciPy computes a Cholesky factor that is upper triangular not
    # lower triangular by default.
    L = chol(G)
    iG = solver(G, np.eye(num_dims))

    # The various computations can be checked as follows:
    # Lp = spla.cholesky(G, lower=True)
    # assert np.allclose(Lp@Lp.T, G)
    # assert np.allclose(L, Lp)
    # assert np.allclose(iG@G, np.eye(num_dims))

    def hamiltonian(q: np.ndarray, p: np.ndarray) -> float:
        """Hamiltonian for sampling from the distribution."""
        U = -log_posterior(q)
        K = 0.5 * p@iG@p
        return U + K

    def sample_momentum(q: np.ndarray) -> np.ndarray:
        """Sample momentum from the conditional distribution of the momentum."""
        return L@np.random.normal(size=q.shape)

    grad_pos_hamiltonian = lambda q: -grad_log_posterior(q)
    grad_mom_hamiltonian = lambda p: iG@p
    vector_field = lambda q, p: (grad_mom_hamiltonian(p), -grad_pos_hamiltonian(q))
    return hamiltonian, grad_pos_hamiltonian, grad_mom_hamiltonian, vector_field, sample_momentum

def softabs_vector_field(
        log_posterior: Callable,
        grad_log_posterior: Callable,
        metric: Callable,
        grad_log_det: Callable,
        grad_quadratic_form: Callable
) -> Tuple[Callable]:
    """Convenience function to create the vector fields used in the softabs metric.
    In this case, we need to specify the gradient of the quadratic kinetic
    energy and the gradient of the log-determinant of the metric. This differs
    from the Riemannian manifold vector field setting because that function
    builds the vector field from the gradient of the metric; we don't have the
    gradient of the softabs metric so a separate approach is required.

    Args:
        log_posterior: Function to compute the log-posterior.
        grad_log_posterior: Function to compute the gradient of the log-posterior.
        metric: Function to compute the Fisher information metric.
        grad_metric: Function to compute the gradient of the Fisher information
            metric.

    Returns:
        hamiltonian: Function to compute the Hamiltonian energy.
        grad_pos_hamiltonian: Function to compute the gradient of the Hamiltonian
            with respect to position.
        grad_mom_hamiltonian: Function to compute the gradient of the Hamiltonian
            with respect to momentum.
        vector_field: The Hamiltonian vector field.
        sample_momentum: Function to sample from the condition distribution of the
            momentum given the position.

    """
    def hamiltonian(q: np.ndarray, p: np.ndarray) -> float:
        """Hamiltonian for sampling from the log-posterior with Riemannian HMC
        distribution.

        """
        U = -log_posterior(q)
        G = metric(q)
        K = -spst.multivariate_normal.logpdf(p, np.zeros_like(q), G)
        return U + K

    def _zeros_and_metric(q: np.ndarray):
        G = metric(q)
        Z = np.zeros_like(q)
        return G, Z

    def sample_momentum(q: np.ndarray) -> np.ndarray:
        """Sample momentum from the conditional distribution of the momentum."""
        G, Z = _zeros_and_metric(q)
        return np.random.multivariate_normal(Z, G)

    def grad_mom_hamiltonian(q: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Gradient of the Hamiltonian with respect to the momentum variable."""
        G = metric(q)
        return solve_psd(G, p)

    def grad_pos_hamiltonian(q: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Gradient of the Hamiltonian with respect to the position variable."""
        a = -grad_log_posterior(q)
        b = 0.5*grad_log_det(q)
        c = 0.5*grad_quadratic_form(q, p)
        return a + b + c

    def vector_field(q: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray]:
        """Hamiltonian vector field."""
        dq = grad_mom_hamiltonian(q, p)
        ndp = grad_pos_hamiltonian(q, p)
        dp = -ndp
        return dq, dp

    return hamiltonian, grad_mom_hamiltonian, grad_pos_hamiltonian, vector_field, sample_momentum

def riemannian_hamiltonian_vector_field(
        log_posterior: Callable,
        grad_log_posterior: Callable,
        metric: Callable,
        grad_metric: Callable,
        grad_log_posterior_and_metric_and_grad_metric: Callable
) -> Tuple[Callable]:
    """Convenience function for building the Riemannian manifold vector fields
    given the log-posterior, the gradient of the log-posterior, the Riemannian
    metric and the gradient of the Riemannian metric. This function computes
    the Hamiltonian, the gradient of the Hamiltonian with respect to position
    and momentum, and the Hamiltonian vector field.

    Args:
        log_posterior: Function to compute the log-posterior.
        grad_log_posterior: Function to compute the gradient of the log-posterior.
        metric: Function to compute the Fisher information metric.
        grad_metric: Function to compute the gradient of the Fisher information
            metric.
        grad_log_posterior_and_metric_and_grad_metric: A function that returns
            all of the gradient of the log-posterior, the Riemannian metric, and
            the derivatives of the metric.

    Returns:
        hamiltonian: Function to compute the Hamiltonian energy.
        grad_pos_hamiltonian: Function to compute the gradient of the Hamiltonian
            with respect to position.
        grad_mom_hamiltonian: Function to compute the gradient of the Hamiltonian
            with respect to momentum.
        vector_field: The Hamiltonian vector field.
        sample_momentum: Function to sample from the condition distribution of the
            momentum given the position.

    """
    def hamiltonian(q: np.ndarray, p: np.ndarray) -> float:
        """Hamiltonian for sampling from the distribution."""
        U = -log_posterior(q)
        G = metric(q)
        K = -spst.multivariate_normal.logpdf(p, np.zeros_like(q), G)
        return U + K

    def grad_mom_hamiltonian(q: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Gradient of the Hamiltonian with respect to the momentum variable."""
        G = metric(q)
        return solve_psd(G, p)

    def vector_field(q: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Hamiltonian vector field."""
        ndims = q.size
        a, G, dG = grad_log_posterior_and_metric_and_grad_metric(q)
        dG = dG.swapaxes(0, -1)
        iG = solve_psd(G, np.eye(ndims))
        dq = iG@p
        A = np.array(np.hsplit(iG@np.hstack(dG), ndims))
        b = -0.5*np.trace(A, axis1=1, axis2=2)
        c = 0.5*dq@dG@dq
        ndp = a + b + c
        return dq, ndp

    def grad_pos_hamiltonian(q: np.ndarray, p: np.ndarray):
        """Gradient of the Hamiltonian with respect to the position variable."""
        _, ndp = vector_field(q, p)
        dp = -ndp
        return dp

    def _zeros_and_metric(q: np.ndarray):
        G = metric(q)
        Z = np.zeros_like(q)
        return G, Z

    def sample_momentum(q: np.ndarray) -> np.ndarray:
        """Sample momentum from the conditional distribution of the momentum."""
        G, Z = _zeros_and_metric(q)
        return np.random.multivariate_normal(Z, G)

    return hamiltonian, grad_pos_hamiltonian, grad_mom_hamiltonian, vector_field, sample_momentum
