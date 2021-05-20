from typing import Callable, Tuple

import numpy as np
import scipy.linalg as spla
import scipy.stats as spst


def posterior_factory(mu: np.ndarray, Sigma: np.ndarray) -> Tuple[Callable]:
    """Implements sampling from a multivariate normal distribution. Constructs
    functions for the log-density of the normal distribution and for the
    gradient of the log-density.

    Args:
        mu: The mean of the multivariate normal distribution.
        Sigma: The covariance matrix of the multivariate normal distribution.

    Returns:
        log_posterior: The log-density of the multivariate normal.
        grad_log_posterior: The gradient of the log-density of the multivariate
            normal distribution.
        metric: The metric for the multivariate normal.

    """
    n = len(mu)
    L = spla.cholesky(Sigma)
    iL = spla.solve_triangular(L, np.eye(n))
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    iSigma = iL@iL.T

    # import numpy as onp
    # L, iL, logdet, iSigma = onp.array(L), onp.array(iL), float(logdet), onp.array(iSigma)
    # n = int(n)

    # Check calculations:
    # >>> np.allclose(logdet, np.log(np.linalg.det(Sigma)))
    # >>> np.allclose(iSigma, np.linalg.inv(Sigma))
    def log_posterior(x: np.ndarray) -> float:
        """Log-density of the multivariate normal distribution.

        Args:
            x: The location at which to evaluate the multivariate normal's
                log-density function.

        Returns:
             out: The value of the log-density.

        """
        o = x - mu
        maha = np.sum((o@iSigma)*o, axis=-1)
        return -0.5*n*np.log(2.0*np.pi) - 0.5*logdet -0.5*maha

    def grad_log_posterior(x: np.ndarray) -> np.ndarray:
        """Compute the gradient of the log-density of the multivariate normal
        distribution.

        Args:
            x: The location at which to evaluate the gradient of the log-density.

        Returns:
            out: The gradient of the log-density.

        """
        return -iSigma@(x - mu)

    def metric() -> np.ndarray:
        """Use the covariance matrix as a constant metric.

        Returns:
            Sigma: The covariance matrix of the multivariate normal distribution.

        """
        return iSigma

    return log_posterior, grad_log_posterior, metric
