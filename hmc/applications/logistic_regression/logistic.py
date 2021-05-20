from typing import Callable, Tuple

import numpy as np
import scipy.stats as spst


# Sigmoid function and its derivatives.
sigmoid = lambda z: 1.0 / (1.0 + np.exp(-z))
sigmoid_p = lambda z: sigmoid(z) * (1.0 - sigmoid(z))
sigmoid_pp = lambda z: sigmoid_p(z) - 2.0*sigmoid(z)*sigmoid_p(z)
sigmoid_ppp = lambda z: sigmoid_pp(z) - 2.0*np.square(sigmoid_p(z)) - 2.0*sigmoid(z)*sigmoid_pp(z)

def posterior_factory(x: np.ndarray, y: np.ndarray, alpha: float) -> Tuple[Callable]:
    """Factory function that yields further functions to compute the log-posterior
    of a Bayesian logistic regression model, the gradient of the log-posterior,
    the Fisher information metric, and the gradient of the Fisher information
    metric.

    Args:
        x: Covariates of the logistic regression.
        y: Binary targets of the logistic regression.
        alpha: The variance of the normal prior over the linear coefficients.

    Returns:
        log_posterior: Function to compute the log-posterior.
        grad_log_posterior: Function to compute the gradient of the log-posterior.
        metric: Function to compute the Fisher information metric.
        grad_metric: Function to compute the gradient of the Fisher information
            metric.

    """
    # Many functions require us to compute the precision of the normal prior.
    ialpha = np.reciprocal(alpha)

    def log_posterior(beta: np.ndarray) -> float:
        """Log-posterior of a Bayesian logistic regression with a Bernoulli likelihood
        and a normal prior over the linear coefficients.

        Args:
            beta: Linear coefficients of the logistic regression.

        Returns:
            lp: Log-posterior of the Bayesian logistic regression.

        """
        z = x@beta
        ll = -np.sum(np.maximum(z, 0.) - z*y + np.log(1.0 + np.exp(-np.abs(z))))
        # These are other equivalent calculations.
        # p = sigmoid(x@beta)
        # _ll = np.sum(y*np.log(p) + (1.-y)*np.log(1.-p))
        # _ll = spst.bernoulli.logpmf(y, p).sum()
        # assert np.allclose(ll, _ll)
        lbeta = -0.5 * ialpha * np.sum(np.square(beta))
        lp = ll + lbeta
        return lp

    def _grad_log_posterior_helper(lin: np.ndarray, beta) -> np.ndarray:
        yp = sigmoid(lin)
        glp = (y - yp)@x - ialpha * beta
        return glp

    def _metric_helper(lin: np.ndarray) -> np.ndarray:
        L = sigmoid_p(lin)
        G = (x.T*L)@x + ialpha * np.eye(x.shape[-1])
        return G

    def _grad_metric_helper(lin: np.ndarray) -> np.ndarray:
        o = sigmoid_pp(lin)
        Q = o[..., np.newaxis] * x
        dG = x.T@(Q[..., np.newaxis] * x[:, np.newaxis]).swapaxes(0, 1)
        return dG

    def grad_log_posterior(beta: np.ndarray) -> np.ndarray:
        """Gradient of the log-posterior of a Bayesian logistic regression model with
        respect to the linear coefficients.

        Args:
            beta: Linear coefficients of the logistic regression.

        Returns:
            out: The gradient of the log-posterior of the logistic regression model.

        """
        return _grad_log_posterior_helper(x@beta, beta)

    def hess_log_posterior(beta: np.ndarray) -> np.ndarray:
        """Hessian of the log-posterior of a Bayesian logistic regression model with
        respect to the linear coefficients. For a linear model, this is just
        the negative Fisher information.

        Args:
            beta: Linear coefficients of the logistic regression.

        Returns:
            out: The Hessian of the log-posterior of the logistic regression model.

        """
        return -metric(beta)

    def metric(beta: np.ndarray) -> np.ndarray:
        """Fisher information metric for Bayesian logistic regression model.

        Args:
            beta: Linear coefficients of the logistic regression.

        Returns:
            G: The Fisher information metric of the Bayesian logistic regression
                model.

        """
        return _metric_helper(x@beta)

    def grad_metric(beta: np.ndarray) -> np.ndarray:
        """The gradient of the Fisher information metric for Bayesian logistic
        regression with respect to the linear coefficients.

        Args:
            beta: Linear coefficients of the logistic regression.

        Returns:
            dG: The gradient of the Fisher information metric.

        """
        return _grad_metric_helper(x@beta)

    def grad_log_posterior_and_metric_and_grad_metric(beta: np.ndarray) -> np.ndarray:
        lin = x@beta
        glp = _grad_log_posterior_helper(lin, beta)
        G = _metric_helper(lin)
        dG = _grad_metric_helper(lin)
        return glp, G, dG

    def hess_metric(beta: np.ndarray) -> np.ndarray:
        """The Hessian of the Fisher information metric for Bayesian logistic
        regression with respect to the linear coefficients.

        Args:
            beta: Linear coefficients of the logistic regression.

        Returns:
            ddG: The Hessian of the Fisher information metric.

        """
        k = beta.size
        o = sigmoid_ppp(x@beta)
        Q = (o[..., np.newaxis]*x)[..., np.newaxis] * x[:, np.newaxis]
        ddG = (Q.T[..., np.newaxis, :] * x.T)@x
        return ddG

    return (
        log_posterior, grad_log_posterior, hess_log_posterior,
        metric, grad_metric, hess_metric,
        grad_log_posterior_and_metric_and_grad_metric
    )
