from typing import Callable, Tuple

import numpy as np


def posterior_factory(y: np.ndarray, sigma_y: float, sigma_theta: float) -> Tuple[Callable]:
    """The banana distribution is a distribution that exhibits a characteristic
    banana-shaped ridge that resembles the posterior that can emerge from
    models that are not identifiable. The distribution is the posterior of the
    following generative model.

        y ~ Normal(theta[0] + theta[1]**2, sigma_sq_y)
        theta[i] ~ Normal(0, sigma_sq_theta)

    Args:
        y: Observations of the banana model.
        sigma_y: Standard deviation of the observations.
        sigma_theta: Standard deviation of prior over linear coefficients.

    Returns:
        log_posterior: Function to compute the log-posterior.
        grad_log_posterior: Function to compute the gradient of the log-posterior.
        hess_log_posterior: Function to compute the Hessian of the log-posterior.
        metric: Function to compute the Fisher information metric.
        grad_metric: Function to compute the gradient of the Fisher information
            metric.
        hess_metric: Function to compute the Hessian of the Fisher information
            metric.

    """
    sigma_sq_y = np.square(sigma_y)
    sigma_sq_theta = np.square(sigma_theta)

    def log_posterior(theta: np.ndarray) -> float:
        """The banana-shaped distribution posterior.

        Args:
            theta: Linear coefficients.

        Returns:
            out: The log-posterior of the banana-shaped distribution.

        """
        p = theta[0] + np.square(theta[1])
        ll = -0.5 / sigma_sq_y * np.square(y - p).sum()
        lp = -0.5 / sigma_sq_theta * np.square(theta).sum()
        return ll + lp

    def grad_log_posterior(theta: np.ndarray) -> np.ndarray:
        """Gradient of the banana-shaped distribution with respect to the linear
        coefficients.

        Args:
            theta: Linear coefficients.

        Returns:
            out: The gradient of the log-posterior of the banana-shaped
                distribution with respect to the linear coefficients.

        """
        p = theta[0] + np.square(theta[1])
        d = np.sum(y - p)
        ga = d / sigma_sq_y - theta[0] / sigma_sq_theta
        gb = 2.0*d / sigma_sq_y * theta[1] - theta[1] / sigma_sq_theta
        return np.hstack((ga, gb))

    def hess_log_posterior(theta: np.ndarray) -> np.ndarray:
        """Hessian of the banana-shaped distribution with respect to the linear
        coefficients.

        Args:
            theta: Linear coefficients.

        Returns:
            out: The Hessian of the log-posterior of the banana-shaped
                distribution with respect to the linear coefficients.

        """
        n = y.size
        p = theta[0] + np.square(theta[1])
        s = 2.0*n*theta[1] / sigma_sq_y
        H = -metric(theta)
        U = np.array([
            [0.0, 0.0],
            [0.0, 2.0 * (y - p).sum() / sigma_sq_y]
        ])
        H += U
        return H

    def metric(theta: np.ndarray) -> np.ndarray:
        """The Fisher information is the negative expected outer product of the
        gradient of the posterior.

        Args:
            theta: Linear coefficients.

        Returns:
            G: The Fisher information metric of the banana-shaped distribution.

        """
        n = y.size
        s = 2.0*n*theta[1] / sigma_sq_y
        G = np.array([[n / sigma_sq_y + 1.0 / sigma_sq_theta, s],
                      [s, 4.0*n*np.square(theta[1]) / sigma_sq_y + 1.0 / sigma_sq_theta]])
        return G

    def grad_metric(theta: np.ndarray) -> np.ndarray:
        """The gradient of the Fisher information metric with respect to the linear
        coefficients.

        Args:
            theta: Linear coefficients.

        Returns:
            dG: The gradient of the Fisher information metric with respect to the
                linear coefficients.

        """
        n = y.size
        dG = np.array([
            [[0.0, 0.0], [0.0, 2.0*n / sigma_sq_y]],
            [[0.0, 2.0*n / sigma_sq_y], [0.0, 8.0*n*theta[1] / sigma_sq_y]]
        ])
        return dG

    def hess_metric(theta: np.ndarray) -> np.ndarray:
        """The Hessian of the Fisher information metric with respect to the linear
        coefficients.

        Args:
            theta: Linear coefficients.

        Returns:
            ddG: The Hessian of the Fisher information metric with respect to the
                linear coefficients.

        """
        l = 8*y.size / sigma_sq_y
        ddG = np.array([
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, l
        ]).reshape((2, 2, 2, 2))
        return ddG

    def grad_log_posterior_and_metric_and_grad_metric(theta):
        glp = grad_log_posterior(theta)
        G = metric(theta)
        dG = grad_metric(theta)
        return glp, G, dG

    return (
        log_posterior, grad_log_posterior, hess_log_posterior,
        metric, grad_metric, hess_metric,
        grad_log_posterior_and_metric_and_grad_metric)

def generate_data(t: float, sigma_y: float, sigma_theta: float, num_obs: int) -> np.ndarray:
    """Generate data from the banana-shaped posterior distribution.

    Args:
        t: Free-parameter determining the thetas.
        sigma_y: Noise standard deviation.
        sigma_theta: Prior standard deviation over the thetas.
        num_obs: Number of observations to generate.

    Returns:
        theta: Linear coefficients of the banana-shaped distribution.
        y: Observations from the unidentifiable model.

    """
    theta = np.array([t, np.sqrt(1.0 - t)])
    y = theta[0] + np.square(theta[1]) + sigma_y * np.random.normal(size=(num_obs, ))
    return theta, y
