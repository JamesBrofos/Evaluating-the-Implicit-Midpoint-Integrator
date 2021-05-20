from typing import Callable, Tuple

import numpy as np
import scipy.special as spsp
import scipy.stats as spst


def forward_transform(q: np.ndarray):
    """Transform parameters frm their constrained representation to their
    unconstrained representation.

    Args:
        q: The constrained parameter representation.

    Returns:
        qt: The unconstrained parameter representation.
        ildj: The logarithm of the Jacobian determinant of the inverse
            transformation.

    """
    sigma, phi, beta = q
    gamma, alpha = np.log(sigma), np.arctanh(phi)
    qt = np.array([gamma, alpha, beta])
    ildj = gamma + np.log(1.0 - np.square(phi))
    return qt, ildj

def inverse_transform(qt: np.ndarray):
    """Transform parameters from their unconstrained representation to their
    constrained representation.

    Args:
        qt: The unconstrained parameter representation.

    Returns:
        q: The constrained parameter representation.

    """
    gamma, alpha, beta = qt
    sigma, phi = np.exp(gamma), np.tanh(alpha)
    q = np.array([sigma, phi, beta])
    fldj = -(gamma + np.log(1.0 - np.square(phi)))
    return q, fldj

def generate_data(T: int, sigma: float=0.15, phi: float=0.98, beta: float=0.65) -> Tuple[np.ndarray]:
    """Generate samples from the stochastic volatility model.

    Args:
        T: Number of subsequent time points at which to generate an observation
            of the stochastic volatility model.
        sigma: Parameter of the stochastic volatility model.
        phi: Parameter of the stochastic volatility model.
        beta: Parameter of the stochastic volatility model.

    Returns:
        x: The stochastic volatilities.
        y: Observations from the stochastic volatility model.

    """
    x = np.random.normal(size=(T, ))
    x[0] = np.random.normal(0.0, sigma / np.sqrt(1.0 - np.square(phi)))
    for t in range(1, T):
        eta = np.random.normal(0.0, sigma)
        x[t] = phi * x[t-1] + eta

    eps = np.random.normal(size=x.shape)
    y = eps * beta * np.exp(0.5 * x)
    return x, y

def inv_chisq_logpdf(x: float, v: float, t: float) -> float:
    """The log-density function of the inverse chi-squared distribution.

    Args:
        x: The location at which to evaluate the density of the inverse
            chi-squared distribution.
        v: Degrees of freedom of inverse chi-squared distribution.
        t: Scale factor of the inverse chi-squared distribution.

    Returns:
        out: The log-density of the inverse chi-squared distribution.

    """
    hv = 0.5*v
    a = hv * (np.log(t) + np.log(hv))
    b = -spsp.gammaln(hv)
    c = -v*t / (2.0 * x)
    d = -(1.0 + hv) * np.log(x)
    return a + b + c + d

def grad_inv_chisq_logpdf(x: float, v: float, t: float) -> float:
    """The gradient of the log-density function of the inverse chi-squared
    distribution.

    Args:
        x: The location at which to evaluate the density of the inverse
            chi-squared distribution.
        v: Degrees of freedom of inverse chi-squared distribution.
        t: Scale factor of the inverse chi-squared distribution.

    Returns:
        out: The gradient of the log-density of the inverse chi-squared
            distribution.

    """
    hv = 0.5*v
    return -(1.0 + hv) / x + 0.5 * v*t / np.square(x)

def hess_inv_chisq_logpdf(x: float, v: float, t: float) -> float:
    """The hessian of the log-density function of the inverse chi-squared
    distribution.

    Args:
        x: The location at which to evaluate the density of the inverse
            chi-squared distribution.
        v: Degrees of freedom of inverse chi-squared distribution.
        t: Scale factor of the inverse chi-squared distribution.

    Returns:
        out: The hessian of the log-density of the inverse chi-squared
            distribution.

    """
    hv = 0.5*v
    return (1.0 + hv) / np.square(x) - v*t / np.power(x, 3.0)

def grad_hess_inv_chisq_logpdf(x: float, v: float, t: float) -> float:
    """The third-order derivative of the log-density function of the inverse
    chi-squared distribution.

    Args:
        x: The location at which to evaluate the density of the inverse
            chi-squared distribution.
        v: Degrees of freedom of inverse chi-squared distribution.
        t: Scale factor of the inverse chi-squared distribution.

    Returns:
        out: The third-order derivative of the log-density of the inverse
            chi-squared distribution.

    """
    hv = 0.5*v
    return -2.0 * (1.0 + hv) / np.power(x, 3.0) + 3.0 * v*t / np.power(x, 4.0)

def grad_beta_logpdf(p: float, alpha: float, beta: float) -> float:
    """Gradient of the log-density function of the Beta distribution.

    Args:
        p: A point on the unit interval at which to evaluate the density.
        alpha: Beta distribution 'success' parameter.
        beta: Beta distribution 'failure' parameter.

    Returns:
        out: The gradient of the log-density of the Beta distribution.

    """
    return (alpha - 1.0) / p - (beta - 1.0) / (1.0 - p)

def hess_beta_logpdf(p: float, alpha: float, beta: float) -> float:
    """Hessian of the log-density function of the Beta distribution.

    Args:
        p: A point on the unit interval at which to evaluate the density.
        alpha: Beta distribution 'success' parameter.
        beta: Beta distribution 'failure' parameter.

    Returns:
        out: The hessian of the log-density of the Beta distribution.

    """
    return (1.0 - alpha) / np.square(p) + (1.0 - beta) / np.square(p - 1.0)

def grad_hess_beta_logpdf(p: float, alpha: float, beta: float) -> float:
    """Third-order derivative of the log-density function of the Beta
    distribution.

    Args:
        p: A point on the unit interval at which to evaluate the density.
        alpha: Beta distribution 'success' parameter.
        beta: Beta distribution 'failure' parameter.

    Returns:
        out: The third-order derivative of the log-density of the Beta
            distribution.

    """
    a = 2 * (alpha - 1.0) / np.power(p, 3.0)
    b = 2 * (beta - 1.0) / np.power(p - 1.0, 3.0)
    return a + b

def log_prior(sigma: float, phi: float, beta: float) -> float:
    """The log-prior density for the stochastic volatility model. The prior
    distribution is as follows (notice that the beta prior is improper):

    sigma ~ InvChiSquare(10.0, 0.05)
    (phi + 1) / 2 ~ Beta(20, 1.5)
    p(beta) = 1 / beta

    Args:
        sigma: Parameter of the stochastic volatility model.
        phi: Parameter of the stochastic volatility model.
        beta: Parameter of the stochastic volatility model.

    Returns:
        out: The log-density of the prior.

    """
    sigmasq = np.square(sigma)
    lbeta = -np.log(beta)
    lphi = spst.beta.logpdf(0.5*(phi + 1.0), 20.0, 1.5)
    lsigmasq = inv_chisq_logpdf(sigmasq, 10.0, 0.05)
    return lbeta + lphi + lsigmasq

def grad_log_prior(gamma: float, alpha: float, beta: float) -> Tuple[float]:
    """The gradient of the log-density of the prior for the stochastic volatility
    model. We reparameterize `sigma` and `phi` to respect parameter
    constraints.

    Args:
        gamma: Parameter of the stochastic volatility model.
        alpha: Parameter of the stochastic volatility model.
        beta: Parameter of the stochastic volatility model.

    Returns:
        dgamma: The gradient with respect to `gamma`.
        dalpha: The gradient with respect to `alpha`.
        dbeta: The gradient with respect to `beta`.

    """
    sigma = np.exp(gamma)
    sigmasq = np.square(sigma)
    phi = np.tanh(alpha)
    phisq = np.square(phi)
    dbeta = -1.0 / beta
    dgamma = 2.0 * sigmasq * grad_inv_chisq_logpdf(sigmasq, 10.0, 0.05)
    dalpha = 0.5 * (1.0 - phisq) * grad_beta_logpdf(0.5*(phi + 1.0), 20.0, 1.5)
    return dgamma, dalpha, dbeta

def hess_log_prior(gamma: float, alpha: float, beta: float) -> np.ndarray:
    """The hessian of the log-density of the prior for the stochastic volatility
    model.

    Args:
        gamma: Parameter of the stochastic volatility model.
        alpha: Parameter of the stochastic volatility model.
        beta: Parameter of the stochastic volatility model.

    Returns:
        H: The Hessian of the log-prior density.

    """
    sigma = np.exp(gamma)
    sigmasq = np.square(sigma)
    phi = np.tanh(alpha)
    phisq = np.square(phi)
    m = 0.5*(phi + 1.0)
    a = 4.0*sigmasq*grad_inv_chisq_logpdf(sigmasq, 10.0, 0.05)
    a += 4.0*np.square(sigmasq)*hess_inv_chisq_logpdf(sigmasq, 10.0, 0.05)
    b = -phi*(1.0 - phisq)*grad_beta_logpdf(m, 20.0, 1.5)
    b += 0.25*np.square(1.0 - phisq)*hess_beta_logpdf(m, 20.0, 1.5)
    c = 1.0 / np.square(beta)
    H = np.array([[a, 0.0, 0.0], [0.0, b, 0.0], [0.0, 0.0, c]])
    return H

def grad_hess_log_prior(gamma: float, alpha: float, beta: float) -> np.ndarray:
    """The tensor of higher-order derivatives of the log-density of the prior for
    the stochastic volatility model.

    Args:
        gamma: Parameter of the stochastic volatility model.
        alpha: Parameter of the stochastic volatility model.
        beta: Parameter of the stochastic volatility model.

    Returns:
        dH: The tensor of higher-order derivatives of the log-prior density.

    """
    sigma = np.exp(gamma)
    sigmasq = np.square(sigma)
    sigmaquad = np.square(sigmasq)
    phi = np.tanh(alpha)
    phisq = np.square(phi)
    m = 0.5*(phi + 1.0)
    r = 1 - phisq
    h = hess_beta_logpdf(m, 20.0, 1.5)
    dHbeta = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, -2.0 / np.power(beta, 3.0)]
    ])
    ghx = grad_hess_inv_chisq_logpdf(sigmasq, 10.0, 0.05)
    a = 8.0*sigmasq*grad_inv_chisq_logpdf(sigmasq, 10.0, 0.05)
    a += 24.0*sigmaquad*hess_inv_chisq_logpdf(sigmasq, 10.0, 0.05)
    a += 8.0*sigmaquad*sigmasq*ghx
    dHgamma = np.array([
        [a, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    ])
    b = r*(3.0*phisq - 1.0)*grad_beta_logpdf(m, 20.0, 1.5)
    b -= 0.5*phi*np.square(r)*h
    b -= phi*np.square(r)*h
    b += np.power(r, 3.0)*grad_hess_beta_logpdf(m, 20.0, 1.5) / 8.0
    dHalpha = np.array([
        [0.0, 0.0, 0.0], [0.0, b, 0.0], [0.0, 0.0, 0.0]
    ])
    return np.array([dHgamma, dHalpha, dHbeta])

def latent_posterior_factory(x: np.ndarray, y: np.ndarray) -> Tuple[Callable]:
    """Factory function that yields further functions to compute the log-posterior
    of the stochastic volatility model given parameters `x`. The factory also
    constructs functions for the gradient of the log-posterior and the Fisher
    information metric.

    Args:
        x: The stochastic volatilities.
        y: Observations from the stochastic volatility model.

    Returns:
        log_posterior: Function to compute the log-posterior.
        grad_log_posterior: Function to compute the gradient of the log-posterior.
        metric: Function to compute the Fisher information metric.
        grad_metric: Function to compute the gradient of the Fisher information
            metric.

    """
    T = x.size

    def _log_posterior(sigma: float, phi: float, beta: float) -> float:
        """The log-posterior of the stochastic volatility model given the stochastic
        volatilities. The inference is over the model parameters `sigma`, `phi`,
        and `beta`.

        Args:
            sigma: Parameter of the stochastic volatility model.
            phi: Parameter of the stochastic volatility model.
            beta: Parameter of the stochastic volatility model.

        Returns:
            lp: The log-posterior of the stochastic volatility model.

        """
        phisq = np.square(phi)
        ly = spst.norm.logpdf(y, 0.0, beta*np.exp(0.5 * x)).sum()
        lxo = spst.norm.logpdf(x[0], 0.0, sigma / np.sqrt(1.0 - phisq))
        lx = spst.norm.logpdf(x[1:], phi*x[:-1], sigma).sum()
        lp = ly + lx + lxo + log_prior(sigma, phi, beta)
        return lp

    def _grad_log_posterior_helper(gamma, alpha, beta, sigmasq, phi, phisq):
        dpgamma, dpalpha, dpbeta = grad_log_prior(gamma, alpha, beta)
        dbeta = (-T / beta
                 + np.sum(np.square(y) / np.exp(x)) / np.power(beta, 3.0)
                 + dpbeta)
        dgamma = (
            -T + np.square(x[0])*(1.0 - phisq) / sigmasq
            + np.sum(np.square(x[1:] - phi*x[:-1])) / sigmasq
            + dpgamma)
        dalpha = (
            -phi + phi*np.square(x[0])*(1.0 - phisq) / sigmasq
            + np.sum(x[:-1] * (x[1:] - phi*x[:-1])) * (1.0 - phisq) / sigmasq
            + dpalpha)
        return np.array([dgamma, dalpha, dbeta])

    def _metric_helper(gamma, alpha, beta, sigmasq, phi, phisq):
        # Note that this ordering of the variables differs from that presented
        # in the Riemannian manifold HMC paper.
        G = np.array([
            #  gamma                                alpha                       beta
            [  2.0*T,                             2.0*phi,                       0.0], # gamma
            [2.0*phi, 2.0*phisq + (T - 1.0)*(1.0 - phisq),                       0.0], # alpha
            [    0.0,                                 0.0, 2.0 * T / np.square(beta)]  # beta
        ])
        # Add in the negative Hessian of the log-prior.
        H = hess_log_prior(gamma, alpha, beta)
        G -= H
        return G

    def _grad_metric_helper(gamma, alpha, beta, sigmasq, phi, phisq):
        dGbeta = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, -4.0 * T / np.power(beta, 3.0)]
        ])
        dGgamma = np.zeros((3, 3))
        a = 2.0*(1.0 - phisq)
        b = 2.0*phi*(3.0 - T)*(1.0 - phisq)
        dGalpha = np.array([
            [0.0,   a, 0.0],
            [  a,   b, 0.0],
            [0.0, 0.0, 0.0]
        ])
        dG = np.array([dGgamma, dGalpha, dGbeta]).swapaxes(0, -1)
        dH = grad_hess_log_prior(gamma, alpha, beta)

        return dG - dH


    def _grad_log_posterior(gamma: float, alpha: float, beta: float) -> np.ndarray:
        """The gradient log-posterior of the stochastic volatility model given the
        stochastic volatilities with respect to the (transformed) parameters
        `gamma`, `alpha`, and `beta`.

        Args:
            gamma: Transformed parameter `sigma` of the stochastic volatility model.
            alpha: Transformed parameter `phi` of the stochastic volatility model.
            beta: Parameter of the stochastic volatility model.

        Returns:
            dgamma: The gradient of the log-posterior with respect to the
                transformed parameter `sigma`.
            dalpha: The gradient of the log-posterior with respect to the
                transformed parameter `phi`.
            dbeta: The gradient of the log-posterior with respect to `beta`.

        """
        sigma = np.exp(gamma)
        sigmasq = np.square(sigma)
        phi = np.tanh(alpha)
        phisq = np.square(phi)
        return _grad_log_posterior_helper(gamma, alpha, beta, sigmasq, phi, phisq)

    def _metric(gamma: float, alpha: float, beta: float) -> np.ndarray:
        """The Fisher information metric of the stochastic volatility model given the
        stochastic volatilities.

        Args:
            gamma: Transformed parameter of the stochastic volatility model.
            alpha: Transformed parameter of the stochastic volatility model.
            beta: Parameter of the stochastic volatility model.

        Returns:
            G: The Fisher information metric.

        """
        sigma = np.exp(gamma)
        sigmasq = np.square(sigma)
        phi = np.tanh(alpha)
        phisq = np.square(phi)
        return _metric_helper(gamma, alpha, beta, sigmasq, phi, phisq)

    def _grad_metric(gamma: float, alpha: float, beta: float) -> np.ndarray:
        """The gradient of the Fisher information metric of the stochastic volatility
        model given the stochastic volatilities with respect to the `sigma`,
        `alpha`, and `beta` parameters of the stochastic volatility model.

        Args:
            gamma: Transformed parameter of the stochastic volatility model.
            alpha: Transformed parameter of the stochastic volatility model.
            beta: Parameter of the stochastic volatility model.

        Returns:
            dG: The gradient of the Fisher information metric.

        """
        sigma = np.exp(gamma)
        sigmasq = np.square(sigma)
        phi = np.tanh(alpha)
        phisq = np.square(phi)
        return _grad_metric_helper(gamma, alpha, beta, sigmasq, phi, phisq)

    def grad_log_posterior_and_metric_and_grad_metric(q):
        gamma, alpha, beta = q
        sigma = np.exp(gamma)
        sigmasq = np.square(sigma)
        phi = np.tanh(alpha)
        phisq = np.square(phi)
        glp = _grad_log_posterior_helper(gamma, alpha, beta, sigmasq, phi, phisq)
        G = _metric_helper(gamma, alpha, beta, sigmasq, phi, phisq)
        dG = _grad_metric_helper(gamma, alpha, beta, sigmasq, phi, phisq)
        return glp, G, dG

    # Convert functions defined for separate arguments to take a vector
    # concatenation of the parameter.
    log_posterior = lambda q: _log_posterior(*inverse_transform(q)[0])
    grad_log_posterior = lambda q: _grad_log_posterior(q[0], q[1], q[2])
    metric = lambda q: _metric(q[0], q[1], q[2])
    grad_metric = lambda q: _grad_metric(q[0], q[1], q[2])

    return (
        log_posterior, grad_log_posterior, metric, grad_metric,
        grad_log_posterior_and_metric_and_grad_metric)

def volatility_posterior_factory(sigma: float, phi: float, beta: float, y: np.ndarray) -> Tuple[Callable]:
    """Factory function that yields further functions to compute the log-posterior
    of the stochastic volatility model given parameters `sigma`, `phi`, and
    `beta`. The factory also constructs functions for the gradient of the
    log-posterior and the Fisher information metric.

    Args:
        sigma: Parameter of the stochastic volatility model.
        phi: Parameter of the stochastic volatility model.
        beta: Parameter of the stochastic volatility model.
        y: Observations from the stochastic volatility model.

    Returns:
        log_posterior: Function to compute the log-posterior.
        grad_log_posterior: Function to compute the gradient of the log-posterior.
        metric: Function to compute the Fisher information metric.

    """
    sigmasq = np.square(sigma)
    phisq = np.square(phi)

    def log_posterior(x: np.ndarray) -> float:
        """The log-posterior of the stochastic volatility model given values of `phi`,
        `sigma`, and `beta`. The posterior is over the stochastic volatilities.

        Args:
            x: The stochastic volatilities.

        Returns:
            lp: The log-posterior of the stochastic volatility model.

        """
        ly = spst.norm.logpdf(y, 0.0, beta*np.exp(0.5 * x)).sum()
        lxo = spst.norm.logpdf(x[0], 0.0, sigma / np.sqrt(1.0 - phisq))
        lx = spst.norm.logpdf(x[1:], phi*x[:-1], sigma).sum()
        lp = ly + lx + lxo
        return lp

    def grad_log_posterior(x: np.ndarray) -> np.ndarray:
        """The gradient of the log-posterior of the stochastic volatility model given
        values of `phi`, `sigma`, and `beta` with respect to the stochastic
        volatilities.

        Args:
            x: The stochastic volatilities.

        Returns:
            out: The gradient of the log-posterior of the stochastic volatility
                model.

        """
        s = (np.square(y / beta) * np.exp(-x) - 1.0) / 2.0
        do = (x[0] - phi * x[1]) / sigmasq
        dn = (x[-1] - phi * x[-2]) / sigmasq
        w = (x[1:-1] - phi * x[:-2]) / sigmasq - phi * (x[2:] - phi * x[1:-1]) / sigmasq
        r = np.hstack((do, w, dn))
        return s - r

    def _metric() -> np.ndarray:
        """The Fisher information metric for the stochastic volatility model given
        values of `phi`, `sigma`, and `beta`.

        Returns:
            G: The Fisher information metric.

        """
        T = y.size
        Id = np.eye(T)
        n = np.arange(T) + 1
        C = np.power(phi, np.abs(n - n[..., np.newaxis])) * sigmasq / (1 - phisq)
        a = np.diag(-phi / sigmasq * np.ones((T-1)), 1)
        b = np.diag(-phi / sigmasq * np.ones((T-1)), -1)
        fl = 1. / sigmasq
        m = (1.0 + phisq) / sigmasq * np.ones(T-2)
        c = np.diag(np.hstack([fl, m, fl]))
        iC = a + b + c
        G = 0.5 * Id + iC
        # One can check the calculation of the inverse of the AR(1) covariance
        # using:
        # assert np.allclose(iC, np.linalg.inv(C))
        return G

    # Precompute the metric.
    G = _metric()

    def metric() -> np.ndarray:
        return G

    return log_posterior, grad_log_posterior, metric
