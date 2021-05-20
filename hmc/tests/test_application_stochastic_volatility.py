import unittest

import numpy as np
import scipy.stats as spst

from hmc.applications.stochastic_volatility import (
    generate_data, latent_posterior_factory, volatility_posterior_factory,
    forward_transform, inverse_transform,
    grad_beta_logpdf, hess_beta_logpdf, grad_hess_beta_logpdf,
    inv_chisq_logpdf, grad_inv_chisq_logpdf, hess_inv_chisq_logpdf,
    grad_hess_inv_chisq_logpdf,
    grad_log_prior, hess_log_prior, grad_hess_log_prior)
from hmc.linalg import solve_tridiagonal

class TestStochasticVolatility(unittest.TestCase):
    def test_stochastic_volatility(self):
        # Generate data from the stochastic volatility model.
        sigma = 0.15
        phi = 0.98
        beta = 0.65
        T = 100
        x, y = generate_data(T, sigma, phi, beta)

        # Check the gradient of the log-posterior when `phi`, `sigma`, and
        # `beta` are fixed.
        log_posterior, grad_log_posterior, metric = volatility_posterior_factory(sigma, phi, beta, y)
        delta = 1e-6
        u = np.random.normal(size=x.shape)
        fd = (log_posterior(x + 0.5*delta*u) - log_posterior(x - 0.5*delta*u)) / delta
        dd = grad_log_posterior(x)@u
        self.assertTrue(np.allclose(fd, dd))
        G = metric()
        rhs = np.random.normal(size=x.shape)
        sol = solve_tridiagonal(G, rhs)
        self.assertTrue(np.allclose(G@sol, rhs))

        # Check the gradient of the log-posterior and the metric when `x` is fixed.
        log_posterior, grad_log_posterior, metric, grad_metric, _ = latent_posterior_factory(x, y)
        gamma = np.log(sigma)
        alpha = np.arctanh(phi)
        qt = np.array([gamma, alpha, beta])
        u = np.random.normal(size=qt.shape)
        fd = (log_posterior(qt + 0.5*delta*u) - log_posterior(qt - 0.5*delta*u)) / delta
        g = grad_log_posterior(qt)@u
        self.assertTrue(np.allclose(fd, g))

        # Check the inverse chi-square gradients.
        sigmasq = np.square(sigma)
        delta = 1e-5
        fd = (
            inv_chisq_logpdf(sigmasq + 0.5*delta, 10.0, 0.05) -
            inv_chisq_logpdf(sigmasq - 0.5*delta, 10.0, 0.05)) / delta
        g = grad_inv_chisq_logpdf(sigmasq, 10.0, 0.05)
        self.assertTrue(np.allclose(fd, g))
        fd = (
            grad_inv_chisq_logpdf(sigmasq + 0.5*delta, 10.0, 0.05) -
            grad_inv_chisq_logpdf(sigmasq - 0.5*delta, 10.0, 0.05)) / delta
        g = hess_inv_chisq_logpdf(sigmasq, 10.0, 0.05)
        self.assertTrue(np.allclose(fd, g))
        fd = (
            hess_inv_chisq_logpdf(sigmasq + 0.5*delta, 10.0, 0.05) -
            hess_inv_chisq_logpdf(sigmasq - 0.5*delta, 10.0, 0.05)) / delta
        g = grad_hess_inv_chisq_logpdf(sigmasq, 10.0, 0.05)
        self.assertTrue(np.allclose(fd, g))

        # Check the gradients of the beta distribution.
        m = 0.5*(phi + 1.0)
        delta = 1e-5
        fd = (
            spst.beta.logpdf(m + 0.5*delta, 20.0, 1.5) -
            spst.beta.logpdf(m - 0.5*delta, 20.0, 1.5)) / delta
        g = grad_beta_logpdf(m, 20.0, 1.5)
        self.assertTrue(np.allclose(fd, g))
        fd = (
            grad_beta_logpdf(m + 0.5*delta, 20.0, 1.5) -
            grad_beta_logpdf(m - 0.5*delta, 20.0, 1.5)) / delta
        g = hess_beta_logpdf(m, 20.0, 1.5)
        self.assertTrue(np.allclose(fd, g))
        fd = (
            hess_beta_logpdf(m + 0.5*delta, 20.0, 1.5) -
            hess_beta_logpdf(m - 0.5*delta, 20.0, 1.5)) / delta
        g = grad_hess_beta_logpdf(m, 20.0, 1.5)
        self.assertTrue(np.allclose(fd, g))

        u = np.random.normal(size=qt.shape)
        dG = grad_metric(qt)
        fd = (metric(qt + 0.5*delta*u) -
              metric(qt - 0.5*delta*u)) / delta
        self.assertTrue(np.allclose(dG@u, fd))

        # Check hessian of the log-posterior.
        g = hess_log_prior(gamma, alpha, beta)
        delta = 1e-5
        fd = np.array([
            (np.array(grad_log_prior(gamma + 0.5*delta, alpha, beta)) -
             np.array(grad_log_prior(gamma - 0.5*delta, alpha, beta))) / delta,
            (np.array(grad_log_prior(gamma, alpha + 0.5*delta, beta)) -
             np.array(grad_log_prior(gamma, alpha - 0.5*delta, beta))) / delta,
            (np.array(grad_log_prior(gamma, alpha, beta + 0.5*delta)) -
             np.array(grad_log_prior(gamma, alpha, beta - 0.5*delta))) / delta])
        self.assertTrue(np.allclose(fd, g))
        # Check tensor of higher-order derivatives of the prior.
        delta = 1e-5
        fd = np.array([
            (hess_log_prior(gamma + 0.5*delta, alpha, beta) -
             hess_log_prior(gamma - 0.5*delta, alpha, beta)) / delta,
            (hess_log_prior(gamma, alpha + 0.5*delta, beta) -
             hess_log_prior(gamma, alpha - 0.5*delta, beta)) / delta,
            (hess_log_prior(gamma, alpha, beta + 0.5*delta) -
             hess_log_prior(gamma, alpha, beta - 0.5*delta)) / delta])
        dH = grad_hess_log_prior(gamma, alpha, beta)
        self.assertTrue(np.allclose(fd, dH))

        # Verify that the logarithm of the Jacobian determinant is correct.
        qo = np.array([sigma, phi, beta])
        qt, ildj = forward_transform(qo)
        iqt, fldj = inverse_transform(qt)
        self.assertTrue(np.allclose(iqt, qo))
        delta = 1e-5
        J = np.array([
            inverse_transform(qt + 0.5*delta*np.array([1.0, 0.0, 0.0]))[0] - inverse_transform(qt - 0.5*delta*np.array([1.0, 0.0, 0.0]))[0],
            inverse_transform(qt + 0.5*delta*np.array([0.0, 1.0, 0.0]))[0] - inverse_transform(qt - 0.5*delta*np.array([0.0, 1.0, 0.0]))[0],
            inverse_transform(qt + 0.5*delta*np.array([0.0, 0.0, 1.0]))[0] - inverse_transform(qt - 0.5*delta*np.array([0.0, 0.0, 1.0]))[0]
        ]) / delta
        self.assertTrue(np.allclose(np.log(np.linalg.det(J)), ildj))
        J = np.array([
            forward_transform(qo + 0.5*delta*np.array([1.0, 0.0, 0.0]))[0] - forward_transform(qo - 0.5*delta*np.array([1.0, 0.0, 0.0]))[0],
            forward_transform(qo + 0.5*delta*np.array([0.0, 1.0, 0.0]))[0] - forward_transform(qo - 0.5*delta*np.array([0.0, 1.0, 0.0]))[0],
            forward_transform(qo + 0.5*delta*np.array([0.0, 0.0, 1.0]))[0] - forward_transform(qo - 0.5*delta*np.array([0.0, 0.0, 1.0]))[0]
        ]) / delta
        self.assertTrue(np.allclose(np.log(np.linalg.det(J)), fldj))
