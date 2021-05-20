import unittest

import numpy as np

from hmc.applications.logistic_regression import posterior_factory, sigmoid


class TestLogisticRegression(unittest.TestCase):
    def test_logistic_regression(self):
        # Generate logistic regression data.
        num_obs, num_dims = 100, 5
        x = np.random.normal(size=(num_obs, num_dims))
        b = np.ones((x.shape[-1], ))
        p = sigmoid(x@b)
        y = np.random.binomial(1, p)
        alpha = 0.5

        # Check the gradients of the posterior using finite differences.
        (
            log_posterior, grad_log_posterior, hess_log_posterior,
            metric, grad_metric, hess_metric, _
        ) = posterior_factory(x, y, alpha)
        delta = 1e-6
        u = np.random.normal(size=b.shape)
        fd = (log_posterior(b + 0.5*delta*u) - log_posterior(b - 0.5*delta*u)) / delta
        dd = grad_log_posterior(b)@u
        self.assertTrue(np.allclose(fd, dd))
        fd = (grad_log_posterior(b + 0.5*delta*u) - grad_log_posterior(b - 0.5*delta*u)) / delta
        dd = hess_log_posterior(b)@u
        self.assertTrue(np.allclose(fd, dd))

        # Check the gradient of the metric using finite differences.
        fd = (metric(b + 0.5*delta*u) - metric(b - 0.5*delta*u)) / delta
        dG = grad_metric(b)
        self.assertTrue(np.allclose(fd, dG@u))
        fd = (grad_metric(b + 0.5*delta*u) - grad_metric(b - 0.5*delta*u)) / delta
        ddG = hess_metric(b)
        self.assertTrue(np.allclose(fd, ddG@u))
