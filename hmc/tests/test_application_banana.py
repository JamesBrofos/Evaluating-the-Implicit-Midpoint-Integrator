import unittest

import numpy as np

from hmc.applications.banana import generate_data, posterior_factory


class TestBanana(unittest.TestCase):
    def test_banana(self):
        # Generate data.
        t = 0.5
        sigma_theta = 2.
        sigma_y = 2.
        theta, y = generate_data(t, sigma_y, sigma_theta, 100)

        # Check the gradients of the posterior using finite differences.
        (
            log_posterior, grad_log_posterior, hess_log_posterior,
            metric, grad_metric, hess_metric, _
        ) = posterior_factory(y, sigma_y, sigma_theta)
        delta = 1e-6
        u = np.random.normal(size=(2, ))
        fd = (log_posterior(theta + 0.5*delta*u) - log_posterior(theta - 0.5*delta*u)) / delta
        dd = grad_log_posterior(theta)@u
        self.assertTrue(np.allclose(fd, dd))
        fd = (grad_log_posterior(theta + 0.5*delta*u) - grad_log_posterior(theta - 0.5*delta*u)) / delta
        dd = hess_log_posterior(theta)@u
        self.assertTrue(np.allclose(fd, dd))

        # Check the gradient of the metric using finite differences.
        fd = (metric(theta + 0.5*delta*u) - metric(theta - 0.5*delta*u)) / delta
        dG = grad_metric(theta)
        self.assertTrue(np.allclose(fd, dG@u))
        u = np.array([1.0, 0.0])
        fd = (metric(theta + 0.5*delta*u) - metric(theta - 0.5*delta*u)) / delta
        self.assertTrue(np.allclose(fd, dG[..., 0]))
        u = np.array([0.0, 1.0])
        fd = (metric(theta + 0.5*delta*u) - metric(theta - 0.5*delta*u)) / delta
        self.assertTrue(np.allclose(fd, dG[..., 1]))

        fd = (metric(theta + 0.5*delta*u) - metric(theta - 0.5*delta*u)) / delta
        dd = grad_metric(theta)@u
        self.assertTrue(np.allclose(fd, dd))
        fd = (grad_metric(theta + 0.5*delta*u) - grad_metric(theta - 0.5*delta*u)) / delta
        dd = hess_metric(theta)@u
        self.assertTrue(np.allclose(fd, dd))
