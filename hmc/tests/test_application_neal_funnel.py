import unittest

import numpy as np

from hmc.applications.neal_funnel import posterior_factory, sample
from hmc.applications.neal_funnel import hess_log_density, grad_hess_log_density
from hmc.linalg import solve_psd


class TestNealFunnel(unittest.TestCase):
    def test_neal_funnel(self):
        num_dims = 3
        alpha = 1e6
        log_density, grad_log_density, metric, grad_log_det, grad_quadratic_form = posterior_factory(alpha)

        x, v = sample(num_dims)
        q = np.hstack((x, v))
        delta = 1e-5
        u = np.random.normal(size=q.shape)
        fd = (log_density(q + 0.5*delta*u) - log_density(q - 0.5*delta*u)) / delta
        g = grad_log_density(q)
        self.assertTrue(np.allclose(fd, g@u))
        H = hess_log_density(q)
        fd = (grad_log_density(q + 0.5*delta*u) - grad_log_density(q - 0.5*delta*u)) / delta
        self.assertTrue(np.allclose(fd, H@u))

        G = metric(q)
        dt, U = np.linalg.eigh(G)
        self.assertTrue(np.all(dt > 0))

        def sample_momentum(q: np.ndarray) -> np.ndarray:
            return np.random.multivariate_normal(np.zeros_like(q), metric(q))

        p = sample_momentum(q)
        dH = grad_hess_log_density(q)
        fd = (hess_log_density(q + 0.5*delta*u) - hess_log_density(q - 0.5*delta*u)) / delta
        self.assertTrue(np.allclose(dH@u, fd))

        def log_abs_det(q: np.ndarray) -> float:
            G = metric(q)
            return np.linalg.slogdet(G)[1]

        def quadratic_form(q: np.ndarray, p: np.ndarray) -> float:
            G = metric(q)
            return p@solve_psd(G, p)

        fd = (log_abs_det(q + 0.5*delta*u) - log_abs_det(q - 0.5*delta*u)) / delta
        self.assertTrue(np.allclose(fd, grad_log_det(q)@u))
        fd = (quadratic_form(q + 0.5*delta*u, p) - quadratic_form(q - 0.5*delta*u, p)) / delta
        self.assertTrue(np.allclose(fd, grad_quadratic_form(q, p)@u))
