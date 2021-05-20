import unittest

import numpy as np
import scipy.stats as spst

from hmc.applications.gaussian import posterior_factory


class TestGaussian(unittest.TestCase):
    def test_gaussian(self):
        n = 10
        mu = np.random.normal(size=(n, ))
        Sigma = np.random.normal(size=(n, n))
        Sigma = Sigma.T@Sigma
        x = np.random.normal(size=(n, ))
        log_posterior, grad_log_posterior, metric = posterior_factory(mu, Sigma)
        lp = log_posterior(x)
        splp = spst.multivariate_normal.logpdf(x, mu, Sigma)
        self.assertTrue(np.allclose(lp, splp))
        u = np.random.normal(size=(n, ))
        delta = 1e-5
        fd = (log_posterior(x + 0.5*delta*u) - log_posterior(x - 0.5*delta*u)) / delta
        g = grad_log_posterior(x)@u
        self.assertTrue(np.allclose(fd, g))

