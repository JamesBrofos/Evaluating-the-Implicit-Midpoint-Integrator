import unittest

import numpy as np

import hmc


class TestGeneralizedLeapfrog(unittest.TestCase):
    def test_generalized_leapfrog(self):
        distr = hmc.applications.banana
        t = 0.5
        sigma_theta = 2.
        sigma_y = 2.
        theta, y = distr.generate_data(t, sigma_y, sigma_theta, 100)
        (
            log_posterior, grad_log_posterior, _,
            metric, grad_metric, _,
            grad_log_posterior_and_metric_and_grad_metric
        ) = distr.posterior_factory(y, sigma_y, sigma_theta)
        (
            hamiltonian, grad_pos_hamiltonian, grad_mom_hamiltonian,
            _, sample_momentum
        ) = hmc.integrators.riemannian_hamiltonian_vector_field(
            log_posterior, grad_log_posterior, metric, grad_metric,
            grad_log_posterior_and_metric_and_grad_metric)

        q = theta
        p = sample_momentum(q)
        step_size = 1e-2
        num_steps = 10
        generalized_leapfrog = hmc.integrators.generalized_leapfrog
        smart_generalized_leapfrog = hmc.integrators.smart_generalized_leapfrog
        (qn, pn), _ = smart_generalized_leapfrog(
            grad_log_posterior, metric, grad_metric,
            grad_log_posterior_and_metric_and_grad_metric,
            (q, p), step_size, num_steps)
        (qm, pm), _ = generalized_leapfrog(
            grad_pos_hamiltonian, grad_mom_hamiltonian,
            (q, p), step_size, num_steps)
        self.assertTrue(np.allclose(qn, qm))
        self.assertTrue(np.allclose(pn, pm))
