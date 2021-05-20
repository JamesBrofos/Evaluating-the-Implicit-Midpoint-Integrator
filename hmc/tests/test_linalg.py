import unittest

import numpy as np
import scipy.linalg as spla


class TestLinalg(unittest.TestCase):
    def test_solveh_banded(self):
        import numpy as onp
        import scipy.linalg as ospla

        num_dims = 15
        b = onp.exp(onp.random.normal(size=(num_dims, )))
        a = 0.1 * onp.random.normal(size=(num_dims-1, ))
        tri = onp.diag(b)
        tri += onp.diag(a, 1)
        tri += onp.diag(a, -1)
        rhs = onp.random.normal(size=(num_dims, ))

        ab = onp.array([
            onp.hstack((0.0, onp.diag(tri, 1))),
            onp.diag(tri)
        ])
        x = ospla.solveh_banded(ab, rhs)
        self.assertTrue(onp.allclose(tri@x, rhs))

        sol = spla.solveh_banded(ab, rhs)
        self.assertTrue(np.allclose(sol, x))
        self.assertTrue(np.allclose(tri@sol, rhs))
