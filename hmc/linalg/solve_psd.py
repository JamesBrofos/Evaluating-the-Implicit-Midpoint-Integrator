import numpy as np
import scipy.linalg as spla


def solve_psd(A: np.ndarray, b: np.ndarray):
    """Solve the system `A x = b` under the assumption that `A` is positive
    definite. The method implemented is to compute the Cholesky factorization
    of `A` and solve the system via forward-backward substitution.

    Args:
        A: Left-hand side of linear system.
        b: Right-hand side of the linear system.

    Returns:
        out: Solution of the linear system.

    """
    L = spla.cholesky(A)
    return spla.cho_solve((L, False), b)
