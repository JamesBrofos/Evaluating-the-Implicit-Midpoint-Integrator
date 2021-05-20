import numpy as np
import scipy.linalg as spla


def cholesky_tridiagonal(tri: np.ndarray) -> np.ndarray:
    """The special structure of a tridiagonal matrix permits its Cholesky factor to
    be computed in linear time instead of cubic time.

    Args:
        tri: Tridiagonal matrix.

    Returns:
        C: The Cholesky factorization of the tridiagonal matrix.

    """

    ab = np.array([
        np.hstack((0.0, np.diag(tri, 1))),
        np.diag(tri, 0)
    ])
    c = spla.cholesky_banded(ab)
    C = np.diag(c[0, 1:], k=-1) + np.diag(c[1, :])
    return C
