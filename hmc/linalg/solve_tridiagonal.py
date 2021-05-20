import numpy as np
import scipy.linalg as spla


def solve_tridiagonal(tri: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """The special structure of a tridiagonal matrix permits it to be used in
    solving a linear system in linear time instead of the usual cubic time.

    Args:
        tri: Tridiagonal matrix.
        rhs: Right-hand side of the linear system.

    Returns:
        out: The solution of the linear system involving a tridiagonal matrix.

    """
    ab = np.array([
        np.hstack((0.0, np.diag(tri, 1))),
        np.diag(tri, 0)
    ])
    return spla.solveh_banded(ab, rhs)
