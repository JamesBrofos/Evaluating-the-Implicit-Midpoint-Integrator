from typing import Callable, Tuple

import numpy as np
import scipy.stats as spst


def sample(num_dims: int) -> Tuple:
    """Sample from Neal's funnel distribution.

    Args:
        num_dims: Number of dimensions, besides the global variance, in Neal's
            funnel distribution.

    Returns:
        x: Conditionally-independent samples from Neal's funnel.
        v: Variance of Neal's funnel.

    """
    v = np.random.normal(0.0, 3.0)
    s = np.exp(-0.5*v)
    x = np.random.normal(0.0, s, size=(num_dims, ))
    return x, v

def _log_density(x: np.ndarray, v: float) -> float:
    """Log-density of Neal's funnel distribution.

    Args:
        x: Conditionally-independent samples from Neal's funnel.
        v: Variance of Neal's funnel.

    Returns:
        out: The log-density of Neal's funnel.

    """
    ldv = spst.norm.logpdf(v, 0.0, 3.0)
    s = np.exp(-0.5*v)
    ldx = spst.norm.logpdf(x, 0.0, s).sum()
    return ldv + ldx

def _grad_log_density(x: np.ndarray, v: float) -> np.ndarray:
    """Gradient of Neal's funnel distribution.

    Args:
        x: Conditionally-independent samples from Neal's funnel.
        v: Variance of Neal's funnel.

    Returns:
        out: The gradient of the log-density of Neal's funnel.

    """
    num_dims = x.size
    s = np.exp(-0.5*v)
    ssq = np.square(s)
    return np.hstack([-x / ssq, -v/9.0 - 0.5 * np.square(x).sum() / ssq + 0.5 * num_dims])

def _hess_log_density(x: np.ndarray, v: float) -> np.ndarray:
    """Hessian of Neal's funnel distribution.

    Args:
        x: Conditionally-independent samples from Neal's funnel.
        v: Variance of Neal's funnel.

    Returns:
        out: The Hessian of the log-density of Neal's funnel.

    """
    num_dims = x.size
    s = np.exp(-0.5*v)
    ssq = np.square(s)
    dvdv = -1.0/9.0 - 0.5 * np.square(x).sum() / ssq
    dvdx = -x / ssq
    dxdx = -np.eye(num_dims) / ssq
    H = np.vstack((
        np.hstack((dxdx, dvdx[..., np.newaxis])),
        np.hstack((dvdx, dvdv))
    ))
    return H

def _grad_hess_log_density(x: np.ndarray, v: float) -> np.ndarray:
    """Gradient of the Hessian of Neal's funnel distribution.

    Args:
        x: Conditionally-independent samples from Neal's funnel.
        v: Variance of Neal's funnel.

    Returns:
        out: The higher-order derivatives of the log-density of Neal's funnel.

    """
    num_dims = x.size
    s = np.exp(-0.5*v)
    ssq = np.square(s)
    dvdvdv = -0.5 * np.square(x).sum() / ssq
    dxdxdv = -np.eye(num_dims) / ssq
    dvdvdx = -x / ssq

    Z = np.zeros((num_dims, num_dims, num_dims))
    da = np.concatenate((Z, dxdxdv[..., np.newaxis]), axis=-1)
    mm = np.hstack((dxdxdv, dvdvdx[..., np.newaxis]))
    rr = np.vstack((mm,
                    np.hstack((dvdvdx, dvdvdv))))
    db = np.concatenate((da, mm[:, np.newaxis]), axis=1)
    dH = np.concatenate((db, rr[np.newaxis]))
    return dH

coth = lambda z: np.reciprocal(np.tanh(z))
csch = lambda z: np.reciprocal(np.sinh(z))
cschsq = lambda z: np.square(csch(z))
softplus = lambda l, alpha: l * coth(alpha * l)
softplus_deriv = lambda l, alpha: coth(alpha * l) - alpha * l * cschsq(alpha * l)

def _metric(x: np.ndarray, v: float, alpha: float) -> np.ndarray:
    """The Riemannian metric is the soft absolute value of the Hessian matrix.

    Args:
        x: Conditionally-independent samples from Neal's funnel.
        v: Variance of Neal's funnel.
        alpha: Parameter controlling the sharpness of the softabs function.

    Returns:
        G: The softabs metric.

    """
    H = _hess_log_density(x, v)
    l, u = np.linalg.eigh(H)
    lt = softplus(l, alpha)
    G = (u*lt)@u.T
    return G

def _j_matrix(l: np.ndarray, lt: np.ndarray, alpha: float) -> np.ndarray:
    """Helper function to compute the `J` matrix in the derivation of the softabs
    metric. Some care must be taken to appropriately handle repeated eigenvalues.

    See [1] for treatment of the softabs metric.

    [1] https://arxiv.org/abs/1212.4693

    Args:
        l: The original eigenvalues.
        lt: The eigenvalues transformed by the softabs function.
        alpha: Parameter controlling the sharpness of the softabs function.

    Returns:
        J: The auxiliary `J` matrix derived in the softabs metric.

    """
    j_den_p = l - l[..., np.newaxis]
    deg = np.abs(j_den_p) < 1e-10
    j_den_p = np.where(deg, np.ones_like(j_den_p), j_den_p)
    deriv = softplus_deriv(l, alpha)
    deriv = np.tile(deriv, (l.size, 1))
    j_num_p = lt - lt[..., np.newaxis]
    j_num_p = np.where(deg, deriv, j_num_p)
    J = j_num_p / j_den_p
    return J

# Helper functions.
hess_log_density = lambda q: _hess_log_density(q[:-1], q[-1])
grad_hess_log_density = lambda q: _grad_hess_log_density(q[:-1], q[-1])

def posterior_factory(alpha: float=1e6) -> Tuple[Callable]:
    """Posterior factory function for Neal's funnel distribution. This is a density
    that exhibits extreme variation in the dimensions and may therefore present
    a challenge for leapfrog integrators. Therefore, the posterior is also
    equipped with the softabs metric which adapts the generalized leapfrog
    integrator to the local geometry. The softabs metric is a transformation of
    the Hessian to make it positive definite.

    It is a curious attribute of this posterior that for a larger size of the
    posterior, larger step-sizes are better behaved that for a smaller size of
    the posterior.

    Args:
        alpha: Parameter controlling the sharpness of the softabs function.

    Returns:
        log_posterior: Function to compute the log-posterior.
        grad_log_posterior: Function to compute the gradient of the log-posterior.
        metric: Function to compute the softabs metric.
        grad_log_det: Gradient of the logarithm of the determinant of the metric.
        grad_quadratic_form: Gradient of the quadratic kinetic energy.

    """
    def grad_log_det(q: np.ndarray) -> np.ndarray:
        """The gradient of the logarithm of the determinant of the metric, which is the
        softabs of the Hessian.

        Args:
            q: The position variable.

        Returns:
            out: The gradient of the logarithm of the metric.

        """
        dH = grad_hess_log_density(q)
        l, U = np.linalg.eigh(hess_log_density(q))
        lt = softplus(l, alpha)
        R = np.diag(np.reciprocal(lt))
        J = _j_matrix(l, lt, alpha)
        return np.trace(U@(R*J)@U.T@dH, axis1=1, axis2=2)

    def grad_quadratic_form(q: np.ndarray, p: np.ndarray) -> np.ndarray:
        """The gradient of the quadratic kinetic energy with respect to the position
        variable wherein the Mahalanobis distance is defined by the inverse
        metric, which is the softabs of the Hessian.

        Args:
            q: The position variable.
            p: The momentum variable.

        Returns:
            out: The gradient of the quadratic potential energy.

        """
        dH = grad_hess_log_density(q)
        l, U = np.linalg.eigh(hess_log_density(q))
        lt = softplus(l, alpha)
        D = np.diag(U.T@p / lt)
        J = _j_matrix(l, lt, alpha)
        return -np.trace(U@D@J@D@U.T@dH, axis1=1, axis2=2)

    # Create functions that take a single parameter representing the
    # concatenation of all the variables in Neal's funnel distribution.
    log_density = lambda q: _log_density(q[:-1], q[-1])
    grad_log_density = lambda q: _grad_log_density(q[:-1], q[-1])
    metric = lambda q: _metric(q[:-1], q[-1], alpha)

    return log_density, grad_log_density, metric, grad_log_det, grad_quadratic_form
