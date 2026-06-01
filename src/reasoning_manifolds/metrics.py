"""Geometric and information-theoretic measures of reasoning dynamics.

All quantities are defined exactly as in the paper §Methods (arXiv:2605.08142).

Key definitions
---------------
* ``intrinsic_dimension``  — TLE estimator with k=20 neighbours (Eq. 6).
* ``information_volume``   — V_l(x) = ½ log det(I + (d/T) Z Zᵀ) (Eq. 14).
* ``reasoning_health``     — H = log(D_world) · V / exp(ε·D_stim) (Eq. 15).

The TLE estimator is provided by the ``perceptual-manifold-geometry`` package.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

EPSILON_DEFAULT: float = 0.1
"""Penalty coefficient ε in Eq. 15. Fixed a priori across all models."""

K_NEIGHBOURS_DEFAULT: int = 20
"""Number of neighbours for the TLE intrinsic-dimension estimator."""


def intrinsic_dimension(
    X: np.ndarray,
    *,
    method: str = "Covariance",
    k: int = K_NEIGHBOURS_DEFAULT,
) -> float:
    """Estimate the intrinsic dimension of a point cloud.

    Wraps :func:`perceptual_manifold_geometry.estimate_intrinsic_dimension`.
    The paper uses the TLE estimator (``method='Covariance'``) with k=20
    neighbours and Euclidean distances throughout.

    Args:
        X: shape ``(m, d)`` array of points.
        method: estimator name passed through to ``pmg``. Default matches the
            paper.
        k: neighbour count. Default matches the paper.

    Returns:
        Estimated intrinsic dimension. ``nan`` on numerical failure.
    """
    import perceptual_manifold_geometry as pmg

    try:
        return float(pmg.estimate_intrinsic_dimension(X, method=method, k=k))
    except TypeError:
        return float(pmg.estimate_intrinsic_dimension(X, method=method))
    except Exception as exc:
        logger.error("intrinsic_dimension failed: %s", exc)
        return float("nan")


def information_volume(Z: np.ndarray, *, d_scale: float = 1.0) -> float:
    """Information volume of a centred trajectory matrix (Eq. 14).

    For a stimulus *x* and layer ℓ the paper defines

    .. math::
        V_\\ell(x) = \\tfrac{1}{2} \\log \\det\\bigl( I + (d_\\ell / T(x))\\,Z_\\ell(x) Z_\\ell(x)^\\top \\bigr).

    Internally we use the equivalent form

    .. math::
        V = \\tfrac{1}{2} \\log_2 \\det\\bigl( I + (d/m)\\,Z^\\top Z \\bigr)

    on the (m × n) centred matrix ``Z`` (rows = tokens, columns = features).

    Args:
        Z: centred trajectory matrix of shape ``(m, n)``.
        d_scale: scale ``d`` in the paper. Defaults to 1.

    Returns:
        Volume in bits. ``inf`` if the determinant is non-positive.
    """
    Z = np.asarray(Z, dtype=np.float64)
    m, n = Z.shape

    centred = Z - Z.mean(axis=0, keepdims=True)
    matrix = np.eye(n) + (d_scale / m) * (centred.T @ centred)

    sign, logdet = np.linalg.slogdet(matrix)
    if sign <= 0:
        logger.warning("information_volume: non-positive determinant (sign=%s)", sign)
        return float("inf")

    return float(0.5 * logdet / np.log(2))


def reasoning_health(
    D_world: float,
    D_stim: float,
    V: float,
    *,
    epsilon: float = EPSILON_DEFAULT,
) -> float:
    """Unified label-free reasoning-health diagnostic ℋ (Eq. 15).

    .. math::
        \\mathcal{H} = \\log(D_\\text{world}) \\cdot \\frac{V}{\\exp(\\varepsilon\\,D_\\text{stim})}.

    Args:
        D_world: intrinsic dimension of the static vocabulary embedding matrix
            (representational expressivity).
        D_stim: stimulus-induced intrinsic dimension of the inference
            trajectory at the final layer (geometric organisation).
        V: information volume of the reasoning manifold at the final layer.
        epsilon: penalty coefficient. Fixed at 0.1 in the paper.

    Returns:
        ℋ. Returns ``nan`` when ``D_world`` is non-positive (log undefined).
    """
    if D_world <= 0:
        return float("nan")
    if not np.isfinite(V):
        V = 0.0
    return float(np.log(D_world) * V / np.exp(epsilon * D_stim))


def reasoning_health_components(
    D_world: float,
    D_stim: float,
    V: float,
    *,
    epsilon: float = EPSILON_DEFAULT,
) -> dict:
    """Return ℋ alongside its three structural inputs."""
    return {
        "D_world": float(D_world),
        "D_stim": float(D_stim),
        "V": float(V),
        "epsilon": float(epsilon),
        "H": reasoning_health(D_world, D_stim, V, epsilon=epsilon),
    }


def sample_trajectory(
    states: np.ndarray,
    *,
    interval: int = 1,
    offset: int = 0,
) -> np.ndarray:
    """Stride-sample tokens along a trajectory.

    The paper computes ID and V from every generated token at the final layer.
    Earlier exploratory code in this repository sub-sampled with a stride; we
    keep the helper for backward compatibility but default to no sub-sampling.
    """
    if interval <= 0:
        raise ValueError("interval must be positive")
    return np.asarray(states)[offset::interval]
