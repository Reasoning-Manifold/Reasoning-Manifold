import logging

import numpy as np

logger = logging.getLogger(__name__)

EPSILON_DEFAULT: float = 0.1
K_NEIGHBOURS_DEFAULT: int = 20


def intrinsic_dimension(
    X: np.ndarray,
    *,
    method: str = "Covariance",
    k: int = K_NEIGHBOURS_DEFAULT,
) -> float:
    import perceptual_manifold_geometry as pmg

    try:
        return float(pmg.estimate_intrinsic_dimension(X, method=method, k=k))
    except TypeError:
        return float(pmg.estimate_intrinsic_dimension(X, method=method))
    except Exception as exc:
        logger.error("intrinsic_dimension failed: %s", exc)
        return float("nan")


def information_volume(Z: np.ndarray, *, d_scale: float = 1.0) -> float:
    # V = ½ log_2 det( I + (d/m) Zᵀ Z )  on the centred (m × n) matrix
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
    if interval <= 0:
        raise ValueError("interval must be positive")
    return np.asarray(states)[offset::interval]
