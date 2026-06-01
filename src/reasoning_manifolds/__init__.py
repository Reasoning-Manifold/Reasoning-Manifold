"""Reasoning Manifolds — code for arXiv:2605.08142.

Public API:
    intrinsic_dimension(X, k=20)
    information_volume(Z)
    reasoning_health(D_world, D_stim, V, epsilon=0.1)
"""

from reasoning_manifolds.metrics import (
    information_volume,
    intrinsic_dimension,
    reasoning_health,
)

__all__ = [
    "intrinsic_dimension",
    "information_volume",
    "reasoning_health",
]

__version__ = "0.1.0"
