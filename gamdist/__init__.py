"""gamdist: Generalized Additive Models fit via ADMM.

The package fits the GLM/GAM zoo -- binary, continuous, or count
outcomes paired with continuous, categorical, or spline-transformed
features, with arbitrary convex regularization on each term -- by
splitting the joint convex problem into per-feature primal steps and a
per-outcome proximal step coordinated by ADMM dual variables. The
modular decomposition follows Chu, Keshavarz, & Boyd's *A Distributed
Algorithm for Fitting Generalized Additive Models*.

Public API:

* :class:`GAM` -- single-response generalized additive model.
* :class:`MultiTaskGAM` -- joint fit of K tasks with optional
  cross-task coupling regularizers.
* :class:`SplineError` -- raised when spline knot selection or basis
  evaluation fails.
* :func:`fit_adaptive_lasso` -- two-stage adaptive-lasso wrapper
  around :meth:`GAM.fit`.
"""

from __future__ import annotations

from .feature import _Feature
from .gamdist import GAM, fit_adaptive_lasso
from .multi_task_gamdist import MultiTaskGAM
from .spline_feature import SplineError

__all__ = [
    "GAM",
    "MultiTaskGAM",
    "SplineError",
    "_Feature",
    "fit_adaptive_lasso",
]
__version__ = "0.2.0"
