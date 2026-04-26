"""gamdist: Generalized Additive Models fit via ADMM."""

from __future__ import annotations

from .feature import _Feature
from .gamdist import GAM, fit_adaptive_lasso
from .spline_feature import SplineError

__all__ = ["GAM", "SplineError", "_Feature", "fit_adaptive_lasso"]
__version__ = "0.2.0"
