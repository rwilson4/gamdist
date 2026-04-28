# Copyright 2017 Match Group, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License"); you
# may not use this file except in compliance with the License. You may
# obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.  See the License for the specific language governing
# permissions and limitations under the License.
#
# Passing untrusted user input may have unintended consequences. Not
# designed to consume input from unknown sources (i.e., the public
# internet).
#
# This file has been modified from the original release by Match Group
# LLC. A description of changes may be found in the change log
# accompanying this source code.

"""Linear feature: a single slope with closed-form regularized solves.

:class:`_LinearFeature` contributes ``m * (x - mean(x))`` to the linear
predictor. The per-feature primal step has a closed-form solution at
every supported regularization combination -- ridge folds into the
denominator, L1 / group lasso / L_inf group lasso into a 1-D
soft-threshold, Huber into a clipped quadratic -- so this branch never
calls cvxpy. Sign / lower / upper convex constraints clip the
unconstrained optimum onto the feasible interval.
"""

from __future__ import annotations

import math
import pickle
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from .feature import _Feature

FloatArray = npt.NDArray[np.float64]
Transform = Callable[[npt.NDArray[Any]], npt.NDArray[Any]]


class _LinearFeature(_Feature):
    """A linear feature: contributes ``m * (x - mean(x))`` to the predictor."""

    def __init__(
        self,
        name: str | None = None,
        transform: Transform | None = None,
        regularization: dict[str, Any] | None = None,
        constraints: dict[str, Any] | None = None,
        load_from_file: str | None = None,
    ) -> None:
        r"""Initialize feature model (independent of data).

        Parameters
        ----------
        name : str
            Name for the feature, used to make plots.
        transform : callable, optional
            Transformation applied to the data (e.g.
            :func:`numpy.log1p`).
        regularization : dict, optional
            Description of regularization terms. Supported keys:

            * ``l1`` -- ``{"coef": lam}``; adds :math:`\lambda |m|`
              to the objective, shrinking the slope toward zero via
              a closed-form 1-D soft-threshold.
            * ``l2`` (ridge) -- ``{"coef": lam}``; adds
              :math:`\lambda m^2` to the objective.
            * ``huber`` -- ``{"coef": lam, "delta": d}``; adds
              :math:`\lambda h_\delta(m)`, where :math:`h_\delta` is
              the standard Huber function
              (:math:`0.5 m^2` for :math:`|m| \le \delta`,
              :math:`\delta |m| - 0.5 \delta^2` otherwise). Behaves
              like ridge for small slopes and like L1 for large
              slopes, bounding the per-coefficient influence of the
              penalty. May be combined with ``l2`` (both smooth at
              the origin) but not with ``l1`` / ``group_lasso`` /
              ``group_lasso_inf``.
            * ``group_lasso`` -- ``{"coef": lam}``; adds

              .. math::

                 \lambda \|f_j\|_2 =
                 \lambda |m| \sqrt{x^\top x},

              which zeros out the entire feature once
              :math:`\lambda` is large enough.
            * ``group_lasso_inf`` -- ``{"coef": lam}``; adds

              .. math::

                 \lambda \|f_j\|_\infty =
                 \lambda |m| \max_i |x_i - \bar{x}|,

              the :math:`L_\infty`-norm variant of the group lasso.
              On a single-slope feature it has the same
              zero-the-slope effect as the :math:`L_2` group lasso,
              but the threshold scales with the data range rather
              than its :math:`L_2` norm.

            ``l1``, ``group_lasso``, and ``group_lasso_inf`` all
            combine additively in the soft-threshold and stack with
            ``l2`` (elastic net). The top-level ``prior`` key (if
            present) provides a scalar prior estimate for the slope.
        constraints : dict, optional
            Optional convex shape constraints on the slope. ``sign``
            takes ``"nonnegative"`` (:math:`m \ge 0`) or
            ``"nonpositive"`` (:math:`m \le 0`); ``lower`` and
            ``upper`` take floats and bound :math:`m` from below /
            above. ``sign`` is shorthand and may not be combined with
            ``lower`` / ``upper``. The single-slope linear feature
            does not support ``monotonic`` / ``convex`` / ``concave``
            (those are well-defined only on multi-coefficient
            features); passing them raises. Constraints clip the
            per-feature primal step's unconstrained closed-form
            solution -- the objective is convex in :math:`m`, so
            projection onto ``[lower, upper]`` is the constrained
            optimum. No cvxpy on this path.
        load_from_file : str, optional
            If provided, restore parameters from this pickle path.
            All other parameters are ignored when loading.
        """
        self.__type__ = "linear"
        if load_from_file is not None:
            self._load(load_from_file)
            return

        if name is None:
            raise ValueError("Feature must have a name.")

        super().__init__(name)

        self._has_transform: bool
        self._transform: Transform
        if transform is not None:
            self._has_transform = True
            self._transform = transform
        else:
            self._has_transform = False

        self._has_l1 = False
        self._has_l2 = False
        self._has_huber = False
        self._has_group_lasso = False
        self._has_group_lasso_inf = False
        self._has_prior = False
        self._coef1: float = 0.0
        self._coef2: float = 0.0
        self._lambda1: float = 0.0
        self._lambda2: float = 0.0
        self._coef_huber: float = 0.0
        self._lambda_huber: float = 0.0
        self._delta_huber: float = 0.0
        self._coef_group_lasso: float = 0.0
        self._lambda_group_lasso: float = 0.0
        self._coef_group_lasso_inf: float = 0.0
        self._lambda_group_lasso_inf: float = 0.0
        self._prior: float = 0.0

        if regularization is not None:
            if "l1" in regularization:
                self._has_l1 = True
                if "coef" in regularization["l1"]:
                    self._coef1 = float(regularization["l1"]["coef"])
                else:
                    raise ValueError(
                        "No coefficient specified for l1 regularization term."
                    )

            if "l2" in regularization:
                self._has_l2 = True
                if "coef" in regularization["l2"]:
                    self._coef2 = float(regularization["l2"]["coef"])
                else:
                    raise ValueError(
                        "No coefficient specified for l2 regularization term."
                    )

            if "group_lasso" in regularization:
                self._has_group_lasso = True
                if "coef" in regularization["group_lasso"]:
                    self._coef_group_lasso = float(
                        regularization["group_lasso"]["coef"]
                    )
                else:
                    raise ValueError(
                        "No coefficient specified for group_lasso regularization term."
                    )

            if "group_lasso_inf" in regularization:
                self._has_group_lasso_inf = True
                if "coef" in regularization["group_lasso_inf"]:
                    self._coef_group_lasso_inf = float(
                        regularization["group_lasso_inf"]["coef"]
                    )
                else:
                    raise ValueError(
                        "No coefficient specified for group_lasso_inf regularization term."
                    )

            if "huber" in regularization:
                self._has_huber = True
                if "coef" not in regularization["huber"]:
                    raise ValueError(
                        "No coefficient specified for huber regularization term."
                    )
                if "delta" not in regularization["huber"]:
                    raise ValueError(
                        "No delta specified for huber regularization term."
                    )
                self._coef_huber = float(regularization["huber"]["coef"])
                self._delta_huber = float(regularization["huber"]["delta"])
                if self._coef_huber < 0.0:
                    raise ValueError("huber coefficient must be non-negative.")
                if self._delta_huber <= 0.0:
                    raise ValueError("huber delta must be positive.")
                if self._has_l1 or self._has_group_lasso or self._has_group_lasso_inf:
                    raise ValueError(
                        "huber regularization on a linear feature cannot be "
                        "combined with l1, group_lasso, or group_lasso_inf."
                    )

            if self._has_l1 or self._has_l2:
                self._has_prior = True
                self._prior = float(regularization.get("prior", 0.0))

        self._lower: float = -np.inf
        self._upper: float = np.inf
        self._has_constraints = False
        if constraints is not None:
            self._has_constraints = True
            for key in constraints:
                if key not in {"sign", "lower", "upper"}:
                    if key in {"monotonic", "convex", "concave", "order"}:
                        raise ValueError(
                            f"linear feature does not support constraint {key!r}; "
                            "monotonicity / convexity require a multi-coefficient "
                            "feature (categorical or spline)."
                        )
                    raise ValueError(f"unknown constraint key {key!r}.")
            if "sign" in constraints and (
                "lower" in constraints or "upper" in constraints
            ):
                raise ValueError(
                    "constraints: 'sign' is shorthand for 'lower' / 'upper' and may "
                    "not be combined with them."
                )
            if "sign" in constraints:
                sign = constraints["sign"]
                if sign == "nonnegative":
                    self._lower = 0.0
                elif sign == "nonpositive":
                    self._upper = 0.0
                else:
                    raise ValueError(
                        f"constraints['sign']: expected 'nonnegative' or "
                        f"'nonpositive', got {sign!r}."
                    )
            if "lower" in constraints:
                self._lower = float(constraints["lower"])
            if "upper" in constraints:
                self._upper = float(constraints["upper"])
            if self._lower > self._upper:
                raise ValueError(
                    f"constraints: lower ({self._lower}) exceeds upper ({self._upper})."
                )

    def initialize(
        self,
        x: npt.NDArray[Any],
        smoothing: float = 1.0,
        save_flag: bool = False,
        save_prefix: str | None = None,
        verbose: bool = False,
        covariate_class_sizes: npt.NDArray[Any] | None = None,
    ) -> None:
        """Compute feature-specific state from data.

        Parameters
        ----------
        x : array
            Observations corresponding to this feature.
        smoothing : float
            Multiplicative factor applied to any regularization
            associated with this model.
        save_flag : bool
            If True, save state to disk after initialization.
        save_prefix : str or None
            Prefix used to derive the save filename. If None, the file
            uses just the feature name.
        verbose : bool
            Print mildly helpful information when True.
        covariate_class_sizes : array or None
            Unused for linear features; present to satisfy the
            ``_Feature`` interface.
        """
        if self._has_transform:
            xx = np.asarray(self._transform(x), dtype=float)
            self._xmean = float(np.mean(xx))
            self._x = xx - self._xmean
        else:
            xa = np.asarray(x, dtype=float)
            self._xmean = float(np.mean(xa))
            self._x = xa - self._xmean

        self._xtx: float = float(self._x.dot(self._x))
        self._x_inf: float = float(np.max(np.abs(self._x))) if self._x.size else 0.0
        self._m: float = 0.0
        self._b: float = 0.0

        if self._has_l1:
            self._lambda1 = self._coef1 * smoothing
        if self._has_l2:
            self._lambda2 = self._coef2 * smoothing
        if self._has_huber:
            self._lambda_huber = self._coef_huber * smoothing
        if self._has_group_lasso:
            self._lambda_group_lasso = self._coef_group_lasso * smoothing
        if self._has_group_lasso_inf:
            self._lambda_group_lasso_inf = self._coef_group_lasso_inf * smoothing

        self._verbose = verbose
        if save_flag:
            self._save_self = True
            if save_prefix is None:
                self._filename = f"{self._name}.pckl"
            else:
                self._filename = f"{save_prefix}_{self._name}.pckl"
            self._save()
        else:
            self._filename = None
            self._save_self = False

    def _save(self) -> None:
        """Save parameters so model fitting can be continued later."""
        assert self._filename is not None
        mv: dict[str, Any] = {
            "has_transform": self._has_transform,
            "has_l1": self._has_l1,
            "has_l2": self._has_l2,
            "has_huber": self._has_huber,
            "has_group_lasso": self._has_group_lasso,
            "has_group_lasso_inf": self._has_group_lasso_inf,
            "has_prior": self._has_prior,
            "verbose": self._verbose,
            "name": self._name,
            "save_self": self._save_self,
            "xmean": self._xmean,
            "x": self._x,
            "xtx": self._xtx,
            "x_inf": self._x_inf,
            "m": self._m,
            "b": self._b,
        }
        if self._has_transform:
            mv["transform"] = self._transform
        if self._has_l1:
            mv["coef1"] = self._coef1
            mv["lambda1"] = self._lambda1
        if self._has_l2:
            mv["coef2"] = self._coef2
            mv["lambda2"] = self._lambda2
        if self._has_huber:
            mv["coef_huber"] = self._coef_huber
            mv["lambda_huber"] = self._lambda_huber
            mv["delta_huber"] = self._delta_huber
        if self._has_group_lasso:
            mv["coef_group_lasso"] = self._coef_group_lasso
            mv["lambda_group_lasso"] = self._lambda_group_lasso
        if self._has_group_lasso_inf:
            mv["coef_group_lasso_inf"] = self._coef_group_lasso_inf
            mv["lambda_group_lasso_inf"] = self._lambda_group_lasso_inf
        if self._has_prior:
            mv["prior"] = self._prior
        if self._has_constraints:
            mv["has_constraints"] = True
            mv["lower"] = self._lower
            mv["upper"] = self._upper

        with open(self._filename, "wb") as f:
            pickle.dump(mv, f)

    def _load(self, filename: str) -> None:
        """Load parameters from a previous model fitting session."""
        with open(filename, "rb") as f:
            mv = pickle.load(f)

        self._filename = filename

        self._has_transform = mv["has_transform"]
        if self._has_transform:
            self._transform = mv["transform"]

        self._has_l1 = mv["has_l1"]
        if self._has_l1:
            self._coef1 = mv["coef1"]
            self._lambda1 = mv["lambda1"]
        self._has_l2 = mv["has_l2"]
        if self._has_l2:
            self._coef2 = mv["coef2"]
            self._lambda2 = mv["lambda2"]
        # has_huber added later; default to False for older pickles.
        self._has_huber = mv.get("has_huber", False)
        if self._has_huber:
            self._coef_huber = mv["coef_huber"]
            self._lambda_huber = mv["lambda_huber"]
            self._delta_huber = mv["delta_huber"]
        else:
            self._coef_huber = 0.0
            self._lambda_huber = 0.0
            self._delta_huber = 0.0
        # has_group_lasso added later; default to False for older pickles.
        self._has_group_lasso = mv.get("has_group_lasso", False)
        if self._has_group_lasso:
            self._coef_group_lasso = mv["coef_group_lasso"]
            self._lambda_group_lasso = mv["lambda_group_lasso"]
        else:
            self._coef_group_lasso = 0.0
            self._lambda_group_lasso = 0.0
        # has_group_lasso_inf added later; default to False for older pickles.
        self._has_group_lasso_inf = mv.get("has_group_lasso_inf", False)
        if self._has_group_lasso_inf:
            self._coef_group_lasso_inf = mv["coef_group_lasso_inf"]
            self._lambda_group_lasso_inf = mv["lambda_group_lasso_inf"]
        else:
            self._coef_group_lasso_inf = 0.0
            self._lambda_group_lasso_inf = 0.0
        self._has_prior = mv["has_prior"]
        if self._has_prior:
            self._prior = mv["prior"]
        # has_constraints added later; default to False for older pickles.
        self._has_constraints = mv.get("has_constraints", False)
        if self._has_constraints:
            self._lower = mv["lower"]
            self._upper = mv["upper"]
        else:
            self._lower = -np.inf
            self._upper = np.inf

        self._verbose = mv["verbose"]
        self._name = mv["name"]
        self._save_self = mv["save_self"]

        self._xmean = mv["xmean"]
        self._x = mv["x"]
        self._xtx = mv["xtx"]
        # x_inf added alongside group_lasso_inf; recover for older pickles.
        if "x_inf" in mv:
            self._x_inf = mv["x_inf"]
        else:
            self._x_inf = float(np.max(np.abs(self._x))) if self._x.size else 0.0
        self._m = mv["m"]
        self._b = mv["b"]

    def optimize(self, fpumz: FloatArray, rho: float) -> FloatArray:
        r"""Solve the per-feature primal step.

        Parameters
        ----------
        fpumz : ndarray of shape ``(m,)``
            Vector representing :math:`\bar{f}^k + u^k - \bar{z}^k`.
        rho : float
            ADMM parameter. Must be positive.

        Returns
        -------
        fkp1 : ndarray of shape ``(m,)``
            This feature's contribution to the response.
        """
        y = self._m * self._x - fpumz

        if self._has_l2:
            denom = self._xtx + 2 * self._lambda2 / rho
        else:
            denom = self._xtx

        b = float(self._x.dot(y))

        if self._has_huber:
            # Huber penalty lambda_h * h_delta(m) is quadratic
            # (lambda_h * m^2 / 2) for |m| <= delta and linear
            # (lambda_h * delta * |m| - const) outside that band. Composes
            # additively with optional ridge; the construction-time guard
            # rules out l1 / group_lasso variants on this branch.
            denom_quad = denom + self._lambda_huber / rho
            m_quad = b / denom_quad
            if abs(m_quad) <= self._delta_huber:
                self._m = m_quad
            elif m_quad > self._delta_huber:
                self._m = (b - self._lambda_huber * self._delta_huber / rho) / denom
            else:
                self._m = (b + self._lambda_huber * self._delta_huber / rho) / denom
        else:
            # L1 (lambda1 * |m|), L2 group lasso (lambda_gl * |m| * sqrt(xtx)),
            # and L_inf group lasso (lambda_gl_inf * |m| * max|x - xmean|) all
            # contribute additively to a 1-D soft-threshold on `b` -- they are
            # each a positive multiple of |m|. With ridge present, the
            # shrinkage is the elastic-net closed form:
            # m = sign(b) * max(|b| - threshold, 0) / denom, and m = 0 when
            # |b| <= threshold.
            threshold = 0.0
            if self._has_l1:
                threshold += self._lambda1 / rho
            if self._has_group_lasso:
                threshold += self._lambda_group_lasso * math.sqrt(self._xtx) / rho
            if self._has_group_lasso_inf:
                threshold += self._lambda_group_lasso_inf * self._x_inf / rho

            if threshold > 0.0:
                if b > threshold:
                    self._m = (b - threshold) / denom
                elif b < -threshold:
                    self._m = (b + threshold) / denom
                else:
                    self._m = 0.0
            else:
                self._m = b / denom

        if self._has_constraints:
            # Objective in m is convex (quadratic + piecewise-linear penalty),
            # so projecting the unconstrained minimizer onto [lower, upper] is
            # the constrained optimum.
            if self._m < self._lower:
                self._m = self._lower
            elif self._m > self._upper:
                self._m = self._upper

        self._b = -self._m * self._xmean

        if self._save_self:
            self._save()

        return self._m * self._x

    def compute_dual_tol(self, y: FloatArray) -> float:
        """Compute this feature's contribution to the dual residual tolerance."""
        ybar = float(np.sum(y))
        xty = float(self._x.dot(y))
        return (xty + 2 * self._xmean * ybar) * xty + (
            1.0 + self._xmean * self._xmean
        ) * ybar * ybar

    def _apply_adaptive_l1(self, gamma: float, eps: float) -> bool:
        """Rewrite ``self._coef1`` as the adaptive-lasso weight from the pilot.

        With a single slope ``m``, the new L1 coefficient is
        ``base / (|m| + eps) ** gamma``. The next ``initialize()`` call
        will rescale this by ``smoothing`` into ``self._lambda1``.
        Returns True iff the feature has a non-zero L1 base coefficient.
        """
        if not self._has_l1 or self._coef1 == 0.0:
            return False
        pilot_dev = abs(float(self._m))
        self._coef1 = float(self._coef1) / (pilot_dev + eps) ** gamma
        return True

    def num_params(self) -> int:
        """Number of parameters in this feature."""
        return 1

    def dof(self) -> float:
        """Effective degrees of freedom contributed by this feature."""
        return 1.0

    def predict(self, X: npt.NDArray[Any]) -> FloatArray:
        """Apply fitted model to feature."""
        if self._has_transform:
            return np.asarray(self._m * self._transform(X) + self._b, dtype=float)
        return np.asarray(self._m * np.asarray(X, dtype=float) + self._b, dtype=float)

    def _plot(self, true_fn: Any = None) -> None:
        """Plot is not yet implemented for linear features."""
        return None

    def __str__(self) -> str:
        return f"Feature {self._name}: beta = {self._m:.06g}\n"
