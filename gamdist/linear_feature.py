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
        load_from_file: str | None = None,
    ) -> None:
        """Initialize feature model (independent of data).

        Parameters
        ----------
        name : str
            Name for feature, used to make plots.
        transform : callable
            Transformation applied to the data (e.g. ``np.log1p``).
        regularization : dict
            Description of regularization terms. ``l2`` (ridge) takes
            ``{"coef": λ}`` and adds ``λ · m²`` to the objective.
            ``group_lasso`` takes ``{"coef": λ}`` and adds
            ``λ · ‖f_j‖₂ = λ · |m| · √(xᵀx)`` to the objective, which
            zeros out the entire feature once ``λ`` is large enough.
            The top-level ``prior`` key (if present) provides a scalar
            prior estimate for the slope. ``l1`` is not yet implemented
            for this feature type — see issue #53; for L1 today, use a
            categorical feature.
        load_from_file : str
            If provided, restore parameters from this pickle path. All
            other parameters are ignored when loading.
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
        self._has_group_lasso = False
        self._has_prior = False
        self._coef1: float = 0.0
        self._coef2: float = 0.0
        self._lambda1: float = 0.0
        self._lambda2: float = 0.0
        self._coef_group_lasso: float = 0.0
        self._lambda_group_lasso: float = 0.0
        self._prior: float = 0.0

        if regularization is not None:
            if "l1" in regularization:
                raise ValueError(
                    "L1 regularization on linear features is not implemented "
                    "(see issue #53). Use a categorical feature for L1 today, "
                    "or restrict this feature's regularization to 'l2'."
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

            if self._has_l1 or self._has_l2:
                self._has_prior = True
                self._prior = float(regularization.get("prior", 0.0))

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
        self._m: float = 0.0
        self._b: float = 0.0

        if self._has_l1:
            self._lambda1 = self._coef1 * smoothing
        if self._has_l2:
            self._lambda2 = self._coef2 * smoothing
        if self._has_group_lasso:
            self._lambda_group_lasso = self._coef_group_lasso * smoothing

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
            "has_group_lasso": self._has_group_lasso,
            "has_prior": self._has_prior,
            "verbose": self._verbose,
            "name": self._name,
            "save_self": self._save_self,
            "xmean": self._xmean,
            "x": self._x,
            "xtx": self._xtx,
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
        if self._has_group_lasso:
            mv["coef_group_lasso"] = self._coef_group_lasso
            mv["lambda_group_lasso"] = self._lambda_group_lasso
        if self._has_prior:
            mv["prior"] = self._prior

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
        # has_group_lasso added later; default to False for older pickles.
        self._has_group_lasso = mv.get("has_group_lasso", False)
        if self._has_group_lasso:
            self._coef_group_lasso = mv["coef_group_lasso"]
            self._lambda_group_lasso = mv["lambda_group_lasso"]
        else:
            self._coef_group_lasso = 0.0
            self._lambda_group_lasso = 0.0
        self._has_prior = mv["has_prior"]
        if self._has_prior:
            self._prior = mv["prior"]

        self._verbose = mv["verbose"]
        self._name = mv["name"]
        self._save_self = mv["save_self"]

        self._xmean = mv["xmean"]
        self._x = mv["x"]
        self._xtx = mv["xtx"]
        self._m = mv["m"]
        self._b = mv["b"]

    def optimize(self, fpumz: FloatArray, rho: float) -> FloatArray:
        """Solve the per-feature primal step.

        Parameters
        ----------
        fpumz : (m,) ndarray
            Vector representing :math:`\\bar{f}^k + u^k - \\bar{z}^k`.
        rho : float
            ADMM parameter. Must be positive.

        Returns
        -------
        fkp1 : (m,) ndarray
            Vector representing this feature's contribution to the response.
        """
        y = self._m * self._x - fpumz

        if self._has_l2:
            denom = self._xtx + 2 * self._lambda2 / rho
        else:
            denom = self._xtx

        b = float(self._x.dot(y))

        if self._has_group_lasso:
            # Group lasso on the per-feature contribution f_j = m * x_centered.
            # ||f_j||_2 = |m| * sqrt(xtx), so the penalty contributes a
            # 1-D soft-threshold on `b` with threshold lambda_gl/rho * sqrt(xtx).
            threshold = self._lambda_group_lasso * math.sqrt(self._xtx) / rho
            if b > threshold:
                self._m = (b - threshold) / denom
            elif b < -threshold:
                self._m = (b + threshold) / denom
            else:
                self._m = 0.0
        else:
            self._m = b / denom

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
