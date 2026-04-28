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

"""Feature ABC.

Defines :class:`_Feature`, the abstract base class every concrete
feature type plugs into the ADMM loop with. The interface is the
modular seam called out in CLAUDE.md: a feature exposes
``initialize`` / ``optimize`` / ``compute_dual_tol`` / ``num_params`` /
``dof`` / ``predict`` / ``_save`` / ``_load``, and the GAM
orchestrator never needs to know whether the feature is linear,
categorical, spline, or something user-defined.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


class _Feature(ABC):
    """Abstract base class for feature types fit by the GAM ADMM loop.

    Concrete subclasses (linear, categorical, spline, ...) implement
    the methods declared abstract here. The :class:`~gamdist.gamdist.GAM`
    orchestrator never inspects feature internals; it only calls these
    interface methods, which is the modular seam called out in
    CLAUDE.md.
    """

    __type__: str
    _name: str
    _filename: str | None

    def __init__(self, name: str) -> None:
        self._name = name

    @abstractmethod
    def initialize(
        self,
        x: npt.NDArray[Any],
        smoothing: float = 1.0,
        save_flag: bool = False,
        save_prefix: str | None = None,
        verbose: bool = False,
        covariate_class_sizes: npt.NDArray[Any] | None = None,
    ) -> None:
        """Compute feature-specific state from training data.

        Parameters
        ----------
        x : ndarray
            Per-observation feature data.
        smoothing : float
            Multiplicative scale applied to every regularization
            coefficient owned by this feature.
        save_flag : bool
            If ``True``, persist state to disk after initialization.
        save_prefix : str, optional
            Prefix used to derive the save filename.
        verbose : bool
            Print mildly helpful information when ``True``.
        covariate_class_sizes : ndarray, optional
            Per-observation covariate class sizes; only meaningful
            for some features and ignored by others.
        """

    @abstractmethod
    def optimize(self, fpumz: FloatArray, rho: float) -> FloatArray:
        r"""Solve the per-feature primal step.

        Parameters
        ----------
        fpumz : ndarray of shape ``(m,)``
            Vector representing :math:`\bar{f}^k + u^k - \bar{z}^k`,
            i.e. the ADMM scaled-residual term.
        rho : float
            ADMM penalty parameter. Must be positive.

        Returns
        -------
        fkp1 : ndarray of shape ``(m,)``
            This feature's contribution to the response.
        """

    @abstractmethod
    def compute_dual_tol(self, y: FloatArray) -> float:
        """Return this feature's contribution to the dual residual tolerance."""

    @abstractmethod
    def num_params(self) -> int:
        """Return the number of parameters in this feature."""

    @abstractmethod
    def dof(self) -> float:
        """Return the effective degrees of freedom contributed by this feature."""

    @abstractmethod
    def predict(self, X: npt.NDArray[Any]) -> FloatArray:
        """Return this feature's contribution to the linear predictor.

        Parameters
        ----------
        X : ndarray
            Per-observation feature data.

        Returns
        -------
        ndarray
            Contribution to :math:`\\eta` for each row of ``X``.
        """

    @abstractmethod
    def _save(self) -> None:
        """Persist feature state to its pickle file."""

    @abstractmethod
    def _load(self, filename: str) -> None:
        """Restore feature state from a pickle file.

        Parameters
        ----------
        filename : str
            Path to the pickle file produced by a previous
            :meth:`_save` call.
        """

    def _plot(self, true_fn: Any = None) -> None:
        return None

    def __str__(self) -> str:
        return f"Feature {self._name}"
