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

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


class _Feature(ABC):
    """Abstract base class for feature types fit by the GAM ADMM loop."""

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
        """Compute feature-specific state from data."""

    @abstractmethod
    def optimize(self, fpumz: FloatArray, rho: float) -> FloatArray:
        """Solve the per-feature primal step and return fitted values."""

    @abstractmethod
    def compute_dual_tol(self, y: FloatArray) -> float:
        """Return this feature's contribution to the dual residual tolerance."""

    @abstractmethod
    def num_params(self) -> int:
        """Number of parameters in this feature."""

    @abstractmethod
    def dof(self) -> float:
        """Effective degrees of freedom contributed by this feature."""

    @abstractmethod
    def predict(self, X: npt.NDArray[Any]) -> FloatArray:
        """Compute this feature's contribution to the linear predictor."""

    @abstractmethod
    def _save(self) -> None:
        """Persist feature state to its pickle file."""

    @abstractmethod
    def _load(self, filename: str) -> None:
        """Restore feature state from a pickle file."""

    def _plot(self, true_fn: Any = None) -> None:
        return None

    def __str__(self) -> str:
        return f"Feature {self._name}"
