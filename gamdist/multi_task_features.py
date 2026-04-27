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

"""Feature classes that span multiple tasks in a ``MultiTaskGAM``.

A :class:`_MultiTaskFeature` owns one set of coefficients shared across
the ``K`` tasks and, optionally, a convex *coupling* regularizer that
ties those ``K`` coefficient vectors together. Coupling lives entirely
inside the feature -- the multi-task orchestrator never sees a penalty
coefficient -- so the seam principle from CLAUDE.md is preserved.

The first concrete subclass is :class:`_MultiTaskLinearFeature`: a
linear contribution :math:`m_k (x_k - \\bar{x}_k)` for each task
:math:`k`. The only coupling penalty implemented in this slice is
*group-lasso across tasks*,

.. math::

   \\lambda \\sqrt{\\sum_k m_k^2\\, x_k^\\top x_k},

the ``K``-task generalization of the existing single-task linear
group-lasso. With :math:`\\lambda = 0` the ``K`` subproblems decouple
and the feature behaves like ``K`` independent
:class:`~gamdist.linear_feature._LinearFeature` instances stitched
together.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from .feature import _Feature

FloatArray = npt.NDArray[np.float64]
Transform = Callable[[npt.NDArray[Any]], npt.NDArray[Any]]


class _MultiTaskFeature(_Feature):
    """Common base class for features shared across multiple tasks.

    Distinguishes a multi-task feature from a single-task one in the
    ADMM orchestrator: ``_MultiTaskGAM`` calls ``optimize_multi`` with a
    list of K ``fpumz`` arrays and expects a list of K contributions
    back. Subclasses also report ``num_tasks``.
    """

    __type__: str
    _num_tasks: int

    def __init__(self, name: str, num_tasks: int) -> None:
        super().__init__(name)
        if num_tasks < 1:
            raise ValueError("num_tasks must be >= 1.")
        self._num_tasks = num_tasks

    @property
    def num_tasks(self) -> int:
        return self._num_tasks

    def optimize(self, fpumz: FloatArray, rho: float) -> FloatArray:
        raise NotImplementedError(
            "_MultiTaskFeature.optimize is not used; the MultiTaskGAM "
            "orchestrator calls optimize_multi() with a list of K fpumz "
            "vectors instead."
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
        raise NotImplementedError(
            "_MultiTaskFeature.initialize is not used; the MultiTaskGAM "
            "orchestrator calls initialize_multi() with a list of K data "
            "arrays instead."
        )

    def predict(self, X: npt.NDArray[Any]) -> FloatArray:
        raise NotImplementedError(
            "_MultiTaskFeature.predict is not used; the MultiTaskGAM "
            "orchestrator calls predict_task(k, X) per task."
        )

    def initialize_multi(
        self,
        xs: list[npt.NDArray[Any]],
        smoothing: float = 1.0,
        verbose: bool = False,
    ) -> None:
        """Compute per-task feature state from K data arrays."""
        raise NotImplementedError

    def optimize_multi(
        self,
        fpumz_list: list[FloatArray],
        rho: float,
    ) -> list[FloatArray]:
        """Solve the per-feature primal step jointly over all K tasks."""
        raise NotImplementedError

    def predict_task(self, k: int, x: npt.NDArray[Any]) -> FloatArray:
        """Compute task k's contribution to the linear predictor on new data."""
        raise NotImplementedError

    def compute_dual_tol_task(self, k: int, y: FloatArray) -> float:
        """Per-task contribution to the dual-residual tolerance."""
        raise NotImplementedError

    def num_params_task(self, k: int) -> int:
        """Per-task parameter count."""
        raise NotImplementedError

    def dof_task(self, k: int) -> float:
        """Per-task effective degrees of freedom."""
        raise NotImplementedError

    # The single-task ABC requires these; persistence is intentionally
    # not part of the first multi-task slice.
    def compute_dual_tol(self, y: FloatArray) -> float:
        raise NotImplementedError

    def num_params(self) -> int:
        return sum(self.num_params_task(k) for k in range(self._num_tasks))

    def dof(self) -> float:
        return float(sum(self.dof_task(k) for k in range(self._num_tasks)))

    def _save(self) -> None:
        raise NotImplementedError(
            "Persistence is not supported for multi-task features in this slice."
        )

    def _load(self, filename: str) -> None:
        raise NotImplementedError(
            "Persistence is not supported for multi-task features in this slice."
        )


class _MultiTaskLinearFeature(_MultiTaskFeature):
    r"""Linear feature shared across K tasks.

    Each task :math:`k` gets its own slope :math:`m_k` and its own
    (centered) data column
    :math:`x_k = X_k[:, \mathrm{name}] - \overline{X_k[:, \mathrm{name}]}`;
    the contribution to task :math:`k`'s linear predictor is
    :math:`m_k x_k`.

    Without a coupling regularizer the ``K`` subproblems decouple and
    the primal step reduces to ``K`` independent 1-D least-squares
    solves (matching :class:`~gamdist.linear_feature._LinearFeature`
    exactly task-by-task).

    Coupling penalties available in this slice:

    ``group_lasso_across_tasks``
        ``regularization={"group_lasso_across_tasks": {"coef": lam}}``
        adds

        .. math::

           \lambda \sqrt{\sum_k m_k^2\, x_k^\top x_k}
           = \lambda\, \bigl\|
           [m_1 \sqrt{x_1^\top x_1}, \ldots,
            m_K \sqrt{x_K^\top x_K}]
           \bigr\|_2

        to the joint objective. This is the ``K``-task generalization
        of the single-task linear group-lasso
        :math:`\lambda |m| \sqrt{x^\top x}`: when :math:`\lambda` is
        large enough the entire ``K``-vector of slopes snaps to zero
        simultaneously, dropping the feature uniformly across all
        tasks. Convex (an :math:`L_2` norm of a linear function of
        :math:`m`); composes additively with the per-task quadratic
        in :meth:`optimize_multi`. With :math:`\lambda = 0` the prox
        decouples and the feature recovers per-task
        ordinary-least-squares behavior.
    """

    def __init__(
        self,
        name: str,
        num_tasks: int,
        transform: Transform | None = None,
        regularization: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(name, num_tasks)
        self.__type__ = "multi_task_linear"

        self._has_transform: bool
        self._transform: Transform
        if transform is not None:
            self._has_transform = True
            self._transform = transform
        else:
            self._has_transform = False

        self._has_group_lasso_across_tasks = False
        self._coef_group_lasso_across_tasks: float = 0.0
        self._lambda_group_lasso_across_tasks: float = 0.0

        if regularization is not None:
            unknown = set(regularization) - {"group_lasso_across_tasks"}
            if unknown:
                raise ValueError(
                    f"_MultiTaskLinearFeature: unsupported regularization keys "
                    f"{sorted(unknown)!r}. The first slice supports only "
                    "'group_lasso_across_tasks'."
                )
            if "group_lasso_across_tasks" in regularization:
                self._has_group_lasso_across_tasks = True
                spec = regularization["group_lasso_across_tasks"]
                if "coef" not in spec:
                    raise ValueError(
                        "No coefficient specified for group_lasso_across_tasks."
                    )
                self._coef_group_lasso_across_tasks = float(spec["coef"])
                if self._coef_group_lasso_across_tasks < 0.0:
                    raise ValueError(
                        "group_lasso_across_tasks coefficient must be non-negative."
                    )

        self._xmean: list[float] = []
        self._x: list[FloatArray] = []
        self._xtx: list[float] = []
        self._m: FloatArray = np.zeros(num_tasks)
        self._b: FloatArray = np.zeros(num_tasks)

    def initialize_multi(
        self,
        xs: list[npt.NDArray[Any]],
        smoothing: float = 1.0,
        verbose: bool = False,
    ) -> None:
        if len(xs) != self._num_tasks:
            raise ValueError(
                f"Expected {self._num_tasks} task data arrays, got {len(xs)}."
            )
        self._xmean = []
        self._x = []
        self._xtx = []
        for x in xs:
            if self._has_transform:
                xx = np.asarray(self._transform(x), dtype=float)
            else:
                xx = np.asarray(x, dtype=float)
            mean = float(np.mean(xx))
            centered = xx - mean
            self._xmean.append(mean)
            self._x.append(centered)
            self._xtx.append(float(centered.dot(centered)))

        if self._has_group_lasso_across_tasks:
            self._lambda_group_lasso_across_tasks = (
                self._coef_group_lasso_across_tasks * smoothing
            )

        self._m = np.zeros(self._num_tasks)
        self._b = np.zeros(self._num_tasks)
        self._verbose = verbose

    def optimize_multi(
        self,
        fpumz_list: list[FloatArray],
        rho: float,
    ) -> list[FloatArray]:
        K = self._num_tasks
        # Per-task unconstrained least-squares optimum.
        # Per-task subproblem (no coupling):
        #     argmin_{m_k}  (rho/2) * ||m_k * x_k - fpumz_k||^2  (... + warm-start
        #     correction matching the single-task linear feature ...)
        # which yields b_k = x_k . y_k where y_k = m_k * x_k - fpumz_k, then
        # m_k = b_k / xtx_k. With group_lasso_across_tasks we apply the
        # K-vector L2 group-lasso prox in the change of variables
        # eta_k = m_k * sqrt(xtx_k), so the penalty becomes λ ‖eta‖_2 and
        # the per-task quadratic stays diagonal (xtx_k cancels out). The
        # closed-form prox is the standard L2 group-lasso shrinkage:
        #     eta = max(0, 1 - λ / (rho * ‖b_eta‖_2)) * b_eta
        # where b_eta = (sqrt(xtx_k) * m_k_unpen)_k.
        b = np.empty(K)
        for k in range(K):
            xk = self._x[k]
            mk = self._m[k]
            yk = mk * xk - fpumz_list[k]
            b[k] = float(xk.dot(yk))

        m_unpen = np.empty(K)
        for k in range(K):
            xtx_k = self._xtx[k]
            m_unpen[k] = b[k] / xtx_k if xtx_k > 0.0 else 0.0

        if (
            self._has_group_lasso_across_tasks
            and self._lambda_group_lasso_across_tasks > 0.0
        ):
            sqrt_xtx = np.sqrt(np.asarray(self._xtx, dtype=float))
            eta = m_unpen * sqrt_xtx
            eta_norm = float(np.linalg.norm(eta))
            threshold = self._lambda_group_lasso_across_tasks / rho
            if eta_norm <= threshold:
                self._m = np.zeros(K)
            else:
                shrink = 1.0 - threshold / eta_norm
                eta_new = shrink * eta
                m_new = np.zeros(K)
                for k in range(K):
                    if sqrt_xtx[k] > 0.0:
                        m_new[k] = eta_new[k] / sqrt_xtx[k]
                self._m = m_new
        else:
            self._m = m_unpen

        for k in range(K):
            self._b[k] = -self._m[k] * self._xmean[k]

        return [self._m[k] * self._x[k] for k in range(K)]

    def predict_task(self, k: int, x: npt.NDArray[Any]) -> FloatArray:
        if self._has_transform:
            xx = np.asarray(self._transform(x), dtype=float)
        else:
            xx = np.asarray(x, dtype=float)
        return np.asarray(self._m[k] * xx + self._b[k], dtype=float)

    def compute_dual_tol_task(self, k: int, y: FloatArray) -> float:
        # Mirrors _LinearFeature.compute_dual_tol per task.
        ybar = float(np.sum(y))
        xty = float(self._x[k].dot(y))
        xmean = self._xmean[k]
        return (xty + 2 * xmean * ybar) * xty + (1.0 + xmean * xmean) * ybar * ybar

    def num_params_task(self, k: int) -> int:
        return 1

    def dof_task(self, k: int) -> float:
        # Group-lasso-across-tasks zeros the entire K-vector at once, so
        # when the feature has been driven to zero, no task contributes a
        # degree of freedom. Otherwise, one slope per task.
        if self._has_group_lasso_across_tasks and math.isclose(
            float(self._m[k]), 0.0, abs_tol=1e-12
        ):
            return 0.0
        return 1.0

    def __str__(self) -> str:
        slopes = ", ".join(f"{m:.06g}" for m in self._m)
        return f"MultiTaskFeature {self._name}: betas = [{slopes}]\n"
