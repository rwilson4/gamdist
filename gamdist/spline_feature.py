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
import os
import pickle
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.linalg as la

from .feature import _Feature

FloatArray = npt.NDArray[np.float64]
Transform = Callable[[npt.NDArray[Any]], npt.NDArray[Any]]


class SplineError(Exception):
    """Raised when spline knot selection or basis evaluation fails."""


def _sample(x: FloatArray) -> FloatArray:
    """Sample knots from an array, culling duplicates and very-close points."""
    num_points = 100
    min_dist = 1e-3
    alpha = 28.2
    max_samples = 1000

    n = len(x)
    if n <= 1:
        return x

    y = np.sort(x)
    z = np.zeros(n)
    z[0] = y[0]
    j = 0
    for i in range(1, n):
        if y[i] > z[j] + min_dist:
            j += 1
            z[j] = y[i]

    if j < num_points:
        return z[0 : (j + 1)]

    num_samples = num_points + int(math.floor(alpha * math.log10(j + 2 - num_points)))
    num_samples = min(num_samples, max_samples)

    if j < num_samples:
        return z[0 : (j + 1)]

    out = np.zeros(num_samples)
    zz = float(j + 1) / num_samples
    for i in range(num_samples - 1):
        k = int(math.floor(i * zz))
        out[i] = z[k]
    out[num_samples - 1] = z[j]
    return out


def _evaluate_spline_basis(X: float, xi: FloatArray) -> FloatArray:
    """Evaluate the cubic regression-spline basis at a single point."""
    K = len(xi)
    if K <= 1:
        raise SplineError("Must have at least two knots.")

    N = np.ones(K)
    N[1] = X

    d = np.zeros(K - 1)
    last_xi = xi[K - 1]
    for k in range(K - 1):
        if X > xi[k]:
            term1 = X - xi[k]
            d[k] = term1 * term1 * term1
        else:
            continue

        if X > last_xi:
            term2 = X - last_xi
            d[k] -= term2 * term2 * term2

        d[k] /= last_xi - xi[k]

    last_d = d[K - 2]
    for k in range(K - 2):
        N[k + 2] = d[k] - last_d

    return N


def _omega_curvature(xi: FloatArray) -> FloatArray:
    """Build the integrated-curvature penalty matrix Omega."""
    K = len(xi)
    if K <= 1:
        raise SplineError("Must have at least two knots.")

    Omega = np.zeros((K, K))

    last_xi = xi[K - 1]
    second_last_xi = xi[K - 2]

    for i in range(2, K):
        term = second_last_xi - xi[i - 2]
        Omega[i, i] = 12.0 * term * term / (last_xi - xi[i - 2])

        for j in range(i + 1, K):
            term1 = (second_last_xi - xi[j - 2]) / (last_xi - xi[j - 2])
            term1 *= 12.0 / (last_xi - xi[i - 2])
            term2 = second_last_xi * (last_xi - 0.5 * (xi[i - 2] + xi[j - 2]))
            term2 -= xi[i - 2] * last_xi
            term2 += xi[j - 2] * (1.5 * xi[i - 2] - 0.5 * xi[j - 2])
            Omega[i, j] = term1 * term2

        for j in range(2, i):
            Omega[i, j] = Omega[j, i]
    return Omega


def _solve_pos(A: FloatArray, B: FloatArray) -> FloatArray:
    """Solve ``A @ X = B`` for symmetric positive-definite ``A``."""
    return la.solve(A, B, assume_a="pos", check_finite=False)


def _determine_smoothing(
    NtN: FloatArray,
    Omega: FloatArray,
    dof: float,
    lmbda_low: float = 0.0,
    lmbda_high: float = 1.0,
    tolerance: float = 1e-12,
) -> float:
    """Bisect on the smoothing parameter to hit the target effective DOF."""
    A = NtN + lmbda_high * Omega
    A_solve_NtN = _solve_pos(A, NtN)

    while A_solve_NtN.trace() > dof:
        lmbda_low = lmbda_high
        lmbda_high *= 2.0
        A = NtN + lmbda_high * Omega
        A_solve_NtN = _solve_pos(A, NtN)

    tolerance *= lmbda_high

    while lmbda_high > lmbda_low + 2.0 * tolerance:
        lmbda = 0.5 * (lmbda_low + lmbda_high)
        A = NtN + lmbda * Omega
        A_solve_NtN = _solve_pos(A, NtN)
        if A_solve_NtN.trace() > dof:
            lmbda_low = lmbda
        else:
            lmbda_high = lmbda
    return 0.5 * (lmbda_low + lmbda_high)


class _SplineFeature(_Feature):
    """A cubic-regression-spline feature with a curvature penalty."""

    def __init__(
        self,
        name: str | None = None,
        transform: Transform | None = None,
        rel_dof: float = 4.0,
        load_from_file: str | None = None,
    ) -> None:
        self.__type__ = "spline"
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

        self._rel_dof = float(rel_dof)

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
        if self._has_transform:
            self._x = np.asarray(self._transform(x), dtype=float)
        else:
            self._x = np.asarray(x, dtype=float)

        self._num_obs = len(self._x)
        self._smoothing = float(smoothing)
        self._verbose = verbose
        self._xi = _sample(self._x)
        num_knots = len(self._xi)

        if self._verbose:
            print(f"Number of observations: {self._num_obs:d}")
            print(f"Number of knots: {num_knots:d}")

        N = np.zeros((self._num_obs, num_knots))
        for i in range(self._num_obs):
            N[i, :] = _evaluate_spline_basis(self._x[i], self._xi)

        self._N = N
        self._NtN = N.transpose().dot(N)
        self._Omega = _omega_curvature(self._xi)
        self._theta = np.zeros(num_knots)
        self._lmbda = _determine_smoothing(self._NtN, self._Omega, self._rel_dof)
        self._computed_cho_factor = False
        self._cho_factor: tuple[FloatArray, bool] | None = None
        self._c: FloatArray = np.zeros(num_knots)
        self._w: FloatArray = np.zeros(num_knots)
        self._constant: float = 0.0

        if self._smoothing == 1.0:
            self._dof: float = self._rel_dof
        else:
            A = self._NtN + self._smoothing * self._lmbda * self._Omega
            self._dof = float(_solve_pos(A, self._NtN).trace())

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
            "name": self._name,
            "has_transform": self._has_transform,
            "rel_dof": self._rel_dof,
            "x": self._x,
            "num_obs": self._num_obs,
            "smoothing": self._smoothing,
            "verbose": self._verbose,
            "xi": np.asarray(self._xi),
            "N": self._N,
            "NtN": self._NtN,
            "Omega": self._Omega,
            "theta": self._theta,
            "lmbda": self._lmbda,
            "computed_cho_factor": self._computed_cho_factor,
            "cho_factor": self._cho_factor,
            "dof": self._dof,
            "save_self": self._save_self,
        }
        if self._has_transform:
            mv["transform"] = self._transform

        with open(self._filename, "wb") as f:
            pickle.dump(mv, f)

    def _load(self, filename: str) -> None:
        """Load parameters from a previous model fitting session."""
        with open(filename, "rb") as f:
            mv = pickle.load(f)

        self._filename = filename
        self._name = mv["name"]
        self._has_transform = mv["has_transform"]
        if self._has_transform:
            self._transform = mv["transform"]
        self._rel_dof = mv["rel_dof"]
        self._x = mv["x"]
        self._num_obs = mv["num_obs"]
        self._smoothing = mv["smoothing"]
        self._verbose = mv["verbose"]
        self._xi = mv["xi"]
        self._N = mv["N"]
        self._NtN = mv["NtN"]
        self._Omega = mv["Omega"]
        self._theta = mv["theta"]
        self._lmbda = mv["lmbda"]
        self._computed_cho_factor = mv["computed_cho_factor"]
        self._cho_factor = mv["cho_factor"]
        self._dof = mv["dof"]
        self._save_self = mv["save_self"]
        self._c = np.zeros(len(self._xi))
        self._w = np.zeros(len(self._xi))
        self._constant = 0.0

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
        y = self._N.dot(self._theta) - fpumz
        Nty = self._N.transpose().dot(y)
        if not self._computed_cho_factor:
            nu = 2.0 * self._smoothing * self._lmbda / rho
            A = self._NtN + nu * self._Omega
            self._cho_factor = la.cho_factor(A)
            self._c = np.mean(self._N, axis=0)
            self._w = la.cho_solve(self._cho_factor, self._c)
            self._constant = 1.0 / float(self._c.dot(self._w))
            self._computed_cho_factor = True

        assert self._cho_factor is not None
        theta_wc = la.cho_solve(self._cho_factor, Nty, check_finite=False)
        self._theta = theta_wc - self._constant * self._c.dot(theta_wc) * self._w

        if self._save_self:
            self._save()

        return self._N.dot(self._theta)

    def compute_dual_tol(self, y: FloatArray) -> float:
        """Compute this feature's contribution to the dual residual tolerance."""
        Aty = self._N.transpose().dot(y)
        return float(Aty.dot(Aty))

    def num_params(self) -> int:
        """Number of parameters in this feature."""
        return int(len(self._theta))

    def dof(self) -> float:
        """Effective degrees of freedom contributed by this feature."""
        return float(self._dof)

    def predict(self, X: npt.NDArray[Any]) -> FloatArray:
        """Apply fitted model to feature."""
        if self._has_transform:
            xx = np.asarray(self._transform(X), dtype=float)
        else:
            xx = np.asarray(X, dtype=float)

        num_obs = len(xx)
        num_knots = len(self._xi)
        N = np.zeros((num_obs, num_knots))
        for i in range(num_obs):
            N[i, :] = _evaluate_spline_basis(xx[i], self._xi)
        return N.dot(self._theta)

    def _plot(self, true_fn: Callable[[FloatArray], FloatArray] | None = None) -> None:
        """Plot the fitted spline against an optional true function."""
        import matplotlib.pyplot as plt

        num_obs = 100
        x_plot = np.linspace(np.min(self._x), np.max(self._x), num=num_obs)

        num_knots = len(self._xi)
        N = np.zeros((num_obs, num_knots))
        for i in range(num_obs):
            N[i, :] = _evaluate_spline_basis(x_plot[i], self._xi)

        y_hat = N.dot(self._theta)

        A = N.transpose().dot(N) + (self._smoothing * self._lmbda) * self._Omega
        S = _solve_pos(A, N.transpose())
        se = np.zeros(num_obs)
        for i in range(num_obs):
            se[i] = la.norm(N[i, :].dot(S))
        ub = y_hat + 2 * se
        lb = y_hat - 2 * se

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.fill_between(x_plot, lb, ub, color="y")
        ax.plot(x_plot, y_hat, "g-")
        if true_fn is not None:
            y_true = true_fn(x_plot)
            ax.plot(x_plot, y_true, "k--")
        plt.xlabel(self._name, fontsize=24)
        plt.ylabel(f"f_{self._name}", fontsize=24)
        plt.title(f"df_lambda = {self._dof:.0f}", fontsize=24)
        plt.gcf().subplots_adjust(bottom=0.15)
        if os.environ.get("MPLBACKEND", "") not in {"Agg", "agg"}:
            plt.show()  # pragma: no cover
        plt.close(fig)

    def __str__(self) -> str:
        return f"Feature {self._name} (spline): {self._dof:0.0f} dof\n"
