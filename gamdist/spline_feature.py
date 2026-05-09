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

"""Cubic regression spline feature with a curvature smoothness penalty.

:class:`_SplineFeature` represents a smooth univariate term in the
linear predictor as a cubic regression spline on adaptively-chosen
knots. The fit is regularized by an integrated squared second
derivative penalty; the smoothing parameter is determined to hit a
target relative-DOF budget (``rel_dof``), not by cross-validation.

Two solver paths share the same objective:

* Closed-form Cholesky path -- used when no shape constraints and no
  group-lasso terms are active.
* cvxpy path -- used when group lasso, L_inf group lasso, or shape
  constraints (sign / lower / upper / monotonic / convex / concave on
  the values evaluated at the knots) are present.

Helper functions :func:`_sample`, :func:`_evaluate_spline_basis`, and
:func:`_omega_curvature` build the basis matrix and curvature penalty
matrix used in both paths.
"""

from __future__ import annotations

import math
import os
import pickle
import warnings
from collections.abc import Callable
from typing import Any

import cvxpy as cvx
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
        if xi[k] < X:
            term1 = X - xi[k]
            d[k] = term1 * term1 * term1
        else:
            continue

        if last_xi < X:
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
        rel_dof : float
            Relative degrees of freedom for the curvature smoother.
        regularization : dict, optional
            Optional extra regularization beyond the built-in
            curvature penalty.

            * ``group_lasso`` -- ``{"coef": lam}``; adds
              :math:`\lambda \|N \theta\|_2`, zeroing out the entire
              spline contribution once :math:`\lambda` is large
              enough.
            * ``group_lasso_inf`` -- ``{"coef": lam}``; adds
              :math:`\lambda \|N \theta\|_\infty`, the
              :math:`L_\infty` variant. Clips the largest pointwise
              contribution rather than applying uniform :math:`L_2`
              contraction.

            The two variants combine additively when both are
            specified.
        constraints : dict, optional
            Optional convex shape constraints on the fitted spline
            evaluated at the knots :math:`\xi`. ``lower`` / ``upper``
            take floats and bound the spline value. ``sign``
            (``"nonnegative"`` or ``"nonpositive"``) is shorthand and
            may not combine with ``lower`` / ``upper``. ``monotonic``
            (``"increasing"`` or ``"decreasing"``) constrains
            consecutive knot values; ``convex`` / ``concave`` bound
            the second differences of the knot values. Knots are
            sorted, so the constraints are applied along the natural
            ordering. Constraint enforcement at the knots is a near-
            global proxy for the smooth cubic spline (sufficient in
            practice for the intended dose-response / regulated use
            cases). Constraints append to the cvxpy program in
            :meth:`optimize` -- the closed-form Cholesky path is only
            used when no constraints and no group lasso are active.
        load_from_file : str, optional
            If provided, restore parameters from this pickle path.
            All other parameters are ignored when loading.
        """
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

        self._has_group_lasso = False
        self._coef_group_lasso: float = 0.0
        self._lambda_group_lasso: float = 0.0
        self._has_group_lasso_inf = False
        self._coef_group_lasso_inf: float = 0.0
        self._lambda_group_lasso_inf: float = 0.0
        self._solver = "CLARABEL"
        if regularization is not None and "group_lasso" in regularization:
            self._has_group_lasso = True
            if "coef" in regularization["group_lasso"]:
                self._coef_group_lasso = float(regularization["group_lasso"]["coef"])
            else:
                raise ValueError(
                    "No coefficient specified for group_lasso regularization term."
                )

        if regularization is not None and "group_lasso_inf" in regularization:
            self._has_group_lasso_inf = True
            if "coef" in regularization["group_lasso_inf"]:
                self._coef_group_lasso_inf = float(
                    regularization["group_lasso_inf"]["coef"]
                )
            else:
                raise ValueError(
                    "No coefficient specified for group_lasso_inf regularization term."
                )

        self._has_constraints = False
        self._constraint_lower: float = -np.inf
        self._constraint_upper: float = np.inf
        self._constraint_monotonic: str | None = None
        self._constraint_convex: bool = False
        self._constraint_concave: bool = False
        if constraints is not None:
            self._has_constraints = True
            allowed = {"sign", "lower", "upper", "monotonic", "convex", "concave"}
            for key in constraints:
                if key not in allowed:
                    raise ValueError(f"unknown constraint key {key!r}.")
            if "sign" in constraints and (
                "lower" in constraints or "upper" in constraints
            ):
                raise ValueError(
                    "constraints: 'sign' is shorthand for 'lower' / 'upper' and "
                    "may not be combined with them."
                )
            if "sign" in constraints:
                sign = constraints["sign"]
                if sign == "nonnegative":
                    self._constraint_lower = 0.0
                elif sign == "nonpositive":
                    self._constraint_upper = 0.0
                else:
                    raise ValueError(
                        f"constraints['sign']: expected 'nonnegative' or "
                        f"'nonpositive', got {sign!r}."
                    )
            if "lower" in constraints:
                self._constraint_lower = float(constraints["lower"])
            if "upper" in constraints:
                self._constraint_upper = float(constraints["upper"])
            if self._constraint_lower > self._constraint_upper:
                raise ValueError(
                    f"constraints: lower ({self._constraint_lower}) exceeds "
                    f"upper ({self._constraint_upper})."
                )
            if "convex" in constraints and "concave" in constraints:
                raise ValueError(
                    "constraints: 'convex' and 'concave' are mutually exclusive."
                )
            if "monotonic" in constraints:
                direction = constraints["monotonic"]
                if direction not in {"increasing", "decreasing"}:
                    raise ValueError(
                        f"constraints['monotonic']: expected 'increasing' or "
                        f"'decreasing', got {direction!r}."
                    )
                self._constraint_monotonic = direction
            if constraints.get("convex"):
                self._constraint_convex = True
            if constraints.get("concave"):
                self._constraint_concave = True

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

        Selects knot locations from the data (see :func:`_sample`),
        builds the basis matrix :math:`N` (see
        :func:`_evaluate_spline_basis`), the curvature penalty
        :math:`\\Omega` (see :func:`_omega_curvature`), and chooses
        the smoothing parameter to hit the requested ``rel_dof``.

        Parameters
        ----------
        x : ndarray
            Observations corresponding to this feature.
        smoothing : float
            Multiplicative scale applied to the curvature penalty.
        save_flag : bool
            If ``True``, save state to disk after initialization.
        save_prefix : str, optional
            Prefix used to derive the save filename.
        verbose : bool
            Print mildly helpful information when ``True``.
        covariate_class_sizes : ndarray, optional
            Unused for spline features; present to satisfy the
            :class:`~gamdist.feature._Feature` interface.
        """
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
        # Spline basis evaluated at the knots themselves -- used by shape
        # constraints to land at the natural cubic-regression-spline anchor
        # points.
        N_xi = np.zeros((num_knots, num_knots))
        for i in range(num_knots):
            N_xi[i, :] = _evaluate_spline_basis(self._xi[i], self._xi)
        self._N_xi = N_xi
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

        if self._has_group_lasso:
            self._lambda_group_lasso = self._coef_group_lasso * smoothing

        if self._has_group_lasso_inf:
            self._lambda_group_lasso_inf = self._coef_group_lasso_inf * smoothing

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
            "N_xi": self._N_xi,
            "NtN": self._NtN,
            "Omega": self._Omega,
            "theta": self._theta,
            "lmbda": self._lmbda,
            "computed_cho_factor": self._computed_cho_factor,
            "cho_factor": self._cho_factor,
            "dof": self._dof,
            "save_self": self._save_self,
            "has_group_lasso": self._has_group_lasso,
            "has_group_lasso_inf": self._has_group_lasso_inf,
            "solver": self._solver,
        }
        if self._has_transform:
            mv["transform"] = self._transform
        if self._has_group_lasso:
            mv["coef_group_lasso"] = self._coef_group_lasso
            mv["lambda_group_lasso"] = self._lambda_group_lasso
        if self._has_group_lasso_inf:
            mv["coef_group_lasso_inf"] = self._coef_group_lasso_inf
            mv["lambda_group_lasso_inf"] = self._lambda_group_lasso_inf
        if self._has_constraints:
            mv["has_constraints"] = True
            mv["constraint_lower"] = self._constraint_lower
            mv["constraint_upper"] = self._constraint_upper
            mv["constraint_monotonic"] = self._constraint_monotonic
            mv["constraint_convex"] = self._constraint_convex
            mv["constraint_concave"] = self._constraint_concave

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
        # N_xi added alongside constraints; recover for older pickles by
        # rebuilding from the persisted knots.
        if "N_xi" in mv:
            self._N_xi = mv["N_xi"]
        else:
            xi = np.asarray(self._xi)
            num_knots = len(xi)
            N_xi = np.zeros((num_knots, num_knots))
            for i in range(num_knots):
                N_xi[i, :] = _evaluate_spline_basis(xi[i], xi)
            self._N_xi = N_xi
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
        # has_group_lasso / solver added later; default for older pickles.
        self._has_group_lasso = mv.get("has_group_lasso", False)
        self._solver = mv.get("solver", "CLARABEL")
        if self._has_group_lasso:
            self._coef_group_lasso = mv["coef_group_lasso"]
            self._lambda_group_lasso = mv["lambda_group_lasso"]
        else:
            self._coef_group_lasso = 0.0
            self._lambda_group_lasso = 0.0
        # has_group_lasso_inf added later; default for older pickles.
        self._has_group_lasso_inf = mv.get("has_group_lasso_inf", False)
        if self._has_group_lasso_inf:
            self._coef_group_lasso_inf = mv["coef_group_lasso_inf"]
            self._lambda_group_lasso_inf = mv["lambda_group_lasso_inf"]
        else:
            self._coef_group_lasso_inf = 0.0
            self._lambda_group_lasso_inf = 0.0
        # has_constraints added later; default to False for older pickles.
        self._has_constraints = mv.get("has_constraints", False)
        if self._has_constraints:
            self._constraint_lower = mv["constraint_lower"]
            self._constraint_upper = mv["constraint_upper"]
            self._constraint_monotonic = mv["constraint_monotonic"]
            self._constraint_convex = mv["constraint_convex"]
            self._constraint_concave = mv["constraint_concave"]
        else:
            self._constraint_lower = -np.inf
            self._constraint_upper = np.inf
            self._constraint_monotonic = None
            self._constraint_convex = False
            self._constraint_concave = False

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
        if self._has_group_lasso or self._has_group_lasso_inf or self._has_constraints:
            self._theta = self._optimize_cvx(fpumz, rho)
        else:
            self._theta = self._optimize_chol(fpumz, rho)

        if self._save_self:
            self._save()

        return self._N.dot(self._theta)

    def _optimize_chol(self, fpumz: FloatArray, rho: float) -> FloatArray:
        """Closed-form Cholesky-based primal step (no group lasso)."""
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
        return theta_wc - self._constant * self._c.dot(theta_wc) * self._w

    def _optimize_cvx(self, fpumz: FloatArray, rho: float) -> FloatArray:
        """Primal step via cvxpy when a group_lasso term is present.

        Group lasso on the per-feature contribution f_j = N theta penalizes
        ``λ · ||N θ||₂`` (or ``λ · ||N θ||_∞`` for the L_inf variant), both
        of which are non-smooth at zero and break the closed-form Cholesky
        path. The fast path is used whenever neither variant is active, so
        this branch only runs when needed.
        """
        target = self._N.dot(self._theta) - fpumz
        K = len(self._theta)
        nu = 2.0 * self._smoothing * self._lmbda / rho
        gamma = 2.0 * self._lambda_group_lasso / rho
        gamma_inf = 2.0 * self._lambda_group_lasso_inf / rho

        theta_var = cvx.Variable(K)
        N_const = cvx.Constant(self._N)
        Omega_const = cvx.psd_wrap(cvx.Constant(self._Omega))
        c_vec = np.mean(self._N, axis=0)

        obj: Any = cvx.sum_squares(
            N_const @ theta_var - cvx.Constant(target)
        ) + nu * cvx.quad_form(theta_var, Omega_const)
        if self._has_group_lasso:
            obj = obj + gamma * cvx.norm(N_const @ theta_var, 2)
        if self._has_group_lasso_inf:
            obj = obj + gamma_inf * cvx.norm_inf(N_const @ theta_var)
        constraints = [cvx.Constant(c_vec) @ theta_var == 0]

        if self._has_constraints:
            # Shape constraints land on the spline value evaluated at the knots
            # (the natural sequence for cubic regression splines). f_xi[i] is
            # spline(xi_i); pinning these gives near-global shape control on a
            # smooth cubic.
            f_xi = cvx.Constant(self._N_xi) @ theta_var
            if np.isfinite(self._constraint_lower):
                constraints.append(f_xi >= self._constraint_lower)
            if np.isfinite(self._constraint_upper):
                constraints.append(f_xi <= self._constraint_upper)
            if self._constraint_monotonic == "increasing":
                constraints.append(cvx.diff(f_xi) >= 0)
            elif self._constraint_monotonic == "decreasing":
                constraints.append(cvx.diff(f_xi) <= 0)
            if self._constraint_convex:
                constraints.append(cvx.diff(f_xi, 2) >= 0)
            if self._constraint_concave:
                constraints.append(cvx.diff(f_xi, 2) <= 0)
        prob = cvx.Problem(cvx.Minimize(obj), constraints)
        # cvxpy's "Solution may be inaccurate" UserWarning escapes the
        # cvxpy.* module filter (cvxpy attributes it to the caller via
        # stacklevel walking). We accept OPTIMAL_INACCURATE explicitly
        # below, so suppress the noise here.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Solution may be inaccurate", category=UserWarning
            )
            try:
                prob.solve(verbose=self._verbose, solver=self._solver)
            except cvx.SolverError:
                # Constrained sub-problems (especially convex / concave shape
                # constraints across many knots) can hit CLARABEL's numerical
                # tolerance with a sum_squares + curvature-penalty + many
                # inequality constraints structure. SCS is a first-order
                # solver that handles the ill-conditioning, so fall back to it
                # when the default fails.
                prob.solve(verbose=self._verbose, solver="SCS")

        if prob.status not in (cvx.OPTIMAL, cvx.OPTIMAL_INACCURATE):
            raise RuntimeError(
                f"Spline feature {self._name!r} failed to converge "
                f"(status={prob.status!r})."
            )
        value = theta_var.value
        if value is None:
            raise RuntimeError(f"Spline feature {self._name!r} produced no solution.")
        return np.asarray(value, dtype=float).ravel()

    def compute_dual_tol(self, y: FloatArray) -> float:
        """Compute this feature's contribution to the dual residual tolerance."""
        Aty = self._N.transpose().dot(y)
        return float(Aty.dot(Aty))

    def num_params(self) -> int:
        """Number of parameters in this feature."""
        return len(self._theta)

    def dof(self) -> float:
        r"""Effective degrees of freedom contributed by this feature.

        The spline's curvature penalty already shrinks the active
        parameter count from ``num_knots`` toward 1; ``self._dof`` is
        the trace of the smoother
        :math:`(N^T N + \lambda \Omega)^{-1} N^T N` set at fit time.

        The one regularization that operates on top of this and can
        therefore reduce ``dof`` further is the group-lasso family
        (``group_lasso`` / ``group_lasso_inf``), which zeros the
        feature's contribution :math:`N\theta` once :math:`\lambda` is
        large enough --- feature selection on a smooth function. The
        penalties act on :math:`\|N\theta\|`, not on :math:`\theta`
        directly, so :math:`\theta` itself can be non-tiny while
        sitting in the (numerical) null space of :math:`N`; the check
        is therefore on the fitted contribution, not :math:`\theta`.
        """
        if self._has_group_lasso or self._has_group_lasso_inf:
            f_hat = self._N @ self._theta
            if float(np.max(np.abs(f_hat))) < 1e-4:
                return 0.0
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
