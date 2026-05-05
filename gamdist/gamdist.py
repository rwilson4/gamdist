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

"""Single-response GAM and the ADMM orchestration loop.

This module contains :class:`GAM` -- the user-facing model -- together
with the family / link tables and the ADMM driver that ties per-feature
primal steps to the per-outcome proximal step.

The public entry points are:

* :class:`GAM` -- configure with a family and link, register features
  with :meth:`GAM.add_feature`, fit with :meth:`GAM.fit`, then call
  :meth:`GAM.predict`, :meth:`GAM.deviance`, :meth:`GAM.aic`,
  :meth:`GAM.summary`, etc.
* :func:`fit_adaptive_lasso` -- two-stage adaptive-lasso wrapper that
  refits with reweighted L1 penalties.

The algorithm follows Chu, Keshavarz, & Boyd's distributed-fitting
paper (citation key ``GAMADMM`` in :class:`GAM`'s docstring); the
convexity-only design rule is captured in CLAUDE.md.
"""

from __future__ import annotations

import os
import pickle
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.linalg as linalg
import scipy.optimize as optimize
import scipy.special as special
import scipy.stats as stats

from . import proximal_operators as po
from .categorical_feature import _CategoricalFeature
from .feature import _Feature
from .linear_feature import _LinearFeature
from .spline_feature import _SplineFeature

FloatArray = npt.NDArray[np.float64]

Family = Literal[
    "normal",
    "binomial",
    "poisson",
    "gamma",
    "exponential",
    "inverse_gaussian",
    "quasi_binomial",
    "quasi_poisson",
    "quantile",
    "huber",
]
Link = Literal[
    "identity",
    "logistic",
    "probit",
    "complementary_log_log",
    "log",
    "reciprocal",
    "reciprocal_squared",
]
FeatureType = Literal["categorical", "linear", "spline"]

# Open work and historical "done" list now live in the GitHub issue
# tracker and changelog.txt respectively; see
# https://github.com/rwilson4/gamdist/issues for the current roadmap.

FAMILIES = [
    "normal",
    "binomial",
    "poisson",
    "gamma",
    "exponential",
    "inverse_gaussian",
    "quasi_binomial",
    "quasi_poisson",
    "quantile",
    "huber",
]

LINKS = [
    "identity",
    "logistic",
    "probit",
    "complementary_log_log",
    "log",
    "reciprocal",
    "reciprocal_squared",
]

FAMILIES_WITH_KNOWN_DISPERSIONS = {
    "binomial": 1,
    "poisson": 1,
    "quantile": 1,
    "huber": 1,
}

CANONICAL_LINKS = {
    "normal": "identity",
    "binomial": "logistic",
    "poisson": "log",
    "gamma": "reciprocal",
    "inverse_gaussian": "reciprocal_squared",
    "quasi_binomial": "logistic",
    "quasi_poisson": "log",
    "quantile": "identity",
    "huber": "identity",
}

# Quasi-likelihood families share the score function (and therefore the
# per-observation prox, deviance, and variance function) with a "base"
# full-likelihood family; only the dispersion parameter differs.
# ``_base_family`` is what the per-observation kernels dispatch on, while
# ``_family`` keeps the user-facing name so ``dispersion()`` knows
# whether to estimate phi by Pearson chi-squared / (n - p) instead of
# fixing it at one.
QUASI_BASE_FAMILY = {
    "quasi_binomial": "binomial",
    "quasi_poisson": "poisson",
}

# (family, link) pairs that have a real, convex proximal-operator
# implementation. Convexity of every per-component subproblem is the
# North Star (see CLAUDE.md): ``GAM`` rejects any other combination at
# construction time so users find out before ``fit()`` runs. Any new
# entry here needs (a) a proof that the prox subproblem is convex and
# (b) a dedicated solver in ``proximal_operators.py`` -- the previous
# silent ``scipy.optimize.minimize_scalar`` fallback is gone.
SUPPORTED_FAMILY_LINK_PAIRS: frozenset[tuple[str, str]] = frozenset(
    CANONICAL_LINKS.items()
)


def _plot_convergence(
    prim_res: list[float],
    prim_tol: list[float],
    dual_res: list[float],
    dual_tol: list[float],
    dev: list[float],
) -> None:
    """Plot ADMM convergence progress.

    Convergence is declared when the primal and dual residuals fall
    below tolerances computed from the data as in [ADMM]_. Some
    analysts prefer to declare convergence based on changes to the
    deviance, so this is also plotted -- specifically
    :math:`\\log(\\mathrm{dev} - \\mathrm{dev}_\\mathrm{final})`,
    with a small constant added to avoid :math:`\\log 0`.

    Parameters
    ----------
    prim_res : array-like
        Primal residuals after each iteration.
    prim_tol : array-like
        Primal tolerances after each iteration.
    dual_res : array-like
        Dual residuals after each iteration.
    dual_tol : array-like
        Dual tolerances after each iteration.
    dev : array-like
        Deviances after each iteration.
    """
    import matplotlib.pyplot as plt

    dev_arr = np.asarray(dev, dtype=float)
    fig = plt.figure(figsize=(12.0, 10.0))

    ax = fig.add_subplot(211)
    ax.plot(range(len(prim_res)), prim_res, "b-", label="Primal Residual")
    ax.plot(range(len(prim_tol)), prim_tol, "b--", label="Primal Tolerance")
    ax.plot(range(len(dual_res)), dual_res, "r-", label="Dual Residual")
    ax.plot(range(len(dual_tol)), dual_tol, "r--", label="Dual Tolerance")
    ax.set_yscale("log")
    plt.xlabel("Iteration", fontsize=24)
    plt.ylabel("Residual", fontsize=24)
    plt.legend(fontsize=24, loc=3)

    ax = fig.add_subplot(212)
    ax.plot(
        range(len(dev_arr)), (dev_arr - dev_arr[-1]) + 1e-10, "b-", label="Deviance"
    )
    ax.set_yscale("log")
    plt.xlabel("Iteration", fontsize=24)
    plt.ylabel("Deviance Suboptimality", fontsize=24)

    plt.gcf().subplots_adjust(bottom=0.1)
    plt.gcf().subplots_adjust(left=0.1)
    if os.environ.get("MPLBACKEND", "") not in {"Agg", "agg"}:
        plt.show()  # pragma: no cover
    plt.close(fig)


def _gamma_dispersion(dof: float, dev: float, num_obs: int) -> float:
    """Gamma dispersion.

    Solves the gamma likelihood equation
        2 * num_obs * (log nu - psi(nu)) - dof / nu = dev
    for nu. The left-hand side minus the right-hand side is monotonically
    decreasing in nu and crosses zero exactly once for valid inputs
    (``dev > 0`` and ``2 * num_obs > dof``), so Brent's method on a wide
    bracket converges reliably.

    The previous damped-Newton implementation used a fixed step of 0.1
    and tol=1e-6 and ran out of iterations on perfectly reasonable
    inputs (e.g. ``dof=3, dev=10, num_obs=50``).

    Parameters
    ----------
    dof : float
        Effective degrees of freedom of the fitted model.
    dev : float
        Total deviance.
    num_obs : int
        Number of observations.

    Returns
    -------
    nu : float
        The shape-parameter MLE. (The function name is a misnomer kept
        for backwards compatibility -- callers expect the shape, which
        is the inverse of the dispersion phi.)

    Raises
    ------
    ValueError
        When the equation has no positive root for the given inputs.
    """
    if dev <= 0.0:
        raise ValueError("Could not estimate gamma dispersion: non-positive deviance.")

    def f(nu: float) -> float:
        return 2.0 * num_obs * (np.log(nu) - special.psi(nu)) - dof / nu - dev

    lo, hi = 1e-8, 1e8
    if f(lo) * f(hi) > 0:
        raise ValueError(
            "Could not estimate gamma dispersion: no sign change in bracket."
        )
    return float(optimize.brentq(f, lo, hi, xtol=1e-10, rtol=1e-10))


class GAM:
    """Generalized Additive Model fit via ADMM.

    Configure with a family and link, register features with
    :meth:`add_feature`, fit with :meth:`fit`, then call
    :meth:`predict`, :meth:`deviance`, :meth:`aic`, :meth:`summary`,
    etc.

    Parameters
    ----------
    family : str, optional
        Family of the model. One of:

        * ``"normal"`` -- continuous responses.
        * ``"binomial"`` -- binary responses (or counts with
          ``covariate_class_sizes`` at fit time).
        * ``"poisson"`` -- counts.
        * ``"gamma"`` -- positive continuous responses (in progress).
        * ``"exponential"`` -- gamma with dispersion fixed at 1.
        * ``"inverse_gaussian"`` -- positive continuous responses
          with cubic mean--variance relationship (in progress).
        * ``"quasi_binomial"`` -- binomial score; dispersion estimated
          from the data via Pearson :math:`\\chi^2 / (n - p)` so
          :meth:`dispersion` reflects over- or under-dispersion
          relative to the exact-binomial baseline of 1.
        * ``"quasi_poisson"`` -- Poisson score; same Pearson
          dispersion estimator, useful when count data has
          ``variance > mean``.
        * ``"quantile"`` -- pinball loss; requires ``tau``.
        * ``"huber"`` -- robust M-estimator; requires ``delta``.

        Multinomial (ordinal / nominal) and proportional-hazards
        families are not yet supported. Required unless loading an
        existing model from file (see ``load_from_file``). Point
        estimates for ``"quasi_binomial"`` / ``"quasi_poisson"``
        coincide with their full-likelihood cousins; only the
        dispersion estimator (and any inferential quantity that
        depends on it) differs.
    link : str, optional
        Link function. Supported links and their canonical families:

        ======================  ====================
        Link                    Canonical for family
        ======================  ====================
        ``identity``            ``normal``
        ``logistic``            ``binomial``
        ``log``                 ``poisson``
        ``reciprocal``          ``gamma``
        ``reciprocal_squared``  ``inverse_gaussian``
        ======================  ====================

        ``probit`` and ``complementary_log_log`` are also accepted
        (non-canonical for ``binomial``). If not specified, the
        canonical link is used. Per the convexity-only design rule
        (CLAUDE.md), only ``(family, link)`` pairs with a dedicated
        convex proximal-operator implementation are accepted.
    dispersion : float, optional
        Dispersion parameter. Some families (``binomial``, ``poisson``)
        have a fixed dispersion independent of the data. For other
        families the dispersion is typically estimated from the data;
        passing a known value here reduces uncertainty in the model.
    estimate_overdispersion : bool, optional
        Whether to estimate over-dispersion for binomial and poisson
        families. Only meaningful when covariate classes are present
        and have at least modest size. See [GLM]_, S4.5. Defaults to
        ``False``.
    name : str, optional
        Name for the model. Used in plots and in deriving filenames
        when ``save_flag=True`` is passed to :meth:`fit`.
    load_from_file : str, optional
        Pickle path produced by a previous ``save_flag=True`` fit.
        When set, every other parameter is ignored.
    tau : float, optional
        Quantile level in :math:`(0, 1)` for ``family="quantile"``
        (pinball loss). ``tau=0.5`` recovers the conditional median.
        Required when ``family="quantile"`` and ignored otherwise.
    delta : float, optional
        Knee parameter for ``family="huber"``. Residuals with
        :math:`|y - \\mu| \\le \\delta` are penalized as
        :math:`0.5 r^2` (least squares); larger residuals are
        penalized linearly, capping their per-observation influence.
        Must be positive and is in the units of ``y``. Required when
        ``family="huber"`` and ignored otherwise.

    References
    ----------
    .. [glmnet] glmnet (R package). The standard package for GAMs in R.
       https://cran.r-project.org/web/packages/glmnet/index.html

    .. [pygam] pygam (Python package). Implements GAMs in Python without
       using ADMM. https://github.com/dswah/pyGAM

    .. [GLM] McCullagh, P. and Nelder, J. A. *Generalized Linear
       Models*. The standard text on GLMs.

    .. [GAM] Hastie, T. and Tibshirani, R. *Generalized Additive
       Models*. The book by the folks who invented GAMs.

    .. [ESL] Hastie, T., Tibshirani, R., and Friedman, J.
       *The Elements of Statistical Learning*. Covers a lot more than
       just GAMs.

    .. [GAMr] Wood, S. N. *Generalized Additive Models: An
       Introduction with R*. Covers more implementation details than
       [GAM]_.

    .. [ADMM] Boyd, S., Parikh, N., Chu, E., Peleato, B., and
       Eckstein, J. *Distributed Optimization and Statistical
       Learning via the Alternating Direction Method of Multipliers*.

    .. [GAMADMM] Chu, E., Keshavarz, A., and Boyd, S. *A Distributed
       Algorithm for Fitting Generalized Additive Models*. Forms the
       basis of this package's approach.
    """

    def __init__(
        self,
        family: Family | None = None,
        link: Link | None = None,
        dispersion: float | None = None,
        estimate_overdispersion: bool = False,
        name: str | None = None,
        load_from_file: str | None = None,
        tau: float | None = None,
        delta: float | None = None,
    ) -> None:

        if load_from_file is not None:
            self._load(load_from_file)
            return

        if family is None:
            raise ValueError("Family not specified.")
        elif family not in FAMILIES:
            raise ValueError(f"{family} family not supported")
        elif family == "exponential":
            # Exponential is a special case of Gamma with a dispersion of 1.
            self._family = "gamma"
            dispersion = 1.0
        else:
            self._family = family

        # quasi_* families share their score function (and therefore the
        # prox, deviance, and variance function) with a base family; only
        # the dispersion estimator differs. Per-observation kernels
        # dispatch on ``_base_family`` so they don't need to learn the
        # quasi names.
        self._base_family = QUASI_BASE_FAMILY.get(self._family, self._family)

        if link is None:
            self._link = CANONICAL_LINKS[self._family]
        elif link in LINKS:
            self._link = link
        else:
            raise ValueError(f"{link} link not supported")

        if self._family == "quantile":
            if tau is None:
                raise ValueError("tau must be specified for the quantile family.")
            if not (0.0 < tau < 1.0):
                raise ValueError(f"tau must be in (0, 1); got {tau}.")
            if self._link != "identity":
                raise ValueError(
                    "quantile family requires link='identity'; "
                    f"got link={self._link!r}."
                )
            self._tau: float | None = float(tau)
        else:
            self._tau = None

        if self._family == "huber":
            if delta is None:
                raise ValueError("delta must be specified for the huber family.")
            if not (delta > 0.0):
                raise ValueError(f"delta must be positive; got {delta}.")
            if self._link != "identity":
                raise ValueError(
                    f"huber family requires link='identity'; got link={self._link!r}."
                )
            self._delta: float | None = float(delta)
        else:
            self._delta = None

        if (self._family, self._link) not in SUPPORTED_FAMILY_LINK_PAIRS:
            canonical = CANONICAL_LINKS[self._family]
            raise ValueError(
                f"({self._family!r}, {self._link!r}) is not a supported "
                "(family, link) combination. Per the convexity-only design "
                "(see CLAUDE.md), only pairs with a dedicated convex "
                f"proximal-operator implementation are accepted; use "
                f"link={canonical!r} for family={self._family!r}."
            )

        if dispersion is not None:
            self._known_dispersion = True
            self._dispersion = dispersion
        elif (
            self._family in FAMILIES_WITH_KNOWN_DISPERSIONS
            and not estimate_overdispersion
        ):
            self._known_dispersion = True
            self._dispersion = FAMILIES_WITH_KNOWN_DISPERSIONS[self._family]
        else:
            self._known_dispersion = False

        if self._link == "identity":
            self._eval_link = lambda x: x
            self._eval_inv_link = lambda x: x
        elif self._link == "logistic":
            self._eval_link = lambda x: special.logit(x)
            self._eval_inv_link = lambda x: special.expit(x)
        elif self._link == "probit":
            # Inverse CDF of the Gaussian distribution
            self._eval_link = lambda x: stats.norm.ppf(x)
            self._eval_inv_link = lambda x: stats.norm.cdf(x)
        elif self._link == "complementary_log_log":
            self._eval_link = lambda x: np.log(-np.log(1.0 - x))
            self._eval_inv_link = lambda x: 1.0 - np.exp(-np.exp(x))
        elif self._link == "log":
            self._eval_link = lambda x: np.log(x)
            self._eval_inv_link = lambda x: np.exp(x)
        elif self._link == "reciprocal":
            self._eval_link = lambda x: 1.0 / x
            self._eval_inv_link = lambda x: 1.0 / x
        elif self._link == "reciprocal_squared":
            self._eval_link = lambda x: 1.0 / (x * x)
            self._eval_inv_link = lambda x: 1.0 / np.sqrt(x)

        self._estimate_overdispersion = estimate_overdispersion
        self._features: dict[str, _Feature] = {}
        self._offset = 0.0
        self._num_features = 0
        self._fitted = False
        self._name = name

    def _save(self) -> None:
        """Save model state to a pickle file."""
        mv: dict[str, Any] = {}
        mv["family"] = self._family
        mv["link"] = self._link
        mv["known_dispersion"] = self._known_dispersion
        if self._known_dispersion:
            mv["dispersion"] = self._dispersion
        mv["tau"] = self._tau
        mv["delta"] = self._delta

        mv["estimate_overdispersion"] = self._estimate_overdispersion
        mv["offset"] = self._offset
        mv["num_features"] = self._num_features
        mv["fitted"] = self._fitted
        mv["name"] = self._name

        features: dict[str, dict[str, Any]] = {}
        for name, feature in self._features.items():
            features[name] = {
                "type": feature.__type__,
                "filename": feature._filename,
            }
        mv["features"] = features

        mv["num_obs"] = self._num_obs
        mv["y"] = self._y
        mv["weights"] = self._weights
        mv["has_covariate_classes"] = self._has_covariate_classes
        if self._has_covariate_classes:
            mv["covariate_class_sizes"] = self._covariate_class_sizes

        mv["f_bar"] = self.f_bar
        mv["z_bar"] = self.z_bar
        mv["u"] = self.u
        mv["prim_res"] = self.prim_res
        mv["dual_res"] = self.dual_res
        mv["prim_tol"] = self.prim_tol
        mv["dual_tol"] = self.dual_tol
        mv["dev"] = self.dev

        filename = f"{self._name}_model.pckl"
        with open(filename, "wb") as f:
            pickle.dump(mv, f)

    def _load(self, filename: str) -> None:
        """Load a saved model from a pickle file."""
        with open(filename, "rb") as f:
            mv = pickle.load(f)

        self._filename = filename
        self._family = mv["family"]
        self._base_family = QUASI_BASE_FAMILY.get(self._family, self._family)
        self._link = mv["link"]
        if (self._family, self._link) not in SUPPORTED_FAMILY_LINK_PAIRS:
            canonical = CANONICAL_LINKS.get(self._family, "<unknown>")
            raise ValueError(
                f"Saved model uses ({self._family!r}, {self._link!r}), which "
                "is no longer a supported (family, link) combination. Per "
                "the convexity-only design (see CLAUDE.md), only pairs "
                "with a dedicated convex proximal-operator implementation "
                "are accepted; refit with link="
                f"{canonical!r} for family={self._family!r}."
            )
        self._known_dispersion = mv["known_dispersion"]
        if self._known_dispersion:
            self._dispersion = mv["dispersion"]
        self._tau = mv.get("tau")
        self._delta = mv.get("delta")

        self._estimate_overdispersion = mv["estimate_overdispersion"]
        self._offset = mv["offset"]
        self._num_features = mv["num_features"]
        self._fitted = mv["fitted"]
        self._name = mv["name"]

        self._features = {}
        features = mv["features"]
        for name, feature in features.items():
            if feature["type"] == "categorical":
                self._features[name] = _CategoricalFeature(
                    load_from_file=feature["filename"]
                )
            elif feature["type"] == "linear":
                self._features[name] = _LinearFeature(
                    load_from_file=feature["filename"]
                )
            elif feature["type"] == "spline":
                self._features[name] = _SplineFeature(
                    load_from_file=feature["filename"]
                )
            else:
                raise ValueError("Invalid feature type")

        self._num_obs = mv["num_obs"]
        self._y = mv["y"]
        self._weights = mv["weights"]
        self._has_covariate_classes = mv["has_covariate_classes"]
        if self._has_covariate_classes:
            self._covariate_class_sizes = mv["covariate_class_sizes"]

        self.f_bar = mv["f_bar"]
        self.z_bar = mv["z_bar"]
        self.u = mv["u"]
        self.prim_res = mv["prim_res"]
        self.dual_res = mv["dual_res"]
        self.prim_tol = mv["prim_tol"]
        self.dual_tol = mv["dual_tol"]
        self.dev = mv["dev"]

        if self._link == "identity":
            self._eval_link = lambda x: x
            self._eval_inv_link = lambda x: x
        elif self._link == "logistic":
            self._eval_link = lambda x: special.logit(x)
            self._eval_inv_link = lambda x: special.expit(x)
        elif self._link == "probit":
            # Inverse CDF of the Gaussian distribution
            self._eval_link = lambda x: stats.norm.ppf(x)
            self._eval_inv_link = lambda x: stats.norm.cdf(x)
        elif self._link == "complementary_log_log":
            self._eval_link = lambda x: np.log(-np.log(1.0 - x))
            self._eval_inv_link = lambda x: 1.0 - np.exp(-np.exp(x))
        elif self._link == "log":
            self._eval_link = lambda x: np.log(x)
            self._eval_inv_link = lambda x: np.exp(x)
        elif self._link == "reciprocal":
            self._eval_link = lambda x: 1.0 / x
            self._eval_inv_link = lambda x: 1.0 / x
        elif self._link == "reciprocal_squared":
            self._eval_link = lambda x: 1.0 / (x * x)
            self._eval_inv_link = lambda x: 1.0 / np.sqrt(x)

    def add_feature(
        self,
        name: str,
        type: FeatureType,
        transform: Callable[[npt.NDArray[Any]], npt.NDArray[Any]] | None = None,
        rel_dof: float | None = None,
        regularization: dict[str, Any] | None = None,
        constraints: dict[str, Any] | None = None,
    ) -> None:
        """Add a feature to the model.

        An implicit constant feature is always included, representing
        the overall average response.

        Parameters
        ----------
        name : str
            Name for the feature. Used internally to keep track of
            features and also used when saving files and in plots.
        type : str
            Type of feature. One of:

            * ``"categorical"`` -- for categorical variables.
            * ``"linear"`` -- for variables with a linear contribution
              to the response.
            * ``"spline"`` -- for variables with a potentially
              nonlinear contribution to the response.
        transform : callable, optional
            Optional transform applied to the feature data. Any
            callable may be used; it is applied to data provided
            during fitting and prediction. Common choices include
            :func:`numpy.log`, :func:`numpy.log1p`, or
            :func:`numpy.sqrt`. The user may wish to start with a base
            feature ``"age"`` and use derived features ``"age_linear"``
            and ``"age_quadratic"`` to permit quadratic models for
            that feature, with potentially different regularization
            applied to each.
        rel_dof : float, optional
            Relative degrees of freedom. Applicable only to spline
            features. The degrees of freedom associated with a spline
            represent how "wiggly" it is allowed to be: a spline with
            two degrees of freedom is just a line. (Since features
            are constrained to have zero mean response over the data,
            linear features only have one degree of freedom.) The
            relative degrees of freedom set the baseline smoothing
            parameter :math:`\\lambda` associated with the feature.
            When :meth:`fit` is called, the user may specify an
            overall ``smoothing`` parameter applied to all features
            to alter the amount of regularization in the entire model;
            the actual degrees of freedom will vary based on the
            amount of smoothing. By default, splines have 4 relative
            degrees of freedom.

            Regularization of any feature effectively reduces the
            degrees of freedom; that adjustment is not yet wired in.
        regularization : dict, optional
            Dictionary specifying the regularization applied to this
            feature. Different types of features support different
            types of regularization. Splines always include a
            :math:`C^2` smoothness penalty controlled via ``rel_dof``;
            ``regularization={"group_lasso": {"coef": lam}}``
            additionally shrinks the entire spline contribution and
            can zero it out. ``group_lasso_inf`` is the
            :math:`L_\\infty`-norm variant
            (:math:`\\lambda \\|f_j\\|_\\infty`); it produces a
            clipping rather than a uniform contraction and is also
            available on linear and categorical features. ``huber``
            is a bounded-influence ridge analogue:
            ``regularization={"huber": {"coef": lam, "delta": d}}``
            adds :math:`\\lambda h_\\delta(\\mathrm{coef})` per
            parameter, with :math:`h_\\delta` quadratic for small
            magnitudes and linear beyond :math:`\\delta`. Available
            on linear and categorical features. Other features have
            more diverse options; see each feature class's docstring.
        constraints : dict, optional
            Optional convex shape constraints on the feature's
            coefficients. ``sign`` (``"nonnegative"`` /
            ``"nonpositive"``), ``lower``, and ``upper`` (floats)
            bound coefficients. ``monotonic`` (``"increasing"`` /
            ``"decreasing"``), ``convex``, and ``concave`` impose
            ordering / second-difference constraints (categorical
            features additionally require an ``order`` list of
            category labels; splines order along the knots). Linear
            features support only ``sign`` / ``lower`` / ``upper``.
            See each feature class's docstring for details.
        """
        f: _Feature
        if type == "categorical":
            f = _CategoricalFeature(
                name, regularization=regularization, constraints=constraints
            )
        elif type == "linear":
            f = _LinearFeature(
                name,
                transform,
                regularization=regularization,
                constraints=constraints,
            )
        elif type == "spline":
            f = _SplineFeature(
                name,
                transform,
                rel_dof if rel_dof is not None else 4.0,
                regularization=regularization,
                constraints=constraints,
            )
        else:
            raise ValueError(f"Features of type {type} not supported.")

        self._features[name] = f
        self._num_features += 1

    def fit(
        self,
        X: pd.DataFrame,
        y: npt.NDArray[Any],
        covariate_class_sizes: npt.NDArray[Any] | None = None,
        weights: npt.NDArray[Any] | None = None,
        optimizer: str = "admm",
        smoothing: float = 1.0,
        save_flag: bool = False,
        verbose: bool = False,
        plot_convergence: bool = False,
        max_its: int = 100,
        n_jobs: int = 1,
    ) -> None:
        """Fit the model to data.

        Parameters
        ----------
        X : pandas.DataFrame
            Dataframe of features. The column names must correspond
            to the names of features added to the model. ``X`` may
            have extra columns corresponding to features not included
            in the model; these are silently ignored. Where
            applicable, the data should be "pre-transformation": any
            transformations specified in :meth:`add_feature` are
            applied here.
        y : array-like
            Response. Depending on the model family, the response may
            need to be in a particular form (for example, for a
            binomial family, ``y`` should contain 0s and 1s); this is
            not checked.
        covariate_class_sizes : array-like, optional
            If observations are grouped into covariate classes, the
            size of those classes should be listed here.
        weights : array-like, optional
            Per-observation weights. Effectively specifies the
            dispersion of each observation.
        optimizer : str
            Optimization method. Currently only ``"admm"`` is
            supported, so this argument has no effect.
        smoothing : float
            Multiplicative scale applied to every feature's
            regularization coefficient. Lets the user set the overall
            smoothing by cross-validation while keeping the relative
            regularization across features fixed. Defaults to ``1.0``,
            which leaves the regularization as specified in
            :meth:`add_feature`.
        save_flag : bool
            Whether to save intermediate results after each iteration.
            Useful for complicated models with massive data sets that
            take a while to fit; if the system crashes during the fit
            the analyst can resume from the last checkpoint. Defaults
            to ``False``.
        verbose : bool
            Print mildly useful information during the fit. Defaults
            to ``False``.
        plot_convergence : bool
            Plot the convergence graph at the end of the fit. Defaults
            to ``False``.
        max_its : int
            Maximum number of ADMM iterations. Defaults to 100.
        n_jobs : int
            Number of threads to use for the per-feature primal step
            within each ADMM iteration. Defaults to 1 (serial); pass
            ``-1`` to use :func:`os.cpu_count`. NumPy / SciPy / cvxpy
            release the GIL during their numeric kernels, so
            threading produces real speedup. Expect a 2-4x ceiling on
            models with several non-trivial features (splines,
            categoricals via cvxpy); pure linear-only models are
            usually faster serial because the per-feature work is too
            cheap to amortize thread-dispatch overhead.

        Notes
        -----
        Many binomial data sets include multiple observations with
        identical features. For example, a data set with features
        ``gender`` and ``country`` and a binary survival response
        might be presented in the compact form

        ============  =============  ==========  ===========
        gender        country        patients    survivors
        ============  =============  ==========  ===========
        M             USA            50          48
        F             USA            70          65
        M             CAN            40          38
        F             CAN            45          43
        ============  =============  ==========  ===========

        This is still a binomial family, just more compact. The
        compact format is not yet supported; in this context it is
        important to check for over-dispersion (see [GLM]_). The
        current implementation assumes no over-dispersion and that
        the number of observations sharing a feature pattern is
        small.
        """
        if n_jobs == -1:
            n_jobs = os.cpu_count() or 1
        if n_jobs < 1:
            raise ValueError(f"n_jobs must be >= 1 or -1; got {n_jobs}")
        # No point in more workers than features.
        n_jobs = min(n_jobs, max(self._num_features, 1))
        if save_flag and self._name is None:
            msg = "Cannot save a GAM with no name."
            msg += " Specify name when instantiating model."
            raise ValueError(msg)

        if len(X) != len(y):
            raise ValueError("Inconsistent number of observations in X and y.")
        if weights is not None and len(weights) != len(y):
            raise ValueError(
                f"weights has length {len(weights)} but y has length {len(y)}."
            )
        if covariate_class_sizes is not None and len(covariate_class_sizes) != len(y):
            raise ValueError(
                f"covariate_class_sizes has length {len(covariate_class_sizes)} "
                f"but y has length {len(y)}."
            )

        self._rho = 0.1
        eps_abs = 1e-3
        eps_rel = 1e-3
        # X may have extra columns not registered as features; ignore them. The
        # real number of features is self._num_features.
        self._num_obs, _num_features_in_X = X.shape

        self._y = np.asarray(y).flatten()
        self._weights = weights

        if covariate_class_sizes is not None:
            self._has_covariate_classes = True
            self._covariate_class_sizes = covariate_class_sizes
            mean_response = float(np.sum(self._y)) / float(
                np.sum(self._covariate_class_sizes)
            )
            self._offset = float(self._eval_link(mean_response))
        else:
            self._has_covariate_classes = False
            self._covariate_class_sizes = None
            if self._family == "quantile":
                # The offset anchors the model's average prediction over
                # training data, since features are mean-zero. For
                # quantile regression that anchor must be the marginal
                # tau-quantile, not the mean.
                assert self._tau is not None
                self._offset = float(np.quantile(self._y, self._tau))
            elif self._family == "huber":
                # Huber loss is robust; the median is the L1 limit of
                # its M-estimator and a sensible anchor regardless of
                # delta.
                self._offset = float(np.median(self._y))
            else:
                self._offset = float(self._eval_link(np.mean(self._y)))

        fj: dict[str, FloatArray] = {}

        for name, feature in self._features.items():
            feature.initialize(
                np.asarray(X[name].values),
                smoothing=smoothing,
                covariate_class_sizes=self._covariate_class_sizes,
                save_flag=save_flag,
                save_prefix=self._name,
            )
            fj[name] = np.zeros(self._num_obs)

        self.f_bar = np.full((self._num_obs,), self._offset / self._num_features)
        self.z_bar = np.zeros(self._num_obs)
        self.u = np.zeros(self._num_obs)
        self.prim_res = []
        self.dual_res = []
        self.prim_tol = []
        self.dual_tol = []
        self.dev = []

        self._pool = ThreadPoolExecutor(max_workers=n_jobs) if n_jobs > 1 else None
        try:
            self._admm_loop(max_its, eps_abs, eps_rel, fj, verbose)
        finally:
            if self._pool is not None:
                self._pool.shutdown(wait=True)
            self._pool = None

        self._fitted = True
        if save_flag:
            self._save()

        if plot_convergence:
            _plot_convergence(
                self.prim_res, self.prim_tol, self.dual_res, self.dual_tol, self.dev
            )

    def _admm_loop(
        self,
        max_its: int,
        eps_abs: float,
        eps_rel: float,
        fj: dict[str, FloatArray],
        verbose: bool,
    ) -> None:
        """Run the ADMM iterations. Extracted from ``fit()`` so the
        ThreadPoolExecutor lifecycle (set up before the loop, torn
        down after) can be expressed as a simple try/finally without
        re-indenting the loop body."""
        z_new = np.zeros(self._num_obs)

        for i in range(max_its):
            if verbose:
                print(f"Iteration {i:d}")
                print("Optimizing primal variables")

            fpumz = self._num_features * (self.f_bar + self.u - self.z_bar)
            fj_new: dict[str, FloatArray] = {}
            f_new = np.full((self._num_obs,), self._offset)
            self._optimize_features(fpumz, fj_new, f_new, verbose)

            f_new /= self._num_features

            if verbose:
                print("Optimizing dual variables")

            z_new = self._optimize(self.u + f_new, self._num_features)

            self.u += f_new - z_new

            prim_res = np.sqrt(self._num_features) * linalg.norm(f_new - z_new)
            dual_res = 0.0
            norm_ax = 0.0
            norm_bz = 0.0
            norm_aty = 0.0
            num_params = 0
            for name, feature in self._features.items():
                dr = (
                    (fj_new[name] - fj[name])
                    + (z_new - self.z_bar)
                    - (f_new - self.f_bar)
                )
                dual_res += dr.dot(dr)
                norm_ax += fj_new[name].dot(fj_new[name])
                zik = fj_new[name] + z_new - f_new
                norm_bz += zik.dot(zik)
                norm_aty += feature.compute_dual_tol(self.u)
                num_params += feature.num_params()

            dual_res = self._rho * np.sqrt(dual_res)
            norm_ax = np.sqrt(norm_ax)
            norm_bz = np.sqrt(norm_bz)
            norm_aty = np.sqrt(norm_aty)

            self.f_bar = f_new
            fj = fj_new
            self.z_bar = z_new
            if self._has_covariate_classes:
                sccs = np.sum(self._covariate_class_sizes)
                prim_tol = np.sqrt(
                    sccs * self._num_features
                ) * eps_abs + eps_rel * np.max([norm_ax, norm_bz])

            else:
                prim_tol = np.sqrt(
                    self._num_obs * self._num_features
                ) * eps_abs + eps_rel * np.max([norm_ax, norm_bz])

            dual_tol = np.sqrt(num_params) * eps_abs + eps_rel * norm_aty

            self.prim_res.append(prim_res)
            self.dual_res.append(dual_res)
            self.prim_tol.append(prim_tol)
            self.dual_tol.append(dual_tol)
            self.dev.append(self.deviance())

            if prim_res < prim_tol and dual_res < dual_tol:
                if verbose:
                    print("Fit converged")
                break
        else:
            if verbose:
                print("Fit did not converge")

    def _optimize_features(
        self,
        fpumz: FloatArray,
        fj_new: dict[str, FloatArray],
        f_new: FloatArray,
        verbose: bool,
    ) -> None:
        """Per-feature primal step. Populates ``fj_new`` and accumulates
        into ``f_new`` in-place. Uses ``self._pool`` (a
        ``ThreadPoolExecutor`` set up by ``fit()``) when present.

        ``self._features`` insertion order is preserved on both the
        serial and parallel paths so ``f_new`` accumulates terms in
        the same order -- floating-point addition is not associative,
        so this matters for serial-vs-parallel bit-parity tests.
        """
        if self._pool is None:
            for name, feature in self._features.items():
                if verbose:
                    print(f"Optimizing {name:s}")
                fj_new[name] = feature.optimize(fpumz, self._rho)
                f_new += fj_new[name]
        else:
            if verbose:
                print(f"Optimizing {self._num_features:d} features in parallel")
            futures = {
                name: self._pool.submit(feature.optimize, fpumz, self._rho)
                for name, feature in self._features.items()
            }
            for name, fut in futures.items():
                fj_new[name] = fut.result()
                f_new += fj_new[name]

    def _optimize(self, upf: FloatArray, N: int) -> FloatArray:
        r"""Optimize :math:`\bar{z}` (the ADMM dual step).

        Solves the optimization problem

        .. math::

           \min_z\; L(N z) + \frac{\rho}{2}\,
           \| N z - N u - N \bar{f} \|_2^2

        where :math:`z` is the variable, :math:`N` is the number of
        features, :math:`u` is the scaled dual variable,
        :math:`\bar{f}` is the average feature response, and
        :math:`L` is the family-and-link-specific negative
        log-likelihood. The minimizer is computed via a proximal
        operator (see [GAMADMM]_):

        .. math::

           \mathrm{prox}_\mu(v) := \arg\min_x\, L(x) +
           \frac{\mu}{2}\, \| x - v \|_2^2.

        We compute :math:`(1 / N) \cdot \mathrm{prox}_\mu(N (u +
        \bar{f}))` with :math:`\mu = \rho`, rather than the paper's
        :math:`\mu = \rho / N`; convergence is much faster this way.
        Several ``(family, link)`` pairs admit closed-form proximal
        operators, making this step very fast.

        Parameters
        ----------
        upf : ndarray
            Vector representing :math:`u + \bar{f}`.
        N : int
            Number of features.

        Returns
        -------
        z : ndarray
            Result of the above optimization.
        """

        # ``__init__`` already restricted (family, link) to the
        # SUPPORTED_FAMILY_LINK_PAIRS allow-list, so every branch below
        # has a dedicated convex prox; there is no fallback path.
        # Different prox operators have slightly different signatures
        # (the binomial variant takes a covariate-class-sizes argument,
        # quantile / huber take an extra shape parameter), so the
        # dispatch is typed as Any to keep mypy quiet here. Quasi
        # families share the score function with their base family, so
        # we dispatch on ``_base_family`` here.
        prox: Any
        if self._base_family == "normal":
            prox = po._prox_normal_identity
        elif self._base_family == "binomial":
            prox = po._prox_binomial_logit
            if self._has_covariate_classes:
                return (1.0 / N) * prox(
                    N * upf,
                    self._rho,
                    self._y,
                    self._covariate_class_sizes,
                    self._weights,
                    self._eval_inv_link,
                )
        elif self._base_family == "poisson":
            prox = po._prox_poisson_log
        elif self._base_family == "gamma":
            prox = po._prox_gamma_reciprocal
        elif self._base_family == "inverse_gaussian":
            prox = po._prox_inv_gaussian_reciprocal_squared
        elif self._base_family == "quantile":
            return (1.0 / N) * po._prox_quantile_identity(
                N * upf,
                self._rho,
                self._y,
                self._tau,  # type: ignore[arg-type]
                w=self._weights,
            )
        elif self._base_family == "huber":
            return (1.0 / N) * po._prox_huber_identity(
                N * upf,
                self._rho,
                self._y,
                self._delta,  # type: ignore[arg-type]
                w=self._weights,
            )
        else:
            raise ValueError(
                f"Family {self._family!s} and Link Function {self._link!s} not (yet) supported."
            )

        return (1.0 / N) * prox(
            N * upf, self._rho, self._y, w=self._weights, inv_link=self._eval_inv_link
        )

    def predict(self, X: pd.DataFrame) -> FloatArray:
        """Apply the fitted model to features.

        Parameters
        ----------
        X : pandas.DataFrame
            Data for which to predict the response. The column names
            must correspond to the names of the features used to fit
            the model. ``X`` may have extra columns corresponding to
            features not in the model; these are silently ignored.
            Where applicable, the data should be "pre-transformation":
            transformations specified at model definition time are
            applied here.

        Returns
        -------
        mu : ndarray
            Predicted mean response for each row of ``X``.
        """
        if not self._fitted:
            raise AttributeError("Model not yet fit.")

        num_points, _ = X.shape
        eta = np.full((num_points,), self._offset)
        for name, feature in self._features.items():
            eta = eta + feature.predict(np.asarray(X[name].values))

        return np.asarray(self._eval_inv_link(eta), dtype=float)

    def _design_matrix(
        self, X: pd.DataFrame | None = None
    ) -> tuple[FloatArray, list[str]]:
        """Build a full-rank design matrix used by ``robust_covariance``.

        Parametrization (matches ``statsmodels`` / ``patsy`` defaults):

        * column 0: intercept (all ones)
        * one column per linear feature: the (transformed) raw values
          ``transform(x)`` -- *not* mean-centered. Centering is a
          reparametrization of the intercept and does not change the
          fitted ``mu`` or the inferential conclusions on the linear
          predictor; using raw columns makes the coefficient align with
          ``statsmodels`` term-by-term.
        * for each categorical feature with ``K`` levels: ``K - 1``
          indicator columns dropping the lexicographically-first level
          (treatment contrast), again matching ``statsmodels`` / ``patsy``.

        Spline features and binomial covariate classes are not yet
        wired up here and raise ``NotImplementedError``.

        Parameters
        ----------
        X : pandas DataFrame or None
            If ``None``, builds the design matrix from the training
            data already absorbed by the features. If a frame is
            supplied, its columns are looked up by feature name and
            the per-feature design block is constructed at those
            values (with the same dropped-category as training).

        Returns
        -------
        D : (n, p) ndarray
        names : list of str
            Column names: ``"Intercept"`` followed by per-feature
            labels (e.g. ``"purchases"``, ``"country[T.GBR]"``).
        """
        if not self._fitted:
            raise AttributeError("Model not yet fit.")
        if self._has_covariate_classes:
            raise NotImplementedError(
                "Inferential design matrix not yet supported for binomial "
                "covariate classes."
            )

        if X is None:
            n = self._num_obs
        else:
            n = len(X)

        cols: list[FloatArray] = [np.ones(n, dtype=float)]
        names: list[str] = ["Intercept"]

        for fname, feature in self._features.items():
            if isinstance(feature, _LinearFeature):
                if X is None:
                    block = np.asarray(feature._x + feature._xmean, dtype=float)
                else:
                    raw = np.asarray(X[fname].values)
                    if feature._has_transform:
                        block = np.asarray(feature._transform(raw), dtype=float)
                    else:
                        block = raw.astype(float)
                cols.append(block)
                names.append(fname)
            elif isinstance(feature, _CategoricalFeature):
                cats = sorted(feature._categories, key=str)
                if X is None:
                    raw = np.array(
                        [feature._categories[idx] for idx in feature.x], dtype=object
                    )
                else:
                    raw = np.asarray(X[fname].values)
                # Drop the lexicographically-first level (treatment contrast).
                for level in cats[1:]:
                    cols.append(np.asarray(raw == level, dtype=float))
                    names.append(f"{fname}[T.{level}]")
            else:
                raise NotImplementedError(
                    f"Feature {fname!r} of type {feature.__type__!r} is not yet "
                    "supported by robust_covariance / _design_matrix. Currently "
                    "only linear and categorical features are wired up."
                )

        return np.column_stack(cols), names

    def _glm_irls_terms(
        self,
        y: npt.NDArray[Any],
        mu: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        """Per-observation IRLS weight ``w`` and eta-scale score ``s``.

        For a GLM with linear predictor ``eta``, mean ``mu = g^{-1}(eta)``
        and variance function ``V(mu)``, the score and Fisher information
        with respect to ``eta`` are

            s_i = (y_i - mu_i) * (dmu/deta)_i / V(mu_i)
            w_i = (dmu/deta)_i^2 / V(mu_i)

        Together they give the sandwich pieces ``X' diag(w) X`` (bread)
        and ``X' diag(s^2) X`` (meat).
        """
        if self._base_family == "normal":
            v = np.ones_like(mu, dtype=float)
        elif self._base_family == "binomial":
            eps = np.finfo(float).eps
            mu_c = np.clip(mu, eps, 1.0 - eps)
            v = mu_c * (1.0 - mu_c)
        elif self._base_family == "poisson":
            v = mu
        elif self._base_family == "gamma":
            v = mu * mu
        elif self._base_family == "inverse_gaussian":
            v = mu * mu * mu
        else:
            raise NotImplementedError(
                f"robust_covariance does not yet support the {self._family!r} "
                "family. Supported: normal, binomial, poisson, gamma, "
                "inverse_gaussian, quasi_binomial, quasi_poisson."
            )

        if self._link == "identity":
            dmu_deta = np.ones_like(mu, dtype=float)
        elif self._link == "log":
            dmu_deta = mu
        elif self._link == "logistic":
            eps = np.finfo(float).eps
            mu_c = np.clip(mu, eps, 1.0 - eps)
            dmu_deta = mu_c * (1.0 - mu_c)
        elif self._link == "probit":
            # eta = Phi^{-1}(mu); dmu/deta = phi(eta)
            eta = stats.norm.ppf(np.clip(mu, 1e-12, 1.0 - 1e-12))
            dmu_deta = stats.norm.pdf(eta)
        elif self._link == "complementary_log_log":
            mu_c = np.clip(mu, 1e-12, 1.0 - 1e-12)
            eta = np.log(-np.log(1.0 - mu_c))
            dmu_deta = np.exp(eta) * (1.0 - mu_c)
        elif self._link == "reciprocal":
            dmu_deta = -mu * mu
        elif self._link == "reciprocal_squared":
            dmu_deta = -0.5 * mu * mu * mu
        else:
            raise NotImplementedError(
                f"robust_covariance does not yet support the {self._link!r} link."
            )

        weight = dmu_deta * dmu_deta / v
        score = (y - mu) * dmu_deta / v
        return np.asarray(weight, dtype=float), np.asarray(score, dtype=float)

    def robust_covariance(self, cov_type: str = "HC0") -> FloatArray:
        """Sandwich (Huber-White) covariance estimator.

        Returns the heteroscedasticity-consistent covariance for the
        parameter vector defined by ``_design_matrix``:

            V = (X' W X)^{-1} (X' diag(s_i^2) X) (X' W X)^{-1}

        where ``W_ii = (dmu/deta)_i^2 / V(mu_i)`` is the IRLS weight
        and ``s_i = (y_i - mu_i) (dmu/deta)_i / V(mu_i)`` is the
        per-observation score on the ``eta`` scale. ``mu`` is the
        fitted mean from this GAM. The dispersion drops out, so the
        estimator is identical for known and estimated dispersion.

        For canonical-link GLMs without regularization, the result
        matches ``statsmodels.GLM(...).fit(cov_type="HC0").cov_params()``
        on the same parametrization to numerical precision. Penalized
        fits return the *naive* sandwich at the penalized point
        estimate; for proper penalized inference a bread that includes
        the penalty Hessian is needed (future work).

        Parameters
        ----------
        cov_type : ``"HC0"`` or ``"HC1"``
            ``"HC0"`` is the standard White estimator. ``"HC1"``
            multiplies by ``n / (n - p)`` for a small-sample
            correction (matches Stata / statsmodels HC1).

        Returns
        -------
        V : (p, p) ndarray
            Covariance for the parameter vector
            ``(intercept, linear..., categorical contrasts...)``,
            in the order produced by ``_design_matrix()``.

        Raises
        ------
        AttributeError
            If the model has not been fit.
        NotImplementedError
            For spline features, covariate classes, observation
            weights, the quantile or huber families, or unsupported
            ``cov_type`` values.
        """
        if cov_type not in ("HC0", "HC1"):
            raise NotImplementedError(
                f"cov_type={cov_type!r} not supported. Use 'HC0' or 'HC1'."
            )
        if not self._fitted:
            raise AttributeError("Model not yet fit.")
        if self._weights is not None:
            raise NotImplementedError(
                "robust_covariance does not yet support observation weights."
            )
        if self._family in ("quantile", "huber"):
            raise NotImplementedError(
                f"robust_covariance does not yet support family={self._family!r}; "
                "M-estimator sandwich SEs are tracked separately."
            )

        D, _ = self._design_matrix()
        mu = self._eval_inv_link(self._num_features * self.f_bar)
        weight, score = self._glm_irls_terms(self._y, mu)

        bread = D.T @ (weight[:, None] * D)
        meat = D.T @ ((score * score)[:, None] * D)
        bread_inv = linalg.inv(bread)
        cov = bread_inv @ meat @ bread_inv

        if cov_type == "HC1":
            n = self._num_obs
            p = D.shape[1]
            if n > p:
                cov = cov * (n / (n - p))

        return np.asarray(cov, dtype=float)

    def confidence_intervals(
        self, X: pd.DataFrame, prediction: bool = False, width: float = 0.95
    ) -> FloatArray:
        """Confidence intervals on predictions.

        .. note::

           Not yet implemented; raises :class:`NotImplementedError`.

        Two notions of confidence interval are useful here. The first
        is a confidence interval on :math:`\\mu`, the mean response;
        this captures the uncertainty in the fitted model. The second
        is a confidence interval on observations of this model. For a
        Gaussian family the model might be a perfect fit with billions
        of observations -- so :math:`\\mu` is known precisely and the
        mean-response interval is tiny -- but observations are still
        spread around the mean, so the prediction interval is wider.
        For a binomial family the estimated mean lies in
        :math:`(0, 1)` and admits a confidence interval, but the
        observed response is always 0 or 1 so a "prediction interval"
        is only meaningful in a pedantic sense.

        When making multiple predictions a "global" set of intervals
        (all predictions inside their intervals with the specified
        joint probability) is sometimes desirable. This function does
        not compute global intervals; each interval is computed *in
        vacuo*.

        Parameters
        ----------
        X : pandas.DataFrame
            Data for which to predict the response. The column names
            must correspond to the names of the features used to fit
            the model. ``X`` may have extra columns corresponding to
            features not in the model; these are silently ignored.
            Where applicable, the data should be "pre-transformation".
        prediction : bool
            If ``True``, return a confidence interval on the predicted
            response; if ``False``, on the mean response. Defaults to
            ``False``.
        width : float
            Desired confidence width in :math:`(0, 1)`. Defaults to
            ``0.95``.

        Returns
        -------
        bounds : ndarray of shape ``(n, 2)``
            Lower and upper bounds on the confidence interval
            associated with each prediction.
        """
        raise NotImplementedError("confidence_intervals is not yet implemented")

    def plot(
        self,
        name: str,
        true_fn: Callable[[FloatArray], FloatArray] | None = None,
    ) -> None:
        """Plot the model component for a particular feature.

        Parameters
        ----------
        name : str
            Name of the feature; must be one of the features in the
            model.
        true_fn : callable, optional
            Function representing the "true" relationship between the
            feature and the response, overlaid on the plot for
            comparison.
        """
        self._features[name]._plot(true_fn=true_fn)

    def residuals(
        self,
        kind: Literal["response", "pearson", "deviance", "anscombe"] = "deviance",
    ) -> FloatArray:
        """Residuals for the fitted model.

        Parameters
        ----------
        kind : ``"response"``, ``"pearson"``, ``"deviance"``, or ``"anscombe"``
            ``"response"`` returns ``y - mu`` (raw error).
            ``"pearson"`` returns the standardized residual
            ``(y - E[y]) / sqrt(Var(y))`` using the family's variance
            function and the estimated dispersion. For binomial with
            covariate classes, ``y`` is the count and ``E[y] = m * mu``.
            ``"deviance"`` returns ``sign(y - E[y]) * sqrt(d_i)``, where
            ``d_i >= 0`` is the per-observation deviance contribution;
            squaring and summing gives the total deviance (Wood 2017
            Sec 3.1.7).
            ``"anscombe"`` returns the family-specific transformation
            ``(A(y) - A(mu)) / (V(mu)^(1/6) * sqrt(phi))`` with
            ``A(t) = integral V(s)^(-1/3) ds``, chosen so the residual
            is approximately standard-normal under correct
            specification (McCullagh & Nelder 1989 Sec 2.4.1; Wood
            2017 Sec 3.1.7).

        Returns
        -------
        residuals : array of shape (n_obs,)
        """
        if not self._fitted:
            raise AttributeError("Model not yet fit.")

        y = self._y
        mu = self._eval_inv_link(self._num_features * self.f_bar)
        if self._has_covariate_classes:
            m: npt.NDArray[Any] | float = self._covariate_class_sizes
        else:
            m = 1.0

        if kind == "response":
            return np.asarray(y - mu, dtype=float)
        if kind == "pearson":
            return self._pearson_residuals(y, mu, m)
        if kind == "deviance":
            d_i = self._unit_deviance_for_residuals(y, mu, m)
            if self._base_family == "binomial":
                return np.asarray(np.sign(y - m * mu) * np.sqrt(d_i), dtype=float)
            return np.asarray(np.sign(y - mu) * np.sqrt(d_i), dtype=float)
        if kind == "anscombe":
            return self._anscombe_residuals(y, mu, m)
        raise ValueError(f"Unknown residual kind: {kind!r}")

    def _pearson_residuals(
        self,
        y: npt.NDArray[Any],
        mu: npt.NDArray[Any],
        m: npt.NDArray[Any] | float,
    ) -> FloatArray:
        """Pearson residuals for each family."""
        phi = self.dispersion()
        if self._base_family == "binomial":
            eps = np.finfo(float).eps
            mu_c = np.clip(mu, eps, 1.0 - eps)
            var = m * mu_c * (1.0 - mu_c)
            return np.asarray((y - m * mu_c) / np.sqrt(var * phi), dtype=float)
        if self._base_family == "normal":
            return np.asarray((y - mu) / np.sqrt(phi), dtype=float)
        if self._base_family == "poisson":
            return np.asarray((y - mu) / np.sqrt(mu * phi), dtype=float)
        if self._base_family == "gamma":
            return np.asarray((y - mu) / np.sqrt(mu * mu * phi), dtype=float)
        if self._base_family == "inverse_gaussian":
            return np.asarray((y - mu) / np.sqrt(mu * mu * mu * phi), dtype=float)
        raise ValueError(f"Unsupported family {self._family!r}")

    def _unit_deviance_for_residuals(
        self,
        y: npt.NDArray[Any],
        mu: npt.NDArray[Any],
        m: npt.NDArray[Any] | float,
    ) -> FloatArray:
        """Per-observation deviance contribution ``d_i >= 0``.

        For binomial this differs from the per-term contribution
        accumulated by ``deviance()``: the ``deviance()`` form omits
        the saturated-likelihood constant (which doesn't affect model
        fitting), but ``d_i`` for residuals must include it so that
        ``sqrt(d_i)`` is real and non-negative.
        """
        if self._base_family == "normal":
            return np.asarray((y - mu) ** 2, dtype=float)
        if self._base_family == "binomial":
            eps = np.finfo(float).eps
            mu_c = np.clip(mu, eps, 1.0 - eps)
            with np.errstate(divide="ignore", invalid="ignore"):
                t1 = np.where(y > 0, y * np.log(y / (m * mu_c)), 0.0)
                t2 = np.where(
                    y < m,
                    (m - y) * np.log((m - y) / (m * (1.0 - mu_c))),
                    0.0,
                )
            return np.maximum(2.0 * (t1 + t2), 0.0)
        if self._base_family == "poisson":
            with np.errstate(divide="ignore", invalid="ignore"):
                t = np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / mu), 0.0)
            return np.maximum(2.0 * (t - (y - mu)), 0.0)
        if self._base_family == "gamma":
            tiny = np.finfo(float).tiny
            y_safe = np.where(y > 0, y, tiny)
            mu_safe = np.where(mu > 0, mu, tiny)
            return np.maximum(
                2.0 * (-np.log(y_safe / mu_safe) + (y - mu_safe) / mu_safe), 0.0
            )
        if self._base_family == "inverse_gaussian":
            return np.asarray((y - mu) ** 2 / (mu * mu * y), dtype=float)
        raise ValueError(f"Unsupported family {self._family!r}")

    def _anscombe_residuals(
        self,
        y: npt.NDArray[Any],
        mu: npt.NDArray[Any],
        m: npt.NDArray[Any] | float,
    ) -> FloatArray:
        r"""Anscombe residuals using the family-specific variance-stabilizing
        transformation ``A(t) = \int V(s)^{-1/3} ds``.

        The residual is

            r_A = (A(y) - A(mu)) / (V(mu)^(1/6) * sqrt(phi))

        with the binomial form scaled by ``sqrt(m)`` because the
        per-trial proportion ``y/m`` has variance ``V(mu)/m``. See
        McCullagh & Nelder (1989) Sec 2.4.1 / Table 2.5.
        """
        phi = self.dispersion()
        if self._base_family == "normal":
            # V(mu) = 1, A(t) = t -- coincides with the Pearson residual.
            return np.asarray((y - mu) / np.sqrt(phi), dtype=float)
        if self._base_family == "binomial":
            eps = np.finfo(float).eps
            mu_c = np.clip(mu, eps, 1.0 - eps)
            prop = np.clip(y / m, eps, 1.0 - eps)
            # A(p) = B(2/3, 2/3) * I(2/3, 2/3, p) where I is the regularized
            # incomplete beta function.
            scale = float(special.beta(2.0 / 3.0, 2.0 / 3.0))
            a_y = scale * special.betainc(2.0 / 3.0, 2.0 / 3.0, prop)
            a_mu = scale * special.betainc(2.0 / 3.0, 2.0 / 3.0, mu_c)
            denom = (mu_c * (1.0 - mu_c)) ** (1.0 / 6.0) * np.sqrt(phi)
            return np.asarray(np.sqrt(m) * (a_y - a_mu) / denom, dtype=float)
        if self._base_family == "poisson":
            # V(mu) = mu, A(t) = (3/2) t^(2/3).
            return np.asarray(
                1.5
                * (y ** (2.0 / 3.0) - mu ** (2.0 / 3.0))
                / mu ** (1.0 / 6.0)
                / np.sqrt(phi),
                dtype=float,
            )
        if self._base_family == "gamma":
            # V(mu) = mu^2, A(t) = 3 t^(1/3); V(mu)^(1/6) = mu^(1/3).
            tiny = np.finfo(float).tiny
            y_safe = np.where(y > 0, y, tiny)
            mu_safe = np.where(mu > 0, mu, tiny)
            return np.asarray(
                3.0
                * (y_safe ** (1.0 / 3.0) - mu_safe ** (1.0 / 3.0))
                / mu_safe ** (1.0 / 3.0)
                / np.sqrt(phi),
                dtype=float,
            )
        if self._base_family == "inverse_gaussian":
            # V(mu) = mu^3, A(t) = log(t); V(mu)^(1/6) = sqrt(mu).
            return np.asarray((np.log(y) - np.log(mu)) / np.sqrt(mu * phi), dtype=float)
        raise ValueError(f"Unsupported family {self._family!r}")

    def plot_residuals(
        self,
        kind: Literal["response", "pearson", "deviance", "anscombe"] = "deviance",
    ) -> Any:
        """Plot residuals vs. fitted values and a normal QQ plot.

        Produces a 1x2 matplotlib figure: residual-vs-fitted on the
        left, Q-Q plot against the standard normal on the right.

        Parameters
        ----------
        kind : ``"response"``, ``"pearson"``, or ``"deviance"``
            Forwarded to ``residuals()``.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        res = self.residuals(kind)
        mu = self._eval_inv_link(self._num_features * self.f_bar)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.scatter(mu, res, s=10, alpha=0.6)
        ax1.axhline(0.0, color="grey", linewidth=0.5)
        ax1.set_xlabel("Fitted (mu)")
        ax1.set_ylabel(f"{kind.capitalize()} residual")
        ax1.set_title("Residuals vs Fitted")

        stats.probplot(res, plot=ax2)
        ax2.set_title("Normal Q-Q")

        fig.tight_layout()
        return fig

    def plot_residuals_vs_predictor(
        self,
        predictor: npt.ArrayLike,
        *,
        kind: Literal["response", "pearson", "deviance", "anscombe"] = "deviance",
        name: str | None = None,
    ) -> Any:
        """Plot residuals against a predictor.

        Useful for diagnosing fit adequacy: a clear pattern (curvature,
        funnel shape, etc.) when ``predictor`` is a feature already in
        the model suggests an unmodeled non-linearity or
        heteroscedasticity. The same plot for a predictor *not* in the
        model is a quick check on whether to add it.

        Parameters
        ----------
        predictor : array-like of shape ``(n_obs,)``
            Predictor values for each training observation. Must match
            the length of the training data used to fit this model.
            Categorical (string / object) predictors are plotted as a
            categorical x-axis.
        kind : ``"response"``, ``"pearson"``, ``"deviance"``, or ``"anscombe"``
            Forwarded to ``residuals()``.
        name : str, optional
            Label for the x-axis and title.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        res = self.residuals(kind)
        pred = np.asarray(predictor)
        if pred.shape != (self._num_obs,):
            raise ValueError(
                f"predictor has shape {pred.shape}, "
                f"expected ({self._num_obs},) to match training data"
            )

        label = name if name is not None else "predictor"
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(pred, res, s=10, alpha=0.6)
        ax.axhline(0.0, color="grey", linewidth=0.5)
        ax.set_xlabel(label)
        ax.set_ylabel(f"{kind.capitalize()} residual")
        ax.set_title(f"Residuals vs {label}")
        fig.tight_layout()
        return fig

    def deviance(
        self,
        X: pd.DataFrame | None = None,
        y: npt.NDArray[Any] | None = None,
        covariate_class_sizes: npt.NDArray[Any] | None = None,
        w: npt.NDArray[Any] | None = None,
    ) -> float:
        r"""Deviance of the fitted model.

        With no arguments, returns the training deviance

        .. math::

           D = 2 \phi \bigl(\ell(y; y) - \ell(\mu; y)\bigr),

        where :math:`\phi` is the dispersion (present only to cancel
        the denominator of the log-likelihood),
        :math:`\ell(y; y)` is the log-likelihood of a model that fits
        the data perfectly, and :math:`\ell(\mu; y)` is the
        log-likelihood of the fitted model on its training data. This
        is the quantity ADMM minimizes during fitting.

        With ``X`` and ``y`` supplied, returns the deviance on the
        provided data set. Combined with cross-validation this is the
        usual way to pick the ``smoothing`` parameter by minimizing
        deviance on a hold-out set.

        Parameters
        ----------
        X : pandas.DataFrame, optional
            Dataframe of features. The column names must correspond
            to the names of features added to the model (see
            :meth:`predict`). Only used in the hold-out branch.
        y : array-like, optional
            Response on the hold-out set. Only used in the hold-out
            branch.
        covariate_class_sizes : array-like, optional
            Array of covariate class sizes for the hold-out data.
        w : array-like, optional
            Per-observation weights for the hold-out branch.

        Returns
        -------
        D : float
            The deviance of the model.
        """
        if X is None or y is None:
            y = self._y
            mu = self._eval_inv_link(self._num_features * self.f_bar)
            w = self._weights
            if self._has_covariate_classes:
                m: npt.NDArray[Any] | float = self._covariate_class_sizes
            else:
                m = 1.0
        else:
            mu = self.predict(X)
            if covariate_class_sizes is not None:
                m = covariate_class_sizes
            else:
                m = 1.0

        return self._deviance_from_mu(y, mu, m=m, w=w)

    def _deviance_from_mu(
        self,
        y: npt.NDArray[Any],
        mu: npt.NDArray[Any],
        m: npt.NDArray[Any] | float = 1.0,
        w: npt.NDArray[Any] | None = None,
    ) -> float:
        """Compute deviance for given response, mean, optional class sizes and weights."""
        if self._base_family == "normal":
            y_minus_mu = y - mu
            if w is None:
                return float(y_minus_mu.dot(y_minus_mu))
            return float(w.dot(y_minus_mu * y_minus_mu))
        if self._base_family == "binomial":
            # Clip mu away from 0 and 1 so log(mu) and log1p(-mu) stay finite.
            eps = np.finfo(float).eps
            mu_c = np.clip(mu, eps, 1.0 - eps)
            if w is None:
                return float(
                    -2.0 * np.sum(y * np.log(mu_c) + (m - y) * np.log1p(-mu_c))
                )
            return float(-2.0 * w.dot(y * np.log(mu_c) + (m - y) * np.log1p(-mu_c)))
        if self._base_family == "poisson":
            # `y * log(y / mu)` has the conventional limit 0 when y = 0.
            y_log_term = np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / mu), 0.0)
            term = y_log_term - (y - mu)
            if w is None:
                return float(2.0 * np.sum(term))
            return float(2.0 * w.dot(term))
        if self._base_family == "gamma":
            # Gamma deviance is undefined at y = 0 or mu <= 0 in the strict
            # sense; clip both to a tiny positive number to keep training
            # iterations from blowing up when mu transiently goes negative
            # (gamma + reciprocal link does not constrain mu).
            tiny = np.finfo(float).tiny
            y_safe = np.where(y > 0, y, tiny)
            mu_safe = np.where(mu > 0, mu, tiny)
            if w is None:
                return float(
                    2.0
                    * np.sum(-1.0 * np.log(y_safe / mu_safe) + (y - mu_safe) / mu_safe)
                )
            return float(
                2.0 * w.dot(-1.0 * np.log(y_safe / mu_safe) + (y - mu_safe) / mu_safe)
            )
        if self._base_family == "inverse_gaussian":
            if w is None:
                return float(np.sum((y - mu) * (y - mu) / (mu * mu * y)))
            return float(w.dot((y - mu) * (y - mu) / (mu * mu * y)))
        if self._base_family == "quantile":
            # Pinball loss: rho_tau(r) = max(tau*r, (tau-1)*r). The
            # quantile family has no proper likelihood, so "deviance"
            # here is twice the empirical pinball loss -- the
            # M-estimator analogue of -2 * log-likelihood used by the
            # ADMM convergence trace and goodness-of-fit summaries.
            tau = float(self._tau)  # type: ignore[arg-type]
            r = y - mu
            term = np.maximum(tau * r, (tau - 1.0) * r)
            if w is None:
                return float(2.0 * np.sum(term))
            return float(2.0 * w.dot(term))
        if self._base_family == "huber":
            # Huber loss is an M-estimator without a proper likelihood;
            # report 2 * sum(L_delta(y - mu)) so the inner (quadratic)
            # region matches normal-family deviance and the convergence
            # trace stays comparable across runs.
            delta = float(self._delta)  # type: ignore[arg-type]
            r = y - mu
            abs_r = np.abs(r)
            term = np.where(abs_r <= delta, 0.5 * r * r, delta * (abs_r - 0.5 * delta))
            if w is None:
                return float(2.0 * np.sum(term))
            return float(2.0 * w.dot(term))
        raise ValueError(f"Unsupported family {self._family!r}")

    def dispersion(self, formula: str = "deviance") -> float:
        """Dispersion of the fitted model.

        Returns the dispersion associated with the model. Some
        families (binomial, poisson) have a fixed dispersion; for
        others the dispersion is estimated from the data.

        Different estimators may be appropriate for different
        families. The current implementation is the deviance-based
        estimator of [GAMr]_, Eqn. 3.10 on p. 110. That section notes
        the estimator works poorly for overdispersed Poisson data
        with a small mean response and offers alternatives that have
        not yet been implemented. This is not currently a concern
        because overdispersion is not supported -- and without
        overdispersion the Poisson dispersion is exactly 1.

        Parameters
        ----------
        formula : str
            Formula for the dispersion. Options:

            * ``"deviance"`` (default)
            * ``"pearson"``
            * ``"fletcher"``
        """
        if self._family == "normal":
            if self._known_dispersion:
                return float(self._dispersion)
            return float(self.deviance() / (self._num_obs - self.dof()))
        if self._family == "binomial":
            if self._known_dispersion:
                return float(self._dispersion)
            if self._estimate_overdispersion:
                return float(self._binomial_overdispersion())
            return 1.0
        if self._family == "poisson":
            if self._known_dispersion:
                return float(self._dispersion)
            if self._estimate_overdispersion:
                return float(self._poisson_overdispersion())
            return 1.0
        if self._family == "quasi_poisson":
            # Quasi-likelihood: same score as Poisson, dispersion phi is
            # always estimated from data via Pearson chi-squared / (n - p).
            # The user can still pin phi by passing ``dispersion=`` to
            # ``GAM(...)``.
            if self._known_dispersion:
                return float(self._dispersion)
            return float(self._poisson_overdispersion())
        if self._family == "quasi_binomial":
            # Quasi-likelihood: same score as Binomial, dispersion
            # estimated by Pearson chi-squared / (n - p) (with the
            # binomial variance function ``mu * (1 - mu)``, scaled by
            # ``m`` for covariate-class data).
            if self._known_dispersion:
                return float(self._dispersion)
            return float(self._binomial_pearson_dispersion())
        if self._family == "gamma":
            if self._known_dispersion:
                return float(self._dispersion)
            return _gamma_dispersion(self.dof(), self.deviance(), self._num_obs)
        if self._family == "inverse_gaussian":
            if self._known_dispersion:
                return float(self._dispersion)
            return float(self.deviance() / (self._num_obs - self.dof()))
        if self._family == "quantile":
            # Pinball loss is an M-estimator, not a likelihood; there
            # is no dispersion to estimate.
            return 1.0
        if self._family == "huber":
            # Huber is an M-estimator. There is no dispersion in the
            # likelihood sense; sandwich SEs would be the right inferential
            # tool (#34) but are out of scope here.
            return 1.0
        raise ValueError(f"Unsupported family {self._family!r}")

    def _binomial_overdispersion(self, formula: str | None = None) -> float:
        r"""Estimate binomial over-dispersion.

        Parameters
        ----------
        formula : str, optional
            Which formula to use, either ``"replication"`` or
            ``"pearson"``. See *Notes*.

        Returns
        -------
        sigma2 : float
            Estimate of over-dispersion. Cached on
            ``self._dispersion`` so repeated calls are O(1).

        Notes
        -----
        When using covariate classes, the observed variance may
        exceed the baseline for the family due to clustering in the
        population (see [GLM]_). That text gives two methodologies
        for estimating over-dispersion. Without covariate classes,
        over-dispersion cannot be estimated.

        The most reliable assessment of over-dispersion is only
        possible when there is replication amongst the covariate
        classes. As an example, suppose we have data on patients from
        two hospitals as in the following table. Note that there are
        three rows for males in hospital 1 -- these could be pooled,
        but keeping them separate lets us estimate over-dispersion
        more reliably.

        ========  ==========  ==========  ===========
        Gender    Hospital    Patients    Survivors
        ========  ==========  ==========  ===========
        M         1           30          15
        M         1           40          19
        M         1           35          15
        F         1           10           8
        M         2           10           3
        M         2           18           6
        F         2           40          30
        ========  ==========  ==========  ===========

        Because we are building a model based on gender and hospital
        alone, we assume the three M/1 entries are drawn from the
        same binomial distribution; that hypothesis can be tested
        with, for example, Welch's t-test. A significant departure
        from the null hypothesis indicates an unobserved source of
        between-replicate variation -- different doctors, different
        time periods, etc. Quantifying the additional variance helps
        produce more accurate confidence intervals.

        When replication is present, [GLM]_ suggests the following.
        Suppose a covariate class (e.g. ``Gender=M, Hospital=1``) has
        :math:`r` replicates. Across all :math:`r` replicates,
        determine the observed success rate :math:`\pi`. In the
        example we have 105 patients and 49 survivors, for
        :math:`\pi = 0.47`. We then compute the variance on
        :math:`r - 1` DOF,

        .. math::

           s^2 = \frac{1}{r - 1} \sum_{j=1}^r
           \frac{(y_j - m_j \pi)^2}{m_j \pi (1 - \pi)},

        where :math:`y_j` is the number of successes in the
        :math:`j`-th replicate and :math:`m_j` the number of trials.
        Per [GLM]_ this is an unbiased estimate of the dispersion
        parameter; on these numbers it gives :math:`s^2 = 0.17`,
        indicating under-dispersion. (These are made up; over-
        dispersion is more common than under-dispersion in practice.)

        Each covariate class with replication yields its own
        dispersion estimate. Assuming the dispersion is independent
        of the covariate classes (which may or may not be true) we
        can pool by replication degree. With the :math:`k`-th class
        having :math:`r_k` replicates and dispersion estimate
        :math:`s_k^2`, the overall estimate is

        .. math::

           s^2 = \frac{\sum_k (r_k - 1)\, s_k^2}{\sum_k (r_k - 1)}.

        This pooling formula is *not* in [GLM]_; that text just says
        to pool the estimates without specifying how. This approach
        makes sense, but that doesn't make it correct.

        When replication is sparse the methodology above breaks down.
        [GLM]_ then advocates the Pearson-residual approach: with
        :math:`\pi_j` the model prediction for the :math:`j`-th
        covariate class,

        .. math::

           s^2 = \frac{1}{n - p} \sum_j
           \frac{(y_j - m_j \pi_j)^2}{m_j \pi_j (1 - \pi_j)}.

        This uses the model prediction instead of the pooled
        observation, and :math:`n - p` for the error DOF instead of
        the number of replicates. It still breaks down when covariate
        class sizes :math:`m_j` are small.

        To use the replicate-based formula at least one covariate
        class must show replication and the degree of replication
        must be at least two. If those conditions fail and the user
        requested the replicate-based formula, the directive is
        silently ignored and the Pearson-based approach used.

        If ``formula`` is not specified, the criterion for using
        replication is: at least two covariate classes show
        replication; the most-replicated class has degree at least 3;
        and the total replication DOF is at least 10. In the example
        above replication classes are M/1 and M/2 with replication 3
        and 2, total DOF :math:`(2-1) + (3-1) = 3` -- below the
        threshold of 10 -- so the Pearson formula is used. These
        criteria are arbitrary and would benefit from further
        research.
        """

        if not self._has_covariate_classes:
            return 1.0

        min_cc_replicates = 1
        min_replication = 2

        des_cc_replicates = 2
        des_replication = 3
        des_replication_dof = 10

        # Determine degree of replication
        #
        # To use the replication formula, we need at least one
        # covariate class with replication, and that covariate class
        # needs replication of at least 2. It might make sense to use
        # a more stringent set of criteria, but this is enough for
        # now.
        #
        # The way we decide whether two observations have the same
        # covariate class is by encoding the covariate class by an
        # index. Each categorical feature has already indexed each
        # category by an internal integer between 0 and n_k - 1, where
        # n_k is the number of categories of the kth feature. (None of
        # this is applicable unless all the features are categorical.
        #
        # We use these internal indices along with the numbers of
        # categories in conjunction with the numpy ravel_multi_index
        # function to map a tuple of category indices into a single
        # integer between 0 and the the product of all category sizes
        # (minus 1).
        #
        # We need to take care to loop over the features in a
        # consistent order, so we create the fnames array just to give
        # an arbitrary but consistent ordering.
        r: dict[int, int] = {}
        covariate_class = np.zeros((self._num_obs,), dtype=np.int64)
        fnames = sorted(self._features)
        for i in range(self._num_obs):
            multi_index = []
            dims = []
            for fname in fnames:
                # Overdispersion is only meaningful when every feature is
                # categorical, so a runtime AttributeError on a non-
                # categorical feature is the correct behavior.
                feat = self._features[fname]
                cindex, csize = feat.category_index(i)  # type: ignore[attr-defined]
                multi_index.append(cindex)
                dims.append(csize)

            cci = int(np.ravel_multi_index(multi_index, dims))
            covariate_class[i] = cci
            r[cci] = r.get(cci, 0) + 1

        num_cc_with_replicates = 0
        max_replication = 0
        replication_dof = 0
        for j in r.values():
            if j > 1:
                num_cc_with_replicates += 1
                replication_dof += j - 1
            if j > max_replication:
                max_replication = j

        if (
            num_cc_with_replicates >= min_cc_replicates
            and max_replication >= min_replication
        ):
            has_replication = True
        else:
            has_replication = False

        if (
            num_cc_with_replicates >= des_cc_replicates
            and max_replication >= des_replication
            and replication_dof >= des_replication_dof
        ):
            has_desired_replication = True
        else:
            has_desired_replication = False

        if formula is None:
            if has_desired_replication:
                formula = "replication"
            else:
                formula = "pearson"

        if has_replication and formula == "replication":
            trials: dict[int, float] = {}
            successes: dict[int, float] = {}
            for i in range(self._num_obs):
                cci = int(covariate_class[i])
                trials[cci] = trials.get(cci, 0.0) + float(
                    self._covariate_class_sizes[i]
                )
                successes[cci] = successes.get(cci, 0.0) + float(self._y[i])

            s2 = 0.0
            for i in range(self._num_obs):
                cci = int(covariate_class[i])
                pi = successes[cci] / trials[cci]
                num = self._y[i] - self._covariate_class_sizes[i] * pi
                denom = self._covariate_class_sizes[i] * pi * (1 - pi)
                s2 += float(num * num / denom)

            s2 /= replication_dof
            self._known_dispersion = True
            self._dispersion = s2
            return s2
        else:
            mu = self._eval_inv_link(self._num_features * self.f_bar)
            m_arr = self._covariate_class_sizes
            bl_var = np.multiply(mu, 1.0 - mu)
            res = self._y - np.multiply(m_arr, mu)
            num_p = np.multiply(res, res)
            denom_p = np.multiply(m_arr, bl_var)
            n_minus_p = self._num_obs - self.dof()
            s2 = float(np.sum(np.divide(num_p, denom_p)) / n_minus_p)
            self._known_dispersion = True
            self._dispersion = s2
            return s2

    def _poisson_overdispersion(self) -> float:
        r"""Estimate the Poisson dispersion as the Pearson chi-square / dof.

        Under the standard Poisson model ``Var(Y) = mu``; if observed
        variance is larger (overdispersion) the dispersion ``phi`` is
        estimated as

            phi = sum_i (y_i - mu_i)^2 / mu_i  /  (n - p)

        where ``p = dof()``. This is the Pearson form of Eqn 3.10 in
        Wood (2017). The replication-based estimator used for
        ``_binomial_overdispersion`` is binomial-specific (it relies on
        covariate-class replicates) and is not applicable here.

        The result is cached on ``self._dispersion`` and
        ``self._known_dispersion`` so subsequent calls are O(1).
        """
        mu = self._eval_inv_link(self._num_features * self.f_bar)
        # Guard against transient mu <= 0 from a non-canonical link.
        tiny = np.finfo(float).tiny
        mu_safe = np.where(mu > 0, mu, tiny)
        res = self._y - mu
        n_minus_p = self._num_obs - self.dof()
        s2 = float(np.sum(res * res / mu_safe) / n_minus_p)
        self._known_dispersion = True
        self._dispersion = s2
        return s2

    def _binomial_pearson_dispersion(self) -> float:
        r"""Estimate the binomial dispersion as the Pearson chi-square / dof.

        Used by ``family='quasi_binomial'`` and matches the Pearson
        branch of ``_binomial_overdispersion`` but works for both
        Bernoulli (no covariate classes) and binomial-counts (with
        covariate classes) data:

            phi = sum_i (y_i - m_i mu_i)^2 / (m_i mu_i (1 - mu_i))
                  / (n - p)

        For Bernoulli data ``m_i = 1``. For an exact-binomial fit
        ``phi`` should be near 1; values markedly larger than 1
        indicate overdispersion. The estimator is undefined when
        ``mu_i in {0, 1}``; we clip ``mu_i`` away from those endpoints
        so transient near-saturation during the ADMM loop does not
        blow up the denominator.
        """
        mu = self._eval_inv_link(self._num_features * self.f_bar)
        eps = np.finfo(float).eps
        mu_c = np.clip(mu, eps, 1.0 - eps)
        if self._has_covariate_classes:
            m_arr = self._covariate_class_sizes
            res = self._y - m_arr * mu_c
            denom = m_arr * mu_c * (1.0 - mu_c)
        else:
            res = self._y - mu_c
            denom = mu_c * (1.0 - mu_c)
        n_minus_p = self._num_obs - self.dof()
        s2 = float(np.sum(res * res / denom) / n_minus_p)
        self._known_dispersion = True
        self._dispersion = s2
        return s2

    def dof(self) -> float:
        """Degrees of freedom: sum of feature DOFs plus the affine intercept."""
        dof = 1.0  # Affine factor
        for _, feature in self._features.items():
            dof += feature.dof()
        return dof

    def aic(self) -> float:
        """Akaike Information Criterion.

        Useful for choosing smoothing parameters. The AIC computed
        here is off by a constant factor, which simplifies the
        computation without affecting model-selection rankings.

        Different authors throw in multiplicative or additive factors
        willy-nilly since they do not affect model selection.
        """
        p = self.dof()
        if not self._known_dispersion:
            # If we are estimating the dispersion, we need to
            # add one to the DOF.
            p += 1

        # Note that the deviance is twice the dispersion times the
        # log-likelihood, so no factor of two required there.
        return self.deviance() / self.dispersion() + 2.0 * p
        # return (self.deviance() / self._num_obs
        #          + 2. * p * self.dispersion() / self._num_obs)

    def aicc(self) -> float:
        """AIC corrected for finite sample size.

        Applies the Hurvich-Tsai small-sample correction to ``aic()``:

            AICc = AIC + 2 * p * (p + 1) / (n - p - 1)

        where ``p`` is the effective number of parameters
        (``dof()``, plus one if the dispersion is being estimated)
        and ``n`` is the number of observations. This is Eqn 6.32
        on p. 304 of [GAMr]. As ``n`` grows the correction term
        vanishes and AICc reduces to AIC. AICc is preferable to AIC
        when ``n`` is small relative to ``p``.

        Returns ``+inf`` for an overparameterized fit
        (``n <= p + 1``), where the correction term is undefined
        and the model has no useful AICc.
        """
        p = self.dof()
        if not self._known_dispersion:
            # Match the convention in aic(): one extra parameter for
            # the estimated dispersion.
            p += 1
        n = self._num_obs
        if n - p - 1 <= 0:
            return float("inf")
        return self.aic() + 2.0 * p * (p + 1) / (n - p - 1)

    def bic(self) -> float:
        """Bayesian Information Criterion.

        Computed as

            BIC = deviance / dispersion + log(n) * p

        where ``p`` is the effective parameter count (``dof()``, plus
        one when dispersion is estimated, matching ``aic()`` and
        ``aicc()``) and ``n`` is the number of observations. As with
        ``aic()``, the BIC is off by the same constant factor; only
        differences are meaningful for model selection.
        """
        p = self.dof()
        if not self._known_dispersion:
            p += 1
        return self.deviance() / self.dispersion() + float(np.log(self._num_obs)) * p

    def null_deviance(self) -> float:
        """Deviance of the intercept-only model on the training data.

        The null model predicts the same mean response for every
        observation: ``mean(y)`` (or ``sum(y) / sum(covariate_class_sizes)``
        when binomial covariate classes were supplied at fit time).
        """
        y = self._y
        if self._has_covariate_classes:
            m: npt.NDArray[Any] | float = self._covariate_class_sizes
            mean_response = float(np.sum(y)) / float(np.sum(m))
        else:
            m = 1.0
            mean_response = float(np.mean(y))
        mu = np.full_like(y, mean_response, dtype=float)
        return self._deviance_from_mu(y, mu, m=m, w=self._weights)

    def r_squared(self) -> float:
        """Deviance-based pseudo-R^2.

        Returns ``1 - deviance() / null_deviance()``. For a normal
        family with identity link this is the conventional R^2; for
        other families it is the deviance pseudo-R^2 of McCullagh and
        Nelder. Returns ``nan`` when the null deviance is zero (the
        response is constant), where pseudo-R^2 is undefined.
        """
        null_dev = self.null_deviance()
        if null_dev == 0.0:
            return float("nan")
        return 1.0 - self.deviance() / null_dev

    def ubre(self, gamma: float = 1.0) -> float:
        """Un-Biased Risk Estimator.

        Returns the Un-Biased Risk Estimator as discussed in Sections
        6.2.1 and 6.2.5 of [GAMr]_. Useful for choosing the smoothing
        parameter when the dispersion is known.

        As discussed in Section 6.2.5 of [GAMr]_, sometimes it is
        helpful to force smoother fits by exaggerating the effective
        degrees of freedom; a value of ``gamma > 1`` may be desirable
        in that case.
        """
        return self.deviance() + 2.0 * gamma * self.dispersion() * self.dof()

    def gcv(self, gamma: float = 1.0) -> float:
        """Generalized Cross Validation score.

        Useful for choosing the smoothing parameter when the
        dispersion is unknown.

        As discussed in Section 6.2.5 of [GAMr]_, sometimes it is
        helpful to force smoother fits by exaggerating the effective
        degrees of freedom; a value of ``gamma > 1`` may be desirable
        in that case.
        """
        denom = self._num_obs - gamma * self.dof()
        return self._num_obs * self.deviance() / (denom * denom)

    def summary(self) -> None:
        """Print summary statistics for the fitted model.

        Prints statistics for the overall model and for each
        individual feature (see the ``__str__`` method on each feature
        class for the per-feature output).

        The overall model section reports:

        ============  =====================================================
        ``phi``       Estimated dispersion parameter. Omitted when
                      dispersion was supplied at construction time or is
                      fixed by the family (e.g. Poisson).
        ``edof``      Estimated degrees of freedom.
        ``Deviance``  Twice the dispersion times the difference between
                      the log-likelihood of the saturated model and that
                      of the fitted model.
        ``AIC``       Akaike Information Criterion.
        ``AICc``      AIC corrected for finite data sets.
        ``BIC``       Bayesian Information Criterion.
        ``R^2``       Deviance-based pseudo-:math:`R^2`.
        ``UBRE``      Unbiased Risk Estimator (when dispersion is known).
        ``GCV``       Generalized Cross Validation (when dispersion is
                      estimated).
        ============  =====================================================

        See the corresponding methods (:meth:`aic`, :meth:`aicc`,
        :meth:`bic`, :meth:`r_squared`, :meth:`ubre`, :meth:`gcv`)
        for definitions.
        """

        print("Model Statistics")
        print("----------------")
        if not self._known_dispersion:
            print(f"phi: {self.dispersion():0.06g}")
        print(f"edof: {self.dof():0.0f}")
        print(f"Deviance: {self.deviance():0.06g}")
        print(f"AIC: {self.aic():0.06g}")
        print(f"AICc: {self.aicc():0.06g}")
        print(f"BIC: {self.bic():0.06g}")
        print(f"R^2: {self.r_squared():0.06g}")

        if self._known_dispersion:
            print(f"UBRE: {self.ubre():0.06g}")
        else:
            print(f"GCV: {self.gcv():0.06g}")

        print()
        print("Features")
        print("--------")

        for _name, feature in self._features.items():
            print(str(feature))


def fit_adaptive_lasso(
    gam: GAM,
    X: pd.DataFrame,
    y: npt.NDArray[Any],
    *,
    gamma: float = 1.0,
    eps: float = 1e-6,
    **fit_kwargs: Any,
) -> GAM:
    """Fit ``gam`` via the two-stage adaptive lasso (Zou, 2006).

    The first stage fits ``gam`` to ``(X, y)`` using whatever L1
    regularization the user configured on each feature -- producing a
    pilot estimate of every coefficient. The second stage rewrites each
    L1-regularized feature's per-coefficient L1 weight to ``base /
    (|pilot| + eps) ** gamma`` and refits. Coefficients with large pilot
    magnitudes are penalized less; tiny pilots (likely noise) are
    penalized more, recovering the "oracle" sparsity pattern with less
    bias on the truly active coefficients than plain L1.

    Both stages are convex (they are ordinary weighted-L1 fits at fixed
    weights), so this is just a thin wrapper over two ``GAM.fit`` calls.

    Parameters
    ----------
    gam : GAM
        Fully configured but not-yet-fit GAM. At least one feature must
        carry an ``regularization={"l1": {"coef": ...}}`` configuration;
        other features and other regularizers (l2, group_lasso, ...)
        ride along untouched. The same ``gam`` object is mutated in
        place and returned.
    X, y : data
        Forwarded to both ``gam.fit`` calls.
    gamma : float
        Adaptive-lasso exponent (typically 1.0). Larger ``gamma`` makes
        the second-stage weights swing harder around the pilot
        magnitudes.
    eps : float
        Stabilizer added to the pilot magnitude before exponentiation,
        so a pilot of exactly zero produces a finite (though large)
        second-stage weight rather than a divide-by-zero.
    **fit_kwargs
        Forwarded to both ``gam.fit`` calls. Both stages see the same
        ``smoothing``, ``max_its``, ``weights``, etc.

    Returns
    -------
    gam : GAM
        The same model, now fit with adaptive L1 weights.
    """
    if gamma <= 0.0:
        raise ValueError(f"gamma must be positive; got {gamma}.")
    if eps <= 0.0:
        raise ValueError(f"eps must be positive; got {eps}.")

    has_l1 = any(getattr(f, "_has_l1", False) for f in gam._features.values())
    if not has_l1:
        raise ValueError(
            "fit_adaptive_lasso requires at least one feature with "
            'regularization={"l1": ...}; without an L1 term to reweight, '
            "the second stage is identical to the first."
        )

    gam.fit(X, y, **fit_kwargs)

    rewrote_any = False
    for feature in gam._features.values():
        apply = getattr(feature, "_apply_adaptive_l1", None)
        if apply is None:
            continue
        if apply(gamma, eps):
            rewrote_any = True
    if not rewrote_any:
        # Defensive: should be impossible given the has_l1 guard above,
        # but feature subclasses without `_apply_adaptive_l1` would land
        # here. Skip the second fit rather than producing identical output.
        return gam

    gam.fit(X, y, **fit_kwargs)
    return gam
