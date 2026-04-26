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
    "quantile": "identity",
    "huber": "identity",
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
    """Plot convergence progress.

    We deem the algorithm to have converged when the prime and dual
    residuals are smaller than tolerances which are themselves computed
    based on the data as in [ADMM]. Some analysts prefer to claim
    convergence when changes to the deviance (a measure of goodness of
    fit). Thus we plot that as well. Specifically, we plot, on a log
    scale, dev - dev_final, where dev_final is the deviance of the final
    model. We add 1e-10 just to avoid taking the logarithm of zero, which
    is completely arbitrary but makes the plot look acceptable.

    Parameters
    ----------
     prim_res : array
         Array of prime residuals after each iteration.
     prim_tol : array
         Array of prime tolerances after each iteration.
     dual_res : array
         Array of dual residuals after each iteration.
     dual_tol : array
         Array of dual tolerances after each iteration.
     dev : array
         Array of deviances after each iteration

    Returns
    -------
     (nothing)
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
        """Generalized Additive Model

        This is the constructor for a Generalized Additive Model.

        References
        ----------
         [glmnet]   glmnet (R package):
                    https://cran.r-project.org/web/packages/glmnet/index.html
                    This is the standard package for GAMs in R and was written by people
                    much smarter than I am!
         [pygam]    pygam (Python package): https://github.com/dswah/pyGAM
                    This is a library in Python that does basically the same thing as this
                    script, but in a different way (not using ADMM).
         [GLM]      Generalized Linear Models by McCullagh and Nelder
                    The standard text on GLMs.
         [GAM]      Generalized Additive Models; by Hastie and Tibshirani
                    The book by the folks who invented GAMs.
         [ESL]      The Elements of Statistical Learning; by Hastie, Tibshirani, and
                    Friedman. Covers a lot more than just GAMs.
         [GAMr]     Generalized Additive Models: an Introduction with R; by Wood.
                    Covers more implementation details than [GAM].
         [ADMM]     Distributed Optimization and Statistical Learning via the Alternating
                    Direction Method of Multipliers; by Boyd, Parikh, Chu, Peleato, and
                    Eckstein. A mouthful, a work of genius.
         [GAMADMM]  A Distributed Algorithm for Fitting Generalized Additive Models;
                    by Chu, Keshavarz, and Boyd
                    Forms the basis of our approach, the inspiration for this package!

        Parameters
        ----------
         family : str or None (default None)
             Family of the model. Currently supported families include:
                'normal' (for continuous responses),
                'binomial' (for binary responses),
                'poisson' (for counts),
                'gamma' (still in progress),
                'inverse_gaussian' (still in progress),
                'quantile' (pinball loss; requires ``tau``),
                'huber' (robust M-estimator; requires ``delta``).
             Not currently supported families that could be supported
             include Multinomial models (ordinal and nominal) and
             proportional hazards models. Required unless loading an
             existing model from file (see load_from_file).
         link : str or None (optional)
             Link function associated with the model. Supported link
             functions include:
                     Link                Canonical For Family
                'identity'                  'normal'
                'logistic'                  'binomial'
                'log'                       'poisson'
                'reciprocal'                'gamma'
                'reciprocal_squared'        'inverse_gaussian'
             Other links worth supporting include probit, log-log
             and complementary log-log link functions. If not
             specified, the canonical link will be used, but non-
             canonical links are still permitted. Certain link/family
             combinations result in a non-convex problem and
             convergence is not guaranteed.
         dispersion : float or None (optional)
             Dispersion parameter associated with the model. Certain
             families (binomial, poisson) have dispersion independent
             of the data. Specifying the dispersion for these families
             does nothing. In other instances, the dispersion is
             typically unknown and must be estimated from the data.
             If the dispersion is known, it can be specified here which
             will reduce the uncertainty of the model.
         estimate_overdispersion : boolean (optional)
             Flag specifying whether to estimate over-dispersion for
             Binomial and Poisson (not yet implemented) families. Is
             only possible when covariate classes are present and have
             at least modest size. See [GLM, S4.5] for
             details. Defaults to False.
         name : str or None (optional)
             Name for model, to be used in plots and in saving files.
         load_from_file : str or None (optional)
             This module uses an iterative approach to fitting models.
             For complicated models with lots of data, each iteration
             can take a long time (though the number of iterations is
             typically less than 100). If the user wishes to pause
             after the end of an iteration, they can pick up where
             the left off by saving results (see the save_flag in .fit)
             and loading them to start the next iterations. Specifying
             this option supercedes all other parameters.
         tau : float or None (optional)
             Quantile level in (0, 1) for ``family='quantile'`` (pinball
             loss). ``tau=0.5`` recovers the conditional median.
             Required when ``family='quantile'`` and ignored otherwise.
         delta : float or None (optional)
             Knee parameter for ``family='huber'``. Residuals with
             ``|y - mu| <= delta`` are penalized as ``0.5 * r^2`` (least
             squares); larger residuals are penalized linearly, capping
             their per-observation influence. Must be positive and is in
             the units of ``y``. Required when ``family='huber'`` and
             ignored otherwise.

        Returns
        -------
         mdl : Generalized Additive Model object

        """

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
    ) -> None:
        """Add a feature

        Add a feature to a Generalized Additive Model. (An implicit
        constant feature is always included, representing the overall
        average response.)

        Parameters
        ----------
         name : str
             Name for feature. Used internally to keep track of
             features and is also used when saving files and in
             plots.
         type : str
             Type of feature. Currently supported options include:
               'categorical' (for categorical variables)
               'linear' (for variables with a linear contribution
                         to the response)
               'spline' (for variables with a potentially nonlinear
                         contribution to the response).
             Other types of features worth supporting include
             piecewise constant functions and monotonic functions.
             Those might end up being regularization terms.
         transform : function or None
             Optional transform applied to feature data, saving
             the user from repetitive boilerplate code. Any function
             may be used; it is applied to data provided during fitting
             and prediction. Common options might include np.log, np.log1p,
             or np.sqrt. The user may wish to start with a base feature
             like 'age' and use derived features 'age_linear', 'age_quadratic'
             to permit quadratic models for that feature, with potentially
             different regularization applied to each.
         rel_dof : float or None
             Relative degrees of freedom. Applicable only to spline features.
             The degrees of freedom associated with a spline represent how
             "wiggly" it is allowed to be. A spline with two degrees of freedom
             is just a line. (Actually, since these features are constrained
             to have zero mean response over the data, linear features
             only have one degree of freedom.) The relative degrees of freedom
             are used to specify the baseline smoothing parameter (lambda)
             associated with a feature. When the model is fit to data, the user
             can specify an overall smoothing parameter applied to all features
             to alter the amount of regularization in the entire model. Thus
             the actual degrees of freedom will vary based on the amount of
             smoothing. The idea is that the analyst may wish to permit some
             features to be more wiggly than others. By default, all
             splines have 4 relative degrees of freedom.

             Regularization of any feature effectively reduces the degrees of
             freedom, and so this term is potentially applicable, but that is
             not yet supported.
        regularization : dictionary or None
             Dictionary specifying the regularization applied to this feature.
             Different types of features support different types of regularization.
             Splines always include a C2 smoothness penalty controlled via
             ``rel_dof``; ``regularization={"group_lasso": {"coef": λ}}``
             additionally shrinks the entire spline contribution and can
             zero it out. Other features have more diverse options
             described in their own documentation.

        Returns
        -------
         (nothing)

        """
        f: _Feature
        if type == "categorical":
            f = _CategoricalFeature(name, regularization=regularization)
        elif type == "linear":
            f = _LinearFeature(name, transform, regularization=regularization)
        elif type == "spline":
            f = _SplineFeature(
                name,
                transform,
                rel_dof if rel_dof is not None else 4.0,
                regularization=regularization,
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
        """Fit a Generalized Additive Model to data.

        Note regarding binomial families: many data sets include
        multiple observations having identical features. For example,
        imagine a data set with features 'gender', and 'country' and
        binary response indicating whether the person died (morbid but
        common in biostatistics). The data might look like this:

           gender   country   patients   survivors
             M        USA       50           48
             F        USA       70           65
             M        CAN       40           38
             F        CAN       45           43

        This still describes a binomial family, but in a more compact
        format than specifying each individual user. We eventually
        want to support this more compact format, but we do not
        currently! In this context, it is important to check for
        over-dispersion (see [GLM]), and I need to learn more first.
        In the current implementation, we assume that there is no
        over-dispersion, and that the number of users having the
        same set of features is small.

        Parameters
        ----------
         X : pandas dataframe
             Dataframe of features. The column names must correspond
             to the names of features added to the model. X may have
             extra columns corresponding to features not included in
             the model; these are simply ignored. Where applicable,
             the data should be "pre-transformation", since this code
             will apply any transformations specified in .add_feature.
         y : array
             Response. Depending on the model family, the response
             may need to be in a particular form (for example, for
             a binomial family, the y's should be either 0 or 1),
             but this is not checked anywhere!
         covariate_class_sizes : array or None.
             If observations are grouped into covariance classes, the
             size of those classes should be listed in this input.
         w : array
             Weights applied to each observation. This is effectively
             specifying the dispersion of each observation.
         optimizer : string
             We use the Alternating Direction Method of Multipliers
             ('admm') to fit the model. We may eventually support more
             methods, but right now this option does nothing.
         smoothing : float
             Smoothing to apply to entire model, used in conjunction
             with other regularization parameters. That is, whatever
             regularization is used for the various features, is
             scaled by this term, allowing the user to set the overall
             smoothing by Cross Validation or whatever they like. This
             allows the user to specify different regularization for
             each feature, while still permitting a one-dimensional
             family of models corresponding to different amounts of
             regularization. Defaults to 1., leaving the regularization
             as specified in .add_feature().
         save_flag : boolean
             Specifies whether to save intermediate results after each
             iteration. Useful for complicated models with massive
             data sets that take a while to fit. If the system crashes
             during the fit, the analyst can pick up where they left
             off instead of starting from scratch. Defaults to False.
         verbose : boolean
             Specifies whether to print mildly useful information to
             the screen during the fit. Defaults to False.
         plot_convergence : boolean
             Specifies whether to plot the convergence graph at the
             end. (I suspect only Convex Optimization nerds like me
             want to see this.) Defaults to False.
         max_its : integer
             Maximum number of iterations. Defaults to 100.
         n_jobs : integer
             Number of threads to use for the per-feature primal step
             within each ADMM iteration. Defaults to 1 (serial); pass
             ``-1`` to use ``os.cpu_count()``. NumPy / SciPy / cvxpy
             release the GIL during their numeric kernels, so threading
             produces real speedup. Expect a 2-4x ceiling on models
             with several non-trivial features (splines, categoricals
             via cvxpy); pure linear-only models are usually faster
             serial because the per-feature work is too cheap to
             amortize the thread-dispatch overhead.

        Returns
        -------
         (nothing)

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
        r"""Optimize \bar{z}.

        Solves the optimization problem:
           minimize L(N*z) + \rho/2 * \| N*z - N*u - N*\bar{f} \|_2^2
        where z is the variable, N is the number of features, u is the scaled
        dual variable, \bar{f} is the average feature response, and L is
        the likelihood function which is different depending on the
        family and link function. This is accomplished via a proximal
        operator, as discussed in [GAMADMM]:
          prox_\mu(v) := argmin_x L(x) + \mu/2 * \| x - v \|_2^2
        I strongly believe that paper contains a typo in this equation, so we
        return (1. / N) * prox_\mu (N * (u + \bar{f}) with \mu = \rho instead
        of \mu = \rho / N as in [GAMADMM]. When implemented as in the paper,
        convergence was much slower, but it did still converge.

        Certain combinations of family and link function result in proximal
        operators with closed form solutions, making this step *very* fast
        (e.g. 3 flops per observation).

        Parameters
        ----------
         upf : array
             Vector representing u + \bar{f}
         N : integer
             Number of features.
         p : Multiprocessing Pool (optional)
             If multiple threads are available, massive data sets may
             benefit from solving this optimization problem in parallel.
             It is up to the individual functions to decide whether to
             actually do this.

        Returns
        -------
         z : array
             Result of the above optimization problem.
        """

        # ``__init__`` already restricted (family, link) to the
        # SUPPORTED_FAMILY_LINK_PAIRS allow-list, so every branch below
        # has a dedicated convex prox; there is no fallback path.
        # Different prox operators have slightly different signatures
        # (the binomial variant takes a covariate-class-sizes argument,
        # quantile / huber take an extra shape parameter), so the
        # dispatch is typed as Any to keep mypy quiet here.
        prox: Any
        if self._family == "normal":
            prox = po._prox_normal_identity
        elif self._family == "binomial":
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
        elif self._family == "poisson":
            prox = po._prox_poisson_log
        elif self._family == "gamma":
            prox = po._prox_gamma_reciprocal
        elif self._family == "inverse_gaussian":
            prox = po._prox_inv_gaussian_reciprocal_squared
        elif self._family == "quantile":
            return (1.0 / N) * po._prox_quantile_identity(
                N * upf,
                self._rho,
                self._y,
                self._tau,  # type: ignore[arg-type]
                w=self._weights,
            )
        elif self._family == "huber":
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
        """Apply fitted model to features.

        Parameters
        ----------
         X : pandas dataframe
             Data for which we wish to predict the response. The
             column names must correspond to the names of the
             features used to fit the model. X may have extra
             columns corresponding to features not in the model;
             these are simply ignored. Where applicable, the data
             should be "pre-transformation", since this code will
             apply any transformations specified while defining
             the model.

        Returns
        -------
         mu : array
             Predicted mean response for each data point.

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
        if self._family == "normal":
            v = np.ones_like(mu, dtype=float)
        elif self._family == "binomial":
            eps = np.finfo(float).eps
            mu_c = np.clip(mu, eps, 1.0 - eps)
            v = mu_c * (1.0 - mu_c)
        elif self._family == "poisson":
            v = mu
        elif self._family == "gamma":
            v = mu * mu
        elif self._family == "inverse_gaussian":
            v = mu * mu * mu
        else:
            raise NotImplementedError(
                f"robust_covariance does not yet support the {self._family!r} "
                "family. Supported: normal, binomial, poisson, gamma, "
                "inverse_gaussian."
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

        NOT YET IMPLEMENTED

        There are two notions of confidence intervals that are
        appropriate. The first is a confidence interval on mu,
        the mean response. This follows from the uncertainty
        associated with the fit model. The second is a confidence
        interval on observations of this model. The distinction
        is best understood by example. For a Gaussian family,
        the model might be a perfect fit to the data, and we
        may have billions of observations, so we know mu perfectly.
        Confidence intervals on the mean response would be very
        small. But the response is Gaussian with a non-zero
        variance, so observations will in general still be spread
        around the mean response. A confidence interval on the
        prediction would be larger.

        Now consider a binomial family. The estimated mean response
        will be some number between 0 and 1, and we can estimate
        a confidence interval for that mean. But the observed
        response is always either 0 or 1, so it doesn't make sense
        to talk about a confidence interval on the prediction
        (except in some pedantic sense perhaps).

        Note that if we are making multiple predictions, it makes
        sense to talk about a "global" set of confidence intervals.
        Such a set has the property that *all* predictions fall
        within their intervals with specified probability. This
        function does not compute global confidence intervals!
        Instead each confidence interval is computed "in vacuo".

        Parameters
        ----------
         X : pandas dataframe
             Data for which we wish to predict the response. The
             column names must correspond to the names of the
             features used to fit the model. X may have extra
             columns corresponding to features not in the model;
             these are simply ignored. Where applicable, the data
             should be "pre-transformation", since this code will
             apply any transformations specified while defining
             the model.
         prediction : boolean
             Specifies whether to return a confidence interval
             on the mean response or on the predicted response.
             (See above.) Defaults to False, leading to a
             confidence interval on the mean response.
         width : float between 0 and 1
             Desired confidence width. Defaults to 0.95.

        Returns
        -------
         mu : (n x 2) array
             Lower and upper bounds on the confidence interval
             associated with each prediction.
        """
        raise NotImplementedError("confidence_intervals is not yet implemented")

    def plot(
        self,
        name: str,
        true_fn: Callable[[FloatArray], FloatArray] | None = None,
    ) -> None:
        """Plot the component of the modelf for a particular feature.

        Parameters
        ----------
         name : str
             Name of feature (must be a feature in the model).
         true_fn : function or None (optional)
             Function representing the "true" relationship
             between the feature and the response.

        Returns
        -------
         (nothing)

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
            if self._family == "binomial":
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
        if self._family == "binomial":
            eps = np.finfo(float).eps
            mu_c = np.clip(mu, eps, 1.0 - eps)
            var = m * mu_c * (1.0 - mu_c)
            return np.asarray((y - m * mu_c) / np.sqrt(var * phi), dtype=float)
        if self._family == "normal":
            return np.asarray((y - mu) / np.sqrt(phi), dtype=float)
        if self._family == "poisson":
            return np.asarray((y - mu) / np.sqrt(mu * phi), dtype=float)
        if self._family == "gamma":
            return np.asarray((y - mu) / np.sqrt(mu * mu * phi), dtype=float)
        if self._family == "inverse_gaussian":
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
        if self._family == "normal":
            return np.asarray((y - mu) ** 2, dtype=float)
        if self._family == "binomial":
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
        if self._family == "poisson":
            with np.errstate(divide="ignore", invalid="ignore"):
                t = np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / mu), 0.0)
            return np.maximum(2.0 * (t - (y - mu)), 0.0)
        if self._family == "gamma":
            tiny = np.finfo(float).tiny
            y_safe = np.where(y > 0, y, tiny)
            mu_safe = np.where(mu > 0, mu, tiny)
            return np.maximum(
                2.0 * (-np.log(y_safe / mu_safe) + (y - mu_safe) / mu_safe), 0.0
            )
        if self._family == "inverse_gaussian":
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
        if self._family == "normal":
            # V(mu) = 1, A(t) = t -- coincides with the Pearson residual.
            return np.asarray((y - mu) / np.sqrt(phi), dtype=float)
        if self._family == "binomial":
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
        if self._family == "poisson":
            # V(mu) = mu, A(t) = (3/2) t^(2/3).
            return np.asarray(
                1.5
                * (y ** (2.0 / 3.0) - mu ** (2.0 / 3.0))
                / mu ** (1.0 / 6.0)
                / np.sqrt(phi),
                dtype=float,
            )
        if self._family == "gamma":
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
        if self._family == "inverse_gaussian":
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
        r"""Deviance

        This function works in one of two ways:

        Firstly, it computes the deviance of the model, defined as
           2 * \phi * (\ell(y; y) - \ell(\mu; y))
        where \phi is the dispersion (which is only in this equation
        to cancel out the denominator of the log-likelihood),
        \ell(y; y) is the log-likelihood of the model that fits the
        data perfectly, and \ell(\mu; y) is the log-likelihood of the
        fitted model on the data used to fit the model. This is
        the quantity we minimize when fitting the model.

        Secondly, it computes the deviance of the model on arbitrary
        data sets. This can be used in conjunction with Cross Validation
        to choose the smoothing parameter by minimizing the deviance
        on the hold-out set.

        Parameters
        ----------
         X : pandas dataframe (optional)
             Dataframe of features. The column names must correspond
             to the names of features added to the model. (See .predict()).
             Only applicable for the second use case described above.
         y : array (optional)
             Response. Only applicable for the second use case.
         covariate_class_sizes : array (optional)
             Array of covariate class sizes.
         w : array (optional)
             Weights for observations. Only applicable for the second
             use case, but optional even then.

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
        if self._family == "normal":
            y_minus_mu = y - mu
            if w is None:
                return float(y_minus_mu.dot(y_minus_mu))
            return float(w.dot(y_minus_mu * y_minus_mu))
        if self._family == "binomial":
            # Clip mu away from 0 and 1 so log(mu) and log1p(-mu) stay finite.
            eps = np.finfo(float).eps
            mu_c = np.clip(mu, eps, 1.0 - eps)
            if w is None:
                return float(
                    -2.0 * np.sum(y * np.log(mu_c) + (m - y) * np.log1p(-mu_c))
                )
            return float(-2.0 * w.dot(y * np.log(mu_c) + (m - y) * np.log1p(-mu_c)))
        if self._family == "poisson":
            # `y * log(y / mu)` has the conventional limit 0 when y = 0.
            y_log_term = np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / mu), 0.0)
            term = y_log_term - (y - mu)
            if w is None:
                return float(2.0 * np.sum(term))
            return float(2.0 * w.dot(term))
        if self._family == "gamma":
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
        if self._family == "inverse_gaussian":
            if w is None:
                return float(np.sum((y - mu) * (y - mu) / (mu * mu * y)))
            return float(w.dot((y - mu) * (y - mu) / (mu * mu * y)))
        if self._family == "quantile":
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
        if self._family == "huber":
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
        """Dispersion

        Returns the dispersion associated with the model. Depending on
        the model family and whether the dispersion was specified by
        the user, the dispersion may or may not be known a
        priori. This function will estimate this parameter when
        appropriate.

        There are different ways of estimating this parameter that may
        be appropriate for different kinds of families. The current
        implementation is based on the deviance, as in Eqn 3.10 on
        p. 110 of GAMr. As discussed in that section, this tends not
        to work well for Poisson data (with overdispersion) when the
        mean response is small. Alternatives are offered in that
        section, but I have not yet implemented them. This is not
        terribly relevant for the current implementation since
        overdispersion is not supported! (When overdispersion is not
        present, the dispersion of the Poisson is exactly 1.)

        My eventual hope is to understand the appropriate methods for
        all the different circumstances and have intelligent defaults
        that can be overridden by opinionated users.

        Parameters
        ----------
         formula : str
             Formula for the dispersion. Options include:
                'deviance' (default)
                'pearson'
                'fletcher'

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
        r"""Over-Dispersion

        Parameters
        ----------
         formula : str
            Which formula to use, either 'replication' or
            'pearson'. See Notes.

        Returns
        -------
         sigma2 : float
            Estimate of over-dispersion. This is also saved as the
            self._dispersion parameter so we only calculate this once
            regardless of how many times this function is called.

        Notes
        -----
        When using covariate classes, the observed variance may exceed
        the baseline for the family due to clustering in the
        population. See GLM for motivation. That text gives two
        methodologies for estimating over-dispersion. When there are
        no covariate classes (multiple observations with identical
        features), estimating over-dispersion is not possible.

        The most reliable assessment of over-dispersion is only
        possible when there is replication amongst the covariate
        classes. This is best illustrated through example. Suppose we
        have data on patients from two hospitals as shown in the table
        below. Note that there are 3 rows corresponding to Men in
        hospital 1. These entries could of course be pooled to give
        the total patients and survivors for this covariate class, but
        because they have not, it permits us to estimate
        over-dispersion more reliably.

        Gender Hospital Patients Survivors
          M       1       30        15
          M       1       40        19
          M       1       35        15
          F       1       10         8
          M       2       10         3
          M       2       18         6
          F       2       40        30

        Because we are building a model based on gender and hospital
        alone, we are assuming that all three entries are drawn from
        the same binomial distribution. We could actually test that
        hypothesis using, for example, Welch's t-Test. If the result
        indicates a significant departure from the null hypothesis,
        there must be some (unobserved) explanation for different
        survival rates. Perhaps the repeated entries correspond to
        different doctors, with some doctors being more effective than
        others. Or perhaps the multiple entries refer to different
        time periods, like before and after a new treatment was
        instituted. Regardless, we can quantify the additional
        variance and use it to make (hopefully) more accurate
        confidence intervals.

        When replication is present, we take the following approach,
        per GLM. Suppose a particular covariate class (e.g. Gender=M,
        Hospital=1) has r replicates. Across all r replicates,
        determine the observed success rate, pi. In our example, we
        have 105 patients and 49 survivors, for a total survival rate
        of pi = 0.47. Next we compute the variance on r-1 DOF:

                  1    r  (y_j - m_j * pi)^2
           s^2 = --- \sum ------------------
                 r-1  j=1  m_j pi * (1 - pi)

        where y_j is the number of successes in the jth replicate, m_j
        is the number of trials in the jth replicate, and s^2 is
        estimated variance. Per GLM, this is an unbiased estimate of
        the dispersion parameter. Filling in our specific numbers, we
        get s^2 = 0.17, indicating under-dispersion. (Important note:
        these are made up numbers, so there is actually more
        consistency in the data than would be exhibited from a true
        binomial model. Over-dispersion is more common than
        under-dispersion.)

        Each covariate class with replication can be used to derive an
        estimate of the dispersion parameter. If we expect the
        dispersion to be independent of the covariate classes (which
        may or may not be true), we can pool these estimates, weighted
        by the degree of replication. If the kth covariate class has
        r_k replicates and dispersion estimate s_k^2, the overall
        estimate of dispersion is:

                  \sum_k (r_k - 1) * s_k^2
           s^2 = -------------------------
                     \sum_k (r_k - 1)

        Another important note: the above formula is *not* present in
        GLM. That text just says to pool the estimates, but does not
        specify how. This approach makes sense to me, but that doesn't
        make it correct!

        When replication is not present, or even if the degree of
        replication is small, the above methodology breaks
        down. Instead, GLM advocates the use of a Pearson-residual
        based approach. If pi_j is the model prediction for the jth
        covariate class, then we estimate dispersion as:

                   1          (y_j - m_j * pi_j)^2
           s^2 = ----- \sum -----------------------
                 n - p   j  m_j * pi_j * (1 - pi_j)

        This is similar to the replicate-based formula, but we are
        using the model prediction for pi_j instead of the pooled
        observations, and we are using the n-p as the error DOF
        instead of the number of replicates. This methodology still
        breaks down when the sizes of the covariate classes, m_j, are
        small.

        In order to use the replicate-based formula, there must be at
        least one covariate class exhibiting replication, and the
        degree of replication must be at least two. If these
        conditions are not met, and the user dictates that we use the
        replicate-based formula, we simply ignore that directive and
        use the Pearson-based approach. (It might be best to issue a
        warning in this case, but we do not do that.)

        If this function is called without specifying which
        methodology to use, we use the following criteria in assessing
        whether there is enough replication to use the first
        approach. First, there must be at least two covariate classes
        exhibiting replication. Second, the degree of replication of
        the most-replicated covariate class must be at least
        3. Finally, the total replication degrees of freedom must be
        at least 10. For example, in the example data set above, there
        are two covariate classes exhibiting replication: Males in
        Hospital 1, and Males in Hospital 2, with 3 and 2 degrees of
        replication, respectively. The degree of replication of the
        most-replicate covariate class is therefore equal to 3. The
        degrees of freedom are (2-1) + (3-1) = 3, which is below the
        threshold of 10. We would therefore use the Pearson-based
        formula in this case.

        These criteria are completely arbitrary! I need to do more
        research to determine the appropriate criteria.

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

    def dof(self) -> float:
        """Degrees of freedom: sum of feature DOFs plus the affine intercept."""
        dof = 1.0  # Affine factor
        for _, feature in self._features.items():
            dof += feature.dof()
        return dof

    def aic(self) -> float:
        """Akaike Information Criterion

        Returns the AIC for the fitted model, useful for choosing
        smoothing parameters. The AIC we compute is actually off
        by a constant factor, making it easier to compute without
        detracting from its role in model selection.

        Different authors seem to throw in multiplicative or additive
        factors willy-nilly since it doesn't affect model selection.
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
        """Un-Biased Risk Estimator

        Returns the Un-Biased Risk Estimator as discussed in Sections
        6.2.1 and 6.2.5 of [GAMr]. This can be used for choosing the
        smoothing parameter when the dispersion is known.

        As discussed in Section 6.2.5 of [GAMr], sometimes it is helpful
        to force smoother fits by exaggerating the effective degrees of
        freedom. In that case, a value of gamma > 1. may be desirable.
        """
        return self.deviance() + 2.0 * gamma * self.dispersion() * self.dof()

    def gcv(self, gamma: float = 1.0) -> float:
        """Generalized Cross Validation

        This function returns the Generalized Cross Validation (GCV)
        score, which can be used for choosing the smoothing parameter
        when the dispersion is unknown.

        As discussed in Section 6.2.5 of [GAMr], sometimes it is helpful
        to force smoother fits by exaggerating the effective degrees of
        freedom. In that case, a value of gamma > 1. may be desirable.
        """
        denom = self._num_obs - gamma * self.dof()
        return self._num_obs * self.deviance() / (denom * denom)

    def summary(self) -> None:
        """Print summary statistics associated with fitted model.

        Prints statistics for the overall model, as well as for
        each individual feature (see the __str__() function in
        each feature type for details about what is printed
        there).

        For the overall model, the following are printed:
           phi:      Estimated dispersion parameter. Omitted
                     if specified or if it is known for the
                     Family (e.g. Poisson).
           edof:     Estimated degrees of freedom.
           Deviance: The difference between the log-likelihood of
                     the model that fits the data perfectly and
                     that of the fitted model, times twice the
                     dispersion.
           AIC:      Akaike Information Criterion.
           AICc:     AIC with correction for finite data sets.
           BIC:      Bayesian Information Criterion.
           R^2:      Deviance-based pseudo-R^2.
           UBRE:     Unbiased Risk Estimator (if dispersion is known).
           GCV:      Generalized Cross Validation (if dispersion is estimated).

        For more details on these parameters, see the documentation
        in the corresponding functions. It would also be nice to have
        confidence intervals at least on the estimated dispersion
        parameter.
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
