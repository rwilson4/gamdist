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
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.linalg as linalg
import scipy.special as special
import scipy.stats as stats

from . import proximal_operators as po
from .categorical_feature import _CategoricalFeature
from .feature import _Feature
from .linear_feature import _LinearFeature
from .spline_feature import _SplineFeature

FloatArray = npt.NDArray[np.float64]

Family = Literal["normal", "binomial", "poisson", "gamma", "exponential", "inverse_gaussian"]
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

# To do:
# - Hierarchical models
# - Piecewise constant fits, total variation regularization
# - Monotone constraint
# - Implement overdispersion for Poisson family
# - Implement Multinomial, Proportional Hazards
# - Implement outlier detection
# - AICc, BIC, R-squared estimate
# - Confidence intervals on mu, predictions (probably need to use Bootstrap but can
#   do so intelligently)
# - Confidence intervals on model parameters, p-values
# - Group lasso penalty (l2 norm -- not squared -- or l_\infty norm on f_j(x_j; p_j))
# - Interactions
# - Runtime optimization (Cython)
# - Fit in parallel
# - Residuals
#   - Compute different types of residuals (Sec 3.1.7 of [GAMr])
#   - Plot residuals against mean response, variance, predictor, unused predictor
#   - QQ plot of residuals
#
# Done:
# - Implement Gaussian, Binomial, Poisson, Gamma, Inv Gaussian,
# - Plot splines
# - Deviance (on training set and test set), AIC, Dispersion, GCV, UBRE
# - Write documentation
# - Check implementation of Gamma dispersion
# - Implement probit, complementary log-log links.
# - Implement Binomial models for covariate classes
# - Constrain spline to have mean prediction 0 over the data
# - Save and load properly
# - Implement overdispersion for Binomial family

FAMILIES = ['normal',
            'binomial',
            'poisson',
            'gamma',
            'exponential',
            'inverse_gaussian'
            ]

LINKS = ['identity',
         'logistic',
         'probit',
         'complementary_log_log',
         'log',
         'reciprocal',
         'reciprocal_squared'
         ]

FAMILIES_WITH_KNOWN_DISPERSIONS = {'binomial': 1,
                                   'poisson': 1
                                   }

CANONICAL_LINKS = {'normal': 'identity',
                   'binomial': 'logistic',
                   'poisson': 'log',
                   'gamma': 'reciprocal',
                   'inverse_gaussian': 'reciprocal_squared'
                   }
# Non-canonical but common link/family combinations include:
# Binomial: probit and complementary log-log
# Gamma: identity and log

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
    ax.plot(range(len(dev_arr)), (dev_arr - dev_arr[-1]) + 1e-10, "b-", label="Deviance")
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

    This function estimates the dispersion of a Gamma family with p
    degrees of freedom and deviance D, and n observations. The
    dispersion nu is that number satisfying
      2*n * (log nu - psi(nu)) - p / nu = D

    We use Newton's method with a learning rate to solve this nonlinear
    equation.

    Parameters
    ----------
     dof : float
         Degrees of freedom
     dev : float
         Deviance
     num_obs : int
         Number of observations

    Returns
    -------
     nu : float
         Estimated dispersion
    """
    beta = 0.1
    tol = 1e-6
    max_its = 100

    nu = 1.
    for i in range(max_its):
        num = 2. * num_obs * (np.log(nu) - special.psi(nu)) - dof / nu - dev
        denom = 2. * num_obs * (1. / nu - special.polygamma(1, nu)) + dof / (nu * nu)
        dnu = num / denom
        nu -= dnu * beta
        if abs(dnu) < tol:
            return nu
    else:
        raise ValueError('Could not estimate gamma dispersion.')

class GAM:
    def __init__(
        self,
        family: Family | None = None,
        link: Link | None = None,
        dispersion: float | None = None,
        estimate_overdispersion: bool = False,
        name: str | None = None,
        load_from_file: str | None = None,
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
                'inverse_gaussian' (still in progress).
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

        Returns
        -------
         mdl : Generalized Additive Model object

        """

        if load_from_file is not None:
            self._load(load_from_file)
            return

        if family is None:
            raise ValueError('Family not specified.')
        elif family not in FAMILIES:
            raise ValueError(f'{family} family not supported')
        elif family == 'exponential':
            # Exponential is a special case of Gamma with a dispersion of 1.
            self._family = 'gamma'
            dispersion = 1.
        else:
            self._family = family

        if link is None:
            self._link = CANONICAL_LINKS[self._family]
        elif link in LINKS:
            self._link = link
        else:
            raise ValueError(f'{link} link not supported')

        if dispersion is not None:
            self._known_dispersion = True
            self._dispersion = dispersion
        elif (self._family in FAMILIES_WITH_KNOWN_DISPERSIONS
              and not estimate_overdispersion):
            self._known_dispersion = True
            self._dispersion = FAMILIES_WITH_KNOWN_DISPERSIONS[self._family]
        else:
            self._known_dispersion = False

        if self._link == 'identity':
            self._eval_link = lambda x: x
            self._eval_inv_link = lambda x: x
        elif self._link == 'logistic':
            self._eval_link = lambda x: special.logit(x)
            self._eval_inv_link = lambda x: special.expit(x)
        elif self._link == 'probit':
            # Inverse CDF of the Gaussian distribution
            self._eval_link = lambda x: stats.norm.ppf(x)
            self._eval_inv_link = lambda x: stats.norm.cdf(x)
        elif self._link == 'complementary_log_log':
            self._eval_link = lambda x: np.log(-np.log(1. - x))
            self._eval_inv_link = lambda x: 1. - np.exp(-np.exp(x))
        elif self._link == 'log':
            self._eval_link = lambda x: np.log(x)
            self._eval_inv_link = lambda x: np.exp(x)
        elif self._link == 'reciprocal':
            self._eval_link = lambda x: 1. / x
            self._eval_inv_link = lambda x: 1. / x
        elif self._link == 'reciprocal_squared':
            self._eval_link = lambda x: 1. / (x * x)
            self._eval_inv_link = lambda x: 1. / np.sqrt(x)

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
        self._family = mv['family']
        self._link = mv['link']
        self._known_dispersion = mv['known_dispersion']
        if self._known_dispersion:
            self._dispersion = mv['dispersion']

        self._estimate_overdispersion = mv['estimate_overdispersion']
        self._offset = mv['offset']
        self._num_features = mv['num_features']
        self._fitted = mv['fitted']
        self._name = mv['name']

        self._features = {}
        features = mv['features']
        for (name, feature) in features.items():
            if feature['type'] == 'categorical':
                self._features[name] = _CategoricalFeature(load_from_file=feature['filename'])
            elif feature['type'] == 'linear':
                self._features[name] = _LinearFeature(load_from_file=feature['filename'])
            elif feature['type'] == 'spline':
                self._features[name] = _SplineFeature(load_from_file=feature['filename'])
            else:
                raise ValueError('Invalid feature type')

        # self._rho = mv['rho']
        self._num_obs = mv['num_obs']
        self._y = mv['y']
        self._weights = mv['weights']
        self._has_covariate_classes = mv['has_covariate_classes']
        if self._has_covariate_classes:
            self._covariate_class_sizes = mv['covariate_class_sizes']

        self.f_bar = mv['f_bar']
        self.z_bar = mv['z_bar']
        self.u = mv['u']
        self.prim_res = mv['prim_res']
        self.dual_res = mv['dual_res']
        self.prim_tol = mv['prim_tol']
        self.dual_tol = mv['dual_tol']
        self.dev = mv['dev']

        if self._link == 'identity':
            self._eval_link = lambda x: x
            self._eval_inv_link = lambda x: x
        elif self._link == 'logistic':
            self._eval_link = lambda x: special.logit(x)
            self._eval_inv_link = lambda x: special.expit(x)
        elif self._link == 'probit':
            # Inverse CDF of the Gaussian distribution
            self._eval_link = lambda x: stats.norm.ppf(x)
            self._eval_inv_link = lambda x: stats.norm.cdf(x)
        elif self._link == 'complementary_log_log':
            self._eval_link = lambda x: np.log(-np.log(1. - x))
            self._eval_inv_link = lambda x: 1. - np.exp(-np.exp(x))
        elif self._link == 'log':
            self._eval_link = lambda x: np.log(x)
            self._eval_inv_link = lambda x: np.exp(x)
        elif self._link == 'reciprocal':
            self._eval_link = lambda x: 1. / x
            self._eval_inv_link = lambda x: 1. / x
        elif self._link == 'reciprocal_squared':
            self._eval_link = lambda x: 1. / (x * x)
            self._eval_inv_link = lambda x: 1. / np.sqrt(x)


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
             Splines implicitly only support regularization of the wiggliness
             via a C2 smoothness penalty. That is controlled via the rel_dof.
             Other features have more diverse options described in their own
             documentation.

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
            f = _SplineFeature(name, transform, rel_dof if rel_dof is not None else 4.0)
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

        Returns
        -------
         (nothing)

        """
        if save_flag and self._name is None:
            msg = 'Cannot save a GAM with no name.'
            msg += ' Specify name when instantiating model.'
            raise ValueError(msg)

        if len(X) != len(y):
            raise ValueError('Inconsistent number of observations in X and y.')

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
            mean_response = float(np.sum(self._y)) / float(np.sum(self._covariate_class_sizes))
            self._offset = float(self._eval_link(mean_response))
        else:
            self._has_covariate_classes = False
            self._covariate_class_sizes = None
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

        z_new = np.zeros(self._num_obs)

        for i in range(max_its):
            if verbose:
                print(f"Iteration {i:d}")
                print("Optimizing primal variables")

            fpumz = self._num_features * (self.f_bar + self.u - self.z_bar)
            fj_new: dict[str, FloatArray] = {}
            f_new = np.full((self._num_obs,), self._offset)
            for name, feature in self._features.items():
                if verbose:
                    print(f"Optimizing {name:s}")
                fj_new[name] = feature.optimize(fpumz, self._rho)
                f_new += fj_new[name]

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
                dr = ((fj_new[name] - fj[name])
                      + (z_new - self.z_bar)
                      - (f_new - self.f_bar))
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
                prim_tol = (np.sqrt(sccs * self._num_features) * eps_abs
                            + eps_rel * np.max([norm_ax, norm_bz]))

            else:
                prim_tol = (np.sqrt(self._num_obs * self._num_features) * eps_abs
                            + eps_rel * np.max([norm_ax, norm_bz]))

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

        self._fitted = True
        if save_flag:
            self._save()

        if plot_convergence:
            _plot_convergence(self.prim_res, self.prim_tol, self.dual_res,
                              self.dual_tol, self.dev)

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

        # Different prox operators have slightly different signatures (e.g. the
        # binomial variant takes a covariate-class-sizes argument), so the
        # dispatch is typed as Any to keep mypy quiet here.
        prox: Any
        if self._family == 'normal':
            if self._link == 'identity':
                prox = po._prox_normal_identity
            else:
                prox = po._prox_normal
        elif self._family == 'binomial':
            if self._link == 'logistic':
                prox = po._prox_binomial_logit
            else:
                prox = po._prox_binomial

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
            prox = po._prox_poisson_log if self._link == "log" else po._prox_poisson
        elif self._family == "gamma":
            prox = po._prox_gamma_reciprocal if self._link == "reciprocal" else po._prox_gamma
        elif self._family == "inverse_gaussian":
            prox = (
                po._prox_inv_gaussian_reciprocal_squared
                if self._link == "reciprocal_squared"
                else po._prox_inv_gaussian
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
                return float(-2.0 * np.sum(y * np.log(mu_c) + (m - y) * np.log1p(-mu_c)))
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
                    2.0 * np.sum(-1.0 * np.log(y_safe / mu_safe) + (y - mu_safe) / mu_safe)
                )
            return float(
                2.0 * w.dot(-1.0 * np.log(y_safe / mu_safe) + (y - mu_safe) / mu_safe)
            )
        if self._family == "inverse_gaussian":
            if w is None:
                return float(np.sum((y - mu) * (y - mu) / (mu * mu * y)))
            return float(w.dot((y - mu) * (y - mu) / (mu * mu * y)))
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
            return 1.0
        if self._family == "gamma":
            if self._known_dispersion:
                return float(self._dispersion)
            return _gamma_dispersion(self.dof(), self.deviance(), self._num_obs)
        if self._family == "inverse_gaussian":
            if self._known_dispersion:
                return float(self._dispersion)
            return float(self.deviance() / (self._num_obs - self.dof()))
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

        if (num_cc_with_replicates >= min_cc_replicates
             and max_replication >= min_replication):
            has_replication = True
        else:
            has_replication = False

        if (num_cc_with_replicates >= des_cc_replicates
             and max_replication >= des_replication
             and replication_dof >= des_replication_dof):
            has_desired_replication = True
        else:
            has_desired_replication = False

        if formula is None:
            if has_desired_replication:
                formula = 'replication'
            else:
                formula = 'pearson'

        if has_replication and formula == "replication":
            trials: dict[int, float] = {}
            successes: dict[int, float] = {}
            for i in range(self._num_obs):
                cci = int(covariate_class[i])
                trials[cci] = trials.get(cci, 0.0) + float(self._covariate_class_sizes[i])
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
        return self.deviance() / self.dispersion() + 2. * p
        # return (self.deviance() / self._num_obs
        #          + 2. * p * self.dispersion() / self._num_obs)

    def aicc(self) -> float:
        # Eqn 6.32 on p. 304 of [GAMr]
        raise NotImplementedError("aicc is not yet implemented")

    def ubre(self, gamma: float = 1.0) -> float:
        """Un-Biased Risk Estimator

        Returns the Un-Biased Risk Estimator as discussed in Sections
        6.2.1 and 6.2.5 of [GAMr]. This can be used for choosing the
        smoothing parameter when the dispersion is known.

        As discussed in Section 6.2.5 of [GAMr], sometimes it is helpful
        to force smoother fits by exaggerating the effective degrees of
        freedom. In that case, a value of gamma > 1. may be desirable.
        """
        return self.deviance() + 2. * gamma * self.dispersion() * self.dof()

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
           UBRE:     Unbiased Risk Estimator (if dispersion is known).
           GCV:      Generalized Cross Validation (if dispersion is estimated).

        For more details on these parameters, see the documentation
        in the corresponding functions. It may also be helpful to
        include an R^2 value where appropriate, and perhaps a p-value
        for the model against the null model having just the affine
        term. It would also be nice to have confidence intervals
        at least on the estimated dispersion parameter.
        """

        print("Model Statistics")
        print("----------------")
        if not self._known_dispersion:
            print(f"phi: {self.dispersion():0.06g}")
        print(f"edof: {self.dof():0.0f}")
        print(f"Deviance: {self.deviance():0.06g}")
        print(f"AIC: {self.aic():0.06g}")

        if self._known_dispersion:
            print(f"UBRE: {self.ubre():0.06g}")
        else:
            print(f"GCV: {self.gcv():0.06g}")

        print()
        print("Features")
        print("--------")

        for _name, feature in self._features.items():
            print(str(feature))
