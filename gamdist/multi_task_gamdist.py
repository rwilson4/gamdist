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

"""Multi-task GAM: K supervised-learning tasks fit jointly under a
shared feature set with optional convex coupling regularization.

Per the design discussion on issue #39 (and CLAUDE.md's seam principle):

  - Per-task primal step is unchanged: each task's task-local features
    run their existing `optimize(fpumz_k, rho)` independently. A
    multi-task feature wraps K of those into a single
    `optimize_multi(fpumz_list, rho)` call, which is the only place the
    cross-task coupling penalty enters.
  - Per-task proximal step is unchanged: K independent prox calls,
    each with its own `(family_k, link_k, y_k)` -- so tasks can use
    different family/link pairs.
  - `MultiTaskGAM` is a separate class whose state (`f_bar`,
    `z_bar`, `u`, residual histories, offset, ...) is a length-K list.
    Single-task `GAM` is unchanged.

The first concrete coupling penalty is group-lasso-across-tasks on
linear features (see `_MultiTaskLinearFeature`); persistence,
`summary()`, plotting, model-selection statistics, and per-task
`tasks=` routing are out of scope for this slice.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.linalg as linalg
import scipy.special as special
import scipy.stats as stats

from . import proximal_operators as po
from .gamdist import (
    CANONICAL_LINKS,
    FAMILIES,
    FAMILIES_WITH_KNOWN_DISPERSIONS,
    LINKS,
    QUASI_BASE_FAMILY,
    SUPPORTED_FAMILY_LINK_PAIRS,
    Family,
    Link,
)
from .multi_task_features import _MultiTaskFeature, _MultiTaskLinearFeature

FloatArray = npt.NDArray[np.float64]
MultiTaskFeatureType = Literal["linear"]


def _link_callables(
    link: str,
) -> tuple[Callable[[FloatArray], FloatArray], Callable[[FloatArray], FloatArray]]:
    """Return ``(link, inv_link)`` callables for a link name.

    Mirrors the inline link/inv_link lambdas in ``GAM.__init__``;
    factored out so a list of K of them can be built without copy-paste.
    """
    if link == "identity":
        return (lambda x: x, lambda x: x)
    if link == "logistic":
        return (
            lambda x: np.asarray(special.logit(x), dtype=float),
            lambda x: np.asarray(special.expit(x), dtype=float),
        )
    if link == "probit":
        return (
            lambda x: np.asarray(stats.norm.ppf(x), dtype=float),
            lambda x: np.asarray(stats.norm.cdf(x), dtype=float),
        )
    if link == "complementary_log_log":
        return (
            lambda x: np.asarray(np.log(-np.log(1.0 - x)), dtype=float),
            lambda x: np.asarray(1.0 - np.exp(-np.exp(x)), dtype=float),
        )
    if link == "log":
        return (
            lambda x: np.asarray(np.log(x), dtype=float),
            lambda x: np.asarray(np.exp(x), dtype=float),
        )
    if link == "reciprocal":
        return (lambda x: 1.0 / x, lambda x: 1.0 / x)
    if link == "reciprocal_squared":
        return (
            lambda x: np.asarray(1.0 / (x * x), dtype=float),
            lambda x: np.asarray(1.0 / np.sqrt(x), dtype=float),
        )
    raise ValueError(f"{link!r} link not supported")


class MultiTaskGAM:
    """Generalized Additive Model fit jointly across K tasks.

    Each task ``k`` has its own ``(family_k, link_k)`` pair, its own
    response ``y_k`` (lengths may differ across tasks), and its own
    feature data; coefficients are tied together by the coupling
    regularizer attached to each multi-task feature.

    Parameters
    ----------
    families : list[str]
        Length-K list of family names. Each entry must be one of the
        single-task families supported by ``GAM``. Tasks may use
        different families.
    links : list[str | None] or None
        Length-K list of link names; defaults to the canonical link of
        each family. ``None`` entries fall back to the canonical link.
    dispersions : list[float | None] or None
        Optional per-task fixed dispersion. ``None`` entries follow
        the single-task ``GAM`` defaults (e.g. binomial / poisson are
        fixed at 1).
    name : str or None
        Optional model name. Persistence is not supported in this
        slice; the name is currently informational only.

    Notes
    -----
    Several features of single-task ``GAM`` are intentionally not yet
    wired up for the multi-task case in this slice and raise
    ``NotImplementedError`` if invoked: ``save_flag``,
    ``load_from_file``, ``summary``, ``plot``, ``aic`` / ``aicc`` /
    ``gcv`` / ``ubre``, ``confidence_intervals``, robust covariance,
    ``quantile`` / ``huber`` families, covariate classes, and the
    ``estimate_overdispersion`` path.
    """

    def __init__(
        self,
        families: list[Family],
        links: list[Link | None] | None = None,
        dispersions: list[float | None] | None = None,
        name: str | None = None,
    ) -> None:
        if not isinstance(families, list) or len(families) < 1:
            raise ValueError("families must be a non-empty list.")
        K = len(families)
        for f in families:
            if f not in FAMILIES:
                raise ValueError(f"{f!r} family not supported")
            if f in {"quantile", "huber"}:
                raise NotImplementedError(
                    f"family {f!r} is not yet supported in MultiTaskGAM "
                    "(needs per-task tau / delta plumbing)."
                )

        if links is None:
            links_resolved: list[str] = [str(CANONICAL_LINKS[fam]) for fam in families]
        else:
            if len(links) != K:
                raise ValueError(
                    f"links has length {len(links)} but families has length {K}."
                )
            links_resolved = []
            for k, lk in enumerate(links):
                if lk is None:
                    links_resolved.append(str(CANONICAL_LINKS[families[k]]))
                elif lk in LINKS:
                    links_resolved.append(lk)
                else:
                    raise ValueError(f"{lk!r} link not supported")

        for k in range(K):
            pair = (families[k], links_resolved[k])
            if pair not in SUPPORTED_FAMILY_LINK_PAIRS:
                canonical = CANONICAL_LINKS[families[k]]
                raise ValueError(
                    f"task {k}: ({families[k]!r}, {links_resolved[k]!r}) is "
                    "not a supported (family, link) combination. Per the "
                    "convexity-only design (see CLAUDE.md), only pairs with "
                    "a dedicated convex proximal-operator implementation are "
                    f"accepted; use link={canonical!r} for "
                    f"family={families[k]!r}."
                )

        if dispersions is None:
            dispersions_resolved: list[float | None] = [None] * K
        elif len(dispersions) != K:
            raise ValueError(
                f"dispersions has length {len(dispersions)} but families has "
                f"length {K}."
            )
        else:
            dispersions_resolved = list(dispersions)

        self._num_tasks = K
        self._families: list[Family] = list(families)
        self._links: list[str] = list(links_resolved)
        self._base_families: list[str] = [
            QUASI_BASE_FAMILY.get(f, f) for f in self._families
        ]
        self._known_dispersion: list[bool] = []
        self._dispersion: list[float] = []
        for k in range(K):
            d = dispersions_resolved[k]
            if d is not None:
                self._known_dispersion.append(True)
                self._dispersion.append(float(d))
            elif self._families[k] in FAMILIES_WITH_KNOWN_DISPERSIONS:
                self._known_dispersion.append(True)
                self._dispersion.append(
                    float(FAMILIES_WITH_KNOWN_DISPERSIONS[self._families[k]])
                )
            else:
                self._known_dispersion.append(False)
                self._dispersion.append(1.0)

        self._eval_link: list[Callable[[FloatArray], FloatArray]] = []
        self._eval_inv_link: list[Callable[[FloatArray], FloatArray]] = []
        for k in range(K):
            link_k, inv_k = _link_callables(self._links[k])
            self._eval_link.append(link_k)
            self._eval_inv_link.append(inv_k)

        self._features: dict[str, _MultiTaskFeature] = {}
        self._num_features = 0
        self._fitted = False
        self._name = name
        # Placeholders so any inherited code that references these
        # single-task attributes does not blow up.
        self._tau = None
        self._delta = None

    def add_feature(
        self,
        name: str,
        type: MultiTaskFeatureType,
        transform: Callable[[npt.NDArray[Any]], npt.NDArray[Any]] | None = None,
        regularization: dict[str, Any] | None = None,
    ) -> None:
        """Add a feature shared across all K tasks.

        The first slice supports only ``type='linear'``. Each task
        ``k`` contributes ``m_k * (x_k - mean(x_k))`` to its own
        linear predictor; the K slopes ``m_1, ..., m_K`` may be
        coupled by a convex coupling regularizer (see
        ``_MultiTaskLinearFeature`` for the supported penalties).
        """
        feat: _MultiTaskFeature
        if type == "linear":
            feat = _MultiTaskLinearFeature(
                name,
                num_tasks=self._num_tasks,
                transform=transform,
                regularization=regularization,
            )
        else:
            raise ValueError(
                f"MultiTaskGAM features of type {type!r} not supported in "
                "this slice (only 'linear' is wired up; categorical / "
                "spline can be added by analogy)."
            )
        self._features[name] = feat
        self._num_features += 1

    def fit(
        self,
        Xs: list[pd.DataFrame],
        ys: list[npt.NDArray[Any]],
        weights: list[npt.NDArray[Any] | None] | None = None,
        smoothing: float = 1.0,
        verbose: bool = False,
        max_its: int = 100,
    ) -> None:
        """Fit the multi-task GAM via K-coupled ADMM.

        Parameters
        ----------
        Xs : list[pandas.DataFrame]
            Length-K list of per-task design matrices. Column names
            must include every feature added via ``add_feature``.
            Per-task row counts may differ.
        ys : list[ndarray]
            Length-K list of per-task response arrays. ``len(ys[k])``
            must equal ``len(Xs[k])``.
        weights : list[ndarray | None] or None
            Optional per-task observation weights. ``None`` (or a
            ``None`` entry) means unit weights for that task.
        smoothing : float
            Multiplicative scale applied to every feature's
            regularization coefficient (matches single-task ``GAM``).
        verbose : bool
            Print per-iteration progress.
        max_its : int
            Maximum ADMM iterations (per the joint stopping rule).
        """
        K = self._num_tasks
        if not isinstance(Xs, list) or len(Xs) != K:
            raise ValueError(f"Xs must be a list of length {K}.")
        if not isinstance(ys, list) or len(ys) != K:
            raise ValueError(f"ys must be a list of length {K}.")
        if weights is None:
            weights_list: list[npt.NDArray[Any] | None] = [None] * K
        else:
            if len(weights) != K:
                raise ValueError(f"weights must be a list of length {K}.")
            weights_list = list(weights)

        for k in range(K):
            if len(Xs[k]) != len(ys[k]):
                raise ValueError(
                    f"task {k}: Xs has {len(Xs[k])} rows, ys has {len(ys[k])}."
                )
            wk = weights_list[k]
            if wk is not None and len(wk) != len(ys[k]):
                raise ValueError(
                    f"task {k}: weights has length {len(wk)} but "
                    f"ys has length {len(ys[k])}."
                )

        self._rho = 0.1
        eps_abs = 1e-3
        eps_rel = 1e-3

        self._num_obs: list[int] = [len(ys[k]) for k in range(K)]
        self._y: list[FloatArray] = [np.asarray(ys[k]).flatten() for k in range(K)]
        self._weights: list[npt.NDArray[Any] | None] = weights_list
        self._offset: list[float] = []
        for k in range(K):
            yk = self._y[k]
            mean_arr = np.asarray(np.mean(yk), dtype=float).reshape(())
            self._offset.append(float(self._eval_link[k](mean_arr)))

        # Initialize features. Each multi-task feature builds K
        # parallel per-task data caches in one call.
        for name, feature in self._features.items():
            xs = [np.asarray(Xs[k][name].values) for k in range(K)]
            feature.initialize_multi(xs, smoothing=smoothing, verbose=verbose)

        N = self._num_features

        # Per-task ADMM state. Each is a list of K arrays (potentially
        # of different lengths).
        self.f_bar: list[FloatArray] = [
            np.full((self._num_obs[k],), self._offset[k] / N if N > 0 else 0.0)
            for k in range(K)
        ]
        self.z_bar: list[FloatArray] = [np.zeros(self._num_obs[k]) for k in range(K)]
        self.u: list[FloatArray] = [np.zeros(self._num_obs[k]) for k in range(K)]
        self.prim_res: list[float] = []
        self.dual_res: list[float] = []
        self.prim_tol: list[float] = []
        self.dual_tol: list[float] = []
        self.dev: list[list[float]] = []

        if N == 0:
            # Degenerate: no features. Fit reduces to per-task offsets.
            self._fitted = True
            return

        fj: dict[str, list[FloatArray]] = {
            name: [np.zeros(self._num_obs[k]) for k in range(K)]
            for name in self._features
        }
        self._admm_loop_multi(max_its, eps_abs, eps_rel, fj, verbose)
        self._fitted = True

    def _admm_loop_multi(
        self,
        max_its: int,
        eps_abs: float,
        eps_rel: float,
        fj: dict[str, list[FloatArray]],
        verbose: bool,
    ) -> None:
        """Run the K-coupled ADMM iterations.

        Mirrors ``GAM._admm_loop`` but holds per-task lists for every
        ADMM-state vector. The per-feature primal step is one call per
        feature (which internally handles all K tasks); the per-task
        proximal step is K independent calls.
        """
        K = self._num_tasks
        N = self._num_features

        for i in range(max_its):
            if verbose:
                print(f"Iteration {i:d}")
                print("Optimizing primal variables")

            fpumz = [N * (self.f_bar[k] + self.u[k] - self.z_bar[k]) for k in range(K)]
            fj_new: dict[str, list[FloatArray]] = {}
            f_new: list[FloatArray] = [
                np.full((self._num_obs[k],), self._offset[k]) for k in range(K)
            ]
            for name, feature in self._features.items():
                contribs = feature.optimize_multi(fpumz, self._rho)
                fj_new[name] = contribs
                for k in range(K):
                    f_new[k] = f_new[k] + contribs[k]

            for k in range(K):
                f_new[k] = f_new[k] / N

            if verbose:
                print("Optimizing dual variables")

            z_new: list[FloatArray] = [
                self._optimize_task(k, self.u[k] + f_new[k], N) for k in range(K)
            ]

            for k in range(K):
                self.u[k] = self.u[k] + (f_new[k] - z_new[k])

            # Per-task primal / dual residuals; we declare convergence
            # only when *every* task is below its tolerance. This is
            # the strongest natural rule and falls out of the existing
            # single-task formulae applied per task.
            prim_res_per_task = np.zeros(K)
            dual_res_per_task = np.zeros(K)
            prim_tol_per_task = np.zeros(K)
            dual_tol_per_task = np.zeros(K)

            for k in range(K):
                prim_res_per_task[k] = np.sqrt(N) * linalg.norm(f_new[k] - z_new[k])

                dual_sq = 0.0
                norm_ax_sq = 0.0
                norm_bz_sq = 0.0
                norm_aty_sq = 0.0
                num_params_k = 0
                for name, feature in self._features.items():
                    dr = (
                        (fj_new[name][k] - fj[name][k])
                        + (z_new[k] - self.z_bar[k])
                        - (f_new[k] - self.f_bar[k])
                    )
                    dual_sq += float(dr.dot(dr))
                    norm_ax_sq += float(fj_new[name][k].dot(fj_new[name][k]))
                    zik = fj_new[name][k] + z_new[k] - f_new[k]
                    norm_bz_sq += float(zik.dot(zik))
                    norm_aty_sq += feature.compute_dual_tol_task(k, self.u[k])
                    num_params_k += feature.num_params_task(k)

                dual_res_per_task[k] = self._rho * np.sqrt(dual_sq)
                norm_ax = np.sqrt(norm_ax_sq)
                norm_bz = np.sqrt(norm_bz_sq)
                norm_aty = np.sqrt(norm_aty_sq)
                prim_tol_per_task[k] = np.sqrt(
                    self._num_obs[k] * N
                ) * eps_abs + eps_rel * max(norm_ax, norm_bz)
                dual_tol_per_task[k] = (
                    np.sqrt(num_params_k) * eps_abs + eps_rel * norm_aty
                )

            self.f_bar = f_new
            self.z_bar = z_new
            fj = fj_new

            # Aggregate per-task residuals/tolerances by max for the
            # joint convergence trace; the per-task arrays are
            # available via ``residuals_per_task`` if a caller wants
            # to inspect them.
            self.prim_res.append(float(np.max(prim_res_per_task)))
            self.dual_res.append(float(np.max(dual_res_per_task)))
            self.prim_tol.append(float(np.max(prim_tol_per_task)))
            self.dual_tol.append(float(np.max(dual_tol_per_task)))
            self.dev.append(self.deviance())

            converged = bool(
                np.all(prim_res_per_task < prim_tol_per_task)
                and np.all(dual_res_per_task < dual_tol_per_task)
            )
            if converged:
                if verbose:
                    print("Fit converged")
                break
        else:
            if verbose:
                print("Fit did not converge")

    def _optimize_task(self, k: int, upf: FloatArray, N: int) -> FloatArray:
        """Per-task proximal step. Mirrors ``GAM._optimize`` but
        parameterized by task index ``k`` so each task uses its own
        ``(family, link, y, weights)``.
        """
        base = self._base_families[k]
        rho = self._rho
        y = self._y[k]
        w = self._weights[k]
        inv = self._eval_inv_link[k]

        prox: Any
        if base == "normal":
            prox = po._prox_normal_identity
        elif base == "binomial":
            prox = po._prox_binomial_logit
            return (1.0 / N) * prox(N * upf, rho, y, None, w, inv)
        elif base == "poisson":
            prox = po._prox_poisson_log
        elif base == "gamma":
            prox = po._prox_gamma_reciprocal
        elif base == "inverse_gaussian":
            prox = po._prox_inv_gaussian_reciprocal_squared
        else:
            raise ValueError(
                f"task {k}: family {self._families[k]!r} with link "
                f"{self._links[k]!r} not supported in MultiTaskGAM."
            )

        return (1.0 / N) * prox(N * upf, rho, y, w=w, inv_link=inv)

    def predict(
        self,
        X: pd.DataFrame | list[pd.DataFrame],
    ) -> list[FloatArray]:
        """Predict per-task mean responses.

        Parameters
        ----------
        X : pandas.DataFrame or list[pandas.DataFrame]
            If a single DataFrame is passed, the same predictor matrix
            is used for every task and the K predictions all have the
            same length. If a list of K DataFrames is passed, each
            task's prediction uses its own DataFrame.

        Returns
        -------
        mus : list of arrays
            Length-K list of predicted mean responses; ``mus[k]`` has
            shape ``(len(X[k]),)``.
        """
        if not self._fitted:
            raise AttributeError("Model not yet fit.")

        K = self._num_tasks
        if isinstance(X, pd.DataFrame):
            Xs = [X for _ in range(K)]
        else:
            if len(X) != K:
                raise ValueError(
                    f"X must be a DataFrame or a list of length {K}; got {len(X)}."
                )
            Xs = list(X)

        mus: list[FloatArray] = []
        for k in range(K):
            num_points = len(Xs[k])
            eta = np.full((num_points,), self._offset[k])
            for name, feature in self._features.items():
                eta = eta + feature.predict_task(k, np.asarray(Xs[k][name].values))
            mus.append(np.asarray(self._eval_inv_link[k](eta), dtype=float))
        return mus

    def deviance(
        self,
        X: list[pd.DataFrame] | None = None,
        y: list[npt.NDArray[Any]] | None = None,
        weights: list[npt.NDArray[Any] | None] | None = None,
    ) -> list[float]:
        """Per-task deviance.

        With no arguments, returns the training deviance of each task
        evaluated against the current ADMM iterate. Otherwise, the
        deviance is computed against the supplied per-task data.
        """
        K = self._num_tasks
        if X is None or y is None:
            ys = self._y
            mus: list[FloatArray] = [
                np.asarray(
                    self._eval_inv_link[k](self._num_features * self.f_bar[k]),
                    dtype=float,
                )
                for k in range(K)
            ]
            ws = self._weights
        else:
            if len(X) != K or len(y) != K:
                raise ValueError(f"X and y must each be lists of length {K}.")
            ys = [np.asarray(y[k]).flatten() for k in range(K)]
            mus = self.predict(list(X))
            if weights is None:
                ws = [None] * K
            else:
                if len(weights) != K:
                    raise ValueError(f"weights must be a list of length {K}.")
                ws = list(weights)

        return [self._deviance_task(k, ys[k], mus[k], w=ws[k]) for k in range(K)]

    def _deviance_task(
        self,
        k: int,
        y: FloatArray,
        mu: FloatArray,
        w: npt.NDArray[Any] | None = None,
    ) -> float:
        """Per-task deviance, dispatching on this task's family.

        Mirrors ``GAM._deviance_from_mu`` but parameterized by task
        index. Covariate-class data is not yet supported in
        ``MultiTaskGAM``, so ``m`` is always 1.
        """
        base = self._base_families[k]
        if base == "normal":
            r = y - mu
            if w is None:
                return float(r.dot(r))
            return float(w.dot(r * r))
        if base == "binomial":
            eps = np.finfo(float).eps
            mu_c = np.clip(mu, eps, 1.0 - eps)
            term = y * np.log(mu_c) + (1.0 - y) * np.log1p(-mu_c)
            if w is None:
                return float(-2.0 * np.sum(term))
            return float(-2.0 * w.dot(term))
        if base == "poisson":
            y_log_term = np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / mu), 0.0)
            term = y_log_term - (y - mu)
            if w is None:
                return float(2.0 * np.sum(term))
            return float(2.0 * w.dot(term))
        if base == "gamma":
            tiny = np.finfo(float).tiny
            y_safe = np.where(y > 0, y, tiny)
            mu_safe = np.where(mu > 0, mu, tiny)
            term = -np.log(y_safe / mu_safe) + (y - mu_safe) / mu_safe
            if w is None:
                return float(2.0 * np.sum(term))
            return float(2.0 * w.dot(term))
        if base == "inverse_gaussian":
            r = y - mu
            term = r * r / (mu * mu * y)
            if w is None:
                return float(np.sum(term))
            return float(w.dot(term))
        raise ValueError(
            f"task {k}: deviance not implemented for family {self._families[k]!r}."
        )

    # --- single-task GAM features intentionally NOT wired up in this
    # slice; raise so callers find out at call time rather than
    # silently getting nonsense. ---

    def summary(self) -> None:
        raise NotImplementedError("summary() is not yet implemented for MultiTaskGAM.")

    def plot(self, name: str, true_fn: Any = None) -> None:
        raise NotImplementedError("plot() is not yet implemented for MultiTaskGAM.")

    def confidence_intervals(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "confidence_intervals() is not yet implemented for MultiTaskGAM."
        )

    def aic(self) -> float:
        raise NotImplementedError("aic() is not yet implemented for MultiTaskGAM.")

    def aicc(self) -> float:
        raise NotImplementedError("aicc() is not yet implemented for MultiTaskGAM.")

    def gcv(self, gamma: float = 1.0) -> float:
        raise NotImplementedError("gcv() is not yet implemented for MultiTaskGAM.")

    def ubre(self, gamma: float = 1.0) -> float:
        raise NotImplementedError("ubre() is not yet implemented for MultiTaskGAM.")

    def _save(self) -> None:
        raise NotImplementedError(
            "Persistence is not yet implemented for MultiTaskGAM."
        )

    def _load(self, filename: str) -> None:
        raise NotImplementedError(
            "Persistence is not yet implemented for MultiTaskGAM."
        )
