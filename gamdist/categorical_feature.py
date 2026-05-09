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

"""Categorical feature: one parameter per level with a zero-sum constraint.

:class:`_CategoricalFeature` contributes a per-level offset to the
linear predictor. The per-feature primal step is solved as a small
QP / SOCP via cvxpy; available regularizers are L1, L2, group lasso,
the L_inf group lasso variant, network lasso, network ridge, and
Huber. Convex shape constraints (sign / lower / upper / monotonic /
convex / concave on a user-supplied category ordering) compose with
those regularizers inside the same cvxpy program.
"""

from __future__ import annotations

import pickle
import warnings
from typing import Any

import cvxpy as cvx
import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.sparse.linalg import splu

from .feature import _Feature

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


class _CategoricalFeature(_Feature):
    """A categorical feature: contributes a per-level offset to the predictor."""

    def __init__(
        self,
        name: str | None = None,
        regularization: dict[str, Any] | None = None,
        constraints: dict[str, Any] | None = None,
        load_from_file: str | None = None,
    ) -> None:
        r"""Initialize feature model (independent of data).

        Parameters
        ----------
        name : str
            Name for the feature, used to make plots.
        regularization : dict, optional
            Description of regularization terms. Supported keys:
            ``l1``, ``l2``, ``huber``, ``group_lasso``,
            ``group_lasso_inf``, ``network_lasso``, ``network_ridge``.

            The ``group_lasso_inf`` variant penalizes

            .. math::

               \lambda \|A q\|_\infty =
               \lambda \max_{c \text{ with obs}} |q_c|,

            which clips the largest per-level effect rather than
            applying the uniform :math:`L_2` contraction induced by
            ``group_lasso``.

            ``huber`` takes ``{"coef": lam, "delta": d}`` (with
            ``coef`` either a scalar or a per-category dict, like
            ``l2``) and adds :math:`\lambda_c h_\delta(q_c)` per
            category, where :math:`h_\delta` is the standard Huber
            function -- ridge for :math:`|q_c| \le \delta`, linear
            with slope :math:`\lambda_c \delta` outside. See the
            package README for details.
        constraints : dict, optional
            Optional convex shape constraints on the per-category
            coefficient vector :math:`q`. ``lower`` / ``upper`` take
            floats and bound every :math:`q_c`. ``sign``
            (``"nonnegative"`` or ``"nonpositive"``) is shorthand for
            one-sided ``lower=0`` / ``upper=0`` and may not combine
            with ``lower`` / ``upper``. ``monotonic``
            (``"increasing"`` or ``"decreasing"``), ``convex``, and
            ``concave`` impose ordering / second-difference
            constraints along a user-supplied ``order`` (a list of
            category labels). The category zero-sum constraint is
            preserved, so ``sign`` constraints will collapse the
            entire vector to zero -- they exist for the occasional
            regulated use case where that is the desired "no
            negative effects" outcome and for symmetry with linear
            features. All constraints append to the existing cvxpy
            program in :meth:`optimize`.
        load_from_file : str, optional
            Pickle path used to restore parameters. If specified,
            other parameters are ignored.
        """
        self.__type__ = "categorical"
        if load_from_file is not None:
            self._load(load_from_file)
            return

        if name is None:
            raise ValueError("Feature must have a name.")

        super().__init__(name)

        self._use_cvx = True
        self._solver = "CLARABEL"
        self._categories: list[Any] = []
        self._has_l1 = False
        self._has_l2 = False
        self._has_huber = False
        self._has_network_lasso = False
        self._has_network_ridge = False
        self._has_group_lasso = False
        self._has_group_lasso_inf = False
        self._has_prior = False
        self._prior_keys: set[Any] | None = None
        self._coef1: float | dict[Any, float] = 0.0
        self._coef2: float | dict[Any, float] = 0.0
        self._coef_huber: float | dict[Any, float] = 0.0
        self._delta_huber: float = 0.0
        self._lambda_network_lasso: float = 0.0
        self._lambda_network_ridge: float = 0.0
        self._lambda_group_lasso: float = 0.0
        self._lambda_group_lasso_inf: float = 0.0
        self._num_edges: int = 0
        self._num_edges_ridge: int = 0
        if regularization is not None:
            if "l1" in regularization:
                self._has_l1 = True
                if "coef" in regularization["l1"]:
                    self._coef1 = regularization["l1"]["coef"]
                else:
                    raise ValueError(
                        "No coefficient specified for l1 regularization term."
                    )

            if "l2" in regularization:
                self._has_l2 = True
                if "coef" in regularization["l2"]:
                    self._coef2 = regularization["l2"]["coef"]
                else:
                    raise ValueError(
                        "No coefficient specified for l2 regularization term."
                    )

            if "network_lasso" in regularization:
                self._has_network_lasso = True
                if "coef" in regularization["network_lasso"]:
                    self._lambda_network_lasso = float(
                        regularization["network_lasso"]["coef"]
                    )
                else:
                    raise ValueError(
                        "No coefficient specified for Network Lasso regularization term."
                    )

                if "edges" in regularization["network_lasso"]:
                    self._edges = regularization["network_lasso"]["edges"]
                    self._num_edges, _ = self._edges.shape
                    for _, row in self._edges.iterrows():
                        if row["node1"] not in self._categories:
                            self._categories.append(row["node1"])
                        if row["node2"] not in self._categories:
                            self._categories.append(row["node2"])
                else:
                    raise ValueError(
                        "Edges not specified for Network Lasso regularization term."
                    )

            if "network_ridge" in regularization:
                self._has_network_ridge = True
                if "coef" in regularization["network_ridge"]:
                    self._lambda_network_ridge = float(
                        regularization["network_ridge"]["coef"]
                    )
                else:
                    raise ValueError(
                        "No coefficient specified for Network Ridge regularization term."
                    )

                if "edges" in regularization["network_ridge"]:
                    self._edges_ridge = regularization["network_ridge"]["edges"]
                    self._num_edges_ridge, _ = self._edges_ridge.shape
                    for _, row in self._edges_ridge.iterrows():
                        if row["node1"] not in self._categories:
                            self._categories.append(row["node1"])
                        if row["node2"] not in self._categories:
                            self._categories.append(row["node2"])
                else:
                    raise ValueError(
                        "Edges not specified for Network Ridge regularization term."
                    )

            if "group_lasso" in regularization:
                self._has_group_lasso = True
                if "coef" in regularization["group_lasso"]:
                    self._lambda_group_lasso = float(
                        regularization["group_lasso"]["coef"]
                    )
                else:
                    raise ValueError(
                        "No coefficient specified for group_lasso regularization term."
                    )

            if "group_lasso_inf" in regularization:
                self._has_group_lasso_inf = True
                if "coef" in regularization["group_lasso_inf"]:
                    self._lambda_group_lasso_inf = float(
                        regularization["group_lasso_inf"]["coef"]
                    )
                else:
                    raise ValueError(
                        "No coefficient specified for group_lasso_inf regularization term."
                    )

            if "huber" in regularization:
                self._has_huber = True
                if "coef" not in regularization["huber"]:
                    raise ValueError(
                        "No coefficient specified for huber regularization term."
                    )
                if "delta" not in regularization["huber"]:
                    raise ValueError(
                        "No delta specified for huber regularization term."
                    )
                self._coef_huber = regularization["huber"]["coef"]
                self._delta_huber = float(regularization["huber"]["delta"])
                if self._delta_huber <= 0.0:
                    raise ValueError("huber delta must be positive.")

            if (self._has_l1 or self._has_l2) and "prior" in regularization:
                self._has_prior = True
                self._prior: Any = regularization["prior"]
                # Track the user-supplied prior keys separately so
                # `_compute_lambda` can apply uniform-coef penalties only to
                # the categories the user named, even after `initialize`
                # rewrites `self._prior` as a dense vector. Without this,
                # a second `initialize()` call (e.g. the refit stage of
                # adaptive lasso) sees `self._prior` as an ndarray and the
                # membership / iteration logic below misbehaves.
                self._prior_keys = set(regularization["prior"].keys())

        self._has_constraints = False
        self._constraint_lower: float = -np.inf
        self._constraint_upper: float = np.inf
        self._constraint_monotonic: str | None = None
        self._constraint_convex: bool = False
        self._constraint_concave: bool = False
        self._constraint_order: list[Any] = []
        if constraints is not None:
            self._has_constraints = True
            allowed = {
                "sign",
                "lower",
                "upper",
                "monotonic",
                "convex",
                "concave",
                "order",
            }
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
            needs_order = (
                self._constraint_monotonic is not None
                or self._constraint_convex
                or self._constraint_concave
            )
            if needs_order:
                if "order" not in constraints:
                    raise ValueError(
                        "constraints: 'monotonic' / 'convex' / 'concave' on a "
                        "categorical feature require an 'order' list of category "
                        "labels defining the sequence."
                    )
                order = list(constraints["order"])
                if len(order) < 2:
                    raise ValueError(
                        "constraints['order']: need at least two categories."
                    )
                if len(set(order)) != len(order):
                    raise ValueError("constraints['order']: categories must be unique.")
                self._constraint_order = order
                # Categories listed in `order` are added to the feature's known
                # category set so the optimize step can reference their indices
                # even if the training data never observes them.
                for cat in order:
                    if cat not in self._categories:
                        self._categories.append(cat)

    def initialize(
        self,
        x: npt.NDArray[Any],
        smoothing: float = 1.0,
        save_flag: bool = False,
        save_prefix: str | None = None,
        verbose: bool = False,
        covariate_class_sizes: npt.NDArray[Any] | None = None,
        na_signifier: Any = None,
    ) -> None:
        """Compute feature-specific state from training data.

        Builds the per-category index, the indicator-style ``A``
        matrix structure, the diagonal ``A^T A`` matrix, and the
        scaled regularization vectors. Calling :meth:`initialize`
        twice (e.g. for the adaptive-lasso refit) is idempotent.

        Parameters
        ----------
        x : ndarray
            Observations corresponding to this feature.
        smoothing : float
            Multiplicative scale applied to every regularization
            coefficient.
        save_flag : bool
            If ``True``, save state to disk after initialization.
        save_prefix : str, optional
            Prefix used to derive the save filename.
        verbose : bool
            Print mildly helpful information when ``True``.
        covariate_class_sizes : ndarray, optional
            Per-observation covariate class sizes.
        na_signifier : object, optional
            Category value treated as missing; predictions for that
            level are pinned to zero.
        """
        self._num_obs = len(x)
        self._verbose = verbose

        self._categories = list(set(x).union(self._categories))
        self._num_categories = len(self._categories)
        self._category_hash: dict[Any, int] = {
            key: i
            for (key, i) in zip(
                self._categories, range(self._num_categories), strict=True
            )
        }

        if self._has_network_lasso:
            D = np.zeros((self._num_edges, self._num_categories))
            _, em = self._edges.shape
            ir = 0
            for _, row in self._edges.iterrows():
                i = self._category_hash[row["node1"]]
                j = self._category_hash[row["node2"]]
                lmbda = float(row["weight"]) if em >= 3 else 1.0
                D[ir, i] = lmbda
                D[ir, j] = -lmbda
                ir += 1
            self._D = sparse.coo_matrix(D).tocsr()
            self._lambda_network_lasso *= smoothing

        if self._has_network_ridge:
            # Build the (weighted) graph Laplacian L so the penalty
            # lambda * sum_{(i,j) in E} w_ij * (q_i - q_j)^2 equals
            # lambda * q^T L q. Symmetric PSD by construction, so the
            # Hessian contribution (2 lambda / rho) * L keeps the
            # subproblem convex.
            _, em = self._edges_ridge.shape
            rows: list[int] = []
            cols: list[int] = []
            data: list[float] = []
            for _, row in self._edges_ridge.iterrows():
                i = self._category_hash[row["node1"]]
                j = self._category_hash[row["node2"]]
                w = float(row["weight"]) if em >= 3 else 1.0
                rows += [i, j, i, j]
                cols += [i, j, j, i]
                data += [w, w, -w, -w]
            self._L = sparse.coo_matrix(
                (data, (rows, cols)),
                shape=(self._num_categories, self._num_categories),
            ).tocsr()
            self._lambda_network_ridge *= smoothing

        if na_signifier is not None and na_signifier in self._categories:
            self._na_index = self._category_hash[na_signifier]
        else:
            self._na_index = -1

        self.x: IntArray = np.zeros(self._num_obs, dtype=np.int64)
        cnt: IntArray = np.zeros(self._num_categories, dtype=np.int64)
        for ix, i in zip(x, range(self._num_obs), strict=True):
            cnt[self._category_hash[ix]] += 1
            self.x[i] = self._category_hash[ix]

        self._lambda1 = np.zeros(self._num_categories)
        if self._has_l1:
            self._lambda1 = self._compute_lambda(self._coef1, smoothing)

        self._lambda2 = np.zeros(self._num_categories)
        if self._has_l2:
            self._lambda2 = self._compute_lambda(self._coef2, smoothing)

        if self._has_group_lasso:
            self._lambda_group_lasso *= smoothing

        if self._has_group_lasso_inf:
            self._lambda_group_lasso_inf *= smoothing

        self._lambda_huber_vec: FloatArray = np.zeros(self._num_categories)
        if self._has_huber:
            if isinstance(self._coef_huber, (int, float)):
                self._lambda_huber_vec[:] = float(self._coef_huber) * smoothing
            else:
                for key, value in self._coef_huber.items():
                    if key in self._category_hash:
                        self._lambda_huber_vec[self._category_hash[key]] = (
                            float(value) * smoothing
                        )

        # Idempotent: a second `initialize()` call (for example, the
        # adaptive-lasso refit) would otherwise try to .items() an
        # ndarray. The original prior keys live in `_prior_keys`; we
        # just keep the existing vector when `_prior` is already one.
        if (
            (self._has_l1 or self._has_l2)
            and self._has_prior
            and isinstance(self._prior, dict)
        ):
            prior_vec = np.zeros(self._num_categories)
            for key, value in self._prior.items():
                prior_vec[self._category_hash[key]] = value
            self._prior = prior_vec

        self._AtA = sparse.dia_matrix(
            (cnt.astype(float), 0),
            shape=(self._num_categories, self._num_categories),
        )
        self.p: FloatArray = np.zeros(self._num_categories)

        if covariate_class_sizes is None:
            self._ccs = cnt.astype(float)
        else:
            self._ccs = self._compute_Atz(
                np.asarray(covariate_class_sizes, dtype=float)
            )

        if self._verbose:
            print(f"Number of categories: {self._num_categories:d}")
            if self._has_network_lasso:
                print(f"Number of edges: {self._num_edges:d}")
            if self._has_network_ridge:
                print(f"Number of network-ridge edges: {self._num_edges_ridge:d}")

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

    def _compute_lambda(
        self, coef: float | dict[Any, float], smoothing: float
    ) -> FloatArray:
        """Build the per-category regularization-weight vector."""
        result = np.zeros(self._num_categories)
        prior_keys = self._prior_keys if self._has_prior else None
        if isinstance(coef, (int, float)):
            scaled = float(coef) * smoothing
            if prior_keys is not None:
                for key in prior_keys:
                    result[self._category_hash[key]] = scaled
            else:
                result[:] = scaled
        else:
            for key, value in coef.items():
                if prior_keys is not None and key not in prior_keys:
                    continue
                result[self._category_hash[key]] = float(value) * smoothing
        return result

    def _save(self) -> None:
        """Save parameters so model fitting can be continued later."""
        assert self._filename is not None
        mv: dict[str, Any] = {
            "num_obs": self._num_obs,
            "categories": self._categories,
            "num_categories": self._num_categories,
            "category_hash": self._category_hash,
            "has_l1": self._has_l1,
            "has_l2": self._has_l2,
            "has_huber": self._has_huber,
            "has_network_lasso": self._has_network_lasso,
            "has_network_ridge": self._has_network_ridge,
            "has_group_lasso": self._has_group_lasso,
            "has_group_lasso_inf": self._has_group_lasso_inf,
            "has_prior": self._has_prior,
            "na_index": self._na_index,
            "x": self.x,
            "p": self.p,
            "AtA": self._AtA,
            "ccs": self._ccs,
            "verbose": self._verbose,
            "use_cvx": self._use_cvx,
            "solver": self._solver,
            "name": self._name,
            "save_self": self._save_self,
        }
        if self._has_l1:
            mv["coef1"] = self._coef1
            mv["lambda1"] = self._lambda1
        if self._has_l2:
            mv["coef2"] = self._coef2
            mv["lambda2"] = self._lambda2
        if self._has_network_lasso:
            mv["num_edges"] = self._num_edges
            mv["D"] = self._D
            mv["edges"] = self._edges
            mv["lambda_network_lasso"] = self._lambda_network_lasso
        if self._has_network_ridge:
            mv["num_edges_ridge"] = self._num_edges_ridge
            mv["L"] = self._L
            mv["edges_ridge"] = self._edges_ridge
            mv["lambda_network_ridge"] = self._lambda_network_ridge
        if self._has_group_lasso:
            mv["lambda_group_lasso"] = self._lambda_group_lasso
        if self._has_group_lasso_inf:
            mv["lambda_group_lasso_inf"] = self._lambda_group_lasso_inf
        if self._has_huber:
            mv["coef_huber"] = self._coef_huber
            mv["delta_huber"] = self._delta_huber
            mv["lambda_huber_vec"] = self._lambda_huber_vec
        if self._has_prior:
            mv["prior"] = self._prior
            if self._prior_keys is not None:
                mv["prior_keys"] = self._prior_keys
        if self._has_constraints:
            mv["has_constraints"] = True
            mv["constraint_lower"] = self._constraint_lower
            mv["constraint_upper"] = self._constraint_upper
            mv["constraint_monotonic"] = self._constraint_monotonic
            mv["constraint_convex"] = self._constraint_convex
            mv["constraint_concave"] = self._constraint_concave
            mv["constraint_order"] = self._constraint_order

        with open(self._filename, "wb") as f:
            pickle.dump(mv, f)

    def _load(self, filename: str) -> None:
        """Load parameters from a previous model fitting session."""
        with open(filename, "rb") as f:
            mv = pickle.load(f)

        self._filename = filename
        self._num_obs = mv["num_obs"]
        self._categories = mv["categories"]
        self._num_categories = mv["num_categories"]
        self._category_hash = mv["category_hash"]
        self._has_l1 = mv["has_l1"]
        if self._has_l1:
            self._lambda1 = mv["lambda1"]
            self._coef1 = mv.get("coef1", 0.0)
        self._has_l2 = mv["has_l2"]
        if self._has_l2:
            self._lambda2 = mv["lambda2"]
            self._coef2 = mv.get("coef2", 0.0)
        self._has_network_lasso = mv["has_network_lasso"]
        if self._has_network_lasso:
            self._num_edges = mv["num_edges"]
            self._D = mv["D"]
            # `edges` was not always persisted; older pickles only stored D.
            if "edges" in mv:
                self._edges = mv["edges"]
            self._lambda_network_lasso = mv["lambda_network_lasso"]
        # has_network_ridge was added later; default to False for older pickles.
        self._has_network_ridge = mv.get("has_network_ridge", False)
        if self._has_network_ridge:
            self._num_edges_ridge = mv["num_edges_ridge"]
            self._L = mv["L"]
            if "edges_ridge" in mv:
                self._edges_ridge = mv["edges_ridge"]
            self._lambda_network_ridge = mv["lambda_network_ridge"]
        else:
            self._lambda_network_ridge = 0.0
        # has_group_lasso was added later; default to False for older pickles.
        self._has_group_lasso = mv.get("has_group_lasso", False)
        if self._has_group_lasso:
            self._lambda_group_lasso = mv["lambda_group_lasso"]
        else:
            self._lambda_group_lasso = 0.0
        # has_group_lasso_inf added later; default to False for older pickles.
        self._has_group_lasso_inf = mv.get("has_group_lasso_inf", False)
        if self._has_group_lasso_inf:
            self._lambda_group_lasso_inf = mv["lambda_group_lasso_inf"]
        else:
            self._lambda_group_lasso_inf = 0.0
        # has_huber added later; default to False for older pickles.
        self._has_huber = mv.get("has_huber", False)
        if self._has_huber:
            self._coef_huber = mv["coef_huber"]
            self._delta_huber = mv["delta_huber"]
            self._lambda_huber_vec = mv["lambda_huber_vec"]
        else:
            self._coef_huber = 0.0
            self._delta_huber = 0.0
            self._lambda_huber_vec = np.zeros(self._num_categories)
        self._has_prior = mv["has_prior"]
        if self._has_prior:
            self._prior = mv["prior"]
            # prior_keys persisted from a later release; older pickles only
            # have the vector form, so reconstruct from non-zero entries.
            if "prior_keys" in mv:
                self._prior_keys = mv["prior_keys"]
            else:
                self._prior_keys = {
                    cat
                    for cat, idx in self._category_hash.items()
                    if self._prior[idx] != 0.0
                }
        else:
            self._prior_keys = None
        # has_constraints added later; default to False for older pickles.
        self._has_constraints = mv.get("has_constraints", False)
        if self._has_constraints:
            self._constraint_lower = mv["constraint_lower"]
            self._constraint_upper = mv["constraint_upper"]
            self._constraint_monotonic = mv["constraint_monotonic"]
            self._constraint_convex = mv["constraint_convex"]
            self._constraint_concave = mv["constraint_concave"]
            self._constraint_order = mv["constraint_order"]
        else:
            self._constraint_lower = -np.inf
            self._constraint_upper = np.inf
            self._constraint_monotonic = None
            self._constraint_convex = False
            self._constraint_concave = False
            self._constraint_order = []
        self._na_index = mv["na_index"]
        self.x = mv["x"]
        self.p = mv["p"]
        self._AtA = mv["AtA"]
        self._ccs = mv.get("ccs", np.zeros(self._num_categories))
        self._verbose = mv["verbose"]
        self._use_cvx = mv["use_cvx"]
        self._solver = mv["solver"]
        self._name = mv["name"]
        self._save_self = mv["save_self"]

    def optimize(self, fpumz: FloatArray, rho: float) -> FloatArray:
        r"""Solve the per-feature primal step via cvxpy.

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
        b = fpumz - self._compute_Az(self.p)
        Atb = self._compute_Atz(b)

        if self._has_l2 and self._has_prior:
            Atb -= (2.0 / rho) * np.multiply(self._lambda2, self._prior)

        q = cvx.Variable(self._num_categories)

        ata_matrix: Any = self._AtA
        if self._has_l2:
            ata_matrix = ata_matrix + sparse.dia_matrix(
                ((2.0 / rho) * self._lambda2, 0),
                shape=(self._num_categories, self._num_categories),
            )
        if self._has_network_ridge:
            ata_matrix = (
                ata_matrix + ((2.0 / rho) * self._lambda_network_ridge) * self._L
            )
        AtA = cvx.Constant(ata_matrix)

        Atb_const = cvx.Constant(Atb)

        obj: Any = cvx.quad_form(q, AtA) + 2.0 * (Atb_const.T @ q)

        c = cvx.Constant(self._ccs)
        constraints: list[Any] = [c.T @ q == 0]

        if self._has_constraints:
            if np.isfinite(self._constraint_lower):
                constraints.append(q >= self._constraint_lower)
            if np.isfinite(self._constraint_upper):
                constraints.append(q <= self._constraint_upper)
            if self._constraint_order:
                idx = [self._category_hash[cat] for cat in self._constraint_order]
                if self._constraint_monotonic == "increasing":
                    for i in range(len(idx) - 1):
                        constraints.append(q[idx[i + 1]] >= q[idx[i]])
                elif self._constraint_monotonic == "decreasing":
                    for i in range(len(idx) - 1):
                        constraints.append(q[idx[i + 1]] <= q[idx[i]])
                if self._constraint_convex:
                    for i in range(1, len(idx) - 1):
                        constraints.append(
                            q[idx[i + 1]] - 2.0 * q[idx[i]] + q[idx[i - 1]] >= 0
                        )
                if self._constraint_concave:
                    for i in range(1, len(idx) - 1):
                        constraints.append(
                            q[idx[i + 1]] - 2.0 * q[idx[i]] + q[idx[i - 1]] <= 0
                        )

        if self._has_l1:
            q_prior = self._prior if self._has_prior else np.zeros(self._num_categories)
            t = cvx.Variable(self._num_categories)
            obj = obj + (self._lambda1 @ t) * (2.0 / rho)
            constraints += [
                t >= 0,
                t + q - q_prior >= 0,
                t - q + q_prior >= 0,
            ]

        if self._has_network_lasso:
            s = cvx.Variable(self._num_edges)
            D = cvx.Constant(self._D)
            obj = obj + (2.0 / rho) * self._lambda_network_lasso * cvx.sum(s)
            constraints += [
                s >= 0,
                s + D @ q >= 0,
                s - D @ q >= 0,
            ]

        if self._has_group_lasso:
            # Group lasso on the per-feature contribution vector
            # f_j = A q in R^n (one entry per observation). Its L2
            # norm equals sqrt(q^T diag(ccs) q) since A is the
            # category-indicator matrix; that's the same as the
            # L2 norm of (sqrt(ccs) elementwise * q). Driving this
            # to zero zeros out the entire feature, giving
            # categorical-variable selection.
            sqrt_ccs = cvx.Constant(np.sqrt(self._ccs))
            obj = obj + (2.0 / rho) * self._lambda_group_lasso * cvx.norm(
                cvx.multiply(sqrt_ccs, q), 2
            )

        if self._has_group_lasso_inf:
            # L_inf-norm group lasso variant: penalize ||A q||_inf =
            # max_{c with obs} |q_c|. Because A is an indicator matrix,
            # categories that never appear in the data don't contribute
            # to ||f_j||_inf -- mask them out so a free-to-roam value at
            # an unused level (e.g., one connected only by network_ridge
            # edges) can't artificially saturate the penalty.
            mask = cvx.Constant((self._ccs > 0).astype(float))
            obj = obj + (2.0 / rho) * self._lambda_group_lasso_inf * cvx.norm_inf(
                cvx.multiply(mask, q)
            )

        if self._has_huber:
            # cvxpy's huber atom is huber(x, M) = x^2 if |x|<=M else
            # 2*M*|x| - M^2, i.e. 2x the standard 0.5-leading Huber.
            # Standard penalty per category is lambda_c * h_delta(q_c) =
            # 0.5 * lambda_c * cvx.huber(q_c, delta); folded into the
            # (2/rho)-scaled obj formulation that's (lambda_c/rho) *
            # cvx.huber(q_c, delta) summed.
            # cvxpy's huber atom is annotated as `M: int = 1` even though
            # the implementation runs `cast_to_const(M)` and accepts any
            # nonnegative scalar. Silence the stub mismatch.
            obj = obj + (1.0 / rho) * (
                cvx.Constant(self._lambda_huber_vec) @ cvx.huber(q, self._delta_huber)  # type: ignore[arg-type]
            )

        prob = cvx.Problem(cvx.Minimize(obj), constraints)
        prob.solve(verbose=self._verbose, solver=self._solver)

        if prob.status not in (cvx.OPTIMAL, cvx.OPTIMAL_INACCURATE):
            raise RuntimeError(
                f"Categorical variable {self._name!r} failed to converge "
                f"(status={prob.status!r})."
            )

        value = q.value
        if value is None:
            raise RuntimeError(
                f"Categorical variable {self._name!r} produced no solution."
            )
        self.p = np.asarray(value, dtype=float).ravel()
        if self._na_index >= 0:
            self.p[self._na_index] = 0.0

        if self._save_self:
            self._save()

        return self._compute_Az(self.p)

    def _compute_Az(self, z: FloatArray) -> FloatArray:
        """Compute ``A @ z`` exploiting the indicator structure of ``A``."""
        m = self._num_obs
        x = self.x
        Az = np.zeros(m)
        for i in range(m):
            Az[i] = z[x[i]]
        return Az

    def _compute_Atz(self, z: FloatArray) -> FloatArray:
        """Compute ``A.T @ z`` exploiting the indicator structure of ``A``."""
        K = self._num_categories
        x = self.x
        Atz = np.zeros(K)
        for i in range(self._num_obs):
            Atz[x[i]] += z[i]
        return Atz

    def compute_dual_tol(self, y: FloatArray) -> float:
        """Compute this feature's contribution to the dual residual tolerance."""
        Aty = self._compute_Atz(y)
        return float(Aty.dot(Aty))

    def _apply_adaptive_l1(self, gamma: float, eps: float) -> bool:
        """Rewrite ``self._coef1`` as adaptive-lasso weights from the pilot.

        For each category that the original L1 configuration touched, the
        new per-category coefficient is ``base_c / (|p_c - prior_c| + eps)
        ** gamma``. Categories that the user did not penalize (``base_c
        == 0``) stay at zero. Returns True iff this feature contributed an
        L1 penalty that was rewritten; the caller uses that to validate
        that adaptive-lasso has something to do.
        """
        if not self._has_l1:
            return False
        base = self._coef1
        if self._has_prior:
            prior_vec = self._prior
        else:
            prior_vec = np.zeros(self._num_categories)
        new_coef: dict[Any, float] = {}
        rewrote = False
        for cat, idx in self._category_hash.items():
            if isinstance(base, dict):
                base_c = float(base.get(cat, 0.0))
            else:
                base_c = float(base)
            if base_c == 0.0:
                continue
            if self._prior_keys is not None and cat not in self._prior_keys:
                continue
            deviation = abs(float(self.p[idx]) - float(prior_vec[idx]))
            new_coef[cat] = base_c / (deviation + eps) ** gamma
            rewrote = True
        self._coef1 = new_coef
        return rewrote

    def num_params(self) -> int:
        """Number of parameters in this feature."""
        return len(self.p)

    def category_index(self, observation: int) -> tuple[int, int]:
        """Return ``(category_index, num_categories)`` for the given training-row index."""
        return int(self.x[observation]), self._num_categories

    def dof(self) -> float:
        r"""Effective degrees of freedom contributed by this feature.

        The count reflects the *fitted* coefficient vector, not just the
        category count, so the value shrinks under any regularization
        that compresses the fit:

        - Unregularized: ``K - 1`` (one zero-sum constraint).
        - **L1**: number of nonzero coefficients (active set), minus the
          zero-sum constraint when at least two are active.
        - **L2 / ridge**: trace of the hat matrix
          :math:`\mathrm{tr}((A^T A + \mathrm{diag}(\lambda))^{-1} A^T A)
          = \sum_c n_c / (n_c + \lambda_c)`, minus the zero-sum constraint.
          Tends to ``K - 1`` as :math:`\lambda \to 0` and to 0 as
          :math:`\lambda \to \infty`.
        - **Huber**: per-category mix of the L2 trace contribution
          (categories with :math:`|q_c| \le \delta`) and an active-set
          contribution of 1 (categories with :math:`|q_c| > \delta`).
        - **Network ridge**: trace of
          :math:`(A^T A + \lambda L)^{-1} A^T A` with the (weighted)
          graph Laplacian :math:`L`, minus the zero-sum constraint.
          Combines additively with an active L2 penalty.
        - **Network lasso**: number of *distinct* fitted values within
          numerical tolerance, minus the zero-sum constraint. Counts
          the actual fused groups produced by the fit.
        - **Group lasso / group_lasso_inf**: 0 if the entire fit is
          zero (the feature got selected out), else the unregularized
          ``K - 1``.

        For combinations of penalties the formula above for the most
        binding penalty applies (``network_lasso`` dominates if active,
        followed by ``network_ridge``, then the per-category penalties).
        """
        if self._num_categories <= 1:
            return 0.0

        K = int(self._num_categories)
        p = np.asarray(self.p, dtype=float)
        zero_sum_credit = 1.0  # We subtract one for the zero-sum constraint
        # whenever the active subspace has at least two free directions.

        # Group-lasso variants are all-or-nothing on this feature: if the
        # whole vector got zeroed out, there is no fitted parameter.
        if (self._has_group_lasso or self._has_group_lasso_inf) and np.max(
            np.abs(p)
        ) < 1e-8:
            return 0.0

        # Network lasso fuses neighbors to identical values. The active
        # parameter count is the number of distinct fitted values, since
        # every fused cluster shares one coefficient.
        if self._has_network_lasso:
            tol = 1e-6
            unique_count = int(np.unique(np.round(p / tol).astype(np.int64)).size)
            return float(max(unique_count - zero_sum_credit, 0.0))

        ata_diag = np.asarray(self._AtA.diagonal(), dtype=float)

        # Network ridge: tr((A^T A + λL + diag(λ_2))^{-1} A^T A).
        # Combines additively with L2 if both are present.
        if self._has_network_ridge:
            M = self._AtA.tocsr() + float(self._lambda_network_ridge) * self._L.tocsr()
            if self._has_l2:
                M = M + sparse.diags(self._lambda2, format="csr")
            edof = _trace_Minv_diag(M, ata_diag)
            return float(max(edof - zero_sum_credit, 0.0))

        # Huber: per-category mix of L2 trace and L1 active-set count.
        if self._has_huber:
            in_l2 = np.abs(p) <= self._delta_huber
            denom = ata_diag + self._lambda_huber_vec
            denom = np.where(denom > 0, denom, 1.0)
            l2_part = np.where(in_l2, ata_diag / denom, 0.0).sum()
            l1_active = int(np.sum((~in_l2) & (np.abs(p) > 1e-8)))
            edof = float(l2_part) + float(l1_active)
            return float(max(edof - zero_sum_credit, 0.0))

        # L2 ridge: closed-form trace formula.
        if self._has_l2:
            denom = ata_diag + self._lambda2
            # Categories with no observations and no penalty contribute 0.
            denom = np.where(denom > 0, denom, 1.0)
            edof = float(np.sum(ata_diag / denom))
            return float(max(edof - zero_sum_credit, 0.0))

        # L1 lasso: active-set count.
        if self._has_l1:
            active = int(np.sum(np.abs(p) > 1e-8))
            if active <= 1:
                return float(active)
            return float(active - zero_sum_credit)

        # Plain unregularized categorical (or only group-lasso with a
        # nonzero fit): K - 1.
        return float(K - 1)

    def predict(self, X: npt.NDArray[Any]) -> FloatArray:
        """Apply fitted model to feature.

        Categories not seen during ``fit()`` are predicted with effect 0
        (the average across training categories, by the zero-sum
        constraint) and trigger a ``UserWarning`` listing them.
        """
        prediction = np.zeros(len(X))
        unseen: set[Any] = set()
        for i in range(len(X)):
            value = X[i]
            if value in self._category_hash:
                prediction[i] = self.p[self._category_hash[value]]
            else:
                unseen.add(value)
        if unseen:
            sample = sorted(unseen, key=str)[:5]
            suffix = "" if len(unseen) <= 5 else f", +{len(unseen) - 5} more"
            warnings.warn(
                f"Feature {self._name!r}: unseen categories predicted "
                f"with effect 0: {sample}{suffix}",
                UserWarning,
                stacklevel=2,
            )
        return prediction

    def _plot(self, true_fn: Any = None) -> None:
        """Plot is not yet implemented for categorical features."""
        return None

    def __str__(self) -> str:
        desc = f"Feature {self._name}\n"
        for cat in self._categories:
            desc += f"  {cat}: {self.p[self._category_hash[cat]]:.06g}\n"
        return desc


def _trace_Minv_diag(M: sparse.spmatrix, diag: FloatArray) -> float:
    r"""Compute :math:`\mathrm{tr}(M^{-1} \mathrm{diag}(d))`.

    Used by ``_CategoricalFeature.dof()`` to evaluate the trace-form
    effective degrees of freedom under network-ridge or L2 + Laplacian
    penalties without materializing :math:`M^{-1}` densely. The trace
    equals :math:`\sum_c d_c \cdot (M^{-1})_{cc}`, so we factor :math:`M`
    once and read off the diagonal of :math:`M^{-1}` by solving against
    the identity column-by-column.
    """
    K = int(M.shape[0])
    lu = splu(M.tocsc())
    diag_inv = np.zeros(K, dtype=float)
    e = np.zeros(K, dtype=float)
    for c in range(K):
        e[c] = 1.0
        diag_inv[c] = float(lu.solve(e)[c])
        e[c] = 0.0
    return float(np.sum(diag * diag_inv))
