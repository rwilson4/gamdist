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

import pickle
from typing import Any

import cvxpy as cvx
import numpy as np
import numpy.typing as npt
from scipy import sparse

from .feature import _Feature

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


class _CategoricalFeature(_Feature):
    """A categorical feature: contributes a per-level offset to the predictor."""

    def __init__(
        self,
        name: str | None = None,
        regularization: dict[str, Any] | None = None,
        load_from_file: str | None = None,
    ) -> None:
        """Initialize feature model (independent of data).

        Parameters
        ----------
        name : str
            Name for feature, used to make plots.
        regularization : dict
            Description of regularization terms (``l1``, ``l2``,
            ``network_lasso``). See the package README for details.
        load_from_file : str
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
        self._solver = "ECOS"
        self._categories: list[Any] = []
        self._has_l1 = False
        self._has_l2 = False
        self._has_network_lasso = False
        self._has_prior = False
        self._coef1: float | dict[Any, float] = 0.0
        self._coef2: float | dict[Any, float] = 0.0
        self._lambda_network_lasso: float = 0.0
        self._num_edges: int = 0
        if regularization is not None:
            if "l1" in regularization:
                self._has_l1 = True
                if "coef" in regularization["l1"]:
                    self._coef1 = regularization["l1"]["coef"]
                else:
                    raise ValueError("No coefficient specified for l1 regularization term.")

            if "l2" in regularization:
                self._has_l2 = True
                if "coef" in regularization["l2"]:
                    self._coef2 = regularization["l2"]["coef"]
                else:
                    raise ValueError("No coefficient specified for l2 regularization term.")

            if "network_lasso" in regularization:
                self._has_network_lasso = True
                if "coef" in regularization["network_lasso"]:
                    self._lambda_network_lasso = float(regularization["network_lasso"]["coef"])
                else:
                    raise ValueError(
                        "No coefficient specified for Network Lasso regularization term."
                    )

                if "edges" in regularization["network_lasso"]:
                    self._edges = regularization["network_lasso"]["edges"]
                    self._num_edges, _ = self._edges.shape
                    for _, row in self._edges.iterrows():
                        if row["country1"] not in self._categories:
                            self._categories.append(row["country1"])
                        if row["country2"] not in self._categories:
                            self._categories.append(row["country2"])
                else:
                    raise ValueError(
                        "Edges not specified for Network Lasso regularization term."
                    )

            if (self._has_l1 or self._has_l2) and "prior" in regularization:
                self._has_prior = True
                self._prior: Any = regularization["prior"]

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
        """Compute feature-specific state from data."""
        self._num_obs = len(x)
        self._verbose = verbose

        self._categories = list(set(x).union(self._categories))
        self._num_categories = len(self._categories)
        self._category_hash: dict[Any, int] = {
            key: i for (key, i) in zip(self._categories, range(self._num_categories), strict=True)
        }

        if self._has_network_lasso:
            D = np.zeros((self._num_edges, self._num_categories))
            _, em = self._edges.shape
            ir = 0
            for _, row in self._edges.iterrows():
                i = self._category_hash[row["country1"]]
                j = self._category_hash[row["country2"]]
                lmbda = float(row["weight"]) if em >= 3 else 1.0
                D[ir, i] = lmbda
                D[ir, j] = -lmbda
                ir += 1
            self._D = sparse.coo_matrix(D).tocsr()
            self._lambda_network_lasso *= smoothing

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

        if (self._has_l1 or self._has_l2) and self._has_prior:
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
            self._ccs = self._compute_Atz(np.asarray(covariate_class_sizes, dtype=float))

        if self._verbose:
            print(f"Number of categories: {self._num_categories:d}")
            if self._has_network_lasso:
                print(f"Number of edges: {self._num_edges:d}")

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

    def _compute_lambda(self, coef: float | dict[Any, float], smoothing: float) -> FloatArray:
        """Build the per-category regularization-weight vector."""
        result = np.zeros(self._num_categories)
        if isinstance(coef, (int, float)):
            scaled = float(coef) * smoothing
            if self._has_prior:
                for key in self._prior:
                    result[self._category_hash[key]] = scaled
            else:
                result[:] = scaled
        else:
            for key, value in coef.items():
                if self._has_prior and key not in self._prior:
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
            "has_network_lasso": self._has_network_lasso,
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
            mv["lambda_network_lasso"] = self._lambda_network_lasso
        if self._has_prior:
            mv["prior"] = self._prior

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
            self._lambda_network_lasso = mv["lambda_network_lasso"]
        self._has_prior = mv["has_prior"]
        if self._has_prior:
            self._prior = mv["prior"]
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
        """Solve the per-feature primal step via cvxpy.

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
        b = fpumz - self._compute_Az(self.p)
        Atb = self._compute_Atz(b)

        if self._has_l2 and self._has_prior:
            Atb -= (2.0 / rho) * np.multiply(self._lambda2, self._prior)

        q = cvx.Variable(self._num_categories)

        if self._has_l2:
            AtA = cvx.Constant(
                self._AtA
                + sparse.dia_matrix(
                    ((2.0 / rho) * self._lambda2, 0),
                    shape=(self._num_categories, self._num_categories),
                )
            )
        else:
            AtA = cvx.Constant(self._AtA)

        Atb_const = cvx.Constant(Atb)

        obj: Any = cvx.quad_form(q, AtA) + 2.0 * (Atb_const.T @ q)

        c = cvx.Constant(self._ccs)
        constraints: list[Any] = [c.T @ q == 0]

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

        prob = cvx.Problem(cvx.Minimize(obj), constraints)
        prob.solve(verbose=self._verbose, solver=self._solver)

        if prob.status not in (cvx.OPTIMAL, cvx.OPTIMAL_INACCURATE):
            raise RuntimeError(
                f"Categorical variable {self._name!r} failed to converge "
                f"(status={prob.status!r})."
            )

        value = q.value
        if value is None:
            raise RuntimeError(f"Categorical variable {self._name!r} produced no solution.")
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

    def num_params(self) -> int:
        """Number of parameters in this feature."""
        return int(len(self.p))

    def category_index(self, observation: int) -> tuple[int, int]:
        """Return ``(category_index, num_categories)`` for the given training-row index."""
        return int(self.x[observation]), self._num_categories

    def dof(self) -> float:
        """Effective degrees of freedom contributed by this feature."""
        return float(len(self.p) - 1)

    def predict(self, X: npt.NDArray[Any]) -> FloatArray:
        """Apply fitted model to feature."""
        prediction = np.zeros(len(X))
        for i in range(len(X)):
            if X[i] in self._category_hash:
                prediction[i] = self.p[self._category_hash[X[i]]]
        return prediction

    def _plot(self, true_fn: Any = None) -> None:
        """Plot is not yet implemented for categorical features."""
        return None

    def __str__(self) -> str:
        desc = f"Feature {self._name}\n"
        for cat in self._categories:
            desc += f"  {cat}: {self.p[self._category_hash[cat]]:.06g}\n"
        return desc
