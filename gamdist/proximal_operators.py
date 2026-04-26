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

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize_scalar
from scipy.special import expit

FloatArray = npt.NDArray[np.float64]
InvLink = Callable[[float], float]


def _prox_normal_identity(
    v: FloatArray,
    mu: float,
    y: FloatArray,
    w: FloatArray | None = None,
    inv_link: InvLink | None = None,
    p: Any = None,
) -> FloatArray:
    if w is None:
        return (y + mu * v) / (1.0 + mu)
    return (w * y + mu * v) / (w + mu)


def _prox_normal(
    v: FloatArray,
    mu: float,
    y: FloatArray,
    w: FloatArray | None = None,
    inv_link: InvLink | None = None,
    p: Any = None,
) -> FloatArray:
    if inv_link is None:
        raise ValueError("inv_link is required.")

    if w is None:

        def obj_fun(x: float, _v: float, _y: float) -> float:
            ilx = inv_link(x)
            return 0.5 * ilx * ilx - _y * ilx + 0.5 * mu * (x - _v) * (x - _v)

        return np.array(
            [minimize_scalar(obj_fun, args=(v[i], y[i])).x for i in range(len(v))]
        )

    def obj_fun_w(x: float, _v: float, _y: float, _w: float) -> float:
        ilx = inv_link(x)
        return _w * (0.5 * ilx * ilx - _y * ilx) + 0.5 * mu * (x - _v) * (x - _v)

    return np.array(
        [minimize_scalar(obj_fun_w, args=(v[i], y[i], w[i])).x for i in range(len(v))]
    )


def _prox_binomial_logit_scalar(xx: Sequence[float]) -> float:
    v = float(xx[0])
    mu = float(xx[1])
    y = float(xx[2])
    m = float(xx[3])
    w_eff = float(xx[4]) if len(xx) >= 5 else 1.0

    def obj(z: float) -> float:
        # log1p(exp(z)) computed stably as np.logaddexp(0, z).
        return (
            w_eff * m * float(np.logaddexp(0.0, z))
            - w_eff * y * z
            + 0.5 * mu * (z - v) * (z - v)
        )

    tol = 1e-3
    max_its = 100
    x = 0.0
    for _ in range(max_its):
        sigma = float(expit(x))
        grad = mu * (x - v) + w_eff * (m * sigma - y)
        hess = mu + w_eff * m * sigma * (1.0 - sigma)
        dx = grad / hess
        if abs(dx) < tol:
            return x
        # Damped Newton: backtrack until the objective decreases.
        f_curr = obj(x)
        step = 1.0
        x_new = x - dx
        while obj(x_new) >= f_curr and step > 1e-12:
            step *= 0.5
            x_new = x - step * dx
        x = x_new

    # Newton failed to converge in max_its; fall back to a bracketed
    # 1-D minimization. The minimizer is convex in z, so this always
    # finds the same optimum that Newton would have reached.
    return float(minimize_scalar(obj).x)


def _prox_binomial_logit(
    v: FloatArray,
    mu: float,
    y: FloatArray,
    ccs: FloatArray | None = None,
    w: FloatArray | None = None,
    inv_link: InvLink | None = None,
    p: Any = None,
) -> FloatArray:
    m = np.ones(y.shape) if ccs is None else ccs
    mu_arr = np.full(v.shape, mu)
    items: list[Sequence[float]]
    if w is None:
        items = [tuple(t) for t in zip(v, mu_arr, y, m, strict=True)]
    else:
        items = [tuple(t) for t in zip(v, mu_arr, y, m, w, strict=True)]
    return np.array([_prox_binomial_logit_scalar(item) for item in items])


def _prox_binomial(
    v: FloatArray,
    mu: float,
    y: FloatArray,
    ccs: FloatArray | None = None,
    w: FloatArray | None = None,
    inv_link: InvLink | None = None,
    p: Any = None,
) -> FloatArray:
    if inv_link is None:
        raise ValueError("inv_link is required.")

    if w is None:

        def obj_fun(x: float, _v: float, _y: float) -> float:
            ilx = inv_link(x)
            m = 1.0
            return (_y - m) * np.log1p(-ilx) - _y * np.log(ilx) + 0.5 * mu * (x - _v) * (x - _v)

        return np.array(
            [minimize_scalar(obj_fun, args=(v[i], y[i])).x for i in range(len(v))]
        )

    def obj_fun_w(x: float, _v: float, _y: float, _w: float) -> float:
        ilx = inv_link(x)
        m = 1.0
        return (
            _w * (_y - m) * np.log1p(-ilx)
            - _w * _y * np.log(ilx)
            + 0.5 * mu * (x - _v) * (x - _v)
        )

    return np.array(
        [minimize_scalar(obj_fun_w, args=(v[i], y[i], w[i])).x for i in range(len(v))]
    )


def _prox_poisson_log_scalar(xx: Sequence[float]) -> float:
    v = float(xx[0])
    mu = float(xx[1])
    y = float(xx[2])
    w_eff = float(xx[3]) if len(xx) >= 4 else 1.0

    def obj(z: float) -> float:
        return w_eff * float(np.exp(z)) - w_eff * y * z + 0.5 * mu * (z - v) * (z - v)

    tol = 1e-3
    max_its = 100
    x = 0.0
    for _ in range(max_its):
        # Cap exp(x) below the float64 overflow threshold so a transient
        # large step doesn't blow up the gradient computation. Damping
        # below pulls x back into a sensible range.
        expx = float(np.exp(min(x, 700.0)))
        grad = mu * (x - v) + w_eff * (expx - y)
        hess = mu + w_eff * expx
        dx = grad / hess
        if abs(dx) < tol:
            return x
        f_curr = obj(x)
        step = 1.0
        x_new = x - dx
        while obj(x_new) >= f_curr and step > 1e-12:
            step *= 0.5
            x_new = x - step * dx
        x = x_new

    return float(minimize_scalar(obj).x)


def _prox_poisson_log(
    v: FloatArray,
    mu: float,
    y: FloatArray,
    w: FloatArray | None = None,
    inv_link: InvLink | None = None,
    p: Any = None,
) -> FloatArray:
    mu_arr = np.full(v.shape, mu)
    items: list[Sequence[float]]
    if w is None:
        items = [tuple(t) for t in zip(v, mu_arr, y, strict=True)]
    else:
        items = [tuple(t) for t in zip(v, mu_arr, y, w, strict=True)]
    return np.array([_prox_poisson_log_scalar(item) for item in items])


def _prox_poisson(
    v: FloatArray,
    mu: float,
    y: FloatArray,
    w: FloatArray | None = None,
    inv_link: InvLink | None = None,
    p: Any = None,
) -> FloatArray:
    if inv_link is None:
        raise ValueError("inv_link is required.")

    if w is None:

        def obj_fun(x: float, _v: float, _y: float) -> float:
            ilx = inv_link(x)
            return (ilx - _y * np.log(ilx)) + 0.5 * mu * (x - _v) * (x - _v)

        return np.array(
            [minimize_scalar(obj_fun, args=(v[i], y[i])).x for i in range(len(v))]
        )

    def obj_fun_w(x: float, _v: float, _y: float, _w: float) -> float:
        ilx = inv_link(x)
        return _w * (ilx - _y * np.log(ilx)) + 0.5 * mu * (x - _v) * (x - _v)

    return np.array(
        [minimize_scalar(obj_fun_w, args=(v[i], y[i], w[i])).x for i in range(len(v))]
    )


def _prox_gamma_reciprocal(
    v: FloatArray,
    mu: float,
    y: FloatArray,
    w: FloatArray | None = None,
    inv_link: InvLink | None = None,
    p: Any = None,
) -> FloatArray:
    if w is None:
        mu_v_minus_w_y = mu * v - y
        return (0.5 / mu) * mu_v_minus_w_y + np.sqrt(
            mu_v_minus_w_y * mu_v_minus_w_y + (4 * mu)
        )
    mu_v_minus_w_y = mu * v - w * y
    return (0.5 / mu) * mu_v_minus_w_y + np.sqrt(
        mu_v_minus_w_y * mu_v_minus_w_y + (4 * mu) * w
    )


def _prox_gamma(
    v: FloatArray,
    mu: float,
    y: FloatArray,
    w: FloatArray | None = None,
    inv_link: InvLink | None = None,
    p: Any = None,
) -> FloatArray:
    if inv_link is None:
        raise ValueError("inv_link is required.")

    if w is None:

        def obj_fun(x: float, _v: float, _y: float) -> float:
            ilx = inv_link(x)
            return (np.log(ilx) + _y / ilx) + 0.5 * mu * (x - _v) * (x - _v)

        return np.array(
            [minimize_scalar(obj_fun, args=(v[i], y[i])).x for i in range(len(v))]
        )

    def obj_fun_w(x: float, _v: float, _y: float, _w: float) -> float:
        ilx = inv_link(x)
        return _w * (np.log(ilx) + _y / ilx) + 0.5 * mu * (x - _v) * (x - _v)

    return np.array(
        [minimize_scalar(obj_fun_w, args=(v[i], y[i], w[i])).x for i in range(len(v))]
    )


def _prox_inv_gaussian_reciprocal_squared_scalar(xx: Sequence[float]) -> float:
    v = xx[0]
    mu = xx[1]
    y = xx[2]
    w: float | None = xx[3] if len(xx) >= 4 else None

    tol = 1e-3
    max_its = 100

    z = 1.0
    if w is None:
        w_y_minus_mu_v = 0.5 * y - mu * v
    else:
        w_y_minus_mu_v = 0.5 * w * y - mu * v

    for _ in range(max_its):
        if w is None:
            num = mu * z * z * z + w_y_minus_mu_v * z - 0.5
            denom = 3 * mu * z * z + w_y_minus_mu_v
        else:
            num = mu * z * z * z + w_y_minus_mu_v * z - 0.5 * w
            denom = 3 * mu * z * z + w_y_minus_mu_v

        dz = num / denom
        z -= dz
        if abs(dz) < tol:
            return z * z

    raise ValueError("Dual variable update failed to converge.")


def _prox_inv_gaussian_reciprocal_squared(
    v: FloatArray,
    mu: float,
    y: FloatArray,
    w: FloatArray | None = None,
    inv_link: InvLink | None = None,
    p: Any = None,
) -> FloatArray:
    mu_arr = np.full(v.shape, mu)
    items: list[Sequence[float]]
    if w is None:
        items = [tuple(t) for t in zip(v, mu_arr, y, strict=True)]
    else:
        items = [tuple(t) for t in zip(v, mu_arr, y, w, strict=True)]
    return np.array([_prox_inv_gaussian_reciprocal_squared_scalar(item) for item in items])


def _prox_inv_gaussian(
    v: FloatArray,
    mu: float,
    y: FloatArray,
    w: FloatArray | None = None,
    inv_link: InvLink | None = None,
    p: Any = None,
) -> FloatArray:
    if inv_link is None:
        raise ValueError("inv_link is required.")

    if w is None:

        def obj_fun(x: float, _v: float, _y: float) -> float:
            ilx = inv_link(x)
            return (-1.0 / ilx + 0.5 * _y * ilx * ilx) + 0.5 * mu * (x - _v) * (x - _v)

        return np.array(
            [minimize_scalar(obj_fun, args=(v[i], y[i])).x for i in range(len(v))]
        )

    def obj_fun_w(x: float, _v: float, _y: float, _w: float) -> float:
        ilx = inv_link(x)
        return _w * (-1.0 / ilx + 0.5 * _y * ilx * ilx) + 0.5 * mu * (x - _v) * (x - _v)

    return np.array(
        [minimize_scalar(obj_fun_w, args=(v[i], y[i], w[i])).x for i in range(len(v))]
    )
