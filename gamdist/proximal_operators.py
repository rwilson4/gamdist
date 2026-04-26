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
            return (
                (_y - m) * np.log1p(-ilx)
                - _y * np.log(ilx)
                + 0.5 * mu * (x - _v) * (x - _v)
            )

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
    v = float(xx[0])
    mu = float(xx[1])
    y = float(xx[2])
    w_eff = float(xx[3]) if len(xx) >= 4 else 1.0

    # Parameterize in eta = z^2, the natural convex coordinate:
    #   F(eta)   = 0.5*w*y*eta - w*sqrt(eta) + 0.5*mu*(eta - v)^2
    #   F'(eta)  = 0.5*w*y - 0.5*w/sqrt(eta) + mu*(eta - v)
    #   F''(eta) = 0.25*w*eta^(-3/2) + mu  > 0  for w, mu, eta > 0,
    # so Newton from any positive eta converges monotonically. The only
    # safeguard needed is to keep eta strictly positive across steps.
    tol = 1e-3
    max_its = 100

    eta = 1.0
    for _ in range(max_its):
        sqrt_eta = np.sqrt(eta)
        grad = 0.5 * w_eff * y - 0.5 * w_eff / sqrt_eta + mu * (eta - v)
        hess = 0.25 * w_eff / (eta * sqrt_eta) + mu
        deta = grad / hess
        if abs(deta) < tol:
            return eta
        # If the full Newton step would push eta non-positive, halve it
        # until it lands in the open positive half-line. The Hessian is
        # globally positive, so a sufficiently short step in the Newton
        # direction is always a descent direction.
        step = 1.0
        eta_new = eta - deta
        while eta_new <= 0.0:
            step *= 0.5
            eta_new = eta - step * deta
        eta = eta_new

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
    return np.array(
        [_prox_inv_gaussian_reciprocal_squared_scalar(item) for item in items]
    )


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


def _prox_huber_identity(
    v: FloatArray,
    mu: float,
    y: FloatArray,
    delta: float,
    w: FloatArray | None = None,
    inv_link: InvLink | None = None,
    p: Any = None,
) -> FloatArray:
    r"""Proximal operator for the Huber loss with identity link.

    Solves, elementwise,
        argmin_x  w * L_delta(y - x) + (mu/2) * (x - v)^2
    where ``L_delta(r) = 0.5 r^2`` for ``|r| <= delta`` and
    ``L_delta(r) = delta * (|r| - 0.5*delta)`` for ``|r| > delta`` is the
    Huber loss with knee ``delta > 0``. The minimum has a closed form:
    a clipped weighted-mean shrinkage, with the linear-region branches
    saturating at ``v +/- w*delta/mu``.
        x* = v + w*delta/mu                 if y - v >   delta*(mu+w)/mu
        x* = v - w*delta/mu                 if y - v <  -delta*(mu+w)/mu
        x* = (w*y + mu*v) / (w + mu)        otherwise.
    Reduces to the normal-identity prox in the inner (quadratic) region
    and to a soft-threshold-style clip in the outer (linear) region;
    bounded influence is what makes Huber robust to outliers.
    """
    if w is None:
        threshold: FloatArray | float = delta * (1.0 + mu) / mu
        upper = v + delta / mu
        lower = v - delta / mu
        quad = (y + mu * v) / (1.0 + mu)
    else:
        threshold = delta * (w + mu) / mu
        upper = v + (w * delta) / mu
        lower = v - (w * delta) / mu
        quad = (w * y + mu * v) / (w + mu)
    diff = y - v
    return np.where(diff > threshold, upper, np.where(diff < -threshold, lower, quad))


def _prox_quantile_identity(
    v: FloatArray,
    mu: float,
    y: FloatArray,
    tau: float,
    w: FloatArray | None = None,
    inv_link: InvLink | None = None,
    p: Any = None,
) -> FloatArray:
    r"""Proximal operator for the pinball / check loss with identity link.

    Solves, elementwise,
        argmin_x  w * rho_tau(y - x) + (mu/2) * (x - v)^2
    where ``rho_tau(r) = max(tau * r, (tau - 1) * r)`` is the pinball loss
    parameterized by quantile level ``tau in (0, 1)``. The minimum is
    attained in closed form via a shifted soft-threshold:
        x* = v + w*tau/mu          if y > v + w*tau/mu        (overshoot)
        x* = v - w*(1-tau)/mu      if y < v - w*(1-tau)/mu    (undershoot)
        x* = y                     otherwise.
    """
    if w is None:
        upper = v + tau / mu
        lower = v - (1.0 - tau) / mu
    else:
        upper = v + (w * tau) / mu
        lower = v - (w * (1.0 - tau)) / mu
    return np.where(y > upper, upper, np.where(y < lower, lower, y))
