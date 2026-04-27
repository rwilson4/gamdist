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

"""Proximal operators for each supported ``(family, link)`` pair.

The ADMM dual step in :meth:`gamdist.GAM._optimize` calls the proximal
operator of the per-observation negative log-likelihood

.. math::

   \\mathrm{prox}_\\mu(v) := \\arg\\min_x\\, L(x) + \\frac{\\mu}{2}\\,
   \\| x - v \\|_2^2

for the model's family and link. Per the convexity-only design rule
(CLAUDE.md), every dispatch entry corresponds to a convex
subproblem and a dedicated solver:

* normal + identity, gamma + reciprocal, quantile + identity, and
  huber + identity have closed forms.
* binomial + logit, poisson + log, and inverse-gaussian +
  reciprocal-squared use damped Newton iterations on the scalar
  dual update with a backtracking safeguard.

All operators broadcast over their ``v``, ``y`` array inputs. Optional
covariate-class sizes (``ccs``), observation weights (``w``), and the
shape parameter for huber / quantile losses are passed through
keyword arguments.
"""

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
    r"""Proximal operator for the Gaussian negative log-likelihood with
    identity link.

    Closed-form weighted-mean shrinkage:

    .. math::

       x^\ast = \frac{w y + \mu v}{w + \mu},

    with :math:`w = 1` when ``w`` is ``None``. ``inv_link`` and ``p``
    are accepted only to keep a uniform dispatch signature with the
    iterative prox operators.
    """
    if w is None:
        return (y + mu * v) / (1.0 + mu)
    return (w * y + mu * v) / (w + mu)


def _prox_binomial_logit_scalar(xx: Sequence[float]) -> float:
    r"""Scalar proximal operator for binomial + logit on one observation.

    Solves

    .. math::

       \min_z\, w m\, \log(1 + e^z) - w y z +
       \tfrac{1}{2} \mu (z - v)^2

    via damped Newton with a backtracking line search, falling back to
    :func:`scipy.optimize.minimize_scalar` only if Newton fails to
    converge. ``xx`` packs ``(v, mu, y, m[, w])``; ``w`` defaults to 1
    when omitted. Convex in :math:`z`, so the iteration is globally
    convergent for any starting point.
    """
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
    """Vectorized proximal operator for the binomial + logit family.

    Iterates :func:`_prox_binomial_logit_scalar` over every observation.
    ``ccs`` are the covariate-class sizes (``m_i``); when ``None`` each
    observation has ``m_i = 1`` (Bernoulli data). ``w`` is the
    optional per-observation weight vector. ``inv_link`` and ``p`` are
    accepted to keep a uniform dispatch signature.
    """
    m = np.ones(y.shape) if ccs is None else ccs
    mu_arr = np.full(v.shape, mu)
    items: list[Sequence[float]]
    if w is None:
        items = [tuple(t) for t in zip(v, mu_arr, y, m, strict=True)]
    else:
        items = [tuple(t) for t in zip(v, mu_arr, y, m, w, strict=True)]
    return np.array([_prox_binomial_logit_scalar(item) for item in items])


def _prox_poisson_log_scalar(xx: Sequence[float]) -> float:
    r"""Scalar proximal operator for Poisson + log on one observation.

    Solves

    .. math::

       \min_z\, w e^z - w y z + \tfrac{1}{2} \mu (z - v)^2

    via damped Newton with backtracking, falling back to
    :func:`scipy.optimize.minimize_scalar` only if Newton fails. ``xx``
    packs ``(v, mu, y[, w])``. Convex in :math:`z`.
    """
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
    """Vectorized proximal operator for the Poisson + log family.

    Iterates :func:`_prox_poisson_log_scalar` over every observation.
    """
    mu_arr = np.full(v.shape, mu)
    items: list[Sequence[float]]
    if w is None:
        items = [tuple(t) for t in zip(v, mu_arr, y, strict=True)]
    else:
        items = [tuple(t) for t in zip(v, mu_arr, y, w, strict=True)]
    return np.array([_prox_poisson_log_scalar(item) for item in items])


def _prox_gamma_reciprocal(
    v: FloatArray,
    mu: float,
    y: FloatArray,
    w: FloatArray | None = None,
    inv_link: InvLink | None = None,
    p: Any = None,
) -> FloatArray:
    r"""Proximal operator for the gamma + reciprocal family.

    Closed form derived from the quadratic in :math:`x`:

    .. math::

       x^\ast = \frac{\mu v - w y}{2 \mu} +
       \sqrt{\Bigl(\frac{\mu v - w y}{2 \mu}\Bigr)^2 +
       \frac{w}{\mu}}.

    With ``w`` set to 1 when ``None``.
    """
    if w is None:
        mu_v_minus_w_y = mu * v - y
        return (0.5 / mu) * mu_v_minus_w_y + np.sqrt(
            mu_v_minus_w_y * mu_v_minus_w_y + (4 * mu)
        )
    mu_v_minus_w_y = mu * v - w * y
    return (0.5 / mu) * mu_v_minus_w_y + np.sqrt(
        mu_v_minus_w_y * mu_v_minus_w_y + (4 * mu) * w
    )


def _prox_inv_gaussian_reciprocal_squared_scalar(xx: Sequence[float]) -> float:
    r"""Scalar proximal operator for inverse-gaussian + reciprocal-squared.

    Reparameterizes in :math:`\eta = z^2`, where the objective becomes

    .. math::

       F(\eta) = \tfrac{1}{2} w y \eta - w \sqrt{\eta} +
       \tfrac{1}{2} \mu (\eta - v)^2,

    which has positive Hessian everywhere on
    :math:`\eta > 0`, so Newton from any positive starting point
    converges. The only safeguard is keeping :math:`\eta` strictly
    positive across steps. Raises :class:`ValueError` if Newton fails
    to converge in 100 iterations.
    """
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
    """Vectorized prox for the inverse-gaussian + reciprocal-squared family.

    Iterates :func:`_prox_inv_gaussian_reciprocal_squared_scalar` over
    every observation.
    """
    mu_arr = np.full(v.shape, mu)
    items: list[Sequence[float]]
    if w is None:
        items = [tuple(t) for t in zip(v, mu_arr, y, strict=True)]
    else:
        items = [tuple(t) for t in zip(v, mu_arr, y, w, strict=True)]
    return np.array(
        [_prox_inv_gaussian_reciprocal_squared_scalar(item) for item in items]
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

    .. math::

       \arg\min_x\, w L_\delta(y - x) +
       \tfrac{1}{2} \mu (x - v)^2,

    where :math:`L_\delta` is the Huber loss with knee
    :math:`\delta > 0`,

    .. math::

       L_\delta(r) = \begin{cases}
       \tfrac{1}{2} r^2 & |r| \le \delta, \\
       \delta\,(|r| - \tfrac{1}{2} \delta) & |r| > \delta.
       \end{cases}

    The minimum has a closed form: a clipped weighted-mean
    shrinkage, with the linear-region branches saturating at
    :math:`v \pm w \delta / \mu`,

    .. math::

       x^\ast = \begin{cases}
       v + w \delta / \mu
         & y - v >  \delta (\mu + w) / \mu, \\
       v - w \delta / \mu
         & y - v < -\delta (\mu + w) / \mu, \\
       (w y + \mu v) / (w + \mu)
         & \text{otherwise.}
       \end{cases}

    This reduces to :func:`_prox_normal_identity` in the inner
    (quadratic) region and to a soft-threshold-style clip in the
    outer (linear) region; bounded influence is what makes Huber
    robust to outliers.
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

    .. math::

       \arg\min_x\, w \rho_\tau(y - x) +
       \tfrac{1}{2} \mu (x - v)^2,

    where :math:`\rho_\tau(r) = \max(\tau r, (\tau - 1) r)` is the
    pinball loss parameterized by quantile level
    :math:`\tau \in (0, 1)`. The minimum is attained in closed form
    via a shifted soft-threshold,

    .. math::

       x^\ast = \begin{cases}
       v + w \tau / \mu          & y > v + w \tau / \mu,
       \quad \text{(overshoot)} \\
       v - w (1 - \tau) / \mu    & y < v - w (1 - \tau) / \mu,
       \quad \text{(undershoot)} \\
       y                          & \text{otherwise.}
       \end{cases}
    """
    if w is None:
        upper = v + tau / mu
        lower = v - (1.0 - tau) / mu
    else:
        upper = v + (w * tau) / mu
        lower = v - (w * (1.0 - tau)) / mu
    return np.where(y > upper, upper, np.where(y < lower, lower, y))
