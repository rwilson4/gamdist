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

import numpy as np
from scipy.optimize import minimize_scalar

def _prox_normal_identity(v, mu, y, w=None, inv_link=None, p=None):
    # Assume there are m elements in v and y.
    # mu is a scalar
    # w is either None, or has m elements.
    if w is None:
        # If no weights, this takes:
        #  - m multiplies
        #  - m+1 adds
        #  - m divides
        return (y + mu * v) / (1. + mu)
    else:
        # If weights, this takes:
        #  - 2m multiplies
        #  - 2m adds
        #  - m divides
        return (w * y + mu * v) / (w + mu)

def _prox_normal(v, mu, y, w=None, inv_link=None, p=None):
    if w is None:
        def obj_fun(x, _v, _y):
            ilx = inv_link(x)
            return 0.5 * ilx * ilx - _y * ilx + 0.5 * mu * (x - _v) * (x - _v)

        return np.array([minimize_scalar(obj_fun, args=(v[i], y[i])).x for i in range(len(v))])

    else:
        def obj_fun(x, _v, _y, _w):
            ilx = inv_link(x)
            return _w * (0.5 * ilx * ilx - _y * ilx) + 0.5 * mu * (x - _v) * (x - _v)

        return np.array([minimize_scalar(obj_fun, args=(v[i], y[i], w[i])).x for i in range(len(v))])

def _prox_binomial_logit_scalar(xx):
    v  = xx[0]
    mu = xx[1]
    y  = xx[2]
    m = xx[3]
    if len(xx) >= 5:
        w = xx[4]
    else:
        w = None

    beta = 0.8
    tol = 1e-3
    max_its = 100

    x = 0.0
    if w is None:
        mu_v_plus_w_y = mu * v + y
    else:
        mu_v_plus_w_y = mu * v + w * y

    for i in range(max_its):
        expx = np.exp(x)
        one_over_one_plus_expx = 1. / (1. + expx)
        if w is None:
            num = mu * x + m * expx * one_over_one_plus_expx - mu_v_plus_w_y
            denom = mu + m * expx  * one_over_one_plus_expx * one_over_one_plus_expx
        else:
            num = mu * x + w * m * expx * one_over_one_plus_expx - mu_v_plus_w_y
            denom = mu + w * m * expx * one_over_one_plus_expx * one_over_one_plus_expx

        dx = num / denom
        x -= dx # * beta
        if abs(dx) < tol:
            return x
    else:
        raise ValueError('Dual variable update failed to converge.')


def _prox_binomial_logit(v, mu, y, ccs=None, w=None, inv_link=None, p=None):
    if ccs is None:
        m = np.ones(y.shape)
    else:
        m = ccs

    if w is None:
        if p is None:
            return np.array(map(_prox_binomial_logit_scalar, zip(v, np.full(v.shape, mu), y, m)))
        else:
            return np.array(p.map(_prox_binomial_logit_scalar, zip(v, np.full(v.shape, mu), y, m)))
    else:
        if p is None:
            return np.array(map(_prox_binomial_logit_scalar, zip(v, np.full(v.shape, mu), y, m, w)))
        else:
            return np.array(p.map(_prox_binomial_logit_scalar, zip(v, np.full(v.shape, mu), y, m, w)))


def _prox_binomial(v, mu, y, ccs=None, w=None, inv_link=None, p=None):
    if w is None:
        def obj_fun(x, _v, _y):
            ilx = inv_link(x)
            m = 1.
            return (_y - m) * np.log1p(-ilx) - _y * np.log(ilx) + 0.5 * mu * (x - _v) * (x - _v)

        return np.array([minimize_scalar(obj_fun, args=(v[i], y[i])).x for i in range(len(v))])
    else:
        def obj_fun(x, _v, _y, _w):
            ilx = inv_link(x)
            m = 1.
            return _w * (_y - m) * np.log1p(-ilx) - _w * _y * np.log(ilx) + 0.5 * mu * (x - _v) * (x - _v)

        return np.array([minimize_scalar(obj_fun, args=(v[i], y[i], w[i])).x for i in range(len(v))])

def _prox_poisson_log_scalar(xx):
    v  = xx[0]
    mu = xx[1]
    y  = xx[2]
    if len(xx) >= 4:
        w = xx[3]
    else:
        w = None

    beta = 0.8
    tol = 1e-3
    max_its = 100

    x = 0.0
    if w is None:
        mu_v_plus_w_y = mu * v + y
    else:
        mu_v_plus_w_y = mu * v + w * y

    for i in range(max_its):
        expx = np.exp(x)
        if w is None:
            num = mu * x + expx - mu_v_plus_w_y
            denom = mu + expx
        else:
            num = mu * x + w * expx - mu_v_plus_w_y
            denom = mu + w * expx

        dx = num / denom
        x -= dx # * beta
        if abs(dx) < tol:
            return x
    else:
        raise ValueError('Dual variable update failed to converge.')


def _prox_poisson_log(v, mu, y, w=None, inv_link=None, p=None):
    if w is None:
        if p is None:
            return np.array(map(_prox_poisson_log_scalar, zip(v, np.full(v.shape, mu), y)))
        else:
            return np.array(p.map(_prox_poisson_log_scalar, zip(v, np.full(v.shape, mu), y)))
    else:
        if p is None:
            return np.array(map(_prox_poisson_log_scalar, zip(v, np.full(v.shape, mu), y, w)))
        else:
            return np.array(p.map(_prox_poisson_log_scalar, zip(v, np.full(v.shape, mu), y, w)))

def _prox_poisson(v, mu, y, w=None, inv_link=None, p=None):
    if w is None:
        def obj_fun(x, _v, _y):
            ilx = inv_link(x)
            return (ilx - _y * np.log(ilx)) + 0.5 * mu * (x - _v) * (x - _v)

        return np.array([minimize_scalar(obj_fun, args=(v[i], y[i])).x for i in range(len(v))])
    else:
        def obj_fun(x, _v, _y, _w):
            ilx = inv_link(x)
            return _w * (ilx - _y * np.log(ilx)) + 0.5 * mu * (x - _v) * (x - _v)

        return np.array([minimize_scalar(obj_fun, args=(v[i], y[i], w[i])).x for i in range(len(v))])


def _prox_gamma_reciprocal(v, mu, y, w=None, inv_link=None, p=None):
    if w is None:
        mu_v_minus_w_y = mu * v - y
        return (0.5 / mu) * mu_v_minus_w_y + np.sqrt( mu_v_minus_w_y * mu_v_minus_w_y + (4 * mu))
    else:
        mu_v_minus_w_y = mu * v - w * y
        return (0.5 / mu) * mu_v_minus_w_y + np.sqrt( mu_v_minus_w_y * mu_v_minus_w_y + (4 * mu) * w)

def _prox_gamma(v, mu, y, w=None, inv_link=None, p=None):
    if w is None:
        def obj_fun(x, _v, _y):
            ilx = inv_link(x)
            return (np.log(ilx) + _y / ilx) + 0.5 * mu * (x - _v) * (x - _v)

        return np.array([minimize_scalar(obj_fun, args=(v[i], y[i])).x for i in range(len(v))])
    else:
        def obj_fun(x, _v, _y, _w):
            ilx = inv_link(x)
            return _w * (np.log(ilx) + _y / ilx) + 0.5 * mu * (x - _v) * (x - _v)

        return np.array([minimize_scalar(obj_fun, args=(v[i], y[i], w[i])).x for i in range(len(v))])

def _prox_inv_gaussian_reciprocal_squared_scalar(xx):
    v = xx[0]
    mu = xx[1]
    y = xx[2]
    if len(xx) >= 4:
        w = xx[3]
    else:
        w = None

    beta = 0.8
    tol = 1e-3
    max_its = 100

    z = 1.0
    if w is None:
        w_y_minus_mu_v = 0.5 * y - mu * v
    else:
        w_y_minus_mu_v = 0.5 * w * y - mu * v

    for i in range(max_its):
        if w is None:
            num = mu * z * z * z + w_y_minus_mu_v * z - 0.5
            denom = 3 * mu * z * z + w_y_minus_mu_v
        else:
            num = mu * z * z * z + w_y_minus_mu_v * z - 0.5 * w
            denom = 3 * mu * z * z + w_y_minus_mu_v

        dz = num / denom
        z -= dz # * beta
        if abs(dz) < tol:
            return z * z
    else:
        raise ValueError('Dual variable update failed to converge.')

def _prox_inv_gaussian_reciprocal_squared(v, mu, y, w=None, inv_link=None, p=None):
    if w is None:
        if p is None:
            return np.array(map(_prox_inv_gaussian_reciprocal_squared_scalar, zip(v, np.full(v.shape, mu), y)))
        else:
            return np.array(p.map(_prox_inv_gaussian_reciprocal_squared_scalar, zip(v, np.full(v.shape, mu), y)))
    else:
        if p is None:
            return np.array(map(_prox_inv_gaussian_reciprocal_squared_scalar, zip(v, np.full(v.shape, mu), y, w)))
        else:
            return np.array(p.map(_prox_inv_gaussian_reciprocal_squared_scalar, zip(v, np.full(v.shape, mu), y, w)))

def _prox_inv_gaussian(v, mu, y, w=None, inv_link=None, p=None):
    if w is None:
        def obj_fun(x, _v, _y):
            ilx = inv_link(x)
            return (-1. / ilx + 0.5 * _y * ilx * ilx) + 0.5 * mu * (x - _v) * (x - _v)

        return np.array([minimize_scalar(obj_fun, args=(v[i], y[i])).x for i in range(len(v))])
    else:
        def obj_fun(x, _v, _y, _w):
            ilx = inv_link(x)
            return _w * (-1. / ilx + 0.5 * _y * ilx * ilx) + 0.5 * mu * (x - _v) * (x - _v)

        return np.array([minimize_scalar(obj_fun, args=(v[i], y[i], w[i])).x for i in range(len(v))])
