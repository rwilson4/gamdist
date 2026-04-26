"""Shared fixtures for the gamdist test suite."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import matplotlib
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

matplotlib.use("Agg")

FloatArray = npt.NDArray[np.float64]


def _identity_link(x: FloatArray) -> FloatArray:
    return x


def _logit_inv_link(x: FloatArray) -> FloatArray:
    return np.exp(x) / (1.0 + np.exp(x))


def _gaussian_family(rng: np.random.Generator, mu: FloatArray) -> FloatArray:
    return mu + rng.normal(size=mu.shape, loc=0.0, scale=np.sqrt(0.1))


def _binomial_family(
    rng: np.random.Generator, mu: FloatArray, ccs: int | FloatArray = 1
) -> FloatArray:
    return rng.binomial(ccs, p=mu).astype(float)


def gmu_purchases(x: FloatArray) -> FloatArray:
    return 0.1 * np.log1p(x) + 0.3


def gmu_gender(x: npt.NDArray[Any]) -> FloatArray:
    z = np.zeros(x.shape)
    z[x == "male"] = 0.1
    z[x == "female"] = -0.5
    return z


def gmu_country(x: npt.NDArray[Any]) -> FloatArray:
    z = np.zeros(x.shape)
    z[x == "USA"] = -0.2
    z[x == "CAN"] = 0.3
    z[x == "GBR"] = 0.4
    return z


def gmu_hft(x: FloatArray) -> FloatArray:
    # Equation 5.22 of Hastie, Friedman, Tibshirani, "Elements of Statistical Learning"
    return np.sin(12.0 * (x + 0.2)) / (x + 0.2)


def generate_data(
    num_obs: int,
    seed: int = 3,
    link: Callable[[FloatArray], FloatArray] = _identity_link,
    family: Callable[[np.random.Generator, FloatArray], FloatArray] | None = None,
    return_mean: bool = False,
    include_hft: bool = False,
) -> tuple[pd.DataFrame, FloatArray]:
    rng = np.random.default_rng(seed)
    purchases = np.array([0, 3, 10, 16, 27, 30])
    p_purchases = np.array([0.1, 0.2, 0.3, 0.3, 0.05, 0.05])
    genders = np.array(["male", "female"])
    p_genders = np.array([0.7, 0.3])
    countries = np.array(["USA", "CAN", "GBR"])

    X = pd.DataFrame(
        {
            "purchases": rng.choice(purchases, size=num_obs, p=p_purchases),
            "gender": rng.choice(genders, size=num_obs, p=p_genders),
            "country": rng.choice(countries, size=num_obs),
            "hft": rng.random(size=num_obs),
        }
    )
    gmu = gmu_purchases(X["purchases"].values.astype(float))
    gmu = gmu + gmu_gender(X["gender"].values)
    gmu = gmu + gmu_country(X["country"].values)
    if include_hft:
        gmu = gmu + gmu_hft(X["hft"].values.astype(float))

    mu = link(gmu)
    if return_mean:
        return X, mu
    fam = family if family is not None else _gaussian_family
    return X, fam(rng, mu)


def generate_covariate_class_data(
    seed: int = 4, return_mean: bool = False
) -> tuple[pd.DataFrame, FloatArray, FloatArray]:
    rng = np.random.default_rng(seed)
    genders = [
        "male",
        "female",
        "male",
        "female",
        "male",
        "female",
        "male",
        "female",
        "male",
        "female",
        "male",
        "female",
        "male",
        "female",
        "male",
        "female",
        "male",
    ]
    countries = [
        "usa",
        "usa",
        "gbr",
        "gbr",
        "can",
        "can",
        "usa",
        "usa",
        "gbr",
        "gbr",
        "can",
        "can",
        "usa",
        "usa",
        "gbr",
        "gbr",
        "can",
    ]
    X = pd.DataFrame({"gender": genders, "country": countries})
    ccs = np.array(
        [
            1000,
            1400,
            2200,
            1300,
            3200,
            1700,
            500,
            1700,
            1400,
            800,
            2600,
            1200,
            1600,
            900,
            400,
            1600,
            1200,
        ],
        dtype=float,
    )
    gmu = gmu_gender(X["gender"].values) + np.where(
        X["country"].values == "usa",
        -0.2,
        np.where(
            X["country"].values == "can",
            0.3,
            np.where(X["country"].values == "gbr", 0.4, 0.0),
        ),
    )
    mu = _logit_inv_link(gmu)
    if return_mean:
        return X, mu, ccs
    return X, _binomial_family(rng, mu, ccs.astype(int)), ccs


def generate_spline_data(
    num_obs: int, seed: int = 5
) -> tuple[pd.DataFrame, FloatArray]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"hft": rng.random(size=num_obs)})
    gmu = gmu_hft(X["hft"].values.astype(float))
    return X, _gaussian_family(rng, gmu)


@pytest.fixture
def linear_data() -> tuple[pd.DataFrame, FloatArray]:
    return generate_data(400, seed=3)


@pytest.fixture
def linear_test_data() -> tuple[pd.DataFrame, FloatArray]:
    return generate_data(100, seed=33)


@pytest.fixture
def logistic_data() -> tuple[pd.DataFrame, FloatArray]:
    return generate_data(
        400, seed=7, link=_logit_inv_link, family=_binomial_family_wrapper
    )


@pytest.fixture
def covariate_data() -> tuple[pd.DataFrame, FloatArray, FloatArray]:
    return generate_covariate_class_data(seed=4)


@pytest.fixture
def spline_data() -> tuple[pd.DataFrame, FloatArray]:
    return generate_spline_data(400, seed=5)


@pytest.fixture
def additive_data() -> tuple[pd.DataFrame, FloatArray]:
    return generate_data(500, seed=11, include_hft=True)


def _binomial_family_wrapper(rng: np.random.Generator, mu: FloatArray) -> FloatArray:
    return _binomial_family(rng, mu, 1)
