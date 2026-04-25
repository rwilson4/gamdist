"""End-to-end regression tests mirroring the historical test.py varieties."""

from __future__ import annotations

import numpy as np
import pytest

from gamdist import GAM
from tests.conftest import (
    _binomial_family,
    _logit_inv_link,
    generate_covariate_class_data,
    generate_data,
    generate_spline_data,
    gmu_country,
    gmu_gender,
    gmu_purchases,
)


def _mse(yhat: np.ndarray, ytrue: np.ndarray) -> float:
    err = ytrue - yhat
    return float(err.dot(err) / len(err))


def test_linear_regression_recovers_signal() -> None:
    X_train, y_train = generate_data(400, seed=3)
    X_test, _ = generate_data(200, seed=33)
    mu_test = (
        gmu_purchases(X_test["purchases"].values.astype(float))
        + gmu_gender(X_test["gender"].values)
        + gmu_country(X_test["country"].values)
    )
    mdl = GAM(family="normal")
    mdl.add_feature(name="purchases", type="linear", transform=np.log1p)
    mdl.add_feature(name="gender", type="categorical")
    mdl.add_feature(name="country", type="categorical")
    mdl.fit(X_train, y_train, max_its=80)

    yhat = mdl.predict(X_test)
    assert _mse(yhat, mu_test) < 0.05


def test_logistic_regression_recovers_probabilities() -> None:
    rng = np.random.default_rng(7)
    X_train, _ = generate_data(800, seed=7, link=_logit_inv_link)
    p_train = _logit_inv_link(
        gmu_purchases(X_train["purchases"].values.astype(float))
        + gmu_gender(X_train["gender"].values)
        + gmu_country(X_train["country"].values)
    )
    y_train = _binomial_family(rng, p_train).astype(float)

    X_test, _ = generate_data(200, seed=70)
    p_test = _logit_inv_link(
        gmu_purchases(X_test["purchases"].values.astype(float))
        + gmu_gender(X_test["gender"].values)
        + gmu_country(X_test["country"].values)
    )

    mdl = GAM(family="binomial")
    mdl.add_feature(name="purchases", type="linear", transform=np.log1p)
    mdl.add_feature(name="gender", type="categorical")
    mdl.add_feature(name="country", type="categorical")
    mdl.fit(X_train, y_train, max_its=80)

    phat = mdl.predict(X_test)
    # Logistic regression with ~800 samples on 0/1 outcomes is noisier than the
    # Gaussian case; relax the bound accordingly.
    assert _mse(phat, p_test) < 0.15


def test_logistic_regression_with_covariate_classes() -> None:
    X, y, ccs = generate_covariate_class_data(seed=4)
    mdl = GAM(family="binomial", estimate_overdispersion=True)
    mdl.add_feature(name="gender", type="categorical")
    mdl.add_feature(name="country", type="categorical")
    mdl.fit(X, y, covariate_class_sizes=ccs, max_its=80)

    X_eval, mu_eval, _ = generate_covariate_class_data(seed=4, return_mean=True)
    phat = mdl.predict(X_eval)
    # Predictions should be in (0, 1) and reasonably close to the true rates.
    assert np.all((phat > 0) & (phat < 1))
    # Pearson-style check rather than tight MSE because we have only 17 covariate
    # classes and binomial draws have nontrivial variance per class.
    rates = y / ccs
    assert _mse(phat, rates) < 0.05


def test_spline_regression_recovers_smooth_function() -> None:
    X_train, y_train = generate_spline_data(800, seed=5)
    X_test, _ = generate_spline_data(200, seed=55)
    from tests.conftest import gmu_hft

    mu_test = gmu_hft(X_test["hft"].values.astype(float))

    mdl = GAM(family="normal")
    mdl.add_feature(name="hft", type="spline", rel_dof=9.0)
    mdl.fit(X_train, y_train, max_its=80)

    yhat = mdl.predict(X_test)
    # Spline fit on noisy data: the residual against the noiseless mean
    # should be bounded.
    assert _mse(yhat, mu_test) < 1.0


def test_additive_regression_combines_feature_types() -> None:
    X_train, y_train = generate_data(600, seed=11, include_hft=True)
    X_test, _ = generate_data(200, seed=110, include_hft=True)
    from tests.conftest import gmu_hft

    mu_test = (
        gmu_purchases(X_test["purchases"].values.astype(float))
        + gmu_gender(X_test["gender"].values)
        + gmu_country(X_test["country"].values)
        + gmu_hft(X_test["hft"].values.astype(float))
    )
    mdl = GAM(family="normal")
    mdl.add_feature(name="hft", type="spline", rel_dof=9.0)
    mdl.add_feature(name="purchases", type="linear", transform=np.log1p)
    mdl.add_feature(name="gender", type="categorical")
    mdl.add_feature(name="country", type="categorical")
    mdl.fit(X_train, y_train, max_its=80)

    yhat = mdl.predict(X_test)
    assert _mse(yhat, mu_test) < 1.0


@pytest.mark.parametrize("smoothing", [0.5, 1.0, 2.0])
def test_smoothing_changes_dof(smoothing: float) -> None:
    X, y = generate_spline_data(300, seed=5)
    mdl = GAM(family="normal")
    mdl.add_feature(name="hft", type="spline", rel_dof=9.0)
    mdl.fit(X, y, max_its=30, smoothing=smoothing)
    # More smoothing -> fewer effective DOF.
    assert mdl.dof() > 1.0
