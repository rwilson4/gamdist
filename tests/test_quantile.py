"""End-to-end tests for the quantile (pinball-loss) family."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from gamdist import GAM


def _heteroskedastic_data(
    n: int, seed: int
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """y_i = x_i + (1 + x_i) * eps_i with eps_i ~ N(0, 1).

    The conditional tau-quantile is then linear in x, namely
    ``x + (1 + x) * Phi^{-1}(tau)``, so a single linear feature suffices
    to represent it exactly.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=n)
    eps = rng.normal(size=n)
    y = x + (1.0 + x) * eps
    X = pd.DataFrame({"x": x})
    return X, y, x


def _true_quantile(x: np.ndarray, tau: float) -> np.ndarray:
    return x + (1.0 + x) * float(norm.ppf(tau))


@pytest.mark.parametrize("tau", [0.1, 0.5, 0.9])
def test_quantile_recovers_conditional_quantile(tau: float) -> None:
    X_train, y_train, _ = _heteroskedastic_data(n=4000, seed=11 + int(tau * 100))
    X_test, _, x_test = _heteroskedastic_data(n=400, seed=999)

    mdl = GAM(family="quantile", tau=tau)
    mdl.add_feature(name="x", type="linear")
    mdl.fit(X_train, y_train, max_its=200)

    yhat = mdl.predict(X_test)
    truth = _true_quantile(x_test, tau)

    err = float(np.sqrt(np.mean((yhat - truth) ** 2)))
    assert err < 0.15, f"tau={tau}: rmse vs. true quantile = {err}"


@pytest.mark.parametrize("tau", [0.2, 0.8])
def test_quantile_empirical_coverage_matches_tau(tau: float) -> None:
    # On a held-out sample, the fraction of observations falling below
    # the predicted tau-quantile should be ~tau. That is the operational
    # contract of a quantile regression and a tau-agnostic check.
    X_train, y_train, _ = _heteroskedastic_data(n=4000, seed=31 + int(tau * 100))
    X_test, y_test, _ = _heteroskedastic_data(n=2000, seed=131 + int(tau * 100))

    mdl = GAM(family="quantile", tau=tau)
    mdl.add_feature(name="x", type="linear")
    mdl.fit(X_train, y_train, max_its=200)

    yhat = mdl.predict(X_test)
    coverage = float(np.mean(y_test <= yhat))
    assert abs(coverage - tau) < 0.05, f"tau={tau}: coverage={coverage}"


def test_quantile_requires_tau() -> None:
    with pytest.raises(ValueError, match="tau"):
        GAM(family="quantile")


@pytest.mark.parametrize("bad_tau", [0.0, 1.0, -0.1, 1.5])
def test_quantile_rejects_tau_out_of_range(bad_tau: float) -> None:
    with pytest.raises(ValueError, match="tau"):
        GAM(family="quantile", tau=bad_tau)


def test_quantile_rejects_non_identity_link() -> None:
    with pytest.raises(ValueError, match="identity"):
        GAM(family="quantile", link="log", tau=0.5)


def test_quantile_deviance_matches_pinball_loss() -> None:
    rng = np.random.default_rng(7)
    n = 200
    X = pd.DataFrame({"x": rng.normal(size=n)})
    y = X["x"].values + rng.normal(size=n) * 0.5

    mdl = GAM(family="quantile", tau=0.3)
    mdl.add_feature(name="x", type="linear")
    mdl.fit(X, y, max_its=100)

    pred = mdl.predict(X)
    r = y - pred
    expected = 2.0 * float(np.sum(np.maximum(0.3 * r, (0.3 - 1.0) * r)))
    assert mdl.deviance() == pytest.approx(expected, rel=1e-6, abs=1e-9)


def test_quantile_save_load_roundtrip() -> None:
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame({"x": rng.normal(size=n)})
    y = X["x"].values + rng.normal(size=n)

    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            mdl = GAM(family="quantile", tau=0.7, name="quantile_roundtrip")
            mdl.add_feature(name="x", type="linear")
            mdl.fit(X, y, max_its=50, save_flag=True)
            yhat_before = mdl.predict(X)

            mdl2 = GAM(load_from_file="quantile_roundtrip_model.pckl")
        finally:
            os.chdir(cwd)

        assert mdl2._family == "quantile"
        assert mdl2._tau == pytest.approx(0.7)
        np.testing.assert_allclose(mdl2.predict(X), yhat_before, rtol=1e-12)
