"""End-to-end tests for the huber (M-estimator) family."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from gamdist import GAM


def _huber_loss(r: np.ndarray, delta: float) -> np.ndarray:
    abs_r = np.abs(r)
    return np.where(abs_r <= delta, 0.5 * r * r, delta * (abs_r - 0.5 * delta))


def _contaminated_data(
    n: int, seed: int, outlier_frac: float = 0.1
) -> tuple[pd.DataFrame, np.ndarray]:
    """y = 1 + 2x + eps with a fraction of large-magnitude outliers in eps."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=n)
    eps = rng.normal(size=n) * 0.2
    n_out = int(round(outlier_frac * n))
    if n_out > 0:
        idx = rng.choice(n, size=n_out, replace=False)
        eps[idx] = rng.choice([-1.0, 1.0], size=n_out) * rng.uniform(
            5.0, 10.0, size=n_out
        )
    y = 1.0 + 2.0 * x + eps
    return pd.DataFrame({"x": x}), y


def test_huber_recovers_slope_under_outliers() -> None:
    # Huber loss should track the underlying slope despite contamination
    # that knocks the OLS estimate around. Both fit on the same training
    # set; the Huber slope ends up closer to the truth (2.0).
    X_train, y_train = _contaminated_data(n=600, seed=0, outlier_frac=0.15)
    truth = 2.0

    huber = GAM(family="huber", delta=1.0)
    huber.add_feature(name="x", type="linear")
    huber.fit(X_train, y_train, max_its=200)

    ols = GAM(family="normal")
    ols.add_feature(name="x", type="linear")
    ols.fit(X_train, y_train, max_its=200)

    # The Huber slope should be at least as close to truth as OLS, and
    # should be within a small tolerance of truth on this data size.
    huber_slope = float(huber._features["x"]._m)  # type: ignore[attr-defined]
    ols_slope = float(ols._features["x"]._m)  # type: ignore[attr-defined]
    assert abs(huber_slope - truth) < abs(ols_slope - truth)
    assert abs(huber_slope - truth) < 0.2


def test_huber_large_delta_approaches_normal() -> None:
    # With delta much larger than any residual, Huber loss collapses to
    # 0.5 * r^2 elementwise, so the fit should match normal-family OLS
    # on clean Gaussian data.
    rng = np.random.default_rng(3)
    n = 400
    x = rng.normal(size=n)
    y = 0.5 + 1.5 * x + rng.normal(size=n) * 0.1
    X = pd.DataFrame({"x": x})

    huber = GAM(family="huber", delta=1e6)
    huber.add_feature(name="x", type="linear")
    huber.fit(X, y, max_its=200)

    ols = GAM(family="normal")
    ols.add_feature(name="x", type="linear")
    ols.fit(X, y, max_its=200)

    np.testing.assert_allclose(huber.predict(X), ols.predict(X), atol=1e-2)


def test_huber_requires_delta() -> None:
    with pytest.raises(ValueError, match="delta"):
        GAM(family="huber")


@pytest.mark.parametrize("bad_delta", [0.0, -0.5, -1e-12])
def test_huber_rejects_non_positive_delta(bad_delta: float) -> None:
    with pytest.raises(ValueError, match="delta"):
        GAM(family="huber", delta=bad_delta)


def test_huber_rejects_non_identity_link() -> None:
    with pytest.raises(ValueError, match="identity"):
        GAM(family="huber", link="log", delta=1.0)


def test_huber_canonical_link_is_identity() -> None:
    mdl = GAM(family="huber", delta=1.0)
    assert mdl._link == "identity"


def test_huber_dispersion_is_one() -> None:
    rng = np.random.default_rng(0)
    n = 100
    X = pd.DataFrame({"x": rng.normal(size=n)})
    y = X["x"].values + rng.normal(size=n) * 0.5

    mdl = GAM(family="huber", delta=1.0)
    mdl.add_feature(name="x", type="linear")
    mdl.fit(X, y, max_its=50)
    assert mdl.dispersion() == 1.0


def test_huber_deviance_matches_huber_loss_sum() -> None:
    rng = np.random.default_rng(7)
    n = 200
    X = pd.DataFrame({"x": rng.normal(size=n)})
    y = X["x"].values + rng.normal(size=n) * 0.5

    delta = 0.7
    mdl = GAM(family="huber", delta=delta)
    mdl.add_feature(name="x", type="linear")
    mdl.fit(X, y, max_its=100)

    pred = mdl.predict(X)
    expected = 2.0 * float(np.sum(_huber_loss(y - pred, delta)))
    assert mdl.deviance() == pytest.approx(expected, rel=1e-6, abs=1e-9)


def test_huber_save_load_roundtrip() -> None:
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame({"x": rng.normal(size=n)})
    y = X["x"].values + rng.normal(size=n)

    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            mdl = GAM(family="huber", delta=1.5, name="huber_roundtrip")
            mdl.add_feature(name="x", type="linear")
            mdl.fit(X, y, max_its=50, save_flag=True)
            yhat_before = mdl.predict(X)

            mdl2 = GAM(load_from_file="huber_roundtrip_model.pckl")
        finally:
            os.chdir(cwd)

        assert mdl2._family == "huber"
        assert mdl2._delta == pytest.approx(1.5)
        np.testing.assert_allclose(mdl2.predict(X), yhat_before, rtol=1e-12)
