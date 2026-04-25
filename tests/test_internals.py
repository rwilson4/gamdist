"""Tests for internal helpers and less-traveled GAM branches."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.special as special

from gamdist import GAM
from gamdist.gamdist import _gamma_dispersion, _plot_convergence
from tests.conftest import generate_covariate_class_data


def test_gamma_dispersion_solves_likelihood_equation() -> None:
    # Inputs that the previous damped-Newton implementation could not
    # solve in 100 iterations. The brentq-based solver returns the unique
    # root of f(nu) = 2*n*(log nu - psi(nu)) - dof/nu - dev.
    nu = _gamma_dispersion(dof=3.0, dev=10.0, num_obs=50)
    assert nu > 0
    f_at_nu = (
        2.0 * 50 * (np.log(nu) - special.psi(nu)) - 3.0 / nu - 10.0
    )
    assert abs(f_at_nu) < 1e-6


def test_gamma_dispersion_zero_deviance_raises() -> None:
    # With dev=0 the likelihood equation has no positive root (the LHS is
    # strictly positive for all finite nu).
    with pytest.raises(ValueError, match="Could not estimate gamma dispersion"):
        _gamma_dispersion(dof=3.0, dev=0.0, num_obs=50)


def test_gamma_dispersion_via_gam_succeeds() -> None:
    # End-to-end: fitting a gamma model and calling .dispersion() must
    # now succeed. The previous implementation raised ValueError on this
    # exact data.
    rng = np.random.default_rng(0)
    n = 100
    X = pd.DataFrame({"x": rng.uniform(0.1, 0.5, size=n)})
    y = rng.gamma(shape=4.0, scale=0.5, size=n)
    mdl = GAM(family="gamma")
    mdl.add_feature(name="x", type="linear")
    mdl.fit(X, y, max_its=30)
    phi = mdl.dispersion()
    assert np.isfinite(phi)
    assert phi > 0


def test_plot_convergence_uses_agg_backend() -> None:
    # Produces a figure but doesn't call show() under the Agg backend.
    _plot_convergence(
        prim_res=[1.0, 0.5, 0.1],
        prim_tol=[1.0, 1.0, 1.0],
        dual_res=[2.0, 1.0, 0.2],
        dual_tol=[1.0, 1.0, 1.0],
        dev=[10.0, 5.0, 4.5],
    )


def test_binomial_overdispersion_replication() -> None:
    # The default covariate-class data has replicated gender x country cells,
    # so dispersion() routes through the replication branch when it exists.
    X, y, ccs = generate_covariate_class_data(seed=4)
    mdl = GAM(family="binomial", estimate_overdispersion=True)
    mdl.add_feature(name="gender", type="categorical")
    mdl.add_feature(name="country", type="categorical")
    mdl.fit(X, y, covariate_class_sizes=ccs, max_its=40)
    disp = mdl.dispersion()
    assert np.isfinite(disp)
    assert disp > 0


def test_binomial_overdispersion_pearson_branch_directly() -> None:
    # Build a fitted model with covariate classes, then force the
    # Pearson-residual branch by calling the helper with a fixed formula.
    X, y, ccs = generate_covariate_class_data(seed=4)
    mdl = GAM(family="binomial", estimate_overdispersion=True)
    mdl.add_feature(name="gender", type="categorical")
    mdl.add_feature(name="country", type="categorical")
    mdl.fit(X, y, covariate_class_sizes=ccs, max_its=40)
    disp = mdl._binomial_overdispersion(formula="pearson")
    assert np.isfinite(disp)
    assert disp > 0


def test_binomial_overdispersion_no_covariate_classes_returns_one() -> None:
    rng = np.random.default_rng(0)
    n = 80
    X = pd.DataFrame({"x": rng.normal(size=n)})
    y = rng.binomial(1, p=0.5, size=n).astype(float)
    mdl = GAM(family="binomial", estimate_overdispersion=True)
    mdl.add_feature(name="x", type="linear")
    mdl.fit(X, y, max_its=20)
    assert mdl._binomial_overdispersion() == 1.0


def test_poisson_dispersion_one_without_estimate_flag() -> None:
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame({"x": rng.normal(size=n)})
    y = rng.poisson(np.exp(0.4 * X["x"].values + 0.5), size=n).astype(float)
    mdl = GAM(family="poisson")  # estimate_overdispersion=False (default)
    mdl.add_feature(name="x", type="linear")
    mdl.fit(X, y, max_its=30)
    assert mdl.dispersion() == 1.0


def test_poisson_overdispersion_pearson_formula() -> None:
    # Numeric check: dispersion() should equal sum((y - mu)^2 / mu) / (n - p)
    # when estimate_overdispersion=True and the data is overdispersed.
    rng = np.random.default_rng(3)
    n = 300
    X = pd.DataFrame({"x": rng.normal(size=n)})
    # Negative-binomial sampling produces overdispersion vs. pure Poisson:
    # mean = mu, var = mu + mu^2 / k. Choose k=2 for moderate overdispersion.
    mu_true = np.exp(0.3 * X["x"].values + 1.0)
    k = 2.0
    p_nb = k / (k + mu_true)
    y = rng.negative_binomial(k, p_nb, size=n).astype(float)

    mdl = GAM(family="poisson", estimate_overdispersion=True)
    mdl.add_feature(name="x", type="linear")
    mdl.fit(X, y, max_its=40)

    disp = mdl.dispersion()
    mu_hat = mdl._eval_inv_link(mdl._num_features * mdl.f_bar)
    expected = float(np.sum((y - mu_hat) ** 2 / mu_hat) / (n - mdl.dof()))
    assert disp == pytest.approx(expected)
    assert disp > 1.0  # We sampled from an overdispersed distribution.


def test_poisson_overdispersion_caches_dispersion() -> None:
    # Calling dispersion() twice must not change the result.
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame({"x": rng.normal(size=n)})
    y = rng.poisson(np.exp(0.5 * X["x"].values), size=n).astype(float)
    mdl = GAM(family="poisson", estimate_overdispersion=True)
    mdl.add_feature(name="x", type="linear")
    mdl.fit(X, y, max_its=30)
    first = mdl.dispersion()
    second = mdl.dispersion()
    assert first == second
    assert mdl._known_dispersion


@pytest.mark.parametrize(
    "link,family",
    [
        ("identity", "normal"),
        ("logistic", "binomial"),
        ("probit", "binomial"),
        ("complementary_log_log", "binomial"),
        ("log", "poisson"),
        # gamma + reciprocal does not constrain mu > 0 and is numerically
        # unstable on small synthetic data; covered by the gamma fit test.
        ("reciprocal_squared", "inverse_gaussian"),
    ],
)
def test_link_function_round_trip(link: str, family: str, tmp_path: Path) -> None:
    """Each link type can be saved and re-loaded with its eval_link/eval_inv_link."""
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        rng = np.random.default_rng(0)
        n = 60
        X = pd.DataFrame({"x": rng.uniform(0.4, 0.7, size=n)})
        # Use simple, well-bounded synthetic responses.
        if family == "binomial":
            y = rng.binomial(1, p=0.5, size=n).astype(float)
        elif family == "poisson":
            y = rng.poisson(2.0, size=n).astype(float)
        elif family in {"gamma", "inverse_gaussian"}:
            y = rng.uniform(0.5, 1.5, size=n)
        else:
            y = X["x"].values + rng.normal(size=n) * 0.1

        mdl = GAM(family=family, link=link, name=f"link_{link}")  # type: ignore[arg-type]
        mdl.add_feature(name="x", type="linear")
        mdl.fit(X, y, max_its=10, save_flag=True)
        before = mdl.predict(X)

        restored = GAM(load_from_file=f"link_{link}_model.pckl")
        after = restored.predict(X)
        np.testing.assert_allclose(after, before, rtol=1e-10, atol=1e-10)
    finally:
        os.chdir(cwd)


def test_summary_with_covariate_classes(capsys: pytest.CaptureFixture[str]) -> None:
    X, y, ccs = generate_covariate_class_data(seed=4)
    mdl = GAM(family="binomial", estimate_overdispersion=True)
    mdl.add_feature(name="gender", type="categorical")
    mdl.add_feature(name="country", type="categorical")
    mdl.fit(X, y, covariate_class_sizes=ccs, max_its=20)
    mdl.summary()
    out = capsys.readouterr().out
    assert "Model Statistics" in out
    assert "Features" in out
