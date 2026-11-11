"""Coefficient tables and Wald / score / likelihood-ratio tests (#47).

As in ``test_robust_covariance.py``, the sharp checks inject
statsmodels' converged linear predictor into gamdist (via ``f_bar``)
so the formulas are compared at an identical ``mu``; ADMM's
``eps_abs = 1e-3`` stopping rule otherwise mixes optimizer error into
the comparison. Even then statsmodels evaluates its Hessian at the
last IRLS iterate, so agreement is ~1e-6 relative on standard errors
(more on tiny p-values), not machine precision. Looser end-to-end
checks cover the real fit path.

statsmodels' GLM ignores the HC1 small-sample factor (its HC1 equals
its HC0), so robust comparisons use HC0; HC1 = HC0 * n / (n - p) is
pinned in ``test_robust_covariance.py``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.stats as scipy_stats

from gamdist import GAM, HypothesisTest

sm = pytest.importorskip("statsmodels.api")

FAMILIES = ["normal", "binomial", "poisson"]


def _data(family: str, seed: int = 7) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = 600
    x1 = rng.normal(size=n)
    x2 = rng.uniform(-1.0, 2.0, size=n)
    cat = rng.choice(["a", "b", "c"], size=n, p=[0.5, 0.3, 0.2])
    cat_eff = np.where(cat == "a", 0.0, np.where(cat == "b", 0.4, -0.3))
    eta = 0.5 + 0.7 * x1 - 0.3 * x2 + cat_eff
    if family == "normal":
        y = eta + rng.normal(scale=0.5, size=n)
    elif family == "binomial":
        y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta))).astype(float)
    elif family == "poisson":
        y = rng.poisson(np.exp(0.3 * eta)).astype(float)
    else:
        raise ValueError(family)
    return pd.DataFrame({"x1": x1, "x2": x2, "cat": cat}), y


def _sm_family(family: str):
    return {
        "normal": sm.families.Gaussian(),
        "binomial": sm.families.Binomial(),
        "poisson": sm.families.Poisson(),
    }[family]


def _fit(family, X, y, features=("x1", "x2", "cat"), **gam_kwargs) -> GAM:
    types = {"x1": "linear", "x2": "linear", "cat": "categorical"}
    mdl = GAM(family=family, **gam_kwargs)
    for f in features:
        mdl.add_feature(name=f, type=types[f])
    mdl.fit(X, y, max_its=200)
    return mdl


def _inject(mdl: GAM, sm_res, D: np.ndarray) -> None:
    """Overwrite gamdist's fitted eta with statsmodels'."""
    eta = np.asarray(sm_res.predict(D, which="linear"), dtype=float)
    mdl.f_bar = eta / mdl._num_features


@pytest.fixture(params=FAMILIES)
def pinned(request):
    """Full and restricted (no ``cat``) models pinned to statsmodels' MLE."""
    family = request.param
    X, y = _data(family)
    full = _fit(family, X, y)
    restricted = _fit(family, X, y, features=("x1", "x2"))
    D, names = full._design_matrix()
    sm_full = sm.GLM(y, D, family=_sm_family(family)).fit()
    sm_restricted = sm.GLM(y, D[:, :3], family=_sm_family(family)).fit()
    _inject(full, sm_full, D)
    _inject(restricted, sm_restricted, D[:, :3])
    return family, full, restricted, sm_full, sm_restricted, D, names


def test_parameter_summary_matches_statsmodels(pinned) -> None:
    _, full, _, sm_full, _, _, names = pinned
    tab = full.parameter_summary()
    assert list(tab.index) == names
    assert list(tab.columns) == [
        "estimate",
        "std_err",
        "z",
        "p_value",
        "ci_lower",
        "ci_upper",
    ]
    np.testing.assert_allclose(tab["estimate"], sm_full.params, rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(tab["std_err"], sm_full.bse, rtol=1e-6)
    np.testing.assert_allclose(tab["z"], sm_full.tvalues, rtol=1e-6)
    np.testing.assert_allclose(tab["p_value"], sm_full.pvalues, rtol=1e-4, atol=1e-14)
    ci = np.asarray(sm_full.conf_int(alpha=0.05))
    np.testing.assert_allclose(tab[["ci_lower", "ci_upper"]], ci, rtol=1e-5, atol=1e-10)


def test_parameter_summary_robust_matches_statsmodels(pinned) -> None:
    family, full, _, _, _, D, _ = pinned
    X, y = _data(family)
    sm_hc = sm.GLM(y, D, family=_sm_family(family)).fit(cov_type="HC0")
    _inject(full, sm_hc, D)
    tab = full.parameter_summary(cov_type="HC0", width=0.9)
    np.testing.assert_allclose(tab["std_err"], sm_hc.bse, rtol=1e-6)
    ci = np.asarray(sm_hc.conf_int(alpha=0.1))
    np.testing.assert_allclose(tab[["ci_lower", "ci_upper"]], ci, rtol=1e-5, atol=1e-10)


def test_wald_test_feature_name_is_joint_contrast_test(pinned) -> None:
    _, full, _, sm_full, _, _, _ = pinned
    res = full.wald_test("cat")
    R = np.eye(5)[3:]
    ref = sm_full.wald_test(R, scalar=True)
    assert isinstance(res, HypothesisTest)
    assert res.test == "wald" and res.distribution == "chi2" and res.df == 2
    np.testing.assert_allclose(res.statistic, ref.statistic, rtol=1e-6)
    np.testing.assert_allclose(res.p_value, ref.pvalue, rtol=1e-4)
    # Same hypothesis, three spellings.
    by_cols = full.wald_test(["cat[T.b]", "cat[T.c]"])
    by_matrix = full.wald_test(R)
    np.testing.assert_allclose(by_cols.statistic, res.statistic, rtol=1e-12)
    np.testing.assert_allclose(by_matrix.statistic, res.statistic, rtol=1e-12)


def test_wald_test_single_coefficient_is_z_squared(pinned) -> None:
    _, full, _, _, _, _, _ = pinned
    tab = full.parameter_summary()
    res = full.wald_test("x1")
    np.testing.assert_allclose(res.statistic, tab.loc["x1", "z"] ** 2, rtol=1e-10)
    np.testing.assert_allclose(res.p_value, tab.loc["x1", "p_value"], rtol=1e-6)


def test_wald_test_nonzero_value_and_f(pinned) -> None:
    _, full, _, sm_full, _, _, _ = pinned
    R = np.array([[0.0, 1.0, 1.0, 0.0, 0.0]])
    res = full.wald_test(R, value=[0.4], use_f=True)
    ref = sm_full.wald_test((R, np.array([0.4])), use_f=True, scalar=True)
    assert res.distribution == "f" and res.df_denom == 600 - 5
    np.testing.assert_allclose(res.statistic, ref.statistic, rtol=1e-6)
    np.testing.assert_allclose(res.p_value, ref.pvalue, rtol=1e-4)


def test_lr_test_matches_deviance_difference(pinned) -> None:
    family, full, restricted, sm_full, sm_restricted, _, _ = pinned
    dev_diff = sm_restricted.deviance - sm_full.deviance
    res = full.lr_test(restricted)
    # Order of arguments does not matter.
    res_rev = restricted.lr_test(full)
    assert res == res_rev
    assert res.test == "likelihood_ratio" and res.df == pytest.approx(2.0)
    if family == "normal":
        # Estimated dispersion -> F test, as in R's anova(test = "F").
        assert res.distribution == "f" and res.df_denom == pytest.approx(595.0)
        expected = (dev_diff / 2.0) / sm_full.scale
        np.testing.assert_allclose(res.statistic, expected, rtol=1e-6)
        np.testing.assert_allclose(
            res.p_value, scipy_stats.f.sf(expected, 2, 595), rtol=1e-4
        )
    else:
        assert res.distribution == "chi2"
        np.testing.assert_allclose(res.statistic, dev_diff, rtol=1e-6)
        lr_sm = 2.0 * (sm_full.llf - sm_restricted.llf)
        np.testing.assert_allclose(res.statistic, lr_sm, rtol=1e-6)


def test_score_test_matches_statsmodels(pinned) -> None:
    _, full, restricted, _, sm_restricted, D, _ = pinned
    res = full.score_test(restricted)
    stat, pval = sm_restricted.score_test(exog_extra=D[:, 3:])[:2]
    assert res.test == "score" and res.df == pytest.approx(2.0)
    np.testing.assert_allclose(res.statistic, np.ravel(stat)[0], rtol=1e-6)
    np.testing.assert_allclose(res.p_value, np.ravel(pval)[0], rtol=1e-4)


def test_wald_score_lr_asymptotically_agree(pinned) -> None:
    family, full, restricted, *_ = pinned
    w = full.wald_test("cat")
    s = full.score_test(restricted)
    lr = full.lr_test(restricted)
    lr_chi2 = lr.statistic * lr.df if lr.distribution == "f" else lr.statistic
    # The trinity differs at O(n^{-1/2}); with n=600 they agree within ~25%.
    for a in (w.statistic, s.statistic):
        assert a == pytest.approx(lr_chi2, rel=0.25)


@pytest.mark.parametrize("family", FAMILIES)
def test_end_to_end_close_to_statsmodels(family: str) -> None:
    X, y = _data(family)
    full = _fit(family, X, y)
    D, _ = full._design_matrix()
    ref = sm.GLM(y, D, family=_sm_family(family)).fit()
    tab = full.parameter_summary()
    np.testing.assert_allclose(tab["estimate"], ref.params, atol=5e-2)
    np.testing.assert_allclose(tab["std_err"], ref.bse, rtol=5e-2)


def test_lr_test_detects_real_and_null_effects() -> None:
    rng = np.random.default_rng(11)
    n = 800
    x = rng.normal(size=n)
    noise = rng.normal(size=n)
    y = rng.poisson(np.exp(0.2 + 0.5 * x)).astype(float)
    X = pd.DataFrame({"x": x, "noise": noise})

    def fit(features):
        m = GAM(family="poisson")
        for f in features:
            m.add_feature(name=f, type="linear")
        m.fit(X, y, max_its=200)
        return m

    m_x, m_xn, m_n = fit(["x"]), fit(["x", "noise"]), fit(["noise"])
    assert m_xn.lr_test(m_x).p_value > 0.01  # noise adds nothing
    assert m_xn.lr_test(m_n).p_value < 1e-10  # x matters


def test_weighted_nonrobust_covariance_matches_statsmodels() -> None:
    X, y = _data("poisson")
    w = np.random.default_rng(3).uniform(0.5, 2.0, size=len(y))
    mdl = GAM(family="poisson")
    for f, t in [("x1", "linear"), ("x2", "linear"), ("cat", "categorical")]:
        mdl.add_feature(name=f, type=t)
    mdl.fit(X, y, weights=w, max_its=200)
    D, _ = mdl._design_matrix()
    ref = sm.GLM(y, D, family=sm.families.Poisson(), var_weights=w).fit()
    _inject(mdl, ref, D)
    np.testing.assert_allclose(
        mdl.parameter_covariance(), ref.cov_params(), rtol=1e-6, atol=1e-14
    )


def test_estimated_overdispersion_scales_covariance() -> None:
    X, y = _data("poisson")
    base = _fit("poisson", X, y)
    over = _fit("poisson", X, y, estimate_overdispersion=True)
    over.f_bar = base.f_bar.copy()
    phi = over.dispersion()
    np.testing.assert_allclose(
        over.parameter_covariance(), phi * base.parameter_covariance(), rtol=1e-10
    )
    # Unknown dispersion switches the LR test to F.
    restricted = _fit(
        "poisson", X, y, features=("x1", "x2"), estimate_overdispersion=True
    )
    assert over.lr_test(restricted).distribution == "f"


def test_regularized_model_warns() -> None:
    X, y = _data("normal")
    mdl = GAM(family="normal")
    mdl.add_feature(name="x1", type="linear", regularization={"l2": {"coef": 200.0}})
    mdl.add_feature(name="x2", type="linear")
    mdl.fit(X, y, max_its=200)
    with pytest.warns(UserWarning, match="regularized"):
        mdl.parameter_summary()


def test_unregularized_model_does_not_warn() -> None:
    X, y = _data("normal")
    mdl = _fit("normal", X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mdl.parameter_summary()
        mdl.wald_test("cat")


def test_coefficients_recover_feature_contributions() -> None:
    X, y = _data("normal")
    mdl = _fit("normal", X, y)
    D, _ = mdl._design_matrix(X)
    np.testing.assert_allclose(D @ mdl.coefficients().to_numpy(), mdl.predict(X))


def test_errors() -> None:
    X, y = _data("normal")
    full = _fit("normal", X, y)
    with pytest.raises(ValueError, match="matches no coefficient"):
        full.wald_test("nope")
    with pytest.raises(ValueError, match="columns"):
        full.wald_test(np.ones((1, 3)))
    with pytest.raises(ValueError, match="length"):
        full.wald_test("cat", value=[0.0])
    with pytest.raises(NotImplementedError):
        full.parameter_covariance(cov_type="HC3")
    with pytest.raises(ValueError, match="width"):
        full.parameter_summary(width=1.5)
    with pytest.raises(ValueError, match="same degrees of freedom"):
        full.lr_test(full)

    other_y = _fit("normal", X, y + 1.0, features=("x1",))
    with pytest.raises(ValueError, match="same response"):
        full.lr_test(other_y)
    other_family = _fit("poisson", *_data("poisson"), features=("x1",))
    with pytest.raises(ValueError, match="family and link"):
        full.lr_test(other_family)

    X2 = X.assign(z=np.random.default_rng(0).normal(size=len(X)))
    not_nested = GAM(family="normal")
    not_nested.add_feature(name="z", type="linear")
    not_nested.fit(X2, y, max_its=100)
    with pytest.raises(ValueError, match="not nested"):
        full.lr_test(not_nested)

    unfit = GAM(family="normal")
    unfit.add_feature(name="x1", type="linear")
    with pytest.raises(AttributeError):
        unfit.parameter_summary()


def test_quantile_family_rejected() -> None:
    X, y = _data("normal")
    mdl = GAM(family="quantile", tau=0.5)
    mdl.add_feature(name="x1", type="linear")
    mdl.fit(X, y, max_its=80)
    with pytest.raises(NotImplementedError, match="M-estimator"):
        mdl.parameter_summary()
    with pytest.raises(NotImplementedError, match="M-estimator"):
        mdl.lr_test(mdl)
