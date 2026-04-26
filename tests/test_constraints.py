"""Tests for convex shape constraints on feature coefficients.

Covers ``constraints={...}`` on ``_LinearFeature`` (sign / box on the
slope), ``_CategoricalFeature`` (box / monotonic / convex / concave on
the per-level coefficient vector along a user-supplied order), and
``_SplineFeature`` (box / monotonic / convex / concave on the spline
evaluated at the knots). For each constraint type, the unconstrained
fit violates the constraint while the constrained fit satisfies it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gamdist import GAM
from gamdist.categorical_feature import _CategoricalFeature
from gamdist.linear_feature import _LinearFeature
from gamdist.spline_feature import _SplineFeature

# ---------------------------------------------------------------------------
# Linear feature: sign / box on the slope
# ---------------------------------------------------------------------------


def test_linear_unknown_constraint_key_raises() -> None:
    with pytest.raises(ValueError, match="unknown constraint key"):
        _LinearFeature(name="x", constraints={"bogus": 1.0})


def test_linear_rejects_monotonic() -> None:
    with pytest.raises(ValueError, match="linear feature does not support"):
        _LinearFeature(name="x", constraints={"monotonic": "increasing"})


def test_linear_sign_invalid_value_raises() -> None:
    with pytest.raises(ValueError, match="expected 'nonnegative' or 'nonpositive'"):
        _LinearFeature(name="x", constraints={"sign": "weird"})


def test_linear_sign_combined_with_lower_raises() -> None:
    with pytest.raises(ValueError, match="shorthand"):
        _LinearFeature(name="x", constraints={"sign": "nonnegative", "lower": -1.0})


def test_linear_box_inverted_raises() -> None:
    with pytest.raises(ValueError, match="exceeds upper"):
        _LinearFeature(name="x", constraints={"lower": 1.0, "upper": 0.0})


def test_linear_sign_nonpositive_clips_positive_unconstrained() -> None:
    # Data has positive correlation; unconstrained slope > 0; sign=nonpositive
    # forces m <= 0 so the projection clips to 0.
    rng = np.random.default_rng(0)
    n = 100
    x = rng.normal(size=n)
    y = 2.0 * (x - x.mean()) + 0.05 * rng.normal(size=n)

    unconstrained = _LinearFeature(name="x")
    unconstrained.initialize(x)
    unconstrained.optimize(-y, rho=1.0)
    assert unconstrained._m > 1.0

    constrained = _LinearFeature(name="x", constraints={"sign": "nonpositive"})
    constrained.initialize(x)
    constrained.optimize(-y, rho=1.0)
    assert constrained._m == 0.0


def test_linear_sign_nonnegative_passes_through() -> None:
    rng = np.random.default_rng(1)
    n = 100
    x = rng.normal(size=n)
    y = 2.0 * (x - x.mean()) + 0.05 * rng.normal(size=n)

    unconstrained = _LinearFeature(name="x")
    unconstrained.initialize(x)
    unconstrained.optimize(-y, rho=1.0)

    constrained = _LinearFeature(name="x", constraints={"sign": "nonnegative"})
    constrained.initialize(x)
    constrained.optimize(-y, rho=1.0)
    # The unconstrained minimum is in the feasible region, so the
    # constrained fit equals the unconstrained fit.
    assert constrained._m == pytest.approx(unconstrained._m, rel=1e-12, abs=1e-12)


def test_linear_box_clips_to_upper() -> None:
    rng = np.random.default_rng(2)
    n = 200
    x = rng.normal(size=n)
    y = 3.0 * (x - x.mean()) + 0.05 * rng.normal(size=n)

    unconstrained = _LinearFeature(name="x")
    unconstrained.initialize(x)
    unconstrained.optimize(-y, rho=1.0)
    assert unconstrained._m > 1.5

    constrained = _LinearFeature(name="x", constraints={"upper": 1.5})
    constrained.initialize(x)
    constrained.optimize(-y, rho=1.0)
    assert constrained._m == pytest.approx(1.5)


def test_linear_box_unconstrained_optimum_inside_box() -> None:
    rng = np.random.default_rng(3)
    n = 100
    x = rng.normal(size=n)
    y = 0.5 * (x - x.mean()) + 0.05 * rng.normal(size=n)

    unconstrained = _LinearFeature(name="x")
    unconstrained.initialize(x)
    unconstrained.optimize(-y, rho=1.0)

    constrained = _LinearFeature(name="x", constraints={"lower": -2.0, "upper": 2.0})
    constrained.initialize(x)
    constrained.optimize(-y, rho=1.0)
    assert constrained._m == pytest.approx(unconstrained._m, rel=1e-12, abs=1e-12)


def test_linear_sign_in_gam(tmp_path: Path) -> None:
    # A signal-then-flip-the-sign scenario: the data wants a positive slope
    # but the user requires a non-positive one. The fitted slope should be
    # zero (clipped at the boundary).
    rng = np.random.default_rng(4)
    n = 300
    x = rng.normal(size=n)
    y = 1.0 * x + 0.1 * rng.normal(size=n)
    X = pd.DataFrame({"x": x})

    mdl = GAM(family="normal")
    mdl.add_feature(name="x", type="linear", constraints={"sign": "nonpositive"})
    mdl.fit(X, y, max_its=40)
    feat = mdl._features["x"]
    assert feat._m <= 1e-9  # type: ignore[attr-defined]


def test_linear_box_save_load_round_trip(tmp_path: Path) -> None:
    feat = _LinearFeature(name="x", constraints={"lower": -0.5, "upper": 0.75})
    feat.initialize(
        np.arange(5.0),
        save_flag=True,
        save_prefix=str(tmp_path / "model"),
    )
    feat._m = 0.5
    feat._b = 0.0
    feat._save()

    restored = _LinearFeature(load_from_file=feat._filename)
    assert restored._has_constraints
    assert restored._lower == pytest.approx(-0.5)
    assert restored._upper == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Categorical feature: box / monotonic / convex / concave
# ---------------------------------------------------------------------------


def test_categorical_unknown_constraint_key_raises() -> None:
    with pytest.raises(ValueError, match="unknown constraint key"):
        _CategoricalFeature(name="g", constraints={"bogus": 1})


def test_categorical_monotonic_invalid_direction_raises() -> None:
    with pytest.raises(ValueError, match="expected 'increasing' or 'decreasing'"):
        _CategoricalFeature(
            name="g", constraints={"monotonic": "weird", "order": ["a", "b"]}
        )


def test_categorical_monotonic_requires_order() -> None:
    with pytest.raises(ValueError, match="require an 'order' list"):
        _CategoricalFeature(name="g", constraints={"monotonic": "increasing"})


def test_categorical_convex_concave_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        _CategoricalFeature(
            name="g",
            constraints={"convex": True, "concave": True, "order": ["a", "b", "c"]},
        )


def test_categorical_order_too_short_raises() -> None:
    with pytest.raises(ValueError, match="at least two categories"):
        _CategoricalFeature(
            name="g", constraints={"monotonic": "increasing", "order": ["a"]}
        )


def test_categorical_order_duplicates_raise() -> None:
    with pytest.raises(ValueError, match="must be unique"):
        _CategoricalFeature(
            name="g",
            constraints={"monotonic": "increasing", "order": ["a", "a", "b"]},
        )


def _fit_categorical(
    x: np.ndarray,
    y: np.ndarray,
    regularization: dict | None = None,
    constraints: dict | None = None,
    iters: int = 30,
) -> _CategoricalFeature:
    feat = _CategoricalFeature(
        name="g", regularization=regularization, constraints=constraints
    )
    feat.initialize(x)
    n = len(y)
    fpumz = -np.asarray(y, dtype=float) * n
    for _ in range(iters):
        feat.optimize(fpumz, rho=1.0)
    return feat


def test_categorical_box_clips_extreme_levels() -> None:
    # Per-level means: a ~ -3, b ~ 0, c ~ +3. Unconstrained fit puts large
    # positive / negative effects on a and c. A tight box [-1, 1] forces them
    # to clip while preserving relative ordering.
    rng = np.random.default_rng(0)
    n = 600
    cats = np.array(["a", "b", "c"] * (n // 3))
    rng.shuffle(cats)
    means = {"a": -3.0, "b": 0.0, "c": 3.0}
    y = np.array([means[c] for c in cats]) + 0.1 * rng.normal(size=len(cats))

    unconstrained = _fit_categorical(cats, y)
    assert unconstrained.p.max() > 2.0
    assert unconstrained.p.min() < -2.0

    constrained = _fit_categorical(cats, y, constraints={"lower": -1.0, "upper": 1.0})
    # Box satisfied (with a small numerical slack from CLARABEL).
    assert constrained.p.max() <= 1.0 + 1e-6
    assert constrained.p.min() >= -1.0 - 1e-6


def test_categorical_monotonic_increasing() -> None:
    # The data has a non-monotone per-category mean (b > c) so the
    # unconstrained fit violates the requested ordering. The
    # monotonic-increasing constraint produces a fit that is non-decreasing
    # along the supplied order.
    rng = np.random.default_rng(1)
    order = ["a", "b", "c", "d"]
    means = {"a": -1.0, "b": 1.0, "c": 0.5, "d": 2.0}
    n_per = 200
    cats = np.repeat(order, n_per)
    y = np.array([means[c] for c in cats]) + 0.1 * rng.normal(size=len(cats))

    unconstrained = _fit_categorical(cats, y)
    h = unconstrained._category_hash
    # Confirm that the unconstrained fit violates the requested ordering.
    assert unconstrained.p[h["b"]] > unconstrained.p[h["c"]]

    constrained = _fit_categorical(
        cats, y, constraints={"monotonic": "increasing", "order": order}
    )
    h2 = constrained._category_hash
    seq = [constrained.p[h2[c]] for c in order]
    diffs = np.diff(seq)
    assert (diffs >= -1e-6).all()


def test_categorical_monotonic_decreasing() -> None:
    rng = np.random.default_rng(2)
    order = ["a", "b", "c", "d"]
    means = {"a": 2.0, "b": 0.5, "c": 1.0, "d": -1.0}
    n_per = 200
    cats = np.repeat(order, n_per)
    y = np.array([means[c] for c in cats]) + 0.1 * rng.normal(size=len(cats))

    constrained = _fit_categorical(
        cats, y, constraints={"monotonic": "decreasing", "order": order}
    )
    h = constrained._category_hash
    seq = [constrained.p[h[c]] for c in order]
    diffs = np.diff(seq)
    assert (diffs <= 1e-6).all()


def test_categorical_convex_constraint() -> None:
    # Unconstrained fit follows a concave shape (peak in the middle); the
    # convex constraint flips the second-difference sign and forces a U
    # (or near-linear) profile along the order.
    rng = np.random.default_rng(3)
    order = ["a", "b", "c", "d", "e"]
    # Concave mean profile: arches upward in the middle.
    means = {"a": 0.0, "b": 1.0, "c": 1.5, "d": 1.0, "e": 0.0}
    n_per = 200
    cats = np.repeat(order, n_per)
    y = np.array([means[c] for c in cats]) + 0.1 * rng.normal(size=len(cats))

    unconstrained = _fit_categorical(cats, y)
    h = unconstrained._category_hash
    seq = np.array([unconstrained.p[h[c]] for c in order])
    # Unconstrained is concave: at least one second-difference is negative.
    assert np.diff(seq, n=2).min() < -0.1

    constrained = _fit_categorical(
        cats, y, constraints={"convex": True, "order": order}
    )
    h2 = constrained._category_hash
    seq2 = np.array([constrained.p[h2[c]] for c in order])
    assert (np.diff(seq2, n=2) >= -1e-6).all()


def test_categorical_concave_constraint() -> None:
    # Mirror of the convex test: convex unconstrained data + concave
    # constraint forces second differences to be non-positive.
    rng = np.random.default_rng(4)
    order = ["a", "b", "c", "d", "e"]
    # Convex mean profile.
    means = {"a": 1.5, "b": 0.5, "c": 0.0, "d": 0.5, "e": 1.5}
    n_per = 200
    cats = np.repeat(order, n_per)
    y = np.array([means[c] for c in cats]) + 0.1 * rng.normal(size=len(cats))

    constrained = _fit_categorical(
        cats, y, constraints={"concave": True, "order": order}
    )
    h = constrained._category_hash
    seq = np.array([constrained.p[h[c]] for c in order])
    assert (np.diff(seq, n=2) <= 1e-6).all()


def test_categorical_constraint_save_load_round_trip(tmp_path: Path) -> None:
    feat = _CategoricalFeature(
        name="g",
        constraints={
            "monotonic": "increasing",
            "lower": -2.0,
            "upper": 2.0,
            "order": ["a", "b", "c"],
        },
    )
    feat.initialize(
        np.array(["a", "b", "c", "a", "b"]),
        save_flag=True,
        save_prefix=str(tmp_path / "model"),
    )

    restored = _CategoricalFeature(load_from_file=feat._filename)
    assert restored._has_constraints
    assert restored._constraint_lower == pytest.approx(-2.0)
    assert restored._constraint_upper == pytest.approx(2.0)
    assert restored._constraint_monotonic == "increasing"
    assert restored._constraint_order == ["a", "b", "c"]


def test_categorical_constraint_in_gam_fit() -> None:
    rng = np.random.default_rng(5)
    order = ["lo", "med", "hi"]
    means = {"lo": -0.5, "med": 1.5, "hi": 0.5}  # not monotone increasing
    n_per = 300
    cats = np.repeat(order, n_per)
    y = np.array([means[c] for c in cats]) + 0.1 * rng.normal(size=len(cats))
    X = pd.DataFrame({"g": cats})

    mdl = GAM(family="normal")
    mdl.add_feature(
        name="g",
        type="categorical",
        constraints={"monotonic": "increasing", "order": order},
    )
    mdl.fit(X, y, max_its=40)
    feat = mdl._features["g"]
    h = feat._category_hash  # type: ignore[attr-defined]
    p = feat.p  # type: ignore[attr-defined]
    seq = [p[h[c]] for c in order]
    assert seq[0] <= seq[1] + 1e-6
    assert seq[1] <= seq[2] + 1e-6


# ---------------------------------------------------------------------------
# Spline feature: box / monotonic / convex / concave
# ---------------------------------------------------------------------------


def test_spline_unknown_constraint_key_raises() -> None:
    with pytest.raises(ValueError, match="unknown constraint key"):
        _SplineFeature(name="x", constraints={"bogus": 1})


def test_spline_monotonic_invalid_direction_raises() -> None:
    with pytest.raises(ValueError, match="expected 'increasing' or 'decreasing'"):
        _SplineFeature(name="x", constraints={"monotonic": "sideways"})


def test_spline_convex_concave_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        _SplineFeature(name="x", constraints={"convex": True, "concave": True})


def _gam_fit_spline(
    x: np.ndarray,
    y: np.ndarray,
    constraints: dict | None = None,
    rel_dof: float = 6.0,
) -> _SplineFeature:
    X = pd.DataFrame({"x": x})
    mdl = GAM(family="normal")
    mdl.add_feature(name="x", type="spline", rel_dof=rel_dof, constraints=constraints)
    mdl.fit(X, y, max_its=60)
    return mdl._features["x"]  # type: ignore[return-value]


def test_spline_monotonic_increasing() -> None:
    # Full sine cycle on [0, 1]: clearly non-monotone. Increasing constraint
    # produces a non-decreasing fit at the knots.
    rng = np.random.default_rng(0)
    n = 300
    x = np.sort(rng.uniform(0.0, 1.0, size=n))
    y = np.sin(2.0 * np.pi * x) + 0.1 * rng.normal(size=n)

    unconstrained = _gam_fit_spline(x, y)
    f_xi = unconstrained._N_xi @ unconstrained._theta
    assert (np.diff(f_xi) < 0).any()

    constrained = _gam_fit_spline(x, y, constraints={"monotonic": "increasing"})
    f_xi_c = constrained._N_xi @ constrained._theta
    assert (np.diff(f_xi_c) >= -1e-4).all()


def test_spline_monotonic_decreasing() -> None:
    rng = np.random.default_rng(1)
    n = 300
    x = np.sort(rng.uniform(0.0, 1.0, size=n))
    y = np.cos(2.0 * np.pi * x) + 0.1 * rng.normal(size=n)

    constrained = _gam_fit_spline(x, y, constraints={"monotonic": "decreasing"})
    f_xi = constrained._N_xi @ constrained._theta
    assert (np.diff(f_xi) <= 1e-4).all()


def test_spline_convex_constraint() -> None:
    # Concave true mean -> convex constraint flips the curvature at the knots.
    rng = np.random.default_rng(2)
    n = 400
    x = np.sort(rng.uniform(-1.0, 1.0, size=n))
    y = -(x**2) + 0.05 * rng.normal(size=n)

    unconstrained = _gam_fit_spline(x, y)
    f_xi = unconstrained._N_xi @ unconstrained._theta
    assert np.diff(f_xi, n=2).min() < -0.01

    constrained = _gam_fit_spline(x, y, constraints={"convex": True})
    f_xi_c = constrained._N_xi @ constrained._theta
    assert (np.diff(f_xi_c, n=2) >= -1e-4).all()


def test_spline_concave_constraint() -> None:
    rng = np.random.default_rng(3)
    n = 400
    x = np.sort(rng.uniform(-1.0, 1.0, size=n))
    y = (x**2) + 0.05 * rng.normal(size=n)

    constrained = _gam_fit_spline(x, y, constraints={"concave": True})
    f_xi = constrained._N_xi @ constrained._theta
    assert (np.diff(f_xi, n=2) <= 1e-4).all()


def test_spline_box_constraint() -> None:
    # Strong full sine on [-1, 1] with amplitude 2: an unconstrained fit
    # exceeds +/-1; box at +/-0.5 forces the spline values at the knots to
    # stay inside the band.
    rng = np.random.default_rng(4)
    n = 400
    x = np.sort(rng.uniform(-1.0, 1.0, size=n))
    y = 2.0 * np.sin(2.0 * np.pi * x) + 0.05 * rng.normal(size=n)

    unconstrained = _gam_fit_spline(x, y)
    f_xi = unconstrained._N_xi @ unconstrained._theta
    assert f_xi.max() > 1.0
    assert f_xi.min() < -1.0

    constrained = _gam_fit_spline(x, y, constraints={"lower": -0.5, "upper": 0.5})
    f_xi_c = constrained._N_xi @ constrained._theta
    assert f_xi_c.max() <= 0.5 + 1e-4
    assert f_xi_c.min() >= -0.5 - 1e-4


def test_spline_constraint_save_load_round_trip(tmp_path: Path) -> None:
    feat = _SplineFeature(
        name="x", constraints={"monotonic": "increasing", "upper": 5.0}
    )
    feat.initialize(
        np.linspace(0.0, 1.0, 50),
        save_flag=True,
        save_prefix=str(tmp_path / "model"),
    )
    restored = _SplineFeature(load_from_file=feat._filename)
    assert restored._has_constraints
    assert restored._constraint_monotonic == "increasing"
    assert restored._constraint_upper == pytest.approx(5.0)
    assert np.isfinite(restored._N_xi).all()


def test_spline_constraint_in_gam_fit_dose_response() -> None:
    # Classic dose-response: noisy realisation of a triangle wave that's
    # non-monotonic; the increasing constraint produces a monotone fit.
    rng = np.random.default_rng(6)
    n = 300
    x = np.sort(rng.uniform(0.0, 1.0, size=n))
    y = np.where(x < 0.5, x, 1.0 - x) + 0.05 * rng.normal(size=n)

    feat = _gam_fit_spline(x, y, constraints={"monotonic": "increasing"})
    f_xi = feat._N_xi @ feat._theta
    assert (np.diff(f_xi) >= -1e-4).all()
