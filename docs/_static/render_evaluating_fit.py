"""Render the residual-diagnostics figures for the evaluating-fit guide.

Run from the repo root with:

    uv run python docs/_static/render_evaluating_fit.py

Writes four PNGs under ``docs/_static/``:

- ``evaluating_fit_mdl3_diagnostics.png`` --- ``mdl3.plot_residuals()``,
  embedded in the "Residuals vs. fitted, and the QQ plot" section.
- ``evaluating_fit_mdl3_vs_walkability.png`` ---
  ``mdl3.plot_residuals_vs_predictor(X['walkability'])``, embedded
  in the "Residuals vs. a predictor" section.
- ``evaluating_fit_mdl1_vs_neighborhood.png`` ---
  ``mdl1.plot_residuals_vs_predictor(X['neighborhood'])``, embedded
  in the diagnose-and-fix cycle.
- ``evaluating_fit_mdl2_vs_walkability.png`` ---
  ``mdl2.plot_residuals_vs_predictor(X['walkability'])`` with a
  rolling-mean overlay, embedded in the diagnose-and-fix cycle.

The script is checked in only as a record of how the figures were
produced, so the images stay reproducible if the rent example, residual
definitions, or styling assumptions change. It is not invoked by the
docs build.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from gamdist import GAM


def build_dataset() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(42)
    n = 500

    sqft = rng.uniform(0.5, 3.0, n)
    neighborhood = rng.choice(["downtown", "midtown", "suburbs", "rural"], n)
    walkability = rng.uniform(0, 1, n)

    neighborhood_effect = (
        pd.Series(neighborhood)
        .map({"downtown": 0.5, "midtown": 0.2, "suburbs": -0.1, "rural": -0.4})
        .to_numpy()
    )
    walk_effect = 0.5 * walkability**0.7

    rent = (
        1.5 + sqft + neighborhood_effect + walk_effect + rng.normal(0, 0.1, n)
    )

    X = pd.DataFrame(
        {"sqft": sqft, "neighborhood": neighborhood, "walkability": walkability}
    )
    return X, rent


def render_mdl3_diagnostics(mdl3: GAM, X: pd.DataFrame, out_dir: Path) -> None:
    """Side-by-side residuals-vs-fitted + Normal Q-Q for mdl3."""
    res = mdl3.residuals("deviance")
    mu = mdl3.predict(X)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.scatter(mu, res, s=10, alpha=0.6)
    ax1.axhline(0.0, color="grey", linewidth=0.5)
    ax1.set_xlabel("Fitted (mu)")
    ax1.set_ylabel("Deviance residual")
    ax1.set_title("Residuals vs Fitted")

    stats.probplot(res, plot=ax2)
    ax2.set_title("Normal Q-Q")
    ax2.get_lines()[0].set_markersize(4)
    ax2.get_lines()[0].set_alpha(0.6)

    fig.tight_layout()
    fig.savefig(out_dir / "evaluating_fit_mdl3_diagnostics.png", dpi=150)
    plt.close(fig)


def render_mdl3_vs_walkability(mdl3: GAM, X: pd.DataFrame, out_dir: Path) -> None:
    """mdl3 residuals against walkability -- featureless cloud."""
    res = mdl3.residuals("deviance")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(X["walkability"], res, s=10, alpha=0.6)
    ax.axhline(0.0, color="grey", linewidth=0.5)
    ax.set_xlabel("walkability")
    ax.set_ylabel("Deviance residual")
    ax.set_title("Residuals vs walkability")
    fig.tight_layout()
    fig.savefig(out_dir / "evaluating_fit_mdl3_vs_walkability.png", dpi=150)
    plt.close(fig)


def render_mdl1_vs_neighborhood(mdl1: GAM, X: pd.DataFrame, out_dir: Path) -> None:
    """mdl1 residuals against the categorical neighborhood predictor."""
    res = mdl1.residuals("deviance")
    levels = ["downtown", "midtown", "suburbs", "rural"]
    positions = {lvl: i for i, lvl in enumerate(levels)}
    jitter = np.random.default_rng(0).uniform(-0.15, 0.15, len(X))
    xs = np.array([positions[n] for n in X["neighborhood"]]) + jitter

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(xs, res, s=10, alpha=0.6)
    ax.axhline(0.0, color="grey", linewidth=0.5)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(levels)
    ax.set_xlabel("neighborhood")
    ax.set_ylabel("Deviance residual")
    ax.set_title("Residuals vs neighborhood")
    fig.tight_layout()
    fig.savefig(out_dir / "evaluating_fit_mdl1_vs_neighborhood.png", dpi=150)
    plt.close(fig)


def render_mdl2_vs_walkability(mdl2: GAM, X: pd.DataFrame, out_dir: Path) -> None:
    """mdl2 residuals against walkability, with a rolling-mean overlay."""
    res = mdl2.residuals("deviance")
    order = np.argsort(X["walkability"].to_numpy())
    w_sorted = X["walkability"].to_numpy()[order]
    r_sorted = res[order]
    smoothed = pd.Series(r_sorted).rolling(40, center=True, min_periods=10).mean()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(w_sorted, r_sorted, s=10, alpha=0.6)
    ax.axhline(0.0, color="grey", linewidth=0.5)
    ax.plot(w_sorted, smoothed, color="C3", linewidth=2, label="rolling mean")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlabel("walkability")
    ax.set_ylabel("Deviance residual")
    ax.set_title("Residuals vs walkability")
    fig.tight_layout()
    fig.savefig(out_dir / "evaluating_fit_mdl2_vs_walkability.png", dpi=150)
    plt.close(fig)


def main() -> None:
    X, y = build_dataset()

    mdl1 = GAM(family="normal")
    mdl1.add_feature("sqft", type="linear")
    mdl1.fit(X, y)

    mdl2 = GAM(family="normal")
    mdl2.add_feature("sqft", type="linear")
    mdl2.add_feature("neighborhood", type="categorical")
    mdl2.fit(X, y)

    mdl3 = GAM(family="normal")
    mdl3.add_feature("sqft", type="linear")
    mdl3.add_feature("neighborhood", type="categorical")
    mdl3.add_feature("walkability", type="spline")
    mdl3.fit(X, y)

    out_dir = Path(__file__).resolve().parent
    render_mdl3_diagnostics(mdl3, X, out_dir)
    render_mdl3_vs_walkability(mdl3, X, out_dir)
    render_mdl1_vs_neighborhood(mdl1, X, out_dir)
    render_mdl2_vs_walkability(mdl2, X, out_dir)
    print(f"wrote 4 PNGs under {out_dir}")


if __name__ == "__main__":
    main()
