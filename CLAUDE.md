# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Design philosophy (North Star)

gamdist exists to fit the **zoo** of supervised-learning models that fall under the GLM/GAM umbrella: binary, continuous, or count outcomes paired with continuous, categorical, or spline-transformed features, with arbitrary regularization (ridge, l1, group lasso, curvature penalties, …) attached to whichever terms want it.

Every such model is a single convex optimization problem — but the joint Hessian is a nightmare to derive and solve as one monolith. ADMM (per Chu, Keshavarz, & Boyd) is what makes the zoo tractable: it splits the problem into a per-feature primal step plus a per-outcome proximal step, coordinated by dual variables. Parallelism is a nice side effect; **the real prize is modularity** — outcomes, features, and regularizers are independent components that mix and match in any combination without anyone needing to know about the others.

Practical consequences for development:

- **Convexity is the constraint that buys us everything.** Non-convex links/families compromise the guarantees ADMM gives us; treat them as second-class (the existing non-canonical fallbacks already are). Don't add features, penalties, or proximal operators that break convexity of the per-component subproblems.
- **Keep the seams clean.** The `_Feature` interface (`initialize`/`optimize`/`compute_dual_tol`/`num_params`/`dof`/`predict`/`_save`/`_load`) and the `(family, link)` → proximal-operator dispatch are the modular boundaries. New feature types or new outcome distributions should plug in without the other side learning anything new.
- **New regularizers belong inside a feature's `optimize` step**, not bolted onto the global loop — that's how ADMM keeps them composable.
- Resist designs that route information across the seams (a feature that needs to know the family, a proximal operator that needs to know which features exist, etc.); if a change seems to require it, flag the tension before implementing.

## Python version

Requires **Python 3.11+**. Code uses `from __future__ import annotations` and full PEP 484 type hints throughout. The package was modernized from Python 2 in v0.2.0; see `changelog.txt`. There is no `setup.py` — packaging is via `pyproject.toml` (hatchling). Real third-party deps are `numpy`, `scipy`, `matplotlib`, `cvxpy`, `pandas`.

## Install / run

Use [uv](https://github.com/astral-sh/uv):

```bash
uv sync                  # runtime deps
uv sync --extra dev      # adds pytest, mypy, ruff
uv run pytest            # 96-test suite, ~7s
```

There is no legacy `test.py` driver; everything goes through `pytest`. End-to-end regression varieties (linear, logistic, covariate, spline, additive, smoothing-DOF) live in `tests/test_regression_varieties.py`.

## Architecture

The package implements GAM fitting via **ADMM** (Alternating Direction Method of Multipliers), following Chu, Keshavarz, & Boyd's distributed-fitting paper. The structure mirrors that decomposition:

- `gamdist/gamdist.py` — `GAM` class. Public API: `add_feature`, `fit`, `predict`, `plot`, `summary`, `deviance`, `confidence_intervals`, `aic`/`aicc`/`gcv`/`ubre`. `fit()` runs the ADMM loop (look for `for i in range(max_its):`): each iteration calls `feature.optimize(fpumz, rho)` per feature (the per-feature primal step), then `GAM._optimize` does the global dual step using a proximal operator selected by `(family, link)`. Convergence is tracked via primal/dual residuals + tolerances against `eps_abs=eps_rel=1e-3`.
- `gamdist/feature.py` — `_Feature` abstract base class (`abc.ABC` with `@abstractmethod` declarations). Concrete subclasses must implement the same interface.
- `gamdist/linear_feature.py`, `categorical_feature.py`, `spline_feature.py` — concrete feature types implementing `__init__`, `initialize(x, ...)`, `optimize(fpumz, rho)`, `compute_dual_tol`, `num_params`, `dof`, `predict`, `_save`/`_load`. Splines use cubic regression splines with an `Omega` curvature penalty; smoothing can be set explicitly or via the `rel_dof` (relative degrees of freedom) target.
- `gamdist/proximal_operators.py` — proximal operators for each `(family, link)` pair: normal+identity has a closed form; binomial+logit and poisson+log use undamped Newton with `tol=1e-3` and `max_its=100` (raises on no-converge); gamma+reciprocal has a closed form; non-canonical links fall back to `scipy.optimize.minimize_scalar`. Logistic link uses `scipy.special.expit`/`logit` to stay well-behaved on extreme linear predictors.

Supported families: `normal`, `binomial`, `poisson`, `gamma`, `exponential` (= gamma with dispersion 1), `inverse_gaussian`. Supported links: `identity`, `logistic`, `probit`, `complementary_log_log`, `log`, `reciprocal`, `reciprocal_squared`. Canonical-link defaults are in `CANONICAL_LINKS`. Some non-canonical combinations are non-convex; convergence is not guaranteed there.

The categorical-feature LP/QP uses CLARABEL by default (ECOS is no longer bundled with cvxpy ≥ 1.4). The `multiprocessing.Pool` parallel-feature path was removed in v0.2.0; "Fit in parallel" remains a documented to-do.

## Persistence

`save_flag=True` on `fit()` writes one pickle per feature plus a top-level `<name>_model.pckl` containing ADMM state (`f_bar`, `z_bar`, `u`, residual histories). `GAM(load_from_file=...)` reconstructs both the model and feature objects. Filenames are derived from the `name=` passed to `GAM()` — `fit(save_flag=True)` requires `name` to be set. Pickle files use binary mode; v0.1 (Python 2) pickles are not loadable.

## Tests / CI

`pytest` suite is 96 tests, 84% line coverage. CI (`.github/workflows/ci.yml`) runs `ruff check`, `mypy gamdist`, and `pytest --cov=gamdist --cov-fail-under=80` on Python 3.11 and 3.12. Lint/type config lives in `pyproject.toml`.

`GAM.confidence_intervals()` is an intentional `NotImplementedError` stub (and tested as such); the corresponding entry on the to-do list at the top of `gamdist/gamdist.py` is still open.

## Conventions

- Files carry a Match Group / Apache 2.0 header — preserve it on edits and add a `changelog.txt` entry for behavior changes (the header explicitly references this log).
- The header also notes the package is **not designed for untrusted input** (uses `pickle`); don't add network/deserialization paths that assume otherwise.
