# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Python version

This is **Python 2** code (originally from 2017). It uses `print` statements (e.g. `print 'MSE:', err`), `dict.iteritems()`, and `map()` returning a list. Do not "modernize" syntax to Python 3 in passing — it is a pervasive change that breaks the whole package.

`setup.py` lists `pickle`, `multiprocessing`, and `math` under `install_requires`; these are stdlib and the entries are spurious. Real third-party deps are `numpy`, `scipy`, `matplotlib`, `cvxpy`.

## Install / run

```bash
pip install -e .                # install package locally
./test.py <variety> [--plot] [--save] [--load]
```

`test.py` is the only test driver — there is no `pytest`/`unittest` suite and no lint config. `<variety>` is one of `linear`, `logistic`, `covariate`, `spline`, `additive`, `cv`. Each variety calls a `test_*` function in `test.py`, generates synthetic data, fits a model, and prints MSE. `--save` writes a `<model_name>_model.pckl` file; `--load` rehydrates it instead of fitting.

## Architecture

The package implements GAM fitting via **ADMM** (Alternating Direction Method of Multipliers), following Chu, Keshavarz, & Boyd's distributed-fitting paper. The structure mirrors that decomposition:

- `gamdist/gamdist.py` — `GAM` class. Public API is `add_feature`, `fit`, `predict`, `plot`, `summary`, `deviance`, `confidence_intervals`, `aic`/`aicc`/`gcv`/`ubre`. `fit()` runs the ADMM loop in `gamdist.py:711`: each iteration calls `feature.optimize(fpumz, rho)` per feature (the per-feature primal step), then `GAM._optimize` does the global dual step using a proximal operator selected by `(family, link)`. Convergence is tracked via primal/dual residuals + tolerances against `eps_abs=eps_rel=1e-3`.
- `gamdist/feature.py` — `_Feature` base class (stub).
- `gamdist/linear_feature.py`, `categorical_feature.py`, `spline_feature.py` — concrete feature types, all implementing the same interface: `__init__`, `initialize(x, ...)`, `optimize(fpumz, rho)`, `compute_dual_tol`, `num_params`, `dof`, `predict`, `_save`/`_load`. Splines use cubic regression splines with an `Omega` curvature penalty; smoothing can be set explicitly or via the `rel_dof` (relative degrees of freedom) target.
- `gamdist/proximal_operators.py` — proximal operators for each `(family, link)` pair: normal+identity has a closed form; binomial+logit and poisson+log use Newton's method; gamma+reciprocal has a closed form; non-canonical links fall back to `scipy.optimize.minimize_scalar`. The `_*_scalar` variants exist so they can (in principle) be dispatched through `multiprocessing.Pool`, but the parallel path is currently disabled (`if False:` at `gamdist.py:719`) because of CPython threading/pickling issues — keep this in mind before "fixing" the unused `_feature_wrapper`.

Supported families: `normal`, `binomial`, `poisson`, `gamma`, `exponential` (= gamma with dispersion 1), `inverse_gaussian`. Supported links: `identity`, `logistic`, `probit`, `complementary_log_log`, `log`, `reciprocal`, `reciprocal_squared`. Canonical-link defaults are in `CANONICAL_LINKS`. Some non-canonical combinations are non-convex; convergence is not guaranteed there.

## Persistence

`save_flag=True` on `fit()` writes one pickle per feature plus a top-level `<name>_model.pckl` containing ADMM state (`f_bar`, `z_bar`, `u`, residual histories). `GAM(load_from_file=...)` reconstructs both the model and feature objects. Filenames are derived from the `name=` passed to `GAM()` — `fit(save_flag=True)` requires `name` to be set.

## Conventions

- Files carry a Match Group / Apache 2.0 header — preserve it on edits and add a changelog entry in `changelog.txt` for behavior changes (the header explicitly references this log).
- The header also notes the package is **not designed for untrusted input** (uses `pickle`); don't add network/deserialization paths that assume otherwise.
- Commented-out `cdef` lines in `spline_feature.py` / `categorical_feature.py` are remnants of an aborted Cython port. Leave them; the "Runtime optimization (Cython)" item in the to-do list at the top of `gamdist.py` is the same effort.
