gamdist
=======

gamdist fits the GLM/GAM zoo — binary, continuous, or count outcomes
paired with continuous, categorical, or spline-transformed features,
with arbitrary convex regularization attached to whichever terms want
it. Every such model is a single convex optimization problem.

The library provides:

- **Feature types**: linear (ridge), categorical (L1, L2, group lasso,
  network lasso, network ridge), and spline with an integrated
  curvature penalty
- **Families and links**: normal, binomial, Poisson, gamma,
  exponential, and inverse Gaussian; identity, logit, probit,
  complementary log-log, log, reciprocal, and reciprocal-squared links
- **Multi-task fitting**: :class:`~gamdist.MultiTaskGAM` for K
  correlated responses with optional cross-task coupling
- **Model selection**: AIC, AICc, GCV, UBRE, and effective degrees of
  freedom for each feature

The ADMM decomposition of [CKB13]_ makes the zoo tractable by
splitting the joint problem into a per-feature primal step plus a
per-outcome proximal step coordinated by dual variables. Parallelism
is a side effect; **the real prize is modularity** — outcomes,
features, and regularizers are independent components that mix and
match in any combination without any side needing to know about the
others. Full citations are on the :doc:`references` page.

Quickstart
----------

The core workflow: construct a :class:`~gamdist.GAM`, register
features with :meth:`~gamdist.GAM.add_feature`, then call
:meth:`~gamdist.GAM.fit`. For binary outcomes, change ``family`` to
``"binomial"`` and the rest of the API stays the same.

.. code-block:: python

    import numpy as np
    import pandas as pd
    from gamdist import GAM

    X = pd.DataFrame(
        {
            "purchases": np.random.choice([0, 3, 10, 16], size=1000),
            "gender": np.random.choice(["male", "female"], size=1000),
        }
    )
    y = (
        0.1 * np.log1p(X["purchases"].values)
        + np.where(X["gender"].values == "male", 0.1, -0.5)
        + np.random.normal(size=1000) * 0.1
    )

    mdl = GAM(family="normal")
    mdl.add_feature(name="purchases", type="linear", transform=np.log1p)
    mdl.add_feature(name="gender", type="categorical")
    mdl.fit(X, y)

    mdl.summary()
    yhat = mdl.predict(X)

The methods in this library follow the algorithm of [CKB13]_, built on
the ADMM framework of [BPC11]_.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api.rst
   references.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
