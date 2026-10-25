"""Simulate one gene mechanistically, sequence it with molecule loss, and recover its burst kinetics.

    python examples/single_gene.py
"""

import numpy as np

from burstkin import models
from burstkin.inference import TelegraphLikelihood, classify_identifiability, compare_models, fit_telegraph, profile_likelihood
from burstkin.scdata import beta_capture
from burstkin.ssa import binomial_capture, simulate_stationary

rng = np.random.default_rng(0)
k_on, k_off, k_syn = 0.2, 0.5, 90.0  # bursts every ~5 lifetimes, ~180 transcripts each
cells = 1500

# 1. exact stochastic simulation of the promoter and its mRNA in every cell
net = models.telegraph(k_on, k_off, k_syn)
mrna = simulate_stationary(net, models.telegraph_initial_state(), cells, burn_in=40.0, rng=rng)[:, 2]

# 2. sequencing captures each transcript with a cell-specific probability
capture = beta_capture(cells, mean=0.2, cv=0.25, rng=rng)
counts = binomial_capture(mrna, capture, rng)

# 3. maximum likelihood, model comparison and profile-likelihood intervals
lik = TelegraphLikelihood(counts, capture)
fit = fit_telegraph(counts, capture, likelihood=lik)
comparison = compare_models(counts, capture, telegraph_fit=fit)
profiles = {t: profile_likelihood(lik, fit, t) for t in ("k_on", "k_off", "k_syn", "burst_size")}

truth = {"k_on": k_on, "k_off": k_off, "k_syn": k_syn, "burst_size": k_syn / k_off}
print(f"best model by BIC: {comparison.best}")
for name, p in profiles.items():
    lo, hi = ("open" if v is None else f"{v:.3g}" for v in p.ci)
    print(f"{name:>10}: truth {truth[name]:8.3g}   estimate {p.mle:8.3g}   95% CI [{lo}, {hi}]")
print("identifiability:", classify_identifiability(profiles))
