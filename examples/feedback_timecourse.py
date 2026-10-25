"""Fit a self-repressing gene to protein snapshots taken at several times after induction.

The model has no closed-form distribution, so the likelihood comes from the
finite state projection of the master equation.

    python examples/feedback_timecourse.py      (about a minute)
"""

import numpy as np

from burstkin import models
from burstkin.fsp_inference import Snapshots, SnapshotLikelihood
from burstkin.ssa import simulate

net = models.autoregulation()  # rho_u=60, rho_b=4, sigma_b=0.4, sigma_u=3
x0 = models.autoregulation_initial_state()
times = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0])

# independent cells at each time point, as in a fixed-cell time course
X = simulate(net, x0, times, n_cells=600, rng=np.random.default_rng(1))
data = Snapshots(times, [X[:, i, 2] for i in range(times.size)])

free = ("rho_u", "rho_b", "sigma_b", "sigma_u")
lik = SnapshotLikelihood(net, x0, "P", data, free, bounds={"P": 100})
fit = lik.fit([{"rho_u": 30.0, "rho_b": 2.0, "sigma_b": 1.0, "sigma_u": 1.0}])

print(f"FSP truncation error at the optimum: {fit.truncation_error:.1e}")
for p in free:
    lo, hi = fit.interval(p)
    print(f"{p:>8}: truth {net.parameters[p]:6.3g}   estimate {fit.params[p]:6.3g}   95% CI [{lo:.3g}, {hi:.3g}]")
