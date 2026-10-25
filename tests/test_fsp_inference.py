import numpy as np
import pytest

from burstkin import analytic as an
from burstkin import models
from burstkin.fsp_inference import Snapshots, SnapshotLikelihood
from burstkin.ssa import simulate


def test_stationary_snapshot_likelihood_equals_closed_form(rng):
    k = (0.6, 1.8, 25.0)
    counts = an.sample_telegraph(*k, 500, rng)
    lik = SnapshotLikelihood(models.telegraph(*k), models.telegraph_initial_state(), "M",
                             Snapshots([np.inf], [counts]), free=("k_on", "k_off", "k_syn"), bounds={"M": 200})
    theta = np.log(k)
    assert -lik.nll(theta) == pytest.approx(an.telegraph_logpmf(counts, *k).sum(), rel=1e-10)


def test_transient_snapshot_likelihood_equals_exact(rng):
    times = np.array([0.2, 1.0, 4.0])
    counts = [rng.poisson(30 * (1 - np.exp(-t)), 300) for t in times]
    lik = SnapshotLikelihood(models.birth_death(30.0, 1.0), [0], "M", Snapshots(times, counts), free=("k",), bounds={"M": 120})
    exact = sum(np.log(an.birth_death_pmf(120, t, 30.0, 1.0)[c]).sum() for t, c in zip(times, counts))
    assert -lik.nll(np.log([30.0])) == pytest.approx(exact, rel=1e-10)


def test_feedback_model_recovered_from_time_course(rng):
    net, x0 = models.autoregulation(), models.autoregulation_initial_state()
    times = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    X = simulate(net, x0, times, n_cells=300, rng=rng)
    free = ("rho_u", "sigma_b", "sigma_u")
    lik = SnapshotLikelihood(net, x0, "P", Snapshots(times, [X[:, i, 2] for i in range(times.size)]), free, {"P": 100})
    fit = lik.fit([{"rho_u": 30.0, "sigma_b": 1.0, "sigma_u": 1.0}])
    truth = net.parameters
    assert fit.nll <= lik.nll(np.log([truth[p] for p in free])) + 1e-6  # at least as good as the truth
    assert fit.truncation_error < 1e-9
    for p in free:
        z = np.log(fit.params[p] / truth[p]) / fit.standard_errors[p]
        assert abs(z) < 3.5, (p, fit.params[p], z)


def test_snapshot_validation():
    with pytest.raises(ValueError):
        Snapshots([1.0, 0.5], [[1], [2]])
    with pytest.raises(ValueError):
        SnapshotLikelihood(models.birth_death(), [0], "M", Snapshots([1.0], [[500]]), ("k",), {"M": 50})
    with pytest.raises(ValueError):
        SnapshotLikelihood(models.birth_death(), [0], "M", Snapshots([1.0], [[5]]), ("nope",), {"M": 50})
