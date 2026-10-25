import numpy as np
import pytest

from burstkin import analytic as an
from burstkin import models
from burstkin.fsp import FSP
from burstkin.ssa import binomial_capture, simulate, simulate_stationary
from burstkin.validate import chisquare_gof


def test_birth_death_transient_matches_exact(rng):
    times = [0.0, 0.3, 1.0, 3.0]
    X = simulate(models.birth_death(15.0, 1.0), [4], times, n_cells=20_000, rng=rng)[:, :, 0]
    assert np.all(X[:, 0] == 4)
    for i, t in enumerate(times[1:], start=1):
        assert chisquare_gof(X[:, i], an.birth_death_pmf(80, t, 15.0, 1.0, n0=4))[2] > 1e-3


def test_telegraph_stationary_matches_closed_form(rng):
    k = (0.3, 0.5, 40.0)  # bimodal: a hard shape to get right by accident
    x = simulate_stationary(models.telegraph(*k), models.telegraph_initial_state(), 20_000, burn_in=40.0, rng=rng)
    assert np.all(x[:, 0] + x[:, 1] == 1)
    assert chisquare_gof(x[:, 2], an.telegraph_pmf(200, *k))[2] > 1e-3


def test_feedback_transient_matches_fsp(rng):
    net, x0 = models.autoregulation(), models.autoregulation_initial_state()
    times = [0.2, 0.6, 2.0]
    X = simulate(net, x0, times, n_cells=8000, rng=rng)
    fsp = FSP(net, x0, {"P": 100})
    m = fsp.marginal(fsp.transient(times)[0], "P")
    for i in range(len(times)):
        assert chisquare_gof(X[:, i, 2], m[i])[2] > 1e-3 / len(times)


def test_reproducible_with_seed():
    net = models.telegraph()
    a = simulate(net, models.telegraph_initial_state(), [1.0, 2.0], n_cells=50, rng=7)
    b = simulate(net, models.telegraph_initial_state(), [1.0, 2.0], n_cells=50, rng=7)
    np.testing.assert_array_equal(a, b)


def test_absorbing_state_terminates():
    """No reaction can fire: every observation must record the initial state."""
    X = simulate(models.birth_death(0.0, 1.0), [0], [1.0, 5.0], n_cells=10, rng=1)
    assert np.all(X == 0)


def test_invalid_times():
    with pytest.raises(ValueError):
        simulate(models.birth_death(), [0], [2.0, 1.0])


def test_binomial_capture(rng):
    counts = np.full(100_000, 50)
    kept = binomial_capture(counts, 0.2, rng)
    assert kept.mean() == pytest.approx(10.0, rel=0.01)
    assert kept.var() == pytest.approx(50 * 0.2 * 0.8, rel=0.03)
    with pytest.raises(ValueError):
        binomial_capture([1, 2], [0.5, 1.5])
