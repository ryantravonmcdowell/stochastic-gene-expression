import numpy as np
import pytest

from burstkin import analytic as an
from burstkin import models
from burstkin.fsp import FSP, conservation_bound, stationary_marginal


def test_state_space_uses_conservation_bounds():
    net = models.telegraph()
    assert conservation_bound(net, models.telegraph_initial_state(), "G_on") == 1
    with pytest.raises(ValueError):
        conservation_bound(net, models.telegraph_initial_state(), "M")
    fsp = FSP(net, models.telegraph_initial_state(), {"M": 50})
    assert fsp.n_states == 2 * 51
    assert np.all(fsp.states[:, 0] + fsp.states[:, 1] == 1)
    assert fsp.index_of([0, 1, 51]) == -1
    assert fsp.index_of([1, 1, 3]) == -1  # inside the box but unreachable


def test_generator_conservation():
    fsp = FSP(models.autoregulation(), models.autoregulation_initial_state(), {"P": 40})
    reflect = fsp.generator(mode="reflect")
    sink = fsp.generator(mode="sink")
    np.testing.assert_allclose(np.asarray(reflect.sum(axis=0)).ravel(), 0.0, atol=1e-12)
    leak = -np.asarray(sink.sum(axis=0)).ravel()
    assert np.all(leak >= -1e-12)
    assert np.all(leak[~fsp.edge_states()] < 1e-12)  # only edge states leak
    assert leak[fsp.edge_states()].min() > 0


@pytest.mark.parametrize("k", [(0.7, 2.3, 30.0), (0.05, 0.08, 120.0), (6.0, 0.5, 40.0)])
def test_stationary_matches_closed_form(k):
    fsp = FSP(models.telegraph(*k), models.telegraph_initial_state(), {"M": 400})
    res = fsp.stationary()
    p = fsp.marginal(res.p, "M")
    np.testing.assert_allclose(p, an.telegraph_pmf(400, *k), atol=1e-12)
    assert res.boundary_mass < 1e-30
    assert res.residual < 1e-10


def test_stationary_when_initial_state_is_improbable():
    """Regression: anchoring on a state with probability ~e^-300 must not break the solve."""
    omega = 300.0
    net = models.annihilation(k=omega, c=1.0 / omega)
    fsp = FSP(net, [0], {"P": 700})
    p = fsp.marginal(fsp.stationary().p, "P")
    mean = (np.arange(p.size) * p).sum()
    assert mean == pytest.approx(omega + 0.125, abs=2e-3)  # exact mean is Omega + 1/8 + O(1/Omega)


@pytest.mark.parametrize("method", ["uniformization", "dense", "expm", "auto"])
def test_transient_matches_exact_birth_death(method):
    times = [0.0, 0.1, 0.5, 2.0, 10.0]
    fsp = FSP(models.birth_death(40.0, 1.0), [5], {"M": 160})
    P, err = fsp.transient(times, method=method)
    for i, t in enumerate(times):
        np.testing.assert_allclose(P[i], an.birth_death_pmf(160, t, 40.0, 1.0, n0=5), atol=1e-12)
    assert np.all(np.abs(err) < 1e-10)


def test_sink_truncation_is_a_lower_bound():
    """FSP theorem: truncated probabilities never exceed the exact ones, and the lost mass bounds the error."""
    fsp = FSP(models.birth_death(25.0, 1.0), [0], {"M": 28})
    times = [0.5, 1.0, 3.0]
    P, err = fsp.transient(times)
    for i, t in enumerate(times):
        exact = an.birth_death_pmf(400, t, 25.0, 1.0)
        assert np.all(P[i] <= exact[:29] + 1e-15)
        l1 = np.abs(exact[:29] - P[i]).sum() + exact[29:].sum()
        assert l1 <= err[i] * (1 + 1e-9) + 1e-14
    assert err[-1] > 0.1  # by t = 3 the bound genuinely truncates (mean 23.8, bound 28)


def test_transient_methods_agree_on_nonlinear_model():
    fsp = FSP(models.autoregulation(), models.autoregulation_initial_state(), {"P": 100})
    times = [0.25, 1.0, 4.0]
    ref, _ = fsp.transient(times, method="expm")
    for method in ("uniformization", "dense"):
        np.testing.assert_allclose(fsp.transient(times, method=method)[0], ref, atol=1e-12)


def test_marginal_of_stack_and_stationary_marginal_helper():
    net = models.telegraph(0.5, 1.0, 20.0)
    pmf, boundary = stationary_marginal(net, models.telegraph_initial_state(), "M", {"M": 10}, tol=1e-12)
    assert pmf.size > 11  # the bound was grown
    np.testing.assert_allclose(pmf, an.telegraph_pmf(pmf.size - 1, 0.5, 1.0, 20.0), atol=1e-11)
    assert boundary < 1e-12
    fsp = FSP(net, models.telegraph_initial_state(), {"M": 60})
    P, _ = fsp.transient([0.5, 1.0])
    assert fsp.marginal(P, "M").shape == (2, 61)


def test_bounds_must_cover_initial_state():
    with pytest.raises(ValueError):
        FSP(models.birth_death(), [10], {"M": 5})
