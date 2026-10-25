import numpy as np
import pytest

from burstkin import analytic as an
from burstkin import models
from burstkin.fsp import FSP
from burstkin.lna import conservation_laws, lna_stationary, lna_trajectory, rre_steady_state


def test_conservation_laws_of_telegraph():
    W, L = conservation_laws(models.telegraph())
    assert W.shape == (3, 2) and L.shape == (1, 3)
    np.testing.assert_allclose(np.abs(L[0]), [2**-0.5, 2**-0.5, 0.0], atol=1e-12)
    np.testing.assert_allclose(W.T @ L.T, 0.0, atol=1e-12)


def test_lna_is_exact_for_the_telegraph_model():
    k = (0.7, 2.3, 30.0)
    res = lna_stationary(models.telegraph(*k), models.telegraph_initial_state())
    m = an.telegraph_moments(*k)
    assert res.mean[2] == pytest.approx(m.mean, rel=1e-10)
    assert res.variance("M") == pytest.approx(m.variance, rel=1e-9)
    assert res.variance("G_on") == pytest.approx(m.p_on * (1 - m.p_on), rel=1e-9)
    assert res.fano("M") == pytest.approx(m.fano, rel=1e-9)


def test_transient_lna_matches_master_equation_for_linear_network():
    net, x0 = models.telegraph(0.7, 2.3, 30.0), models.telegraph_initial_state()
    times = [0.0, 0.2, 1.0, 3.0]
    mean, cov = lna_trajectory(net, x0, times)
    fsp = FSP(net, x0, {"M": 200})
    m = fsp.marginal(fsp.transient(times)[0], "M")
    n = np.arange(m.shape[1])
    for i in range(len(times)):
        mu = (n * m[i]).sum()
        assert mean[i, 2] == pytest.approx(mu, rel=1e-8, abs=1e-12)
        assert cov[i, 2, 2] == pytest.approx(((n - mu) ** 2 * m[i]).sum(), rel=1e-8, abs=1e-12)


def test_annihilation_fano_and_error_order():
    errors = []
    omegas = np.array([40.0, 80.0, 160.0, 320.0])
    for omega in omegas:
        net = models.annihilation(k=omega, c=1.0 / omega)
        res = lna_stationary(net, [0])
        assert res.fano("P") == pytest.approx(0.75, abs=1e-10)
        fsp = FSP(net, [0], {"P": int(omega + 20 * np.sqrt(omega) + 40)})
        p = fsp.marginal(fsp.stationary().p, "P")
        mean = (np.arange(p.size) * p).sum()
        errors.append(mean - res.mean[0])
    # the first correction to the mean is the constant 1/8, so the relative error falls as 1/Omega
    np.testing.assert_allclose(errors, 0.125, atol=2e-3)
    slope = np.polyfit(np.log(omegas), np.log(np.array(errors) / omegas), 1)[0]
    assert slope == pytest.approx(-1.0, abs=0.02)


def test_rre_fixed_point_of_feedback_model():
    net = models.autoregulation()
    phi = rre_steady_state(net, models.autoregulation_initial_state())
    assert phi[0] + phi[1] == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(net.drift(phi), 0.0, atol=1e-9)
    assert np.all(phi >= 0)
