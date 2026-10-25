import numpy as np
import pytest
from scipy import stats

from burstkin import analytic as an

REGIMES = [
    (0.7, 2.3, 30.0),  # bursty, unimodal
    (0.05, 0.08, 120.0),  # slow switching, bimodal
    (6.0, 0.5, 40.0),  # mostly on, near Poisson
    (0.01, 50.0, 2000.0),  # rare large bursts
]


def test_log_hyp1f1_matches_mpmath(rng):
    mpmath = pytest.importorskip("mpmath")
    mpmath.mp.dps = 40
    for _ in range(150):
        a, b, z = 10 ** rng.uniform(-3, 3), 10 ** rng.uniform(-3, 3.5), 10 ** rng.uniform(-3, 3.5)
        ref = float(mpmath.log(mpmath.hyp1f1(a, b, z)))
        assert an.log_hyp1f1(a, b, z) == pytest.approx(ref, rel=1e-12, abs=1e-12)


def test_log_hyp1f1_derivatives(rng):
    for _ in range(20):
        a, b, z = 10 ** rng.uniform(-1, 2), 10 ** rng.uniform(-1, 2.5), 10 ** rng.uniform(-1, 2.5)
        _, ga, gb, gz = an.log_hyp1f1(a, b, z, grad=True)
        h = 1e-6
        for g, d in ((ga, (h, 0, 0)), (gb, (0, h, 0)), (gz, (0, 0, h))):
            fd = (an.log_hyp1f1(a + d[0], b + d[1], z + d[2]) - an.log_hyp1f1(a - d[0], b - d[1], z - d[2])) / (2 * h)
            assert g == pytest.approx(fd, rel=1e-6, abs=1e-8)


def test_log_hyp1f1_at_zero_and_domain():
    f, ga, gb, gz = an.log_hyp1f1(0.4, 3.0, 0.0, grad=True)
    assert (f, ga, gb) == (0.0, 0.0, 0.0)
    assert gz == pytest.approx(0.4 / 3.0)
    with pytest.raises(ValueError):
        an.log_hyp1f1(-1.0, 2.0, 1.0)


@pytest.mark.parametrize("k", REGIMES)
def test_pmf_normalised_with_exact_moments(k):
    p = an.telegraph_pmf(4000, *k)
    n = np.arange(p.size)
    m = an.telegraph_moments(*k)
    # log P carries -k_syn + log 1F1(..., k_syn): the cancellation limits absolute
    # accuracy to roughly 10 * k_syn * machine epsilon (about 4e-12 at k_syn = 2000)
    assert p.sum() == pytest.approx(1.0, abs=max(1e-13, 10 * k[2] * np.finfo(float).eps))
    assert (n * p).sum() == pytest.approx(m.mean, rel=1e-10)
    assert ((n - m.mean) ** 2 * p).sum() == pytest.approx(m.variance, rel=1e-9)
    for order in (1, 2, 3):
        falling = np.prod([n - j for j in range(order)], axis=0)
        assert (falling * p).sum() == pytest.approx(an.telegraph_factorial_moment(order, *k), rel=1e-9)


@pytest.mark.parametrize("k", REGIMES[:3])
def test_series_matches_gauss_jacobi_quadrature(k):
    n = np.arange(200)
    np.testing.assert_allclose(an.telegraph_pmf(199, *k), an.telegraph_pmf_quadrature(n, *k, order=300), atol=1e-10)


def test_fast_path_matches_reference(rng):
    for _ in range(20):
        k_on, k_off = 10 ** rng.uniform(-2, 2, 2)
        k_syn = 10 ** rng.uniform(-1, 3.7)
        s = rng.uniform(0.05, 1.0, 150)
        n = an.sample_telegraph(k_on, k_off, k_syn, 150, rng, capture=s)
        ref = an.telegraph_logpmf_grad(n, k_on, k_off, k_syn * s)
        fast = an.telegraph_loglik_terms(n, k_on, k_off, k_syn * s, grad=True)
        for r, f in zip(ref, fast):
            np.testing.assert_allclose(f, r, rtol=1e-9, atol=1e-9)


def test_zero_rate_cells():
    logp, d_on, d_off, d_rate = an.telegraph_loglik_terms(np.array([0, 3]), 0.5, 1.5, np.array([0.0, 0.0]), grad=True)
    assert logp[0] == 0.0 and logp[1] == -np.inf
    assert d_rate[0] == pytest.approx(-0.5 / 2.0)  # -E[p] for p ~ Beta(0.5, 1.5)


def test_bursty_limit_is_negative_binomial():
    """With k_off -> inf at fixed burst size the error falls like 1/k_off."""
    k_on, burst = 0.4, 12.0
    n = np.arange(400)
    nb = np.exp(an.bursty_limit_logpmf(n, k_on, burst))
    errors = [np.abs(an.telegraph_pmf(399, k_on, k_off, burst * k_off) - nb).max() for k_off in (1e1, 1e2, 1e3)]
    assert errors[1] < errors[0] / 5 and errors[2] < errors[1] / 5
    assert errors[2] < 1e-3


def test_constitutive_limit_is_poisson():
    n = np.arange(100)
    p = an.telegraph_pmf(99, 1e4, 1e-2, 20.0)
    np.testing.assert_allclose(p, stats.poisson.pmf(n, 20.0), atol=1e-4)


def test_moment_inversion():
    for k in REGIMES:
        m = [an.telegraph_factorial_moment(o, *k) for o in (1, 2, 3)]
        np.testing.assert_allclose(an.telegraph_from_factorial_moments(*m), k, rtol=1e-8)
    lam = 7.0  # Poisson moments: the telegraph rates are not defined
    assert an.telegraph_from_factorial_moments(lam, lam**2, lam**3) is None


def test_moment_estimator_with_capture(rng):
    k = (0.5, 2.0, 60.0)
    s = rng.beta(30, 70, 200_000)
    counts = an.sample_telegraph(*k, s.size, rng, capture=s)
    est = an.telegraph_from_moments(counts, s)
    np.testing.assert_allclose(est, k, rtol=0.15)


def test_sampler_matches_pmf(rng):
    from burstkin.validate import chisquare_gof

    k = (0.3, 0.4, 50.0)
    x = an.sample_telegraph(*k, 30_000, rng)
    assert chisquare_gof(x, an.telegraph_pmf(200, *k))[2] > 1e-3


def test_birth_death_pmf():
    p = an.birth_death_pmf(200, 0.7, 30.0, 1.0, n0=12)
    n = np.arange(p.size)
    assert p.sum() == pytest.approx(1.0, abs=1e-12)
    assert (n * p).sum() == pytest.approx(12 * np.exp(-0.7) + 30 * (1 - np.exp(-0.7)), rel=1e-10)


def test_nbinom_and_poisson_match_scipy():
    n = np.arange(60)
    mean, size = 8.0, 1.7
    np.testing.assert_allclose(an.nbinom_logpmf(n, mean, size), stats.nbinom.logpmf(n, size, size / (size + mean)), rtol=1e-12)
    np.testing.assert_allclose(an.poisson_logpmf(n, mean), stats.poisson.logpmf(n, mean), rtol=1e-12)


def test_invalid_counts():
    with pytest.raises(ValueError):
        an.telegraph_logpmf([-1], 1.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        an.telegraph_logpmf([1.5], 1.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        an.telegraph_logpmf([1], 0.0, 1.0, 1.0)
