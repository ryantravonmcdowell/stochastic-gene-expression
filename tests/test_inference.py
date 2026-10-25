import numpy as np
import pytest
from scipy import stats

from burstkin import analytic as an
from burstkin.inference import (
    TelegraphLikelihood,
    classify_identifiability,
    compare_models,
    fit_nbinom,
    fit_telegraph,
    profile_likelihood,
)


def test_gradient_matches_finite_differences(rng):
    s = rng.beta(30, 70, 300)
    counts = an.sample_telegraph(0.4, 1.5, 50.0, 300, rng, capture=s)
    for capture in (None, s):
        lik = TelegraphLikelihood(counts, capture)
        for theta in (np.log([0.4, 1.5, 50.0]), np.log([3.0, 0.2, 10.0]), np.log([0.02, 40.0, 1500.0])):
            _, g = lik.nll_and_grad(theta)
            h = 1e-5
            fd = [(lik.nll(theta + h * e) - lik.nll(theta - h * e)) / (2 * h) for e in np.eye(3)]
            np.testing.assert_allclose(g, fd, rtol=1e-6, atol=1e-6)


def test_compressed_and_per_cell_likelihoods_agree(rng):
    counts = an.sample_telegraph(0.5, 2.0, 30.0, 500, rng)
    a = TelegraphLikelihood(counts).loglik(0.5, 2.0, 30.0)
    b = TelegraphLikelihood(counts, np.ones(counts.size)).loglik(0.5, 2.0, 30.0)
    assert a == pytest.approx(b, rel=1e-12)
    assert a == pytest.approx(an.telegraph_logpmf(counts, 0.5, 2.0, 30.0).sum(), rel=1e-10)


def test_large_sample_recovery(rng):
    k = (0.4, 1.5, 40.0)
    fit = fit_telegraph(an.sample_telegraph(*k, 20_000, rng))
    assert fit.converged and not fit.at_bound
    assert fit.burst_frequency == pytest.approx(k[0], rel=0.08)
    assert fit.burst_size == pytest.approx(k[2] / k[1], rel=0.08)


def test_profile_interval_ends_sit_on_the_threshold(rng):
    s = rng.beta(30, 70, 800)
    counts = an.sample_telegraph(0.5, 2.0, 60.0, 800, rng, capture=s)
    lik = TelegraphLikelihood(counts, s)
    fit = fit_telegraph(counts, s, likelihood=lik)
    prof = profile_likelihood(lik, fit, "k_on")
    lo, hi = prof.ci
    assert lo < fit.k_on < hi
    assert not prof.mle_improved
    from burstkin.inference import _ProfileProblem, DEFAULT_BOUNDS

    problem = _ProfileProblem(lik, "k_on", DEFAULT_BOUNDS)
    threshold = stats.chi2.ppf(0.95, 1)
    for end in (lo, hi):
        f, _ = problem.minimize(np.log(end), problem.free(fit.theta))
        assert 2 * (f + fit.loglik) == pytest.approx(threshold, abs=0.02)


def test_identifiable_and_bursty_regimes(rng):
    slow = an.sample_telegraph(0.2, 0.3, 80.0, 1000, rng)  # slow switching: everything identifiable
    lik = TelegraphLikelihood(slow)
    fit = fit_telegraph(slow, likelihood=lik)
    profs = {t: profile_likelihood(lik, fit, t) for t in ("k_on", "k_off", "k_syn", "burst_size")}
    assert classify_identifiability(profs) == "full"

    bursty = an.sample_telegraph(0.3, 30.0, 600.0, 1000, rng)  # short bursts: only frequency and size
    lik = TelegraphLikelihood(bursty)
    fit = fit_telegraph(bursty, likelihood=lik)
    profs = {t: profile_likelihood(lik, fit, t) for t in ("k_on", "k_off", "k_syn", "burst_size")}
    assert classify_identifiability(profs) == "burst"
    assert profs["burst_size"].closed and profs["k_on"].closed
    lo, hi = profs["burst_size"].ci
    assert lo < 20.0 < hi
    # where k_off or k_syn runs away, the other rate ends up pinned at its bound
    assert profs["k_off"].nuisance_at_bound.any() or profs["k_syn"].nuisance_at_bound.any()


def test_profile_grid_points(rng):
    counts = an.sample_telegraph(0.5, 1.0, 30.0, 400, rng)
    lik = TelegraphLikelihood(counts)
    fit = fit_telegraph(counts, likelihood=lik)
    coarse = profile_likelihood(lik, fit, "burst_size")
    fine = profile_likelihood(lik, fit, "burst_size", grid_points=12)
    assert fine.values.size == coarse.values.size + 12 - 2  # the grid's two end points already exist
    assert fine.ci == pytest.approx(coarse.ci)
    assert np.all(fine.deviance > -1e-6)
    assert fine.nuisance_at_bound.shape == fine.values.shape


def test_model_selection(rng):
    poisson = rng.poisson(12.0, 2000)
    assert compare_models(poisson).best == "poisson"
    nb = an.sample_telegraph(0.8, 200.0, 2000.0, 2000, rng)  # deep bursty limit
    assert compare_models(nb).best == "nbinom"
    bimodal = an.sample_telegraph(0.1, 0.1, 60.0, 2000, rng)
    cmp = compare_models(bimodal)
    assert cmp.best == "telegraph"
    assert cmp.lrt_pvalue["telegraph_vs_nbinom"] < 1e-6


def test_capture_scale_only_rescales_synthesis(rng):
    s = rng.beta(30, 70, 800)
    counts = an.sample_telegraph(0.5, 2.0, 40.0, 800, rng, capture=s)
    a = fit_telegraph(counts, s)
    b = fit_telegraph(counts, 0.5 * s)
    assert b.k_on == pytest.approx(a.k_on, rel=1e-4)
    assert b.k_off == pytest.approx(a.k_off, rel=1e-4)
    assert b.k_syn == pytest.approx(2 * a.k_syn, rel=1e-4)
    assert b.loglik == pytest.approx(a.loglik, abs=1e-6)


def test_nbinom_fit_matches_moments(rng):
    r, mean = 1.5, 20.0
    counts = rng.negative_binomial(r, r / (r + mean), 50_000)
    mu, size, _ = fit_nbinom(counts)
    assert mu == pytest.approx(mean, rel=0.03)
    assert size == pytest.approx(r, rel=0.05)


def test_input_validation():
    with pytest.raises(ValueError):
        TelegraphLikelihood([1, -2, 3])
    with pytest.raises(ValueError):
        TelegraphLikelihood([1, 2, 3], capture=[0.5, 0.5])
    with pytest.raises(ValueError):
        TelegraphLikelihood([1, 2, 3], capture=[0.5, 0.0, 0.5])
