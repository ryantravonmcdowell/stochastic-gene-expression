"""Built-in validation: every method is checked against an independent one.

``run_checks()`` is what ``burstkin validate`` prints. The quick set takes
about a minute; ``full=True`` adds a confidence-interval coverage study and
the recovery of a nonlinear feedback model from time-course data.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy import stats

from . import analytic, models
from .fsp import FSP
from .fsp_inference import Snapshots, SnapshotLikelihood
from .inference import TelegraphLikelihood, fit_telegraph, profile_likelihood
from .lna import lna_stationary, lna_trajectory
from .ssa import simulate, simulate_stationary


@dataclass
class Check:
    name: str
    description: str
    value: float
    threshold: str
    passed: bool
    seconds: float = 0.0


def chisquare_gof(samples, pmf, min_expected: float = 5.0):
    """Pearson chi-square test of integer samples against ``pmf`` on ``0..len(pmf)-1``.

    Adjacent bins are pooled until each expects at least ``min_expected``
    counts; the last bin also absorbs any probability beyond the support.
    Returns ``(statistic, dof, p_value)``.
    """
    samples = np.asarray(samples, dtype=np.int64)
    pmf = np.asarray(pmf, dtype=float)
    n = samples.size
    observed = np.bincount(np.minimum(samples, pmf.size - 1), minlength=pmf.size).astype(float)
    expected = pmf * n
    expected[-1] += max(1.0 - pmf.sum(), 0.0) * n
    obs, exp = [], []
    acc_o = acc_e = 0.0
    for o, e in zip(observed, expected):
        acc_o, acc_e = acc_o + o, acc_e + e
        if acc_e >= min_expected:
            obs.append(acc_o)
            exp.append(acc_e)
            acc_o = acc_e = 0.0
    if obs:
        obs[-1] += acc_o
        exp[-1] += acc_e
    obs, exp = np.array(obs), np.array(exp)
    stat = float(((obs - exp) ** 2 / exp).sum())
    dof = obs.size - 1
    return stat, dof, float(stats.chi2.sf(stat, dof))


# ----------------------------------------------------------------- the checks
def _kummer_vs_mpmath(rng):
    import mpmath

    mpmath.mp.dps = 40
    worst = 0.0
    for _ in range(200):
        a, b, z = 10 ** rng.uniform(-3, 3), 10 ** rng.uniform(-3, 3.5), 10 ** rng.uniform(-3, 3.5)
        ref = float(mpmath.log(mpmath.hyp1f1(a, b, z)))
        worst = max(worst, abs(float(analytic.log_hyp1f1(a, b, z)) - ref) / max(1.0, abs(ref)))
    return worst


def _fast_vs_reference(rng):
    worst = 0.0
    for _ in range(25):
        k_on, k_off = 10 ** rng.uniform(-2, 2, 2)
        k_syn = 10 ** rng.uniform(0, 3.5)
        s = rng.uniform(0.05, 1.0, 200)
        n = analytic.sample_telegraph(k_on, k_off, k_syn, 200, rng, capture=s)
        ref = analytic.telegraph_logpmf_grad(n, k_on, k_off, k_syn * s)
        fast = analytic.telegraph_loglik_terms(n, k_on, k_off, k_syn * s, grad=True)
        for r, f in zip(ref, fast):
            worst = max(worst, float(np.max(np.abs(r - f) / np.maximum(1.0, np.abs(r)))))
    return worst


def _gradient_vs_fd(rng):
    s = rng.beta(30, 70, 400)
    counts = analytic.sample_telegraph(0.4, 1.5, 50.0, 400, rng, capture=s)
    lik = TelegraphLikelihood(counts, s)
    worst = 0.0
    for theta in (np.log([0.4, 1.5, 50.0]), np.log([2.0, 0.3, 20.0]), np.log([0.05, 20.0, 900.0])):
        _, g = lik.nll_and_grad(theta)
        h = 1e-5
        fd = np.array([(lik.nll(theta + h * e) - lik.nll(theta - h * e)) / (2 * h) for e in np.eye(3)])
        worst = max(worst, float(np.max(np.abs(g - fd) / np.maximum(1.0, np.abs(fd)))))
    return worst


def _quadrature_vs_series(_rng):
    worst = 0.0
    for k in ((0.7, 2.3, 30.0), (0.3, 0.2, 60.0), (4.0, 1.0, 15.0)):
        n = np.arange(150)
        worst = max(worst, float(np.abs(analytic.telegraph_pmf(149, *k) - analytic.telegraph_pmf_quadrature(n, *k, order=300)).max()))
    return worst


def _fsp_vs_closed_form(_rng):
    worst = 0.0
    for k_on, k_off, k_syn in ((0.7, 2.3, 30.0), (0.05, 0.08, 120.0), (6.0, 0.5, 40.0)):
        fsp = FSP(models.telegraph(k_on, k_off, k_syn), models.telegraph_initial_state(), {"M": 400})
        p = fsp.marginal(fsp.stationary().p, "M")
        worst = max(worst, float(np.abs(p - analytic.telegraph_pmf(400, k_on, k_off, k_syn)).max()))
    return worst


def _birth_death_transient(_rng):
    times = [0.1, 0.5, 2.0, 10.0]
    fsp = FSP(models.birth_death(40.0, 1.0), [5], {"M": 160})
    P, _ = fsp.transient(times)
    return float(max(np.abs(P[i] - analytic.birth_death_pmf(160, t, 40.0, 1.0, n0=5)).max() for i, t in enumerate(times)))


def _uniformization_vs_expm(_rng):
    fsp = FSP(models.autoregulation(), models.autoregulation_initial_state(), {"P": 100})
    times = [0.25, 1.0, 4.0]
    a, _ = fsp.transient(times, method="uniformization")
    b, _ = fsp.transient(times, method="expm")
    c, _ = fsp.transient(times, method="dense")
    return float(max(np.abs(a - b).max(), np.abs(a - c).max()))


def _lna_exact_for_linear(_rng):
    k = (0.7, 2.3, 30.0)
    net = models.telegraph(*k)
    x0 = models.telegraph_initial_state()
    res = lna_stationary(net, x0)
    exact = analytic.telegraph_moments(*k)
    worst = abs(res.variance("M") / exact.variance - 1)
    times = [0.2, 1.0, 3.0]
    mean, cov = lna_trajectory(net, x0, times)
    fsp = FSP(net, x0, {"M": 200})
    m = fsp.marginal(fsp.transient(times)[0], "M")
    n = np.arange(m.shape[1])
    for i in range(len(times)):
        mu = (n * m[i]).sum()
        var = ((n - mu) ** 2 * m[i]).sum()
        worst = max(worst, abs(mean[i, 2] / mu - 1), abs(cov[i, 2, 2] / var - 1))
    return float(worst)


def _lna_system_size(_rng):
    """Slope of log(relative error of the LNA mean) against log(system size)."""
    omegas = np.array([40.0, 80.0, 160.0, 320.0])
    errors = []
    for omega in omegas:
        net = models.annihilation(k=omega, c=1.0 / omega)
        fsp = FSP(net, [0], {"P": int(omega + 20 * np.sqrt(omega) + 40)})
        p = fsp.marginal(fsp.stationary().p, "P")
        mean = (np.arange(p.size) * p).sum()
        errors.append(abs(lna_stationary(net, [0]).mean[0] / mean - 1))
    return float(np.polyfit(np.log(omegas), np.log(errors), 1)[0])


def _ssa_stationary(rng):
    k = (0.7, 2.3, 30.0)
    x = simulate_stationary(models.telegraph(*k), models.telegraph_initial_state(), 20000, burn_in=30.0, rng=rng)
    return chisquare_gof(x[:, 2], analytic.telegraph_pmf(300, *k))[2]


def _ssa_transient(rng):
    net, x0 = models.autoregulation(), models.autoregulation_initial_state()
    times = [0.25, 0.5, 1.0, 4.0]
    X = simulate(net, x0, times, n_cells=4000, rng=rng)
    fsp = FSP(net, x0, {"P": 100})
    m = fsp.marginal(fsp.transient(times)[0], "P")
    p_values = [chisquare_gof(X[:, i, 2], m[i])[2] for i in range(len(times))]
    return float(min(1.0, min(p_values) * len(times)))  # Bonferroni


def _capture_thinning(_rng):
    k, s = (0.7, 2.3, 30.0), 0.3
    fsp = FSP(models.telegraph(*k), models.telegraph_initial_state(), {"M": 200})
    true_pmf = fsp.marginal(fsp.stationary().p, "M")
    m = np.arange(true_pmf.size)
    thinned = np.array([(true_pmf * stats.binom.pmf(n, m, s)).sum() for n in range(80)])
    return float(np.abs(thinned - analytic.telegraph_pmf(79, k[0], k[1], k[2] * s)).max())


def _capture_scale(rng):
    s = rng.beta(30, 70, 800)
    counts = analytic.sample_telegraph(0.5, 2.0, 40.0, 800, rng, capture=s)
    a = fit_telegraph(counts, s)
    b = fit_telegraph(counts, 0.5 * s)
    return float(max(abs(b.k_on / a.k_on - 1), abs(b.k_off / a.k_off - 1), abs(b.k_syn / (2 * a.k_syn) - 1)))


def _moments_roundtrip(_rng):
    worst = 0.0
    for k in ((0.3, 2.0, 50.0), (5.0, 0.5, 20.0), (0.05, 0.02, 300.0)):
        m = [analytic.telegraph_factorial_moment(o, *k) for o in (1, 2, 3)]
        est = analytic.telegraph_from_factorial_moments(*m)
        worst = max(worst, max(abs(e / t - 1) for e, t in zip(est, k)))
    return float(worst)


def _mle_large_sample(rng):
    k = (0.4, 1.5, 40.0)
    counts = analytic.sample_telegraph(*k, 20000, rng)
    fit = fit_telegraph(counts)
    return float(max(abs(fit.burst_frequency / k[0] - 1), abs(fit.burst_size / (k[2] / k[1]) - 1)))


def coverage_study(replicates: int = 200, n_cells: int = 500, seed: int = 11, k=(0.5, 2.0, 40.0)) -> dict:
    """Fraction of 95% profile-likelihood intervals that contain the truth."""
    rng = np.random.default_rng(seed)
    hits = {"k_on": 0, "burst_size": 0}
    truth = {"k_on": k[0], "burst_size": k[2] / k[1]}
    for _ in range(replicates):
        counts = analytic.sample_telegraph(*k, n_cells, rng)
        lik = TelegraphLikelihood(counts)
        fit = fit_telegraph(counts, likelihood=lik)
        for target in hits:
            lo, hi = profile_likelihood(lik, fit, target).ci
            lo = 0.0 if lo is None else lo
            hi = np.inf if hi is None else hi
            hits[target] += lo <= truth[target] <= hi
    return {t: h / replicates for t, h in hits.items()}


def _coverage(_rng):
    cov = coverage_study()
    return float(min(cov.values()))


def _feedback_recovery(rng):
    net, x0 = models.autoregulation(), models.autoregulation_initial_state()
    times = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    X = simulate(net, x0, times, n_cells=300, rng=rng)
    free = ("rho_u", "sigma_b", "sigma_u")
    lik = SnapshotLikelihood(net, x0, "P", Snapshots(times, [X[:, i, 2] for i in range(times.size)]), free, {"P": 100})
    fit = lik.fit([{"rho_u": 30.0, "sigma_b": 1.0, "sigma_u": 1.0}])
    return float(max(abs(np.log(fit.params[p] / net.parameters[p])) / fit.standard_errors[p] for p in free))


QUICK = [
    ("kummer_series_vs_mpmath", "log 1F1 by the positive Kummer series vs 40-digit mpmath (200 random a, b, z over 6 decades)", _kummer_vs_mpmath, "lt", 1e-12),
    ("fast_likelihood_vs_reference", "shared-switching likelihood and gradient vs the general series", _fast_vs_reference, "lt", 1e-9),
    ("gradient_vs_finite_differences", "analytic gradient of the telegraph likelihood vs central differences", _gradient_vs_fd, "lt", 1e-5),
    ("series_vs_gauss_jacobi", "stationary pmf: Kummer series vs Gauss-Jacobi quadrature over the Beta mixing law", _quadrature_vs_series, "lt", 1e-10),
    ("fsp_stationary_vs_closed_form", "master equation (FSP) vs closed-form telegraph distribution, three regimes", _fsp_vs_closed_form, "lt", 1e-12),
    ("fsp_transient_vs_exact", "FSP transient vs exact birth-death distribution (binomial + Poisson)", _birth_death_transient, "lt", 1e-12),
    ("uniformization_vs_expm", "three transient solvers agree on the nonlinear feedback model", _uniformization_vs_expm, "lt", 1e-12),
    ("lna_exact_for_linear_network", "LNA mean/variance vs exact moments (stationary and transient) for the telegraph model", _lna_exact_for_linear, "lt", 1e-8),
    ("lna_error_order", "LNA error vs system size for 2P -> 0: log-log slope should be -1", _lna_system_size, "near-1", 0.02),
    ("ssa_stationary_gof", "SSA (20,000 cells) vs closed-form distribution: chi-square p-value", _ssa_stationary, "gt", 1e-3),
    ("ssa_transient_gof", "SSA (4,000 cells) vs FSP for the feedback model at 4 times: Bonferroni p-value", _ssa_transient, "gt", 1e-3),
    ("capture_thinning_identity", "binomial capture of the FSP distribution vs Poisson-Beta with k_syn * s", _capture_thinning, "lt", 1e-12),
    ("capture_scale_invariance", "halving every capture efficiency doubles k_syn and leaves k_on, k_off unchanged", _capture_scale, "lt", 1e-4),
    ("moment_inversion_roundtrip", "exact factorial moments -> method-of-moments rates", _moments_roundtrip, "lt", 1e-10),
    ("mle_large_sample", "MLE from 20,000 cells: max relative error of burst frequency and size", _mle_large_sample, "lt", 0.06),
]

# exact 99.9% binomial acceptance region for 200 replicates at the nominal 95%
_COVERAGE_BAND = tuple(float(v) / 200 for v in stats.binom.interval(0.999, 200, 0.95))

FULL = [
    ("profile_ci_coverage", "coverage of 95% profile-likelihood intervals (200 replicates x 500 cells), worst of burst frequency and size", _coverage, "between", _COVERAGE_BAND),
    ("feedback_timecourse_recovery", "feedback model fitted to SSA snapshots (300 cells x 6 times): max |z| of 3 rates", _feedback_recovery, "lt", 3.5),
]


def _passes(kind, value, limit) -> tuple:
    if kind == "lt":
        return value < limit, f"< {limit:g}"
    if kind == "gt":
        return value > limit, f"> {limit:g}"
    if kind == "near-1":
        return abs(value + 1.0) < limit, f"-1 +/- {limit:g}"
    if kind == "between":
        return limit[0] <= value <= limit[1], f"in [{limit[0]:g}, {limit[1]:g}]"
    raise ValueError(kind)


def run_checks(full: bool = False, seed: int = 2026, progress=None) -> list:
    """Run the validation checks; ``progress(check)`` is called after each one."""
    results = []
    for name, description, fn, kind, limit in QUICK + (FULL if full else []):
        rng = np.random.default_rng([seed, len(results)])
        start = time.perf_counter()
        try:
            value = fn(rng)
        except ImportError as exc:  # mpmath is optional
            results.append(Check(name, f"{description} (skipped: {exc.name} not installed)", np.nan, "-", True, 0.0))
            continue
        ok, threshold = _passes(kind, value, limit)
        results.append(Check(name, description, value, threshold, bool(ok), time.perf_counter() - start))
        if progress:
            progress(results[-1])
    return results


def format_table(checks: list) -> str:
    width = max(len(c.name) for c in checks)
    lines = [f"{'check':<{width}}  {'value':>11}  {'required':<16} {'result':<6}  time"]
    for c in checks:
        status = "PASS" if c.passed else "FAIL"
        lines.append(f"{c.name:<{width}}  {c.value:>11.3g}  {c.threshold:<16} {status:<6}  {c.seconds:5.1f}s")
    n_fail = sum(not c.passed for c in checks)
    lines.append(f"{len(checks) - n_fail}/{len(checks)} checks passed")
    return "\n".join(lines)
