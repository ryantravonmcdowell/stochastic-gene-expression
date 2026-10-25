"""Maximum-likelihood inference of transcriptional burst kinetics.

For a gene measured in many cells, cell ``i`` reports a count drawn from the
telegraph model's stationary distribution with transcription rate
``k_syn * s_i``, where ``s_i`` is the probability that a transcript in that
cell is captured. Rates are in units of the mRNA decay rate.

The log-likelihood and its exact gradient come from
:func:`burstkin.analytic.telegraph_logpmf_grad`. Uncertainty is quantified by
profile likelihood, which also diagnoses identifiability: in the bursty
regime (short active periods) the data pin down burst frequency ``k_on`` and
burst size ``k_syn / k_off`` but not ``k_off`` and ``k_syn`` separately, and
the profiles of those two run flat into the parameter bounds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, stats

from .analytic import nbinom_logpmf, poisson_logpmf, telegraph_from_moments, telegraph_loglik_terms

PARAMS = ("k_on", "k_off", "k_syn")
DEFAULT_BOUNDS = {"k_on": (1e-3, 1e3), "k_off": (1e-3, 1e3), "k_syn": (1e-2, 1e4)}
PROFILE_TARGETS = ("k_on", "k_off", "k_syn", "burst_size")


def _validate_counts(counts, capture):
    counts = np.asarray(counts)
    if counts.ndim != 1 or counts.size == 0:
        raise ValueError("counts must be a non-empty 1-d array")
    if np.any(counts < 0) or np.any(counts != np.round(counts)):
        raise ValueError("counts must be non-negative integers")
    counts = counts.astype(np.int64)
    if capture is not None:
        capture = np.asarray(capture, dtype=float)
        if capture.shape != counts.shape:
            raise ValueError("capture must have one efficiency per cell")
        if np.any(capture <= 0) or np.any(capture > 1):
            raise ValueError("capture efficiencies must lie in (0, 1]")
    return counts, capture


def _log_bounds(bounds):
    """Lower and upper bounds of ``log(k_on, k_off, k_syn)``."""
    arr = np.log(np.array([bounds[p] for p in PARAMS], dtype=float))
    return arr[:, 0], arr[:, 1]


# ------------------------------------------------------------------ likelihood
class TelegraphLikelihood:
    """Telegraph-model likelihood of one gene's counts across cells."""

    def __init__(self, counts, capture=None) -> None:
        counts, capture = _validate_counts(counts, capture)
        self.counts, self.capture = counts, capture
        self.n_cells = counts.size
        if capture is None:  # cells are exchangeable: sum over distinct counts
            values, weights = np.unique(counts, return_counts=True)
            self._n, self._w, self._s = values.astype(float), weights.astype(float), np.ones(values.size)
        else:
            self._n, self._w, self._s = counts.astype(float), np.ones(counts.size), capture
        self.evaluations = 0

    def loglik(self, k_on: float, k_off: float, k_syn: float) -> float:
        return float(self._w @ telegraph_loglik_terms(self._n, k_on, k_off, k_syn * self._s))

    def nll_and_grad(self, theta):
        """Negative log-likelihood and its gradient in ``theta = log(k_on, k_off, k_syn)``."""
        k_on, k_off, k_syn = np.exp(theta)
        logp, d_on, d_off, d_rate = telegraph_loglik_terms(self._n, k_on, k_off, k_syn * self._s, grad=True)
        self.evaluations += 1
        w = self._w
        nll = -float(w @ logp)
        grad = -np.array([k_on * (w @ d_on), k_off * (w @ d_off), k_syn * (w @ (d_rate * self._s))])
        return nll, grad

    def nll(self, theta) -> float:
        return self.nll_and_grad(theta)[0]


# -------------------------------------------------------------------- fitting
@dataclass
class TelegraphFit:
    k_on: float
    k_off: float
    k_syn: float
    loglik: float
    n_cells: int
    converged: bool
    at_bound: tuple = ()
    start: str = ""

    @property
    def burst_frequency(self) -> float:
        return self.k_on

    @property
    def burst_size(self) -> float:
        return self.k_syn / self.k_off

    @property
    def theta(self) -> np.ndarray:
        return np.log([self.k_on, self.k_off, self.k_syn])

    def as_dict(self) -> dict:
        return {
            "k_on": self.k_on,
            "k_off": self.k_off,
            "k_syn": self.k_syn,
            "burst_frequency": self.burst_frequency,
            "burst_size": self.burst_size,
            "loglik": self.loglik,
            "converged": self.converged,
            "at_bound": ";".join(self.at_bound),
        }


def starting_points(counts, capture=None, bounds=None) -> list:
    """Labelled starting points ``(label, theta)`` for the optimiser.

    Method-of-moments (when admissible), a negative-binomial (bursty) guess,
    and a small grid over switching rates with ``k_syn`` set to match the mean.
    """
    bounds = bounds or DEFAULT_BOUNDS
    lo, hi = _log_bounds(bounds)
    counts = np.asarray(counts, dtype=float)
    s = np.ones_like(counts) if capture is None else np.asarray(capture, dtype=float)
    mean_obs = max(counts.mean(), 1e-3)
    mean_true = mean_obs / s.mean()
    starts = []
    mom = telegraph_from_moments(counts, capture)
    if mom is not None:
        starts.append(("moments", np.log(mom)))
    var = counts.var()
    if var > mean_obs:  # bursty guess: NB size = k_on, NB mean / size = burst size
        k_on = mean_obs**2 / (var - mean_obs)
        burst = mean_true / k_on
        starts.append(("bursty", np.log([k_on, 20.0, 20.0 * burst])))
    for k_on, k_off in ((0.1, 0.1), (0.2, 3.0), (1.0, 1.0), (3.0, 0.3), (1.0, 10.0)):
        starts.append((f"grid({k_on:g},{k_off:g})", np.log([k_on, k_off, mean_true * (k_on + k_off) / k_on])))
    return [(label, np.clip(theta, lo, hi)) for label, theta in starts]


def _minimize(fun, x0, lo, hi, maxiter=1000, ftol=1e-13, gtol=1e-8):
    return optimize.minimize(
        fun,
        x0,
        jac=True,
        method="L-BFGS-B",
        bounds=list(zip(lo, hi)),
        options={"maxiter": maxiter, "ftol": ftol, "gtol": gtol, "maxcor": 20},
    )


def fit_telegraph(counts, capture=None, bounds=None, likelihood: TelegraphLikelihood | None = None) -> TelegraphFit:
    """Maximum-likelihood ``(k_on, k_off, k_syn)`` from multiple starting points.

    Every start gets a short exploratory run; the two best are then optimised
    to full precision. In testing this reached the same optimum as converging
    every start, at a fraction of the cost. Profile likelihoods flag the rare
    case where a better optimum turns up (``Profile.mle_improved``).
    """
    bounds = bounds or DEFAULT_BOUNDS
    lik = likelihood or TelegraphLikelihood(counts, capture)
    lo, hi = _log_bounds(bounds)
    scouts = []
    for label, theta0 in starting_points(lik.counts, lik.capture, bounds):
        res = _minimize(lik.nll_and_grad, theta0, lo, hi, maxiter=25, ftol=1e-9, gtol=1e-5)
        scouts.append((res.fun, label, res.x))
    scouts.sort(key=lambda s: s[0])
    best, best_label = None, ""
    for _, label, theta0 in scouts[:2]:
        res = _minimize(lik.nll_and_grad, theta0, lo, hi)
        if best is None or res.fun < best.fun - 1e-9:
            best, best_label = res, label
    theta = best.x
    tol = 1e-6
    at_bound = tuple(
        f"{p}={'lower' if theta[i] <= lo[i] + tol else 'upper'}"
        for i, p in enumerate(PARAMS)
        if theta[i] <= lo[i] + tol or theta[i] >= hi[i] - tol
    )
    k_on, k_off, k_syn = np.exp(theta)
    return TelegraphFit(
        k_on=float(k_on),
        k_off=float(k_off),
        k_syn=float(k_syn),
        loglik=-float(best.fun),
        n_cells=lik.n_cells,
        converged=bool(best.success),
        at_bound=at_bound,
        start=best_label,
    )


# ---------------------------------------------------------- profile likelihood
@dataclass
class Profile:
    """Profile likelihood of one quantity; ``ci`` ends are ``None`` where the interval reaches a bound.

    ``nuisance_at_bound[i]`` records whether, at ``values[i]``, any of the
    other rates sits on its bound, i.e. where the box constraints rather than
    the data may be shaping the profile.
    """

    target: str
    mle: float
    values: np.ndarray
    deviance: np.ndarray
    ci: tuple
    level: float
    mle_improved: bool = False
    nuisance_at_bound: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))

    @property
    def closed(self) -> bool:
        return self.ci[0] is not None and self.ci[1] is not None


class _ProfileProblem:
    """Profile of ``target`` over the two remaining log-parameters ``u``."""

    def __init__(self, lik: TelegraphLikelihood, target: str, bounds) -> None:
        if target == "burst_frequency":
            target = "k_on"
        if target not in PROFILE_TARGETS:
            raise ValueError(f"cannot profile {target!r}; choose from {PROFILE_TARGETS}")
        self.lik, self.target = lik, target
        self.lo, self.hi = _log_bounds(bounds)

    def value(self, theta) -> float:
        if self.target == "burst_size":
            return float(theta[2] - theta[1])
        return float(theta[PARAMS.index(self.target)])

    def free(self, theta) -> np.ndarray:
        if self.target == "burst_size":
            return np.array([theta[0], theta[1]])
        return np.delete(theta, PARAMS.index(self.target))

    def theta(self, v, u) -> np.ndarray:
        if self.target == "burst_size":
            return np.array([u[0], u[1], v + u[1]])
        return np.insert(u, PARAMS.index(self.target), v)

    def reduce_grad(self, g) -> np.ndarray:
        if self.target == "burst_size":
            return np.array([g[0], g[1] + g[2]])
        return np.delete(g, PARAMS.index(self.target))

    def range(self):
        if self.target == "burst_size":  # log k_syn - log k_off
            return self.lo[2] - self.hi[1], self.hi[2] - self.lo[1]
        i = PARAMS.index(self.target)
        return self.lo[i], self.hi[i]

    def free_bounds(self, v):
        if self.target == "burst_size":
            lo = np.array([self.lo[0], max(self.lo[1], self.lo[2] - v)])
            hi = np.array([self.hi[0], min(self.hi[1], self.hi[2] - v)])
            return lo, hi
        i = PARAMS.index(self.target)
        return np.delete(self.lo, i), np.delete(self.hi, i)

    def minimize(self, v, u0):
        lo, hi = self.free_bounds(v)

        def fun(u):
            f, g = self.lik.nll_and_grad(self.theta(v, u))
            return f, self.reduce_grad(g)

        # the profile is needed to ~1e-6 in log-likelihood, not to machine precision
        res = _minimize(fun, np.clip(u0, lo, hi), lo, hi, ftol=1e-11, gtol=1e-6)
        return float(res.fun), res.x

    def at_bound(self, v, u, tol: float = 1e-6) -> bool:
        lo, hi = self.free_bounds(v)
        return bool(np.any(u <= lo + tol) or np.any(u >= hi - tol))


def profile_likelihood(
    lik: TelegraphLikelihood,
    fit: TelegraphFit,
    target: str,
    bounds=None,
    level: float = 0.95,
    first_step: float = 0.1,
    max_steps: int = 60,
    grid_points: int = 0,
) -> Profile:
    """Walk the profile outwards from the MLE until it crosses the chi-square threshold.

    Steps grow while the profile is flat and the crossing is located by root
    finding, so each side costs a handful of constrained optimisations.
    ``grid_points > 0`` adds that many evenly spaced (log-scale) evaluations
    across the explored range, for plotting smooth curves.
    """
    problem = _ProfileProblem(lik, target, bounds or DEFAULT_BOUNDS)
    threshold = stats.chi2.ppf(level, 1) / 2.0
    nll_hat = -fit.loglik
    v_hat, u_hat = problem.value(fit.theta), problem.free(fit.theta)
    v_min, v_max = problem.range()
    points = {v_hat: (0.0, u_hat)}
    ends = []
    for direction in (-1.0, 1.0):
        v_prev, u_prev, d_prev, step = v_hat, u_hat, 0.0, first_step
        end = None
        for _ in range(max_steps):
            v = v_prev + direction * step
            at_limit = v <= v_min if direction < 0 else v >= v_max
            if at_limit:
                v = v_min if direction < 0 else v_max
            f, u = problem.minimize(v, u_prev)
            d = f - nll_hat
            points[v] = (d, u)
            if d >= threshold:
                end = _crossing(problem, v_prev, v, u_prev, nll_hat + threshold)
                break
            if at_limit:
                break
            if d - d_prev < threshold / 6.0:
                step = min(step * 1.8, 2.0)
            v_prev, u_prev, d_prev = v, u, d
        ends.append(end)

    if grid_points > 0:
        explored = np.array(sorted(points))
        for v in np.linspace(explored[0], explored[-1], grid_points):
            nearest = explored[np.argmin(np.abs(explored - v))]
            f, u = problem.minimize(v, points[nearest][1])
            points[v] = (f - nll_hat, u)

    vs = np.array(sorted(points))
    ds = np.array([points[v][0] for v in vs])
    return Profile(
        target=problem.target,
        mle=float(np.exp(v_hat)),
        values=np.exp(vs),
        deviance=2.0 * ds,
        ci=tuple(None if e is None else float(np.exp(e)) for e in ends),
        level=level,
        mle_improved=bool(ds.min() < -1e-4),
        nuisance_at_bound=np.array([problem.at_bound(v, points[v][1]) for v in vs]),
    )


def _crossing(problem: _ProfileProblem, v_in, v_out, u_start, level_nll) -> float:
    def g(v):
        return problem.minimize(v, u_start)[0] - level_nll

    lo, hi = sorted((v_in, v_out))
    g_lo, g_hi = g(lo), g(hi)
    if g_lo * g_hi > 0:  # optimiser noise near the threshold; fall back to the outer point
        return float(v_out)
    return float(optimize.brentq(g, lo, hi, xtol=1e-4))


def classify_identifiability(profiles: dict) -> str:
    """Practical identifiability from profile intervals (Raue et al. 2009).

    ``full``: intervals for burst frequency, burst size, ``k_off`` and ``k_syn``
    are all closed; ``burst``: burst frequency and size are closed but
    ``k_off`` or ``k_syn`` is open (or was not profiled); ``none`` otherwise.
    Profile both ``k_off`` and ``k_syn``: with bounded parameters one of them
    can close only because the other has reached its bound.
    """
    freq, size = profiles.get("k_on"), profiles.get("burst_size")
    if freq is None or size is None or not (freq.closed and size.closed):
        return "none"
    others = [profiles[p] for p in ("k_off", "k_syn") if p in profiles]
    if others and all(p.closed for p in others):
        return "full"
    return "burst"


# ---------------------------------------------------------- model comparison
@dataclass
class ModelComparison:
    loglik: dict
    n_params: dict
    aic: dict
    bic: dict
    best: str
    lrt_pvalue: dict
    poisson_mean: float
    nbinom: tuple
    telegraph: TelegraphFit = field(repr=False)


def fit_poisson(counts, capture=None):
    counts, capture = _validate_counts(counts, capture)
    s = np.ones(counts.size) if capture is None else capture
    mean = counts.sum() / s.sum()
    return float(mean), float(poisson_logpmf(counts, mean * s).sum())


def fit_nbinom(counts, capture=None):
    """Negative binomial with per-cell mean ``mu * s_i`` and shared size ``r``: returns ``(mu, r, loglik)``."""
    counts, capture = _validate_counts(counts, capture)
    s = np.ones(counts.size) if capture is None else capture
    n = counts.astype(float)
    from scipy.special import digamma

    def fun(theta):
        mu, r = np.exp(theta)
        m = mu * s
        logp = nbinom_logpmf(n, m, r)
        d_mu = n / mu - s * (r + n) / (r + m)
        d_r = digamma(n + r) - digamma(r) + np.log(r / (r + m)) + 1.0 - (r + n) / (r + m)
        return -logp.sum(), -np.array([mu * d_mu.sum(), r * d_r.sum()])

    mean = max(n.mean(), 1e-6) / s.mean()
    var = n.var()
    r0 = n.mean() ** 2 / (var - n.mean()) if var > n.mean() else 100.0
    lo, hi = np.log([1e-8, 1e-4]), np.log([1e7, 1e7])
    res = _minimize(fun, np.clip(np.log([mean, r0]), lo, hi), lo, hi)
    mu, r = np.exp(res.x)
    return float(mu), float(r), -float(res.fun)


def _boundary_lrt(ll_big: float, ll_small: float) -> float:
    """p-value with the 50:50 chi-square(0)/chi-square(1) null of a boundary test (Self & Liang 1987)."""
    stat = 2.0 * (ll_big - ll_small)
    return 1.0 if stat <= 0 else float(0.5 * stats.chi2.sf(stat, 1))


def compare_models(counts, capture=None, telegraph_fit: TelegraphFit | None = None) -> ModelComparison:
    """Poisson (constitutive), negative binomial (bursty limit) and telegraph, ranked by BIC.

    The models are nested at parameter boundaries, so the likelihood-ratio
    p-values use the boundary-corrected null distribution; for the
    telegraph-vs-NB test this is an approximation.
    """
    counts, capture = _validate_counts(counts, capture)
    mu_p, ll_p = fit_poisson(counts, capture)
    mu_nb, r_nb, ll_nb = fit_nbinom(counts, capture)
    tel = telegraph_fit or fit_telegraph(counts, capture)
    ll_nb = max(ll_nb, ll_p)  # NB contains Poisson in the limit r -> inf
    ll_t = max(tel.loglik, ll_nb)  # telegraph contains NB in the limit k_off -> inf
    loglik = {"poisson": ll_p, "nbinom": ll_nb, "telegraph": ll_t}
    k = {"poisson": 1, "nbinom": 2, "telegraph": 3}
    n = counts.size
    aic = {m: 2 * k[m] - 2 * loglik[m] for m in loglik}
    bic = {m: k[m] * np.log(n) - 2 * loglik[m] for m in loglik}
    best = min(bic, key=bic.get)
    return ModelComparison(
        loglik=loglik,
        n_params=k,
        aic=aic,
        bic=bic,
        best=best,
        lrt_pvalue={
            "nbinom_vs_poisson": _boundary_lrt(ll_nb, ll_p),
            "telegraph_vs_nbinom": _boundary_lrt(ll_t, ll_nb),
        },
        poisson_mean=mu_p,
        nbinom=(mu_nb, r_nb),
        telegraph=tel,
    )
