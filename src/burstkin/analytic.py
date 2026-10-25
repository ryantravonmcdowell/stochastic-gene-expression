"""Closed-form results for stochastic gene expression.

The central object is the stationary mRNA distribution of the telegraph model
(Peccoud & Ycart 1995). With rates in units of the decay rate,

    P(n) = k_syn^n / n! * B(k_on + n, k_off) / B(k_on, k_off)
           * 1F1(k_on + n; k_on + k_off + n; -k_syn),

i.e. a Poisson distribution whose mean ``k_syn * p`` is mixed over
``p ~ Beta(k_on, k_off)``. The confluent hypergeometric function with a
negative argument is an alternating series that loses all precision when
``k_syn`` is large. Kummer's transformation ``1F1(a; b; -z) = e^-z 1F1(b-a; b; z)``
turns it into a series of positive terms, which is summed here in log space
with a rigorous bound on the truncated tail, together with its derivatives.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import special, stats

_MAX_CHUNK = 512


# --------------------------------------------------------------- Kummer series
def log_hyp1f1(a, b, z, grad: bool = False, rtol: float = 1e-16, max_terms: int = 2_000_000):
    """``log 1F1(a; b; z)`` for ``a > 0``, ``b > 0``, ``z >= 0`` (broadcasting).

    With ``grad=True`` also returns the partial derivatives of the log with
    respect to ``a``, ``b`` and ``z``.

    Every term ``t_k = (a)_k / (b)_k * z^k / k!`` is positive and the ratio
    ``t_{k+1} / t_k`` is unimodal in ``k``, so once it is falling and below
    one the remaining tail is bounded by a geometric series. Summation stops
    when that bound is below ``rtol`` times the partial sum.
    """
    a, b, z = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (a, b, z)))
    shape = a.shape
    a, b, z = a.ravel(), b.ravel(), z.ravel()
    if a.size and not (np.all(a > 0) and np.all(b > 0) and np.all(z >= 0)):
        raise ValueError("log_hyp1f1 requires a > 0, b > 0 and z >= 0")

    log_f = np.zeros(a.size)
    ga = np.zeros(a.size)
    gb = np.zeros(a.size)
    gz = a / b  # d/dz log 1F1 at z = 0; overwritten for z > 0
    pos = np.flatnonzero(z > 0)
    if pos.size:
        ap, bp, zp = a[pos], b[pos], z[pos]
        # The term ratio equals one at the positive root of
        # k^2 + (b + 1 - z) k + (b - a z) = 0, which locates the largest term.
        p = bp + 1.0 - zp
        disc = np.maximum(p * p - 4.0 * (bp - ap * zp), 0.0)
        k_peak = np.maximum((-p + np.sqrt(disc)) / 2.0, 0.0)
        need = k_peak + 8.0 * np.sqrt(k_peak + 1.0) + 16.0
        bucket = np.floor(np.log2(need)).astype(int)
        for level in np.unique(bucket):
            idx = pos[bucket == level]
            chunk = min(2 ** (int(level) + 1), _MAX_CHUNK)
            res = _kummer_series(a[idx], b[idx], z[idx], chunk, grad, rtol, max_terms)
            log_f[idx] = res[0]
            if grad:
                ga[idx], gb[idx], gz[idx] = res[1], res[2], res[3]
    if grad:
        return log_f.reshape(shape), ga.reshape(shape), gb.reshape(shape), gz.reshape(shape)
    return log_f.reshape(shape)


def _kummer_series(a, b, z, chunk, grad, rtol, max_terms):
    n = a.size
    log_sum = np.zeros(n)  # log of the partial sum; t_0 = 1
    log_term = np.zeros(n)  # log of the last term summed
    sum_da = np.zeros(n)  # d log t_k / da at the last term
    sum_db = np.zeros(n)
    ga, gb, gz = np.zeros(n), np.zeros(n), np.zeros(n)
    log_rtol = np.log(rtol)
    todo = np.arange(n)
    k0 = 0
    while todo.size:
        if k0 > max_terms:
            raise RuntimeError("1F1 series did not converge")
        ks = k0 + np.arange(chunk, dtype=float)
        A = a[todo, None] + ks
        B = b[todo, None] + ks
        ratio = A * z[todo, None] / (B * (ks + 1.0))  # t_{k+1} / t_k
        log_t = log_term[todo, None] + np.cumsum(np.log(ratio), axis=1)  # terms k0+1 .. k0+chunk
        peak = log_t.max(axis=1)
        w = np.exp(log_t - peak[:, None])
        w_sum = w.sum(axis=1)
        log_chunk = peak + np.log(w_sum)
        new_sum = np.logaddexp(log_sum[todo], log_chunk)
        if grad:
            # d log t_k / da = sum_{m<k} 1/(a+m);  d/db = -sum_{m<k} 1/(b+m);  d/dz = k/z
            da_k = sum_da[todo, None] + np.cumsum(1.0 / A, axis=1)
            db_k = sum_db[todo, None] - np.cumsum(1.0 / B, axis=1)
            keep_old = np.exp(log_sum[todo] - new_sum)
            take_new = np.exp(log_chunk - new_sum) / w_sum
            ga[todo] = keep_old * ga[todo] + take_new * (w * da_k).sum(axis=1)
            gb[todo] = keep_old * gb[todo] + take_new * (w * db_k).sum(axis=1)
            gz[todo] = keep_old * gz[todo] + take_new * (w * (ks + 1.0)).sum(axis=1) / z[todo]
            sum_da[todo] = da_k[:, -1]
            sum_db[todo] = db_k[:, -1]
        log_sum[todo] = new_sum
        log_term[todo] = log_t[:, -1]

        k_next = k0 + chunk
        r_next = (a[todo] + k_next) * z[todo] / ((b[todo] + k_next) * (k_next + 1.0))
        falling = r_next <= ratio[:, -1]
        with np.errstate(divide="ignore", invalid="ignore"):
            log_tail = log_term[todo] + np.log(r_next) - np.log1p(-r_next)
        done = falling & (r_next < 1.0) & (log_tail - log_sum[todo] < log_rtol)
        todo = todo[~done]
        k0 = k_next
    return log_sum, ga, gb, gz


# ----------------------------------------------- many cells, shared switching
_BLOCK_ELEMENTS = 2_000_000


def _shared_switching_terms(n, k_on, k_off, rate, grad, rtol=1e-16):
    """``log 1F1(k_off; k_on + k_off + n_i; rate_i)`` for many cells at once.

    In a likelihood every cell shares ``k_on`` and ``k_off``; only the count
    ``n_i`` and the effective transcription rate ``rate_i = k_syn * s_i``
    differ. The log of term ``k`` then splits into pieces that are computed
    once per evaluation:

        log t_k = [lgamma(a+k) - lgamma(a) - lgamma(k+1)]          (shared)
                - [lgamma(c+n_i+k) - lgamma(c+n_i)]                 (table lookup)
                + k log rate_i,                     a = k_off, c = k_on + k_off,

    so the per-cell work is a gather, an exponential and a matrix product
    instead of a logarithm per term.

    Only a window ``[k_lo, k_lo + K)`` around the largest term is summed. The
    term ratio is unimodal in ``k``, which gives rigorous geometric bounds on
    both skipped tails; a cell whose bounds exceed ``rtol`` is redone with a
    wider window.
    """
    n = np.asarray(n, dtype=np.int64)
    rate = np.asarray(rate, dtype=float)
    a, total = float(k_off), float(k_on + k_off)
    log_f = np.zeros(n.size)
    ga = np.zeros(n.size)
    gb = np.zeros(n.size)
    gz = a / (total + n)  # value at rate = 0
    pos = np.flatnonzero(rate > 0)
    if pos.size == 0:
        return log_f, ga, gb, gz
    b = total + n[pos]
    z = rate[pos]
    p = b + 1.0 - z
    disc = np.maximum(p * p - 4.0 * (b - a * z), 0.0)
    k_peak = np.maximum((-p + np.sqrt(disc)) / 2.0, 0.0)
    margin = 10.0 * np.sqrt(k_peak + 1.0) + 24.0
    k_lo = np.maximum(np.floor((k_peak - margin) / 64.0) * 64.0, 0.0).astype(np.int64)
    terms = (np.ceil((k_peak + margin - k_lo) / 32.0) * 32.0).astype(np.int64)
    log_rtol = np.log(rtol)
    pending = []
    for lo in np.unique(k_lo):
        for t in np.unique(terms[k_lo == lo]):
            pending.append((pos[(k_lo == lo) & (terms == t)], int(lo), int(t)))
    while pending:
        idx_all, start_k, n_terms = pending.pop()
        k = start_k + np.arange(n_terms, dtype=float)
        shared = special.gammaln(a + k) - special.gammaln(a) - special.gammaln(k + 1.0)
        if grad:
            shared_da = special.digamma(a + k) - special.digamma(a)
        block = max(1, _BLOCK_ELEMENTS // n_terms)
        for first in range(0, idx_all.size, block):
            idx = idx_all[first : first + block]
            nn = n[idx]
            zz = rate[idx]
            j = np.arange(nn.max() + start_k + n_terms + 1, dtype=float)
            table = special.gammaln(total + j)
            cols = nn[:, None] + (start_k + np.arange(n_terms))
            log_t = table[cols]
            np.subtract(shared, log_t, out=log_t)
            log_t += table[nn][:, None]
            log_t += k[None, :] * np.log(zz)[:, None]
            peak = log_t.max(axis=1)
            # Terms below e^-700 of the largest cannot change a double-precision
            # sum; clipping them avoids slow denormal arithmetic.
            w = np.exp(np.maximum(log_t - peak[:, None], -700.0))
            w_sum = w.sum(axis=1)
            log_sum = peak + np.log(w_sum)
            bb = total + nn

            # upper tail beyond the last term t_K (K = start_k + n_terms - 1)
            kk = start_k + n_terms - 1.0
            r_last = (a + kk - 1.0) * zz / ((bb + kk - 1.0) * kk)
            r_next = (a + kk) * zz / ((bb + kk) * (kk + 1.0))
            with np.errstate(divide="ignore", invalid="ignore"):
                log_tail = log_t[:, -1] + np.log(r_next) - np.log1p(-r_next)
            ok = (r_next <= r_last) & (r_next < 1.0) & (log_tail - log_sum < log_rtol)
            # lower tail below t_{start_k}: every ratio there is >= min(r_0, r_{start_k - 1})
            if start_k > 0:
                rho = np.minimum(a * zz / bb, (a + start_k - 1.0) * zz / ((bb + start_k - 1.0) * start_k))
                with np.errstate(divide="ignore", invalid="ignore"):
                    log_low = log_t[:, 0] - np.log(rho - 1.0)
                ok &= (rho > 1.0) & (log_low - log_sum < log_rtol)

            if ok.all():
                good, wo, so = idx, w, w_sum
                log_f[idx] = log_sum
            else:
                good, wo, so = idx[ok], w[ok], w_sum[ok]
                log_f[good] = log_sum[ok]
                pending.append((idx[~ok], 0, 2 * (start_k + n_terms)))
            if grad and good.size:
                psi = special.digamma(total + j)
                sel = slice(None) if good is idx else ok
                ga[good] = (wo @ shared_da) / so
                gb[good] = -((wo * psi[cols[sel]]).sum(axis=1) / so - psi[nn[sel]])
                gz[good] = (wo @ k) / so / rate[good]
    return log_f, ga, gb, gz


def telegraph_loglik_terms(n, k_on: float, k_off: float, rate, grad: bool = False):
    """Telegraph log-probabilities for cells sharing ``k_on, k_off`` with per-cell rates.

    ``rate`` is the effective transcription rate of each cell (``k_syn`` times
    its capture efficiency). With ``grad=True`` returns
    ``(logp, d/dk_on, d/dk_off, d/drate)``.
    """
    n, rate = np.broadcast_arrays(np.asarray(n), np.asarray(rate, dtype=float))
    shape = n.shape
    n, rate = n.ravel(), rate.ravel()
    if n.size and (np.any(n < 0) or np.any(n != np.floor(n))):
        raise ValueError("counts must be non-negative integers")
    if not (k_on > 0 and k_off > 0 and np.all(rate >= 0)):
        raise ValueError("telegraph model needs k_on > 0, k_off > 0 and rates >= 0")
    n = n.astype(np.int64)
    nf = n.astype(float)
    total = k_on + k_off
    log_f, ga, gb, gz = _shared_switching_terms(n, k_on, k_off, rate, grad)
    logp = (
        special.gammaln(k_on + nf)
        - special.gammaln(k_on)
        + special.gammaln(total)
        - special.gammaln(total + nf)
        - special.gammaln(nf + 1.0)
        + special.xlogy(nf, rate)
        - rate
        + log_f
    ).reshape(shape)
    if not grad:
        return logp
    psi_tot = special.digamma(total) - special.digamma(total + nf)
    d_on = special.digamma(k_on + nf) - special.digamma(k_on) + psi_tot + gb
    d_off = psi_tot + ga + gb
    with np.errstate(divide="ignore", invalid="ignore"):
        d_rate = np.where(nf > 0, nf / rate, 0.0) - 1.0 + gz
    return logp, d_on.reshape(shape), d_off.reshape(shape), d_rate.reshape(shape)


# ------------------------------------------------------------ telegraph model
def _broadcast_counts(n, *params):
    n = np.asarray(n)
    if n.size and (np.any(n < 0) or np.any(n != np.floor(n))):
        raise ValueError("counts must be non-negative integers")
    arrays = np.broadcast_arrays(n.astype(float), *(np.asarray(p, dtype=float) for p in params))
    return arrays


def _check_telegraph(k_on, k_off, k_syn):
    if not (np.all(k_on > 0) and np.all(k_off > 0) and np.all(k_syn >= 0)):
        raise ValueError("telegraph model needs k_on > 0, k_off > 0 and k_syn >= 0")


def telegraph_logpmf(n, k_on, k_off, k_syn):
    """Log stationary probability of ``n`` mRNAs (rates in units of the decay rate)."""
    n, k_on, k_off, k_syn = _broadcast_counts(n, k_on, k_off, k_syn)
    _check_telegraph(k_on, k_off, k_syn)
    total = k_on + k_off
    return (
        special.gammaln(k_on + n)
        - special.gammaln(k_on)
        + special.gammaln(total)
        - special.gammaln(total + n)
        - special.gammaln(n + 1.0)
        + special.xlogy(n, k_syn)
        - k_syn
        + log_hyp1f1(k_off, total + n, k_syn)
    )


def telegraph_logpmf_grad(n, k_on, k_off, k_syn):
    """Log-probability and its derivatives with respect to ``k_on``, ``k_off``, ``k_syn``."""
    n, k_on, k_off, k_syn = _broadcast_counts(n, k_on, k_off, k_syn)
    _check_telegraph(k_on, k_off, k_syn)
    total = k_on + k_off
    log_f, ga, gb, gz = log_hyp1f1(k_off, total + n, k_syn, grad=True)
    lgam_on = special.gammaln(k_on + n) - special.gammaln(k_on)
    lgam_tot = special.gammaln(total) - special.gammaln(total + n)
    logp = lgam_on + lgam_tot - special.gammaln(n + 1.0) + special.xlogy(n, k_syn) - k_syn + log_f
    psi_tot = special.digamma(total) - special.digamma(total + n)
    d_on = special.digamma(k_on + n) - special.digamma(k_on) + psi_tot + gb
    d_off = psi_tot + ga + gb
    with np.errstate(divide="ignore", invalid="ignore"):
        d_syn = np.where(n > 0, n / k_syn, 0.0) - 1.0 + gz
    return logp, d_on, d_off, d_syn


def telegraph_pmf(n_max: int, k_on: float, k_off: float, k_syn: float) -> np.ndarray:
    """Stationary probabilities of 0..n_max mRNAs."""
    return np.exp(telegraph_logpmf(np.arange(n_max + 1), k_on, k_off, k_syn))


def telegraph_pmf_quadrature(n, k_on: float, k_off: float, k_syn: float, order: int = 256) -> np.ndarray:
    """Independent evaluation of the same pmf by Gauss-Jacobi quadrature over ``p ~ Beta``.

    Used only to cross-check :func:`telegraph_logpmf`; it is accurate when
    ``k_syn`` is moderate (the Poisson factor must be smooth on the node grid).
    """
    y, w = special.roots_jacobi(order, k_off - 1.0, k_on - 1.0)
    p = (1.0 + y) / 2.0
    w = w / w.sum()
    n = np.asarray(n)
    return (w * stats.poisson.pmf(n[..., None], k_syn * p)).sum(axis=-1)


@dataclass(frozen=True)
class TelegraphMoments:
    mean: float
    variance: float
    fano: float
    p_on: float
    burst_size: float
    burst_frequency: float


def telegraph_moments(k_on: float, k_off: float, k_syn: float) -> TelegraphMoments:
    """Exact stationary moments: by the law of total variance over ``p ~ Beta``."""
    total = k_on + k_off
    mean = k_syn * k_on / total
    variance = mean + k_syn**2 * k_on * k_off / (total**2 * (total + 1.0))
    return TelegraphMoments(
        mean=mean,
        variance=variance,
        fano=variance / mean if mean > 0 else np.nan,
        p_on=k_on / total,
        burst_size=k_syn / k_off,
        burst_frequency=k_on,
    )


def telegraph_factorial_moment(order: int, k_on: float, k_off: float, k_syn: float) -> float:
    """``E[n (n-1) ... (n-order+1)] = k_syn^order E[p^order]`` for ``p ~ Beta(k_on, k_off)``."""
    j = np.arange(order)
    return float(k_syn**order * np.prod((k_on + j) / (k_on + k_off + j)))


def telegraph_from_factorial_moments(m1: float, m2: float, m3: float):
    """Invert the first three factorial moments to ``(k_on, k_off, k_syn)``; ``None`` if inadmissible.

    With ``b_1 = m_1``, ``b_2 = m_2 / m_1``, ``b_3 = m_3 / m_2`` and
    ``b_k = k_syn (k_on + k - 1) / (k_on + k_off + k - 1)``,

        k_on + k_off = 2 (b_2 - b_3) / (b_1 - 2 b_2 + b_3)
        k_syn        = (k_on + k_off) (b_2 - b_1) + b_2
        k_on         = b_1 (k_on + k_off) / k_syn.
    """
    if min(m1, m2, m3) <= 0:
        return None
    b1, b2, b3 = m1, m2 / m1, m3 / m2
    denom = b1 - 2.0 * b2 + b3
    if denom == 0:
        return None
    total = 2.0 * (b2 - b3) / denom
    k_syn = total * (b2 - b1) + b2
    if not (np.isfinite(total) and total > 0 and k_syn > 0):
        return None
    k_on = b1 * total / k_syn
    k_off = total - k_on
    if not (k_on > 0 and k_off > 0):
        return None
    return float(k_on), float(k_off), float(k_syn)


def telegraph_from_moments(counts, capture=None):
    """Method-of-moments estimate ``(k_on, k_off, k_syn)`` from counts, or ``None``.

    Each cell's count is a binomial sample of its mRNA with capture efficiency
    ``s_i``, which scales the k-th factorial moment by ``E[s^k]``; dividing by
    it recovers the moments of the true copy number.
    """
    n = np.asarray(counts, dtype=float)
    s = np.ones_like(n) if capture is None else np.asarray(capture, dtype=float)
    m = [np.mean(n), np.mean(n * (n - 1)), np.mean(n * (n - 1) * (n - 2))]
    r = [m[k] / np.mean(s ** (k + 1)) for k in range(3)]
    return telegraph_from_factorial_moments(*r)


def sample_telegraph(k_on, k_off, k_syn, size, rng=None, capture=None) -> np.ndarray:
    """Exact draws from the stationary distribution, optionally after capture.

    Binomial capture of a Poisson(lambda) count is Poisson(s * lambda), so a
    captured draw is Poisson(k_syn * s * p) with ``p ~ Beta(k_on, k_off)``.
    """
    rng = np.random.default_rng(rng)
    p = rng.beta(k_on, k_off, size)
    lam = k_syn * p if capture is None else k_syn * p * np.asarray(capture, dtype=float)
    return rng.poisson(lam)


# --------------------------------------------------- simpler reference models
def poisson_logpmf(n, mean):
    n, mean = _broadcast_counts(n, mean)
    return special.xlogy(n, mean) - mean - special.gammaln(n + 1.0)


def nbinom_logpmf(n, mean, size):
    """Negative binomial with the given mean and size (shape) ``r``; variance ``mean + mean^2 / r``."""
    n, mean, size = _broadcast_counts(n, mean, size)
    log_denom = np.log(size + mean)
    return (
        special.gammaln(n + size)
        - special.gammaln(size)
        - special.gammaln(n + 1.0)
        + size * (np.log(size) - log_denom)
        + special.xlogy(n, mean)
        - n * log_denom
    )


def bursty_limit_logpmf(n, burst_frequency, burst_size):
    """Telegraph model when bursts are short (``k_off -> inf`` at fixed ``k_syn / k_off``).

    The stationary law becomes negative binomial with size ``k_on`` and mean
    ``k_on * burst_size`` (Shahrezaei & Swain 2008).
    """
    return nbinom_logpmf(n, np.asarray(burst_frequency) * np.asarray(burst_size), burst_frequency)


def birth_death_pmf(n_max: int, t: float, k: float, gamma: float = 1.0, n0: int = 0) -> np.ndarray:
    """Exact transient distribution of the birth-death process started at ``n0``.

    Survivors of the initial molecules are ``Binomial(n0, e^{-gamma t})`` and
    newly made ones are ``Poisson(k / gamma (1 - e^{-gamma t}))``.
    """
    survive = np.exp(-gamma * t)
    lam = k / gamma * (1.0 - survive)
    pois = stats.poisson.pmf(np.arange(n_max + 1), lam)
    if n0 == 0:
        return pois
    binom = stats.binom.pmf(np.arange(n0 + 1), n0, survive)
    return np.convolve(binom, pois)[: n_max + 1]
