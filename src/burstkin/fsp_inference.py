"""Likelihood inference for networks without a closed-form distribution.

Snapshot data - one species counted in independent cells at several times,
as in smFISH or flow-cytometry time courses - have likelihood

    L(theta) = prod_t prod_i  p_t(n_i ; theta),

where ``p_t`` is the master-equation solution at time ``t``. Here ``p_t``
comes from the finite state projection. Because the sink-boundary FSP
solution never exceeds the true probability, the computed likelihood is a
lower bound on the exact one, and the gap is bounded by the reported
truncation error.

Uncertainty is estimated from the observed Fisher information (a finite-
difference Hessian of the negative log-likelihood in log-parameters).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
from scipy import optimize

from .fsp import FSP
from .network import ReactionNetwork


@dataclass
class Snapshots:
    """Counts of one species in independent cells; ``times`` may include ``np.inf`` for steady state."""

    times: np.ndarray
    counts: list

    def __post_init__(self) -> None:
        self.times = np.asarray(self.times, dtype=float)
        self.counts = [np.asarray(c, dtype=np.int64) for c in self.counts]
        if len(self.counts) != self.times.size:
            raise ValueError("need one count array per time point")
        finite = self.times[np.isfinite(self.times)]
        if np.any(np.diff(finite) < 0) or np.any(finite < 0):
            raise ValueError("finite times must be non-decreasing and >= 0")


@dataclass
class SnapshotFit:
    params: dict
    nll: float
    converged: bool
    standard_errors: dict = field(default_factory=dict)  # on the natural-log scale
    truncation_error: float = 0.0

    def interval(self, name: str, z: float = 1.96) -> tuple:
        """Approximate Wald interval, symmetric on the log scale."""
        se = self.standard_errors.get(name, np.nan)
        value = self.params[name]
        return float(value * np.exp(-z * se)), float(value * np.exp(z * se))


class SnapshotLikelihood:
    """FSP likelihood of snapshot counts of ``species`` for the parameters in ``free``.

    Parameters
    ----------
    network, x0 : model and the (deterministic) initial state at time 0.
    species : the observed species.
    data : :class:`Snapshots`.
    free : parameter names to estimate (on the log scale).
    bounds : FSP copy-number bounds (see :class:`burstkin.fsp.FSP`).
    fixed : values for parameters that are not estimated (defaults otherwise).
    """

    def __init__(
        self,
        network: ReactionNetwork,
        x0,
        species: str,
        data: Snapshots,
        free: Sequence[str],
        bounds: Mapping[str, int],
        fixed: Mapping[str, float] | None = None,
    ) -> None:
        self.network = network
        self.species = species
        self.data = data
        self.free = tuple(free)
        unknown = set(self.free) - set(network.parameters)
        if unknown:
            raise ValueError(f"unknown parameters {sorted(unknown)}")
        self.fixed = dict(fixed or {})
        self.fsp = FSP(network, x0, bounds)
        n_max = self.fsp.bounds[network.index(species)]
        self.histograms = []
        for c in data.counts:
            if c.size and c.max() > n_max:
                raise ValueError(f"observed {species} count {c.max()} exceeds the FSP bound {n_max}")
            self.histograms.append(np.bincount(c, minlength=n_max + 1).astype(float))
        self._finite = np.isfinite(data.times)
        self.last_truncation_error = 0.0
        self.evaluations = 0

    def params(self, theta) -> dict:
        return {**self.fixed, **dict(zip(self.free, np.exp(theta)))}

    def marginals(self, theta) -> np.ndarray:
        """Model distribution of the observed species at each time, shape (n_times, bound + 1)."""
        rates = self.network.rate_vector(self.params(theta))
        out = np.empty((self.data.times.size, len(self.histograms[0])))
        error = 0.0
        if self._finite.any():
            P, err = self.fsp.transient(self.data.times[self._finite], rates)
            out[self._finite] = self.fsp.marginal(P, self.species)
            error = float(err.max())
        if (~self._finite).any():
            st = self.fsp.stationary(rates)
            out[~self._finite] = self.fsp.marginal(st.p, self.species)
            error = max(error, st.boundary_mass)
        self.last_truncation_error = error
        return out

    def nll(self, theta) -> float:
        self.evaluations += 1
        marg = self.marginals(theta)
        with np.errstate(divide="ignore"):
            logm = np.log(np.maximum(marg, 1e-300))
        return -float(sum(h @ lm for h, lm in zip(self.histograms, logm)))

    def hessian(self, theta, step: float = 1e-3) -> np.ndarray:
        """Central finite-difference Hessian of the negative log-likelihood."""
        theta = np.asarray(theta, dtype=float)
        k = theta.size
        H = np.empty((k, k))
        f0 = self.nll(theta)
        for i in range(k):
            for j in range(i, k):
                ei, ej = np.eye(k)[i] * step, np.eye(k)[j] * step
                if i == j:
                    H[i, i] = (self.nll(theta + ei) - 2 * f0 + self.nll(theta - ei)) / step**2
                else:
                    H[i, j] = H[j, i] = (
                        self.nll(theta + ei + ej)
                        - self.nll(theta + ei - ej)
                        - self.nll(theta - ei + ej)
                        + self.nll(theta - ei - ej)
                    ) / (4 * step**2)
        return H

    def fit(
        self,
        starts: Sequence[Mapping[str, float]],
        bounds: Mapping[str, tuple] | None = None,
        uncertainty: bool = True,
    ) -> SnapshotFit:
        """Maximise the likelihood from each start (natural-scale parameter dicts) and keep the best.

        ``bounds`` maps parameter names to ``(low, high)`` on the natural
        scale; the default allows 1e-3 to 1e3 (in units of the decay rate).
        The likelihood is accurate to ~1e-12, so central finite differences
        give reliable gradients for L-BFGS-B.
        """
        bounds = dict(bounds or {})
        box = [tuple(np.log(bounds.get(p, (1e-3, 1e3)))) for p in self.free]
        best = None
        for start in starts:
            theta0 = np.clip(np.log([start[p] for p in self.free]), [b[0] for b in box], [b[1] for b in box])
            res = optimize.minimize(
                self.nll, theta0, method="L-BFGS-B", jac="3-point", bounds=box,
                options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-5},
            )
            if best is None or res.fun < best.fun:
                best = res
        theta = best.x
        self.marginals(theta)
        fit = SnapshotFit(
            params={p: float(v) for p, v in zip(self.free, np.exp(theta))},
            nll=float(best.fun),
            converged=bool(best.success),
            truncation_error=self.last_truncation_error,
        )
        if uncertainty:
            H = self.hessian(theta)
            try:
                cov = np.linalg.inv(H)
                se = np.sqrt(np.clip(np.diag(cov), 0, None))
                fit.standard_errors = {p: float(s) for p, s in zip(self.free, se)}
            except np.linalg.LinAlgError:
                fit.standard_errors = {p: float("nan") for p in self.free}
        return fit
