"""Reaction rate equations and the linear noise approximation (LNA).

The LNA (van Kampen's system-size expansion to first order) describes copy
numbers as the deterministic solution plus Gaussian fluctuations whose
covariance obeys

    d Sigma / dt = J Sigma + Sigma J^T + D,   D = S diag(a(phi)) S^T,

with ``J`` the Jacobian of the rate equations. For networks of zeroth- and
first-order reactions this is exact for the first two moments; for
higher-order reactions its error shrinks with system size.

Conservation laws (for example a fixed number of gene copies) make ``J``
singular. Fluctuations can only move inside the stoichiometric subspace, so
the Lyapunov equation is solved there, in an orthonormal basis ``W`` of the
column space of ``S``, which is exact rather than a regularisation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg, optimize
from scipy.integrate import solve_ivp

from .network import ReactionNetwork


def conservation_laws(network: ReactionNetwork):
    """Orthonormal bases ``(W, L)``: ``W`` spans the stoichiometric subspace, rows of ``L`` are conservation laws."""
    s = network.stoichiometry.astype(float)
    u, sv, _ = linalg.svd(s)
    tol = max(s.shape) * np.finfo(float).eps * (sv[0] if sv.size else 0.0)
    rank = int(np.sum(sv > tol))
    return u[:, :rank], u[:, rank:].T


def rre_steady_state(network: ReactionNetwork, x0, params=None, horizon: float | None = None) -> np.ndarray:
    """Stable fixed point of the rate equations reached from ``x0``.

    The equations are first relaxed with a stiff integrator, then the fixed
    point is polished by Newton's method on the reduced system: the drift
    projected onto the stoichiometric subspace plus the conservation laws.
    """
    rates = network._resolve_rates(params)
    s = network.stoichiometry.astype(float)
    w, cons = conservation_laws(network)
    x0 = np.asarray(x0, dtype=float)
    totals = cons @ x0

    def rhs(_t, phi):
        return s @ network.macroscopic_propensities(phi, rates)

    def jac(_t, phi):
        return s @ network.macroscopic_jacobian(phi, rates)

    phi = x0.copy()
    positive = rates[rates > 0]
    span = horizon if horizon is not None else 10.0 / positive.min()
    for _ in range(16):
        sol = solve_ivp(rhs, (0.0, span), phi, method="LSODA", jac=jac, rtol=1e-10, atol=1e-12)
        phi = sol.y[:, -1]
        if np.linalg.norm(rhs(0.0, phi)) < 1e-6 * (1.0 + np.linalg.norm(phi)):
            break
        span *= 2.0

    def reduced(phi):
        return np.concatenate([w.T @ rhs(0.0, phi), cons @ phi - totals])

    def reduced_jac(phi):
        return np.vstack([w.T @ jac(0.0, phi), cons])

    root = optimize.root(reduced, phi, jac=reduced_jac, method="hybr", options={"xtol": 1e-14})
    phi = root.x
    scale = 1.0 + np.abs(phi).max()
    if np.abs(reduced(phi)).max() > 1e-8 * scale or phi.min() < -1e-9 * scale:
        raise RuntimeError("rate equations did not converge to a non-negative steady state")
    return np.maximum(phi, 0.0)


@dataclass
class LNAResult:
    species: tuple
    mean: np.ndarray
    covariance: np.ndarray
    jacobian: np.ndarray
    diffusion: np.ndarray

    def _i(self, species: str) -> int:
        return self.species.index(species)

    def variance(self, species: str) -> float:
        i = self._i(species)
        return float(self.covariance[i, i])

    def fano(self, species: str) -> float:
        return self.variance(species) / float(self.mean[self._i(species)])

    def cv2(self, species: str) -> float:
        return self.variance(species) / float(self.mean[self._i(species)]) ** 2


def lna_stationary(network: ReactionNetwork, x0, params=None) -> LNAResult:
    """Stationary mean and covariance in the linear noise approximation."""
    rates = network._resolve_rates(params)
    phi = rre_steady_state(network, x0, rates)
    s = network.stoichiometry.astype(float)
    a = network.macroscopic_propensities(phi, rates)
    jac = s @ network.macroscopic_jacobian(phi, rates)
    diff = s @ np.diag(a) @ s.T
    w, _ = conservation_laws(network)
    jac_r = w.T @ jac @ w
    if np.any(np.linalg.eigvals(jac_r).real >= 0):
        raise ValueError("steady state is not linearly stable; the LNA does not apply")
    cov_r = linalg.solve_continuous_lyapunov(jac_r, -(w.T @ diff @ w))
    cov = w @ cov_r @ w.T
    return LNAResult(network.species, phi, (cov + cov.T) / 2.0, jac, diff)


def lna_trajectory(network: ReactionNetwork, x0, t_eval, params=None):
    """Mean and covariance over time from a deterministic initial state.

    Returns ``(mean, cov)`` with shapes (len(t_eval), n) and (len(t_eval), n, n).
    """
    rates = network._resolve_rates(params)
    s = network.stoichiometry.astype(float)
    n = network.n_species
    t_eval = np.atleast_1d(np.asarray(t_eval, dtype=float))

    def rhs(_t, y):
        phi, cov = y[:n], y[n:].reshape(n, n)
        a = network.macroscopic_propensities(phi, rates)
        jac = s @ network.macroscopic_jacobian(phi, rates)
        dcov = jac @ cov + cov @ jac.T + s @ np.diag(a) @ s.T
        return np.concatenate([s @ a, dcov.ravel()])

    y0 = np.concatenate([np.asarray(x0, dtype=float), np.zeros(n * n)])
    if t_eval.max() == 0:
        ys = np.repeat(y0[None, :], t_eval.size, axis=0)
    else:
        sol = solve_ivp(rhs, (0.0, t_eval.max()), y0, t_eval=t_eval, method="LSODA", rtol=1e-11, atol=1e-13)
        if not sol.success:
            raise RuntimeError(sol.message)
        ys = sol.y.T
    return ys[:, :n], ys[:, n:].reshape(-1, n, n)
