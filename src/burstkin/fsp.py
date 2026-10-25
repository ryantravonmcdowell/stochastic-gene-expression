"""Finite state projection (FSP) of the chemical master equation.

The master equation ``dp/dt = A p`` is an infinite linear system over all copy
number vectors. The FSP (Munsky & Khammash 2006) keeps the states reachable
from the initial condition inside a box of copy-number bounds and solves the
resulting finite system exactly with sparse linear algebra.

Two boundary treatments are available:

* ``"sink"``: probability that flows out of the box is lost. The truncated
  solution is then a lower bound on the true one, state by state, and the
  lost mass ``1 - sum(p)`` bounds the total error (the FSP theorem). This is
  used for transient solutions and for likelihoods built from them.
* ``"reflect"``: transitions that would leave the box are dropped, which keeps
  the generator conservative so a stationary distribution exists. Its quality
  is reported as the probability sitting on states next to the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply, spsolve

from .network import ReactionNetwork


@dataclass
class StationaryResult:
    p: np.ndarray
    boundary_mass: float
    residual: float


class FSP:
    """Truncated master equation for ``network`` started from ``x0``.

    Parameters
    ----------
    network : the reaction network.
    x0 : initial state; states are enumerated by reachability from it.
    bounds : maximum copy number of each species (mapping or sequence). Species
        missing from a mapping must be covered by a conservation law (such as
        the gene copies ``G_off + G_on = 1``), which then supplies the bound.
    """

    def __init__(self, network: ReactionNetwork, x0, bounds, max_states: int = 2_000_000) -> None:
        self.network = network
        self.x0 = np.asarray(x0, dtype=np.int64)
        if isinstance(bounds, dict):
            b = np.array(
                [bounds[s] if s in bounds else conservation_bound(network, self.x0, s) for s in network.species],
                dtype=np.int64,
            )
        else:
            b = np.asarray(bounds, dtype=np.int64)
        if b.shape != (network.n_species,) or np.any(b < self.x0):
            raise ValueError("bounds must give one maximum per species, each >= the initial state")
        self.bounds = b
        self._radix = b + 1
        self._strides = np.concatenate([np.cumprod(self._radix[::-1])[::-1][1:], [1]]).astype(np.int64)
        box = int(np.prod(self._radix.astype(float)))
        if box > 50 * max_states:
            raise ValueError(f"bounding box has {box} states; tighten the bounds")
        self.states = self._enumerate(max_states)
        self.n_states = len(self.states)
        self._lookup = -np.ones(box, dtype=np.int64)
        self._lookup[self._encode(self.states)] = np.arange(self.n_states)
        self._build_structure()

    # ----------------------------------------------------------- state space
    def _encode(self, x: np.ndarray) -> np.ndarray:
        return x @ self._strides

    def _inside(self, x: np.ndarray) -> np.ndarray:
        return np.all((x >= 0) & (x <= self.bounds), axis=-1)

    def _enumerate(self, max_states: int) -> np.ndarray:
        """Breadth-first search over transitions with positive propensity."""
        net = self.network
        visited = np.zeros(int(np.prod(self._radix.astype(float))), dtype=bool)
        frontier = self.x0[None, :]
        visited[self._encode(frontier)] = True
        found = [frontier]
        change = net.stoichiometry.T
        total = 1
        while frontier.size:
            h = net.combinatorial_factors(frontier)
            new = []
            for j in range(net.n_reactions):
                y = frontier[h[:, j] > 0] + change[j]
                y = y[self._inside(y)]
                if y.size:
                    codes = self._encode(y)
                    fresh = ~visited[codes]
                    codes, y = codes[fresh], y[fresh]
                    codes, first = np.unique(codes, return_index=True)
                    visited[codes] = True
                    new.append(y[first])
            frontier = np.concatenate(new) if new else np.empty((0, net.n_species), dtype=np.int64)
            total += len(frontier)
            if total > max_states:
                raise ValueError(f"more than {max_states} reachable states; tighten the bounds")
            if frontier.size:
                found.append(frontier)
        states = np.concatenate(found)
        return states[np.argsort(self._encode(states), kind="stable")]

    def index_of(self, x) -> np.ndarray:
        """Index of each state in ``x`` (shape (..., n_species)), or -1 if not in the projection."""
        x = np.asarray(x, dtype=np.int64)
        idx = np.full(x.shape[:-1], -1, dtype=np.int64)
        inside = self._inside(x)
        idx[inside] = self._lookup[self._encode(x[inside])]
        return idx

    # ------------------------------------------------------------- generator
    def _build_structure(self) -> None:
        """Unit-rate transition pattern; the generator is linear in the rate constants."""
        net = self.network
        h = net.combinatorial_factors(self.states)  # (states, reactions)
        rows, cols, vals, rx_of, leaving = [], [], [], [], []
        src_all = np.arange(self.n_states)
        for j in range(net.n_reactions):
            active = h[:, j] > 0
            src = src_all[active]
            dst = self.index_of(self.states[active] + net.stoichiometry[:, j])
            inside = dst >= 0
            # transfer into the target state
            rows.append(dst[inside])
            cols.append(src[inside])
            vals.append(h[active, j][inside])
            rx_of.append(np.full(inside.sum(), j))
            leaving.append(np.zeros(inside.sum(), dtype=bool))
            # outflow from the source (diagonal); flagged when the target is outside
            rows.append(src)
            cols.append(src)
            vals.append(-h[active, j])
            rx_of.append(np.full(src.size, j))
            leaving.append(~inside)
        self._rows = np.concatenate(rows)
        self._cols = np.concatenate(cols)
        self._vals = np.concatenate(vals)
        self._rx = np.concatenate(rx_of)
        self._leaving = np.concatenate(leaving)
        # states with at least one transition out of the box (for any positive rates)
        self._edge_reactions = [
            np.unique(self._cols[self._leaving & (self._rx == j)]) for j in range(net.n_reactions)
        ]

    def generator(self, rates=None, mode: str = "sink") -> sparse.csc_matrix:
        """Sparse generator ``A`` with ``dp/dt = A p`` (columns are source states)."""
        if mode not in ("sink", "reflect"):
            raise ValueError("mode must be 'sink' or 'reflect'")
        c = self.network._resolve_rates(rates)
        vals = self._vals * c[self._rx]
        keep = slice(None) if mode == "sink" else ~self._leaving
        return sparse.csc_matrix(
            (vals[keep], (self._rows[keep], self._cols[keep])), shape=(self.n_states, self.n_states)
        )

    def edge_states(self, rates=None) -> np.ndarray:
        """Boolean mask of states from which a reaction with positive rate leaves the box."""
        c = self.network._resolve_rates(rates)
        mask = np.zeros(self.n_states, dtype=bool)
        for j, idx in enumerate(self._edge_reactions):
            if c[j] > 0:
                mask[idx] = True
        return mask

    # --------------------------------------------------------------- solvers
    def initial_distribution(self) -> np.ndarray:
        p0 = np.zeros(self.n_states)
        p0[self.index_of(self.x0)] = 1.0
        return p0

    def transient(self, t_eval, rates=None, p0=None, method: str = "auto"):
        """Distribution at each time in ``t_eval`` (sink boundary).

        Returns ``(P, error)`` with ``P`` of shape (len(t_eval), n_states) and
        ``error[i] = 1 - P[i].sum()``, the probability that has left the box.

        Methods (all accurate to ~1e-14):

        * ``"uniformization"`` expands ``exp(A t)`` in powers of the
          non-negative matrix ``I + A / Lambda``. Every term is positive, so
          there is no cancellation, and one sequence of matrix-vector products
          serves all output times; the cost grows with ``Lambda * t``.
        * ``"dense"`` uses the scaling-and-squaring matrix exponential on a
          dense copy of ``A``; the cost grows with ``n_states^3`` but barely
          with the rates, so it wins for small, stiff systems.
        * ``"expm"`` uses scipy's sparse ``expm_multiply`` (Al-Mohy & Higham),
          kept as an independent cross-check.
        * ``"auto"`` picks uniformization or dense from an operation count.
        """
        t_eval = np.atleast_1d(np.asarray(t_eval, dtype=float))
        if np.any(np.diff(t_eval) < 0) or np.any(t_eval < 0):
            raise ValueError("t_eval must be non-decreasing and >= 0")
        A = self.generator(rates, "sink")
        p = self.initial_distribution() if p0 is None else np.asarray(p0, dtype=float).copy()
        if method == "auto":
            lam_t = float(-A.diagonal().min()) * float(t_eval.max())
            steps = lam_t + 12.0 * np.sqrt(lam_t) + 40.0
            # rough cost model in flop-equivalents: each sparse product carries
            # interpreter overhead worth ~1.5e5 flops; a dense exponential costs
            # ~20 n^3 per distinct interval. Only speed depends on the choice.
            unif_cost = steps * (1.5e5 + 2.0 * A.nnz)
            dense_cost = np.unique(np.diff(np.concatenate([[0.0], t_eval]))).size * 20.0 * self.n_states**3
            method = "dense" if dense_cost < unif_cost else "uniformization"
        if method == "uniformization":
            out = _uniformization(A, p, t_eval)
        elif method == "dense":
            from scipy.linalg import expm

            dense = A.toarray()
            out = np.empty((t_eval.size, self.n_states))
            cache = {}
            t_prev = 0.0
            for i, t in enumerate(t_eval):
                dt = t - t_prev
                if dt > 0:
                    if dt not in cache:
                        cache[dt] = expm(dense * dt)
                    p = cache[dt] @ p
                    np.maximum(p, 0.0, out=p)
                out[i] = p
                t_prev = t
        elif method == "expm":
            out = np.empty((t_eval.size, self.n_states))
            t_prev = 0.0
            for i, t in enumerate(t_eval):
                if t > t_prev:
                    p = expm_multiply(A * (t - t_prev), p)
                    np.maximum(p, 0.0, out=p)
                out[i] = p
                t_prev = t
        else:
            raise ValueError("method must be 'auto', 'uniformization', 'dense' or 'expm'")
        return out, 1.0 - out.sum(axis=1)

    def stationary(self, rates=None) -> StationaryResult:
        """Stationary distribution of the reflecting truncation.

        ``A p = 0`` has a one-dimensional null space for an irreducible chain;
        the balance equation of one reference state is redundant, so it is
        replaced by fixing that state's (unnormalised) probability to one.
        """
        A = self.generator(rates, "reflect")
        ref = int(self.index_of(self.x0))
        p = self._solve_with_reference(A, ref)
        if p is None or p[ref] < 1e-8 * p.max() or p.min() < -1e-10 * p.max():
            # The initial state is improbable, so the anchored solution spans a
            # huge dynamic range. Solve the scale-free system (normalisation
            # replaces one balance equation) to locate the bulk, then re-anchor
            # on its most probable state.
            q = self._solve_normalized(A, ref)
            p = self._solve_with_reference(A, int(np.argmax(q)))
            if p is None:
                p = q
        if p.min() < -1e-9 * p.max():
            raise np.linalg.LinAlgError("stationary solution has negative mass; is the chain irreducible?")
        np.maximum(p, 0.0, out=p)
        p /= p.sum()
        residual = float(np.abs(A @ p).sum())
        edge = self.edge_states(rates)
        return StationaryResult(p=p, boundary_mass=float(p[edge].sum()), residual=residual)

    def _solve_with_reference(self, A: sparse.csc_matrix, ref: int):
        """Fix the unnormalised probability of state ``ref`` to one; ``None`` if the solve overflows."""
        keep = np.ones(self.n_states, dtype=bool)
        keep[ref] = False
        sub = A[keep][:, keep].tocsc()
        rhs = -A[keep][:, [ref]].toarray().ravel()
        with np.errstate(all="ignore"):
            x = spsolve(sub, rhs)
        if not np.all(np.isfinite(x)):
            return None
        p = np.empty(self.n_states)
        p[keep] = x
        p[ref] = 1.0
        total = p.sum()
        if not np.isfinite(total) or total <= 0:
            return None
        return p / total

    def _solve_normalized(self, A: sparse.csc_matrix, row: int) -> np.ndarray:
        """Replace one balance equation by ``sum(p) = 1`` and solve."""
        B = A.tolil(copy=True)
        B[row, :] = np.ones(self.n_states)
        rhs = np.zeros(self.n_states)
        rhs[row] = 1.0
        p = spsolve(B.tocsc(), rhs)
        if not np.all(np.isfinite(p)):
            raise np.linalg.LinAlgError("stationary system is singular; is the chain irreducible?")
        return p

    # ------------------------------------------------------------- marginals
    def marginal(self, p, species: str) -> np.ndarray:
        """Distribution of one species' copy number, length ``bound + 1``.

        ``p`` may be one distribution (n_states,) or a stack (k, n_states).
        """
        i = self.network.index(species)
        p = np.asarray(p)
        n = self.bounds[i] + 1
        if p.ndim == 1:
            return np.bincount(self.states[:, i], weights=p, minlength=n)
        return np.stack([np.bincount(self.states[:, i], weights=row, minlength=n) for row in p])


def _uniformization(A: sparse.csc_matrix, p0: np.ndarray, times: np.ndarray, log_tol: float = -40.0) -> np.ndarray:
    """``exp(A t) p0`` for every ``t`` in ``times`` by uniformization (Jensen's method).

    With ``Lambda >= max_i |A_ii|``, ``M = I + A / Lambda`` is non-negative and
    ``exp(A t) = sum_k Poisson(k; Lambda t) M^k``. The vectors ``M^k p0`` are
    generated once and combined for all time points in a single matrix
    product with Poisson weights computed in log space. Terms stop once every
    remaining weight is below ``exp(log_tol)`` beyond the largest mode, which
    bounds the omitted probability by about ``1e-17``.
    """
    from scipy.special import gammaln

    out = np.zeros((times.size, p0.size))
    rate = float(-A.diagonal().min())
    if rate <= 0 or times.max() == 0:
        out[:] = p0
        return out
    M = (sparse.identity(A.shape[0], format="csr") + A.tocsr() / rate).tocsr()
    lam = rate * times
    positive = lam > 0
    out[~positive] = p0
    lam_pos = lam[positive]
    # last term: beyond every mode and with all weights below exp(log_tol)
    k = np.arange(int(np.ceil(lam.max() + 12.0 * np.sqrt(lam.max()) + 40)) + 1, dtype=float)
    log_w = k[:, None] * np.log(lam_pos)[None, :] - lam_pos[None, :] - gammaln(k + 1.0)[:, None]
    tail = (k > lam.max()) & (log_w.max(axis=1) < log_tol)
    n_terms = int(np.argmax(tail)) + 1 if tail.any() else k.size
    weights = np.exp(np.maximum(log_w[:n_terms], -745.0)).T  # (times, terms)
    chunk = max(1, min(n_terms, 20_000_000 // max(p0.size, 1)))
    acc = np.zeros((lam_pos.size, p0.size))
    V = np.empty((chunk, p0.size))
    v = p0.copy()
    for start in range(0, n_terms, chunk):
        stop = min(start + chunk, n_terms)
        for i in range(stop - start):
            V[i] = v
            v = M @ v
        acc += weights[:, start:stop] @ V[: stop - start]
    out[positive] = acc
    return out


def conservation_bound(network: ReactionNetwork, x0, species: str) -> int:
    """Largest copy number of ``species`` allowed by a conservation law.

    Solves ``min l . x0`` over non-negative ``l`` with ``l^T S = 0`` and
    ``l_species = 1``: any such ``l`` gives ``x_species <= l . x(t) = l . x0``.
    Raises if the species is not covered by a non-negative conservation law.
    """
    from scipy.optimize import linprog

    i = network.index(species)
    n_s = network.n_species
    a_eq = np.vstack([network.stoichiometry.T, np.eye(n_s)[i]])
    b_eq = np.concatenate([np.zeros(network.n_reactions), [1.0]])
    res = linprog(np.asarray(x0, dtype=float), A_eq=a_eq, b_eq=b_eq, bounds=[(0, None)] * n_s, method="highs")
    if res.status != 0:
        raise ValueError(f"species {species!r} is not conserved; give it an explicit bound")
    return int(np.floor(res.fun + 1e-9))


def stationary_marginal(network: ReactionNetwork, x0, species: str, bounds: dict, tol: float = 1e-12,
                        rates=None, max_doublings: int = 8):
    """Stationary distribution of ``species``, growing its bound until the boundary mass is below ``tol``.

    Returns ``(pmf, boundary_mass)``.
    """
    bounds = dict(bounds)
    for _ in range(max_doublings + 1):
        fsp = FSP(network, x0, bounds)
        res = fsp.stationary(rates)
        if res.boundary_mass < tol:
            return fsp.marginal(res.p, species), res.boundary_mass
        bounds[species] = 2 * bounds[species]
    raise RuntimeError(f"boundary mass {res.boundary_mass:.2e} still above {tol:g}")
