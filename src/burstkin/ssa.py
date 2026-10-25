"""Exact stochastic simulation (Gillespie's direct method), vectorised over cells.

Every cell is an independent realisation of the same network. One loop
iteration advances every unfinished cell by one reaction event, so the cost is
set by the busiest cell while the numpy work is shared across all of them.
"""

from __future__ import annotations

import numpy as np

from .network import ReactionNetwork


def simulate(
    network: ReactionNetwork,
    x0,
    t_obs,
    n_cells: int = 1,
    params=None,
    rng: np.random.Generator | int | None = None,
    max_events: int = 50_000_000,
) -> np.ndarray:
    """Simulate ``n_cells`` trajectories and record their states at ``t_obs``.

    Parameters
    ----------
    network : reaction network to simulate.
    x0 : initial state, shape (n_species,), or (n_cells, n_species).
    t_obs : non-decreasing observation times (>= 0).
    params : optional parameter overrides (mapping) or a full rate vector.
    rng : numpy Generator or seed.
    max_events : safety limit on the total number of loop iterations.

    Returns
    -------
    states : int array, shape (n_cells, len(t_obs), n_species).
    """
    rng = np.random.default_rng(rng)
    t_obs = np.atleast_1d(np.asarray(t_obs, dtype=float))
    if t_obs.ndim != 1 or np.any(np.diff(t_obs) < 0) or np.any(t_obs < 0):
        raise ValueError("t_obs must be a non-decreasing 1-d array of times >= 0")
    rates = network._resolve_rates(params)
    change = network.stoichiometry.T.astype(np.int64)  # (reactions, species)
    n_obs, n_species, n_rx = t_obs.size, network.n_species, network.n_reactions

    x0 = np.asarray(x0, dtype=np.int64)
    state = np.broadcast_to(x0, (n_cells, n_species)).copy()
    if np.any(state < 0):
        raise ValueError("initial copy numbers must be non-negative")
    time = np.zeros(n_cells)
    next_obs = np.zeros(n_cells, dtype=np.intp)
    out = np.empty((n_cells, n_obs, n_species), dtype=np.int64)
    active = np.arange(n_cells)

    for _ in range(max_events):
        if active.size == 0:
            return out
        a = network.combinatorial_factors(state[active]) * rates
        cum = np.cumsum(a, axis=1)
        total = cum[:, -1]
        with np.errstate(divide="ignore"):
            dt = rng.standard_exponential(active.size) / total  # total == 0 -> inf
        t_new = time[active] + dt

        # The state is constant on [time, t_new): record every observation in it.
        while True:
            k = next_obs[active]
            pending = k < n_obs
            pending[pending] = t_obs[k[pending]] < t_new[pending]
            if not pending.any():
                break
            cells = active[pending]
            out[cells, k[pending]] = state[cells]
            next_obs[cells] += 1

        keep = next_obs[active] < n_obs
        active, cum, total, t_new = active[keep], cum[keep], total[keep], t_new[keep]
        if active.size == 0:
            return out
        u = rng.random(active.size) * total
        j = np.minimum((cum <= u[:, None]).sum(axis=1), n_rx - 1)
        state[active] += change[j]
        time[active] = t_new
    raise RuntimeError(f"simulation did not finish within {max_events} events")


def simulate_stationary(
    network: ReactionNetwork,
    x0,
    n_cells: int,
    burn_in: float,
    params=None,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """States of ``n_cells`` independent cells after running each for ``burn_in``.

    ``burn_in`` should be several times the slowest relaxation time of the
    network (for the telegraph model, several multiples of ``1 / delta`` and
    of ``1 / (k_on + k_off)``).
    """
    return simulate(network, x0, [burn_in], n_cells=n_cells, params=params, rng=rng)[:, 0, :]


def binomial_capture(counts, capture, rng: np.random.Generator | int | None = None) -> np.ndarray:
    """Keep each molecule independently with the cell's capture probability.

    This is the standard model of molecule loss in single-cell RNA sequencing:
    a cell with ``m`` transcripts and efficiency ``s`` yields ``Binomial(m, s)``.
    """
    rng = np.random.default_rng(rng)
    counts = np.asarray(counts, dtype=np.int64)
    capture = np.asarray(capture, dtype=float)
    if np.any((capture < 0) | (capture > 1)):
        raise ValueError("capture efficiencies must lie in [0, 1]")
    return rng.binomial(counts, np.broadcast_to(capture, counts.shape))
