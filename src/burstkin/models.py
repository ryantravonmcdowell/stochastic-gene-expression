"""Gene-expression models used throughout the package.

Time is measured in units of the product's lifetime: unless stated otherwise
the degradation rate is 1, so every other rate is "events per lifetime".
"""

from __future__ import annotations

import numpy as np

from .network import Reaction, ReactionNetwork


def birth_death(k: float = 10.0, gamma: float = 1.0) -> ReactionNetwork:
    """Constitutive expression: ``0 -> M`` at rate ``k``, ``M -> 0`` at rate ``gamma``."""
    return ReactionNetwork(
        species=("M",),
        reactions=(
            Reaction({}, {"M": 1}, "k", "synthesis"),
            Reaction({"M": 1}, {}, "gamma", "degradation"),
        ),
        parameters={"k": k, "gamma": gamma},
        name="birth-death",
    )


def telegraph(k_on: float = 0.5, k_off: float = 2.0, k_syn: float = 30.0, delta: float = 1.0) -> ReactionNetwork:
    """The two-state (telegraph) model of transcriptional bursting.

    The promoter switches between an inactive and an active state; mRNA is
    made only from the active state and decays at rate ``delta``.
    Burst frequency is ``k_on`` and mean burst size is ``k_syn / k_off``.
    """
    return ReactionNetwork(
        species=("G_off", "G_on", "M"),
        reactions=(
            Reaction({"G_off": 1}, {"G_on": 1}, "k_on", "activation"),
            Reaction({"G_on": 1}, {"G_off": 1}, "k_off", "inactivation"),
            Reaction({"G_on": 1}, {"G_on": 1, "M": 1}, "k_syn", "transcription"),
            Reaction({"M": 1}, {}, "delta", "decay"),
        ),
        parameters={"k_on": k_on, "k_off": k_off, "k_syn": k_syn, "delta": delta},
        name="telegraph",
    )


def telegraph_initial_state(active: bool = False, mrna: int = 0) -> np.ndarray:
    """State vector ``(G_off, G_on, M)`` for a single gene copy."""
    return np.array([0, 1, mrna] if active else [1, 0, mrna], dtype=np.int64)


def annihilation(k: float = 10.0, c: float = 0.1) -> ReactionNetwork:
    """``0 -> P`` at rate ``k`` and ``P + P -> 0`` with stochastic rate constant ``c``.

    The simplest network with a second-order reaction, used to show where the
    linear noise approximation stops being exact. With ``k = Omega`` and
    ``c = 1 / Omega`` the system size is ``Omega``.
    """
    return ReactionNetwork(
        species=("P",),
        reactions=(
            Reaction({}, {"P": 1}, "k", "synthesis"),
            Reaction({"P": 2}, {}, "c", "annihilation"),
        ),
        parameters={"k": k, "c": c},
        name="annihilation",
    )


def autoregulation(
    rho_u: float = 60.0,
    rho_b: float = 4.0,
    sigma_b: float = 0.4,
    sigma_u: float = 3.0,
    delta: float = 1.0,
) -> ReactionNetwork:
    """A gene that represses itself.

    The free promoter ``G`` makes protein at rate ``rho_u``. A protein can bind
    it (``G + P -> G_b``, rate ``sigma_b`` per protein), after which output
    drops to the leak rate ``rho_b`` until the protein unbinds (``sigma_u``).
    The binding step is bimolecular, so the network is nonlinear and has no
    simple closed-form distribution; it is solved with the finite state
    projection and fitted to time-course snapshots.
    """
    return ReactionNetwork(
        species=("G", "G_b", "P"),
        reactions=(
            Reaction({"G": 1}, {"G": 1, "P": 1}, "rho_u", "free synthesis"),
            Reaction({"G_b": 1}, {"G_b": 1, "P": 1}, "rho_b", "leaky synthesis"),
            Reaction({"G": 1, "P": 1}, {"G_b": 1}, "sigma_b", "binding"),
            Reaction({"G_b": 1}, {"G": 1, "P": 1}, "sigma_u", "unbinding"),
            Reaction({"P": 1}, {}, "delta", "decay"),
        ),
        parameters={"rho_u": rho_u, "rho_b": rho_b, "sigma_b": sigma_b, "sigma_u": sigma_u, "delta": delta},
        name="autoregulation",
    )


def autoregulation_initial_state(protein: int = 0) -> np.ndarray:
    """State vector ``(G, G_b, P)`` with a free promoter."""
    return np.array([1, 0, protein], dtype=np.int64)
