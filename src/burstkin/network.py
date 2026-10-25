"""Chemical reaction networks with stochastic mass-action kinetics.

A network is a list of species and a list of reaction channels. Each channel
has integer reactant and product stoichiometries and a *named* rate constant,
so the same topology can be evaluated at many parameter values (which is what
inference needs).

Two propensity conventions are provided:

* ``propensities`` - the stochastic (master-equation) form
  ``a_j(x) = c_j * prod_i C(x_i, r_ij)`` where ``C`` is the binomial
  coefficient, i.e. the number of distinct reactant combinations. This is what
  the SSA and the finite state projection use.
* ``macroscopic_propensities`` - the large-copy-number limit
  ``a_j(phi) = c_j * prod_i phi_i**r_ij / r_ij!`` used by the reaction rate
  equations and the linear noise approximation.

For networks whose reactions are all zeroth or first order the two agree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import factorial
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class Reaction:
    """One reaction channel ``reactants -> products`` with rate parameter ``rate``."""

    reactants: Mapping[str, int]
    products: Mapping[str, int]
    rate: str
    name: str = field(default="")

    def describe(self) -> str:
        def side(terms: Mapping[str, int]) -> str:
            parts = [f"{c} {s}" if c != 1 else s for s, c in terms.items() if c]
            return " + ".join(parts) if parts else "0"

        label = f"{self.name}: " if self.name else ""
        return f"{label}{side(self.reactants)} -> {side(self.products)}  [{self.rate}]"


class ReactionNetwork:
    """Species, reactions and default parameter values."""

    def __init__(
        self,
        species: Sequence[str],
        reactions: Sequence[Reaction],
        parameters: Mapping[str, float],
        name: str = "",
    ) -> None:
        self.species = tuple(species)
        self.reactions = tuple(reactions)
        self.parameters = {k: float(v) for k, v in parameters.items()}
        self.name = name
        if len(set(self.species)) != len(self.species):
            raise ValueError("species names must be unique")

        index = {s: i for i, s in enumerate(self.species)}
        n_s, n_r = len(self.species), len(self.reactions)
        reactant = np.zeros((n_s, n_r), dtype=np.int64)
        product = np.zeros((n_s, n_r), dtype=np.int64)
        for j, rx in enumerate(self.reactions):
            for target, side in ((reactant, rx.reactants), (product, rx.products)):
                for s, c in side.items():
                    if s not in index:
                        raise ValueError(f"reaction {j} uses unknown species {s!r}")
                    if c < 0:
                        raise ValueError("stoichiometric coefficients must be non-negative")
                    target[index[s], j] += int(c)
            if rx.rate not in self.parameters:
                raise ValueError(f"reaction {j} uses unknown parameter {rx.rate!r}")

        self._index = index
        self.reactant_orders = reactant
        self.stoichiometry = product - reactant  # (species, reactions)
        # (species, reaction, order) triples with a nonzero order
        self._order_terms = [
            (i, j, int(reactant[i, j])) for i in range(n_s) for j in range(n_r) if reactant[i, j] > 0
        ]

    # ------------------------------------------------------------------ basics
    @property
    def n_species(self) -> int:
        return len(self.species)

    @property
    def n_reactions(self) -> int:
        return len(self.reactions)

    def index(self, species: str) -> int:
        try:
            return self._index[species]
        except KeyError:
            raise KeyError(f"unknown species {species!r}; have {self.species}") from None

    def rate_vector(self, params: Mapping[str, float] | None = None) -> np.ndarray:
        """Rate constant of every reaction, with ``params`` overriding the defaults."""
        values = self.parameters if not params else {**self.parameters, **params}
        unknown = set(params or ()) - set(self.parameters)
        if unknown:
            raise KeyError(f"unknown parameters {sorted(unknown)}")
        return np.array([values[rx.rate] for rx in self.reactions], dtype=float)

    def with_parameters(self, **params: float) -> "ReactionNetwork":
        self.rate_vector(params)  # validates names
        return ReactionNetwork(self.species, self.reactions, {**self.parameters, **params}, self.name)

    def state(self, **counts: int) -> np.ndarray:
        """Build a state vector from keyword counts; unspecified species are zero."""
        x = np.zeros(self.n_species, dtype=np.int64)
        for s, c in counts.items():
            x[self.index(s)] = c
        return x

    def is_linear(self) -> bool:
        """True if every reaction is of order zero or one."""
        return bool(np.all(self.reactant_orders.sum(axis=0) <= 1))

    def describe(self) -> str:
        lines = [f"network {self.name or '(unnamed)'}: species {', '.join(self.species)}"]
        lines += ["  " + rx.describe() for rx in self.reactions]
        lines += ["  parameters: " + ", ".join(f"{k}={v:g}" for k, v in self.parameters.items())]
        return "\n".join(lines)

    # ---------------------------------------------------------- propensities
    def _resolve_rates(self, rates) -> np.ndarray:
        if rates is None:
            return self.rate_vector()
        if isinstance(rates, Mapping):
            return self.rate_vector(rates)
        rates = np.asarray(rates, dtype=float)
        if rates.shape != (self.n_reactions,):
            raise ValueError(f"expected {self.n_reactions} rates, got shape {rates.shape}")
        return rates

    def combinatorial_factors(self, x) -> np.ndarray:
        """``h_j(x) = prod_i C(x_i, r_ij)``: propensities at unit rate constants."""
        x = np.asarray(x, dtype=float)
        h = np.ones(x.shape[:-1] + (self.n_reactions,))
        for i, j, r in self._order_terms:
            xi = x[..., i]
            if r == 1:
                h[..., j] *= xi
            else:
                falling = np.ones_like(xi)
                for m in range(r):
                    falling = falling * np.maximum(xi - m, 0.0)
                h[..., j] *= falling / factorial(r)
        return h

    def propensities(self, x, rates=None) -> np.ndarray:
        """Stochastic mass-action propensities, shape ``x.shape[:-1] + (n_reactions,)``."""
        return self.combinatorial_factors(x) * self._resolve_rates(rates)

    def macroscopic_propensities(self, phi, rates=None) -> np.ndarray:
        phi = np.asarray(phi, dtype=float)
        a = np.ones(phi.shape[:-1] + (self.n_reactions,)) * self._resolve_rates(rates)
        for i, j, r in self._order_terms:
            a[..., j] *= phi[..., i] ** r / factorial(r)
        return a

    def macroscopic_jacobian(self, phi, rates=None) -> np.ndarray:
        """``d a_j / d phi_k`` of the macroscopic propensities, shape (reactions, species)."""
        phi = np.asarray(phi, dtype=float)
        rates = self._resolve_rates(rates)
        jac = np.zeros((self.n_reactions, self.n_species))
        for j in range(self.n_reactions):
            terms = [(i, r) for i, jj, r in self._order_terms if jj == j]
            for k, rk in terms:
                d = rates[j] * rk * phi[k] ** (rk - 1) / factorial(rk)
                for i, r in terms:
                    if i != k:
                        d *= phi[i] ** r / factorial(r)
                jac[j, k] = d
        return jac

    def drift(self, phi, rates=None) -> np.ndarray:
        """Right-hand side of the reaction rate equations, ``S a(phi)``."""
        return self.stoichiometry @ self.macroscopic_propensities(phi, rates)

    def __repr__(self) -> str:
        return f"ReactionNetwork({self.name!r}, species={self.species}, reactions={self.n_reactions})"
