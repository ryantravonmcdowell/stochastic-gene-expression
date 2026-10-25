"""burstkin: stochastic gene expression and transcriptional burst kinetics.

Exact simulation, master-equation solvers, the linear noise approximation and
likelihood-based inference of burst frequency and burst size from
single-cell counts, each checked against the others.
"""

from . import analytic, fsp, inference, lna, models, network, ssa
from .analytic import telegraph_logpmf, telegraph_moments, telegraph_pmf
from .fsp import FSP
from .inference import TelegraphLikelihood, compare_models, fit_telegraph, profile_likelihood
from .lna import lna_stationary
from .models import annihilation, autoregulation, birth_death, telegraph
from .network import Reaction, ReactionNetwork
from .ssa import simulate, simulate_stationary

__version__ = "0.1.0"

__all__ = [
    "FSP",
    "Reaction",
    "ReactionNetwork",
    "TelegraphLikelihood",
    "analytic",
    "annihilation",
    "autoregulation",
    "birth_death",
    "compare_models",
    "fit_telegraph",
    "fsp",
    "inference",
    "lna",
    "lna_stationary",
    "models",
    "network",
    "profile_likelihood",
    "simulate",
    "simulate_stationary",
    "ssa",
    "telegraph",
    "telegraph_logpmf",
    "telegraph_moments",
    "telegraph_pmf",
]
