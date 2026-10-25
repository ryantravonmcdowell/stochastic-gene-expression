"""Gene-by-gene burst-kinetics analysis of a count matrix.

For each gene: quality filter, maximum-likelihood telegraph fit, model
comparison against Poisson and negative-binomial alternatives, profile
likelihood intervals and an identifiability call. Genes are independent, so
they can be analysed in parallel processes.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from .inference import (
    ModelComparison,
    TelegraphFit,
    TelegraphLikelihood,
    classify_identifiability,
    compare_models,
    fit_telegraph,
    profile_likelihood,
)

# Both k_off and k_syn are profiled: in the bursty regime either one's interval
# can close artificially when the other reaches its bound, but not both.
DEFAULT_PROFILES = ("k_on", "k_off", "k_syn", "burst_size")


@dataclass
class GeneResult:
    gene: str
    n_cells: int
    mean: float
    fano: float
    status: str
    best_model: str = ""
    identifiability: str = ""
    fit: TelegraphFit | None = None
    profiles: dict = field(default_factory=dict)
    comparison: ModelComparison | None = None
    seconds: float = 0.0
    message: str = ""

    def row(self) -> dict:
        r = {
            "gene": self.gene,
            "status": self.status,
            "n_cells": self.n_cells,
            "mean": self.mean,
            "fano": self.fano,
            "best_model": self.best_model,
            "identifiability": self.identifiability,
        }
        f = self.fit
        r.update(
            {
                "k_on": f.k_on if f else np.nan,
                "k_off": f.k_off if f else np.nan,
                "k_syn": f.k_syn if f else np.nan,
                "burst_frequency": f.burst_frequency if f else np.nan,
                "burst_size": f.burst_size if f else np.nan,
                "loglik": f.loglik if f else np.nan,
            }
        )
        for name in DEFAULT_PROFILES:
            p = self.profiles.get(name)
            lo, hi = (p.ci if p else (None, None))
            r[f"{name}_ci_low"] = np.nan if lo is None else lo
            r[f"{name}_ci_high"] = np.nan if hi is None else hi
        if self.comparison:
            for m, v in self.comparison.bic.items():
                r[f"bic_{m}"] = v
            r["p_telegraph_vs_nbinom"] = self.comparison.lrt_pvalue["telegraph_vs_nbinom"]
        r["seconds"] = self.seconds
        r["message"] = self.message
        return r


def analyse_gene(
    counts,
    capture=None,
    gene: str = "",
    profiles=DEFAULT_PROFILES,
    min_mean: float = 0.05,
    min_nonzero: int = 10,
    bounds=None,
) -> GeneResult:
    """Fit, compare and profile one gene; never raises for a numerical failure."""
    start = time.perf_counter()
    counts = np.asarray(counts, dtype=np.int64)
    mean = float(counts.mean())
    var = float(counts.var())
    result = GeneResult(gene=gene, n_cells=counts.size, mean=mean, fano=var / mean if mean > 0 else np.nan, status="ok")
    if mean < min_mean or np.count_nonzero(counts) < min_nonzero:
        result.status = "low_expression"
        result.seconds = time.perf_counter() - start
        return result
    try:
        lik = TelegraphLikelihood(counts, capture)
        fit = fit_telegraph(counts, capture, bounds=bounds, likelihood=lik)
        result.fit = fit
        result.comparison = compare_models(counts, capture, telegraph_fit=fit)
        result.best_model = result.comparison.best
        for target in profiles:
            result.profiles[target] = profile_likelihood(lik, fit, target, bounds=bounds)
        if profiles:
            result.identifiability = classify_identifiability(result.profiles)
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
        result.status = "failed"
        result.message = f"{type(exc).__name__}: {exc}"
    result.seconds = time.perf_counter() - start
    return result


def _analyse_one(args):
    counts, capture, gene, options = args
    return analyse_gene(counts, capture, gene, **options)


def _tasks(counts, capture, genes, options, block: int = 256):
    """One task per gene, densifying ``block`` columns at a time (so sparse input stays sparse)."""
    for start in range(0, counts.shape[1], block):
        chunk = counts[:, start : start + block]
        chunk = np.asarray(chunk.toarray() if hasattr(chunk, "toarray") else chunk)
        for j in range(chunk.shape[1]):
            yield chunk[:, j].copy(), capture, genes[start + j], options


def analyse_genes(counts, capture=None, genes=None, n_jobs: int = 1, progress=None, **options) -> list:
    """Analyse every column of a cells x genes matrix (dense or scipy sparse); results keep the column order.

    ``progress`` is called as ``progress(done, total, result)`` after each gene.
    """
    if not hasattr(counts, "toarray"):
        counts = np.asarray(counts)
    n_genes = counts.shape[1]
    genes = list(genes) if genes is not None else [f"gene{j}" for j in range(n_genes)]
    results = []
    if n_jobs == 1:
        for task in _tasks(counts, capture, genes, options):
            results.append(_analyse_one(task))
            if progress:
                progress(len(results), n_genes, results[-1])
        return results
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        # submit in windows so that only a bounded number of dense columns is in flight
        pending = []
        for task in _tasks(counts, capture, genes, options):
            pending.append(pool.submit(_analyse_one, task))
            if len(pending) >= 4 * n_jobs:
                results.append(pending.pop(0).result())
                if progress:
                    progress(len(results), n_genes, results[-1])
        for fut in pending:
            results.append(fut.result())
            if progress:
                progress(len(results), n_genes, results[-1])
    return results
