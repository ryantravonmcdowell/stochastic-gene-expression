"""Regenerate every figure in docs/images and the numbers quoted in the README.

    python scripts/make_figures.py            (about 25 minutes on a laptop)
    python scripts/make_figures.py profiles   (one step: distributions, lna, profiles,
                                               feedback, genome, coverage)

Writes docs/images/*.png and docs/results.json. All randomness is seeded.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from burstkin import analytic as an
from burstkin import models, plots
from burstkin.cli import summarize
from burstkin.fsp import FSP
from burstkin.fsp_inference import Snapshots, SnapshotLikelihood
from burstkin.inference import TelegraphLikelihood, classify_identifiability, fit_telegraph, profile_likelihood
from burstkin.lna import lna_stationary
from burstkin.pipeline import analyse_genes
from burstkin.scdata import beta_capture, estimate_capture, simulate_experiment
from burstkin.ssa import simulate, simulate_stationary
from burstkin.validate import chisquare_gof, coverage_study

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "docs" / "images"


def log(msg):
    print(msg, flush=True)


def distributions(results):
    regimes = [
        ("Bursty", "k_on 0.4, k_off 8, k_syn 160", (0.4, 8.0, 160.0), 90),
        ("Bimodal", "k_on 0.3, k_off 0.2, k_syn 60", (0.3, 0.2, 60.0), 100),
        ("Near-constitutive", "k_on 8, k_off 1, k_syn 40", (8.0, 1.0, 40.0), 80),
    ]
    rng = np.random.default_rng(1)
    panels = []
    for title, subtitle, k, n_max in regimes:
        net = models.telegraph(*k)
        burn_in = 10.0 + 10.0 / (k[0] + k[1])
        x = simulate_stationary(net, models.telegraph_initial_state(), 20_000, burn_in, rng=rng)[:, 2]
        fsp = FSP(net, models.telegraph_initial_state(), {"M": 3 * n_max})
        p_fsp = fsp.marginal(fsp.stationary().p, "M")[: n_max + 1]
        closed = an.telegraph_pmf(n_max, *k)
        ssa = np.bincount(np.minimum(x, n_max), minlength=n_max + 1) / x.size
        panels.append({"title": title, "subtitle": subtitle, "ssa": ssa, "closed": closed, "fsp": p_fsp})
        results.setdefault("distributions", {})[title] = {
            "max_abs_fsp_vs_closed": float(np.abs(p_fsp - closed).max()),
            "ssa_chisquare_p": chisquare_gof(x, an.telegraph_pmf(3 * n_max, *k))[2],
        }
    plots.distributions(panels, IMAGES / "distributions.png")


def lna(results):
    omegas = np.array([5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0])
    err_mean, err_var = [], []
    for omega in omegas:
        net = models.annihilation(k=omega, c=1.0 / omega)
        fsp = FSP(net, [0], {"P": int(omega + 20 * np.sqrt(omega) + 40)})
        p = fsp.marginal(fsp.stationary().p, "P")
        n = np.arange(p.size)
        mean = (n * p).sum()
        var = ((n - mean) ** 2 * p).sum()
        res = lna_stationary(net, [0])
        err_mean.append(abs(res.mean[0] / mean - 1))
        err_var.append(abs(res.covariance[0, 0] / var - 1))
    plots.lna_scaling(omegas, np.array(err_mean), np.array(err_var), IMAGES / "lna_scaling.png")
    results["lna_slope_mean"] = float(np.polyfit(np.log(omegas), np.log(err_mean), 1)[0])


def profiles(results):
    rng = np.random.default_rng(3)
    rows = []
    for title, k in (("Slow switching: every rate identifiable", (0.2, 0.3, 80.0)),
                     ("Short bursts: only frequency and size", (0.3, 30.0, 600.0))):
        s = beta_capture(1000, 0.3, 0.25, rng)
        counts = an.sample_telegraph(*k, 1000, rng, capture=s)
        lik = TelegraphLikelihood(counts, s)
        fit = fit_telegraph(counts, s, likelihood=lik)
        profs = {t: profile_likelihood(lik, fit, t, grid_points=15) for t in ("k_on", "k_off", "k_syn", "burst_size")}
        truth = {"k_on": k[0], "k_off": k[1], "k_syn": k[2], "burst_size": k[2] / k[1]}
        rows.append((title, profs, truth))
        results.setdefault("profiles", {})[title] = {
            t: {"mle": p.mle, "ci": p.ci, "truth": truth[t]} for t, p in profs.items()
        } | {"class": classify_identifiability(profs)}
        log(f"  {title}: {classify_identifiability(profs)}")
    plots.profiles(rows, IMAGES / "profiles.png")


def feedback(results):
    net, x0 = models.autoregulation(), models.autoregulation_initial_state()
    times = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    X = simulate(net, x0, times, n_cells=1000, rng=np.random.default_rng(5))
    data = Snapshots(times, [X[:, i, 2] for i in range(times.size)])
    free = ("rho_u", "rho_b", "sigma_b", "sigma_u")
    lik = SnapshotLikelihood(net, x0, "P", data, free, {"P": 100})
    t0 = time.perf_counter()
    fit = lik.fit([{"rho_u": 30.0, "rho_b": 2.0, "sigma_b": 1.0, "sigma_u": 1.0},
                   {"rho_u": 100.0, "rho_b": 10.0, "sigma_b": 0.1, "sigma_u": 10.0}])
    fitted = lik.marginals(np.log([fit.params[p] for p in free]))
    plots.feedback_fit(times, data.counts, fitted, IMAGES / "feedback_fit.png")
    results["feedback"] = {
        "seconds": time.perf_counter() - t0,
        "truncation_error": fit.truncation_error,
        "params": {p: {"truth": net.parameters[p], "estimate": fit.params[p], "ci95": fit.interval(p)} for p in free},
    }


def coverage(results):
    cov = coverage_study(replicates=200)
    results["coverage"] = {"replicates": 200, "n_cells": 500, **cov}
    log(f"  coverage {cov}")


def genome(results):
    exp = simulate_experiment(600, 40, 0.3, 0.25, rng=7)
    capture = estimate_capture(exp.counts, 0.3)
    t0 = time.perf_counter()
    res = analyse_genes(exp.counts, capture, exp.genes, n_jobs=4)
    rows = [r.row() for r in res]
    truth = {t["gene"]: t for t in exp.truth_rows()}
    plots.genome_recovery(rows, truth, IMAGES / "genome_recovery.png")
    summary = summarize(rows, truth)
    log(summary)
    results["genome"] = {
        "seconds": time.perf_counter() - t0,
        "capture_correlation": float(np.corrcoef(capture, exp.capture)[0, 1]),
        "summary": summary,
        "classes": {c: sum(r["identifiability"] == c for r in rows) for c in ("full", "burst", "none")},
        "rows": rows,
        "truth": truth,
    }


def main(which=None):
    IMAGES.mkdir(parents=True, exist_ok=True)
    out = ROOT / "docs" / "results.json"
    steps = {"distributions": distributions, "lna": lna, "profiles": profiles, "feedback": feedback,
             "genome": genome, "coverage": coverage}
    for name, step in steps.items():
        if which and name not in which:
            continue
        t0 = time.perf_counter()
        log(f"{name} ...")
        produced = {}
        step(produced)
        # merge into the file as it is now, so separate runs of different steps don't erase each other
        results = json.loads(out.read_text()) if out.exists() else {}
        results.update(produced)
        out.write_text(json.dumps(results, indent=2, default=lambda o: None if o is None else float(o)))
        log(f"{name} done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
