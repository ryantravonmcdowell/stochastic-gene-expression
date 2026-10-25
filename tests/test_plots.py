"""Every figure renders from small inputs (the published figures come from scripts/make_figures.py)."""

import numpy as np

from burstkin import analytic as an
from burstkin import plots
from burstkin.inference import TelegraphLikelihood, fit_telegraph, profile_likelihood


def _is_png(path):
    return path.exists() and path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_distribution_and_lna_figures(tmp_path):
    pmf = an.telegraph_pmf(40, 0.5, 2.0, 30.0)
    panel = {"title": "t", "subtitle": "s", "ssa": pmf, "closed": pmf, "fsp": pmf}
    plots.distributions([panel, panel], tmp_path / "d.png")
    plots.lna_scaling(np.array([10.0, 20.0, 40.0]), np.array([1e-2, 5e-3, 2.5e-3]), np.array([8e-3, 4e-3, 2e-3]),
                      tmp_path / "l.png")
    assert _is_png(tmp_path / "d.png") and _is_png(tmp_path / "l.png")


def test_profile_and_recovery_figures(tmp_path, rng):
    counts = an.sample_telegraph(0.5, 1.0, 30.0, 300, rng)
    lik = TelegraphLikelihood(counts)
    fit = fit_telegraph(counts, likelihood=lik)
    profs = {t: profile_likelihood(lik, fit, t) for t in ("k_on", "k_off", "k_syn", "burst_size")}
    truth = {"k_on": 0.5, "k_off": 1.0, "k_syn": 30.0, "burst_size": 30.0}
    plots.profiles([("gene", profs, truth)], tmp_path / "p.png")

    row = {"gene": "g", "status": "ok", "identifiability": "full", "burst_frequency": fit.k_on,
           "burst_size": fit.burst_size, "k_on_ci_low": np.nan, "k_on_ci_high": 1.0,
           "burst_size_ci_low": 10.0, "burst_size_ci_high": np.nan}
    plots.genome_recovery([row], {"g": {"burst_frequency": 0.5, "burst_size": 30.0}}, tmp_path / "g.png")
    assert _is_png(tmp_path / "p.png") and _is_png(tmp_path / "g.png")


def test_feedback_figure(tmp_path, rng):
    times = [0.5, 1.0]
    fitted = np.vstack([an.birth_death_pmf(40, t, 20.0) for t in times])
    counts = [rng.poisson(20 * (1 - np.exp(-t)), 200) for t in times]
    plots.feedback_fit(times, counts, fitted, tmp_path / "f.png")
    assert _is_png(tmp_path / "f.png")
