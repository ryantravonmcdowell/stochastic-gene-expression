"""Command-line interface.

    burstkin validate [--full]           run the cross-checks
    burstkin simulate -o counts.csv      synthetic scRNA-seq experiment with known kinetics
    burstkin fit counts.csv -o out.csv   infer burst kinetics for every gene
    burstkin gene --k-on .5 ...          simulate one gene mechanistically and recover it
    burstkin demo -o demo_output         simulate, fit and plot a small experiment
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np


def _print(*args):
    print(*args, flush=True)


# ------------------------------------------------------------------ commands
def cmd_validate(args) -> int:
    from .validate import format_table, run_checks

    _print("running cross-checks" + (" (full: includes coverage study, several minutes)" if args.full else ""))
    checks = run_checks(full=args.full, seed=args.seed, progress=lambda c: _print(
        f"  {'PASS' if c.passed else 'FAIL'}  {c.name}  ({c.seconds:.1f}s)"))
    _print()
    _print(format_table(checks))
    return 0 if all(c.passed for c in checks) else 1


def cmd_simulate(args) -> int:
    from .scdata import simulate_experiment, write_counts, write_table

    exp = simulate_experiment(args.cells, args.genes, args.capture, args.capture_cv, rng=args.seed)
    write_counts(args.out, exp.counts, exp.genes, exp.cells)
    if args.truth:
        write_table(args.truth, exp.truth_rows())
    if args.capture_out:
        write_table(args.capture_out, [{"cell": c, "capture": s} for c, s in zip(exp.cells, exp.capture)])
    _print(f"wrote {args.cells} cells x {args.genes} genes to {args.out}")
    return 0


def _progress(done, total, result):
    status = result.status if result.status != "ok" else (result.identifiability or result.best_model)
    _print(f"  [{done:>4}/{total}] {result.gene:<14} {status:<15} {result.seconds:6.1f}s")


def cmd_fit(args) -> int:
    from .pipeline import analyse_genes
    from .scdata import estimate_capture, read_counts, read_table, write_table

    counts, genes, cells = read_counts(args.counts, genes_in_rows=args.genes_in_rows)
    if args.capture_file:
        table = {r["cell"]: float(r["capture"]) for r in read_table(args.capture_file)}
        capture = np.array([table[c] for c in cells])
        source = f"from {args.capture_file}"
    elif args.mean_capture > 0:
        try:
            capture = estimate_capture(counts, args.mean_capture)
        except ValueError as exc:
            _print(f"error: {exc}. Remove empty cells first, or pass --capture-file, or --mean-capture 0.")
            return 2
        source = f"from library size, mean {args.mean_capture:g}"
    else:
        capture = None
        source = "none (counts treated as complete)"
    profiles = () if args.profiles == "none" else tuple(p for p in args.profiles.split(",") if p)
    _print(f"{counts.shape[0]} cells x {counts.shape[1]} genes; capture efficiency {source}")
    t0 = time.perf_counter()
    results = analyse_genes(counts, capture, genes, n_jobs=args.jobs, progress=_progress,
                            profiles=profiles, min_mean=args.min_mean)
    write_table(args.out, [r.row() for r in results])
    _print(f"wrote {args.out} ({time.perf_counter() - t0:.0f}s)")
    return 0


def cmd_gene(args) -> int:
    from . import models
    from .analytic import telegraph_moments
    from .pipeline import analyse_gene
    from .scdata import beta_capture
    from .ssa import binomial_capture, simulate_stationary

    rng = np.random.default_rng(args.seed)
    net = models.telegraph(args.k_on, args.k_off, args.k_syn)
    burn_in = 10.0 + 10.0 / (args.k_on + args.k_off)
    _print(f"simulating {args.cells} cells with the SSA (burn-in {burn_in:.1f} mRNA lifetimes)")
    mrna = simulate_stationary(net, models.telegraph_initial_state(), args.cells, burn_in, rng=rng)[:, 2]
    if args.capture < 1:
        capture = beta_capture(args.cells, args.capture, 0.25, rng)
        counts = binomial_capture(mrna, capture, rng)
    else:
        capture, counts = None, mrna
    res = analyse_gene(counts, capture, gene="simulated")
    if res.status != "ok":
        _print(f"gene not fitted ({res.status}) {res.message}".rstrip())
        return 1
    m = telegraph_moments(args.k_on, args.k_off, args.k_syn)
    truth = {"k_on": args.k_on, "k_off": args.k_off, "k_syn": args.k_syn, "burst_size": m.burst_size}
    _print(f"mean count {res.mean:.2f}, Fano factor {res.fano:.2f}; best model by BIC: {res.best_model}")
    _print(f"{'quantity':<12}{'truth':>10}{'estimate':>11}   95% profile interval")
    est = {"k_on": res.fit.k_on, "k_off": res.fit.k_off, "k_syn": res.fit.k_syn, "burst_size": res.fit.burst_size}
    for key in ("k_on", "k_off", "k_syn", "burst_size"):
        p = res.profiles.get(key)
        ci = "-" if p is None else "[{}, {}]".format(*("open" if v is None else f"{v:.3g}" for v in p.ci))
        _print(f"{key:<12}{truth[key]:>10.3g}{est[key]:>11.3g}   {ci}")
    _print(f"identifiability: {res.identifiability}")
    return 0


def cmd_demo(args) -> int:
    from . import plots
    from .pipeline import analyse_genes
    from .scdata import estimate_capture, simulate_experiment, write_counts, write_table

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    exp = simulate_experiment(args.cells, args.genes, 0.3, 0.25, rng=args.seed)
    write_counts(out / "counts.csv", exp.counts, exp.genes, exp.cells)
    write_table(out / "truth.csv", exp.truth_rows())
    _print(f"simulated {args.cells} cells x {args.genes} genes -> {out / 'counts.csv'}")
    if args.true_capture:
        capture, note = exp.capture, "true values"
    else:
        try:
            capture = estimate_capture(exp.counts, 0.3)
            note = f"estimated from library size (r = {np.corrcoef(capture, exp.capture)[0, 1]:.2f} with truth)"
        except ValueError:  # with few genes some cells have no counts at all
            capture, note = exp.capture, "true values (too few genes to estimate from library size)"
    _print(f"capture efficiency: {note}")
    t0 = time.perf_counter()
    results = analyse_genes(exp.counts, capture, exp.genes, n_jobs=args.jobs, progress=_progress)
    rows = [res.row() for res in results]
    write_table(out / "results.csv", rows)
    truth = {t["gene"]: t for t in exp.truth_rows()}
    plots.genome_recovery(rows, truth, out / "genome_recovery.png")
    _print(f"fitted in {time.perf_counter() - t0:.0f}s -> {out / 'results.csv'}, {out / 'genome_recovery.png'}")
    _print(summarize(rows, truth))
    return 0


def summarize(rows, truth) -> str:
    """Accuracy and interval coverage of a demo run, by identifiability class."""
    lines = []
    ok = [r for r in rows if r["status"] == "ok"]
    classes = {c: [r for r in ok if r["identifiability"] == c] for c in ("full", "burst", "none")}
    lines.append("identifiability: " + ", ".join(f"{c} {len(v)}" for c, v in classes.items())
                 + f"; skipped {len(rows) - len(ok)}")
    for key, ci in (("burst_frequency", "k_on"), ("burst_size", "burst_size")):
        usable = [r for r in ok if r["identifiability"] in ("full", "burst")]
        if not usable:
            continue
        err = np.array([abs(np.log(r[key] / truth[r["gene"]][key])) for r in usable])
        covered = sum(
            (not np.isfinite(r[f"{ci}_ci_low"]) or r[f"{ci}_ci_low"] <= truth[r["gene"]][key])
            and (not np.isfinite(r[f"{ci}_ci_high"]) or truth[r["gene"]][key] <= r[f"{ci}_ci_high"])
            for r in usable
        )
        lines.append(f"{key}: median fold error {np.exp(np.median(err)):.2f}x over {len(usable)} identifiable genes; "
                     f"95% interval contains the truth for {covered}/{len(usable)}")
    return "\n".join(lines)


# --------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="burstkin", description="Stochastic gene expression and burst-kinetics inference.")
    sub = parser.add_subparsers(dest="command", required=True)

    v = sub.add_parser("validate", help="run the built-in cross-checks")
    v.add_argument("--full", action="store_true", help="also run the coverage study and feedback-model recovery")
    v.add_argument("--seed", type=int, default=2026)
    v.set_defaults(func=cmd_validate)

    s = sub.add_parser("simulate", help="simulate a single-cell experiment with known burst kinetics")
    s.add_argument("--cells", type=int, default=600)
    s.add_argument("--genes", type=int, default=40)
    s.add_argument("--capture", type=float, default=0.3, help="mean capture efficiency")
    s.add_argument("--capture-cv", type=float, default=0.25, help="coefficient of variation of capture efficiency")
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("-o", "--out", default="counts.csv")
    s.add_argument("--truth", help="also write the true kinetics to this CSV")
    s.add_argument("--capture-out", help="also write the true capture efficiencies to this CSV")
    s.set_defaults(func=cmd_simulate)

    f = sub.add_parser("fit", help="infer burst kinetics for every gene in a count matrix")
    f.add_argument("counts", help="CSV/TSV (cells as rows) or MatrixMarket .mtx")
    f.add_argument("--genes-in-rows", action="store_true", help="matrix is genes x cells (10x .mtx files are)")
    f.add_argument("--capture-file", help="CSV with columns cell,capture")
    f.add_argument("--mean-capture", type=float, default=0.1,
                   help="assumed mean capture efficiency when estimating it from library size; 0 disables capture")
    f.add_argument("--profiles", default="k_on,k_off,k_syn,burst_size",
                   help="comma-separated quantities to profile, or 'none'")
    f.add_argument("--min-mean", type=float, default=0.05, help="skip genes with a lower mean count")
    f.add_argument("-j", "--jobs", type=int, default=1, help="parallel worker processes")
    f.add_argument("-o", "--out", default="burst_kinetics.csv")
    f.set_defaults(func=cmd_fit)

    g = sub.add_parser("gene", help="simulate one gene with the SSA and recover its kinetics")
    g.add_argument("--k-on", type=float, default=0.5)
    g.add_argument("--k-off", type=float, default=2.0)
    g.add_argument("--k-syn", type=float, default=60.0)
    g.add_argument("--cells", type=int, default=1000)
    g.add_argument("--capture", type=float, default=0.3, help="mean capture efficiency (1 = perfect capture)")
    g.add_argument("--seed", type=int, default=0)
    g.set_defaults(func=cmd_gene)

    d = sub.add_parser("demo", help="simulate, fit and plot a small genome-wide experiment")
    d.add_argument("-o", "--out", default="demo_output")
    d.add_argument("--cells", type=int, default=600)
    d.add_argument("--genes", type=int, default=40)
    d.add_argument("--seed", type=int, default=7)
    d.add_argument("--true-capture", action="store_true", help="use the true capture efficiencies instead of estimating them")
    d.add_argument("-j", "--jobs", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    d.set_defaults(func=cmd_demo)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
