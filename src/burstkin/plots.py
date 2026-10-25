"""Figures (matplotlib, static PNG).

One visual system throughout: a light chart surface, hairline solid grids,
2 px lines, ringed markers, at most three categorical hues in a fixed order
(blue, orange, aqua; validated for colour-vision deficiency on all pairs),
a legend whenever two or more series share a panel, and text in ink colours
rather than series colours.
"""

from __future__ import annotations

import numpy as np

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
WASH = "#2a78d6"  # used at low alpha for intervals


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.size": 9.5,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.titlecolor": INK,
            "axes.labelcolor": INK_2,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "legend.frameon": False,
            "legend.labelcolor": INK_2,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
        }
    )
    return plt


def _style(ax, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.6)


def _save(fig, path):
    fig.savefig(path, dpi=160, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)


def _ringed(ax, x, y, color, label=None, size=5.5, **kw):
    return ax.plot(x, y, "o", color=color, markersize=size, markeredgecolor=SURFACE, markeredgewidth=1.2,
                   linestyle="none", label=label, **kw)


# ------------------------------------------------------------------ figures
def distributions(panels, path):
    """Stationary mRNA distributions: SSA histogram, closed form, and FSP (one panel per regime).

    ``panels``: list of dicts with ``title``, ``subtitle``, ``ssa`` (empirical
    pmf), ``closed`` and ``fsp`` (pmfs on the same support).
    """
    plt = _plt()
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.0))
    for ax, panel in zip(np.atleast_1d(axes), panels):
        _style(ax)
        n = np.arange(len(panel["closed"]))
        ax.bar(n, panel["ssa"], width=0.72, color=BLUE, alpha=0.35, linewidth=0, label="SSA, 20,000 cells")
        ax.plot(n, panel["closed"], color=ORANGE, linewidth=1.8, label="closed form")
        step = max(1, len(n) // 30)
        _ringed(ax, n[::step], panel["fsp"][::step], AQUA, label="master equation (FSP)", size=4.5)
        ax.set_title(panel["title"], loc="left")
        ax.text(0.98, 0.97, panel["subtitle"], transform=ax.transAxes, color=INK_2, fontsize=7.5,
                ha="right", va="top")
        ax.set_xlabel("mRNA per cell")
        ax.set_xlim(-1, len(n))
        ax.set_ylim(0, None)
    np.atleast_1d(axes)[0].set_ylabel("probability")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout()
    _save(fig, path)


def lna_scaling(omegas, err_mean, err_var, path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    _style(ax, "both")
    ax.loglog(omegas, err_mean, color=BLUE, linewidth=1.8, label="mean")
    _ringed(ax, omegas, err_mean, BLUE)
    ax.loglog(omegas, err_var, color=ORANGE, linewidth=1.8, label="variance")
    _ringed(ax, omegas, err_var, ORANGE)
    guide = err_mean[0] * omegas[0] / np.asarray(omegas)
    ax.loglog(omegas, guide * 1.6, color=MUTED, linewidth=0.8)
    ax.text(omegas[1], guide[1] * 1.9, "slope -1", color=INK_2, fontsize=8)
    ax.set_xlabel("system size  Ω")
    ax.set_ylabel("relative error of the LNA")
    ax.set_title("LNA error for 2P → ∅ shrinks as 1/Ω", loc="left")
    ax.legend(loc="lower left")
    fig.tight_layout()
    _save(fig, path)


def _value_axis(ax, values, truth):
    """Linear axis for a narrow range, log axis with plain decade labels for a wide one."""
    from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullFormatter

    lo, hi = min(values.min(), truth), max(values.max(), truth)
    fmt = FuncFormatter(lambda v, _: f"{v:g}")
    if hi / lo < 8:
        ax.xaxis.set_major_locator(MaxNLocator(4))
    else:
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=6))
        ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_formatter(fmt)


def profiles(rows, path, threshold=3.841):
    """Profile-likelihood curves; ``rows``: list of (row title, {target: Profile}, {target: truth}).

    Hollow markers are profile points where another rate sits on its bound,
    i.e. where the box constraints rather than the data shape the curve.
    """
    plt = _plt()
    from matplotlib.lines import Line2D

    targets = [("k_on", "burst frequency  k_on"), ("k_off", "inactivation  k_off"),
               ("k_syn", "transcription  k_syn"), ("burst_size", "burst size  k_syn / k_off")]
    fig, axes = plt.subplots(len(rows), len(targets), figsize=(12.4, 2.75 * len(rows) + 0.6), squeeze=False)
    ymax = threshold * 2.2
    for r, (title, profs, truth) in enumerate(rows):
        for c, (key, label) in enumerate(targets):
            ax = axes[r, c]
            _style(ax, "y")
            p = profs[key]
            _value_axis(ax, p.values, truth[key])
            lo = p.ci[0] if p.ci[0] is not None else p.values.min()
            hi = p.ci[1] if p.ci[1] is not None else p.values.max()
            ax.axvspan(lo, hi, color=WASH, alpha=0.08, linewidth=0)
            ax.axhline(threshold, color=MUTED, linewidth=0.8)
            ax.axvline(truth[key], color=ORANGE, linewidth=1.4)
            ax.plot(p.values, p.deviance, color=BLUE, linewidth=1.8, zorder=3)  # the axes clip the line
            pinned = p.nuisance_at_bound
            on_scale = p.deviance <= ymax
            free = ~pinned & on_scale
            # off-scale points held at a bound sit on the top edge: they explain cliffs in the curve
            # (and an empty unclipped line would claim a layout box at the figure origin)
            if free.any():
                ax.plot(p.values[free], p.deviance[free], "o", color=BLUE, markersize=4.5,
                        markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=4)
            if pinned.any():
                ax.plot(p.values[pinned], np.minimum(p.deviance[pinned], ymax * 0.985), "o",
                        markerfacecolor=SURFACE, markeredgecolor=BLUE, markersize=4.5, markeredgewidth=1.2,
                        zorder=4, clip_on=False)
            ax.set_ylim(-0.25, ymax)
            ax.set_xlim(*ax.get_xlim())  # freeze limits before off-scale markers can widen them
            notes = [
                ax.text(p.mle, threshold + 0.15, "95% cut-off", color=INK_2, fontsize=7.5, va="bottom", ha="center"),
                ax.text(truth[key], ymax * 0.97, " truth", color=INK_2, fontsize=7.5, va="top", clip_on=True),
            ]
            if p.ci[1] is None or p.ci[0] is None:
                notes.append(ax.text(0.97, 0.06, "interval open:\nnot identifiable", transform=ax.transAxes,
                                     ha="right", va="bottom", fontsize=7.5, color=INK_2))
            for note in notes:  # in-panel annotations must not drive the layout
                note.set_in_layout(False)
            ax.set_xlabel(label)
            if c == 0:
                ax.set_ylabel("deviance")
                ax.set_title(title, loc="left")
    handles = [
        Line2D([], [], color=BLUE, linewidth=1.8, marker="o", markersize=4.5, markeredgecolor=SURFACE, label="profile"),
        Line2D([], [], color=BLUE, linewidth=0, marker="o", markersize=4.5, markerfacecolor=SURFACE,
               label="another rate held at its bound"),
        Line2D([], [], color=ORANGE, linewidth=1.4, label="true value"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    _save(fig, path)


CLASS_COLORS = {"full": BLUE, "burst": ORANGE, "none": AQUA}
CLASS_LABELS = {
    "full": "all three rates identifiable",
    "burst": "only burst frequency and size",
    "none": "not identifiable",
}


def genome_recovery(rows, truth, path):
    """Estimated vs true burst frequency and size with 95% profile intervals, coloured by identifiability.

    ``rows``: result rows (dicts) from :mod:`burstkin.pipeline`; ``truth``:
    dict gene -> dict with ``burst_frequency`` and ``burst_size``.
    """
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    panels = [("burst_frequency", "k_on", "burst frequency (per mRNA lifetime)"),
              ("burst_size", "burst_size", "burst size (mRNA per burst)")]
    ok_rows = [r for r in rows if r["status"] == "ok"]
    for ax, (key, ci_key, label) in zip(axes, panels):
        _style(ax, "both")
        ax.set_xscale("log")
        ax.set_yscale("log")
        true_all = np.array([truth[r["gene"]][key] for r in ok_rows])
        lo_lim, hi_lim = true_all.min() / 3, true_all.max() * 3
        for cls in ("full", "burst", "none"):
            sel = [r for r in ok_rows if r["identifiability"] == cls]
            if not sel:
                continue
            x = np.array([truth[r["gene"]][key] for r in sel])
            y = np.array([r[key] for r in sel])
            lo = np.array([r[f"{ci_key}_ci_low"] if np.isfinite(r[f"{ci_key}_ci_low"]) else lo_lim for r in sel])
            hi = np.array([r[f"{ci_key}_ci_high"] if np.isfinite(r[f"{ci_key}_ci_high"]) else hi_lim for r in sel])
            ax.vlines(x, np.minimum(lo, y), np.maximum(hi, y), color=CLASS_COLORS[cls], linewidth=1.0, alpha=0.55)
            # estimates beyond the panel are pinned to its edge rather than dropped
            _ringed(ax, x, np.clip(y, lo_lim * 1.08, hi_lim / 1.08), CLASS_COLORS[cls],
                    label=f"{CLASS_LABELS[cls]} ({len(sel)})")
        ax.plot([lo_lim, hi_lim], [lo_lim, hi_lim], color=MUTED, linewidth=0.8)
        ax.set_xlim(lo_lim, hi_lim)
        ax.set_ylim(lo_lim, hi_lim)
        ax.set_xlabel(f"true {label}")
        ax.set_ylabel("estimate with 95% interval")
    axes[0].set_title("Burst frequency", loc="left")
    axes[1].set_title("Burst size", loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    _save(fig, path)


def feedback_fit(times, counts, fitted, path, truth=None):
    """Protein snapshots of the feedback model with the fitted master-equation solution at each time."""
    plt = _plt()
    k = len(times)
    cols = 3
    rows_n = int(np.ceil(k / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(3.3 * cols, 2.5 * rows_n), squeeze=False)
    for i, ax in enumerate(axes.ravel()):
        if i >= k:
            ax.set_visible(False)
            continue
        _style(ax)
        n = np.arange(fitted.shape[1])
        hist = np.bincount(counts[i], minlength=n.size)[: n.size] / max(len(counts[i]), 1)
        ax.bar(n, hist, width=0.72, color=BLUE, alpha=0.35, linewidth=0, label="SSA snapshot")
        ax.plot(n, fitted[i], color=ORANGE, linewidth=1.8, label="fitted model (FSP)")
        upper = int(np.searchsorted(np.cumsum(fitted[i]), 0.9995)) + 3
        ax.set_xlim(-1, max(upper, 10))
        ax.set_title(f"t = {times[i]:g} lifetimes", loc="left")
        if i % cols == 0:
            ax.set_ylabel("fraction of cells")
        if i >= k - cols:
            ax.set_xlabel("protein per cell")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    _save(fig, path)
