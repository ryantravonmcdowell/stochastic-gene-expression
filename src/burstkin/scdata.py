"""Single-cell count matrices: simulated experiments, capture efficiency and file I/O.

Matrices are stored cells x genes. In a droplet scRNA-seq experiment each
transcript in cell ``i`` is captured with probability ``s_i`` (typically 5-30%),
so the observed count of a gene is a binomial thinning of its true copy number.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .analytic import sample_telegraph


@dataclass
class Experiment:
    """A simulated experiment with known ground truth."""

    counts: np.ndarray  # (cells, genes)
    capture: np.ndarray  # (cells,) true capture efficiencies
    genes: list
    truth: dict  # name -> (genes,) array of k_on, k_off, k_syn

    @property
    def cells(self) -> list:
        return [f"cell{i:05d}" for i in range(self.counts.shape[0])]

    def truth_rows(self) -> list:
        return [
            {
                "gene": g,
                "k_on": self.truth["k_on"][j],
                "k_off": self.truth["k_off"][j],
                "k_syn": self.truth["k_syn"][j],
                "burst_frequency": self.truth["k_on"][j],
                "burst_size": self.truth["k_syn"][j] / self.truth["k_off"][j],
            }
            for j, g in enumerate(self.genes)
        ]


def draw_kinetics(
    n_genes: int,
    rng=None,
    k_on=(0.05, 3.0),
    k_off=(0.5, 40.0),
    burst_size=(2.0, 80.0),
) -> dict:
    """Log-uniform burst kinetics; ``k_syn = burst_size * k_off``.

    The default ranges span the burst frequencies (~0.05-3 per mRNA lifetime)
    and burst sizes (a few to ~100 transcripts) reported for mammalian genes.
    """
    rng = np.random.default_rng(rng)

    def log_uniform(lo_hi):
        lo, hi = np.log(lo_hi)
        return np.exp(rng.uniform(lo, hi, n_genes))

    on, off, size = log_uniform(k_on), log_uniform(k_off), log_uniform(burst_size)
    return {"k_on": on, "k_off": off, "k_syn": size * off}


def beta_capture(n_cells: int, mean: float = 0.3, cv: float = 0.25, rng=None) -> np.ndarray:
    """Capture efficiencies from a Beta distribution with the given mean and coefficient of variation."""
    rng = np.random.default_rng(rng)
    var = (cv * mean) ** 2
    if not 0 < var < mean * (1 - mean):
        raise ValueError("cv too large for this mean")
    nu = mean * (1 - mean) / var - 1.0
    return rng.beta(mean * nu, (1 - mean) * nu, n_cells)


def simulate_experiment(
    n_cells: int = 600,
    n_genes: int = 40,
    mean_capture: float = 0.3,
    capture_cv: float = 0.25,
    rng=None,
    kinetics: dict | None = None,
) -> Experiment:
    """Counts of independent telegraph-model genes sampled with per-cell capture.

    Each cell's mRNA numbers are drawn from the exact stationary distribution
    and thinned binomially with that cell's efficiency (equivalently, Poisson
    with rate ``k_syn * s_i * p``).
    """
    rng = np.random.default_rng(rng)
    kin = kinetics if kinetics is not None else draw_kinetics(n_genes, rng)
    n_genes = len(kin["k_on"])
    capture = beta_capture(n_cells, mean_capture, capture_cv, rng)
    counts = np.empty((n_cells, n_genes), dtype=np.int64)
    for j in range(n_genes):
        counts[:, j] = sample_telegraph(kin["k_on"][j], kin["k_off"][j], kin["k_syn"][j], n_cells, rng, capture)
    genes = [f"gene{j:03d}" for j in range(n_genes)]
    return Experiment(counts=counts, capture=capture, genes=genes, truth={k: np.asarray(v) for k, v in kin.items()})


def estimate_capture(counts, mean_capture: float) -> np.ndarray:
    """Per-cell capture efficiency from library size, scaled to an assumed mean.

    Library size is proportional to capture efficiency only on average over
    many genes, and the absolute scale is not identifiable from counts alone.
    A wrong ``mean_capture`` rescales every ``k_syn`` and burst size by the same
    factor but leaves ``k_on`` and ``k_off`` unchanged (see the tests).
    """
    library = np.asarray(counts.sum(axis=1), dtype=float).ravel()  # dense or sparse
    if np.any(library <= 0):
        raise ValueError("every cell needs at least one count to estimate its capture efficiency")
    return np.clip(mean_capture * library / library.mean(), 1e-6, 1.0)


# ------------------------------------------------------------------ file I/O
def read_counts(path, genes_in_rows: bool = False):
    """Read a count matrix; returns ``(counts cells x genes, genes, cells)``.

    Supports delimited text (``.csv``/``.tsv``: header row of column names,
    first column of row names; returned dense) and MatrixMarket ``.mtx`` files
    with optional ``genes.tsv``/``features.tsv`` and ``barcodes.tsv`` alongside
    (returned as a sparse CSC matrix). 10x-style ``.mtx`` files are genes x
    cells, so pass ``genes_in_rows=True`` for them.
    """
    path = Path(path)
    if path.suffix == ".mtx" or path.name.endswith(".mtx.gz"):
        import warnings

        from scipy import sparse
        from scipy.io import mmread

        with warnings.catch_warnings():  # scipy >= 1.15 warns about its sparse return type
            warnings.simplefilter("ignore", DeprecationWarning)
            matrix = sparse.csc_matrix(mmread(str(path)))
        rows = _side_names(path.parent, ("genes.tsv", "features.tsv") if genes_in_rows else ("barcodes.tsv",), matrix.shape[0], "row")
        cols = _side_names(path.parent, ("barcodes.tsv",) if genes_in_rows else ("genes.tsv", "features.tsv"), matrix.shape[1], "col")
        if np.any(matrix.data < 0) or np.any(matrix.data != np.round(matrix.data)):
            raise ValueError("counts must be non-negative integers")
        matrix = matrix.astype(np.int64)
        # kept sparse (cells x genes, column-compressed): genes are densified one block at a time
        if genes_in_rows:
            return matrix.T.tocsc(), list(rows), list(cols)
        return matrix, list(cols), list(rows)
    else:
        delimiter = "\t" if path.suffix in (".tsv", ".txt") else ","
        with open(path, newline="") as fh:
            reader = csv.reader(fh, delimiter=delimiter)
            header = next(reader)
            rows, values = [], []
            for line in reader:
                if not line:
                    continue
                rows.append(line[0])
                values.append([float(v) for v in line[1:]])
        cols = header[1:]
        matrix = np.array(values)
    if np.any(matrix < 0) or np.any(matrix != np.round(matrix)):
        raise ValueError("counts must be non-negative integers")
    matrix = matrix.astype(np.int64)
    if genes_in_rows:
        return matrix.T, list(rows), list(cols)
    return matrix, list(cols), list(rows)


def _side_names(folder: Path, candidates, n: int, prefix: str) -> list:
    for name in candidates:
        f = folder / name
        if f.exists():
            with open(f) as fh:
                names = [line.split("\t")[0].strip() for line in fh if line.strip()]
            if len(names) == n:
                return names
    return [f"{prefix}{i}" for i in range(n)]


def write_counts(path, counts, genes, cells) -> None:
    """Write a cells x genes matrix as CSV with a header row and cell names."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cell", *genes])
        for name, row in zip(cells, np.asarray(counts)):
            w.writerow([name, *row.tolist()])


def write_table(path, rows: list) -> None:
    """Write a list of dicts as CSV (columns from the first row)."""
    if not rows:
        Path(path).write_text("")
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(v) for k, v in r.items()})


def read_table(path) -> list:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        return "" if not np.isfinite(v) else f"{float(v):.6g}"
    return v
