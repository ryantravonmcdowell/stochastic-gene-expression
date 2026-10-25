import numpy as np
import pytest

from burstkin.pipeline import analyse_gene, analyse_genes
from burstkin.scdata import (
    beta_capture,
    estimate_capture,
    read_counts,
    read_table,
    simulate_experiment,
    write_counts,
    write_table,
)


def test_simulated_experiment_is_reproducible():
    a = simulate_experiment(200, 10, rng=3)
    b = simulate_experiment(200, 10, rng=3)
    np.testing.assert_array_equal(a.counts, b.counts)
    assert a.counts.shape == (200, 10)
    rows = a.truth_rows()
    assert rows[0]["burst_size"] == pytest.approx(rows[0]["k_syn"] / rows[0]["k_off"])


def test_beta_capture_moments(rng):
    s = beta_capture(200_000, 0.3, 0.25, rng)
    assert s.mean() == pytest.approx(0.3, rel=0.01)
    assert s.std() / s.mean() == pytest.approx(0.25, rel=0.03)
    with pytest.raises(ValueError):
        beta_capture(10, 0.5, 5.0)


def test_capture_estimated_from_library_size():
    exp = simulate_experiment(1000, 200, mean_capture=0.2, rng=5)
    s = estimate_capture(exp.counts, 0.2)
    assert s.mean() == pytest.approx(0.2, rel=1e-6)
    assert np.corrcoef(s, exp.capture)[0, 1] > 0.6


def test_csv_and_mtx_round_trip(tmp_path):
    counts = np.array([[1, 0, 5], [2, 3, 0]])
    write_counts(tmp_path / "c.csv", counts, ["g1", "g2", "g3"], ["c1", "c2"])
    got, genes, cells = read_counts(tmp_path / "c.csv")
    np.testing.assert_array_equal(got, counts)
    assert genes == ["g1", "g2", "g3"] and cells == ["c1", "c2"]

    from scipy.io import mmwrite
    from scipy.sparse import coo_matrix

    folder = tmp_path / "tenx"
    folder.mkdir()
    mmwrite(str(folder / "matrix.mtx"), coo_matrix(counts.T))  # genes x cells, as 10x writes it
    (folder / "features.tsv").write_text("g1\tGene1\ng2\tGene2\ng3\tGene3\n")
    (folder / "barcodes.tsv").write_text("c1\nc2\n")
    got, genes, cells = read_counts(folder / "matrix.mtx", genes_in_rows=True)
    assert got.shape == (2, 3) and hasattr(got, "toarray")  # stays sparse, cells x genes
    np.testing.assert_array_equal(got.toarray(), counts)
    assert genes == ["g1", "g2", "g3"] and cells == ["c1", "c2"]
    np.testing.assert_allclose(estimate_capture(got, 0.1), estimate_capture(counts, 0.1))


def test_sparse_and_dense_input_give_identical_results():
    from scipy import sparse

    exp = simulate_experiment(200, 3, rng=2)
    dense = analyse_genes(exp.counts, exp.capture, exp.genes, profiles=())
    sp = analyse_genes(sparse.csc_matrix(exp.counts), exp.capture, exp.genes, profiles=())
    for a, b in zip(dense, sp):
        assert a.gene == b.gene and a.status == b.status
        if a.fit:
            assert a.fit.loglik == b.fit.loglik


def test_table_round_trip(tmp_path):
    write_table(tmp_path / "t.csv", [{"a": 1.5, "b": "x", "c": np.nan}])
    assert read_table(tmp_path / "t.csv") == [{"a": "1.5", "b": "x", "c": ""}]


def test_low_expression_gene_is_skipped():
    res = analyse_gene(np.zeros(500, dtype=int))
    assert res.status == "low_expression" and res.fit is None
    assert res.row()["status"] == "low_expression"


def test_gene_analysis_end_to_end(rng):
    from burstkin.analytic import sample_telegraph

    s = rng.beta(30, 70, 600)
    counts = sample_telegraph(0.15, 0.25, 120.0, 600, rng, capture=s)
    res = analyse_gene(counts, s, gene="g")
    assert res.status == "ok"
    assert res.best_model == "telegraph"
    row = res.row()
    for key in ("k_on", "k_off", "k_syn", "burst_size"):
        assert row[f"{key}_ci_low"] <= row[key] <= row[f"{key}_ci_high"]


def test_parallel_matches_serial():
    exp = simulate_experiment(200, 2, rng=11)
    serial = analyse_genes(exp.counts, exp.capture, exp.genes, n_jobs=1, profiles=())
    parallel = analyse_genes(exp.counts, exp.capture, exp.genes, n_jobs=2, profiles=())
    for a, b in zip(serial, parallel):
        assert a.gene == b.gene
        if a.fit:
            assert a.fit.k_on == pytest.approx(b.fit.k_on, rel=1e-12)
