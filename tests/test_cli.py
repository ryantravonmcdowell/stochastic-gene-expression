from burstkin.cli import main
from burstkin.scdata import read_table


def test_simulate_then_fit(tmp_path, capsys):
    counts = tmp_path / "counts.csv"
    truth = tmp_path / "truth.csv"
    capture = tmp_path / "capture.csv"
    assert main(["simulate", "--cells", "150", "--genes", "3", "--seed", "4", "-o", str(counts),
                 "--truth", str(truth), "--capture-out", str(capture)]) == 0
    out = tmp_path / "fit.csv"
    assert main(["fit", str(counts), "--capture-file", str(capture), "--profiles", "none", "-o", str(out)]) == 0
    rows = read_table(out)
    assert [r["gene"] for r in rows] == [r["gene"] for r in read_table(truth)]
    assert "wrote" in capsys.readouterr().out


def test_gene_command(capsys):
    assert main(["gene", "--cells", "400", "--k-on", "0.3", "--k-off", "0.4", "--k-syn", "60", "--seed", "2"]) == 0
    text = capsys.readouterr().out
    assert "burst_size" in text and "identifiability" in text


def test_validate_command(capsys):
    assert main(["validate"]) == 0
    assert "15/15 checks passed" in capsys.readouterr().out
