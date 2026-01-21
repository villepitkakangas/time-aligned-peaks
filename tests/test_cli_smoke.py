
# tests/test_cli_smoke.py
import csv
import subprocess
import sys
from pathlib import Path

def _write_csv(path: Path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

def test_cli_generates_figure_and_artifacts(tmp_path: Path):
    # Prepare tiny synthetic primary & secondary CSVs sharing YYYY-MM timeline
    primary = tmp_path / "primary.csv"
    secondary = tmp_path / "secondary.csv"
    figure = tmp_path / "out.png"
    peaks_csv = tmp_path / "peaks.csv"
    matrix_csv = tmp_path / "matrix.csv"

    # Two primary series, two secondary series, 4 timestamps (month granularity)
    # Construct simple peaks at t=2 for primary col0 and secondary col1
    header = ["date", "p1", "p2"]
    rows_p = [
        ["2025-01", 0, 1],
        ["2025-02", 1, 2],
        ["2025-03", 2, 1],  # p1 peak (up then down)
        ["2025-04", 1, 1],
    ]
    _write_csv(primary, header, rows_p)

    header_s = ["date", "s1", "s2"]
    rows_s = [
        ["2025-01", 5, 5],
        ["2025-02", 6, 7],
        ["2025-03", 6, 8],  # s2 peak (up then down)
        ["2025-04", 6, 7],
    ]
    _write_csv(secondary, header_s, rows_s)

    # Run the single-file CLI (adjust path if you add a console_script entry point)
    script = Path(__file__).resolve().parents[1] / "time_aligned_peaks_final.py"
    cmd = [
        sys.executable, str(script),
        "--primary", str(primary),
        "--secondary", str(secondary),
        "--secondary-date-format", "%Y-%m",
        "--title", "CLI Smoke",
        "--save-figure", str(figure),
        "--output-peaks", str(peaks_csv),
        "--output-peak-matrix", str(matrix_csv),
    ]
    subprocess.run(cmd, check=True)

    # Artifacts exist and have expected minimal content
    assert figure.exists() and figure.stat().st_size > 0
    assert peaks_csv.exists()
    assert matrix_csv.exists()

    # Check that 'overlap' column exists and is 0/1
    import pandas as pd
    mdf = pd.read_csv(matrix_csv)
    assert "overlap" in mdf.columns
    assert set(mdf["overlap"].unique()).issubset({0, 1})
