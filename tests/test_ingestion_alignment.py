
# tests/test_ingestion_alignment.py
import io
import pandas as pd
from pathlib import Path

from time_aligned_peaks_final import (
    read_table,
    detect_and_fix_orientation,
    parse_secondary_index,
    apply_secondary_index_to_primary_by_position,
)

def test_csv_delimiter_sniffing(tmp_path: Path):
    # Create a semicolon-separated CSV
    p = tmp_path / "demo.csv"
    p.write_text("A;B;C\n1;2;3\n4;5;6\n", encoding="utf-8")
    df = read_table(p)  # delimiter=None -> sniff
    assert df.shape == (2, 3)
    assert list(df.columns) == ["A", "B", "C"]

def test_orientation_autodetect_transposed():
    # Fake a transposed table with timestamp-like headers
    cols = ["2025-01", "2025-02", "2025-03"]
    df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=cols)
    fixed = detect_and_fix_orientation(df, "Primary")
    # Should transpose because headers look like timestamps
    assert fixed.shape == (3, 2)  # transposed

def test_parse_secondary_index_and_apply_to_primary(tmp_path: Path):
    # Secondary with YYYY-MM dates in first column
    sec = pd.DataFrame({
        "date": ["2025-01", "2025-02", "2025-03"],
        "y": [10, 20, 15]
    })
    spath = tmp_path / "secondary.csv"
    sec.to_csv(spath, index=False)

    # Primary without dates
    pri = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    ppath = tmp_path / "primary.csv"
    pri.to_csv(ppath, index=False)

    sec_df = read_table(spath)
    sec_df, has_dt = parse_secondary_index(sec_df, date_format="%Y-%m", tz=None)
    assert has_dt and isinstance(sec_df.index, pd.DatetimeIndex)

    pri_df = read_table(ppath)
    # adopt secondary index by position
    aligned = apply_secondary_index_to_primary_by_position(pri_df, sec_df.index)
    assert isinstance(aligned.index, pd.DatetimeIndex)
    assert aligned.shape == pri_df.shape
