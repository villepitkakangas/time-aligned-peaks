
# tests/test_peak_detection.py
import numpy as np
import pandas as pd

# Import directly from your package/module. If only the single-file exists,
# use: from time_aligned_peaks_final import find_peaks_per_row
from time_aligned_peaks_final import find_peaks_per_row

def test_slope_change_detects_simple_peaks():
    # Construct 3 series with a known peak at t=2
    # s1: 0 -> 1 -> 2 -> 1 (peak at index 2)
    # s2: 5 -> 6 -> 7 -> 6 (peak at index 2)
    # s3: 0 -> 0 -> 0 -> 0 (no peaks)
    data = {
        "s1": [0, 1, 2, 1],
        "s2": [5, 6, 7, 6],
        "s3": [0, 0, 0, 0],
    }
    df = pd.DataFrame(data)

    peaks = find_peaks_per_row(df)
    # peaks is a list of arrays per timestamp; timestamps 0 and last have empty arrays.
    # At t=2, both s1 and s2 peak -> column indices [0, 1]
    assert len(peaks) == 4
    assert peaks[0].size == 0
    assert set(peaks[2].tolist()) == {0, 1}
    assert peaks[3].size == 0

def test_no_peaks_on_monotonic_segments():
    df = pd.DataFrame({"a": [0, 1, 2, 3, 4], "b": [4, 3, 2, 1, 0]})
    peaks = find_peaks_per_row(df)
    # strictly monotone → no slope change (up then down) within interior rows
    assert all(arr.size == 0 for arr in peaks)

def test_nan_handling_does_not_raise():
    df = pd.DataFrame({"x": [np.nan, 1.0, 2.0, np.nan, 1.0]})
    peaks = find_peaks_per_row(df)
    # Should run, may or may not detect anything depending on neighbors
    assert isinstance(peaks, list)
