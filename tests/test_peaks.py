
import pandas as pd
from time_aligned_peaks.peaks import find_peaks_per_row

def test_basic_slope_change_peaks():
    s = pd.Series([1, 3, 5, 3, 1])
    df = pd.DataFrame({'A': s})
    peaks = find_peaks_per_row(df)
    idxs = [list(p) for p in peaks]
    assert idxs[2] == [0]
