
import pandas as pd
from time_aligned_peaks.peaks import peaks_to_binary_matrix, find_peaks_per_row

def test_peaks_to_binary_matrix():
    idx = pd.RangeIndex(0,5)
    primary = pd.DataFrame({'A':[0,1,3,1,0]}, index=idx)
    secondary = pd.DataFrame({'B':[0,2,1,0,0]}, index=idx)
    p_peaks = find_peaks_per_row(primary)
    s_peaks = find_peaks_per_row(secondary)
    df = peaks_to_binary_matrix(idx, p_peaks, s_peaks, primary.columns.tolist(), secondary.columns.tolist())
    assert list(df.columns) == ['time','A','B','overlap']
    assert df.loc[df['time']==2, 'A'].item() == 1
    assert df.loc[df['time']==1, 'B'].item() == 1
    assert df['overlap'].sum() == 0
