
import numpy as np
import pandas as pd


def find_peaks_per_row(df: pd.DataFrame):
    """Slope-change peak detection per row (per timestamp)."""
    values = df.to_numpy()
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    T, N = values.shape
    peaks = []
    for t in range(T):
        if t == 0 or t == T - 1:
            peaks.append(np.array([], dtype=int))
            continue
        prev, cur, nxt = values[t - 1], values[t], values[t + 1]
        deltas1, deltas2 = cur - prev, nxt - cur
        valid = np.isfinite(deltas1) & np.isfinite(deltas2)
        mask = (deltas1 > 0) & (deltas2 < 0) & valid
        peaks.append(np.where(mask)[0])
    return peaks


def indices_to_names(idx_arr, columns):
    return [str(columns[i]) for i in idx_arr]


def peaks_to_binary_matrix(unified_index, primary_peaks, secondary_peaks, primary_cols, secondary_cols, include_overlap=True):
    """Build a wide 0/1 matrix indicating peak presence per series at each timestamp."""
    import pandas as pd
    n = len(unified_index)
    P = len(primary_cols)
    S = len(secondary_cols)
    pm = np.zeros((n, P), dtype=np.int8)
    sm = np.zeros((n, S), dtype=np.int8)
    for i in range(n):
        if len(primary_peaks[i]) > 0:
            pm[i, primary_peaks[i]] = 1
        if len(secondary_peaks[i]) > 0:
            sm[i, secondary_peaks[i]] = 1
    data = {'time': list(unified_index)}
    for j, name in enumerate(primary_cols):
        data[str(name)] = pm[:, j]
    for j, name in enumerate(secondary_cols):
        data[str(name)] = sm[:, j]
    df = pd.DataFrame(data)
    if include_overlap:
        overlap = (pm.sum(axis=1) > 0) & (sm.sum(axis=1) > 0)
        df['overlap'] = overlap.astype(np.int8)
    return df
