
import csv
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def read_table(path: Path, sheet_name=0, delimiter: Optional[str]=None) -> pd.DataFrame:
    """Read CSV/TSV or Excel into a DataFrame. Auto-detect delimiter for CSV/TSV."""
    ext = path.suffix.lower()
    if ext in ('.csv', '.tsv'):
        if delimiter is None:
            with open(path, 'r', newline='') as f:
                sample = f.read(2048)
                detected = csv.Sniffer().sniff(sample)
                delimiter = detected.delimiter
                logging.info(f"Auto-detected delimiter for {path.name}: '{delimiter}'")
        return pd.read_csv(path, sep=delimiter)
    elif ext in ('.xls', '.xlsx'):
        engine = 'openpyxl' if ext == '.xlsx' else None
        return pd.read_excel(path, engine=engine, sheet_name=sheet_name)
    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: .csv, .tsv, .xls, .xlsx"
        )


def parse_secondary_index(df: pd.DataFrame, date_format: str = '%Y-%m', tz: Optional[str] = None):
    """Parse the first column as datetime index; return (df, has_datetime)."""
    first_col = df.columns[0]
    parsed = pd.to_datetime(df[first_col].astype(str), format=date_format, errors='coerce')
    if parsed.notna().sum() == 0:
        # No valid dates; drop the first column
        df = df.drop(columns=[first_col])
        return df, False
    # Keep valid rows and sort chronologically
    mask_valid = parsed.notna()
    df = df.loc[mask_valid].copy()
    parsed = parsed.loc[mask_valid]
    order = np.argsort(parsed.values)
    df = df.iloc[order].copy()
    parsed = parsed.iloc[order]
    if tz:
        parsed = parsed.dt.tz_localize(tz) if parsed.dt.tz is None else parsed.dt.tz_convert(tz)
    df = df.set_index(parsed)
    if first_col in df.columns:
        df = df.drop(columns=[first_col])
    return df, True


def is_running_number_column(series: pd.Series) -> bool:
    arr = pd.to_numeric(series, errors='coerce')
    if arr.isna().any():
        return False
    vals = arr.to_numpy()
    return len(vals) > 1 and np.all(np.diff(vals) == 1)


def drop_running_number_column_if_no_datetime(df: pd.DataFrame):
    """Drop the first detected running-number column; return (df, dropped_col or None)."""
    for col in df.columns:
        if is_running_number_column(df[col]):
            return df.drop(columns=[col]).copy(), col
    return df, None


def apply_secondary_index_to_primary_by_position(primary_df: pd.DataFrame, secondary_index: pd.Index) -> pd.DataFrame:
    """Assign secondary timestamps to primary by row position, truncating or padding as needed."""
    n, m = len(primary_df), len(secondary_index)
    if n == 0:
        return pd.DataFrame(index=secondary_index)
    if n <= m:
        primary_df.index = secondary_index[:n]
        return primary_df.reindex(secondary_index)
    else:
        logging.info(f"Warning: primary has {n} rows but secondary has {m}. Truncating primary.")
        primary_df = primary_df.iloc[:m].copy()
        primary_df.index = secondary_index
        return primary_df


def resample_df(df: pd.DataFrame, freq: Optional[str]):
    """Resample only if index is DatetimeIndex."""
    return df.resample(freq).mean() if freq and isinstance(df.index, pd.DatetimeIndex) else df


def detect_and_fix_orientation(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Try to detect transposed tables and fix orientation."""
    rows, cols = df.shape
    def looks_like_timestamps(series: pd.Series) -> bool:
        parsed = pd.to_datetime(series, errors='coerce')
        return parsed.notna().sum() > len(series) * 0.8
    # Headers
    if looks_like_timestamps(pd.Series(df.columns)):
        logging.info(f"Auto-detect: {name} headers look like timestamps -> transposed.")
        return df.transpose()
    # First row/column
    if looks_like_timestamps(df.iloc[0]):
        logging.info(f"Auto-detect: {name} first row looks like timestamps -> normal.")
        return df
    if looks_like_timestamps(df.iloc[:, 0]):
        logging.info(f"Auto-detect: {name} first column looks like timestamps -> normal.")
        return df
    # Fallback shape heuristic
    if rows > cols * 3:
        logging.info(f"Auto-detect: {name} appears transposed (rows={rows}, cols={cols}). Transposing...")
        return df.transpose()
    logging.info(f"Auto-detect: {name} orientation seems normal.")
    return df
