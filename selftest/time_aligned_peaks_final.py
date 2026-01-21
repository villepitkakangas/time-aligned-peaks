"""
    Time-aligned Peaks
    Author: Ville Pitkäkangas <ville.pitkakangas@centria.fi>
    License: MIT
"""

#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
import logging

# ----------------------------
# Utilities
# ----------------------------

def read_table(path: Path, sheet_name=0, delimiter=None) -> pd.DataFrame:
    """
    Read a data table from CSV/TSV or Excel.
    Auto-detect delimiter for CSV/TSV if not provided.
    """
    ext = path.suffix.lower()
    if ext in (".csv", ".tsv"):
        # Auto-detect delimiter if not provided
        if delimiter is None:
            with open(path, "r", newline="") as f:
                sample = f.read(2048)  # Read a small chunk for sniffing
                detected = csv.Sniffer().sniff(sample)
                delimiter = detected.delimiter
                logging.info(f"Auto-detected delimiter for {path.name}: '{delimiter}'")
        return pd.read_csv(path, sep=delimiter)
    elif ext in (".xls", ".xlsx"):
        engine = "openpyxl" if ext == ".xlsx" else None
        return pd.read_excel(path, engine=engine, sheet_name=sheet_name)
    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: .csv, .tsv, .xls, .xlsx"
        )

def parse_secondary_index(df: pd.DataFrame, date_format: str = "%Y-%m", tz: str | None = None):
    """Parse secondary's first column as dates; if it fully fails, drop that column."""
    first_col = df.columns[0]
    parsed = pd.to_datetime(df[first_col].astype(str), format=date_format, errors="coerce")
    if parsed.notna().sum() == 0:
        # No valid dates at all → treat first col as non-date and drop it
        df = df.drop(columns=[first_col])
        return df, False
    # Keep only valid rows and sort chronologically
    mask_valid = parsed.notna()
    df = df.loc[mask_valid].copy()
    parsed = parsed.loc[mask_valid]
    order = np.argsort(parsed.values)
    df = df.iloc[order].copy()
    parsed = parsed.iloc[order]
    if tz:
        parsed = parsed.dt.tz_localize(tz) if parsed.dt.tz is None else parsed.dt.tz_convert(tz)
    df = df.set_index(parsed)
    # Ensure the original date column is not left in the data
    if first_col in df.columns:
        df = df.drop(columns=[first_col])
    return df, True

def is_running_number_column(series: pd.Series) -> bool:
    arr = pd.to_numeric(series, errors="coerce")
    if arr.isna().any():
        return False
    vals = arr.to_numpy()
    return len(vals) > 1 and np.all(np.diff(vals) == 1)

def drop_running_number_column_if_no_datetime(df: pd.DataFrame):
    """Drop the first detected 'running number' column if no datetime is present."""
    for col in df.columns:
        if is_running_number_column(df[col]):
            return df.drop(columns=[col]).copy(), col
    return df, None

def apply_secondary_index_to_primary_by_position(primary_df: pd.DataFrame, secondary_index: pd.Index):
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

def resample_df(df: pd.DataFrame, freq: str | None):
    """Resample only if index is DatetimeIndex."""
    return df.resample(freq).mean() if freq and isinstance(df.index, pd.DatetimeIndex) else df

def set_axis_frequency(ax: plt.Axes, index: pd.Index):
    """Set date tick locators/formatters and rotate labels if crowded."""
    if not isinstance(index, pd.DatetimeIndex):
        return
    freq = pd.infer_freq(index)
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    if freq:
        f = freq.upper()
        if f.startswith("M"):
            locator, formatter = mdates.MonthLocator(), mdates.DateFormatter("%Y-%m")
        elif f.startswith("D"):
            locator, formatter = mdates.DayLocator(), mdates.DateFormatter("%Y-%m-%d")
        elif f.startswith("A") or f.startswith("Y"):
            locator, formatter = mdates.YearLocator(), mdates.DateFormatter("%Y")
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.figure.canvas.draw()
    if len(ax.get_xticklabels()) > 12:
        plt.setp(ax.get_xticklabels(), rotation=90)
        
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

# ----------------------------
# Peak detection
# ----------------------------
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

# ----------------------------
# Timeline visualization (GridSpec-based)
# ----------------------------

from matplotlib.colors import to_rgb

def invert_color(color):
    r, g, b = to_rgb(color)
    return (1 - r, 1 - g, 1 - b)

def brightness(color):
    r, g, b = to_rgb(color)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def is_combined_similar(color1, color2, rgb_threshold=0.6, brightness_threshold=0.2):
    """Return True if colors are similar in RGB space OR brightness."""
    # RGB distance
    c1, c2 = np.array(to_rgb(color1)), np.array(to_rgb(color2))
    rgb_distance = np.linalg.norm(c1 - c2)

    # Brightness difference
    brightness_diff = abs(brightness(color1) - brightness(color2))

    return rgb_distance < rgb_threshold or brightness_diff < brightness_threshold


def add_peak_timeline(
    fig, gs, unified_index, primary_peaks, secondary_peaks, overlap_mask,
    primary_cols, secondary_cols, line_colors, ax, bottom_lw, right_lw,
    top_lw, left_lw, overlap_color, overlap_thickness, border_color, split_color,
    fill_color, fill_alpha, overlap_mode, bar_width_mode='local', bar_shrink=0.95
):
    """
    Draw the bottom timeline rectangle:
    - Order matches legend: primary (top, same order), then secondary (same order).
    - Split line constrained to rectangle width.
    - Overlap lines clipped to rectangle height.
    - Bars offset left by half tick width, full cell height.
    - Y-scale stays fixed on zoom, X-scale follows main plot.
    """

    num_primary = len(primary_cols)
    num_secondary = len(secondary_cols)
    num_vars = num_primary + num_secondary
    timeline_ax = fig.add_subplot(gs[1], sharex=ax)
    timeline_ax.set_ylim(-0.5, num_vars)
    timeline_ax.set_autoscale_on(False)
    timeline_ax.set_navigate(False)
    timeline_ax.invert_yaxis()
    timeline_ax.axis('off')
    fig.canvas.draw()
    x_min, x_max = ax.get_xlim()
    rect_start, rect_end = x_min, x_max
    rect_width = rect_end - rect_start
    base_lw = ax.spines['left'].get_linewidth() if ax.spines else 2.0
    timeline_ax.add_patch(Rectangle((rect_start, 0), rect_width, num_vars,
                  fill=bool(fill_color), facecolor=(fill_color if fill_color else None),
                  alpha=(fill_alpha if fill_color else None), linewidth=base_lw,
                  edgecolor=border_color, zorder=1))
    timeline_ax.hlines(y=num_vars, xmin=rect_start, xmax=rect_end, colors=border_color, linewidth=bottom_lw, zorder=3)
    timeline_ax.vlines(x=rect_end, ymin=0, ymax=num_vars, colors=border_color, linewidth=right_lw, zorder=3)
    timeline_ax.hlines(y=0, xmin=rect_start, xmax=rect_end, colors=border_color, linewidth=top_lw, zorder=3)
    timeline_ax.vlines(x=rect_start, ymin=0, ymax=num_vars, colors=border_color, linewidth=left_lw, zorder=3)
    split_y = num_primary
    timeline_ax.hlines(y=split_y, xmin=rect_start, xmax=rect_end, colors=split_color, linewidth=base_lw, zorder=2)

    x_numeric = mdates.date2num(unified_index)

    def _local_span(i, x, shrink=0.95):
        n = len(x)
        if n == 1:
            w = shrink * 1.0
            return x[0] - 0.5 * w, x[0] + 0.5 * w
        left_gap = (x[i] - x[i - 1]) if i > 0 else (x[1] - x[0])
        right_gap = (x[i + 1] - x[i]) if i < n - 1 else (x[-1] - x[-2])
        eps = 1e-9
        left_gap = max(left_gap, eps)
        right_gap = max(right_gap, eps)
        return x[i] - 0.5 * shrink * left_gap, x[i] + 0.5 * shrink * right_gap

    diffs = np.diff(x_numeric) if len(x_numeric) > 1 else np.array([])
    global_bar_width = None
    if bar_width_mode in ('median', 'min') and diffs.size:
        if bar_width_mode == 'median':
            global_bar_width = bar_shrink * float(np.median(diffs))
        else:
            global_bar_width = bar_shrink * float(np.min(diffs))
    elif bar_width_mode in ('median', 'min'):
        global_bar_width = bar_shrink * 1.0

    bar_height = 1.0
    for i in range(len(unified_index)):
        x_center = x_numeric[i]
        if bar_width_mode == 'local':
            x_left, x_right = _local_span(i, x_numeric, shrink=bar_shrink)
            bar_width = x_right - x_left
        else:
            bar_width = global_bar_width if global_bar_width is not None else bar_shrink * 1.0
            x_left = x_center - 0.5 * bar_width
        for col_idx in primary_peaks[i]:
            row = col_idx
            color = line_colors['primary'][col_idx]
            timeline_ax.add_patch(Rectangle((x_left, row), bar_width, bar_height,
                                            color=color, zorder=2, linewidth=0, antialiased=False))
        for col_idx in secondary_peaks[i]:
            row = num_primary + col_idx
            color = line_colors['secondary'][col_idx]
            timeline_ax.add_patch(Rectangle((x_left, row), bar_width, bar_height,
                                            color=color, zorder=2, linewidth=0, antialiased=False))

    for i, overlap in enumerate(overlap_mask):
        if overlap:
            x_center = x_numeric[i]
            original_color = overlap_color
            for row_idx in range(num_primary + num_secondary):
                ymin = row_idx
                ymax = row_idx + 1
                segment_color = original_color
                peak_exists = False
                if row_idx < len(primary_cols):
                    peak_exists = row_idx in primary_peaks[i]
                    var_color = line_colors['primary'][row_idx]
                else:
                    sec_idx = row_idx - len(primary_cols)
                    peak_exists = sec_idx in secondary_peaks[i]
                    var_color = line_colors['secondary'][sec_idx]
                if peak_exists:
                    if overlap_mode == 'inverse1':
                        segment_color = invert_color(original_color)
                    elif overlap_mode == 'inverse2' and is_combined_similar(original_color, var_color):
                        segment_color = invert_color(original_color)
                timeline_ax.vlines(x_center, ymin=ymin, ymax=ymax,
                                   colors=segment_color, linewidth=overlap_thickness, zorder=5)
    timeline_ax.set_xlim(rect_start, rect_end)



def detect_and_fix_orientation(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Improved orientation detection:
    - Check if headers look like timestamps (indicates transposed).
    - Check first row and column for timestamp density.
    - Fallback: shape heuristic.
    """
    rows, cols = df.shape

    def looks_like_timestamps(series: pd.Series) -> bool:
        parsed = pd.to_datetime(series, errors="coerce")
        return parsed.notna().sum() > len(series) * 0.8

    # Check headers
    if looks_like_timestamps(pd.Series(df.columns)):
        logging.info(f"Auto-detect: {name} headers look like timestamps → transposed.")
        return df.transpose()

    # Check first row and column
    if looks_like_timestamps(df.iloc[0]):
        logging.info(f"Auto-detect: {name} first row looks like timestamps → normal.")
        return df
    if looks_like_timestamps(df.iloc[:, 0]):
        logging.info(f"Auto-detect: {name} first column looks like timestamps → normal.")
        return df

    # Fallback heuristic
    if rows > cols * 3:
        logging.info(f"Auto-detect: {name} appears transposed (rows={rows}, cols={cols}). Transposing...")
        return df.transpose()

    logging.info(f"Auto-detect: {name} orientation seems normal.")
    return df


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", required=True)
    parser.add_argument("--secondary", required=True)
    parser.add_argument("--primary-sheet", default=0)
    parser.add_argument("--secondary-sheet", default=0)
    parser.add_argument("--secondary-date-format", default="%Y-%m")
    parser.add_argument("--tz", default=None)
    parser.add_argument("--resample", default=None)
    parser.add_argument("--fill-missing", choices=["zero", "none", "ffill", "bfill"], default="none")
    parser.add_argument("--output-peaks", default=None)
    parser.add_argument("--title", default="Combined Data Plot")
    parser.add_argument("--timeline-bottom-lw", type=float, default=2.5,
                    help="Line width for bottom edge of timeline rectangle")
    parser.add_argument("--timeline-right-lw", type=float, default=2.5,
                    help="Line width for right edge of timeline rectangle")
    parser.add_argument("--timeline-left-lw", type=float, default=2.5,
                    help="Line width for left edge of timeline rectangle")
    parser.add_argument("--timeline-top-lw", type=float, default=2.5,
                    help="Line width for top edge of timeline rectangle")
    parser.add_argument("--save-figure", default=None,
                    help="Path to save the combined figure (main plot + timeline)")
    parser.add_argument("--figure-format", default=None,
                    help="Image format for saved figure (e.g., png, pdf, svg). If not provided, inferred from file extension.")
    parser.add_argument("--primary-cols", type=str, default=None,
                    help="Comma-separated column names for primary table")
    parser.add_argument("--secondary-cols", type=str, default=None,
                    help="Comma-separated column names for secondary table")
    parser.add_argument("--delimiter", default=None,
                    help="Custom delimiter for CSV/TSV (auto-detected if not provided)")
    parser.add_argument("--primary-transposed", action="store_true",
                        help="Indicate that the primary table is transposed (columns are observations)")
    parser.add_argument("--secondary-transposed", action="store_true",
                        help="Indicate that the secondary table is transposed (columns are observations)")
    parser.add_argument("--auto-detect-orientation", action="store_true",
                        help="Automatically detect if tables are transposed and fix orientation")
    parser.add_argument("--force-orientation", type=str, default=None,
        help="Force orientation: format 'primary=normal|transposed secondary=normal|transposed'")
    parser.add_argument("--log-level", type=str, default="INFO",
        help="Set logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL")
    parser.add_argument("--legend-primary-prefix", default="Primary", help="Prefix for primary series in legend")
    parser.add_argument("--legend-secondary-prefix", default="Secondary", help="Prefix for secondary series in legend")
    parser.add_argument("--xlabel", default="Time", help="Custom X-axis label")
    parser.add_argument("--ylabel", default="Values", help="Custom Y-axis label")
    parser.add_argument("--overlap-style", choices=['none','lineplot','sync-timeline'], default='lineplot',
        help="Control overlap markers in main plot: none, lineplot (current style), sync-timeline (same as timeline)")
    parser.add_argument("--overlap-color", default='red',
        help="Color for overlap markers in main plot")
    parser.add_argument("--overlap-thickness", type=float, default=1.0,
        help="Line thickness for overlap markers in main plot")
    parser.add_argument("--timeline-overlap-color", default='red',
        help="Color for overlap markers in timeline visualization")
    parser.add_argument("--timeline-overlap-thickness", type=float, default=1.0,
        help="Line thickness for overlap markers in timeline visualization")
    parser.add_argument("--timeline-border-color", default='black',
        help="Color for timeline rectangle border")
    parser.add_argument("--timeline-split-color", default='0.5',
        help="Color for split line between primary and secondary in timeline")
    parser.add_argument("--timeline-fill-color", default=None,
        help="Fill color for timeline rectangle background (e.g., 'lightgray'); default None = no fill")
    parser.add_argument("--timeline-fill-alpha", type=float, default=1.0,
        help="Transparency (alpha) for timeline rectangle fill; default 1.0")
    parser.add_argument("--timeline-overlap-mode", choices=['default', 'inverse1', 'inverse2'], default='default',
        help="Overlap line color mode: default=no inverse, inverse1=invert inside rectangle if any peak exists, inverse2=invert inside rectangle if too similar"
    )
    parser.add_argument('--timeline-bar-width-mode', choices=['local', 'median', 'min'], default='local')
    parser.add_argument('--timeline-bar-shrink', type=float, default=0.95)
    parser.add_argument('--output-peak-matrix', default=None)

    args = parser.parse_args()


    # Configure logging level based on CLI argument
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {args.log_level}')
    logging.basicConfig(level=numeric_level, format='%(asctime)s [%(levelname)s] %(message)s')

    # --- Read Excel ---
    primary_raw = read_table(Path(args.primary), sheet_name=args.primary_sheet, delimiter=args.delimiter)
    secondary_raw = read_table(Path(args.secondary), sheet_name=args.secondary_sheet, delimiter=args.delimiter)

    # Apply orientation overrides or auto-detection
    force_orientation = {}
    if args.force_orientation:
        for part in args.force_orientation.split():
            key, val = part.split("=")
            force_orientation[key.strip()] = val.strip()

    if "primary" in force_orientation:
        primary_raw = primary_raw if force_orientation["primary"] == "normal" else primary_raw.transpose()
    elif args.auto_detect_orientation:
        primary_raw = detect_and_fix_orientation(primary_raw, "Primary")
    elif args.primary_transposed:
        primary_raw = primary_raw.transpose()

    if "secondary" in force_orientation:
        secondary_raw = secondary_raw if force_orientation["secondary"] == "normal" else secondary_raw.transpose()
    elif args.auto_detect_orientation:
        secondary_raw = detect_and_fix_orientation(secondary_raw, "Secondary")
    elif args.secondary_transposed:
        secondary_raw = secondary_raw.transpose()
        
    # --- parse first column as dates (or drop it if fully invalid) ---
    secondary_df, sec_has_datetime = parse_secondary_index(
        secondary_raw, date_format=args.secondary_date_format, tz=args.tz
    )

    # --- Primary: try to parse same-named date column; else drop running-number column ---
    primary_df = primary_raw.copy()
    sec_first_col = secondary_raw.columns[0]
    if sec_first_col in primary_df.columns:
        parsed_primary = pd.to_datetime(primary_df[sec_first_col], errors="coerce")
        if parsed_primary.notna().sum() > 0:
            primary_df = primary_df.set_index(parsed_primary)
            primary_df = primary_df[primary_df.index.notna()].sort_index()
            if sec_first_col in primary_df.columns:
                primary_df = primary_df.drop(columns=[sec_first_col])
        else:
            primary_df = primary_df.drop(columns=[sec_first_col])

    if not isinstance(primary_df.index, pd.DatetimeIndex):
        primary_df, dropped_col = drop_running_number_column_if_no_datetime(primary_df)
        if dropped_col:
            logging.info(f"Dropped running-number column: '{dropped_col}'")

    # --- If primary has no dates, adopt secondary's timeline by position; else create synthetic index ---
    if not isinstance(primary_df.index, pd.DatetimeIndex):
        if sec_has_datetime:
            primary_df = apply_secondary_index_to_primary_by_position(primary_df, secondary_df.index)
        else:
            synthetic_index = pd.RangeIndex(start=0, stop=max(len(primary_df), len(secondary_df)), step=1)
            primary_df.index = synthetic_index[:len(primary_df)]
            secondary_df.index = synthetic_index[:len(secondary_df)]

    # --- Optional resampling ---
    primary_df = resample_df(primary_df, args.resample)
    secondary_df = resample_df(secondary_df, args.resample)

    # --- Align to unified index ---
    unified_index = primary_df.index.union(secondary_df.index)
    primary_aligned = primary_df.reindex(unified_index)
    secondary_aligned = secondary_df.reindex(unified_index)

    # Override column names if provided
    if args.primary_cols is not None:
        names = [name.strip() for name in args.primary_cols.split(",")]
        if len(names) != len(primary_aligned.columns):
            raise ValueError(f"Expected {len(primary_aligned.columns)} primary names, got {len(names)}")
        primary_aligned.columns = names

    if args.secondary_cols is not None:
        names = [name.strip() for name in args.secondary_cols.split(",")]
        if len(names) != len(secondary_aligned.columns):
            raise ValueError(f"Expected {len(secondary_aligned.columns)} secondary names, got {len(names)}")
        secondary_aligned.columns = names

    # --- Missing values handling ---
    if args.fill_missing == "zero":
        primary_aligned = primary_aligned.fillna(0)
        secondary_aligned = secondary_aligned.fillna(0)
    elif args.fill_missing == "ffill":
        primary_aligned = primary_aligned.ffill()
        secondary_aligned = secondary_aligned.ffill()
    elif args.fill_missing == "bfill":
        primary_aligned = primary_aligned.bfill()
        secondary_aligned = secondary_aligned.bfill()

    # --- Ensure numeric types for peak detection ---
    primary_aligned = primary_aligned.apply(pd.to_numeric, errors="coerce")
    secondary_aligned = secondary_aligned.apply(pd.to_numeric, errors="coerce")

    # --- Layout (GridSpec) ---
    num_vars = len(primary_aligned.columns) + len(secondary_aligned.columns)
    fig = plt.figure(figsize=(12, 6 + 0.3 * num_vars))
    gs = gridspec.GridSpec(2, 1, height_ratios=[6, num_vars], hspace=0.45)

    # --- Main plot ---
    ax = fig.add_subplot(gs[0])
    line_colors = {"primary": [], "secondary": []}
    for col in primary_aligned.columns:
        line, = ax.plot(primary_aligned.index, primary_aligned[col], label=f"{args.legend_primary_prefix}: {col}")
        line_colors["primary"].append(line.get_color())
    for col in secondary_aligned.columns:
        line, = ax.plot(secondary_aligned.index, secondary_aligned[col], label=f"{args.legend_secondary_prefix}: {col}")
        line_colors["secondary"].append(line.get_color())

    ax.set_title(args.title)
    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    set_axis_frequency(ax, unified_index)

    # --- Peak detection ---
    primary_peaks = find_peaks_per_row(primary_aligned)
    secondary_peaks = find_peaks_per_row(secondary_aligned)

    overlap_mask = [
        (len(primary_peaks[i]) > 0 and len(secondary_peaks[i]) > 0)
        for i in range(len(unified_index))
    ]
    # Render the overlap positions
    for i, overlap in enumerate(overlap_mask):
        if not overlap:
            continue
        if args.overlap_style == 'none':
            continue
        elif args.overlap_style == 'lineplot':
            ax.axvline(unified_index[i], color=args.overlap_color, alpha=0.15, linewidth=args.overlap_thickness)
        elif args.overlap_style == 'sync-timeline':
            ax.axvline(unified_index[i], color=args.timeline_overlap_color, linewidth=args.timeline_overlap_thickness)


    # --- Timeline visualization ---
    add_peak_timeline(
        fig, gs, unified_index, primary_peaks, secondary_peaks, overlap_mask,
        list(primary_aligned.columns), list(secondary_aligned.columns),
        line_colors, ax, args.timeline_bottom_lw, args.timeline_right_lw,
        args.timeline_top_lw, args.timeline_left_lw,
        args.timeline_overlap_color, args.timeline_overlap_thickness,
        args.timeline_border_color, args.timeline_split_color, args.timeline_fill_color,
        args.timeline_fill_alpha, args.timeline_overlap_mode,
        args.timeline_bar_width_mode, args.timeline_bar_shrink
    )


    if args.save_figure:
        fmt = args.figure_format if args.figure_format else Path(args.save_figure).suffix.lstrip('.').lower()
        plt.savefig(args.save_figure, dpi=300, bbox_inches='tight', format=fmt)
        logging.info(f"Figure saved to {args.save_figure}")

    plt.show()

    if args.output_peaks is not None:
        # --- Save peak report ---
        rows = []
        for i, ts in enumerate(unified_index):
            sec_names = indices_to_names(secondary_peaks[i], secondary_aligned.columns)
            pri_names = indices_to_names(primary_peaks[i], primary_aligned.columns)
            rows.append({
                "row": i,
                "time": ts,
                "secondary_peak_columns": ", ".join(sec_names),
                "primary_peak_columns": ", ".join(pri_names),
                "overlap": overlap_mask[i],
            })
        pd.DataFrame(rows).to_csv(args.output_peaks, index=False)
        logging.info(f"Peak report saved to {args.output_peaks}")
    if args.output_peak_matrix is not None:
        matrix_df = peaks_to_binary_matrix(
            unified_index, primary_peaks, secondary_peaks,
            list(primary_aligned.columns), list(secondary_aligned.columns), include_overlap=True
        )
        matrix_df.to_csv(args.output_peak_matrix, index=False)
        logging.info(f"Peak matrix saved to {args.output_peak_matrix}")


if __name__ == "__main__":
    main()
