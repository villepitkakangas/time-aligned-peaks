#!/usr/bin/env python3
"""
TAP Self-Test Tool
------------------
A small interactive tool to assess how quickly and accurately a user can detect
primary–secondary overlapping peak events using (A) a traditional overlaid line plot
and (B) the Time-Aligned Peaks (TAP) dual-panel view (line plot + stacked peak timeline).

It computes "ground truth" overlaps using the same slope-change peak detection
criterion as TAP and compares the user's clicked timestamps to the nearest index values.

Outputs:
  - results JSON with timing and metrics per condition
  - CSV with per-timestamp classification for each condition
  - annotated PNGs of the two sessions (optional, if --save-figs specified)

Requirements: pandas, numpy, matplotlib, openpyxl (if reading .xlsx)

Usage (example):
  python tap_selftest.py \
    --primary examples/primary.csv \
    --secondary examples/secondary.xlsx \
    --secondary-sheet 0 \
    --secondary-date-format "%Y-%m" \
    --title "TAP Self-Test" \
    --out-dir selftest_out \
    --save-figs

If time_aligned_peaks_final.py is available in the same directory, the tool will
import its ingestion/alignment functions; otherwise, it falls back to internal
simplified ingestion.
"""
import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.widgets import Button

import time

# ----------------------- Try to import TAP helpers -----------------------
TAP = None
try:
    import importlib
    TAP = importlib.import_module('time_aligned_peaks_final')
except Exception:
    TAP = None

# ----------------------- Internal fallbacks (simplified) -----------------------
import csv
from pathlib import Path

def _read_table(path: Path, sheet_name=0, delimiter=None) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext in ('.csv', '.tsv'):
        if delimiter is None:
            with open(path, 'r', newline='') as f:
                sample = f.read(2048)
                try:
                    detected = csv.Sniffer().sniff(sample)
                    delimiter = detected.delimiter
                except Exception:
                    delimiter = ',' if ext == '.csv' else '\t'
        return pd.read_csv(path, sep=delimiter)
    elif ext in ('.xls', '.xlsx'):
        engine = 'openpyxl' if ext == '.xlsx' else None
        return pd.read_excel(path, engine=engine, sheet_name=sheet_name)
    else:
        raise ValueError(f"Unsupported file type '{ext}'.")


def _parse_secondary_index(df: pd.DataFrame, date_format: str = "%Y-%m", tz: str | None = None):
    first_col = df.columns[0]
    parsed = pd.to_datetime(df[first_col].astype(str), format=date_format, errors='coerce')
    if parsed.notna().sum() == 0:
        df = df.drop(columns=[first_col])
        return df, False
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

def _resample_df(df: pd.DataFrame, freq: str | None):
    return df.resample(freq).mean() if freq and isinstance(df.index, pd.DatetimeIndex) else df


def _find_peaks_per_row(df: pd.DataFrame):
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

# ----------------------- Core utilities -----------------------

def align_and_detect(primary_path: Path,
                     secondary_path: Path,
                     primary_sheet=0,
                     secondary_sheet=0,
                     secondary_date_format="%Y-%m",
                     resample=None,
                     delimiter=None,
                     tz=None,
                     primary_cols=None,
                     secondary_cols=None):
    """Load, align (union index), and detect peaks for primary/secondary.
    Returns: dict with aligned DataFrames, unified_index, lists of peak indices per row,
    and overlap_mask (bool per timestamp).
    """
    if TAP is not None:
        read_table = TAP.read_table
        parse_secondary_index = TAP.parse_secondary_index
        resample_df = TAP.resample_df
        find_peaks_per_row = TAP.find_peaks_per_row
    else:
        read_table = _read_table
        parse_secondary_index = _parse_secondary_index
        resample_df = _resample_df
        find_peaks_per_row = _find_peaks_per_row

    primary_raw = read_table(Path(primary_path), sheet_name=primary_sheet, delimiter=delimiter)
    secondary_raw = read_table(Path(secondary_path), sheet_name=secondary_sheet, delimiter=delimiter)

    # Parse secondary dates (or drop first col if invalid)
    secondary_df, sec_has_datetime = parse_secondary_index(
        secondary_raw, date_format=secondary_date_format, tz=tz
    )

    # Primary: try to parse a same-named date column as secondary's first col; else leave as-is
    primary_df = primary_raw.copy()
    sec_first_col = secondary_raw.columns[0]
    if sec_first_col in primary_df.columns:
        parsed_primary = pd.to_datetime(primary_df[sec_first_col], errors='coerce')
        if parsed_primary.notna().sum() > 0:
            primary_df = primary_df.set_index(parsed_primary)
            primary_df = primary_df[primary_df.index.notna()].sort_index()
            if sec_first_col in primary_df.columns:
                primary_df = primary_df.drop(columns=[sec_first_col])
        else:
            primary_df = primary_df.drop(columns=[sec_first_col])

    # If primary has no datetime index and secondary does, adopt by position
    if not isinstance(primary_df.index, pd.DatetimeIndex):
        if sec_has_datetime:
            n, m = len(primary_df), len(secondary_df.index)
            if n == 0:
                primary_df = pd.DataFrame(index=secondary_df.index)
            elif n <= m:
                primary_df.index = secondary_df.index[:n]
                primary_df = primary_df.reindex(secondary_df.index)
            else:
                primary_df = primary_df.iloc[:m].copy()
                primary_df.index = secondary_df.index
        else:
            # Synthetic RangeIndex
            L = max(len(primary_df), len(secondary_df))
            synthetic = pd.RangeIndex(start=0, stop=L, step=1)
            primary_df.index = synthetic[:len(primary_df)]
            secondary_df.index = synthetic[:len(secondary_df)]

    # Optional resampling
    primary_df = resample_df(primary_df, resample)
    secondary_df = resample_df(secondary_df, resample)

    # Align to unified index
    unified_index = primary_df.index.union(secondary_df.index)
    primary_aligned = primary_df.reindex(unified_index)
    secondary_aligned = secondary_df.reindex(unified_index)

    # Override column names if provided
    if primary_cols is not None:
        names = [n.strip() for n in primary_cols.split(',')]
        if len(names) != len(primary_aligned.columns):
            raise ValueError(f"Expected {len(primary_aligned.columns)} primary names, got {len(names)}")
        primary_aligned.columns = names
    if secondary_cols is not None:
        names = [n.strip() for n in secondary_cols.split(',')]
        if len(names) != len(secondary_aligned.columns):
            raise ValueError(f"Expected {len(secondary_aligned.columns)} secondary names, got {len(names)}")
        secondary_aligned.columns = names

    # Ensure numeric
    primary_aligned = primary_aligned.apply(pd.to_numeric, errors='coerce')
    secondary_aligned = secondary_aligned.apply(pd.to_numeric, errors='coerce')

    # Detect peaks and overlaps
    primary_peaks = find_peaks_per_row(primary_aligned)
    secondary_peaks = find_peaks_per_row(secondary_aligned)
    overlap_mask = [ (len(primary_peaks[i])>0 and len(secondary_peaks[i])>0) for i in range(len(unified_index)) ]

    return dict(
        primary=primary_aligned,
        secondary=secondary_aligned,
        unified_index=unified_index,
        primary_peaks=primary_peaks,
        secondary_peaks=secondary_peaks,
        overlap_mask=np.array(overlap_mask, dtype=bool)
    )

# ----------------------- Interactive session helpers -----------------------

class ClickCollector:
    def __init__(self, ax, xlabel='Time', title='Click timestamps where overlaps occur; Press Done when finished'):
        self.ax = ax
        self.fig = ax.figure
        self.cid = None
        self.xs = []
        self.pts = []
        self.start_time = None
        self.end_time = None
        # Buttons
        self.done_ax = self.fig.add_axes([0.80, 0.01, 0.08, 0.05])
        self.clear_ax = self.fig.add_axes([0.70, 0.01, 0.08, 0.05])
        self.undo_ax = self.fig.add_axes([0.60, 0.01, 0.08, 0.05])
        self.btn_done = Button(self.done_ax, 'Done')
        self.btn_clear = Button(self.clear_ax, 'Clear')
        self.btn_undo = Button(self.undo_ax, 'Undo')
        self.btn_done.on_clicked(self._on_done)
        self.btn_clear.on_clicked(self._on_clear)
        self.btn_undo.on_clicked(self._on_undo)
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)

    def connect(self):
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self._onclick)
        self.start_time = time.perf_counter()  # monotonic timer

    def disconnect(self):
        if self.cid is not None:
            self.fig.canvas.mpl_disconnect(self.cid)
            self.cid = None

    def _onclick(self, event):
        # Ignore clicks outside the target axes
        if event.inaxes != self.ax:
            return

        # Ignore clicks while the toolbar is in pan or zoom mode
        toolbar = None
        # Cross-backend lookup for toolbar
        if hasattr(self.fig.canvas, 'manager') and hasattr(self.fig.canvas.manager, 'toolbar'):
            toolbar = self.fig.canvas.manager.toolbar
        elif hasattr(self.fig.canvas, 'toolbar'):
            toolbar = self.fig.canvas.toolbar

        if toolbar and getattr(toolbar, 'mode', ''):
            # mode is non-empty → a tool is active (pan or zoom); ignore this click
            return

        # Only accept left-button clicks
        if getattr(event, 'button', None) != 1:
            return

        x = event.xdata
        if x is None:
            return

        self.xs.append(x)
        # Visual marker
        p = self.ax.axvline(x, color='tab:red', alpha=0.6, linestyle='--')
        self.pts.append(p)
        self.fig.canvas.draw_idle()


    def _on_done(self, _):
        self.end_time = time.perf_counter()    # monotonic timer
        plt.close(self.fig)

    def _on_clear(self, _):
        self.xs.clear()
        for p in self.pts:
            p.remove()
        self.pts.clear()
        self.fig.canvas.draw_idle()

    def _on_undo(self, _):
        if self.xs:
            self.xs.pop()
        if self.pts:
            p = self.pts.pop()
            p.remove()
        self.fig.canvas.draw_idle()

# ----------------------- Visualization -----------------------

def draw_lineplot(ax, primary_df, secondary_df, title, xlabel, ylabel, legend_primary_prefix='Primary', legend_secondary_prefix='Secondary'):
    line_colors = {'primary': [], 'secondary': []}
    for col in primary_df.columns:
        line, = ax.plot(primary_df.index, primary_df[col], label=f"{legend_primary_prefix}: {col}")
        line_colors['primary'].append(line.get_color())
    for col in secondary_df.columns:
        line, = ax.plot(secondary_df.index, secondary_df[col], label=f"{legend_secondary_prefix}: {col}")
        line_colors['secondary'].append(line.get_color())
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    # Date formatting if datetime index
    if isinstance(primary_df.index, pd.DatetimeIndex) or isinstance(secondary_df.index, pd.DatetimeIndex):
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
    return line_colors


def draw_tap_view(primary_df, secondary_df, unified_index, primary_peaks, secondary_peaks,
                  title, xlabel, ylabel,
                  overlap_color='red', overlap_thickness=1.0,
                  show_overlap_markers=False):
    """Draw line plot + stacked peak timeline (no vertical overlap markers by default)."""
    # Top line plot
    num_vars = len(primary_df.columns) + len(secondary_df.columns)
    fig = plt.figure(figsize=(12, 6 + 0.3 * max(1, num_vars)))
    gs = fig.add_gridspec(2, 1, height_ratios=[6, num_vars], hspace=0.45)
    ax = fig.add_subplot(gs[0])
    line_colors = draw_lineplot(ax, primary_df, secondary_df, title, xlabel, ylabel)

    # Bottom timeline
    if TAP is not None and hasattr(TAP, 'add_peak_timeline'):
        TAP.add_peak_timeline(
            fig, gs, unified_index, primary_peaks, secondary_peaks,
            # Compute overlap mask but don't draw markers by default
            [(len(primary_peaks[i])>0 and len(secondary_peaks[i])>0) for i in range(len(unified_index))],
            list(primary_df.columns), list(secondary_df.columns), line_colors, ax,
            bottom_lw=2.0, right_lw=2.0, top_lw=2.0, left_lw=2.0,
            overlap_color=overlap_color, overlap_thickness=overlap_thickness,
            border_color='black', split_color='0.5', fill_color=None, fill_alpha=1.0,
            overlap_mode='default', bar_width_mode='local', bar_shrink=0.95
        )
    else:
        # Minimal fallback timeline: draw filled rectangles for peaks (primary on top, secondary below divider)
        ax2 = fig.add_subplot(gs[1], sharex=ax)
        ax2.set_ylim(-0.5, num_vars)
        ax2.invert_yaxis()
        ax2.axis('off')
        x_numeric = mdates.date2num(unified_index) if isinstance(unified_index, pd.DatetimeIndex) else np.arange(len(unified_index))
        n = len(unified_index)
        # local bar width
        def local_span(i, x, shrink=0.95):
            if n == 1:
                w = shrink * 1.0
                return x[0]-0.5*w, x[0]+0.5*w
            left_gap = (x[i]-x[i-1]) if i>0 else (x[1]-x[0])
            right_gap = (x[i+1]-x[i]) if i<n-1 else (x[-1]-x[-2])
            left_gap = max(left_gap, 1e-9)
            right_gap = max(right_gap, 1e-9)
            return x[i]-0.5*shrink*left_gap, x[i]+0.5*shrink*right_gap
        bar_height = 1.0
        num_primary = len(primary_df.columns)
        # Colors approximate line colors
        for i in range(n):
            xl, xr = local_span(i, x_numeric, 0.95)
            bw = xr - xl
            # primary
            for col_idx in primary_peaks[i]:
                y = col_idx
                ax2.add_patch(plt.Rectangle((xl, y), bw, bar_height, color='C0', ec=None, lw=0, zorder=2))
            # secondary
            for col_idx in secondary_peaks[i]:
                y = num_primary + col_idx
                ax2.add_patch(plt.Rectangle((xl, y), bw, bar_height, color='C1', ec=None, lw=0, zorder=2))
        # divider
        ax2.hlines(num_primary, xmin=x_numeric[0], xmax=x_numeric[-1], colors='0.5', linewidth=2)

    # Optionally draw overlap markers on top axis
    if show_overlap_markers:
        for i, ts in enumerate(unified_index):
            if len(primary_peaks[i])>0 and len(secondary_peaks[i])>0:
                ax.axvline(ts, color=overlap_color, alpha=0.15, linewidth=overlap_thickness)

    return fig, ax

# ----------------------- Metrics -----------------------


def snap_clicks_to_index(click_xs, ax, index):
    """
    Snap user click positions to the nearest value in 'index'.
    Handles datetime-like and numeric indices robustly.
    """
    # Helper: robust datetime check
    def _is_datetime_like(idx):
        if isinstance(idx, pd.DatetimeIndex):
            return True
        # dtype check
        if pd.api.types.is_datetime64_any_dtype(idx):
            return True
        # fallback: try converting; if many values convert, treat as datetime-like
        try:
            converted = pd.to_datetime(idx, errors='coerce')
            return converted.notna().sum() > 0
        except Exception:
            return False

    if _is_datetime_like(index):
        # Normalize index to numpy datetime64 for distance computation
        idx_vals = pd.to_datetime(index, errors='coerce').to_numpy('datetime64[ns]')
        snapped = []
        for x in click_xs:
            # Matplotlib date axis reports x as float (days); convert to datetime64[ns]
            dt = mdates.num2date(x)                        # datetime (naive)
            ts = np.datetime64(pd.Timestamp(dt).tz_localize(None))
            # nearest index position
            pos = int(np.argmin(np.abs(idx_vals - ts)))
            snapped.append(index[pos])
        return pd.Index(snapped)

    else:
        # Treat index as numeric: cast to float array for distance computation
        try:
            idx_vals = np.asarray(index, dtype=float)
        except Exception:
            # Last resort: use positional snapping if casting fails
            idx_vals = np.arange(len(index), dtype=float)

        snapped = []
        for x in click_xs:
            pos = int(np.argmin(np.abs(idx_vals - float(x))))
            snapped.append(index[pos])

def compute_metrics(snapped_clicks: pd.Index, index: pd.Index, overlap_mask: np.ndarray):
    # Unique user selections
    user_set = set(snapped_clicks.unique())
    # Ground truth set
    gt_set = set(index[overlap_mask])
    tp = len(user_set & gt_set)
    fp = len(user_set - gt_set)
    fn = len(gt_set - user_set)
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2*prec*rec/(prec+rec) if prec+rec>0 else 0.0
    jacc = tp / len(user_set | gt_set) if (user_set or gt_set) else 0.0
    return {
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'jaccard': jacc,
        'user_count': len(user_set),
        'gt_count': len(gt_set)
    }

# ----------------------- Main routine -----------------------

def main():
    p = argparse.ArgumentParser(description='TAP Self-Test Tool')
    p.add_argument('--primary', required=True)
    p.add_argument('--secondary', required=True)
    p.add_argument('--primary-sheet', default=0)
    p.add_argument('--secondary-sheet', default=0)
    p.add_argument('--secondary-date-format', default='%Y-%m')
    p.add_argument('--resample', default=None)
    p.add_argument('--delimiter', default=None)
    p.add_argument('--tz', default=None)
    p.add_argument('--primary-cols', default=None)
    p.add_argument('--secondary-cols', default=None)
    p.add_argument('--title', default='TAP Self-Test')
    p.add_argument('--xlabel', default='Time')
    p.add_argument('--ylabel', default='Values')
    p.add_argument('--out-dir', default='tap_selftest_output')
    p.add_argument('--save-figs', action='store_true')
    p.add_argument('--reverse-order', action='store_true', help='Show TAP view first, then line plot')
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = align_and_detect(
        primary_path=Path(args.primary),
        secondary_path=Path(args.secondary),
        primary_sheet=args.primary_sheet,
        secondary_sheet=args.secondary_sheet,
        secondary_date_format=args.secondary_date_format,
        resample=args.resample,
        delimiter=args.delimiter,
        tz=args.tz,
        primary_cols=args.primary_cols,
        secondary_cols=args.secondary_cols
    )

    primary = data['primary']
    secondary = data['secondary']

    unified_index = data['unified_index']
    primary_peaks = data['primary_peaks']
    secondary_peaks = data['secondary_peaks']
    overlap_mask = data['overlap_mask']

    sessions = []

    def run_lineplot_session():
        fig, ax = plt.subplots(figsize=(12, 6))
        draw_lineplot(ax, primary, secondary, title=f"{args.title} — Line Plot Only", xlabel=args.xlabel, ylabel=args.ylabel)
        collector = ClickCollector(ax)
        collector.connect()
        plt.show()
        collector.disconnect()
        return collector

    def run_tap_session():
        fig, ax = draw_tap_view(primary, secondary, unified_index, primary_peaks, secondary_peaks,
                                title=f"{args.title} — TAP (Dual View)", xlabel=args.xlabel, ylabel=args.ylabel,
                                show_overlap_markers=False)
        collector = ClickCollector(ax)
        collector.connect()
        plt.show()
        collector.disconnect()
        return collector

    order = ['line', 'tap']
    if args.reverse_order:
        order = ['tap', 'line']

    results = {}

    for mode in order:
        if mode == 'line':
            col = run_lineplot_session()
            label = 'lineplot'
        else:
            col = run_tap_session()
            label = 'tap'
        elapsed = (col.end_time - col.start_time) if (col.start_time and col.end_time) else None
        snapped = snap_clicks_to_index(col.xs, col.ax, unified_index)
        metrics = compute_metrics(snapped, unified_index, overlap_mask)
        results[label] = {
            'elapsed_seconds': elapsed,
            'n_clicks': len(col.xs),
            'n_snapped_unique': len(pd.Index(snapped).unique()),
            'metrics': metrics
        }
        # Save per-timestamp classification CSV
        sel_set = set(pd.Index(snapped).unique())
        rows = []
        for ts, gt in zip(unified_index, overlap_mask):
            rows.append({'time': ts, 'user_selected': ts in sel_set, 'ground_truth_overlap': bool(gt)})
        df_out = pd.DataFrame(rows)
        df_out.to_csv(out_dir / f'session_{label}_selections.csv', index=False)

    # Save summary JSON
    with open(out_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n=== TAP Self-Test Summary ===")
    for label in ['lineplot', 'tap']:
        if label in results:
            r = results[label]
            print(f"\nSession: {label}")
            print(f"  Time (s): {r['elapsed_seconds']:.2f}" if r['elapsed_seconds'] is not None else "  Time (s): n/a")
            m = r['metrics']
            print(f"  Precision: {m['precision']:.3f}  Recall: {m['recall']:.3f}  F1: {m['f1']:.3f}  Jaccard: {m['jaccard']:.3f}")
            print(f"  TP: {m['true_positives']}  FP: {m['false_positives']}  FN: {m['false_negatives']}")

if __name__ == '__main__':
    main()
