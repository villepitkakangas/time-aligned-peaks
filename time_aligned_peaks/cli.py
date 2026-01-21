
import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .io import (
    read_table, parse_secondary_index, drop_running_number_column_if_no_datetime,
    apply_secondary_index_to_primary_by_position, resample_df, detect_and_fix_orientation
)
from .peaks import find_peaks_per_row, indices_to_names, peaks_to_binary_matrix
from .plotting import set_axis_frequency
from .timeline import add_peak_timeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--primary', required=True)
    parser.add_argument('--secondary', required=True)
    parser.add_argument('--primary-sheet', default=0)
    parser.add_argument('--secondary-sheet', default=0)
    parser.add_argument('--secondary-date-format', default='%Y-%m')
    parser.add_argument('--tz', default=None)
    parser.add_argument('--resample', default=None)
    parser.add_argument('--fill-missing', choices=['zero', 'none', 'ffill', 'bfill'], default='none')
    parser.add_argument('--output-peaks', default=None)
    parser.add_argument('--output-peak-matrix', default=None)
    parser.add_argument('--title', default='Combined Data Plot')
    parser.add_argument('--timeline-bottom-lw', type=float, default=2.5)
    parser.add_argument('--timeline-right-lw', type=float, default=2.5)
    parser.add_argument('--timeline-left-lw', type=float, default=2.5)
    parser.add_argument('--timeline-top-lw', type=float, default=2.5)
    parser.add_argument('--save-figure', default=None)
    parser.add_argument('--figure-format', default=None)
    parser.add_argument('--primary-cols', type=str, default=None)
    parser.add_argument('--secondary-cols', type=str, default=None)
    parser.add_argument('--delimiter', default=None)
    parser.add_argument('--primary-transposed', action='store_true')
    parser.add_argument('--secondary-transposed', action='store_true')
    parser.add_argument('--auto-detect-orientation', action='store_true')
    parser.add_argument('--force-orientation', type=str, default=None)
    parser.add_argument('--log-level', type=str, default='INFO')
    parser.add_argument('--legend-primary-prefix', default='Primary')
    parser.add_argument('--legend-secondary-prefix', default='Secondary')
    parser.add_argument('--xlabel', default='Time')
    parser.add_argument('--ylabel', default='Values')
    parser.add_argument('--overlap-style', choices=['none', 'lineplot', 'sync-timeline'], default='lineplot')
    parser.add_argument('--overlap-color', default='red')
    parser.add_argument('--overlap-thickness', type=float, default=1.0)
    parser.add_argument('--timeline-overlap-color', default='red')
    parser.add_argument('--timeline-overlap-thickness', type=float, default=1.0)
    parser.add_argument('--timeline-border-color', default='black')
    parser.add_argument('--timeline-split-color', default='0.5')
    parser.add_argument('--timeline-fill-color', default=None)
    parser.add_argument('--timeline-fill-alpha', type=float, default=1.0)
    parser.add_argument('--timeline-overlap-mode', choices=['default', 'inverse1', 'inverse2'], default='default')
    parser.add_argument('--timeline-bar-width-mode', choices=['local', 'median', 'min'], default='local')
    parser.add_argument('--timeline-bar-shrink', type=float, default=0.95)
    parser.add_argument('--dpi', type=int, default=300)

    args = parser.parse_args()

    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {args.log_level}')
    logging.basicConfig(level=numeric_level, format='%(asctime)s [%(levelname)s] %(message)s')

    primary_raw = read_table(Path(args.primary), sheet_name=args.primary_sheet, delimiter=args.delimiter)
    secondary_raw = read_table(Path(args.secondary), sheet_name=args.secondary_sheet, delimiter=args.delimiter)

    force_orientation = {}
    if args.force_orientation:
        for part in args.force_orientation.split():
            key, val = part.split('=')
            force_orientation[key.strip()] = val.strip()
    if 'primary' in force_orientation:
        primary_raw = primary_raw if force_orientation['primary'] == 'normal' else primary_raw.transpose()
    elif args.auto_detect_orientation:
        primary_raw = detect_and_fix_orientation(primary_raw, 'Primary')
    elif args.primary_transposed:
        primary_raw = primary_raw.transpose()

    if 'secondary' in force_orientation:
        secondary_raw = secondary_raw if force_orientation['secondary'] == 'normal' else secondary_raw.transpose()
    elif args.auto_detect_orientation:
        secondary_raw = detect_and_fix_orientation(secondary_raw, 'Secondary')
    elif args.secondary_transposed:
        secondary_raw = secondary_raw.transpose()

    secondary_df, sec_has_datetime = parse_secondary_index(
        secondary_raw, date_format=args.secondary_date_format, tz=args.tz
    )

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
    if not isinstance(primary_df.index, pd.DatetimeIndex):
        primary_df, dropped_col = drop_running_number_column_if_no_datetime(primary_df)
        if dropped_col:
            logging.info(f"Dropped running-number column: '{dropped_col}'")

    if not isinstance(primary_df.index, pd.DatetimeIndex):
        if sec_has_datetime:
            primary_df = apply_secondary_index_to_primary_by_position(primary_df, secondary_df.index)
        else:
            synthetic_index = pd.RangeIndex(start=0, stop=max(len(primary_df), len(secondary_df)), step=1)
            primary_df.index = synthetic_index[:len(primary_df)]
            secondary_df.index = synthetic_index[:len(secondary_df)]

    primary_df = resample_df(primary_df, args.resample)
    secondary_df = resample_df(secondary_df, args.resample)

    unified_index = primary_df.index.union(secondary_df.index)
    primary_aligned = primary_df.reindex(unified_index)
    secondary_aligned = secondary_df.reindex(unified_index)

    if args.primary_cols is not None:
        names = [name.strip() for name in args.primary_cols.split(',')]
        if len(names) != len(primary_aligned.columns):
            raise ValueError(f"Expected {len(primary_aligned.columns)} primary names, got {len(names)}")
        primary_aligned.columns = names
    if args.secondary_cols is not None:
        names = [name.strip() for name in args.secondary_cols.split(',')]
        if len(names) != len(secondary_aligned.columns):
            raise ValueError(f"Expected {len(secondary_aligned.columns)} secondary names, got {len(names)}")
        secondary_aligned.columns = names

    if args.fill_missing == 'zero':
        primary_aligned = primary_aligned.fillna(0)
        secondary_aligned = secondary_aligned.fillna(0)
    elif args.fill_missing == 'ffill':
        primary_aligned = primary_aligned.ffill()
        secondary_aligned = secondary_aligned.ffill()
    elif args.fill_missing == 'bfill':
        primary_aligned = primary_aligned.bfill()
        secondary_aligned = secondary_aligned.bfill()

    primary_aligned = primary_aligned.apply(pd.to_numeric, errors='coerce')
    secondary_aligned = secondary_aligned.apply(pd.to_numeric, errors='coerce')

    import matplotlib.gridspec as gridspec
    num_vars = len(primary_aligned.columns) + len(secondary_aligned.columns)
    fig = plt.figure(figsize=(12, 6 + 0.3 * num_vars))
    gs = gridspec.GridSpec(2, 1, height_ratios=[6, num_vars], hspace=0.45)

    ax = fig.add_subplot(gs[0])
    line_colors = {'primary': [], 'secondary': []}
    for col in primary_aligned.columns:
        line, = ax.plot(primary_aligned.index, primary_aligned[col], label=f"{args.legend_primary_prefix}: {col}")
        line_colors['primary'].append(line.get_color())
    for col in secondary_aligned.columns:
        line, = ax.plot(secondary_aligned.index, secondary_aligned[col], label=f"{args.legend_secondary_prefix}: {col}")
        line_colors['secondary'].append(line.get_color())
    ax.set_title(args.title)
    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    set_axis_frequency(ax, unified_index)

    primary_peaks = find_peaks_per_row(primary_aligned)
    secondary_peaks = find_peaks_per_row(secondary_aligned)

    overlap_mask = [
        (len(primary_peaks[i]) > 0 and len(secondary_peaks[i]) > 0)
        for i in range(len(unified_index))
    ]
    for i, overlap in enumerate(overlap_mask):
        if not overlap:
            continue
        if args.overlap_style == 'none':
            continue
        elif args.overlap_style == 'lineplot':
            ax.axvline(unified_index[i], color=args.overlap_color, alpha=0.15, linewidth=args.overlap_thickness)
        elif args.overlap_style == 'sync-timeline':
            ax.axvline(unified_index[i], color=args.timeline_overlap_color, linewidth=args.timeline_overlap_thickness)

    add_peak_timeline(
        fig, gs, unified_index, primary_peaks, secondary_peaks, overlap_mask,
        list(primary_aligned.columns), list(secondary_aligned.columns),
        line_colors, ax, args.timeline_bottom_lw, args.timeline_right_lw,
        args.timeline_top_lw, args.timeline_left_lw,
        args.timeline_overlap_color, args.timeline_overlap_thickness,
        args.timeline_border_color, args.timeline_split_color,
        args.timeline_fill_color, args.timeline_fill_alpha, args.timeline_overlap_mode,
        bar_width_mode=args.timeline_bar_width_mode, bar_shrink=args.timeline_bar_shrink
    )

    if args.save_figure:
        fmt = args.figure_format if args.figure_format else Path(args.save_figure).suffix.lstrip('.').lower()
        plt.savefig(args.save_figure, dpi=args.dpi, bbox_inches='tight', format=fmt)
        logging.info(f"Figure saved to {args.save_figure}")
    plt.show()

    if args.output_peaks is not None:
        rows = []
        for i, ts in enumerate(unified_index):
            sec_names = indices_to_names(secondary_peaks[i], secondary_aligned.columns)
            pri_names = indices_to_names(primary_peaks[i], primary_aligned.columns)
            rows.append({
                'row': i,
                'time': ts,
                'secondary_peak_columns': ', '.join(sec_names),
                'primary_peak_columns': ', '.join(pri_names),
                'overlap': overlap_mask[i]
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

if __name__ == '__main__':
    main()
