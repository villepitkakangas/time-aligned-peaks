from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

from time_aligned_peaks.io import read_table, parse_secondary_index
from time_aligned_peaks.plotting import set_axis_frequency
from time_aligned_peaks.peaks import find_peaks_per_row
from time_aligned_peaks.timeline import add_peak_timeline

# Read input datasets
primary_raw = read_table(Path("examples/synthetic_primary.csv"))
secondary_raw = read_table(Path("examples/synthetic_secondary.csv"),
                          sheet_name="Sheet 1")

# Parse secondary index to match times
secondary_df, sec_has_datetime = parse_secondary_index(
    secondary_raw, date_format='%Y-%m', tz=None
)

# Merge and align datasets
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

unified_index = primary_df.index.union(secondary_df.index)
primary_aligned = primary_df.reindex(unified_index)
secondary_aligned = secondary_df.reindex(unified_index)

primary_aligned = primary_aligned.apply(pd.to_numeric, errors='coerce')
secondary_aligned = secondary_aligned.apply(pd.to_numeric, errors='coerce')

# Calculate the total number of variables and initialise visualisation
num_vars = len(primary_aligned.columns) + len(secondary_aligned.columns)
fig = plt.figure(figsize=(12, 6 + 0.3 * num_vars))
gs = gridspec.GridSpec(2, 1, height_ratios=[6, num_vars], hspace=0.45)

# Top panel
ax = fig.add_subplot(gs[0])
line_colors = {'primary': [], 'secondary': []}
for col in primary_aligned.columns:
    line, = ax.plot(primary_aligned.index, primary_aligned[col],
                    label=f"Primary: {col}")
    line_colors['primary'].append(line.get_color())
for col in secondary_aligned.columns:
    line, = ax.plot(secondary_aligned.index, secondary_aligned[col],
                    label=f"Secondary: {col}")
    line_colors['secondary'].append(line.get_color())

# Labels
ax.set_title("Combined Data Plot")
ax.set_xlabel("Time")
ax.set_ylabel("Values")
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
set_axis_frequency(ax, unified_index)

# Peak detection
primary_peaks = find_peaks_per_row(primary_aligned)
secondary_peaks = find_peaks_per_row(secondary_aligned)

# Get peak overlaps
overlap_mask = [
    (len(primary_peaks[i]) > 0 and len(secondary_peaks[i]) > 0)
    for i in range(len(unified_index))
]
# Draw peak overlaps to top panel
for i, overlap in enumerate(overlap_mask):
    if not overlap:
        continue
    ax.axvline(unified_index[i], color="red", linewidth=1.0)

# Bottom panel
add_peak_timeline(
    fig, gs, unified_index, primary_peaks, secondary_peaks, overlap_mask,
    list(primary_aligned.columns), list(secondary_aligned.columns),
    line_colors, ax, 1, 1,
    1, 1,
    "red", 1,
    "black", "0.5",
    None, 1.0, "default",
    bar_width_mode="local", bar_shrink=1.0
)

plt.show()
