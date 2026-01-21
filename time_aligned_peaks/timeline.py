
import numpy as np
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.colors import to_rgb


def invert_color(color):
    r, g, b = to_rgb(color)
    return (1 - r, 1 - g, 1 - b)


def brightness(color):
    r, g, b = to_rgb(color)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def is_combined_similar(color1, color2, rgb_threshold=0.6, brightness_threshold=0.2):
    c1 = np.array(to_rgb(color1))
    c2 = np.array(to_rgb(color2))
    rgb_distance = np.linalg.norm(c1 - c2)
    brightness_diff = abs(brightness(color1) - brightness(color2))
    return rgb_distance < rgb_threshold or brightness_diff < brightness_threshold


def add_peak_timeline(
    fig, gs, unified_index, primary_peaks, secondary_peaks, overlap_mask,
    primary_cols, secondary_cols, line_colors, ax, bottom_lw, right_lw,
    top_lw, left_lw, overlap_color, overlap_thickness, border_color, split_color,
    fill_color, fill_alpha, overlap_mode,
    bar_width_mode='local', bar_shrink=0.95
):
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
