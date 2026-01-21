
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


def set_axis_frequency(ax, index):
    """Set date tick locators/formatters and rotate labels if crowded."""
    if not isinstance(index, pd.DatetimeIndex):
        return
    freq = pd.infer_freq(index)
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    if freq:
        f = freq.upper()
        if f.startswith('M'):
            locator, formatter = mdates.MonthLocator(), mdates.DateFormatter('%Y-%m')
        elif f.startswith('D'):
            locator, formatter = mdates.DayLocator(), mdates.DateFormatter('%Y-%m-%d')
        elif f.startswith('A') or f.startswith('Y'):
            locator, formatter = mdates.YearLocator(), mdates.DateFormatter('%Y')
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.figure.canvas.draw()
    if len(ax.get_xticklabels()) > 12:
        plt.setp(ax.get_xticklabels(), rotation=90)
