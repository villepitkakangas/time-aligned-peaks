
# Time-Aligned Peaks (TAP)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/1139156471.svg)](https://doi.org/10.5281/zenodo.18328885)

## Table of Contents
- Statement of Need
- Features
- Installation
- Quick Start
- Command-line Usage
- Command-line Arguments
- Programmatic Usage
- Using the Makefile
- Testing
- Design Evaluation
- Downstream Reuse (ML & Analytics)
- Support & Maintenance
- Citation
- Provenance & Relationship to Previous Work
- Licence

## Statement of Need
Time-Aligned Peaks (TAP) is a Python tool for analysing **peak co-occurrence** across multiple time series. It augments standard line plots with a synchronised timeline panel that shows, per timestamp, which **primary** and **secondary** series have detected **peaks**, and highlights **overlaps** (coincident peaks across groups). TAP addresses a gap in existing visualisation libraries by providing a **compact, reproducible representation** of peak dynamics and exporting machine-readable artefacts for downstream analysis.

## Features
- Robust table ingestion (CSV/TSV/Excel) with delimiter sniffing and sheet selection.
- Orientation handling (auto-detect row/column transposition) and timestamp negotiation.
- Optional resampling and missing-data strategies.
- Simple, explainable slope-change peak detection.
- Combined figure output (main line plot + peak timeline) and two CSV exports: a long-form report and a **binary peak matrix**.
- Customizable timeline bar width modes (`local|median|min`) and shrink factor.

## Requirements
- Python 3.9+
- numpy >= 1.23
- pandas >= 2.0
- matplotlib >= 3.5
- openpyxl >= 3.1 (for Excel support)

## Installation (from source)
Currently, only development install is supported:
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e .
```

## Quick Start
```bash
python -m pip install -e .

time-aligned-peaks   --primary examples/synthetic_primary.csv   --secondary examples/synthetic_secondary.csv   --secondary-date-format "%Y-%m"   --timeline-bar-width-mode local   --timeline-bar-shrink 0.95   --save-figure demo.png   --output-peaks peaks_report.csv   --output-peak-matrix peaks_matrix.csv
```

## Command-line Usage
General:
```bash
time-aligned-peaks   --primary examples/synthetic_primary.csv   --secondary examples/synthetic_secondary.csv   --secondary-date-format "%Y-%m"   --title "Synthetic demo"   --save-figure output_demo.png   --output-peaks peaks_demo.csv   --output-peak-matrix peaks_matrix.csv
```
POSIX shells (Linux/macOS, WSL, Git Bash):
```bash
time-aligned-peaks \
  --primary examples/synthetic_primary.csv \
  --secondary examples/synthetic_secondary.csv \
  --secondary-date-format "%Y-%m" \
  --title "Synthetic demo" \
  --save-figure output_demo.png \
  --output-peaks peaks_demo.csv \
  --output-peak-matrix peaks_matrix.csv
```
Windows PowerShell (recommended on Windows):
```bash
time-aligned-peaks `
  --primary examples/synthetic_primary.csv `
  --secondary examples/synthetic_secondary.csv `
  --secondary-date-format "%Y-%m" `
  --title "Synthetic demo" `
  --save-figure output_demo.png `
  --output-peaks peaks_demo.csv `
  --output-peak-matrix peaks_matrix.csv
```
Windows Command Prompt (cmd.exe):
```bash
time-aligned-peaks ^
  --primary examples\synthetic_primary.csv ^
  --secondary examples\synthetic_secondary.csv ^
  --secondary-date-format "%%Y-%%m" ^
  --title "Synthetic demo" ^
  --save-figure output_demo.png ^
  --output-peaks peaks_demo.csv ^
  --output-peak-matrix peaks_matrix.csv
```


## Command-line Arguments
>NOTE: COLOR can be a colour name (e.g., "red") a float, or a tuple of floats ranging from 0.0 to 1.0.

Positional overview
- `--primary`: Path to the primary table (CSV/TSV/XLS/XLSX).
- `--secondary`: Path to the secondary table (CSV/TSV/XLS/XLSX).

File reading & orientation
- `--primary-sheet INT or STR` — Excel sheet index/name for primary (default: 0).
- `--secondary-sheet INT or STR` — Excel sheet index/name for secondary (default: 0).
- `--delimiter CHAR` — Explicit CSV/TSV delimiter; otherwise auto‑detected.
- `--primary-transposed` — Treat primary as transposed (columns are observations).
- `--secondary-transposed` — Treat secondary as transposed.
- `--auto-detect-orientation` — Auto‑detect and fix orientation from timestamp heuristics.
- `--force-orientation "primary=normal|transposed secondary=normal|transposed"` — Override orientation explicitly.

Indexing, time parsing & alignment
- `--secondary-date-format STR` — Parse the first column of the secondary table as dates using this format (default: "%Y-%m"). If parsing completely fails, that column is dropped.
- `--tz STR` — Localise/convert parsed timestamps to timezone (e.g., "Europe/Helsinki").
- `--resample STR` — Resample by Pandas offset alias (e.g., D, W, M) only when a DatetimeIndex exists; mean aggregation is used.
- If the primary table has no dates:
  - When the secondary table has dates, **secondary timestamps are assigned to primary by row position**.
  - Otherwise, both are aligned to a synthetic RangeIndex.

Column selection & naming
- `--primary-cols "A,B,C"` — Comma‑separated column names for primary; the count must match detected columns.
- `--secondary-cols "X,Y"` — Comma‑separated column names for secondary; the count must match detected columns.

Missing‑value policy
- `--fill-missing {none,zero,ffill,bfill} — Missing‑value handling (default: none).
  - `none`: Missing values are not handled.
  - `zeros`: Missing values are replaced with zeros.
  - `ffill`: Missing values are replaced with the last non-missing value.
  - `bfill`: Missing values are replaced with the next non-missing value.

Plot basics
- `--title STR` — Figure title (default: "Combined Data Plot").
- `--xlabel STR` — X‑axis label (default: "Time"). --ylabel STR (default: "Values").
- `--legend-primary-prefix STR` — Legend prefix for primary (default: "Primary").
- `--legend-secondary-prefix STR` — Legend prefix for secondary (default: "Secondary").

Figure resolution
- `--dpi INT` — Controls the resolution of raster output formats (PNG/JPG). It is ignored for vector formats (SVG/PDF).

Overlap markers (main plot)
- `--overlap-style {none,lineplot,sync-timeline}` — Marker style in the **top** line plot (default: lineplot).
   - `lineplot`: faint vertical lines at overlap timestamps (uses `--overlap-color` and `--overlap-thickness`).
   - `sync-timeline`: uses the same colour/thickness as the timeline overlap settings.
   - `none`: no markers.
- `--overlap-color COLOR` — Colour for main‑plot overlap markers (default: red).
- `--overlap-thickness FLOAT` — Line thickness for main‑plot overlap markers (default: 1.0).

Timeline (bottom panel)
- Frame & fill
  - `--timeline-border-color COLOR` (default: black)
  - `--timeline-split-color COLOR` (default: 0.5)
  - `--timeline-fill-color COLOR` (default: none = transparent)
  - `--timeline-fill-alpha FLOAT` (default: 1.0)
  - `--timeline-bottom-lw|right-lw|left-lw|top-lw FLOAT` — Border widths (default: 2.5 each).
- Overlap markers inside timeline
  - `--timeline-overlap-color COLOR` (default: red)
  - `--timeline-overlap-thickness FLOAT` (default: 1.0)
  - `--timeline-overlap-mode {default,inverse1,inverse2}` (default: default)
    - `default`: always use the default colour
    - `inverse1`: invert the marker colour **inside** any peak rectangle.
    - `inverse2`: invert only when the marker colour would be too similar to the series colour (combined RGB/brightness heuristic).
- Bar widths & spacing
  - `--timeline-bar-width-mode {local,median,min}` (default: local)
    - `local`: width derived from **adjacent tick spacing** per timestamp (per-timestamp width from local gaps). 
    - `median|min`: global width from the median/minimum inter‑tick spacing (global `median(diff)` or `min(diff)` width).
  - `--timeline-bar-shrink FLOAT` (default: 0.95) — Scale factor (0–1) to avoid touching seams and to create **gaps** between bars for clarity.

> Notes: The bottom panel’s y‑axis is fixed for navigation clarity; the x‑axis follows the main plot. Series order is **primary on top** (in legend order), then **secondary**.

Outputs
- `--save-figure PATH` — Save the combined figure (top line plot + bottom timeline).
- `--figure-format {png,pdf,svg,...}` — Explicit format; otherwise inferred from PATH extension.
- `--output-peaks PATH` — Save a **peak report CSV** with, per timestamp: row index, time, comma‑separated secondary/primary peak column names, and a boolean overlap flag.
- `--output-peak-matrix PATH` — Save a **binary peak matrix CSV** with columns time, per‑series 0/1 peak presence, and an overlap column (1 if any primary & any secondary peak co‑occur).

Logging
- `--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}` — Default: INFO.

## Exit Status
The script raises errors (non‑zero exit) on invalid options (e.g., name count mismatch in --primary-cols / --secondary-cols) or unsupported file types.

## Programmatic Usage
```python
import pandas as pd
from time_aligned_peaks import find_peaks_per_row, peaks_to_binary_matrix

# Build or load aligned DataFrames primary_df, secondary_df and an index
unified_index = primary_df.index.union(secondary_df.index)
primary_aligned = primary_df.reindex(unified_index)
secondary_aligned = secondary_df.reindex(unified_index)

p_peaks = find_peaks_per_row(primary_aligned)
s_peaks = find_peaks_per_row(secondary_aligned)

matrix_df = peaks_to_binary_matrix(
    unified_index, p_peaks, s_peaks,
    list(primary_aligned.columns), list(secondary_aligned.columns), include_overlap=True
)
```

## Using the Makefile
```bash
make demo        # local per-timestamp bar widths
make demo_median # global median width
make demo_min    # global minimum width
make test        # run unit tests
```

## Testing

A test suite covering the core functionality is included in the `tests/` folder.

1. Create and activate a virtual environment (recommended):
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate
```   
2. Install test dependencies:
```bash
pip install -r requirements-tests.txt
```
3. Run the test suite from the repository root:
```bash
pytest -q
```
Tests include both unit and integration tests. All tests have been verified on Python 3.9, 3.10, and 3.13 on Windows.

## Design Evaluation
The `selftest/` folder contains an optional usability test tool and example outputs. It is not required for normal operation.

## Downstream Reuse (ML & Analytics)

TAP matrices are written as standard CSV files and can be loaded directly into common analysis and ML toolkits.  
Example (pandas + scikit-learn):

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load a TAP matrix
df = pd.read_csv("output/matrix.csv", index_col=0)

# Convert to numeric features
X = df.values

# Example: compute correlations (or train a simple model)
corr = df.corr()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Correlation matrix:")
print(corr)
```

## Documentation
See CONTRIBUTION.md for contribution guidelines and CODE_OF_CONDUCT.md for the code of conduct.

## Support & Maintenance
This software is provided as-is. If you encounter bugs or questions, please open a GitHub Issue. Issues are handled on a best-effort basis.

## Citation
If you use Time-Aligned Peaks (TAP) in academic work, please cite:
Ville Pitkäkangas, *Time-Aligned Peaks (TAP)*, software package, 2026.  
See `CITATION.cff` for the preferred citation.

DOI: 10.5281/zenodo.18328885

## Provenance & Relationship to Previous Work
Time‑Aligned Peaks (TAP) is an independent software implementation created by the author. TAP generalises and extends a static visualisation concept (“piikkivisualisointi”) developed by the author within the LYHTY project at Centria University of Applied Sciences (2021).
No LYHTY project code, assets, figures, data, or proprietary material are reused in TAP.
All TAP source code is original and released under the MIT Licence.

## Licence
MIT Licence (see LICENSE-MIT file).
