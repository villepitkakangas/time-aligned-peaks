
import pandas as pd
import numpy as np
from pathlib import Path

# Create examples directory
Path('examples').mkdir(exist_ok=True)

# Generate a small monthly datetime index
dates = pd.date_range('2024-01-01', periods=18, freq='MS')  # Month start

# Primary: two sensors/series
rng = np.random.default_rng(42)
primary_df = pd.DataFrame({
    'date': dates.strftime('%Y-%m'),  # string month-year to mimic CSV common format
    'P1': (np.sin(np.linspace(0, 3*np.pi, len(dates))) * 10 + 20 + rng.normal(0, 1.5, len(dates))).round(2),
    'P2': (np.cos(np.linspace(0, 3*np.pi, len(dates))) * 7 + 15 + rng.normal(0, 1.2, len(dates))).round(2),
})
primary_df.to_csv('examples/primary.csv', index=False)

# Secondary: two indicators/series, stored in Excel with a date column formatted as %Y-%m
secondary_df = pd.DataFrame({
    'date': dates.strftime('%Y-%m'),
    'S1': (np.sin(np.linspace(0.5, 3.5*np.pi, len(dates))) * 8 + 18 + rng.normal(0, 1.0, len(dates))).round(2),
    'S2': (np.cos(np.linspace(0.5, 3.5*np.pi, len(dates))) * 6 + 12 + rng.normal(0, 1.1, len(dates))).round(2),
})
secondary_df.to_excel('examples/secondary.xlsx', index=False)

# Also include a tiny README to explain formats
readme = '''
Example data for TAP
====================

primary.csv
-----------
Columns:
- date: string formatted as %Y-%m (e.g., 2024-01, 2024-02, ...)
- P1, P2: numeric series (simulated sensor values)

secondary.xlsx
--------------
Sheet 0
Columns:
- date: string formatted as %Y-%m (e.g., 2024-01, 2024-02, ...)
- S1, S2: numeric series (simulated indicator values)

These files are small synthetic datasets intended for demos and the self-test tool.
They are intentionally month-granular to align with the CLI example that uses
--secondary-date-format "%Y-%m".
'''
Path('examples/README.txt').write_text(readme)

print('Wrote examples/primary.csv and examples/secondary.xlsx')
