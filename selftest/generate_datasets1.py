
import pandas as pd
import numpy as np
from pathlib import Path

Path('examples').mkdir(exist_ok=True)

dates = pd.date_range('2024-01-01', periods=18, freq='MS')  # Month start
rng = np.random.default_rng(42)

# Base signals
P1 = (np.sin(np.linspace(0, 3*np.pi, len(dates))) * 10 + 20 + rng.normal(0, 1.5, len(dates)))
P2 = (np.cos(np.linspace(0, 3*np.pi, len(dates))) * 7 + 15 + rng.normal(0, 1.2, len(dates)))
S1 = (np.sin(np.linspace(0.5, 3.5*np.pi, len(dates))) * 8 + 18 + rng.normal(0, 1.0, len(dates)))
S2 = (np.cos(np.linspace(0.5, 3.5*np.pi, len(dates))) * 6 + 12 + rng.normal(0, 1.1, len(dates)))

# Inject aligned spikes at the same timestamps to guarantee overlaps
overlap_idx = [5, 10, 14]  # months where both groups peak
for i in overlap_idx:
    # create local maxima: add spike at i (neighbors unchanged)
    P1[i] += 8.0; P2[i] += 6.0
    S1[i] += 7.0; S2[i] += 5.0

primary_df = pd.DataFrame({
    'date': dates.strftime('%Y-%m'),
    'P1': np.round(P1, 2),
    'P2': np.round(P2, 2),
})
primary_df.to_csv('examples/primary1.csv', index=False)

secondary_df = pd.DataFrame({
    'date': dates.strftime('%Y-%m'),
    'S1': np.round(S1, 2),
       'S2': np.round(S2, 2),
})
secondary_df.to_excel('examples/secondary1.xlsx', index=False)


Path('examples/README1.txt').write_text(
    "Synthetic examples with guaranteed overlaps at indices: " + str(overlap_idx) + "\n" + \
    "Files: primary1.csv secondary1.xlsx"
)