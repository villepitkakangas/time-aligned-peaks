
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
