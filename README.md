# EDUFAIR
The repo is a small Python/Streamlit app for exam score normalization and audit. It ingests a totals CSV with student, shift, and raw score data, computes shift-level percentiles, turns those into preliminary and final ranks, and can break ties using subject percentiles or date of birth. The user-facing interface, including CSV upload, sample-data generation, rank tables, per-student explanation, and CSV download, is in app.py.

The reusable normalization logic is also mirrored in edufair_core.py. In practice, this looks like a prototype or audit tool for comparing candidates fairly across shifts rather than a general-purpose app.

