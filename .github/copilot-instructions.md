# Copilot instructions — MV_Financial_Dashboard_Dec2025

Short, actionable notes to help AI coding agents make useful, safe changes.

1) Big picture
- Single-file Streamlit dashboard: main app logic is in `app.py` (views, data engine,
  helpers, and PDF export live together). Treat `app.py` as the primary surface to edit.
- Data is currently simulated in-memory by `load_data_final()` (daily rows for 2024–2025).
  There is an `acquisition_history.csv` file in the repo root but it is not used by
  the app; do not assume file-backed data without explicit changes to `app.py`.

2) How to run locally
- Install dependencies: `pip install -r requirements.txt` (use a venv).
- Run the app with Streamlit: `streamlit run app.py` (this is the expected dev flow).

3) Key patterns to follow
- Caching: expensive data builders use `@st.cache_data`. When changing cached functions,
  restart the Streamlit process or call cache-clear routines so UI picks up changes.
- UI is organized into view functions: `show_home()`, `show_history()`, etc. Add
  sidebar controls and chart layout inside these functions. Example: to add a filter,
  edit the `st.expander("🔎 Filter & Baseline Comparison Settings")` block in
  `show_history()`.
- Chart helpers: use `plot_line_chart(df, x, y, title, color=None)` to produce
  Plotly figures that follow existing styling (markers, unified hover, white theme).
- CSV / PDF exports: use `convert_df_to_csv(df)` to prepare CSV bytes and
  `create_pdf(rec, kpi_data)` to produce a PDF bytes payload. `create_pdf` returns
  `pdf.output(dest='S').encode('latin-1')` — keep that encoding when returning files.

4) Repository-specific caveats & TODOs discovered
- Duplicate function: `load_data_final()` appears defined more than once inside
  `app.py`. Consolidate into a single implementation when refactoring.
- Asset references: `MagnaVitaLogo.jpeg` and other static assets are referenced
  but not present in the repo. Either add assets to the repo root or guard image
  loads with try/except (the code already has a fallback).
- `requirements.txt` contains some stdlib modules listed (e.g., `io`, `datetime`). Only
  install the third-party packages: `streamlit,pandas,numpy,plotly,pytz,fpdf,xlsxwriter`.

5) Integration points / future work
- There is an in-code comment indicating future integration with an "AlayaCare API." If
  you implement API fetching, add a separate data-loading function, apply `@st.cache_data`,
  and do not overwrite the simulated generator until integration is validated.

6) Typical edits and examples
- Add a new chart: create a helper similar to `plot_line_chart` and call it from the
  desired view. Keep layout responsibilities in view functions, computations in helpers.
- Add a sidebar control: update the `st.expander("🔎 Filter & Baseline Comparison Settings")`
  block and use the control values to filter `df_master` before passing to plotting helpers.
- Fix PDF content: change `create_pdf()` only to adjust visual layout or fields; it returns
  bytes ready for `st.download_button` or HTTP responses.

7) Testing & debugging
- Streamlit hot-reloads UI but may keep `@st.cache_data` entries — when in doubt,
  restart with `Ctrl-C` and re-run `streamlit run app.py`.
- Use `st.write()` to surface intermediate dataframes in a view during development.

8) PR expectations
- Keep changes minimal and self-contained. If you change data shapes (column names),
  update all helper usages (plot helpers, `get_historical_avg`, PDF table code).
- Update `README.md` if you change how the app is run or its public configuration.

If anything above is unclear or you want extra patterns (tests, CI, or modularization
guidance), tell me which area to expand and I'll update this file.
