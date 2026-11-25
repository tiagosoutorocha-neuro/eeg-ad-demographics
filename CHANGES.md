# Changes

## 1.1.0 — 2026-11-25

Additions only: notebooks 00–08, the analysis plan (hash `d694ec84c0e48616`), every pre-specified result and the 546
key numbers compared by notebook 08 are unchanged.

**Post hoc amendment 1 — recalibration of the transported models (notebook 09)**
- Decided after the transport results (RQ3) were seen: the frozen models ranked participants well in the other cohort
  but their probabilities were miscalibrated. Asks whether a local recalibration (intercept only; intercept and slope),
  estimated in the target cohort without retraining the model, corrects the probabilities and the decisions at
  P(AD) = 0.5. Leave-one-subject-out within the target cohort; 2,000 stratified bootstrap resamples.
- Written in `config/post_hoc_plan.json`, which states the hash of the plan it amends and is frozen in
  `outputs/post_hoc/` on its first run. It reads only the predictions saved by notebook 05 and writes only to
  `outputs/post_hoc/` (Table S7, recalibrated predictions, `recalibration.json`).

**Post hoc amendment 2 — hyperparameters, group spectra, montage error (notebooks 10 and 11)**
- `config/post_hoc_plan_2.json`, frozen the same way (`outputs/post_hoc/post_hoc_plan_2_frozen.json`).
- Notebook 10: the pipeline re-run with 10 Random Forest configurations (one hyperparameter changed at a time); every
  configuration is reported and none is chosen (Table S11, Fig. S5). The frozen configuration must reproduce the
  predictions of notebooks 02 and 05 exactly, or the notebook stops.
- Notebook 11 (descriptive): the EEG files are read again with the same preprocessing code as the features to draw the
  group-average spectra (Fig. S2; only group curves go to the shareable file), and the electrode positions behind
  Table S1 are saved for the montage error map (Fig. S1).
- `features.py`: the preprocessing of `features_19ch` moved, unchanged, into `preprocessed_epochs`, which both
  notebooks 01 and 11 use; the features are bitwise identical (checked on synthetic recordings).

**Figures and tables of the article (notebook 12)**
- Seven main figures (study design, participant flow, ROC curves, one figure per research question, feature
  consistency) and five supplementary figures (montage error, group spectra, E-value, domain shift, hyperparameters),
  at journal sizes, sans-serif lettering, PDF with embedded fonts and PNG at 300 dpi; Fig. 1 also as SVG.
- Every table in one Word file, `outputs/article/article_tables.docx`, numbered in the order of first citation:
  Table 1 (related work, to be completed by hand; the row of this study is filled), Table 2 (acquisition,
  preprocessing by the data providers and harmonisation; new), Table 3 (participants), Table 4 (main results),
  Table 5 (confounding by design); Tables S1–S12, including the feature definitions (Table S2, new); the TRIPOD+AI
  checklist as an unnumbered supplementary file; draft figure legends.
- The analysis plan and both amendments are copied to `outputs/article/supplementary_files/`.
- New dependency: `python-docx` 1.2.0, used only for the Word file (without it, an HTML file is written instead).

**Running**
- `run_all.py` runs 00 → 12 by default and sets no time limit per cell by default (the null bootstrap of notebook 03
  can run for more than 6 hours on real data).
- `make_synthetic.py` accepts `--small`/`--fdt` (as used by the tests and the README) as well as `-small`/`-fdt`.

## 1.0.0 — 2026-11-23

First public version: a refactored English translation of the original analysis notebook (Portuguese, run on
2026-11-20; see `provenance/original_run/`).

**Translation and structure (no analytic change)**
- One notebook split into nine (`00`–`08`) plus the `eegad` package; notebooks exchange data through files in
  `outputs/` (exact float round-trip), not kernel state.
- Analysis plan, variable names, feature names (`Razao_*` -> `Ratio_*`), tables, figures and messages in English.
- Paths come from `config/paths.yaml` / environment variables instead of being written in the code.
- The ds004504 manifest moved to `config/ds004504_v1.0.9_manifest.json`.
- New: `scripts/run_all.py`, `scripts/make_synthetic.py`, unit and end-to-end tests, notebook 08 (reproduction check);
  notebook 00 warns when the library versions differ from those of the original run.
- Verified on identical synthetic data, in quick test mode and with the full settings of the analysis plan: bit-for-bit
  identical features, out-of-fold and transport predictions, null-bootstrap statistics and matched pairs, and identical
  values for all 546 key numbers of the original outputs.

**Intentional changes after the original run (descriptive outputs only)**
- Table S7 (included vs excluded BrainLat participants) now compares participants of the *same diagnostic group*
  (excluded controls vs included controls). The original table compared the excluded participants with all included
  participants, mixing AD and controls, which made the excluded group look cognitively different when it was not.
- The results summary no longer lists the identifiers of excluded BrainLat participants (they stay in
  `outputs/derived/`, which is not shared).
- Shared outputs (`share.zip`, `preprint_package.zip`) contain aggregated results only.
