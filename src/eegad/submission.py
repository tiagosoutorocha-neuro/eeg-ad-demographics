"""Submission material: TRIPOD+AI checklist pre-filled with where each item is answered, draft declarations,
the preprint (medRxiv) package and the shareable zip of aggregated results."""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import pandas as pd

READY, PARTIAL, WRITE, NA = "ready in the notebooks", "partial", "write in the manuscript", "not applicable (say why)"

# (section, item, applies to, item summary, where in the manuscript, where in the notebooks, status)
TRIPOD_AI = [
    ("Title", "1", "D;E", "Identify development/evaluation of a multivariable prediction model, target population and outcome",
     "Title", "working title in config/analysis_plan.json", PARTIAL),
    ("Abstract", "2", "D;E", "See TRIPOD+AI for Abstracts", "Abstract", "—", WRITE),
    ("Introduction", "3a", "D;E", "Healthcare context and rationale, including existing models",
     "Introduction", "README (References); AHEPA benchmark", PARTIAL),
    ("Introduction", "3b", "D;E", "Target population, intended purpose in the care pathway, intended users",
     "Introduction", "—", WRITE),
    ("Introduction", "3c", "D;E", "Known health inequalities between sociodemographic groups",
     "Introduction", "education (RQ1, RQ2); Latin American cohort", PARTIAL),
    ("Introduction", "4", "D;E", "Objectives, including whether the study develops, evaluates or both",
     "Introduction", "README (Objective, RQ1-RQ3)", READY),
    ("Methods — data", "5a", "D;E", "Data sources for development and evaluation, rationale and representativeness",
     "Methods", "notebooks 00 and 01; dataset DOIs", READY),
    ("Methods — data", "5b", "D;E", "Dates of data collection, start and end of accrual",
     "Methods", "dataset descriptors (Prado 2023; Miltiadous 2023)", WRITE),
    ("Methods — participants", "6a", "D;E", "Study setting, number and location of centres",
     "Methods", "Table 1 (BrainLat sites); ds004504: one centre", READY),
    ("Methods — participants", "6b", "D;E", "Eligibility criteria", "Methods", "notebook 01; flow table F0 and fig0", READY),
    ("Methods — participants", "6c", "D;E", "Treatments received and how they were handled",
     "Methods", "not available in the datasets: state it", WRITE),
    ("Methods — data preparation", "7", "D;E", "Pre-processing and quality checks, and whether similar across groups",
     "Methods", "notebook 01 (19 channels, 40 Hz band, epochs); S1; S2", READY),
    ("Methods — outcome", "8a", "D;E", "Outcome definition, how and when assessed, consistency across groups",
     "Methods", "clinical diagnosis of each dataset (descriptors)", PARTIAL),
    ("Methods — outcome", "8b", "D;E", "Qualifications of outcome assessors if subjective",
     "Methods", "dataset descriptors", WRITE),
    ("Methods — outcome", "8c", "D;E", "Blinding of outcome assessment",
     "Methods", "diagnosis made before and independently of the features; state it", WRITE),
    ("Methods — predictors", "9a", "D", "How initial predictors were chosen; any pre-selection",
     "Methods", "analysis plan: the 11 features of the conference paper, fixed in advance", READY),
    ("Methods — predictors", "9b", "D;E", "Definition of all predictors, how and when measured, blinding",
     "Methods", "src/eegad/features.py; demographics in notebook 01", READY),
    ("Methods — predictors", "9c", "D;E", "Qualifications of predictor assessors if subjective",
     "Methods", "automatic features", NA),
    ("Methods — sample size", "10", "D;E", "How study size was arrived at and why it is sufficient",
     "Methods", "sample = available datasets; CI widths in the summary", WRITE),
    ("Methods — missing data", "11", "D;E", "Handling of missing data and reasons for omitted data",
     "Methods", "complete cases; F0; S7", READY),
    ("Methods — analysis", "12a", "D", "How data were used and partitioned, given sample size",
     "Methods", "LOSO (notebook 02); transport (notebook 05)", READY),
    ("Methods — analysis", "12b", "D", "Handling of predictors (functional form, rescaling, standardisation)",
     "Methods", "RF without scaling needs; score as logit per SD (src/eegad/stats.py)", READY),
    ("Methods — analysis", "12c", "D", "Model type, rationale, building steps, hyperparameters, internal validation",
     "Methods", "analysis plan (fixed hyperparameters); notebook 02; SVM/XGBoost sensitivity (notebook 06)", READY),
    ("Methods — analysis", "12d", "D;E", "Heterogeneity across clusters: handling and quantification",
     "Methods", "RQ1c (cohort x EEG, notebook 03); site check (notebook 03)", READY),
    ("Methods — analysis", "12e", "D;E", "All performance measures and plots, with rationale; model comparisons",
     "Methods", "AUC, delta AUC, OR/SD, calibration slope, CITL, Brier, threshold metrics", READY),
    ("Methods — analysis", "12f", "E", "Model updating (e.g., recalibration) after evaluation",
     "Methods", "no updating; per-cohort standardisation is secondary (notebook 05)", READY),
    ("Methods — analysis", "12g", "E", "How predictions were calculated for evaluation",
     "Methods", "frozen models in outputs/models/; predictions in outputs/predictions/", READY),
    ("Methods — class imbalance", "13", "D;E", "Class-imbalance methods, why and how; recalibration",
     "Methods", "balanced class weights in the RF (analysis plan); no resampling", READY),
    ("Methods — fairness", "14", "D;E", "Approaches to address model fairness",
     "Methods", "confounding by education (RQ2); fairness not formally assessed: state it", PARTIAL),
    ("Methods — model output", "15", "D", "Model output and how classification thresholds were chosen",
     "Methods", "probability; 0.5 threshold descriptive only", READY),
    ("Methods — training vs evaluation", "16", "D;E", "Differences between development and evaluation data",
     "Methods", "Table 1; S2 (domain shift)", READY),
    ("Methods — ethics", "17", "D;E", "Ethics committee and informed consent or waiver",
     "Methods", "preprint_package/DECLARATIONS_draft.md", WRITE),
    ("Open science", "18a", "D;E", "Funding source and role of funders", "Declarations", "—", WRITE),
    ("Open science", "18b", "D;E", "Conflicts of interest and financial disclosures", "Declarations", "—", WRITE),
    ("Open science", "18c", "D;E", "Where the protocol can be accessed, or that none was prepared",
     "Methods", "config/analysis_plan.json + provenance/original_run (hash and freeze time)", READY),
    ("Open science", "18d", "D;E", "Registration details, or that the study was not registered",
     "Methods", "not externally registered: state it", WRITE),
    ("Open science", "18e", "D;E", "Availability of study data", "Declarations", "dataset DOIs; BrainLat access terms", PARTIAL),
    ("Open science", "18f", "D;E", "Availability of analytical code", "Declarations",
     "this repository; archive a release on Zenodo (DOI)", PARTIAL),
    ("Patient and public involvement", "19", "D;E", "Patient and public involvement, or state none",
     "Methods", "none (secondary data): state it", WRITE),
    ("Results — participants", "20a", "D;E", "Participant flow, numbers with and without the outcome (diagram)",
     "Results; Fig. 1", "notebook 01: fig0_participant_flow; F0", READY),
    ("Results — participants", "20b", "D;E", "Participant characteristics by data source, predictors, missing data",
     "Results; Table 1", "T1; T1b; T1c; S7", READY),
    ("Results — participants", "20c", "E", "Compare predictors and outcome between evaluation and development data",
     "Results", "T1; S2 (SMD between cohorts)", READY),
    ("Results — model development", "21", "D;E", "Number of participants and outcome events in each analysis",
     "Results", "F0 (n per analysis); T2", READY),
    ("Results — model specification", "22", "D", "Full model (code, object or API) for predictions and evaluation",
     "Declarations", "outputs/models/*.pkl and *.json; this repository", READY),
    ("Results — performance", "23a", "D;E", "Performance estimates with confidence intervals, incl. key subgroups",
     "Results; Table 2", "T2; fig1-fig3", READY),
    ("Results — performance", "23b", "D;E", "Heterogeneity in performance across clusters, if examined",
     "Results", "RQ1c; performance per cohort", READY),
    ("Results — model updating", "24", "E", "Results of any model updating", "Results",
     "no updating; per-cohort standardisation secondary", NA),
    ("Discussion", "25", "D;E", "Overall interpretation, including fairness, versus objectives and previous studies",
     "Discussion", "readings by the plan's rules (summary)", WRITE),
    ("Discussion", "26", "D;E", "Limitations: representativeness, size, overfitting, missing data; effects on bias",
     "Discussion", "README (Limitations); S7; T1b", PARTIAL),
    ("Discussion", "27a", "D", "How poor-quality or unavailable input data should be handled in practice",
     "Discussion", "—", WRITE),
    ("Discussion", "27b", "D", "Whether users interact with inputs/model and the expertise required",
     "Discussion", "—", WRITE),
    ("Discussion", "27c", "D;E", "Next steps for research toward applicability and generalisability",
     "Discussion", "—", WRITE),
]


def tripod_table():
    return pd.DataFrame(TRIPOD_AI, columns=["section", "item", "applies to", "item summary (TRIPOD+AI)",
                                            "where in the manuscript", "where in the notebooks / source", "status"])


def write_tripod(out: Path):
    t = tripod_table()
    t.to_csv(out / "tables" / "TRIPOD_AI_checklist.csv", index=False, encoding="utf-8-sig")
    md = ["# TRIPOD+AI — pre-filled checklist", "",
          "Reference: Collins GS et al. BMJ 2024;385:e078378 — https://doi.org/10.1136/bmj-2023-078378", "",
          "Item summaries only; for submission use the official form with manuscript page numbers.", "",
          "| " + " | ".join(t.columns) + " |", "|" + "---|" * len(t.columns)]
    md += ["| " + " | ".join(str(v).replace("|", "/") for v in r) + " |" for r in t.itertuples(index=False)]
    (out / "tables" / "TRIPOD_AI_checklist.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return t


def declarations_draft(plan_hash, frozen_at, labels_read_at, original):
    return f"""# Draft declarations — review and complete the [FILL IN] fields

## Ethics statement
This study is a secondary analysis of de-identified data from two datasets: BrainLat (Prado et al., 2023,
https://doi.org/10.1038/s41597-023-02806-8) and the OpenNeuro dataset ds004504 v1.0.9
(https://doi.org/10.18112/openneuro.ds004504.v1.0.9; Miltiadous et al., 2023, https://doi.org/10.3390/data8060095).
Ethics approval and informed consent for the original data collection were obtained by the original investigators, as
described in the data descriptors. [FILL IN: position of the local research ethics committee on secondary analysis of
public, de-identified data. Confirm with the committee before writing any waiver.]

## Data availability
ds004504 is openly available on OpenNeuro (https://doi.org/10.18112/openneuro.ds004504.v1.0.9). BrainLat data are
available from the BrainLat consortium under its data access terms (https://doi.org/10.1038/s41597-023-02806-8)
[FILL IN: exact link and access conditions].

## Code availability
All analyses can be reproduced with the code at [FILL IN: GitHub URL and Zenodo DOI]. The analysis plan was frozen
before any diagnostic label of the Greek cohort was read: original specification hash {original['hash']}, frozen on
{original['frozen_at']} (this is an internal, time-stamped analysis plan, not an external preregistration). The public
code is a refactored, English-language reproduction of that analysis (plan hash {plan_hash}, frozen on {frozen_at};
Greek labels first read on {labels_read_at}, after the freeze); notebook 08 compares its results with the original run.

## Reporting guideline
Reported following TRIPOD+AI (Collins et al., 2024, https://doi.org/10.1136/bmj-2023-078378); checklist in the
Supplementary Material.

## Funding
[FILL IN: funding sources and role of the funders, or "This research received no specific funding."]

## Competing interests
[FILL IN: payments or services from third parties in the past 36 months, or "The authors declare no competing interests."]

## Author contributions
[FILL IN: CRediT]

## Prior presentation
[FILL IN if an earlier conference paper on BrainLat is published: cite it and state what is new here — second cohort,
independent cross-cohort tests, confounding removed by design and calibration under transport.]
"""


PREPRINT_README = """# Posting the preprint on medRxiv

1. Write the manuscript and produce a single PDF with a title page (title, authors, affiliations and abstract), text,
   figures and tables. Supplementary material goes in separate files: the S* tables, T1b, T1c, S7 and the TRIPOD+AI
   checklist (CSV or MD).
2. All authors must agree to the deposit.
3. Complete DECLARATIONS_draft.md: ethics, competing interests, funding and data availability are mandatory on medRxiv.
4. Archive the code (this repository, tagged release) on Zenodo to obtain a DOI and put it in the code statement.
5. Licence: CC BY is the simplest and is compatible with open-access publication agreements that require CC BY.
6. Submit at https://www.medrxiv.org/submit-a-manuscript (subject area: Neurology). Each depositing author needs an
   account.
7. medRxiv does not post manuscripts already accepted or published. Post before journal acceptance; submitting to the
   journal at the same time is allowed.
8. Once posted, cite the preprint DOI in the cover letter and in the declarations.
9. If a conference paper with the BrainLat analysis is published, cite it in the preprint and the manuscript and state
   what is new.

Contents of this folder:
- figures/: fig0 (participant flow), fig1 (ROC), fig2 (incremental OR), fig3 (confounding), PDF and PNG
- tables/: T1, T1b (severity), T1c (MMSE stages), T2, T2b, F0 (flow), S1-S7, TRIPOD_AI_checklist
- supplementary/: results_summary.md, plan_frozen.json, run_log.json, environment.txt, ds004504_download.json
"""

# Aggregated outputs only: nothing at the subject level (BrainLat data-use terms).
AGGREGATED_TABLES = ("T1_participants.csv", "T1b_severity_by_cohort.csv", "T1c_greek_mmse_stages.csv",
                     "T2_main_results.csv", "T2b_confounding_by_design.csv", "F0_participant_flow.csv",
                     "S1_brainlat_channel_map_1020.csv", "S2_features_by_cohort.csv",
                     "S3_balance_BrainLat_matching.csv", "S3_balance_BrainLat_overlap.csv",
                     "S3_balance_Greek_overlap.csv", "S4_exploratory_severity.csv", "S5_sensitivity.csv",
                     "S6_features_beyond_demographics.csv", "S7_brainlat_included_vs_excluded.csv",
                     "TRIPOD_AI_checklist.csv", "TRIPOD_AI_checklist.md", "null_bootstrap_BrainLat.json",
                     "null_bootstrap_Greek.json", "confounding_design.json")
SUPPLEMENTARY = ("results_summary.md", "plan_frozen.json", "run_log.json", "environment.txt", "ds004504_download.json")


def build_preprint_package(out: Path, declarations: str):
    pkg = out / "preprint_package"
    if pkg.exists():
        shutil.rmtree(pkg)
    for sub in ("figures", "tables", "supplementary"):
        (pkg / sub).mkdir(parents=True, exist_ok=True)
    for f in sorted((out / "figures").glob("*")):
        if f.suffix in (".pdf", ".png"):
            shutil.copy2(f, pkg / "figures" / f.name)
    for name in AGGREGATED_TABLES:
        if (out / "tables" / name).exists():
            shutil.copy2(out / "tables" / name, pkg / "tables" / name)
    for name in SUPPLEMENTARY:
        if (out / name).exists():
            shutil.copy2(out / name, pkg / "supplementary" / name)
    (pkg / "DECLARATIONS_draft.md").write_text(declarations, encoding="utf-8")
    (pkg / "README_PREPRINT.md").write_text(PREPRINT_README, encoding="utf-8")
    f_zip = out / "preprint_package.zip"
    with zipfile.ZipFile(f_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(pkg.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(out))
    return pkg, f_zip


def build_share_zip(out: Path):
    """outputs/share.zip: aggregated results only (safe to send to co-authors)."""
    f_zip = out / "share.zip"
    with zipfile.ZipFile(f_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for name in SUPPLEMENTARY:
            if (out / name).exists():
                z.write(out / name, name)
        for name in ("key_results.json", "verdicts.json", "alerts.json"):
            if (out / "results" / name).exists():
                z.write(out / "results" / name, f"results/{name}")
        for name in AGGREGATED_TABLES:
            if (out / "tables" / name).exists():
                z.write(out / "tables" / name, f"tables/{name}")
        for f in sorted((out / "figures").glob("*")):
            if f.suffix in (".pdf", ".png"):
                z.write(f, f"figures/{f.name}")
        if (out / "reproduction_check.csv").exists():
            z.write(out / "reproduction_check.csv", "reproduction_check.csv")
    return f_zip
