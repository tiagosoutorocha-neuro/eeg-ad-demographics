"""All tables of the article in one Word file (outputs/article/article_tables.docx), formatted for the manuscript
(three-line tables, captions, notes), plus draft figure legends. Built only from files already in outputs/.

If python-docx is not installed, the same content is written as HTML (Word opens it too)."""
from __future__ import annotations

import datetime
import html
from pathlib import Path

import numpy as np
import pandas as pd

from .article_figures import FEATURE_LABELS

COVARIATE_NAMES = (("Age + sex_fem + years_education + site_cl", "age + sex + education + site"),
                   ("Age + sex_fem + years_education", "age + sex + education"),
                   ("Age + sex_fem", "age + sex"), ("Age + years_education", "age + education"))


# ---- formatting helpers ---------------------------------------------------------------------------------------------
def _read(out: Path, name: str) -> pd.DataFrame | None:
    f = out / "tables" / name
    # as text, exactly as written by the notebooks (keeps trailing zeros such as 0.500)
    return pd.read_csv(f, encoding="utf-8-sig", dtype=str, keep_default_na=False) if f.exists() else None


def _num(x, nd=3):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x)
    return "" if not np.isfinite(x) else f"{x:.{nd}f}"


def _ci(a, lo, hi, nd=3):
    return f"{_num(a, nd)} [{_num(lo, nd)}, {_num(hi, nd)}]"


def _clean_text(s):
    if not isinstance(s, str):
        return "" if s is None or (isinstance(s, float) and np.isnan(s)) else str(s)
    for a, b in COVARIATE_NAMES:
        s = s.replace(a, b)
    return s


def _blank(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype(object).where(df.notna(), "").map(_clean_text)


def _feature(f):
    return FEATURE_LABELS.get(f, f)


# ---- the tables -----------------------------------------------------------------------------------------------------
C1_C7 = [
    ("C1", "Subject-level independence of the outer evaluation",
     "Leave-one-subject-out within each cohort; transport between cohorts trains on one cohort only", "Methods; nb 02, 05"),
    ("C2", "Fully nested pipeline (no data-driven step outside the training folds)",
     "No feature or channel selection and no tuning: features, channels and hyperparameters fixed in the plan (their "
     "sensitivity is shown post hoc in Table S11); RMS normalisation within each recording. Secondary RQ3 analysis "
     "standardises features within each cohort without labels (declared)", "Methods; analysis plan"),
    ("C3", "Segmentation and unit of analysis",
     "Non-overlapping 4-s epochs; training at the epoch level; subject P(AD) = mean over epochs; all epochs used",
     "Methods; Table 2"),
    ("C4", "Transparency of the preprocessing",
     "What each data provider did and what this study applied are listed side by side (Table 2): 19 channels of the "
     "10-20 system (BioSemi correspondence in Table S1 and Fig. S1); 0.5-45 Hz FIR; Greek data low-passed at 40 Hz "
     "(Butterworth, order 8) like the distributed BrainLat; average reference; no epoch rejection",
     "Methods; Table 2"),
    ("C5", "Reproducibility of the validation protocol",
     "Analysis plan frozen before the Greek diagnoses were read; code, seeds and settings public; full re-run "
     "reproduced all 546 key numbers", "Methods; code availability"),
    ("C6", "Within-subject structure exploited", "Epochs kept as separate training samples", "Methods"),
    ("C7", "Robust outer evaluation", "Leave-one-subject-out (every participant held out once) plus external transport",
     "Methods"),
]

# What the data providers did, from their data descriptors (Prado et al., 2023, Sci Data 10:889; Miltiadous et al.,
# 2023, Data 8:95). Checked against the papers on 2026-11-25.
DISTRIBUTED = [
    ("Source", "BrainLat (Prado et al., 2023): AD and controls from Argentina and Chile",
     "ds004504 v1.0.9 (Miltiadous et al., 2023): AHEPA University Hospital, Thessaloniki"),
    ("Condition", "Resting state, eyes closed, 10 min", "Resting state, eyes closed"),
    ("Recording system", "BioSemi Active-Two, 128 active Ag/AgCl electrodes", "Nihon Kohden EEG 2100, 19 electrodes "
                                                                              "of the 10-20 system"),
    ("Reference at recording", "Linked mastoids", "A1 and A2 (mastoids); Cz for common-mode rejection"),
    ("Processing by the data provider",
     "Average reference; Butterworth band-pass 0.5-40 Hz (order 8, zero phase); ICA correction of blinks and eye "
     "movements; malfunctioning channels interpolated (3.2 ± 1.1 per recording)",
     "Butterworth band-pass 0.5-45 Hz; re-referenced to A1-A2; artifact subspace reconstruction (0.5-s windows, "
     "SD 17); ICA (RunICA), components labelled eye or jaw by ICLabel removed"),
]
SAME = "Same as BrainLat"


def _median_iqr(v, nd=1):
    v = pd.to_numeric(pd.Series(v), errors="coerce").dropna()
    if v.empty:
        return ""
    q1, q2, q3 = np.percentile(v, [25, 50, 75])
    return f"{q2:.{nd}f} [{q1:.{nd}f}, {q3:.{nd}f}]"


def acquisition_table(out: Path, plan: dict) -> tuple[pd.DataFrame, list] | None:
    """Table 2: what each data provider did (left of the harmonisation) and what this study applied (same code for
    both cohorts), with the recordings actually analysed. Numbers from outputs/derived and outputs/tables."""
    fb, fg = out / "derived" / "brainlat_subjects.csv", out / "derived" / "greek_subjects.csv"
    if not (fb.exists() and fg.exists()):
        return None
    sb, sg = pd.read_csv(fb), pd.read_csv(fg)
    s1 = _read(out, "S1_brainlat_channel_map_1020.csv")
    prep = plan["preprocessing"]
    hb = prep["greek_band_harmonization"]
    ang = (f" (mean {pd.to_numeric(s1.angle_deg).mean():.1f}°, maximum {pd.to_numeric(s1.angle_deg).max():.1f}°; "
           "Table S1, Fig. S1)") if s1 is not None else ""
    rows, groups = [], []

    def add(group, field, bl, gr):
        rows.append({"": field, "BrainLat": bl, "Greek cohort (ds004504)": gr})
        groups.append(group)

    g1, g2, g3 = ("As distributed (data descriptors)", "Applied in this study (the same code for both cohorts)",
                  "Recordings analysed")
    for field, bl, gr in DISTRIBUTED:
        add(g1, field, bl, gr)
    add(g1, "Sampling rate", f"{sb.sfreq.median():g} Hz", f"{sg.sfreq.median():g} Hz")
    add(g2, "Channels", f"Nearest BioSemi electrode to each of the 19 positions of the 10-20 system{ang}",
        "The 19 recorded electrodes")
    add(g2, "Reference", f"Average of the {len(plan['channels_1020'])} channels", SAME)
    add(g2, "Pass band", "FIR 0.5-45 Hz (zero phase), on data distributed at 0.5-40 Hz",
        f"Butterworth low-pass {hb['h_freq']:g} Hz (order {hb['order']}, zero phase), as distributed in BrainLat; "
        "then FIR 0.5-45 Hz (zero phase)")
    add(g2, "Segmentation", f"Non-overlapping {prep['epoch_s']:g}-s epochs; no epoch rejection (epochs with a "
                            "non-finite feature are dropped)", SAME)
    add(g2, "Scaling", "Division by the root mean square of the whole recording (the features do not depend on scale)",
        SAME)
    add(g2, "Spectral estimate", f"Welch, {prep['welch_resolution_hz']:g}-Hz resolution, mean of the 19 channels "
                                 "(Table S2)", SAME)
    add(g3, "Participants (AD / HC)", f"{int((sb.y == 1).sum())} / {int((sb.y == 0).sum())}",
        f"{int((sg.y == 1).sum())} / {int((sg.y == 0).sum())}")
    add(g3, "Recording analysed, min, median [IQR]", _median_iqr(sb.duration_min), _median_iqr(sg.duration_min))
    add(g3, "4-s epochs per participant, median [IQR]", _median_iqr(sb.epochs_used, 0), _median_iqr(sg.epochs_used, 0))
    add(g3, "Epochs analysed (dropped)", f"{int(sb.epochs_used.sum()):,} ({int(sb.epochs_invalid.sum())})",
        f"{int(sg.epochs_used.sum()):,} ({int(sg.epochs_invalid.sum())})")

    def marks(s):
        m = pd.to_numeric(s.boundary_marks, errors="coerce").fillna(0)
        return (f"{int((m > 0).sum())} of {len(s)} (median {m[m > 0].median():.0f} per file)" if (m > 0).any()
                else "none")
    add(g3, "Files with discontinuities ('boundary' marks)", marks(sb), marks(sg))
    return pd.DataFrame(rows), groups


def feature_definitions(plan: dict) -> pd.DataFrame:
    """Table S2: the 11 features as computed in features.py."""
    from .features import BANDS, CONN_BANDS

    band = lambda b: f"{BANDS[b][0]:g}-{BANDS[b][1]:g} Hz"
    cband = lambda b: f"{CONN_BANDS[b][0]:g}-{CONN_BANDS[b][1]:g} Hz"
    n = len(plan["channels_1020"])
    pairs = n * (n - 1) // 2
    rel = ("Power in the band divided by the power in 0.5-45 Hz (trapezoidal integration of the Welch spectrum "
           f"averaged over the {n} channels)")
    plv = (f"Phase-locking value of each of the {pairs} channel pairs, averaged; phase from a 4th-order Butterworth "
           "band-pass (zero phase) and the Hilbert transform")
    rows = [("Rel_Theta_mean", rel, band("Theta")), ("Rel_Alpha_mean", rel, band("Alpha")),
            ("Rel_Beta_mean", rel, band("Beta")), ("Rel_Gamma_mean", rel, band("Gamma")),
            ("Ratio_Theta_Alpha", "Relative θ power divided by relative α power", "θ, α"),
            ("Ratio_Theta_Beta", "Relative θ power divided by relative β power", "θ, β"),
            ("Spectral_Entropy", f"Shannon entropy (bits) of the channel-averaged spectrum normalised to sum 1",
             "0.5-45 Hz"),
            ("PLV_Theta_mean", plv, cband("Theta")), ("PLV_Alpha_mean", plv, cband("Alpha")),
            ("PLV_Beta_mean", plv, cband("Beta")),
            ("Corr_Broadband_mean", f"Mean absolute Pearson correlation over the {pairs} channel pairs",
             "0.5-45 Hz")]
    return pd.DataFrame([{"feature": _feature(f), "definition": d, "band": b} for f, d, b in rows
                         if f in plan["features"]["all"]])


RELATED_COLUMNS = ["study", "data (AD/HC)", "train/test split", "demographic baseline", "confounding control",
                   "external cohort", "calibration", "main result (validation)"]


def related_work_table(out: Path, keys: dict) -> pd.DataFrame:
    """Table 1: placeholder with the columns of the related-work table and the row of this study (filled from the
    results); the other rows are written by hand from the full texts."""
    def k(name):
        v = keys.get(name, {}).get("value")
        return float(v) if v is not None else np.nan

    t1 = _read(out, "T1_participants.csv")
    n = {}
    if t1 is not None:
        for _, r in t1[t1["variable"] == "n"].iterrows():
            n["BrainLat" if r["cohort"].startswith("BrainLat") else "Greek"] = f"{r['AD']}/{r['HC']}"
    this = {"study": "This study",
            "data (AD/HC)": f"BrainLat ({n.get('BrainLat', '?')}); ds004504 ({n.get('Greek', '?')})",
            "train/test split": "Leave-one-subject-out within each cohort",
            "demographic baseline": "Yes: incremental value beyond age, sex (and education in BrainLat), null-bootstrap "
                                    "p",
            "confounding control": "Matched pairs, comparable pairs, overlap weights",
            "external cohort": "Yes: each frozen model applied once to the other cohort",
            "calibration": "Yes: slope, calibration-in-the-large, Brier",
            "main result (validation)": f"AUC {_num(k('auc.brainlat.eeg.value'))} (BrainLat) and "
                                        f"{_num(k('auc.greek.eeg.value'))} (Greek), LOSO; transported "
                                        f"{_num(k('rq3a.auc.value'))} and {_num(k('rq3b.auc.value'))}"}
    todo = {c: "" for c in RELATED_COLUMNS}
    todo["study"] = "[to be completed from the full texts]"
    return pd.DataFrame([todo, this], columns=RELATED_COLUMNS)


def build_blocks(out: Path, ctx_info: dict) -> list:
    """Document model: list of (kind, payload) with kinds heading, paragraph, table, legend. Tables are numbered in
    the order of first citation in the manuscript (see the guide to figures and tables)."""
    from .reporting import read_json

    plan = ctx_info.get("plan")
    kf = out / "results" / "key_results.json"
    keys = read_json(kf) if kf.exists() else {}
    B = []
    add = B.append
    add(("title", "Tables and figure legends — What Does Resting-State EEG Add to Demographics in Alzheimer's "
                  "Disease? Incremental Value, Confounding, and Transportability Across Two Cohorts"))
    add(("paragraph", f"Generated {datetime.datetime.now():%Y-%m-%d %H:%M} from {out} · analysis plan "
                      f"{ctx_info['plan_hash']} · mode: {ctx_info['mode']}"))
    if ctx_info["mode"] != "full":
        add(("warning", "QUICK TEST MODE: these numbers only check that the code runs. Do not use them."))

    add(("heading", "Main tables"))
    add(("table", {"label": "Table 1", "caption": "How previous studies evaluated resting-state EEG classifiers of "
                                                   "Alzheimer's disease (related work).",
                   "df": related_work_table(out, keys), "landscape": True,
                   "note": "To be completed by hand from the full texts (one row per study, citing the original "
                           "study); 'not reported' when a study does not say. The metric is always shown next to its "
                           "validation scheme. The row of this study is filled from the results."}))
    if plan is not None:
        acq = acquisition_table(out, plan)
        if acq is not None:
            df, groups = acq
            add(("table", {"label": "Table 2", "caption": "EEG acquisition, preprocessing by the data providers and the "
                                                           "harmonisation applied in this study.",
                           "df": df, "groups": groups, "landscape": True,
                           "note": "Differences that the harmonisation cannot remove: recording system, original "
                                   "reference and artefact cleaning (ICA in both cohorts; artifact subspace "
                                   "reconstruction only in the Greek data, which leaves the discontinuities counted "
                                   "in the last row). Epochs were not dropped at discontinuities. IQR, interquartile "
                                   "range."}))
    t1 = _read(out, "T1_participants.csv")
    if t1 is not None:
        t = _blank(t1)
        add(("table", {"label": "Table 3", "caption": "Participants with analysable EEG, by cohort and diagnosis.",
                       "df": t[["variable", "AD", "HC", "p", "test"]].rename(columns={"variable": ""}),
                       "groups": list(t["cohort"]),
                       "note": "Mean ± SD (n with the score when incomplete) or n (%). p: Mann–Whitney or Fisher "
                               "test. AD, Alzheimer's disease; HC, healthy controls; MoCA, Montreal Cognitive "
                               "Assessment; IFS, INECO Frontal Screening; MMSE, Mini-Mental State Examination."}))
    t2 = _read(out, "T2_main_results.csv")
    if t2 is not None:
        t = _blank(t2)
        add(("table", {"label": "Table 4", "caption": "Main results: incremental value (RQ1), discrimination, "
                                                       "confounding by design (RQ2) and transportability (RQ3).",
                       "df": t[["cohort", "analysis", "estimate", "p", "reading"]], "groups": list(t["block"]),
                       "landscape": True,
                       "note": "OR/SD: odds ratio of AD per 1 SD of the EEG score beyond the demographics (profile "
                               "likelihood 95% CI, descriptive). Null bootstrap: p from the parametric bootstrap that "
                               "re-runs the whole leave-one-subject-out procedure under the null (sequential stopping "
                               "of Besag and Clifford); analytic: naive likelihood-ratio p. Readings follow the rules "
                               "fixed in the analysis plan; 'not demonstrated' never means 'no information'."}))
    t2b = _read(out, "T2b_confounding_by_design.csv")
    if t2b is not None:
        t = pd.DataFrame({"cohort": t2b["cohort"], "analysis": t2b["analysis"],
                          "AUC or concordance [95% CI]": [_ci(a, l, h) for a, l, h in zip(t2b.AUC, t2b.lo, t2b.hi)],
                          "n": t2b["n"]})
        add(("table", {"label": "Table 5", "caption": "Discrimination of the EEG score with the confounders removed by "
                                                       "design (RQ2).", "df": _blank(t),
                       "note": "Reading rule: survives if the 95% CI lies above 0.5. ESS, effective sample size "
                               "(AD/HC) after overlap weighting. Covariate-adjusted AUC (Janes and Pepe) is a check "
                               "outside the reading rule."}))

    add(("heading", "Supplementary tables"))
    s1 = _read(out, "S1_brainlat_channel_map_1020.csv")
    if s1 is not None:
        t = pd.DataFrame({"10-20 position": s1.iloc[:, 0], "BioSemi electrode (BrainLat)": s1.iloc[:, 1],
                          "angle (°)": [_num(a, 1) for a in s1.iloc[:, 2]]})
        add(("table", {"label": "Table S1", "caption": "Correspondence between the BioSemi-128 electrodes of BrainLat "
                                                        "and the 19 positions of the 10-20 system.", "df": t,
                       "note": "Nearest electrode on the unit sphere with a one-to-one assignment (Hungarian "
                               "algorithm); angle between the electrode and the standard position (Fig. S1)."}))
    if plan is not None:
        add(("table", {"label": "Table S2", "caption": "The 11 EEG features.", "df": feature_definitions(plan),
                       "note": "Computed for every 4-s epoch on the 19 harmonised channels; δ (0.5-4 Hz) enters only "
                               "the denominator of the relative powers. The classifier is trained on the epoch values; "
                               "subject-level summaries are means over epochs."}))
    s7 = _read(out, "S7_brainlat_included_vs_excluded.csv")
    if s7 is not None:
        add(("table", {"label": "Table S3", "caption": "BrainLat: participants without analysable EEG compared with "
                                                        "the included participants of the same diagnostic group.",
                       "df": _blank(s7)}))
    t1b, t1c = _read(out, "T1b_severity_by_cohort.csv"), _read(out, "T1c_greek_mmse_stages.csv")
    if t1b is not None:
        add(("table", {"label": "Table S4a", "caption": "Severity of cognitive impairment by cohort.",
                       "df": _blank(t1b), "note": "Hedges g with 95% CI (AD − HC). The scales differ: the MoCA is "
                                                  "harder than the MMSE, so equal scores do not mean equal "
                                                  "severity."}))
    if t1c is not None:
        add(("table", {"label": "Table S4b", "caption": "Greek AD group by MMSE stage (Perneczky et al., 2006).",
                       "df": _blank(t1c)}))
    s2 = _read(out, "S2_features_by_cohort.csv")
    if s2 is not None:
        s2 = s2.rename(columns={s2.columns[0]: "feature"})
        t = pd.DataFrame({"feature": [_feature(f) for f in s2["feature"]],
                          **{c: [_num(v, 3) for v in s2[c]] for c in s2.columns[1:-1]},
                          "SMD between cohorts (HC)": [_num(v, 2) for v in s2[s2.columns[-1]]]})
        add(("table", {"label": "Table S5", "caption": "EEG features by cohort and diagnosis (median per epoch) and "
                                                        "difference between the cohorts in the controls.", "df": t,
                       "note": "SMD: standardised mean difference of the subject means, Greek − BrainLat, in the "
                               "controls; |SMD| > 0.8 indicates a large domain shift (Fig. S4)."}))
    for name, cap in (("S3_balance_BrainLat_matching.csv", "BrainLat, 1:1 matching on education"),
                      ("S3_balance_BrainLat_overlap.csv", "BrainLat, overlap weights"),
                      ("S3_balance_Greek_overlap.csv", "Greek cohort, overlap weights")):
        s3 = _read(out, name)
        if s3 is not None:
            t = s3.copy()
            for c in t.columns[1:]:
                t[c] = [_num(v, 2) for v in t[c]]
            add(("table", {"label": "Table S6" + {"S3_balance_BrainLat_matching.csv": "a",
                                                  "S3_balance_BrainLat_overlap.csv": "b",
                                                  "S3_balance_Greek_overlap.csv": "c"}[name],
                           "caption": f"Covariate balance before and after adjustment: {cap}.", "df": t,
                           "note": "SMD: standardised mean difference AD − HC; |SMD| < 0.1 is usually taken as "
                                   "good balance."}))
    rec = out / "post_hoc" / "recalibration_table.csv"
    if rec.exists():
        t = pd.read_csv(rec, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        note_file = out / "post_hoc" / "recalibration_note.txt"
        add(("table", {"label": "Table S7", "caption": "Post hoc: local recalibration of the transported models.",
                       "df": _blank(t), "landscape": True,
                       "note": note_file.read_text(encoding="utf-8") if note_file.exists() else ""}))
    s6 = _read(out, "S6_features_beyond_demographics.csv")
    if s6 is not None:
        s6 = s6.rename(columns={s6.columns[0]: "feature"})
        s6["feature"] = [_feature(f) for f in s6["feature"]]
        add(("table", {"label": "Table S8", "caption": "Each feature beyond the demographics (descriptive).",
                       "df": _blank(s6), "note": "OR per 1 SD of the subject mean of the feature; q: "
                                                 "Benjamini–Hochberg within cohort. Outside the reading rules."}))
    s4 = _read(out, "S4_exploratory_severity.csv")
    if s4 is not None:
        t = pd.DataFrame({"cohort": s4["cohort"], "scale (AD only)": s4["scale"], "n": s4["n"],
                          "partial ρ [95% CI]": [_ci(r, l, h, 2) for r, l, h in zip(s4.rho_partial, s4.lo, s4.hi)],
                          "p": [_num(p, 3) for p in s4["p"]], "adjusted for": [_clean_text(c) for c in s4.covariates]})
        add(("table", {"label": "Table S9", "caption": "Exploratory: EEG score and severity within the AD group.",
                       "df": _blank(t), "note": "Partial Spearman correlation; bootstrap 95% CI. Exploratory, "
                                                "not corrected for multiplicity."}))
    s5 = _read(out, "S5_sensitivity.csv")
    if s5 is not None:
        add(("table", {"label": "Table S10", "caption": "Sensitivity analyses.", "df": _blank(s5), "landscape": True,
                       "note": "p of the sensitivity variants: naive likelihood-ratio test (no null bootstrap). "
                               "BL, BrainLat; GR, Greek cohort."}))
    hp = out / "post_hoc" / "hyperparameter_table.csv"
    if hp.exists():
        t = pd.read_csv(hp, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        note_file = out / "post_hoc" / "hyperparameter_note.txt"
        add(("table", {"label": "Table S11", "caption": "Post hoc: sensitivity of the results to the Random Forest "
                                                         "hyperparameters.", "df": _blank(t), "landscape": True,
                       "note": note_file.read_text(encoding="utf-8") if note_file.exists() else ""}))
    else:
        add(("warning", "Table S11 (hyperparameters) is missing: run notebook 10."))
    add(("table", {"label": "Table S12", "caption": "Validity criteria of the AHEPA benchmark (Miltiadous et al., "
                                                     "2026) and how this study meets them.",
                   "df": pd.DataFrame(C1_C7, columns=["criterion", "requirement", "this study", "where"]),
                   "note": "All seven criteria are met (validity tier 1)."}))
    tri = _read(out, "TRIPOD_AI_checklist.csv")
    if tri is not None:
        cols = [c for c in tri.columns if c in ("section", "item", "item summary (TRIPOD+AI)",
                                                 "where in the manuscript", "status")]
        add(("table", {"label": "Supplementary checklist", "caption": "TRIPOD+AI checklist (pre-filled; update the "
                                                                       "locations when the manuscript is final). "
                                                                       "Submitted as a separate file, without a table "
                                                                       "number.", "df": _blank(tri[cols]),
                       "landscape": True}))
    add(("heading", "Figure legends (draft)"))
    for lab, txt in FIGURE_LEGENDS:
        add(("legend", (lab, txt)))
    return B


FIGURE_LEGENDS = [
    ("Fig. 1", "Study design. (a) The two cohorts, recording systems, groups and available demographics. (b) The "
               "harmonised pipeline applied identically to both cohorts: common 19-electrode montage, pass band and "
               "reference, 4-s epochs, 11 features and a Random Forest evaluated with leave-one-subject-out "
               "cross-validation. (c) The three research questions."),
    ("Fig. 2", "Participant flow in each cohort, with the number excluded at each step and why."),
    ("Fig. 3", "Receiver operating characteristic curves of the EEG score (within-cohort and from the other cohort's "
               "model), the demographic model and the demographic model plus the EEG score, in each cohort."),
    ("Fig. 4", "RQ1, incremental value. (a) Odds ratio of AD per 1 SD of the EEG score beyond the demographics, with "
               "95% CI, in each cohort, in both cohorts together, for the score of the other cohort's model "
               "(independent) and with site added (check). (b, c) Null distributions of the test statistic from the "
               "parametric bootstrap that re-runs the whole leave-one-subject-out procedure under the null "
               "hypothesis; vertical line, observed value; dashed curve, reference distribution of the naive test."),
    ("Fig. 5", "RQ2, confounding by design. (a) Discrimination of the EEG score when AD and controls are made "
               "comparable in the demographics (pairs, comparable pairs, overlap weights; open symbols: whole-cohort "
               "reference and covariate-adjusted check). (b, c) Covariate balance before and after adjustment "
               "(standardised mean differences; grey band, |SMD| < 0.1)."),
    ("Fig. 6", "RQ3, transportability. (a, b) Calibration of the frozen models in the other cohort (points: "
               "quintiles of predicted probability with Wilson 95% intervals; lines: logistic calibration curves) "
               "and after local recalibration (post hoc, leave-one-out within the target cohort). Ticks on the top "
               "and bottom axes: predictions of the AD and control participants. (c, d) Predicted probabilities by "
               "diagnosis before and after recalibration; dashed line, threshold 0.5."),
    ("Fig. 7", "Each EEG feature beyond the demographics in both cohorts (odds ratio per 1 SD with 95% CI; filled "
               "symbols: Benjamini–Hochberg q < 0.05 within cohort). Descriptive, outside the reading rules."),
    ("Fig. S1", "Correspondence between the BrainLat electrodes and the 10-20 system, seen from above (nose up). Open "
                "circles, the 19 standard positions; filled circles, the BioSemi electrode used for each, coloured by "
                "the angle between them; grey dots, the other BioSemi electrodes. Markers only: the error is defined "
                "at the 19 positions, so it is not interpolated. Descriptive (post hoc amendment 2)."),
    ("Fig. S2", "Group-average power spectra after harmonisation (relative power density: each epoch's spectrum, "
                "averaged over the 19 channels, divided by its power in 0.5-45 Hz; mean of the participants with "
                "bootstrap 95% bands). (a) BrainLat; (b) Greek cohort; (c) the controls of the two cohorts. Dashed "
                "line, the 40 Hz harmonisation cutoff; shaded, the frequency bands. Descriptive (post hoc amendment 2)."),
    ("Fig. S3", "E-value for the unmeasured education in the Greek cohort: combinations of the associations of an "
                "unmeasured confounder with the EEG score and with AD (risk-ratio scale) that would explain away "
                "the estimate or move its 95% CI to include 1; dashed line, association of education with AD in "
                "BrainLat (per 1 SD)."),
    ("Fig. S4", "Domain shift: standardised differences between the cohorts in the controls, compared with the "
                "differences between AD and controls within each cohort; dashed lines, |SMD| = 0.8."),
    ("Fig. S5", "Post hoc sensitivity to the Random Forest hyperparameters (one changed at a time; shaded row, the "
                "configuration fixed in the analysis plan). (a) Within-cohort AUC (leave-one-subject-out); (b) odds "
                "ratio of AD per 1 SD of the EEG score beyond the demographics; (c) AUC of the frozen model in the "
                "other cohort. Points with 95% CI. No configuration was selected."),
]


# ---- writers --------------------------------------------------------------------------------------------------------
def write_docx(blocks: list, path: Path) -> Path:
    from docx import Document
    from docx.enum.section import WD_ORIENT
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt, RGBColor

    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Times New Roman"
    st.font.size = Pt(10)
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)          # A4
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(sec, side, Cm(2.0))

    def column_widths(df, total_cm, cm_per_char=0.15, pad=0.35):
        """Column widths (cm) that never split a word or number: each column gets at least its longest token, then
        the room left is shared by how long the text in each column usually is (wrapped above ~40 characters)."""
        import re as _re
        mins, nat = [], []
        for c in df.columns:
            cells = [str(v) for v in df[c]]
            tokens = [t for v in cells + [str(c)] for t in _re.split(r"[\s\-]+", v) if t]
            mins.append(max((len(t) for t in tokens), default=2) * cm_per_char + pad)
            nat.append(float(np.clip(np.percentile([len(v) for v in cells] or [2], 90), 3, 40)) * cm_per_char + pad)
        mins, nat = np.asarray(mins), np.maximum(np.asarray(nat), mins)
        if nat.sum() <= total_cm:
            w = nat * total_cm / nat.sum()
        elif mins.sum() >= total_cm:
            w = mins * total_cm / mins.sum()
        else:
            k = (total_cm - mins.sum()) / (nat - mins).sum()
            w = mins + k * (nat - mins)
        return [Cm(float(x)) for x in w]

    def set_orientation(landscape):
        """New section (on a new page) when the orientation changes; returns True if one was added."""
        cur = doc.sections[-1]
        want = WD_ORIENT.LANDSCAPE if landscape else WD_ORIENT.PORTRAIT
        if cur.orientation == want:
            return False
        new = doc.add_section()
        new.orientation = want
        w, h = new.page_width, new.page_height
        if (landscape and w < h) or (not landscape and w > h):
            new.page_width, new.page_height = h, w
        return True

    def border(cell, **edges):
        tcPr = cell._tc.get_or_add_tcPr()
        b = tcPr.find(qn("w:tcBorders"))
        if b is None:
            b = OxmlElement("w:tcBorders")
            tcPr.append(b)
        for edge, val in edges.items():
            el = OxmlElement(f"w:{edge}")
            el.set(qn("w:val"), "single" if val else "nil")
            if val:
                el.set(qn("w:sz"), str(val))
                el.set(qn("w:color"), "000000")
            b.append(el)

    def para(text_, size=10, bold=False, italic=False, color=None, align=None, space_after=4):
        p = doc.add_paragraph()
        r = p.add_run(text_)
        r.font.size, r.bold, r.italic = Pt(size), bold, italic
        if color:
            r.font.color.rgb = RGBColor.from_string(color)
        if align:
            p.alignment = align
        p.paragraph_format.space_after = Pt(space_after)
        return p

    for idx, (kind, payload) in enumerate(blocks):
        if kind == "title":
            para(payload, 13, bold=True)
        elif kind == "paragraph":
            para(payload, 8.5, color="555555")
        elif kind == "warning":
            para(payload, 11, bold=True, color="C00000")
        elif kind == "heading":
            nxt = blocks[idx + 1] if idx + 1 < len(blocks) else (None, None)
            landscape_next = nxt[0] == "table" and bool(nxt[1].get("landscape"))   # heading on the table's page
            if not set_orientation(landscape_next):
                doc.add_page_break()
            para(payload, 12, bold=True, space_after=8)
        elif kind == "legend":
            lab, txt = payload
            p = doc.add_paragraph()
            r = p.add_run(lab + " ")
            r.bold = True
            p.add_run(txt)
            p.paragraph_format.space_after = Pt(6)
        elif kind == "table":
            spec = payload
            set_orientation(bool(spec.get("landscape")))
            p = doc.add_paragraph()
            r = p.add_run(spec["label"] + ". ")
            r.bold = True
            p.add_run(spec["caption"])
            p.paragraph_format.space_before = Pt(10)
            p.paragraph_format.keep_with_next = True
            df = spec["df"].reset_index(drop=True)
            groups = spec.get("groups")
            n_rows = len(df) + 1 + (len(dict.fromkeys(groups)) if groups else 0)
            tab = doc.add_table(rows=n_rows, cols=len(df.columns))
            tab.autofit = False
            size = Pt(8)
            usable = (29.7 if spec.get("landscape") else 21.0) - 4.0
            widths = column_widths(df, usable)
            trPr = tab.rows[0]._tr.get_or_add_trPr()                # repeat the header row on every page
            hdr = OxmlElement("w:tblHeader")
            hdr.set(qn("w:val"), "true")
            trPr.append(hdr)
            for j, c in enumerate(df.columns):
                cell = tab.cell(0, j)
                cell.text = ""
                run = cell.paragraphs[0].add_run(str(c))
                run.bold, run.font.size = True, size
                border(cell, top=8, bottom=6)
            i, prev = 1, None
            for k, row in df.iterrows():
                if groups and groups[k] != prev:
                    gcell = tab.cell(i, 0).merge(tab.cell(i, len(df.columns) - 1))
                    gcell.text = ""
                    rr = gcell.paragraphs[0].add_run(str(groups[k]))
                    rr.bold, rr.italic, rr.font.size = True, True, size
                    prev, i = groups[k], i + 1
                for j, v in enumerate(row):
                    cell = tab.cell(i, j)
                    cell.text = ""
                    cell.paragraphs[0].add_run(str(v)).font.size = size
                i += 1
            for j in range(len(df.columns)):
                border(tab.cell(n_rows - 1, j), bottom=8)
            for j, col in enumerate(tab.columns):
                col.width = widths[j]
            for row in tab.rows:
                for j, cell in enumerate(row.cells):
                    cell.paragraphs[0].paragraph_format.space_after = Pt(0)
                    if j < len(widths):
                        cell.width = widths[j]
            if spec.get("note"):
                para(spec["note"], 7.5, italic=True, space_after=10)
    doc.save(path)
    return path


def write_html(blocks: list, path: Path) -> Path:
    parts = ["<html><head><meta charset='utf-8'><style>body{font-family:'Times New Roman',serif;font-size:10pt}"
             "table{border-collapse:collapse;margin:6px 0;font-size:8pt}th{border-top:1.5px solid #000;"
             "border-bottom:1px solid #000;padding:2px 6px;text-align:left}td{padding:2px 6px}"
             "tr:last-child td{border-bottom:1.5px solid #000}.g{font-weight:bold;font-style:italic}"
             ".note{font-size:7.5pt;font-style:italic}</style></head><body>"]
    for kind, payload in blocks:
        if kind == "title":
            parts.append(f"<h2>{html.escape(payload)}</h2>")
        elif kind in ("paragraph", "warning"):
            parts.append(f"<p style='color:{'#C00000' if kind == 'warning' else '#555'}'>{html.escape(payload)}</p>")
        elif kind == "heading":
            parts.append(f"<h3>{html.escape(payload)}</h3>")
        elif kind == "legend":
            parts.append(f"<p><b>{html.escape(payload[0])}</b> {html.escape(payload[1])}</p>")
        elif kind == "table":
            df, groups = payload["df"].reset_index(drop=True), payload.get("groups")
            parts.append(f"<p><b>{html.escape(payload['label'])}.</b> {html.escape(payload['caption'])}</p><table>")
            parts.append("<tr>" + "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns) + "</tr>")
            prev = None
            for k, row in df.iterrows():
                if groups and groups[k] != prev:
                    parts.append(f"<tr><td class='g' colspan='{len(df.columns)}'>{html.escape(str(groups[k]))}</td></tr>")
                    prev = groups[k]
                parts.append("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>")
            parts.append("</table>")
            if payload.get("note"):
                parts.append(f"<p class='note'>{html.escape(payload['note'])}</p>")
    parts.append("</body></html>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def write_article_tables(out: Path, ctx_info: dict) -> Path:
    folder = out / "article"
    folder.mkdir(parents=True, exist_ok=True)
    blocks = build_blocks(out, ctx_info)
    try:
        import docx  # noqa: F401
    except ImportError:
        print("python-docx is not installed (pip install python-docx): writing HTML instead (Word opens it).")
        return write_html(blocks, folder / "article_tables.html")
    return write_docx(blocks, folder / "article_tables.docx")
