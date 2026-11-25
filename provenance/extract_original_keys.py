"""Extract the key numerical results of the ORIGINAL run (Portuguese notebook) into key_results.json.

The original notebook wrote formatted tables (mostly 2-3 decimals) and a few files at full precision. This script
reads only aggregated outputs (no subject-level file) and records, for each value, how many decimals were reported,
so that notebooks/08_reproduction_check.ipynb can tell whether the new run reproduces it.

Usage: python provenance/extract_original_keys.py provenance/original_run provenance/original_run/key_results.json
(also accepts an original output folder with RESUMO_JOURNAL.md and tabelas/)
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

SRC, DST = Path(sys.argv[1]), Path(sys.argv[2])
T = SRC / "tabelas" if (SRC / "tabelas").exists() else SRC / "tables"      # original layout or provenance/
KEYS = {}
FEATURE_EN = {"Razao_Theta_Alpha": "Ratio_Theta_Alpha", "Razao_Theta_Beta": "Ratio_Theta_Beta"}
NUM = r"-?\d+(?:\.\d+)?"


def put(key, text_or_value, source, decimals="auto"):
    """Store a value. Strings like '< 0.001' become kind 'lt'; decimals are inferred from the text."""
    if isinstance(text_or_value, str):
        s = text_or_value.strip()
        if s.startswith("<"):
            KEYS[key] = {"value": float(s.lstrip("< ").strip()), "kind": "lt", "decimals": None, "source": source}
            return
        if s in ("n/d", "n/a", ""):
            KEYS[key] = {"value": None, "kind": "number", "decimals": None, "source": source}
            return
        dec = len(s.split(".")[1]) if "." in s else 0
        KEYS[key] = {"value": float(s), "kind": "number", "decimals": dec if decimals == "auto" else decimals,
                     "source": source}
    else:
        KEYS[key] = {"value": None if text_or_value is None else float(text_or_value), "kind": "number",
                     "decimals": None if decimals == "auto" else decimals, "source": source}


def ci(text):
    m = re.search(rf"({NUM}|n/d) \[({NUM}|n/d), ({NUM}|n/d)\]", text)
    return m.groups() if m else (None, None, None)


def put_ci(prefix, text, source, names=("", "lo", "hi")):
    a, lo, hi = ci(text)
    for n, v in zip(names, (a, lo, hi)):
        if v is not None:
            put(f"{prefix}.{n}" if n else prefix, v, source)


def metrics(prefix, text, source):
    for pt, en in (("sens", "sens"), ("espec", "spec"), ("acurácia balanceada", "bal_acc"), ("acurácia", "acc"),
                   ("F1", "F1"), ("MCC", "MCC")):
        m = re.search(rf"(?:^|, |; ){pt} ({NUM})", text)
        if m:
            put(f"{prefix}.{en}", m.group(1), source)


# ---- summary (RESUMO_JOURNAL.md): items that are not in the tables ---------------------------------------------
resumo = next(f for f in (SRC / "RESUMO_JOURNAL.md", SRC / "RESUMO_JOURNAL_redacted.md") if f.exists()).read_text(
    encoding="utf-8")
ITEMS = {}
for line in resumo.splitlines():
    cells = [c.strip() for c in re.split(r"(?<!\\)\|", line)[1:-1]]
    if len(cells) == 5:
        ITEMS[cells[1].replace("\\|", "|")] = cells[2].replace("\\|", "|")


def item(name):
    return ITEMS.get(name, "")


src = "RESUMO_JOURNAL.md"
for key, name in (("rq1a", "BrainLat: escore EEG além de idade, sexo e escolaridade"),
                  ("rq1b", "Grego: escore EEG além de idade e sexo (unilateral)")):
    v = item(name)
    m = re.search(rf"χ²\(1\) = ({NUM})", v)
    if m:
        put(f"{key}.lr", m.group(1), src)
v = item("Grego: E-value (escolaridade não medida)")
m = re.search(rf"({NUM}) \(OR\); ({NUM})", v)
if m:
    put("rq1b.e_value", m.group(1), src)
    put("rq1b.e_value_ci", m.group(2), src)
v = item("Referência para o E-value (BrainLat)")
m = re.search(rf"EEG ({NUM}) \(idade \+ sexo\) → ({NUM}) \(\+ escolaridade\); escolaridade OR/DP ({NUM})", v)
if m:
    put("rq1b.reference.eeg_or_age_sex", m.group(1), src)
    put("rq1b.reference.eeg_or_age_sex_education", m.group(2), src)
    put("rq1b.reference.education_or", m.group(3), src)
for cohort_pt, cohort in (("BrainLat", "brainlat"), ("Grego", "greek")):
    put_ci(f"delta_auc.{cohort}.combined_minus_demographics",
           item(f"{cohort_pt}: ΔAUC Demographics + EEG score − demographics"), src, ("value", "lo", "hi"))
    put_ci(f"delta_auc.{cohort}.eeg_minus_demographics",
           item(f"{cohort_pt}: ΔAUC EEG score (RF, LOSO) − demographics"), src, ("value", "lo", "hi"))
v = item("Conjunto: escore EEG além das covariáveis de cada coorte (efeito comum)")
m = re.search(rf"χ²\(1\) = ({NUM})", v)
if m:
    put("rq1c.pooled.lr", m.group(1), src)
v = item("Heterogeneidade (coorte × EEG)")
m = re.search(rf"χ²\(1\) = ({NUM})", v)
if m:
    put("rq1c.heterogeneity.lr", m.group(1), src)
for key, name in (("rq3a", "BrainLat → grego (modelo congelado): AUC"), ("rq3b", "Grego → BrainLat (modelo congelado): AUC")):
    m = re.search(rf"Brier ({NUM})", item(name))
    if m:
        put(f"{key}.brier", m.group(1), src)
v = item("BrainLat: AUC nos pares 1:1 por escolaridade (±2 anos)")
m = re.search(rf"mediana ({NUM}), faixa ({NUM})–({NUM})", v)
if m:
    put("rq2.brainlat.matched.tiebreak_median", m.group(1), src)
    put("rq2.brainlat.matched.tiebreak_min", m.group(2), src)
    put("rq2.brainlat.matched.tiebreak_max", m.group(3), src)
v = item("Mudança de domínio (controles, |SMD| > 0,8 entre coortes)")
put("domain_shift.n_features_smd_gt_0.8", str(len([x for x in v.split(",") if x.strip() and x.strip() != "nenhuma"])), src)
for cohort_pt, cohort in (("BrainLat", "brainlat"), ("Grego", "greek")):
    m = re.search(r"(\d+) de (\d+) \((\d+) AD, (\d+) HC\); (\d+) épocas", item(f"{cohort_pt}: participantes com EEG analisável"))
    if m:
        put(f"cohort.{cohort}.n_analysable", m.group(1), src)
        put(f"cohort.{cohort}.n_with_eeg_file", m.group(2), src)
        put(f"cohort.{cohort}.n_ad", m.group(3), src)
        put(f"cohort.{cohort}.n_hc", m.group(4), src)
        put(f"cohort.{cohort}.n_epochs", m.group(5), src)

# ---- Table 2 (formatted) -----------------------------------------------------------------------------------------
t2 = pd.read_csv(T / "T2_resultados_principais.csv", dtype=str, keep_default_na=False)
src = "T2_resultados_principais.csv"
for _, r in t2.iterrows():
    a, c, est, p = r["análise"], r["coorte"], r["estimativa"], r["p"]
    coh = {"BrainLat": "brainlat", "Grego": "greek", "Pooled": "pooled"}[c]
    if a.startswith("Within-cohort EEG score beyond"):
        key = "rq1a" if coh == "brainlat" else "rq1b"
        put_ci(f"{key}.or", est, src, ("value", "lo", "hi"))
        m = re.search(r"analytic (< 0\.001|[\d.]+)", p)
        put(f"{key}.p_analytic", m.group(1), src)
    elif "(independent)" in a:
        put_ci(f"rq1.independent.{coh}.or", est, src, ("value", "lo", "hi"))
        put(f"rq1.independent.{coh}.p" if coh == "brainlat" else f"rq1.independent.{coh}.p_one_sided",
            p.replace(" one-sided", ""), src)
    elif a == "Check: + site":
        put_ci("rq1a.site_check.or", est, src, ("value", "lo", "hi"))
        put("rq1a.site_check.p", p, src)
    elif coh == "pooled":
        put_ci("rq1c.pooled.or", est, src, ("value", "lo", "hi"))
        m = re.match(r"(< 0\.001|[\d.]+); cohort × EEG (< 0\.001|[\d.]+)", p)
        put("rq1c.pooled.p", m.group(1), src)
        put("rq1c.heterogeneity.p", m.group(2), src)
    elif r["bloco"].startswith("Discrimination"):
        if a.startswith("EEG score, threshold"):
            metrics(f"threshold.{coh}", est, src)
        else:
            name = {"Age": "age", "Age + sex": "age_sex", "Age + sex + education": "age_sex_education",
                    "EEG score (RF, LOSO)": "eeg", "Demographics + EEG score": "demographics_plus_eeg"}[a]
            if name != "eeg":       # the EEG AUC is taken at full precision from T2b
                put_ci(f"auc.{coh}.{name}", est, src, ("value", "lo", "hi"))
    elif r["bloco"].startswith("RQ3"):
        std = "padronizado" in a
        key = ("rq3a" if coh == "greek" else "rq3b") + ("_std" if std else "")
        put_ci(f"{key}.auc", est, src, ("value", "lo", "hi"))
        m = re.search(rf"calibration slope ({NUM}); calibration-in-the-large ({NUM})", p)
        put(f"{key}.calibration_slope", m.group(1), src)
        put(f"{key}.citl", m.group(2), src)
        if not std:
            metrics(f"{key}.threshold", est, src)

# ---- Table 2b (full precision) ----------------------------------------------------------------------------------
t2b = pd.read_csv(T / "T2b_confundimento_por_desenho.csv")
src = "T2b_confundimento_por_desenho.csv"
NAMES = {"Full cohort": "full", "Education-matched 1:1 (±2 y)": "matched",
         "Comparable-pair concordance (education ±2 y)": "comparable_pairs",
         "Overlap-weighted (age, sex, education)": "overlap", "Covariate-adjusted AUC (age, sex, education)": "adjusted",
         "Overlap-weighted (age, sex)": "overlap", "Comparable-pair concordance (same sex, age ±2 y)": "comparable_pairs",
         "Covariate-adjusted AUC (age, sex)": "adjusted"}
for _, r in t2b.iterrows():
    coh = "brainlat" if r["coorte"] == "BrainLat" else "greek"
    name = NAMES[r["análise"]]
    for col, sfx in (("AUC", "value"), ("lo", "lo"), ("hi", "hi")):
        put(f"rq2.{coh}.{name}.{sfx}", float(r[col]), src)
    if name == "full":
        for col, sfx in (("AUC", "value"), ("lo", "lo"), ("hi", "hi")):
            put(f"auc.{coh}.eeg.{sfx}", float(r[col]), src)
    m = re.match(r"(\d+) (pairs|participants)", str(r["n"]))
    if m and name in ("matched", "comparable_pairs"):
        put(f"rq2.{coh}.{name}.n_pairs", m.group(1), src)

# ---- null bootstrap (full precision) ----------------------------------------------------------------------------
for cohort_pt, key in (("BrainLat", "rq1a"), ("Grego", "rq1b")):
    f = T / f"bootstrap_nulo_{cohort_pt}.json"
    if f.exists():
        b = json.loads(f.read_text(encoding="utf-8"))
        put(f"{key}.null_bootstrap.t_obs", b["T_obs"], f.name)
        put(f"{key}.null_bootstrap.p", b["p"], f.name)
        put(f"{key}.null_bootstrap.replicates", b["replicas"], f.name)
        put(f"{key}.null_bootstrap.exceedances", b["excedencias"], f.name)

# ---- design (full precision) ------------------------------------------------------------------------------------
f = T / "desenho_confundimento.json"
d = json.loads(f.read_text(encoding="utf-8"))
put("design.matched_pairs", d["n_pares_1a1"], f.name)
put("design.comparable_pairs.brainlat", d["n_pares_comparaveis_brainlat"], f.name)
put("design.comparable_pairs.greek", d["n_pares_comparaveis_grego"], f.name)
for cohort_pt, coh in (("BrainLat", "brainlat"), ("Grego", "greek")):
    o = d[f"overlap_{cohort_pt}"]
    put(f"design.overlap.{coh}.ess_ad", o["ess_AD"], f.name)
    put(f"design.overlap.{coh}.ess_hc", o["ess_HC"], f.name)
    put(f"design.overlap.{coh}.n", o["n"], f.name)
for fname, prefix in (("S3_balanco_BrainLat_pareamento.csv", "balance.brainlat_matching"),
                      ("S3_balanco_BrainLat_overlap.csv", "balance.brainlat_overlap"),
                      ("S3_balanco_Grego_overlap.csv", "balance.greek_overlap")):
    b = pd.read_csv(T / fname)
    for _, r in b.iterrows():
        var = r["variável"].split(" (")[0].lower().replace(" ", "_")
        for col in [c for c in b.columns if c.startswith("SMD")]:
            sfx = {"SMD_todos": "smd_all", "SMD_pareados": "smd_matched", "SMD_bruto": "smd_raw",
                   "SMD_ponderado": "smd_weighted"}[col]
            put(f"{prefix}.{var}.{sfx}", float(r[col]), fname)

# ---- channel map, domain shift (full precision) -----------------------------------------------------------------
s1 = pd.read_csv(T / "S1_mapa_canais_brainlat_1020.csv")
put("channels.angle_mean", float(s1["angulo_graus"].mean()), "S1_mapa_canais_brainlat_1020.csv")
put("channels.angle_max", float(s1["angulo_graus"].max()), "S1_mapa_canais_brainlat_1020.csv")
s2 = pd.read_csv(T / "S2_features_por_coorte.csv", index_col=0)
for feat, r in s2.iterrows():
    put(f"domain_shift.smd.{FEATURE_EN.get(feat, feat)}", float(r.iloc[-1]), "S2_features_por_coorte.csv")

# ---- exploratory (full precision) -------------------------------------------------------------------------------
s4 = pd.read_csv(T / "S4_exploratoria_gravidade.csv")
for _, r in s4.iterrows():
    coh = "brainlat" if r["coorte"] == "BrainLat" else "greek"
    scale = r["escala"].split(" ")[0].lower()
    for col, sfx in (("n", "n"), ("rho_simples", "rho_simple"), ("rho_parcial", "rho_partial"), ("IC_inf", "lo"),
                     ("IC_sup", "hi"), ("p", "p")):
        put(f"exploratory.{coh}.{scale}.{sfx}", float(r[col]), "S4_exploratoria_gravidade.csv")

# ---- per-feature descriptive (formatted) ------------------------------------------------------------------------
s6 = pd.read_csv(T / "S6_features_alem_da_demografia.csv", index_col=0, dtype=str)
for feat, r in s6.iterrows():
    fe = FEATURE_EN.get(feat, feat)
    put_ci(f"features.{fe}.brainlat.or", r["BrainLat OR/SD [95% CI]"], "S6", ("value", "lo", "hi"))
    put(f"features.{fe}.brainlat.q", r["BrainLat q"], "S6")
    put_ci(f"features.{fe}.greek.or", r["Greek OR/SD [95% CI]"], "S6", ("value", "lo", "hi"))
    put(f"features.{fe}.greek.q", r["Greek q"], "S6")

# ---- sensitivity (formatted) ------------------------------------------------------------------------------------
s5 = pd.read_csv(T / "S5_sensibilidade.csv", dtype=str)
VAR = {"Principal (RF, 11 features)": "main", "RF, 7 spectral features": "rf_spectral7",
       "RF, 10 features (no gamma)": "rf_no_gamma", "SVM, 11 features": "svm", "XGBoost, 11 features": "xgboost"}
for _, r in s5.iterrows():
    v = VAR.get(r["variante"], "rf_gamma_epochs_removed" if "gamma RP" in r["variante"] else None)
    if v is None:
        continue
    p = f"sensitivity.{v}"
    put_ci(f"{p}.auc_brainlat", r["AUC BrainLat"], "S5", ("value", "lo", "hi"))
    put_ci(f"{p}.rq1a_or", r["RQ1a OR/DP"], "S5", ("value", "lo", "hi"))
    put(f"{p}.rq1a_p", r["RQ1a p"], "S5")
    put_ci(f"{p}.auc_greek", r["AUC grego"], "S5", ("value", "lo", "hi"))
    put_ci(f"{p}.rq1b_or", r["RQ1b OR/DP"], "S5", ("value", "lo", "hi"))
    put(f"{p}.rq1b_p_one_sided", r["RQ1b p unilateral"], "S5")
    put_ci(f"{p}.rq3a_auc", r["RQ3a BL→GR"], "S5", ("value", "lo", "hi"))
    put_ci(f"{p}.rq3b_auc", r["RQ3b GR→BL"], "S5", ("value", "lo", "hi"))

# ---- Table 1 (formatted) ----------------------------------------------------------------------------------------
t1 = pd.read_csv(T / "T1_participantes.csv", dtype=str, keep_default_na=False)
VARS = {"Age (years)": "age", "Education (years)": "education", "MoCA": "moca", "IFS": "ifs", "MMSE": "mmse",
        "Recording analysed (min)": "recording_min", "Epochs per participant": "epochs"}
for _, r in t1.iterrows():
    coh = "brainlat" if r["coorte"] == "BrainLat" else "greek"
    var = r["variável"]
    if var == "n":
        put(f"t1.{coh}.n_ad", r["AD"], "T1")
        put(f"t1.{coh}.n_hc", r["HC"], "T1")
    elif var in VARS:
        for grp in ("AD", "HC"):
            m = re.match(rf"({NUM}) ± ({NUM})", r[grp])
            if m:
                put(f"t1.{coh}.{VARS[var]}.{grp.lower()}.mean", m.group(1), "T1")
                put(f"t1.{coh}.{VARS[var]}.{grp.lower()}.sd", m.group(2), "T1")
        if r["p"]:
            put(f"t1.{coh}.{VARS[var]}.p", r["p"], "T1")
    elif var.startswith("Female"):
        for grp in ("AD", "HC"):
            put(f"t1.{coh}.female.{grp.lower()}.n", r[grp].split(" ")[0], "T1")
        put(f"t1.{coh}.female.p", r["p"], "T1")

# ---- section 13 tables (only in runs that included it) ----------------------------------------------------------
if (T / "T1b_gravidade_por_coorte.csv").exists():
    t1b = pd.read_csv(T / "T1b_gravidade_por_coorte.csv", dtype=str, keep_default_na=False)
    for _, r in t1b[t1b["grupo"] == "AD"].iterrows():
        coh = "brainlat" if r["coorte"] == "BrainLat" else "greek"
        scale = r["escala"].split(" ")[0].lower()
        put_ci(f"severity.{coh}.{scale}.hedges_g", r["g de Hedges AD − HC [IC 95%]"], "T1b", ("value", "lo", "hi"))
if (T / "T1c_estagios_MMSE_grego.csv").exists():
    t1c = pd.read_csv(T / "T1c_estagios_MMSE_grego.csv")
    for _, r in t1c.iterrows():
        put(f"severity.greek.mmse_stage.{r.iloc[0].split(' ')[0]}", int(r.iloc[1]), "T1c")
if (T / "F0_fluxo_participantes.csv").exists():
    f0 = pd.read_csv(T / "F0_fluxo_participantes.csv", dtype=str, keep_default_na=False)
    STEP = {"Listed in the distributed demographic files (AD and HC folders)": "brainlat.demographic_files",
            "Resting-state EEG file (.set) available": None, "AD and cognitively normal controls": "greek.ad_and_controls",
            "Listed in participants.tsv (ds004504 v1.0.9)": "greek.participants_tsv"}
    for _, r in f0.iterrows():
        coh = "brainlat" if r["coorte"] == "BrainLat" else "greek"
        step = STEP.get(r["etapa"], "")
        if r["etapa"] == "Resting-state EEG file (.set) available":
            step = f"{coh}.set_files"
        elif r["etapa"].startswith("Analysable EEG"):
            step = f"{coh}.analysable"
        elif r["etapa"].startswith("Complete covariates"):
            step = f"{coh}.complete_covariates"
        elif r["etapa"].startswith("AD with"):
            step = f"{coh}.exploratory"
        if step:
            for col in ("n", "AD", "HC"):
                put(f"flow.{step}.{col.lower()}", r[col], "F0")

out = {"_meta": {"source_folder": str(SRC.name), "n_keys": len(KEYS),
                 "note": "Values from the original (Portuguese) notebook outputs; 'decimals' = reported precision "
                         "(null = full precision); kind 'lt' = reported as '< value'."},
       "keys": dict(sorted(KEYS.items()))}
DST.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"{len(KEYS)} keys -> {DST}")
