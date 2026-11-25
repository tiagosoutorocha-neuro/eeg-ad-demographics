"""Descriptive tables: participants (Table 1) and cognitive severity by cohort (Table 1b)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .reporting import fmt_ci, fmt_p
from .stats import hedges_g, mean_sd, median_iqr, p_fisher, p_mwu, value_range


def participants_table(s, cohort):
    """Table 1 rows for one cohort (AD vs HC)."""
    AD, HC = s[s.y == 1], s[s.y == 0]
    L = [{"cohort": cohort, "variable": "n", "AD": len(AD), "HC": len(HC), "p": "", "test": ""},
         {"cohort": cohort, "variable": "Age (years)", "AD": mean_sd(AD.Age), "HC": mean_sd(HC.Age),
          "p": fmt_p(p_mwu(AD.Age, HC.Age)), "test": "Mann–Whitney"}]
    fa, fh = AD.sex_fem.dropna(), HC.sex_fem.dropna()
    tab = [[int(fa.sum()), int((1 - fa).sum())], [int(fh.sum()), int((1 - fh).sum())]]
    L.append({"cohort": cohort, "variable": "Female, n (%)", "AD": f"{int(fa.sum())} ({100 * fa.mean():.1f})",
              "HC": f"{int(fh.sum())} ({100 * fh.mean():.1f})", "p": fmt_p(p_fisher(tab)), "test": "Fisher"})
    if "years_education" in s:
        L.append({"cohort": cohort, "variable": "Education (years)", "AD": mean_sd(AD.years_education),
                  "HC": mean_sd(HC.years_education), "p": fmt_p(p_mwu(AD.years_education, HC.years_education)),
                  "test": "Mann–Whitney"})
    if "site" in s:
        tab = [[int((AD.site == "Argentina").sum()), int((AD.site == "Chile").sum())],
               [int((HC.site == "Argentina").sum()), int((HC.site == "Chile").sum())]]
        L.append({"cohort": cohort, "variable": "Site (Argentina / Chile)", "AD": f"{tab[0][0]} / {tab[0][1]}",
                  "HC": f"{tab[1][0]} / {tab[1][1]}", "p": fmt_p(p_fisher(tab)), "test": "Fisher"})
    for col, label in (("moca_total", "MoCA"), ("ifs_total_score", "IFS"), ("mmse", "MMSE")):
        if col in s and s[col].notna().any():
            L.append({"cohort": cohort, "variable": label, "AD": mean_sd(AD[col], True), "HC": mean_sd(HC[col], True),
                      "p": fmt_p(p_mwu(AD[col], HC[col])), "test": "Mann–Whitney"})
    L.append({"cohort": cohort, "variable": "Recording analysed (min)", "AD": mean_sd(AD.duration_min),
              "HC": mean_sd(HC.duration_min), "p": fmt_p(p_mwu(AD.duration_min, HC.duration_min)),
              "test": "Mann–Whitney"})
    L.append({"cohort": cohort, "variable": "Epochs per participant", "AD": mean_sd(AD.epochs_used, nd=0),
              "HC": mean_sd(HC.epochs_used, nd=0), "p": fmt_p(p_mwu(AD.epochs_used, HC.epochs_used)),
              "test": "Mann–Whitney"})
    return pd.DataFrame(L)


def participants_keys(s, cohort_key):
    """Unrounded Table 1 numbers, for the reproduction check."""
    AD, HC = s[s.y == 1], s[s.y == 0]
    k = {f"t1.{cohort_key}.n_ad": len(AD), f"t1.{cohort_key}.n_hc": len(HC)}
    cols = {"age": "Age", "education": "years_education", "moca": "moca_total", "ifs": "ifs_total_score",
            "mmse": "mmse", "recording_min": "duration_min", "epochs": "epochs_used"}
    for name, col in cols.items():
        if col not in s or s[col].notna().sum() == 0:
            continue
        for grp, d in (("ad", AD), ("hc", HC)):
            v = d[col].dropna()
            if len(v):
                k[f"t1.{cohort_key}.{name}.{grp}.mean"] = float(v.mean())
                k[f"t1.{cohort_key}.{name}.{grp}.sd"] = float(v.std())
        k[f"t1.{cohort_key}.{name}.p"] = p_mwu(AD[col], HC[col])
    fa, fh = AD.sex_fem.dropna(), HC.sex_fem.dropna()
    k[f"t1.{cohort_key}.female.ad.n"] = int(fa.sum())
    k[f"t1.{cohort_key}.female.hc.n"] = int(fh.sum())
    k[f"t1.{cohort_key}.female.p"] = p_fisher([[int(fa.sum()), int((1 - fa).sum())], [int(fh.sum()), int((1 - fh).sum())]])
    return k


def severity_rows(cohort, s, col, label):
    """Rows of Table 1b for one scale: n, mean ± SD, median [IQR], range, and Hedges' g (AD − HC)."""
    AD, HC = s.loc[s.y == 1, col], s.loc[s.y == 0, col]
    g = hedges_g(AD, HC)
    rows = []
    for grp, v in (("AD", AD), ("HC", HC)):
        rows.append({"cohort": cohort, "scale": label, "group": grp,
                     "n with score / n": f"{v.notna().sum()} / {len(v)}", "mean ± SD": mean_sd(v),
                     "median [IQR]": median_iqr(v), "range": value_range(v),
                     "Hedges g AD − HC [95% CI]": fmt_ci(*g, 2) if grp == "AD" else ""})
    return rows, g


MMSE_STAGES = ([-0.1, 10, 20, 25, 29, 30],
               ["severe (0–10)", "moderate (11–20)", "mild (21–25)", "questionable (26–29)", "none (30)"])


def mmse_stages(mmse):
    """AD participants by MMSE stage (Perneczky et al., 2006)."""
    mm = pd.Series(mmse).dropna()
    st = pd.cut(mm, bins=MMSE_STAGES[0], labels=MMSE_STAGES[1])
    tab = st.value_counts().reindex(st.cat.categories).rename_axis("stage (MMSE)").reset_index(name="n AD")
    tab["%"] = (100 * tab["n AD"] / max(len(mm), 1)).round(1)
    return tab, len(mm)


def compare_groups(I, E, label_i="included", label_e="excluded"):
    """Included vs excluded participants of the same diagnostic group (descriptive)."""
    rows = [{"variable": "n", label_i: len(I), label_e: len(E), "p": "", "test": ""}]
    if len(E) == 0:
        return pd.DataFrame(rows)
    t = [[int((I.site == "Chile").sum()), int((I.site != "Chile").sum())],
         [int((E.site == "Chile").sum()), int((E.site != "Chile").sum())]]
    rows.append({"variable": "Chile, n (%)", label_i: f"{t[0][0]} ({100 * t[0][0] / max(len(I), 1):.0f})",
                 label_e: f"{t[1][0]} ({100 * t[1][0] / max(len(E), 1):.0f})", "p": fmt_p(p_fisher(t)),
                 "test": "Fisher"})
    fi, fe = I.sex_fem.dropna(), E.sex_fem.dropna()
    t = [[int(fi.sum()), int((1 - fi).sum())], [int(fe.sum()), int((1 - fe).sum())]]
    rows.append({"variable": "Female, n (%)",
                 label_i: f"{int(fi.sum())} ({100 * fi.mean():.0f})" if len(fi) else "—",
                 label_e: f"{int(fe.sum())} ({100 * fe.mean():.0f})" if len(fe) else "—",
                 "p": fmt_p(p_fisher(t)) if len(fi) and len(fe) else "n/a", "test": "Fisher"})
    for label, col in (("Age (years)", "Age"), ("Education (years)", "years_education"), ("MoCA", "moca_total")):
        rows.append({"variable": label, label_i: mean_sd(I[col], True), label_e: mean_sd(E[col], True),
                     "p": fmt_p(p_mwu(I[col], E[col])), "test": "Mann–Whitney"})
    return pd.DataFrame(rows)
