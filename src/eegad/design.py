"""Confounding removed by design: propensity score, overlap weights, balance, 1:1 matching, covariate-adjusted
AUC and concordance in comparable pairs."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_auc_score

from .stats import logit_fit, safe_logit, sigmoid, standardize


def propensity_score(tab, covs):
    """P(AD | covariates) by logistic regression; Firth if the MLE does not converge or there is separation."""
    Xd = np.column_stack([np.ones(len(tab)), standardize(tab[list(covs)].to_numpy(float))])
    fit, method = logit_fit(Xd, tab["y"].to_numpy(int)), "MLE"
    if (not fit["converged"]) or np.max(np.abs(fit["beta"][1:])) > 10:
        fit, method = logit_fit(Xd, tab["y"].to_numpy(int), firth=True), "Firth"
    return sigmoid(Xd @ fit["beta"]), method


def ess(w):
    """Effective sample size of a set of weights."""
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / np.sum(w ** 2)) if np.sum(w ** 2) > 0 else np.nan


def weighted_auc(yv, pv, wv):
    yv, pv, wv = np.asarray(yv, int), np.asarray(pv, float), np.asarray(wv, float)
    pa, ph, wa, wh = pv[yv == 1], pv[yv == 0], wv[yv == 1], wv[yv == 0]
    K = (pa[:, None] > ph[None, :]) + 0.5 * (pa[:, None] == ph[None, :])
    return float(wa @ K @ wh / (wa.sum() * wh.sum()))


def overlap_weighted_auc_ci(tab, covs, col_p, n_boot=2000, seed=42):
    """AUC with overlap weights; CI by stratified bootstrap, re-estimating the propensity score in each replicate."""
    t = tab.dropna(subset=list(covs) + [col_p, "y"]).reset_index(drop=True)
    ps, method = propensity_score(t, covs)
    w = np.where(t["y"] == 1, 1 - ps, ps)
    a = weighted_auc(t["y"], t[col_p], w)
    rng = np.random.default_rng(seed)
    i1, i0 = np.where(t.y == 1)[0], np.where(t.y == 0)[0]
    bw = []
    for _ in range(n_boot):
        tb = t.iloc[np.r_[rng.choice(i1, len(i1)), rng.choice(i0, len(i0))]].reset_index(drop=True)
        ps_b, _ = propensity_score(tb, covs)
        bw.append(weighted_auc(tb["y"], tb[col_p], np.where(tb["y"] == 1, 1 - ps_b, ps_b)))
    lo, hi = np.percentile(bw, [2.5, 97.5])
    yy = t["y"].to_numpy(int)
    return {"AUC": a, "lo": float(lo), "hi": float(hi), "ps_method": method, "n": len(t),
            "ess_AD": ess(w[yy == 1]), "ess_HC": ess(w[yy == 0])}


def covariate_adjusted_auc(tab, covs, col_p="p_eeg", n_boot=2000, seed=42):
    """Covariate-adjusted AUC (Janes & Pepe, 2009): the score (logit) is standardised by the linear regression in
    the CONTROLS; the AUC of the residuals estimates P(score_AD > score_HC | same covariates). Stratified bootstrap CI."""
    t = tab.dropna(subset=list(covs) + [col_p, "y"]).reset_index(drop=True)

    def est(tt):
        m = safe_logit(tt[col_p])
        Z = np.column_stack([np.ones(len(tt)), tt[list(covs)].to_numpy(float)])
        hc = tt["y"].to_numpy(int) == 0
        beta = np.linalg.lstsq(Z[hc], m[hc], rcond=None)[0]
        return float(roc_auc_score(tt["y"], m - Z @ beta))

    a = est(t)
    rng = np.random.default_rng(seed)
    i1, i0 = np.where(t.y == 1)[0], np.where(t.y == 0)[0]
    boots = [est(t.iloc[np.r_[rng.choice(i1, len(i1)), rng.choice(i0, len(i0))]].reset_index(drop=True))
             for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"AUC": a, "lo": float(lo), "hi": float(hi), "n": len(t)}


def balance(tab, variables, w=None):
    """Group means and standardised mean difference (SMD), optionally weighted. variables = {column: label}."""
    yy = tab["y"].to_numpy(int)
    ww = np.ones(len(tab)) if w is None else np.asarray(w, float)
    rows = []
    for var, name in variables.items():
        if var not in tab.columns:
            continue
        x = tab[var].to_numpy(float)
        ok = np.isfinite(x)
        a, h = ok & (yy == 1), ok & (yy == 0)
        if a.sum() == 0 or h.sum() == 0:
            continue
        m1, m0 = np.average(x[a], weights=ww[a]), np.average(x[h], weights=ww[h])
        v1 = np.average((x[a] - m1) ** 2, weights=ww[a])
        v0 = np.average((x[h] - m0) ** 2, weights=ww[h])
        rows.append({"variable": name, "AD": m1, "HC": m0,
                     "SMD": (m1 - m0) / np.sqrt((v1 + v0) / 2) if v1 + v0 > 0 else 0.0})
    return pd.DataFrame(rows)


def match_1to1(tab, caliper, var="years_education", tiebreak="Age", rng=None):
    """Optimal 1:1 matching without replacement within a caliper; rng given: random tie-breaking among valid pairs."""
    ad = tab[tab.y == 1].dropna(subset=[var]).reset_index(drop=True)
    hc = tab[tab.y == 0].dropna(subset=[var]).reset_index(drop=True)
    empty = pd.DataFrame(columns=["pair", "ad_id", "hc_id", "var_ad", "var_hc", "age_ad", "age_hc"])
    if len(ad) == 0 or len(hc) == 0:
        return empty
    de = np.abs(ad[var].to_numpy(float)[:, None] - hc[var].to_numpy(float)[None, :])
    if rng is None:
        da = np.nan_to_num(np.abs(ad[tiebreak].to_numpy(float)[:, None]
                                  - hc[tiebreak].to_numpy(float)[None, :]), nan=99.0)
        cost = np.where(de <= caliper, de + 1e-4 * da, 1e6)
    else:
        cost = np.where(de <= caliper, rng.random(de.shape), 1e6)
    r, c = linear_sum_assignment(cost)
    ok = cost[r, c] < 1e6
    r, c = r[ok], c[ok]
    if len(r) == 0:
        return empty
    return pd.DataFrame({
        "pair": np.arange(len(r)), "ad_id": ad["subject_id"].to_numpy()[r], "hc_id": hc["subject_id"].to_numpy()[c],
        "var_ad": ad[var].to_numpy()[r], "var_hc": hc[var].to_numpy()[c],
        "age_ad": ad["Age"].to_numpy()[r], "age_hc": hc["Age"].to_numpy()[c]})


def matched_auc_ci(pairs, P, n_boot=2000, seed=42):
    """AUC in the matched subset; CI by bootstrap of pairs (Austin & Small, 2014)."""
    pa, ph = P.loc[list(pairs.ad_id)].to_numpy(float), P.loc[list(pairs.hc_id)].to_numpy(float)
    n = len(pa)
    yv = np.r_[np.ones(n), np.zeros(n)]
    a = float(roc_auc_score(yv, np.r_[pa, ph]))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        boots.append(roc_auc_score(yv, np.r_[pa[i], ph[i]]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return a, float(lo), float(hi)


def comparable_pair_concordance(tab_p, eligible, col_p="p_eeg", n_boot=2000, seed=42):
    """P(P_AD > P_HC) over the eligible AD-HC pairs; CI by bootstrap of subjects within each group."""
    ad = tab_p[tab_p.y == 1].reset_index(drop=True)
    hc = tab_p[tab_p.y == 0].reset_index(drop=True)
    Mm = np.asarray(eligible(ad, hc), float)
    if Mm.sum() == 0:
        return {"C": np.nan, "lo": np.nan, "hi": np.nan, "n_pairs": 0, "n_AD": 0, "n_HC": 0}
    pa, ph = ad[col_p].to_numpy(float), hc[col_p].to_numpy(float)
    K = (pa[:, None] > ph[None, :]) + 0.5 * (pa[:, None] == ph[None, :])
    KM = K * Mm
    C = float(KM.sum() / Mm.sum())
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        a = np.bincount(rng.integers(0, len(pa), len(pa)), minlength=len(pa)).astype(float)
        b = np.bincount(rng.integers(0, len(ph), len(ph)), minlength=len(ph)).astype(float)
        den = a @ Mm @ b
        if den > 0:
            boots.append((a @ KM @ b) / den)
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
    return {"C": C, "lo": float(lo), "hi": float(hi), "n_pairs": int(Mm.sum()),
            "n_AD": int((Mm.sum(axis=1) > 0).sum()), "n_HC": int((Mm.sum(axis=0) > 0).sum())}


def eligible_education(caliper):
    """Comparable pairs in BrainLat: |education difference| <= caliper."""
    return lambda a, h: np.abs(a["years_education"].to_numpy(float)[:, None]
                               - h["years_education"].to_numpy(float)[None, :]) <= caliper


def eligible_sex_age(age_caliper):
    """Comparable pairs in the Greek cohort: same sex and |age difference| <= caliper."""
    return lambda a, h: ((a["sex_fem"].to_numpy(float)[:, None] == h["sex_fem"].to_numpy(float)[None, :])
                         & (np.abs(a["Age"].to_numpy(float)[:, None] - h["Age"].to_numpy(float)[None, :])
                            <= age_caliper))
