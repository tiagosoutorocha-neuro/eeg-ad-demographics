"""Statistics: logistic regression (MLE/Firth), likelihood-ratio tests, profile CIs, AUC with bootstrap CIs,
E-values, partial Spearman correlation and small descriptive helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq
from sklearn.metrics import roc_auc_score


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def safe_logit(p):
    p = np.clip(np.asarray(p, float), 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def standardize(M):
    M = np.asarray(M, float)
    sd = M.std(axis=0, ddof=1)
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
    return (M - M.mean(axis=0)) / sd


# ---- logistic regression ----------------------------------------------------------------------------------------
def logit_fit(X, y, firth=False, fixed=None, max_iter=200, tol=1e-8):
    """Logistic regression by Newton-Raphson (X already has the intercept column).

    firth=True: Firth's penalised likelihood (Heinze & Schemper, 2002).
    fixed={column: value}: coefficients held fixed (used for profile likelihood and calibration-in-the-large).
    """
    X, y = np.asarray(X, float), np.asarray(y, float)
    n, k = X.shape
    fixed = fixed or {}
    free = np.array([j for j in range(k) if j not in fixed], dtype=int)
    beta = np.zeros(k)
    for j, v in fixed.items():
        beta[j] = v

    def loglik(b):
        p = sigmoid(X @ b)
        ll = float(np.sum(y * np.log(np.clip(p, 1e-300, None)) + (1 - y) * np.log(np.clip(1 - p, 1e-300, None))))
        if firth:
            w = p * (1 - p)
            sign, logdet = np.linalg.slogdet(X.T @ (X * w[:, None]))
            ll += 0.5 * logdet if sign > 0 else -np.inf
        return ll

    ll, converged, it = loglik(beta), False, 0
    for it in range(1, max_iter + 1):
        p = sigmoid(X @ beta)
        w = p * (1 - p)
        info = X.T @ (X * w[:, None])
        if firth:
            info_inv = np.linalg.pinv(info)
            h = np.einsum("ij,jk,ik->i", X, info_inv, X) * w
            score = X.T @ (y - p + h * (0.5 - p))
        else:
            score = X.T @ (y - p)
        sub = info[np.ix_(free, free)]
        try:
            step = np.linalg.solve(sub, score[free])
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(sub, score[free], rcond=None)[0]
        largest = np.max(np.abs(step)) if len(step) else 0.0
        if largest > 5:
            step = step * 5 / largest
        t = 1.0
        while True:
            b_new = beta.copy()
            b_new[free] += t * step
            ll_new = loglik(b_new)
            if ll_new >= ll - 1e-10 or t < 1e-6:
                break
            t /= 2
        diff = np.max(np.abs(b_new - beta)) if len(step) else 0.0
        beta, ll = b_new, ll_new
        if diff < tol:
            converged = True
            break
    p = sigmoid(X @ beta)
    info = X.T @ (X * (p * (1 - p))[:, None])
    try:
        cov = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        cov = np.full((k, k), np.nan)
    return {"beta": beta, "ll": ll, "cov": cov, "converged": converged, "n_iter": it, "p": p}


def profile_ci(Xd, yv, j, firth, fit_full, level=0.95):
    """Profile-likelihood (penalised, if Firth) confidence interval for coefficient j."""
    target = stats.chi2.ppf(level, 1)
    b_hat, ll_max = fit_full["beta"][j], fit_full["ll"]
    v = fit_full["cov"][j, j]
    se = float(np.sqrt(v)) if np.isfinite(v) and v > 0 else 0.5

    def f(b):
        return 2 * (ll_max - logit_fit(Xd, yv, firth=firth, fixed={j: b})["ll"]) - target

    limits = []
    for sign in (-1, 1):
        step, k = max(se, 0.05), 0
        b_out = b_hat + sign * step
        while f(b_out) < 0 and k < 25:
            step *= 1.8
            b_out = b_hat + sign * step
            k += 1
        try:
            limits.append(brentq(f, min(b_hat, b_out), max(b_hat, b_out), xtol=1e-6))
        except Exception:
            limits.append(np.nan)
    return limits[0], limits[1]


def nested_lr(X_red, X_full, yv, j):
    """Likelihood-ratio test (1 df) for column j of X_full, with the plan's Firth rule."""
    res = {}
    for method, firth in (("MLE", False), ("Firth", True)):
        fr, ff = logit_fit(X_red, yv, firth=firth), logit_fit(X_full, yv, firth=firth)
        lr = max(2 * (ff["ll"] - fr["ll"]), 0.0)
        lo_b, hi_b = profile_ci(X_full, yv, j, firth, ff)
        b = float(ff["beta"][j])
        p2 = float(stats.chi2.sf(lr, 1))
        res[method] = {"LR": lr, "p": p2, "p_one_sided": p2 / 2 if b > 0 else 1 - p2 / 2, "beta": b,
                       "OR": float(np.exp(b)),
                       "OR_lo": float(np.exp(lo_b)) if np.isfinite(lo_b) else np.nan,
                       "OR_hi": float(np.exp(hi_b)) if np.isfinite(hi_b) else np.nan,
                       "converged": bool(fr["converged"] and ff["converged"]),
                       "max_beta": float(max(np.max(np.abs(fr["beta"][1:])), np.max(np.abs(ff["beta"][1:]))))}
    separation = (not res["MLE"]["converged"]) or res["MLE"]["max_beta"] > 10
    method = "Firth" if separation else "MLE"
    return {"method": method, **res[method]}


def incremental_test(tab, covs, col_p, label):
    """EEG score (logit of P(AD), standardised within the cohort) beyond the covariates: OR per 1 SD and LR test."""
    covs = list(covs)
    t = tab.dropna(subset=covs + [col_p, "y"]).reset_index(drop=True)
    yv = t["y"].to_numpy(int)
    z = standardize(safe_logit(t[col_p])[:, None])[:, 0]
    X_red = np.column_stack([np.ones(len(yv)), standardize(t[covs].to_numpy(float))])
    X_full = np.column_stack([X_red, z])
    r = nested_lr(X_red, X_full, yv, X_full.shape[1] - 1)
    return {"cohort": label, "covariates": " + ".join(covs), "n": len(yv), "n_AD": int(yv.sum()),
            "n_HC": int((1 - yv).sum()), **r}


def e_value(or_, lo, hi, common_outcome=True):
    """E-value (VanderWeele & Ding, 2017) for the OR and for the CI limit closest to 1."""
    def rr(o):
        return np.sqrt(o) if common_outcome else o

    def ev(r):
        r = 1 / r if r < 1 else r
        return float(r + np.sqrt(r * (r - 1)))

    if not np.isfinite(or_):
        return np.nan, np.nan
    e_pt = ev(rr(or_))
    if or_ >= 1:
        e_ci = 1.0 if (not np.isfinite(lo) or lo <= 1) else ev(rr(lo))
    else:
        e_ci = 1.0 if (not np.isfinite(hi) or hi >= 1) else ev(rr(hi))
    return e_pt, e_ci


def partial_spearman(x, y, Z=None):
    """Partial Spearman rho: correlation of the residuals of the ranks of x and y regressed on the ranks of Z."""
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    if Z is not None and np.asarray(Z).shape[1] > 0:
        Z = np.asarray(Z, float)
        R = np.column_stack([np.ones(len(rx))] + [stats.rankdata(Z[:, k]) for k in range(Z.shape[1])])
        rx = rx - R @ np.linalg.lstsq(R, rx, rcond=None)[0]
        ry = ry - R @ np.linalg.lstsq(R, ry, rcond=None)[0]
    if np.std(rx) == 0 or np.std(ry) == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def partial_spearman_ci(x, y, Z=None, n_boot=2000, seed=42):
    x, y = np.asarray(x, float), np.asarray(y, float)
    Z = None if Z is None else np.asarray(Z, float)
    rho = partial_spearman(x, y, Z)
    k = 0 if Z is None else Z.shape[1]
    df = len(x) - 2 - k
    p = (float(2 * stats.t.sf(abs(rho * np.sqrt(df / max(1 - rho ** 2, 1e-12))), df))
         if np.isfinite(rho) and df > 0 else np.nan)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        r = partial_spearman(x[i], y[i], None if Z is None else Z[i])
        if np.isfinite(r):
            boots.append(r)
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
    return rho, float(lo), float(hi), p


# ---- discrimination ------------------------------------------------------------------------------------------
def auc_ci(y, p, n_boot=2000, seed=42):
    """AUC and 95% CI by stratified bootstrap of subjects."""
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    if len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]
    boots = []
    for _ in range(n_boot):
        idx = np.r_[rng.choice(i1, len(i1)), rng.choice(i0, len(i0))]
        boots.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(roc_auc_score(y, p)), float(lo), float(hi)


def delta_auc_boot(y, p1, p2, n_boot=2000, seed=42):
    """AUC difference (p1 - p2) in the SAME subjects, with a paired bootstrap CI."""
    y, p1, p2 = np.asarray(y).astype(int), np.asarray(p1, float), np.asarray(p2, float)
    d = roc_auc_score(y, p1) - roc_auc_score(y, p2)
    rng = np.random.default_rng(seed)
    i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]
    diffs = []
    for _ in range(n_boot):
        idx = np.r_[rng.choice(i1, len(i1)), rng.choice(i0, len(i0))]
        diffs.append(roc_auc_score(y[idx], p1[idx]) - roc_auc_score(y[idx], p2[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(d), float(lo), float(hi)


def threshold_metrics(y, p, threshold=0.5):
    """Metrics at a fixed threshold (descriptive), comparable with the ds004504 literature."""
    y, yh = np.asarray(y).astype(int), (np.asarray(p, float) >= threshold).astype(int)
    tp, tn = int(((y == 1) & (yh == 1)).sum()), int(((y == 0) & (yh == 0)).sum())
    fp, fn = int(((y == 0) & (yh == 1)).sum()), int(((y == 1) & (yh == 0)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return {"sens": sens, "spec": spec, "bal_acc": (sens + spec) / 2, "acc": (tp + tn) / max(len(y), 1),
            "F1": 2 * tp / max(2 * tp + fp + fn, 1), "MCC": (tp * tn - fp * fn) / den if den > 0 else 0.0}


def benjamini_hochberg(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = np.empty(len(p))
    q[o] = np.minimum.accumulate((p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    return np.minimum(q, 1.0)


# ---- descriptive helpers ---------------------------------------------------------------------------------------
def p_fisher(tab):
    tab = np.asarray(tab)
    if tab.shape != (2, 2):
        return np.nan
    r = stats.fisher_exact(tab)
    return float(getattr(r, "pvalue", r[1]))


def p_mwu(a, b):
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    if len(a) < 2 or len(b) < 2 or (a.nunique() == 1 and b.nunique() == 1 and a.iloc[0] == b.iloc[0]):
        return np.nan
    return float(stats.mannwhitneyu(a, b).pvalue)


def mean_sd(s, with_n=False, nd=1):
    s = pd.Series(s).dropna()
    if len(s) == 0:
        return "—"
    txt = f"{s.mean():.{nd}f} ± {s.std():.{nd}f}"
    return f"{txt} ({len(s)})" if with_n else txt


def median_iqr(s):
    s = pd.Series(s).dropna()
    if len(s) == 0:
        return "—"
    return f"{s.median():.1f} [{s.quantile(.25):.1f}, {s.quantile(.75):.1f}]"


def value_range(s):
    s = pd.Series(s).dropna()
    return "—" if len(s) == 0 else f"{s.min():.0f}–{s.max():.0f}"


def hedges_g(a, b):
    """Hedges' g (a - b) with an approximate 95% CI."""
    a, b = pd.Series(a).dropna().astype(float), pd.Series(b).dropna().astype(float)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return np.nan, np.nan, np.nan
    sp = np.sqrt(((n1 - 1) * a.var() + (n2 - 1) * b.var()) / (n1 + n2 - 2))
    if not np.isfinite(sp) or sp == 0:
        return np.nan, np.nan, np.nan
    g = (1 - 3 / (4 * (n1 + n2) - 9)) * (a.mean() - b.mean()) / sp
    se = np.sqrt((n1 + n2) / (n1 * n2) + g ** 2 / (2 * (n1 + n2)))
    return g, g - 1.96 * se, g + 1.96 * se
