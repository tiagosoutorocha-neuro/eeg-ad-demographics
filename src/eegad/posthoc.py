"""Post hoc amendment 1 (notebook 09): local recalibration of the transported models of RQ3.

Kept apart from the pre-specified analysis: it has its own plan (config/post_hoc_plan.json), which states the hash of
the analysis plan it amends and is frozen in outputs/post_hoc/ before anything is computed; it only reads the
predictions saved by notebook 05 and writes to outputs/post_hoc/. Nothing in outputs/results/ is touched, so the
readings of RQ1-RQ3 and the reproduction check of notebook 08 do not change. Amendment 2 (notebooks 10 and
11) uses the same freezing mechanism with config/post_hoc_plan_2.json.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from .config import plan_hash
from .stats import logit_fit, safe_logit, sigmoid, threshold_metrics

POST_HOC_PLAN = Path("config") / "post_hoc_plan.json"          # amendment 1 (notebook 09)
POST_HOC_PLAN_2 = Path("config") / "post_hoc_plan_2.json"      # amendment 2 (notebooks 10 and 11)
METHODS = ("intercept_only", "intercept_and_slope")
METHOD_LABELS = {"frozen": "Frozen model (no recalibration)",
                 "intercept_only": "Recalibrated: intercept only (post hoc)",
                 "intercept_and_slope": "Recalibrated: intercept and slope (post hoc)"}


# ---- amendment plan ------------------------------------------------------------------------------------------------
def freeze_post_hoc_plan(ctx, plan_file: Path = POST_HOC_PLAN) -> tuple[dict, str, str]:
    """Read a post hoc amendment (config/post_hoc_plan*.json), check that it amends the plan in force, and freeze it
    in outputs/post_hoc/<name>_frozen.json (first run) or check that it did not change since (later runs).
    Returns (plan, hash, frozen_at)."""
    plan_file = Path(plan_file)
    ph = json.loads((ctx.repo / plan_file).read_text(encoding="utf-8"))
    h = plan_hash(ph)
    if ph["amends"]["plan_hash"] != ctx.plan_hash:
        raise RuntimeError(f"{plan_file} amends the analysis plan {ph['amends']['plan_hash']}, but the plan in "
                           f"force is {ctx.plan_hash}.")
    folder = ctx.out / "post_hoc"
    folder.mkdir(parents=True, exist_ok=True)
    f = folder / f"{plan_file.stem}_frozen.json"
    if f.exists():
        old = json.loads(f.read_text(encoding="utf-8"))
        if old["hash"] != h:
            raise RuntimeError(f"{plan_file} changed after it was frozen on {old['frozen_at']}. Undo the change, "
                               f"or write it as a new amendment and delete {f} on purpose (and say so in the paper).")
        frozen_at = old["frozen_at"]
        print(f"Post hoc amendment {ph['amendment']} matches the version frozen on {frozen_at} (hash {h}).")
    else:
        frozen_at = datetime.datetime.now().isoformat(timespec="seconds")
        f.write_text(json.dumps({"hash": h, "frozen_at": frozen_at, "amends_plan_hash": ctx.plan_hash, "plan": ph},
                                indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Post hoc amendment {ph['amendment']} frozen now ({frozen_at}), hash {h}: {f}")
    return ph, h, frozen_at


# ---- recalibration -------------------------------------------------------------------------------------------------
def _design(p):
    return np.column_stack([np.ones(len(p)), safe_logit(p)])


def fit_recalibration(y, p, method):
    """Parameters (a, b) of logit(p') = a + b * logit(p); b = 1 for the intercept-only version."""
    X, y = _design(p), np.asarray(y, int)
    fixed = {1: 1.0} if method == "intercept_only" else None
    r = logit_fit(X, y, fixed=fixed)
    if not r["converged"]:
        r = logit_fit(X, y, firth=True, fixed=fixed)
    return np.asarray(r["beta"], float), bool(r["converged"])


def apply_recalibration(p, beta):
    return sigmoid(_design(p) @ np.asarray(beta, float))


def recalibrate_loo(y, p, method):
    """Leave-one-subject-out: the recalibrated probability of each participant uses (a, b) estimated without them."""
    y, p = np.asarray(y, int), np.asarray(p, float)
    out = np.empty(len(y))
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        beta, _ = fit_recalibration(y[keep], p[keep], method)
        out[i] = apply_recalibration(p[i:i + 1], beta)[0]
    return out


# ---- metrics -------------------------------------------------------------------------------------------------------
def calibration_metrics(y, p, threshold=0.5):
    """Same definitions as notebook 05: slope and intercept with the slope fixed at 1 (calibration-in-the-large)."""
    y, p = np.asarray(y, int), np.asarray(p, float)
    X = _design(p)
    cal, citl = logit_fit(X, y), logit_fit(X, y, fixed={1: 1.0})
    return {"calibration_slope": float(cal["beta"][1]) if cal["converged"] else np.nan,
            "citl": float(citl["beta"][0]) if citl["converged"] else np.nan,
            "brier": float(brier_score_loss(y, p)), "auc": float(roc_auc_score(y, p)) if len(set(y)) == 2 else np.nan,
            "mean_p": float(p.mean()), "prevalence": float(y.mean()), "n": int(len(y)), "n_AD": int(y.sum()),
            "n_HC": int((1 - y).sum()), "metrics": threshold_metrics(y, p, threshold)}


def bootstrap_intervals(y, p_ref, p_new, n_boot, seed):
    """95% percentile intervals (participants resampled, stratified by diagnosis; probabilities held fixed) for the
    calibration slope, calibration-in-the-large and Brier of p_new, and for Brier(p_new) - Brier(p_ref)."""
    y, p_ref, p_new = np.asarray(y, int), np.asarray(p_ref, float), np.asarray(p_new, float)
    rng = np.random.default_rng(seed)
    i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]
    rows = []
    for _ in range(int(n_boot)):
        idx = np.r_[rng.choice(i1, len(i1)), rng.choice(i0, len(i0))]
        yb, pb = y[idx], p_new[idx]
        X = _design(pb)
        cal, citl = logit_fit(X, yb), logit_fit(X, yb, fixed={1: 1.0})
        rows.append((cal["beta"][1] if cal["converged"] else np.nan, citl["beta"][0] if citl["converged"] else np.nan,
                     brier_score_loss(yb, pb), brier_score_loss(yb, pb) - brier_score_loss(yb, p_ref[idx])))
    B = np.asarray(rows, float)

    def ci(col):
        v = B[:, col][np.isfinite(B[:, col])]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [np.nan, np.nan]

    return {"calibration_slope": ci(0), "citl": ci(1), "brier": ci(2), "delta_brier": ci(3),
            "resamples_without_convergence": int(np.sum(~np.isfinite(B[:, 0]) | ~np.isfinite(B[:, 1])))}


def wilson_interval(k, n, z=1.959964):
    if n == 0:
        return np.nan, np.nan
    ph = k / n
    den = 1 + z ** 2 / n
    centre = (ph + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(ph * (1 - ph) / n + z ** 2 / (4 * n ** 2)) / den
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def calibration_bins(y, p, n_bins=5):
    """Observed proportion of AD in quantile bins of the predicted probability, with Wilson 95% intervals."""
    y, p = np.asarray(y, int), np.asarray(p, float)
    order = np.argsort(p, kind="mergesort")
    rows = []
    for chunk in np.array_split(order, n_bins):
        if len(chunk) == 0:
            continue
        k, n = int(y[chunk].sum()), len(chunk)
        lo, hi = wilson_interval(k, n)
        rows.append({"mean_predicted": float(p[chunk].mean()), "observed": k / n, "lo": lo, "hi": hi, "n": n})
    return pd.DataFrame(rows)


def logistic_calibration_curve(y, p, grid):
    """Fitted relation observed vs predicted: sigmoid(a + b * logit(grid)), with (a, b) from y ~ logit(p)."""
    beta, _ = fit_recalibration(y, p, "intercept_and_slope")
    return apply_recalibration(grid, beta)


# ---- the whole amendment for one transport direction ----------------------------------------------------------------
def recalibration_analysis(pred, label, threshold, n_boot, seed, n_bins=5):
    """pred: subject-level predictions of a frozen model in the target cohort (columns subject_id, y, p)."""
    d = pred.dropna(subset=["y", "p"]).sort_values("subject_id").reset_index(drop=True)
    y, p = d["y"].to_numpy(int), d["p"].to_numpy(float)
    res = {"label": label, "versions": {}, "equations": {}}
    probs = {"frozen": p}
    res["versions"]["frozen"] = calibration_metrics(y, p, threshold)
    for m in METHODS:
        probs[m] = recalibrate_loo(y, p, m)
        res["versions"][m] = calibration_metrics(y, probs[m], threshold)
        res["versions"][m]["ci"] = bootstrap_intervals(y, p, probs[m], n_boot, seed)
        beta, conv = fit_recalibration(y, p, m)
        res["equations"][m] = {"a": float(beta[0]), "b": float(beta[1]), "converged": conv}
    res["versions"]["frozen"]["ci"] = bootstrap_intervals(y, p, p, n_boot, seed)
    res["bins"] = {k: calibration_bins(y, v, n_bins).to_dict(orient="list") for k, v in probs.items()}
    subj = d[["subject_id", "y"]].assign(p_frozen=p, **{f"p_{m}": probs[m] for m in METHODS})
    return res, subj


RECALIBRATION_NOTE = ("Post hoc (amendment 1). Recalibrated probabilities are leave-one-subject-out within the target cohort "
                 "(each participant's parameters were estimated without them); the frozen models were not retrained. "
                 "A recalibration is a monotone transformation, so the ranking of the participants and the AUC are "
                 "those of the frozen model (pooled leave-one-out probabilities are slightly pessimistic for "
                 "discrimination and calibration slope; see recalibration.json). Threshold metrics at P(AD) = 0.5. "
                 "Intervals: {n_boot} bootstrap resamples of participants, stratified by diagnosis, with the "
                 "probabilities held fixed.")


def recalibration_table(results: dict) -> pd.DataFrame:
    """Table S7 of the article: one row per direction and version."""
    def f(x, nd=3):
        return "n/a" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"

    rows = []
    for key, r in results.items():
        for v in ("frozen",) + METHODS:
            m, ci = r["versions"][v], r["versions"][v]["ci"]
            eq = r["equations"].get(v)
            rows.append({
                "direction": r["label"], "version": METHOD_LABELS[v], "n (AD/HC)": f"{m['n']} ({m['n_AD']}/{m['n_HC']})",
                "calibration slope [95% CI]": f"{f(m['calibration_slope'], 2)} [{f(ci['calibration_slope'][0], 2)}, "
                                              f"{f(ci['calibration_slope'][1], 2)}]",
                "calibration-in-the-large [95% CI]": f"{f(m['citl'], 2)} [{f(ci['citl'][0], 2)}, {f(ci['citl'][1], 2)}]",
                "Brier [95% CI]": f"{f(m['brier'])} [{f(ci['brier'][0])}, {f(ci['brier'][1])}]",
                "change in Brier vs frozen [95% CI]": ("—" if v == "frozen" else
                                                       f"{f(m['brier'] - r['versions']['frozen']['brier'])} "
                                                       f"[{f(ci['delta_brier'][0])}, {f(ci['delta_brier'][1])}]"),
                "sensitivity": f(m["metrics"]["sens"]), "specificity": f(m["metrics"]["spec"]),
                "balanced accuracy": f(m["metrics"]["bal_acc"]), "MCC": f(m["metrics"]["MCC"]),
                "AUC": f(m["auc"]) if v == "frozen" else "as frozen (see note)",
                "recalibration (whole target cohort)": ("—" if eq is None else
                                                        f"logit p' = {eq['a']:.2f} + {eq['b']:.2f} × logit p"),
            })
    return pd.DataFrame(rows)
