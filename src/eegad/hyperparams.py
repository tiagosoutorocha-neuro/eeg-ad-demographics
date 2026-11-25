"""Post hoc amendment 2, part A (notebook 10): the frozen pipeline re-run with a small grid of Random Forest
hyperparameters, changed one at a time. Every configuration is reported; none is chosen.

The frozen configuration goes through exactly the same code as notebooks 02 and 05 (models.run_loso and the same
pipeline), so it must reproduce their predictions; the notebook checks this before showing anything."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .models import cache_get, cache_key, cache_put, fingerprint, positive_class_score
from .stats import auc_ci, incremental_test, logit_fit, safe_logit

RF_KEYS = ("n_estimators", "max_depth", "max_features", "min_samples_leaf", "bootstrap", "class_weight",
           "random_state")


def configurations(ctx, grid):
    """Full RF settings of each grid entry: the plan's settings with one change. In quick test mode the number of
    trees follows the quick setting (and 'n_estimators = 1000' becomes 1000/300 times it)."""
    base = {k: ctx.plan["classifiers"]["RandomForest"][k] for k in RF_KEYS}
    base["n_estimators"] = ctx.rt["rf_trees"]
    out = []
    for g in grid:
        c = dict(base)
        for k, v in g["change"].items():
            if k == "n_estimators":
                v = int(round(v * ctx.rt["rf_trees"] / ctx.plan["classifiers"]["RandomForest"]["n_estimators"]))
            c[k] = v
        out.append({"label": g["label"], "frozen": not g["change"], "params": c})
    return out


def rf_pipeline(params, n_jobs):
    """Same pipeline as models.model_factories (StandardScaler + RandomForest), with the given settings."""
    return Pipeline([("scaler", StandardScaler()),
                     ("clf", RandomForestClassifier(**params, n_jobs=n_jobs))])


def loso_config(df, feats, params, name, ctx):
    """Leave-one-subject-out exactly as models.run_loso, for any RF settings; cached in outputs/cache/."""
    feats = list(feats)
    X, y = df[feats].to_numpy(float), df["y"].to_numpy(int)
    g = df["subject_id"].astype(str).to_numpy()
    key = cache_key(name=name, feats=feats, params=params, data=fingerprint(df, feats + ["y", "subject_id"]))
    hit = cache_get(ctx, name, key)
    if hit is not None:
        print(f"[cache] {name}")
        return hit["pred"]
    rows, t0 = [], time.time()
    folds = list(LeaveOneGroupOut().split(X, y, g))
    for k, (tr, te) in enumerate(folds):
        if len(np.unique(y[tr])) < 2:
            s = np.full(len(te), float(y[tr][0]))
        else:
            m = rf_pipeline(params, ctx.n_jobs).fit(X[tr], y[tr])
            s = positive_class_score(m, X[te])
        rows.append({"subject_id": g[te][0], "y": int(y[te][0]), "p": float(np.mean(s)), "n_epochs": int(len(te)),
                     "fold": k})
    pred = pd.DataFrame(rows)
    print(f"  {name}: {len(folds)} folds in {time.time() - t0:.0f} s")
    cache_put(ctx, name, key, {"pred": pred, "seconds": time.time() - t0})
    return pred


def transport_config(df_tr, df_te, feats, params, name, ctx):
    """Train on the whole source cohort, apply once to the other (as models.transport, without saving the model)."""
    feats = list(feats)
    key = cache_key(name=name, feats=feats, params=params, train=fingerprint(df_tr, feats + ["y", "subject_id"]),
                    test=fingerprint(df_te, feats + ["subject_id"]))
    hit = cache_get(ctx, name, key)
    if hit is not None:
        print(f"[cache] {name}")
        return hit["pred"]
    m = rf_pipeline(params, ctx.n_jobs).fit(df_tr[feats].to_numpy(float), df_tr["y"].to_numpy(int))
    pe = positive_class_score(m, df_te[feats].to_numpy(float))
    pred = (pd.DataFrame({"subject_id": df_te["subject_id"].astype(str).to_numpy(), "p": pe})
            .groupby("subject_id", sort=True)["p"].mean().reset_index())
    cache_put(ctx, name, key, {"pred": pred})
    return pred


def transport_metrics(pred, subj_target, n_boot, seed):
    s = pred.merge(subj_target[["subject_id", "y"]], on="subject_id", how="inner")
    a = auc_ci(s["y"], s["p"], n_boot, seed)
    cal = logit_fit(np.column_stack([np.ones(len(s)), safe_logit(s["p"])]), s["y"].to_numpy(int))
    return {"auc": a, "calibration_slope": float(cal["beta"][1]) if cal["converged"] else np.nan, "n": len(s)}


def within_metrics(pred, subjects, covs, cohort, n_boot, seed):
    a = auc_ci(pred["y"], pred["p"], n_boot, seed)
    s = subjects.merge(pred[["subject_id", "p"]].rename(columns={"p": "p_eeg"}), on="subject_id", how="left")
    r = incremental_test(s, covs, "p_eeg", cohort)
    return {"auc": a, "OR": r["OR"], "OR_lo": r["OR_lo"], "OR_hi": r["OR_hi"], "LR": r["LR"], "method": r["method"],
            "n": r["n"]}


def sensitivity_table(results):
    """Table S11: one row per configuration."""
    def ci(v, nd=3):
        return f"{v[0]:.{nd}f} [{v[1]:.{nd}f}, {v[2]:.{nd}f}]"

    def num(x):
        return "∞" if not np.isfinite(x) else "> 999" if x > 999 else f"{x:.2f}"

    def orci(r):
        return f"{num(r['OR'])} [{num(r['OR_lo'])}, {num(r['OR_hi'])}]"

    rows = []
    for r in results:
        rows.append({"configuration": r["label"] + (" (frozen)" if r["frozen"] and "Frozen" not in r["label"] else ""),
                     "BrainLat AUC [95% CI]": ci(r["BrainLat"]["auc"]),
                     "BrainLat OR/SD [95% CI]": orci(r["BrainLat"]), "BrainLat LR χ²": f"{r['BrainLat']['LR']:.2f}",
                     "Greek AUC [95% CI]": ci(r["Greek"]["auc"]),
                     "Greek OR/SD [95% CI]": orci(r["Greek"]), "Greek LR χ²": f"{r['Greek']['LR']:.2f}",
                     "BrainLat → Greek AUC [95% CI]": ci(r["RQ3a"]["auc"]),
                     "Greek → BrainLat AUC [95% CI]": ci(r["RQ3b"]["auc"]),
                     "calibration slope (BL → GR / GR → BL)":
                         f"{r['RQ3a']['calibration_slope']:.2f} / {r['RQ3b']['calibration_slope']:.2f}"})
    return pd.DataFrame(rows)


HYPERPARAMETER_NOTE = ("Post hoc (amendment 2). One Random Forest hyperparameter changed at a time; everything else "
                       "as in the analysis plan. All configurations are shown and none was selected. AUC: "
                       "leave-one-subject-out within the cohort, or the frozen model applied once to the other cohort; "
                       "bootstrap 95% CI ({n_boot} resamples). OR/SD: odds ratio of AD per 1 SD of the EEG score beyond "
                       "the demographics of the cohort (profile-likelihood 95% CI); LR χ²: likelihood-ratio statistic "
                       "(1 df). No p-values: the null bootstrap of the frozen configuration (Table 4) is the only valid "
                       "test, and the naive p is liberal.")
