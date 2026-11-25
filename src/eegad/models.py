"""Classifiers, leave-one-subject-out (LOSO), transport of frozen models between cohorts, the parametric null
bootstrap of the whole LOSO (RQ1) and subject-level leave-one-out logistic regression."""
from __future__ import annotations

import datetime
import hashlib
import json
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .stats import incremental_test, logit_fit, sigmoid, standardize

try:
    from xgboost import XGBClassifier
    XGB_OK = True
except ImportError:  # the XGBoost sensitivity analysis is skipped
    XGB_OK = False


# ---- cache -----------------------------------------------------------------------------------------------------
def fingerprint(df, cols):
    """Fingerprint of the data (numbers rounded to 10 decimals, so it does not depend on the last bit)."""
    d = df[list(cols)].copy()
    num = d.select_dtypes("number").columns
    d[num] = d[num].astype(float).round(10)
    return hashlib.sha1(pd.util.hash_pandas_object(d, index=False).values.tobytes()).hexdigest()[:12]


def cache_key(**kw):
    return hashlib.sha1(json.dumps(kw, sort_keys=True, default=str).encode()).hexdigest()[:16]


def cache_get(ctx, name, key):
    f = ctx.out / "cache" / f"{name}_{key}.pkl"
    if f.exists():
        with open(f, "rb") as fh:
            return pickle.load(fh)
    return None


def cache_put(ctx, name, key, obj):
    with open(ctx.out / "cache" / f"{name}_{key}.pkl", "wb") as fh:
        pickle.dump(obj, fh)


# ---- models ----------------------------------------------------------------------------------------------------
def positive_class_score(model, X):
    """Probability of the AD class (label 1), from the right column of predict_proba."""
    proba = model.predict_proba(X)
    classes = np.asarray(getattr(model, "classes_", [0, 1]))
    if 1 in classes:
        return proba[:, int(np.where(classes == 1)[0][0])]
    return np.zeros(len(X))


def model_factories(ctx):
    """{model name: factory(y_train, groups_train) -> unfitted pipeline}, built from the analysis plan."""
    cfg = ctx.plan["classifiers"]

    def rf(y_tr=None, g_tr=None):
        c = cfg["RandomForest"]
        return Pipeline([("scaler", StandardScaler()),
                         ("clf", RandomForestClassifier(
                             n_estimators=ctx.rt["rf_trees"], max_depth=c["max_depth"], max_features=c["max_features"],
                             min_samples_leaf=c["min_samples_leaf"], bootstrap=c["bootstrap"],
                             class_weight=c["class_weight"], random_state=c["random_state"], n_jobs=ctx.n_jobs))])

    def xgb(y_tr, g_tr=None):
        c = cfg["XGBoost"]
        y_tr = np.asarray(y_tr)
        spw = float(np.sum(y_tr == 0)) / max(float(np.sum(y_tr == 1)), 1.0)   # training fold only
        return Pipeline([("scaler", StandardScaler()),
                         ("clf", XGBClassifier(
                             n_estimators=ctx.rt["xgb_trees"], learning_rate=c["learning_rate"],
                             max_depth=c["max_depth"], subsample=c["subsample"],
                             colsample_bytree=c["colsample_bytree"], eval_metric=c["eval_metric"],
                             scale_pos_weight=spw, random_state=c["random_state"], n_jobs=ctx.n_jobs, verbosity=0))])

    def svm(y_tr, g_tr):
        c = cfg["SVM"]
        base = SVC(kernel=c["kernel"], C=c["C"], gamma=c["gamma"], random_state=c["random_state"])
        if c["calibration_grouped_by_subject"]:
            cv = list(StratifiedGroupKFold(n_splits=c["calibration_folds"]).split(
                np.zeros(len(y_tr)), np.asarray(y_tr), np.asarray(g_tr)))
        else:
            cv = c["calibration_folds"]
        return Pipeline([("scaler", StandardScaler()),
                         ("clf", CalibratedClassifierCV(base, cv=cv, method=c["calibration"]))])

    f = {"RandomForest": rf, "SVM": svm}
    if XGB_OK:
        f["XGBoost"] = xgb
    return f


def run_loso(df, feats, model, name, ctx, verbose=True):
    """Leave-one-subject-out; returns DataFrame(subject_id, y, p, n_epochs, fold). Cached by name + data + settings."""
    feats = list(feats)
    X = df[feats].to_numpy(float)
    y = df["y"].to_numpy(int)
    g = df["subject_id"].astype(str).to_numpy()
    key = cache_key(name=name, feats=feats, model=model, data=fingerprint(df, feats + ["y", "subject_id"]),
                    cfg=ctx.plan["classifiers"][model], rt=ctx.rt)
    hit = cache_get(ctx, name, key)
    if hit is not None:
        if verbose:
            print(f"[cache] {name}: {len(hit['pred'])} subjects (originally run in {hit['seconds']:.0f} s)")
        return hit["pred"]
    factories = model_factories(ctx)
    folds = list(LeaveOneGroupOut().split(X, y, g))
    rows, t0 = [], time.time()
    for k, (tr, te) in enumerate(folds):
        if len(np.unique(y[tr])) < 2:
            s = np.full(len(te), float(y[tr][0]))
        else:
            m = factories[model](y[tr], g[tr])
            m.fit(X[tr], y[tr])
            s = positive_class_score(m, X[te])
        rows.append({"subject_id": g[te][0], "y": int(y[te][0]), "p": float(np.mean(s)),
                     "n_epochs": int(len(te)), "fold": k})
        if verbose and ((k + 1) % 10 == 0 or k + 1 == len(folds)):
            dt = time.time() - t0
            print(f"  {name}: {k + 1}/{len(folds)} folds | {dt:.0f} s | ~{dt / (k + 1) * (len(folds) - k - 1):.0f} s left")
    pred = pd.DataFrame(rows)
    cache_put(ctx, name, key, {"pred": pred, "seconds": time.time() - t0})
    return pred


def transport(df_tr, df_te, feats, model, name, ctx):
    """Train on the whole source cohort and apply ONCE to the other cohort (without its labels). The fitted model,
    its predictions and a metadata file are stored in outputs/models/; a later run reuses them if nothing changed."""
    feats = list(feats)
    meta = {"name": name, "model": model, "feats": feats, "plan_hash": ctx.plan_hash, "rt": ctx.rt,
            "train_data": fingerprint(df_tr, feats + ["y", "subject_id"]),
            "test_data": fingerprint(df_te, feats + ["subject_id"])}
    f_json = ctx.out / "models" / f"{name}_{ctx.tag}.json"
    f_pred = ctx.out / "models" / f"{name}_{ctx.tag}_predictions.csv"
    if f_json.exists() and f_pred.exists():
        old = json.loads(f_json.read_text(encoding="utf-8"))
        if all(old.get(k) == meta[k] for k in ("model", "feats", "plan_hash", "train_data", "test_data")):
            print(f"[already applied on {old['applied_at']}] {name}: not re-applied")
            return pd.read_csv(f_pred, dtype={"subject_id": str}, float_precision="round_trip")
        ctx.log.alert(f"{name}: data, features or plan changed since it was applied on {old['applied_at']}; re-applied.")
    Xtr, ytr = df_tr[feats].to_numpy(float), df_tr["y"].to_numpy(int)
    m = model_factories(ctx)[model](ytr, df_tr["subject_id"].astype(str).to_numpy()).fit(Xtr, ytr)
    f_mod = ctx.out / "models" / f"{name}_{ctx.tag}.pkl"
    with open(f_mod, "wb") as fh:
        pickle.dump(m, fh)
    pe = positive_class_score(m, df_te[feats].to_numpy(float))
    pred = (pd.DataFrame({"subject_id": df_te["subject_id"].astype(str).to_numpy(), "p": pe})
            .groupby("subject_id", sort=True)["p"].mean().reset_index())
    pred.to_csv(f_pred, index=False)
    meta.update({"applied_at": datetime.datetime.now().isoformat(timespec="seconds"),
                 "model_sha1": hashlib.sha1(f_mod.read_bytes()).hexdigest()[:12], "n_test_subjects": len(pred)})
    f_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return pred


def null_bootstrap(df_ep, subj, covs, name, ctx, one_sided=False, model="RandomForest", feats=None):
    """p-value of the EEG score beyond the covariates by parametric bootstrap under H0 (Shang et al., 2025, adapted
    to a binary outcome): simulate y* from the null model P(AD | covariates), re-run the WHOLE LOSO with y* and
    recompute the statistic. Keeps the correlation between EEG and covariates, which is what makes the naive test
    liberal. Each replicate is cached; Besag & Clifford (1991) sequential stopping."""
    feats = ctx.features if feats is None else list(feats)
    h_stop = ctx.plan["rq1"]["null_bootstrap"]["stop_after_exceedances"]
    B = ctx.rt["null_B"]
    t = subj.dropna(subset=list(covs) + ["p_eeg"]).reset_index(drop=True)
    obs = incremental_test(t, covs, "p_eeg", name)

    def stat(r):
        return float(np.sign(r["beta"]) * np.sqrt(r["LR"])) if one_sided else float(r["LR"])

    T_obs = stat(obs)
    Xc = np.column_stack([np.ones(len(t)), standardize(t[list(covs)].to_numpy(float))])
    fit = logit_fit(Xc, t["y"].to_numpy(int))
    if (not fit["converged"]) or np.max(np.abs(fit["beta"][1:])) > 10:
        fit = logit_fit(Xc, t["y"].to_numpy(int), firth=True)
    p0 = sigmoid(Xc @ fit["beta"])
    Ystar = (np.random.default_rng(ctx.seed + 1000).random((B, len(t))) < p0[None, :]).astype(int)
    ids = t["subject_id"].astype(str).to_numpy()
    sid_ep = df_ep["subject_id"].astype(str).to_numpy()
    y_ep = df_ep["y"].to_numpy(int)
    T, exceed, t0 = [], 0, time.time()
    for b in range(B):
        mapping = dict(zip(ids, Ystar[b]))
        d = df_ep.copy()
        d["y"] = [int(mapping.get(s, yy)) for s, yy in zip(sid_ep, y_ep)]   # subjects without covariates keep y
        pred = run_loso(d, feats, model, f"{name}_null", ctx, verbose=False)
        s_ = t[["subject_id"] + list(covs)].merge(pred[["subject_id", "y", "p"]], on="subject_id")
        Tb = stat(incremental_test(s_, covs, "p", name))
        T.append(Tb)
        exceed += int(Tb >= T_obs)
        if (b + 1) % 10 == 0 or exceed >= h_stop or b + 1 == B:
            dt = time.time() - t0
            print(f"  null bootstrap {name}: {b + 1}/{B} replicates | {exceed} >= observed | {dt / 60:.1f} min")
        if exceed >= h_stop:          # Besag & Clifford (1991): p clearly above 0.05
            break
    nb = len(T)
    p = exceed / nb if exceed >= h_stop else (exceed + 1) / (nb + 1)
    return {"T_obs": T_obs, "T": np.asarray(T), "p": float(p), "replicates": nb, "exceedances": exceed,
            "stopped_early": exceed >= h_stop, "one_sided": one_sided, "naive": obs}


def loo_logistic(tab, cols):
    """Balanced logistic regression, leave-one-subject-out at the subject level."""
    X, yv = tab[cols].to_numpy(float), tab["y"].to_numpy(int)
    p = np.zeros(len(yv))
    for i in range(len(yv)):
        tr = np.arange(len(yv)) != i
        mdl = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                            LogisticRegression(max_iter=5000, class_weight="balanced"))
        mdl.fit(X[tr], yv[tr])
        p[i] = mdl.predict_proba(X[i:i + 1])[0, 1]
    return pd.DataFrame({"subject_id": tab["subject_id"].to_numpy(), "y": yv, "p": p})


def standardize_within_cohort(d, features):
    """Each feature standardised with the mean and SD of all epochs of its own cohort (no labels)."""
    d = d.copy()
    sd = d[features].std(ddof=1).replace(0, 1.0)
    d[features] = (d[features] - d[features].mean()) / sd
    return d
