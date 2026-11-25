import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from eegad.design import (balance, comparable_pair_concordance, covariate_adjusted_auc, match_1to1,
                          overlap_weighted_auc_ci, propensity_score)


def _cohort(n=120, seed=0):
    rng = np.random.default_rng(seed)
    age = rng.normal(70, 8, n)
    edu = rng.normal(12, 4, n)
    sex = (rng.random(n) < 0.5).astype(float)
    y = (rng.random(n) < 1 / (1 + np.exp(-(0.08 * (age - 70) - 0.2 * (edu - 12))))).astype(int)
    p = 1 / (1 + np.exp(-(1.5 * y + rng.normal(size=n) - 0.75)))
    return pd.DataFrame({"subject_id": [f"s{i}" for i in range(n)], "y": y, "Age": age, "years_education": edu,
                         "sex_fem": sex, "p_eeg": p})


def test_overlap_weights_balance_the_covariates_exactly():
    t = _cohort()
    covs = ["Age", "years_education", "sex_fem"]
    ps, method = propensity_score(t, covs)
    w = np.where(t.y == 1, 1 - ps, ps)
    b = balance(t, {c: c for c in covs}, w)
    assert method == "MLE"
    assert np.allclose(b["SMD"], 0, atol=1e-6)        # exact mean balance (Li et al., 2018)


def test_matching_respects_the_caliper_without_reuse():
    t = _cohort()
    pairs = match_1to1(t, 2)
    assert (np.abs(pairs.var_ad - pairs.var_hc) <= 2).all()
    assert pairs.ad_id.is_unique and pairs.hc_id.is_unique


def test_concordance_with_all_pairs_eligible_equals_auc():
    t = _cohort()
    c = comparable_pair_concordance(t, lambda a, h: np.ones((len(a), len(h))), "p_eeg", n_boot=50, seed=1)
    assert np.isclose(c["C"], roc_auc_score(t.y, t.p_eeg))


def test_weighted_and_adjusted_auc_are_in_range():
    t = _cohort()
    ow = overlap_weighted_auc_ci(t, ["Age", "years_education"], "p_eeg", n_boot=50, seed=1)
    aa = covariate_adjusted_auc(t, ["Age", "years_education"], "p_eeg", n_boot=50, seed=1)
    for r in (ow, aa):
        assert 0.5 < r["AUC"] < 1 and r["lo"] <= r["AUC"] <= r["hi"]
