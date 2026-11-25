import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from eegad.stats import (auc_ci, benjamini_hochberg, delta_auc_boot, e_value, hedges_g, incremental_test, logit_fit,
                         nested_lr, partial_spearman, safe_logit, standardize, threshold_metrics)


def test_logit_fit_recovers_coefficients():
    rng = np.random.default_rng(0)
    n = 4000
    X = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n)])
    beta = np.array([-0.5, 1.0, -0.7])
    y = (rng.random(n) < 1 / (1 + np.exp(-X @ beta))).astype(int)
    fit = logit_fit(X, y)
    assert fit["converged"]
    assert np.allclose(fit["beta"], beta, atol=0.15)


def test_firth_is_finite_under_separation():
    x = np.r_[np.linspace(-2, -0.1, 20), np.linspace(0.1, 2, 20)]
    y = (x > 0).astype(int)
    X = np.column_stack([np.ones(len(x)), x])
    mle = logit_fit(X, y)
    firth = logit_fit(X, y, firth=True)
    assert abs(mle["beta"][1]) > 10 or not mle["converged"]
    assert np.isfinite(firth["beta"]).all() and abs(firth["beta"][1]) < 20


def test_nested_lr_and_incremental_test():
    rng = np.random.default_rng(1)
    n = 300
    age = rng.normal(70, 7, n)
    score = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(0.05 * (age - 70) + 1.2 * score)))).astype(int)
    p = 1 / (1 + np.exp(-score))
    r = incremental_test(pd.DataFrame({"y": y, "Age": age, "p": p}), ["Age"], "p", "test")
    assert r["n"] == n and r["OR"] > 1 and r["p"] < 0.001 and r["OR_lo"] < r["OR"] < r["OR_hi"]
    assert 0 <= r["p_one_sided"] <= 1 and r["method"] == "MLE"
    X0 = np.column_stack([np.ones(n), standardize(age[:, None])])
    X1 = np.column_stack([X0, rng.normal(size=n)])          # unrelated column
    r0 = nested_lr(X0, X1, y, 2)
    assert r0["LR"] >= 0 and 0 <= r0["p"] <= 1


def test_e_value_known_values():
    # common outcome: RR = sqrt(OR); E = RR + sqrt(RR (RR - 1))
    e_pt, e_ci = e_value(4.0, 2.25, 9.0)
    assert np.isclose(e_pt, 2 + np.sqrt(2))
    assert np.isclose(e_ci, 1.5 + np.sqrt(1.5 * 0.5))
    assert e_value(2.0, 0.8, 5.0)[1] == 1.0          # CI includes 1


def test_auc_ci_and_delta():
    rng = np.random.default_rng(2)
    y = np.r_[np.ones(40), np.zeros(40)].astype(int)
    p = np.r_[rng.normal(1, 1, 40), rng.normal(0, 1, 40)]
    a, lo, hi = auc_ci(y, p, n_boot=300, seed=1)
    assert np.isclose(a, roc_auc_score(y, p)) and lo < a < hi
    d, dlo, dhi = delta_auc_boot(y, p, p, n_boot=100, seed=1)
    assert d == 0 and dlo == 0 and dhi == 0


def test_threshold_metrics():
    m = threshold_metrics([1, 1, 0, 0], [0.9, 0.2, 0.1, 0.6], 0.5)
    assert m["sens"] == 0.5 and m["spec"] == 0.5 and m["acc"] == 0.5 and m["bal_acc"] == 0.5


def test_benjamini_hochberg():
    q = benjamini_hochberg([0.01, 0.04, 0.03, 0.2])
    assert np.allclose(q, [0.04, 0.0533333, 0.0533333, 0.2], atol=1e-6)


def test_hedges_g_and_partial_spearman():
    a, b = pd.Series([10, 12, 14, 16]), pd.Series([20, 22, 24, 26])
    g, lo, hi = hedges_g(a, b)
    assert g < 0 and lo < g < hi
    x = np.arange(20.0)
    assert np.isclose(partial_spearman(x, x ** 3), 1.0)


def test_safe_logit_is_clipped():
    assert np.isfinite(safe_logit([0.0, 1.0])).all()
