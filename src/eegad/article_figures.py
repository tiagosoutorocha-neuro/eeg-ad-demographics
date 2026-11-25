"""Article figures, one per research question (notebook 12). They only read files already in outputs/.

Fig. 4  RQ1  incremental value: odds ratios + null-bootstrap distributions
Fig. 5  RQ2  confounding by design: confounding-free AUCs + covariate balance (Love plots)
Fig. 6  RQ3  transportability: calibration of the frozen models and post hoc recalibration
Fig. 7  per-feature consistency between the cohorts (descriptive)
Fig. S1 angular error of the BrainLat 10-20 correspondence (descriptive; amendment 2)
Fig. S2 group-average spectra by cohort and diagnosis (descriptive; amendment 2)
Fig. S3 E-value (unmeasured education in the Greek cohort)
Fig. S4 domain shift: differences between cohorts in the controls vs disease effects within each cohort
Fig. S5 hyperparameter sensitivity of the Random Forest (post hoc; amendment 2)
"""
from __future__ import annotations

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter
from scipy import stats

from .figures import apply_style
from .posthoc import calibration_bins, logistic_calibration_curve

COL1, COL2 = 3.31, 6.85            # Springer column widths (84 and 174 mm), inches
COHORT = {"BrainLat": "#D55E00", "Greek": "#0072B2"}
GROUP = {"AD": "#762A83", "HC": "#1B7837", "FTD": "#9E9E9E"}
VERSION = {"frozen": "#E69F00", "intercept_and_slope": "#009E73", "intercept_only": "#56B4E9"}
FEATURE_LABELS = {"Rel_Theta_mean": "Relative θ power", "Rel_Alpha_mean": "Relative α power",
                  "Rel_Beta_mean": "Relative β power", "Rel_Gamma_mean": "Relative γ power",
                  "Ratio_Theta_Alpha": "θ/α ratio", "Ratio_Theta_Beta": "θ/β ratio",
                  "Spectral_Entropy": "Spectral entropy", "PLV_Theta_mean": "Phase locking (PLV), θ",
                  "PLV_Alpha_mean": "Phase locking (PLV), α", "PLV_Beta_mean": "Phase locking (PLV), β",
                  "Corr_Broadband_mean": "Broadband correlation"}


def article_style():
    """Style of the existing figures, with the sans-serif lettering recommended by Springer (Arial/Helvetica)."""
    apply_style()
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
                         "mathtext.fontset": "dejavusans", "svg.fonttype": "none"})


def panel_label(ax, letter, x=-0.12, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fmt_p(p, one_sided=False):
    if p is None or not np.isfinite(p):
        return "n/a"
    s = "< 0.001" if p < 0.001 else f"= {p:.3f}"
    return f"one-sided p {s}" if one_sided else f"p {s}"


def log2_ticks(ax, vmin, vmax, max_ticks=10):
    """Readable ticks on a log axis: powers of 2 for moderate ranges, 1-2-5 decades for wide ones."""
    if vmax / vmin <= 2.0 ** 9:
        k_lo, k_hi = int(np.floor(np.log2(vmin))), int(np.ceil(np.log2(vmax)))
        step = max(1, int(np.ceil((k_hi - k_lo + 1) / max_ticks)))
        ticks = {2.0 ** k for k in range(k_lo, k_hi + 1, step)} | {1.0}
    else:
        d_lo, d_hi = int(np.floor(np.log10(vmin))), int(np.ceil(np.log10(vmax)))
        mult = (1,) if d_hi - d_lo > 4 else (1, 3)
        ticks = {m * 10.0 ** d for d in range(d_lo, d_hi + 1) for m in mult} | {1.0}
    ticks = sorted(t for t in ticks if vmin <= t <= vmax)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_formatter(NullFormatter())


# ---- Fig. 2 — participant flow at article size --------------------------------------------------------------------
def _flow_text(r):
    ftd = r.get("FTD")
    extra = f", FTD {int(ftd)}" if ftd is not None and pd.notna(ftd) and float(ftd) > 0 else ""
    return f"{r['step']}\nn = {int(r['n'])} (AD {int(r['AD'])}{extra}, HC {int(r['HC'])})"


def _flow_panel(ax, title, rows):
    import textwrap
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=8, fontweight="bold")
    rows = rows.reset_index(drop=True)
    ys = np.linspace(0.93, 0.06, len(rows))
    wrap = lambda s, w: "\n".join(textwrap.fill(t, w) for t in s.split("\n"))
    for i, (y, r) in enumerate(zip(ys, rows.to_dict("records"))):
        ax.text(0.33, y, wrap(_flow_text(r), 36), ha="center", va="center", fontsize=6.3,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", lw=0.7))
        if i < len(rows) - 1:
            ax.annotate("", xy=(0.33, ys[i + 1] + 0.055), xytext=(0.33, y - 0.055),
                        arrowprops=dict(arrowstyle="-|>", lw=0.7, color="black"))
            exc = r.get("excluded_at_step")
            n_exc = int(float(exc)) if exc is not None and pd.notna(exc) and str(exc).strip() != "" else 0
            if n_exc > 0:
                ym = (y + ys[i + 1]) / 2
                ax.annotate("", xy=(0.66, ym), xytext=(0.33, ym),
                            arrowprops=dict(arrowstyle="-|>", lw=0.7, color="0.35"))
                reason = "" if pd.isna(r.get("reason")) else re.sub(r"\s*\(n = \d+\)", "", str(r.get("reason")))
                ax.text(0.82, ym, wrap(f"Excluded: n = {n_exc}\n{reason}", 22), ha="center", va="center",
                        fontsize=6.0, color="0.2",
                        bbox=dict(boxstyle="round,pad=0.35", fc="#F2F2F2", ec="0.5", lw=0.6))


def fig_flow(flow):
    """Participant flow (same content as fig0) sized for the article (174 mm wide, lettering >= 6 pt)."""
    fig, axs = plt.subplots(1, 2, figsize=(COL2, 5.6))
    _flow_panel(axs[0], "BrainLat (Argentina and Chile)", flow[flow.cohort == "BrainLat"])
    _flow_panel(axs[1], "Greek cohort (ds004504)", flow[flow.cohort.isin(["Greek", "Grego"])])
    fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.01, wspace=0.05)
    return fig


# ---- Fig. 4 — RQ1 ---------------------------------------------------------------------------------------------------
def rq1_rows(inc, tr):
    """Rows of the odds-ratio panel from results/incremental.json and results/transport.json."""
    I, P, H, B, ind = inc["INC"], inc["R_POOL"], inc["R_HET"], inc["BOOT"], tr["INC_IND"]
    g = lambda r: (r["OR"], r["OR_lo"], r["OR_hi"])
    return [
        {"label": "BrainLat — within-cohort EEG score\n(beyond age, sex, education)", "est": g(I["BrainLat"]),
         "note": f"null bootstrap {fmt_p(B['BrainLat']['p'])}", "kind": "primary", "cohort": "BrainLat"},
        {"label": "Greek — within-cohort EEG score\n(beyond age, sex)", "est": g(I["Greek"]),
         "note": f"null bootstrap {fmt_p(B['Greek']['p'], True)}", "kind": "primary", "cohort": "Greek"},
        {"label": "Both cohorts — common EEG effect", "est": g(P),
         "note": f"{fmt_p(P['p'])}; cohort × EEG {fmt_p(H['p'])}", "kind": "pooled", "cohort": None},
        {"label": "BrainLat — score of the Greek model\n(independent)", "est": g(ind["BrainLat (Greek model)"]),
         "note": fmt_p(ind["BrainLat (Greek model)"]["p"]), "kind": "independent", "cohort": "BrainLat"},
        {"label": "Greek — score of the BrainLat model\n(independent)", "est": g(ind["Greek (BrainLat model)"]),
         "note": fmt_p(ind["Greek (BrainLat model)"]["p_one_sided"], True), "kind": "independent", "cohort": "Greek"},
        {"label": "BrainLat — adjusted also for site (check)", "est": g(I["BrainLat + site"]),
         "note": fmt_p(I["BrainLat + site"]["p"]), "kind": "check", "cohort": "BrainLat"},
    ]


def draw_or_forest(ax, rows):
    marker = {"primary": "o", "pooled": "D", "independent": "s", "check": "^"}
    n = len(rows)
    vals = []
    for i, r in enumerate(rows):
        yy = n - 1 - i
        o, lo, hi = r["est"]
        color = COHORT.get(r["cohort"], "black") if r["cohort"] else "black"
        filled = r["kind"] in ("primary", "pooled")
        ax.errorbar(o, yy, xerr=[[o - lo if np.isfinite(lo) else 0], [hi - o if np.isfinite(hi) else 0]],
                    fmt=marker[r["kind"]], color=color, mfc=color if filled else "white", mec=color, ms=5,
                    capsize=2, lw=0.9)
        ax.text(1.02, yy, r["note"], transform=ax.get_yaxis_transform(), va="center", fontsize=6.8)
        vals += [v for v in (o, lo, hi) if np.isfinite(v) and v > 0]
    for cut in (n - 2.5, n - 3.5, n - 5.5):
        ax.axhline(cut, color="0.85", lw=0.6)
    ax.axvline(1, color="0.5", lw=0.7, ls="--")
    ax.set_yticks(range(n))
    ax.set_yticklabels([r["label"] for r in rows][::-1], fontsize=6.8)
    ax.set_xscale("log")
    vmin, vmax = min(vals) / 1.4, max(vals) * 1.4
    ax.set_xlim(vmin, vmax)
    log2_ticks(ax, vmin, vmax)
    ax.set_xlabel("Odds ratio per 1 SD of the EEG score, beyond the demographics (95% CI, log scale)")
    ax.set_ylim(-0.6, n - 0.4)
    clean(ax)


def draw_null_distribution(ax, boot, naive_p, cohort):
    T = np.asarray(boot["T"], float)
    t_obs, one = float(boot["T_obs"]), bool(boot["one_sided"])
    lo = min(T.min(), t_obs, -3.5 if one else 0.0)
    hi = max(T.max(), t_obs) * (1.08 if max(T.max(), t_obs) > 0 else 0.9) + 0.3
    bins = np.linspace(lo, hi, 28)
    heights, _, _ = ax.hist(T, bins=bins, density=True, color="0.72", edgecolor="white", lw=0.4,
                            label=f"null replicates (n = {boot['replicates']})")
    xs = np.linspace(lo, hi, 400)
    if one:
        ax.plot(xs, stats.norm.pdf(xs), color="0.25", lw=0.8, ls="--", label="N(0, 1): naive test")
        xlabel = "Signed root of the likelihood-ratio statistic"
    else:
        xs = xs[xs > 0.03]
        ax.plot(xs, stats.chi2.pdf(xs, 1), color="0.25", lw=0.8, ls="--", label="χ² (1 df): naive test")
        xlabel = "Likelihood-ratio statistic, χ² (1 df)"
    ax.axvline(t_obs, color=COHORT[cohort], lw=1.6, label="observed")
    ax.set_xlim(lo, hi)
    top = max(float(np.max(heights)), 0.42 if one else 0.0) or 1.0
    ax.set_ylim(0, top * 2.5)
    p_txt = fmt_p(boot["p"], one)
    stop = (f"stopped after {boot['exceedances']} exceedances" if boot["stopped_early"]
            else f"all {boot['replicates']} replicates run")
    txt = (f"observed = {t_obs:.2f}\n{boot['exceedances']} of {boot['replicates']} null replicates ≥ observed\n"
           f"null bootstrap {p_txt}\n({stop})\nnaive {fmt_p(naive_p, one)}")
    right_side = t_obs < (lo + hi) / 2
    ax.text(0.98 if right_side else 0.02, 0.97, txt, transform=ax.transAxes, va="top",
            ha="right" if right_side else "left", fontsize=6.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", lw=0.5))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    leg = ax.legend(fontsize=6.0, loc="upper right" if right_side else "upper left", frameon=True,
                    bbox_to_anchor=(0.98 if right_side else 0.02, 0.62))
    leg.get_frame().set_edgecolor("none")
    leg.get_frame().set_alpha(0.9)
    clean(ax)


def fig_rq1(inc, tr, boot_bl, boot_gr):
    fig = plt.figure(figsize=(COL2, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.55, wspace=0.32,
                          left=0.27, right=0.80, top=0.95, bottom=0.08)
    ax = fig.add_subplot(gs[0, :])
    draw_or_forest(ax, rq1_rows(inc, tr))
    panel_label(ax, "a", x=-0.36)
    ax.set_title("Does the EEG score add information beyond the demographics?", fontsize=8, loc="left")
    I = inc["INC"]
    for j, (boot, naive, cohort, title) in enumerate((
            (boot_bl, I["BrainLat"]["p"], "BrainLat", "BrainLat: null distribution (two-sided)"),
            (boot_gr, I["Greek"]["p_one_sided"], "Greek", "Greek cohort: null distribution (one-sided)"))):
        axn = fig.add_axes([0.08 + 0.47 * j, 0.08, 0.40, 0.33])
        draw_null_distribution(axn, boot, naive, cohort)
        axn.set_title(title, fontsize=7.5, loc="left")
        panel_label(axn, "bc"[j], x=-0.2)
    return fig


# ---- Fig. 5 — RQ2 ---------------------------------------------------------------------------------------------------
DESIGN_SHORT = {"Full cohort": "Whole cohort (reference)",
                "Education-matched 1:1 (±2 y)": "1:1 pairs matched on education (±2 y)",
                "Comparable-pair concordance (education ±2 y)": "All pairs with education ±2 y",
                "Overlap-weighted (age, sex, education)": "Overlap weights (age, sex, education)",
                "Covariate-adjusted AUC (age, sex, education)": "Covariate-adjusted AUC (check)",
                "Overlap-weighted (age, sex)": "Overlap weights (age, sex)",
                "Comparable-pair concordance (same sex, age ±2 y)": "All pairs of same sex and age ±2 y",
                "Covariate-adjusted AUC (age, sex)": "Covariate-adjusted AUC (check)"}


def draw_design_auc(ax, tab):
    tab = tab.reset_index(drop=True)
    ys, labels, y = [], [], 0.0
    prev = None
    for _, r in tab.iterrows():
        if prev is not None and r["cohort"] != prev:
            y += 0.8
        ys.append(y)
        labels.append(DESIGN_SHORT.get(r["analysis"], r["analysis"]))
        prev = r["cohort"]
        y += 1
    ys = np.max(ys) - np.asarray(ys)
    for yy, (_, r) in zip(ys, tab.iterrows()):
        c = COHORT["BrainLat" if r["cohort"] == "BrainLat" else "Greek"]
        ref = r["analysis"] == "Full cohort"
        check = "Covariate-adjusted" in r["analysis"]
        ax.errorbar(r["AUC"], yy, xerr=[[r["AUC"] - r["lo"]], [r["hi"] - r["AUC"]]], fmt="o",
                    color="0.55" if ref else c, mfc="white" if (check or ref) else ("0.55" if ref else c),
                    ms=4.5, capsize=2, lw=0.9)
        ax.text(1.01, yy, str(r["n"]), transform=ax.get_yaxis_transform(), va="center", fontsize=6.3, color="0.3")
    ax.axvline(0.5, color="0.5", lw=0.7, ls="--")
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=6.8)
    ax.set_xlim(max(0.0, min(0.35, float(np.nanmin(tab["lo"])) - 0.05)), 1.0)
    ax.set_xlabel("AUC or concordance of the EEG score (95% CI)")
    cohorts = list(dict.fromkeys(tab["cohort"]))
    for coh in cohorts:
        yy = ys[(tab["cohort"] == coh).to_numpy()]
        ax.text(-0.02, yy.max() + 0.55, "BrainLat" if coh == "BrainLat" else "Greek cohort (ds004504)",
                transform=ax.get_yaxis_transform(), ha="right", fontsize=7, fontweight="bold",
                color=COHORT["BrainLat" if coh == "BrainLat" else "Greek"])
    ax.set_ylim(-0.6, ys.max() + 1.0)
    clean(ax)


def draw_love(ax, before, afters, variables, legend=True):
    """before: Series SMD by variable; afters: list of (label, Series, marker)."""
    ax.axvspan(-0.1, 0.1, color="0.92", lw=0)
    ax.axvline(0, color="0.5", lw=0.6)
    yy = np.arange(len(variables))[::-1]
    ax.scatter([before[v] for v in variables], yy, marker="o", facecolor="white", edgecolor="0.25", s=22,
               zorder=3, label="before adjustment")
    for label, s, mk in afters:
        ax.scatter([s[v] for v in variables], yy, marker=mk, color="black", s=22, zorder=4, label=label)
    for v, y in zip(variables, yy):
        xs = [before[v]] + [s[v] for _, s, _ in afters]
        ax.plot([min(xs), max(xs)], [y, y], color="0.7", lw=0.6, zorder=2)
    ax.set_yticks(yy)
    ax.set_yticklabels(variables, fontsize=6.8)
    allv = [abs(before[v]) for v in variables] + [abs(s[v]) for _, s, _ in afters for v in variables]
    lim = max(1.0, float(np.nanmax(allv)) * 1.12)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-0.7, len(variables) - 0.3)
    ax.set_xlabel("Standardised mean difference, AD − HC", fontsize=7)
    if legend:
        ax.legend(frameon=False, fontsize=6, loc="upper left", bbox_to_anchor=(-0.05, -0.3), ncol=1)
    clean(ax)


def _smd_series(tab, col):
    names = {"Education (years)": "Education", "Age (years)": "Age", "Female (proportion)": "Female sex"}
    return pd.Series({names.get(v, v): float(x) for v, x in zip(tab["variable"], tab[col])})


def fig_rq2(t2b, bal_bl_match, bal_bl_ow, bal_gr_ow):
    fig = plt.figure(figsize=(COL2, 4.3))
    ax = fig.add_axes([0.30, 0.13, 0.30, 0.80])
    draw_design_auc(ax, t2b)
    panel_label(ax, "a", x=-0.85, y=1.0)
    ax.set_title("Discrimination when the groups are made comparable", fontsize=7.5, loc="left", x=-0.75)
    ax_b = fig.add_axes([0.79, 0.66, 0.19, 0.26])
    before = _smd_series(bal_bl_ow, "SMD_raw")
    draw_love(ax_b, before, [("1:1 matching on education", _smd_series(bal_bl_match, "SMD_matched"), "s"),
                             ("overlap weights", _smd_series(bal_bl_ow, "SMD_weighted"), "D")],
              ["Education", "Age", "Female sex"])
    ax_b.set_title("BrainLat: covariate balance", fontsize=7.5, loc="left")
    panel_label(ax_b, "b", x=-0.55)
    ax_c = fig.add_axes([0.79, 0.13, 0.19, 0.17])
    draw_love(ax_c, _smd_series(bal_gr_ow, "SMD_raw"), [("overlap weights", _smd_series(bal_gr_ow, "SMD_weighted"), "D")],
              ["Age", "Female sex"], legend=False)
    ax_c.set_title("Greek cohort: covariate balance", fontsize=7.5, loc="left")
    panel_label(ax_c, "c", x=-0.55)
    return fig


# ---- Fig. 6 — RQ3 ---------------------------------------------------------------------------------------------------
def draw_calibration(ax, subj, res, versions=("frozen", "intercept_and_slope"), n_bins=5):
    y = subj["y"].to_numpy(int)
    ax.plot([0, 1], [0, 1], color="0.6", lw=0.7, ls=":")
    handles = []
    names = {"frozen": "frozen model", "intercept_and_slope": "recalibrated (post hoc)",
             "intercept_only": "recalibrated, intercept only (post hoc)"}
    for k, v in enumerate(versions):
        p = subj[f"p_{v}"].to_numpy(float)
        col = VERSION[v]
        b = calibration_bins(y, p, n_bins)
        yerr = [np.clip(b["observed"] - b["lo"], 0, None), np.clip(b["hi"] - b["observed"], 0, None)]
        ax.errorbar(b["mean_predicted"], b["observed"], yerr=yerr,
                    fmt="o" if v == "frozen" else "s", color=col, ms=3.8, capsize=1.5, lw=0.8, zorder=3)
        grid = np.linspace(max(0.01, p.min()), min(0.99, p.max()), 200)
        ax.plot(grid, logistic_calibration_curve(y, p, grid), color=col, lw=1.2, zorder=2)
        m = res["versions"][v]
        handles.append(Line2D([], [], color=col, marker="o" if v == "frozen" else "s", ms=3.8, lw=1.2,
                              label=f"{names[v]}: slope {m['calibration_slope']:.2f}, "
                                    f"CITL {m['citl']:.2f}, Brier {m['brier']:.3f}"))
    ax.plot(subj.loc[subj.y == 1, "p_frozen"], np.full((subj.y == 1).sum(), 1.015), "|", color=GROUP["AD"], ms=4,
            clip_on=False)
    ax.plot(subj.loc[subj.y == 0, "p_frozen"], np.full((subj.y == 0).sum(), -0.015), "|", color=GROUP["HC"], ms=4,
            clip_on=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Predicted P(AD)")
    ax.set_ylabel("Observed proportion with AD")
    ax.legend(handles=handles, frameon=False, fontsize=6.0, loc="upper left", bbox_to_anchor=(-0.04, -0.17),
              handlelength=1.6)
    clean(ax)


def draw_probability_strips(ax, subj, res, threshold, versions=("frozen", "intercept_and_slope")):
    rng = np.random.default_rng(0)
    centres = {}
    for k, v in enumerate(versions):
        for j, (g, lab) in enumerate(((0, "HC"), (1, "AD"))):
            x0 = k * 2.6 + j
            centres[(v, lab)] = x0
            p = subj.loc[subj.y == g, f"p_{v}"].to_numpy(float)
            ax.scatter(x0 + rng.uniform(-0.22, 0.22, len(p)), p, s=7, color=GROUP[lab], alpha=0.75, lw=0)
            ax.plot([x0 - 0.3, x0 + 0.3], [np.median(p)] * 2, color="black", lw=1.2)
        m = res["versions"][v]["metrics"]
        ax.text(k * 2.6 + 0.5, 1.04, f"sens {m['sens']:.2f} · spec {m['spec']:.2f}", ha="center", fontsize=6.3,
                transform=ax.get_xaxis_transform())
    ax.plot([-0.6, 3.9], [threshold, threshold], color="0.35", lw=0.8, ls="--")
    ax.text(3.98, threshold, f"{threshold:g}", va="center", fontsize=6.3, color="0.35")
    ax.set_xticks([centres[k] for k in centres])
    ax.set_xticklabels([lab for (_, lab) in centres], fontsize=6.8)
    for k, v in enumerate(versions):
        ax.text(k * 2.6 + 0.5, -0.2, "frozen model" if v == "frozen" else "recalibrated (post hoc)",
                ha="center", va="top", fontsize=6.8, transform=ax.get_xaxis_transform())
    ax.set_xlim(-0.6, 4.4)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Predicted P(AD)")
    clean(ax)


def fig_rq3(subj_a, res_a, subj_b, res_b, threshold, auc_a, auc_b):
    fig = plt.figure(figsize=(COL2, 6.3))
    for j, (subj, res, auc, title) in enumerate(((subj_a, res_a, auc_a, "BrainLat model → Greek cohort"),
                                                 (subj_b, res_b, auc_b, "Greek model → BrainLat"))):
        ax = fig.add_axes([0.09 + 0.49 * j, 0.50, 0.40, 0.42])
        draw_calibration(ax, subj, res)
        ax.set_title(f"{title}\nAUC {auc[0]:.2f} [{auc[1]:.2f}, {auc[2]:.2f}]", fontsize=7.5, loc="left", pad=8)
        panel_label(ax, "ab"[j], x=-0.2, y=1.1)
        ax2 = fig.add_axes([0.09 + 0.49 * j, 0.08, 0.40, 0.22])
        draw_probability_strips(ax2, subj, res, threshold)
        panel_label(ax2, "cd"[j], x=-0.2, y=1.08)
    return fig


# ---- Fig. 7 — per-feature consistency ------------------------------------------------------------------------------
def feature_rows(keys, features):
    """Per-feature OR/SD and BH q from results/key_results.json (keys features.<feature>.<cohort>.*)."""
    def k(name):
        v = keys.get(name)
        return np.nan if v is None or v.get("value") is None else float(v["value"])
    rows = []
    for f in features:
        for coh, key in (("BrainLat", "brainlat"), ("Greek", "greek")):
            rows.append({"feature": f, "cohort": coh, "OR": k(f"features.{f}.{key}.or.value"),
                         "lo": k(f"features.{f}.{key}.or.lo"), "hi": k(f"features.{f}.{key}.or.hi"),
                         "q": k(f"features.{f}.{key}.q")})
    return pd.DataFrame(rows)


def fig_features(rows, spectral):
    feats = list(dict.fromkeys(rows["feature"]))
    fig, ax = plt.subplots(figsize=(COL2 * 0.78, 4.0))
    ypos, y = {}, 0.0
    for i, f in enumerate(feats):
        if i > 0 and (f in spectral) != (feats[i - 1] in spectral):
            y += 0.7
        ypos[f] = y
        y += 1
    top = max(ypos.values())
    vals = []
    for _, r in rows.iterrows():
        yy = top - ypos[r["feature"]] + (0.17 if r["cohort"] == "BrainLat" else -0.17)
        c = COHORT[r["cohort"]]
        filled = np.isfinite(r["q"]) and r["q"] < 0.05
        ax.errorbar(r["OR"], yy, xerr=[[r["OR"] - r["lo"]], [r["hi"] - r["OR"]]], fmt="o", color=c,
                    mfc=c if filled else "white", mec=c, ms=4.2, capsize=1.5, lw=0.8)
        vals += [v for v in (r["OR"], r["lo"], r["hi"]) if np.isfinite(v) and v > 0]
    ax.axvline(1, color="0.5", lw=0.7, ls="--")
    ax.set_yticks([top - ypos[f] for f in feats])
    ax.set_yticklabels([FEATURE_LABELS.get(f, f) for f in feats], fontsize=7)
    ax.set_xscale("log")
    vmin, vmax = min(vals) / 1.3, max(vals) * 1.3
    ax.set_xlim(vmin, vmax)
    log2_ticks(ax, vmin, vmax)
    ax.set_xlabel("Odds ratio of AD per 1 SD of the feature, beyond the demographics (95% CI, log scale)")
    n_sp = sum(f in spectral for f in feats)
    ax.text(1.01, top - (n_sp - 1) / 2, "spectral", transform=ax.get_yaxis_transform(), rotation=90,
            va="center", fontsize=6.8, color="0.35")
    ax.text(1.01, top - ypos[feats[-1]] + (len(feats) - n_sp - 1) / 2, "connectivity",
            transform=ax.get_yaxis_transform(), rotation=90, va="center", fontsize=6.8, color="0.35")
    same = sum(((g.set_index("cohort")["OR"] - 1).prod() > 0) for _, g in rows.groupby("feature"))
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=COHORT["BrainLat"], label="BrainLat"),
                       Line2D([], [], marker="o", ls="", color=COHORT["Greek"], label="Greek cohort"),
                       Line2D([], [], marker="o", ls="", color="0.3", mfc="white", label="open: q ≥ 0.05"),
                       Line2D([], [], marker="o", ls="", color="0.3", label="filled: q < 0.05 (BH, within cohort)")],
              frameon=False, fontsize=6.3, loc="lower right")
    ax.set_title(f"Same direction in both cohorts: {same} of {len(feats)} features", fontsize=7.5, loc="left")
    ax.set_ylim(-0.7, top + 0.7)
    clean(ax)
    fig.tight_layout()
    return fig


# ---- Fig. S3 — E-value ---------------------------------------------------------------------------------------------
def fig_evalue(or_pt, or_lo, e_pt, e_ci, education_or=None):
    rr_pt, rr_lo = np.sqrt(or_pt), np.sqrt(or_lo)
    hi = max(12.0, e_pt * 2.2)
    fig, ax = plt.subplots(figsize=(COL1 * 1.15, 3.2))
    for rr, e, col, lab in ((rr_pt, e_pt, "#C0392B", "explains away the estimate"),
                            (rr_lo, e_ci, "#E69F00", "moves the 95% CI to include 1")):
        a = np.geomspace(rr * 1.0005, hi, 400)
        b = rr * (a - 1) / (a - rr)
        ok = (b > 1) & (b <= hi)
        ax.plot(a[ok], b[ok], color=col, lw=1.3, label=f"confounder that {lab}")
        ax.plot([e], [e], "o", color=col, ms=5)
        ax.annotate(f"E-value {e:.2f}", (e, e), xytext=(6, 4), textcoords="offset points", fontsize=6.8, color=col)
    if education_or is not None and np.isfinite(education_or) and education_or > 0:
        rr_edu = np.sqrt(education_or if education_or >= 1 else 1 / education_or)
        ax.axhline(rr_edu, color="0.45", lw=0.8, ls="--")
        ax.text(hi, rr_edu, f"education–AD association in BrainLat\n(per 1 SD; RR ≈ {rr_edu:.1f})", ha="right",
                va="bottom", fontsize=6.2, color="0.35")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, hi)
    ax.set_ylim(1, hi)
    for axis in (ax.xaxis, ax.yaxis):
        ticks = [t for t in (1, 2, 3, 5, 10, 20, 50) if t <= hi]
        axis.set_ticks(ticks)
        axis.set_ticklabels([str(t) for t in ticks])
        axis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Confounder–EEG score association (risk ratio)")
    ax.set_ylabel("Confounder–AD association (risk ratio)")
    ax.legend(frameon=False, fontsize=6.2, loc="lower right", bbox_to_anchor=(1.0, 0.08))
    clean(ax)
    fig.tight_layout()
    return fig


# ---- Fig. S4 — domain shift ----------------------------------------------------------------------------------------
def subject_means(df_ep, features):
    return df_ep.groupby("subject_id").agg({**{f: "mean" for f in features}, "y": "first"})


def smd(a, b):
    return float((b.mean() - a.mean()) / np.sqrt((a.var() + b.var()) / 2))


def domain_shift_rows(df_bl, df_gr, features):
    sb, sg = subject_means(df_bl, features), subject_means(df_gr, features)
    rows = []
    for f in features:
        rows.append({"feature": f,
                     "HC: Greek − BrainLat": smd(sb.loc[sb.y == 0, f], sg.loc[sg.y == 0, f]),
                     "AD − HC within BrainLat": smd(sb.loc[sb.y == 0, f], sb.loc[sb.y == 1, f]),
                     "AD − HC within the Greek cohort": smd(sg.loc[sg.y == 0, f], sg.loc[sg.y == 1, f])})
    return pd.DataFrame(rows)


def fig_domain_shift(rows):
    fig, ax = plt.subplots(figsize=(COL2 * 0.72, 3.8))
    feats = list(rows["feature"])
    yy = np.arange(len(feats))[::-1]
    style = {"HC: Greek − BrainLat": ("D", "black", "controls: Greek − BrainLat (domain shift)"),
             "AD − HC within BrainLat": ("o", COHORT["BrainLat"], "AD − HC within BrainLat"),
             "AD − HC within the Greek cohort": ("s", COHORT["Greek"], "AD − HC within the Greek cohort")}
    for col, (mk, c, lab) in style.items():
        ax.scatter(rows[col], yy, marker=mk, color=c, s=20 if mk != "D" else 24, zorder=3, label=lab)
    for y in yy:
        ax.axhline(y, color="0.93", lw=0.6, zorder=1)
    ax.axvline(0, color="0.5", lw=0.6)
    for x in (-0.8, 0.8):
        ax.axvline(x, color="0.6", lw=0.6, ls="--")
    ax.set_yticks(yy)
    ax.set_yticklabels([FEATURE_LABELS.get(f, f) for f in feats], fontsize=7)
    ax.set_xlabel("Standardised mean difference (subject means)")
    ax.legend(frameon=False, fontsize=6.3, loc="upper center", bbox_to_anchor=(0.45, -0.14), ncol=2)
    clean(ax)
    fig.tight_layout()
    return fig


# ---- Fig. S1 — angular error of the BrainLat 10-20 correspondence (amendment 2, part C) -----------------------------
def _project(xyz):
    """Azimuthal equidistant projection (vertex at the centre, nose up), as in the study-design figure."""
    U = np.asarray(xyz, float)
    theta = np.arccos(np.clip(U[:, 2], -1, 1))
    phi = np.arctan2(U[:, 1], U[:, 0])
    rho = theta / (np.pi / 2)
    return np.column_stack([rho * np.cos(phi), rho * np.sin(phi)])


def fig_montage_error(electrodes, targets, spacing):
    """Head seen from above: the BrainLat electrodes (grey), the 19 standard positions (open circles) and the
    electrode used for each (filled, coloured by the angle; line to its position). Markers only, no interpolation:
    the error exists only at the 19 positions."""
    from matplotlib.colors import Normalize
    from matplotlib.patches import Circle, Polygon

    fig = plt.figure(figsize=(COL1 * 1.25, COL1 * 1.18))
    ax = fig.add_axes([0.02, 0.03, 0.80, 0.94])
    cax = fig.add_axes([0.84, 0.22, 0.035, 0.56])
    xy_e = _project(electrodes[["x", "y", "z"]].to_numpy())
    xy_t = _project(targets[["x", "y", "z"]].to_numpy())
    r_head = max(1.0, float(np.max(np.linalg.norm(xy_t, axis=1))) * 1.12)
    keep = np.linalg.norm(xy_e, axis=1) <= r_head * 1.02
    ax.add_patch(Circle((0, 0), r_head, fc="none", ec="black", lw=0.9, zorder=1))
    ax.add_patch(Polygon([(-0.09 * r_head, 0.99 * r_head), (0, 1.11 * r_head), (0.09 * r_head, 0.99 * r_head)],
                         closed=True, fc="none", ec="black", lw=0.9))
    for sgn in (-1, 1):
        ax.add_patch(plt.matplotlib.patches.Ellipse((sgn * 1.02 * r_head, 0), 0.08 * r_head, 0.22 * r_head,
                                                    fc="none", ec="black", lw=0.9))
    ax.scatter(xy_e[keep, 0], xy_e[keep, 1], s=4, color="0.72", lw=0, zorder=2)
    norm = Normalize(0, max(5.0, float(np.ceil(targets["angle_deg"].max()))))
    cmap = plt.get_cmap("viridis")
    by_name = dict(zip(electrodes["electrode"], xy_e))
    s1 = dict(zip(electrodes["assigned_to"], electrodes["electrode"]))
    for (c, ang), (x, y) in zip(targets[["channel_1020", "angle_deg"]].to_numpy(), xy_t):
        e = by_name.get(s1.get(c))
        if e is not None:
            ax.plot([x, e[0]], [y, e[1]], color="0.25", lw=0.6, zorder=3)
            ax.scatter([e[0]], [e[1]], s=26, color=cmap(norm(ang)), ec="black", lw=0.4, zorder=5)
        ax.scatter([x], [y], s=46, facecolor="none", ec="black", lw=0.7, zorder=4)
        ax.text(x, y + 0.075 * r_head, c, ha="center", va="bottom", fontsize=6.2, zorder=6)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Angle to the electrode used (°)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    ax.set_xlim(-1.2 * r_head, 1.2 * r_head)
    ax.set_ylim(-1.2 * r_head, 1.2 * r_head)
    ax.set_aspect("equal")
    ax.axis("off")
    txt = (f"mean {targets['angle_deg'].mean():.1f}°, maximum {targets['angle_deg'].max():.1f}°\n"
           f"nearest 10-20 neighbours: median {spacing['median_neighbour_angle_deg']:.0f}° apart")
    ax.text(0, -1.17 * r_head, txt, ha="center", va="bottom", fontsize=6.5, color="0.2")
    handles = [Line2D([], [], marker="o", ls="", mfc="none", mec="black", ms=6, label="10-20 position"),
               Line2D([], [], marker="o", ls="", color=cmap(0.6), mec="black", ms=5, label="BioSemi electrode used"),
               Line2D([], [], marker="o", ls="", color="0.72", ms=3, label="other BioSemi electrodes")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.02, 1.02), frameon=False, fontsize=6,
              handletextpad=0.3)
    return fig


# ---- Fig. S2 — group-average spectra (amendment 2, part B) ----------------------------------------------------------
BANDS_PLOT = (("δ", 0.5, 4), ("θ", 4, 8), ("α", 8, 13), ("β", 13, 30), ("γ", 30, 45))


def _band_shading(ax, y_text):
    for k, (lab, lo, hi) in enumerate(BANDS_PLOT):
        if k % 2:
            ax.axvspan(lo, hi, color="0.95", zorder=0, lw=0)
        ax.text(np.sqrt(lo * hi), y_text, lab, ha="center", va="top", fontsize=7, color="0.35",
                transform=ax.get_xaxis_transform())


def fig_spectra(groups, cutoff_hz=40.0):
    """(a, b) Mean relative spectrum of AD and controls in each cohort with 95% bands; (c) the controls of the two
    cohorts together (domain shift). Log-log axes; dashed line: the 40 Hz harmonisation cutoff."""
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.45), sharey=True)
    panels = [("a", "BrainLat", [("BrainLat", "AD"), ("BrainLat", "HC")]),
              ("b", "Greek cohort (ds004504)", [("Greek", "AD"), ("Greek", "HC")]),
              ("c", "Controls of the two cohorts", [("BrainLat", "HC"), ("Greek", "HC")])]
    style = {("BrainLat", "AD"): (GROUP["AD"], "-"), ("BrainLat", "HC"): (GROUP["HC"], "-"),
             ("Greek", "AD"): (GROUP["AD"], "-"), ("Greek", "HC"): (GROUP["HC"], "-")}
    for ax, (letter, title, curves) in zip(axes, panels):
        _band_shading(ax, 0.995)
        for cohort, group in curves:
            d = groups[(groups.cohort == cohort) & (groups.group == group)].sort_values("frequency_hz")
            if d.empty:
                continue
            color, ls = style[(cohort, group)]
            if letter == "c":
                color = COHORT[cohort]
            lab = (f"{group} (n = {int(d['n'].iloc[0])})" if letter != "c"
                   else f"{'BrainLat' if cohort == 'BrainLat' else 'Greek'} controls (n = {int(d['n'].iloc[0])})")
            ax.fill_between(d["frequency_hz"], d["lo"], d["hi"], color=color, alpha=0.2, lw=0)
            ax.plot(d["frequency_hz"], d["mean"], color=color, ls=ls, lw=1.1, label=lab)
        ax.axvline(cutoff_hz, color="0.4", lw=0.7, ls="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.5, 45)
        ax.set_xticks([1, 2, 4, 8, 13, 30, 45])
        ax.set_xticklabels(["1", "2", "4", "8", "13", "30", "45"])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("Frequency (Hz)")
        ax.set_title(title, fontsize=8, pad=12)
        ax.legend(frameon=False, fontsize=6.2, loc="lower left")
        panel_label(ax, letter, x=-0.16, y=1.03)
        clean(ax)
    axes[0].set_ylabel("Relative power density (1/Hz)")
    fig.tight_layout(w_pad=0.6)
    return fig


# ---- Fig. S5 — hyperparameter sensitivity (amendment 2, part A) -----------------------------------------------------
OR_CAP = 1000.0


def fig_hyperparameters(results):
    """One row per configuration (frozen first, highlighted): (a) AUC within each cohort, (b) OR per SD beyond the
    demographics (log scale; intervals beyond 1000 are cut, with an arrow), (c) AUC of the frozen model in the other
    cohort. Points with 95% CI."""
    labels = [r["label"] for r in results]
    yy = np.arange(len(results))[::-1].astype(float)
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 0.24 * len(results) + 1.25), sharey=True)
    off = {"BrainLat": 0.14, "Greek": -0.14, "RQ3a": 0.14, "RQ3b": -0.14}
    spec = [("a", "Within-cohort AUC", (("BrainLat", "BrainLat", COHORT["BrainLat"], "o"),
                                         ("Greek", "Greek", COHORT["Greek"], "s"))),
            ("b", "Incremental value: OR/SD", (("BrainLat", "BrainLat", COHORT["BrainLat"], "o"),
                                                          ("Greek", "Greek", COHORT["Greek"], "s"))),
            ("c", "Transported AUC", (("RQ3a", "BrainLat → Greek", COHORT["BrainLat"], "o"),
                                      ("RQ3b", "Greek → BrainLat", COHORT["Greek"], "s")))]
    for ax, (letter, title, series) in zip(axes, spec):
        for y, r in zip(yy, results):
            if r["frozen"]:
                ax.axhspan(y - 0.45, y + 0.45, color="#FFF3D6", zorder=0, lw=0)
        for key, lab, color, mk in series:
            if letter == "b":
                v = np.array([[r[key]["OR"], r[key]["OR_lo"], r[key]["OR_hi"]] for r in results], float)
            else:
                v = np.array([r[key]["auc"] for r in results], float)
            ys = yy + off[key]
            if letter == "b":                                   # very wide profile CIs are cut at OR_CAP (arrow)
                v = np.where(np.isfinite(v), v, np.inf)
                cut = v[:, 2] > OR_CAP
                v = np.clip(v, 1 / OR_CAP, OR_CAP)
                ax.scatter(v[cut, 2], ys[cut], marker=">", color=color, s=12, zorder=3)
            ax.hlines(ys, v[:, 1], v[:, 2], color=color, lw=1.0)
            ax.scatter(v[:, 0], ys, color=color, marker=mk, s=14, zorder=3, label=lab)
        if letter == "b":
            ax.set_xscale("log")
            ax.axvline(1, color="0.5", lw=0.6)
            vals = np.array([[r[k]["OR_lo"], r[k]["OR_hi"]] for r in results for k in ("BrainLat", "Greek")], float)
            vals = np.clip(vals[np.isfinite(vals) & (vals > 0)], 1 / OR_CAP, OR_CAP)
            lo_, hi_ = max(vals.min() * 0.8, 0.05), min(vals.max() * 1.25, OR_CAP * 1.3)
            ax.set_xlim(lo_, hi_)
            log2_ticks(ax, lo_, hi_, max_ticks=7)
        else:
            ax.axvline(0.5, color="0.5", lw=0.6, ls="--")
            ax.set_xlim(0.3, 1.0)
        ax.set_title(title, fontsize=7.6)
        ax.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.06),
                  ncol=2, handletextpad=0.2, columnspacing=0.8)
        panel_label(ax, letter, x=-0.04 if letter != "a" else -0.62, y=1.005)
        clean(ax)
    axes[0].set_yticks(yy)
    axes[0].set_yticklabels(labels, fontsize=6.6)
    axes[0].set_ylim(yy.min() - 0.6, yy.max() + 0.6)
    fig.tight_layout(w_pad=0.5)
    return fig
