"""Figures: style, ROC curves (Fig. 1), incremental odds ratios (Fig. 2), confounding-free discrimination
(Fig. 3) and the participant flow diagram."""
from __future__ import annotations

import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter
from sklearn.metrics import roc_auc_score, roc_curve

COL1, COL2 = 3.31, 6.85   # Springer column widths (84 mm and 174 mm), inches
COLORS = {"EEG": "#D55E00", "EEG_other": "#CC79A7", "Demo": "#0072B2", "Demo_EEG": "#009E73"}


def apply_style():
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 0.6, "lines.linewidth": 1.0,
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })
    import logging
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


def clean_axes(ax):
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)


def _curve(ax, y, p, color, label, style="-"):
    fpr, tpr, _ = roc_curve(y, p)
    ax.plot(fpr, tpr, style, color=color, label=f"{label}: {roc_auc_score(y, p):.2f}")


def roc_figure(panels):
    """panels: list of dicts with title, eeg (y, p), other (y, p), demo (label, y, p), combined (y, p)."""
    fig, axes = plt.subplots(1, len(panels), figsize=(COL2, 3.0))
    for ax, pn in zip(np.atleast_1d(axes), panels):
        _curve(ax, *pn["eeg"], COLORS["EEG"], "EEG, within-cohort model")
        _curve(ax, *pn["other"], COLORS["EEG_other"], "EEG, other cohort's model", "--")
        _curve(ax, *pn["demo"][1:], COLORS["Demo"], pn["demo"][0])
        _curve(ax, *pn["combined"], COLORS["Demo_EEG"], f"{pn['demo'][0]} + EEG", ":")
        ax.plot([0, 1], [0, 1], color="0.75", lw=0.6)
        ax.set_title(pn["title"])
        ax.set_xlabel("1 − specificity")
        ax.set_ylabel("Sensitivity")
        ax.legend(frameon=False, loc="lower right", title="AUC", title_fontsize=6.5, fontsize=6.5, handlelength=1.8)
        ax.set_aspect("equal")
        clean_axes(ax)
    plt.tight_layout()
    return fig


def or_forest_figure(rows, n_main=5):
    """rows: list of (label, OR, lo, hi); the first n_main rows are the main analyses (row 3 = pooled)."""
    fig, ax = plt.subplots(figsize=(COL2, 0.22 * len(rows) + 0.7))
    for i, (_, o, lo, hi) in enumerate(rows):
        yy = len(rows) - 1 - i
        main = i < n_main
        ax.errorbar(o, yy, xerr=[[o - lo if np.isfinite(lo) else 0], [hi - o if np.isfinite(hi) else 0]],
                    fmt="D" if i == 2 else "s" if i in (3, 4) else "o", color="black" if main else "0.55",
                    ms=4 if main else 3, capsize=2, lw=0.8)
    ax.axhline(len(rows) - 3.5, color="0.85", lw=0.6)
    ax.axhline(len(rows) - 5.5, color="0.85", lw=0.6)
    ax.axvline(1, color="0.6", lw=0.6, ls="--")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1])
    ax.set_xscale("log")
    vals = np.array([v for r in rows for v in r[1:] if np.isfinite(v) and v > 0])
    k_lo, k_hi = int(np.floor(np.log2(vals.min() / 1.5))), int(np.ceil(np.log2(vals.max() * 1.5)))
    step = max(1, int(np.ceil((k_hi - k_lo + 1) / 12)))          # at most ~12 powers-of-two ticks
    ticks = [2.0 ** k for k in range(k_lo, k_hi + 1, step)] + ([1.0] if step > 1 else [])
    ticks = sorted(t for t in set(ticks) if vals.min() / 1.5 <= t <= vals.max() * 1.5)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Odds ratio per 1 SD of the EEG score, beyond demographics (95% CI)")
    clean_axes(ax)
    plt.tight_layout()
    return fig


def confounding_figure(tab, short_names):
    """tab: rows with cohort, analysis, AUC, lo, hi (Table 2b)."""
    tc = tab.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(COL1, 0.27 * len(tc) + 0.75))
    for i, row in tc.iterrows():
        yy = len(tc) - 1 - i
        color = COLORS["EEG"] if row["cohort"] == "BrainLat" else COLORS["Demo"]
        ax.errorbar(row["AUC"], yy, xerr=[[row["AUC"] - row["lo"]], [row["hi"] - row["AUC"]]], fmt="o", color=color,
                    ms=3.5, capsize=2, lw=0.8)
    ax.axvline(0.5, color="0.6", lw=0.6, ls="--")
    ax.set_yticks(range(len(tc)))
    ax.set_yticklabels([short_names.get(a, a) for a in tc["analysis"]][::-1])
    ax.set_xlabel("AUC or concordance (95% CI)")
    ax.set_xlim(max(0.0, min(0.45, np.nanmin(tc["lo"]) - 0.05)), 1.0)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=COLORS["EEG"], ms=3.5, label="BrainLat"),
                       Line2D([], [], marker="o", ls="", color=COLORS["Demo"], ms=3.5, label="Greek")],
              frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=6.5)
    clean_axes(ax)
    plt.tight_layout()
    return fig


# ---- participant flow ---------------------------------------------------------------------------------------
def _box(ax, x, y, txt, grey=False, width=38):
    ax.text(x, y, "\n".join(textwrap.fill(t, width) for t in txt.split("\n")), ha="center", va="center",
            fontsize=7.2, bbox=dict(boxstyle="round,pad=0.45", fc="#f2f2f2" if grey else "white",
                                    ec="0.45" if grey else "black", lw=0.8))


def _flow_panel(ax, title, rows):
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=9.5, fontweight="bold")
    ys = np.linspace(0.90, 0.07, len(rows))
    for i, (y, r) in enumerate(zip(ys, rows.itertuples())):
        ftd = getattr(r, "FTD", np.nan)
        extra = f", FTD {int(ftd)}" if pd.notna(ftd) else ""
        _box(ax, 0.33, y, f"{r.step}\nn = {r.n} (AD {r.AD}{extra}, HC {r.HC})")
        if i < len(rows) - 1:
            ax.annotate("", xy=(0.33, ys[i + 1] + 0.055), xytext=(0.33, y - 0.055),
                        arrowprops=dict(arrowstyle="-|>", lw=0.8, color="black"))
            if str(r.excluded_at_step) not in ("", "0", "nan"):
                ym = (y + ys[i + 1]) / 2
                ax.annotate("", xy=(0.66, ym), xytext=(0.33, ym), arrowprops=dict(arrowstyle="-|>", lw=0.8, color="0.3"))
                _box(ax, 0.83, ym, f"Excluded: n = {r.excluded_at_step}\n{r.reason}", grey=True, width=30)


def flow_figure(flow):
    """flow: table with cohort, step, n, AD, HC, excluded_at_step, reason (and FTD for the Greek first row)."""
    fig, axs = plt.subplots(1, 2, figsize=(11, 7.5))
    _flow_panel(axs[0], "BrainLat (Argentina and Chile)", flow[flow.cohort == "BrainLat"])
    _flow_panel(axs[1], "Greek cohort (ds004504)", flow[flow.cohort == "Greek"])
    fig.tight_layout()
    return fig
