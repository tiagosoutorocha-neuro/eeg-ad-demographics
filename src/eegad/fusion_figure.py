"""Fig. 2 - how the demographics and the EEG were combined: two-stage late fusion (stacking). Schematic, drawn with
matplotlib only (no data): (a) stage 1, the Random Forest on the EEG features alone, leave-one-subject-out, gives one
out-of-fold score per participant; (b) stage 2, the score enters a logistic model next to the demographics (Model 0
against Model 1); (c) what each comparison answers; (d) why the demographics were not put in the Random Forest (early
fusion). Same size and visual language as Fig. 1 (design_figure.py); saved also as SVG with editable text."""
from __future__ import annotations

import textwrap

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

from .article_figures import article_style

INK, SOFT, EDGE, MUTED = "#222222", "#F4F4F4", "#8A8A8A", "#5A6270"
BLUE, BLUE_L = "#0072B2", "#CFE3F1"          # EEG score (RQ1 colour in Fig. 1)
GREY, GREY_L = "#9AA0A6", "#E3E6EA"          # demographics / training participants
ORANGE, ORANGE_L = "#D55E00", "#F9DCC8"      # participant left out (as in Fig. 1)
TREE, STRIKE = "#2E7D32", "#B03A2E"
BASE = 6.3                                   # body text size (pt), as in Fig. 1


def late_fusion_figure():
    """Return the figure (7.1 in wide, like Fig. 1). Nothing is read from outputs/."""
    article_style()
    plt.rcParams.update({"mathtext.default": "regular", "pdf.fonttype": 42})
    W, H = 100.0, 100.0 * 5.7 / 7.1
    fig = plt.figure(figsize=(7.1, 5.7))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    def box(x, y, w, h, fc="white", ec="none", lw=0.0, r=1.0, z=1, ls="-"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                                    zorder=z, linestyle=ls))

    def text(x, y, s, right=None, size=BASE, wrap=None, **kw):   # right: layout limit, for reading only
        kw.setdefault("color", INK)
        kw.setdefault("va", "center")
        if wrap:
            s = "\n".join(textwrap.fill(p, wrap) for p in s.split("\n"))
        t = ax.text(x, y, s, fontsize=size, zorder=6, **kw)
        return t

    def arrow(x0, y0, x1, y1, color=INK, lw=0.9, ms=7, ls="-", z=3):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw,
                                     linestyle=ls, zorder=z, shrinkA=0, shrinkB=0))

    def line(xs, ys, color=INK, lw=0.9, z=3, **kw):
        ax.plot(xs, ys, color=color, lw=lw, zorder=z, solid_capstyle="round", solid_joinstyle="round", **kw)

    def heading(x, y, letter, title):
        text(x, y, letter, size=10, weight="bold")
        text(x + 3.2, y, title, size=8, weight="bold")

    def trees(x, y, h=5.0, color=TREE):
        for dx, s in ((-2.1, 0.85), (0.0, 1.0), (2.1, 0.85)):
            hh = h * s
            ax.add_patch(Polygon([[x + dx - 1.05, y], [x + dx + 1.05, y], [x + dx, y + hh]], closed=True, fc=color,
                                 ec="none", zorder=4))
            ax.add_patch(Rectangle((x + dx - 0.18, y - 0.7), 0.36, 0.7, fc="#6D4C41", ec="none", zorder=4))

    def vdots(x, y, color=MUTED, gap=0.55, r=0.17):
        for k in (-1, 0, 1):
            ax.add_patch(Circle((x, y + k * gap), r, fc=color, ec="none", zorder=5))

    def epoch_block(x, y, w, h, fc, ec, rows=3, cols=6, z=4.0):
        cw, rh = w / cols, h / rows
        ax.add_patch(Rectangle((x, y), w, h, fc="white", ec="none", zorder=z))
        for r in range(rows):
            for c in range(cols):
                ax.add_patch(Rectangle((x + c * cw + 0.12, y + r * rh + 0.12), cw - 0.24, rh - 0.24, fc=fc, ec="none",
                                       zorder=z + 0.1))
        ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=ec, lw=0.7, zorder=z + 0.2))

    # ================================================================================================================
    # a. Stage 1
    AX0, AX1, TY0, TY1 = 1.0, 48.5, 41.0, 74.0
    heading(AX0, 77.2, "a", "Stage 1 · EEG only: one score per participant")
    box(AX0, TY0, AX1 - AX0, TY1 - TY0, fc="white", ec=EDGE, lw=0.9, r=1.4)
    text(AX0 + 1.6, TY1 - 2.2, "Epoch level · Random Forest · leave-one-subject-out (LOSO)", AX1 - 0.5, size=6,
         color=MUTED)

    BX, BW, BH = 6.0, 8.0, 3.6
    # row 1 - training: all other participants -> Random Forest
    R1 = 61.0                                     # bottom of the stack
    for k in (2, 1, 0):                           # back to front
        epoch_block(BX + 0.7 * k, R1 + 0.7 * k, BW, BH, GREY_L, GREY, z=4.0 + 0.5 * (2 - k))
    stack_r, stack_mid = BX + BW + 1.4, R1 + 0.7 + BH / 2
    text(BX + BW / 2 + 0.7, 68.3, "all other participants", size=BASE, ha="center")
    text(BX + BW / 2 + 0.7, 66.9, "11 EEG features × 4-s epochs", size=5.4, color=MUTED, ha="center")
    TX, TYB = 26.0, 61.0                          # trees
    trees(TX, TYB)
    arrow(stack_r + 0.4, stack_mid, TX - 3.6, stack_mid, ms=6)
    text((stack_r + TX - 3.4) / 2, stack_mid + 0.5, "train", size=5.6, color=MUTED, ha="center", va="bottom")
    text(TX + 3.9, TYB + 2.6, "Random Forest,\ntrained without $P_i$", AX1 - 0.4, size=BASE, ha="left", va="center",
         linespacing=1.15)

    # row 2 - the left-out participant -> P(AD) per epoch -> mean
    bbase, hs = 49.4, [0.55, 0.8, 0.62, 0.9, 0.7, 0.48, 0.85, 0.66]
    step, bwid = 0.92, 0.64
    bx0 = TX - (len(hs) * step - (step - bwid)) / 2
    for k, hgt in enumerate(hs):
        ax.add_patch(Rectangle((bx0 + k * step, bbase), bwid, 5.2 * hgt, fc=BLUE, ec="none", zorder=4))
    bx1 = bx0 + len(hs) * step - (step - bwid)
    line([bx0 - 0.3, bx1 + 0.3], [bbase, bbase], color=EDGE, lw=0.6)
    mh = bbase + 5.2 * float(np.mean(hs))
    line([bx0 - 0.3, bx1 + 0.3], [mh, mh], color=INK, lw=0.8, ls=(0, (2, 1.4)))
    arrow(TX, TYB - 1.0, TX, bbase + 5.2 * max(hs) + 0.6, ms=6)
    text(TX + 0.8, (TYB - 1.0 + bbase + 5.2 * max(hs) + 0.6) / 2, "predicts each epoch", size=5.6, color=MUTED,
         ha="left")

    PIY = bbase + 0.6                             # the left-out participant
    epoch_block(BX, PIY, BW, BH, ORANGE_L, ORANGE)
    text(BX - 0.8, PIY + BH / 2, "$P_i$", size=6.8, ha="right", color=ORANGE, weight="bold")
    text(BX + BW / 2, PIY + BH + 1.3, "left-out participant", size=BASE, ha="center", color=ORANGE, weight="bold")
    arrow(BX + BW + 0.4, PIY + BH / 2, bx0 - 0.7, PIY + BH / 2, color=ORANGE, lw=1.0, ms=6)

    SX, SCY = 37.6, bbase + 2.5
    box(SX - 2.4, SCY - 2.3, 4.8, 4.6, fc=BLUE, r=0.6, z=4)
    text(SX, SCY, "$s_i$", size=9, color="white", ha="center", weight="bold")
    arrow(bx1 + 0.6, SCY, SX - 2.6, SCY, ms=6)
    text((bx1 + SX - 2.4) / 2 + 0.2, SCY + 0.5, "mean", size=5.6, color=MUTED, ha="center", va="bottom")
    text(TX, bbase - 0.8, "P(AD) of each\nepoch of $P_i$", size=BASE, ha="center", va="top", linespacing=1.15)
    text(SX, bbase - 0.8, "EEG score\nof $P_i$", AX1 - 0.3, size=BASE, ha="center", va="top", linespacing=1.15)
    text(AX0 + 1.6, 42.9, "Repeated for i = 1, …, n: every participant's score comes from a model that never saw that "
         "participant.", AX1 - 0.6, wrap=74, va="center", linespacing=1.2)

    # ================================================================================================================
    # b. Stage 2
    BX0, BX1 = 51.5, 99.0
    heading(BX0, 77.2, "b", "Stage 2 · EEG score + demographics")
    box(BX0, TY0, BX1 - BX0, TY1 - TY0, fc="white", ec=EDGE, lw=0.9, r=1.4)
    text(BX0 + 1.6, TY1 - 2.2, "Participant level · logistic regression", BX1 - 0.5, size=6, color=MUTED)

    cols = [("s", 3.8, True), ("age", 4.4, False), ("sex", 4.0, False), ("education", 6.9, False)]
    tx0, ttop, hh, rh = 56.0, 69.2, 2.6, 2.3
    rows = [("$P_1$", False), ("$P_2$", False), ("$P_i$", True), ("$P_n$", False)]
    x = tx0
    xs = []
    for k, (name, w, is_s) in enumerate(cols):
        xs.append((x, x + w - 0.25))
        ax.add_patch(Rectangle((x, ttop - hh), w - 0.25, hh, fc=BLUE if is_s else GREY, ec="none", zorder=4))
        text(x + (w - 0.25) / 2, ttop - hh / 2, name, size=6, color="white", weight="bold", ha="center",
             style="italic" if is_s else "normal")
        for r, (lab, held) in enumerate(rows):
            y = ttop - hh - 0.25 - (r + 1) * rh
            fc = (BLUE if held else BLUE_L) if is_s else GREY_L
            ax.add_patch(Rectangle((x, y), w - 0.25, rh - 0.25, fc=fc, ec="none", zorder=4))
            if is_s and held:
                text(x + (w - 0.25) / 2, y + (rh - 0.25) / 2, "$s_i$", size=6.3, color="white", ha="center")
        x += w + (0.6 if is_s else 0.0)
    t_end = x - 0.25
    for r, (lab, held) in enumerate(rows):
        y = ttop - hh - 0.25 - (r + 1) * rh + (rh - 0.25) / 2
        text(tx0 - 0.7, y, lab, size=6.3, ha="right", color=ORANGE if held else INK,
             weight="bold" if held else "normal")
    t_bot = ttop - hh - 0.25 - len(rows) * rh
    for (xa, xb), lab, c in ((xs[0], "from stage 1", BLUE), ((xs[1][0], xs[-1][1]), "demographics", GREY)):
        line([xa, xa, xb, xb], [t_bot - 0.4, t_bot - 0.9, t_bot - 0.9, t_bot - 0.4], color=c, lw=0.7)
        text((xa + xb) / 2, t_bot - 1.4, lab, size=5.8, color=BLUE if c == BLUE else MUTED, ha="center", va="top",
             weight="bold" if c == BLUE else "normal")

    # s_i from panel a into row P_i
    yi = ttop - hh - 0.25 - 3 * rh + (rh - 0.25) / 2
    line([SX + 2.6, 50.0, 50.0, 53.3], [SCY, SCY, yi, yi], color=BLUE, lw=1.3, z=7)
    arrow(53.2, yi, 54.0, yi, color=BLUE, lw=1.3, ms=7, z=7)

    # the two nested models
    MX0, MX1 = 77.3, 97.9
    box(MX0, 61.4, MX1 - MX0, 8.6, fc=SOFT, r=0.8, z=2)
    text(MX0 + 1.0, 68.4, "Model 0: demographics only", MX1 - 0.3, size=6.6, weight="bold")
    text(MX0 + 1.0, 66.5, "logit P(AD) = $β_0$ + $β_1$ age\n+ $β_2$ sex + $β_3$ education", MX1 - 0.3, size=BASE, va="top",
         linespacing=1.3)
    box(MX0, 49.6, MX1 - MX0, 10.0, fc=SOFT, ec=BLUE, lw=1.0, r=0.8, z=2)
    text(MX0 + 1.0, 58.0, "Model 1: adds the EEG score", MX1 - 0.3, size=6.6, weight="bold")
    t0 = text(MX0 + 1.0, 55.6, "Model 0 +", size=7.0)
    fig.canvas.draw()
    REND = fig.canvas.get_renderer()
    INV = ax.transData.inverted()
    xe = INV.transform(t0.get_window_extent(REND))[1, 0]
    text(xe + 0.7, 55.6, "γ · z", size=7.0, color=BLUE, weight="bold")
    text(MX0 + 1.0, 53.4, "z: logit of s, standardised\nOR per SD of the score = exp(γ)", MX1 - 0.3, size=5.8,
         color=MUTED, va="top", linespacing=1.3)
    arrow(t_end + 0.5, 65.8, MX0 - 0.4, 65.8, ms=6)
    arrow(t_end + 0.5, 56.0, MX0 - 0.4, 56.0, ms=6)
    text(BX0 + 1.6, 46.0, "Both models are fitted to the same participants; they differ only by the EEG "
         "score, so the gain from Model 0 to Model 1 is what the EEG adds.", BX1 - 0.5, wrap=78, va="center",
         linespacing=1.2)
    text(BX0 + 1.6, 43.0, "Greek cohort: age and sex only (education not recorded).", BX1 - 0.5, size=5.8,
         color=MUTED)

    # ================================================================================================================
    # c. What each comparison answers
    CX0, CX1, CY0, CY1 = 1.0, 73.6, 5.2, 34.6
    heading(CX0, 37.8, "c", "What each comparison answers")
    box(CX0, CY0, CX1 - CX0, CY1 - CY0, fc="white", ec=EDGE, lw=0.9, r=1.4)
    cw = (CX1 - CX0 - 4 * 1.1) / 3
    cards = [CX0 + 1.1 + k * (cw + 1.1) for k in range(3)]
    GY0, GY1 = 23.2, 29.6                          # band for the small graphics
    for k, x0 in enumerate(cards):
        box(x0, CY0 + 1.1, cw, CY1 - CY0 - 2.2, fc=SOFT, r=0.9, z=1)
    heads = ["Does the EEG add to demographics? (RQ1)", "How well do the models discriminate?",
             "Independent check of RQ1"]
    for x0, hd in zip(cards, heads):
        text(x0 + 1.0, CY1 - 2.6, hd, x0 + cw - 0.4, size=6.6, weight="bold", wrap=30, va="center", linespacing=1.15)

    # c1: null distribution
    x0 = cards[0]
    gx0, gx1 = x0 + 2.0, x0 + cw - 2.0
    edges = np.linspace(0, 1, 15)
    dens = np.exp(-4.2 * edges[:-1]) + 0.02
    dens = dens / dens.max()
    bw = (gx1 - gx0) / len(dens)
    for k, d in enumerate(dens):
        ax.add_patch(Rectangle((gx0 + k * bw + 0.1, GY0), bw - 0.2, (GY1 - GY0 - 1.2) * d, fc=GREY, ec="none",
                               zorder=4))
    line([gx0, gx1], [GY0, GY0], color=EDGE, lw=0.6)
    xo = gx0 + 0.72 * (gx1 - gx0)
    line([xo, xo], [GY0, GY1 - 0.4], color=BLUE, lw=1.3, z=5)
    text(xo + 0.4, GY1 - 1.0, "observed", size=5.6, color=BLUE, weight="bold")
    text(gx0 + 3.0, GY1 + 0.2, "simulated with\nno EEG effect", size=5.4, color=MUTED, linespacing=1.1, va="top")
    text(x0 + 1.0, GY0 - 1.6,
         "Likelihood-ratio test of Model 1 against Model 0; effect size: OR per SD of the score. The p-value is not "
         "taken from the χ² table: diagnoses are simulated from Model 0 (no EEG effect), stages 1 and 2 are re-run on "
         "each simulated data set, and p is the share of simulated statistics at least as large as the observed one.",
         x0 + cw - 0.3, size=BASE, wrap=35, va="top", linespacing=1.22)

    # c2: ROC curves
    x0 = cards[1]
    rx0, ry0, rs = x0 + 2.4, GY0, GY1 - GY0
    ax.add_patch(Rectangle((rx0, ry0), rs, rs, fc="white", ec=EDGE, lw=0.6, zorder=3))
    line([rx0, rx0 + rs], [ry0, ry0 + rs], color=EDGE, lw=0.6, ls=(0, (2, 1.5)))
    u = np.linspace(0, 1, 50)
    for pw, c, lw_ in ((0.55, GREY, 1.1), (0.25, BLUE, 1.3)):
        line(rx0 + rs * u, ry0 + rs * u ** pw, color=c, lw=lw_, z=4)
    lx = rx0 + rs + 1.4
    line([lx, lx + 2.0], [ry0 + rs - 1.0] * 2, color=BLUE, lw=1.3)
    text(lx + 2.5, ry0 + rs - 1.0, "Model 1", x0 + cw - 0.3, size=5.8)
    line([lx, lx + 2.0], [ry0 + rs - 3.0] * 2, color=GREY, lw=1.1)
    text(lx + 2.5, ry0 + rs - 3.0, "Model 0", x0 + cw - 0.3, size=5.8)
    text(lx, ry0 + 0.8, "ROC curves", size=5.4, color=MUTED)
    text(x0 + 1.0, GY0 - 1.6,
         "AUC of the EEG score alone (out of fold, from stage 1), of Model 0 and of Model 1. Both logistic models "
         "are refitted "
         "leaving one participant out, so no participant is scored by a model fitted on them. The difference in AUC is "
         "reported as descriptive.",
         x0 + cw - 0.3, size=BASE, wrap=35, va="top", linespacing=1.22)

    # c3: the other cohort's model
    x0 = cards[2]
    cy = (GY0 + GY1) / 2
    box(x0 + 1.2, cy - 2.2, 6.6, 4.4, fc="white", ec=EDGE, lw=0.7, r=0.5, z=3)
    text(x0 + 4.5, cy, "other\ncohort", size=5.6, ha="center", linespacing=1.05)
    trees(x0 + 12.2, cy - 1.6, h=3.6)
    arrow(x0 + 8.1, cy, x0 + 9.4, cy, ms=5)
    text(x0 + 12.2, cy - 2.5, "frozen", size=5.2, color=MUTED, ha="center", va="top")
    box(x0 + 16.6, cy - 1.9, 3.8, 3.8, fc=BLUE, r=0.5, z=4)
    text(x0 + 18.5, cy, "s′", size=7.5, color="white", ha="center", weight="bold")
    arrow(x0 + 15.0, cy, x0 + 16.4, cy, ms=5)
    text(x0 + 1.0, GY0 - 1.6,
         "s is replaced by s′, the score of the model trained on the other cohort, which never saw any participant of "
         "this cohort. Model 1 against Model 0 is then tested with the ordinary likelihood-ratio test, which is valid "
         "here because s′ is independent of these diagnoses.",
         x0 + cw - 0.3, size=BASE, wrap=35, va="top", linespacing=1.22)

    # ================================================================================================================
    # d. Not used: early fusion
    DX0, DX1 = 75.4, 99.0
    heading(DX0, 37.8, "d", "Not used: early fusion")
    box(DX0, CY0, DX1 - DX0, CY1 - CY0, fc="white", ec=EDGE, lw=0.9, r=1.4)
    text(DX0 + 1.4, CY1 - 2.6, "Demographics added to the\nRandom Forest's epoch table", DX1 - 0.3, size=6.6,
         weight="bold", linespacing=1.15)
    ex0, etop, er, ecw = DX0 + 4.2, 26.6, 1.25, 1.35
    ncol_eeg, ncol_dem = 6, 3
    ax.add_patch(Rectangle((ex0, etop), ncol_eeg * ecw - 0.2, 1.3, fc=GREY, ec="none", zorder=4))
    text(ex0 + (ncol_eeg * ecw - 0.2) / 2, etop + 0.65, "EEG features", size=5.0, color="white", ha="center",
         weight="bold")
    dx = ex0 + ncol_eeg * ecw + 0.4
    ax.add_patch(Rectangle((dx, etop), ncol_dem * ecw - 0.2, 1.3, fc=INK, ec="none", zorder=4))
    text(dx + (ncol_dem * ecw - 0.2) / 2, etop + 0.65, "a  s  e", size=5.0, color="white", ha="center",
         weight="bold")
    rng = np.random.default_rng(3)
    erows = ["1", "2", "3", "⋮", "m"]
    for r, lab in enumerate(erows):
        y = etop - 0.2 - (r + 1) * er
        if lab == "⋮":
            vdots(ex0 + (ncol_eeg * ecw) / 2, y + er / 2, gap=0.35, r=0.13)
            vdots(dx + (ncol_dem * ecw) / 2, y + er / 2, gap=0.35, r=0.13)
            continue
        text(ex0 - 0.6, y + er / 2 - 0.1, lab, size=5.2, ha="right", color=MUTED)
        for c in range(ncol_eeg):
            g = 0.80 + 0.12 * rng.random()
            ax.add_patch(Rectangle((ex0 + c * ecw, y), ecw - 0.2, er - 0.2, fc=(g, g, g + 0.02), ec="none", zorder=4))
        for c, col in enumerate(("#6B7280", "#8B93A0", "#A7AEB8")):
            ax.add_patch(Rectangle((dx + c * ecw, y), ecw - 0.2, er - 0.2, fc=col, ec="none", zorder=4))
    e_bot = etop - 0.2 - len(erows) * er
    text(ex0, etop + 2.3, "all epochs of $P_1$", size=5.2, color=MUTED, ha="left")
    line([dx + ncol_dem * ecw + 0.2, dx + ncol_dem * ecw + 0.6, dx + ncol_dem * ecw + 0.6, dx + ncol_dem * ecw + 0.2],
         [etop - 0.3, etop - 0.3, e_bot, e_bot], color=INK, lw=0.7)
    text(dx + ncol_dem * ecw + 1.0, (etop + e_bot) / 2, "same\nvalues\nin every\nepoch", DX1 - 0.3, size=5.2,
         linespacing=1.05, color=INK)
    text(DX0 + 1.4, e_bot - 1.4, "a, s, e: age, sex, education", DX1 - 0.3, size=5.2, color=MUTED, va="top")
    text(DX0 + 1.4, e_bot - 3.6,
         "Each participant's demographics would be copied to ≈100–200 epochs (pseudoreplication), the forest would mix "
         "them with the EEG, and no single coefficient would measure what the EEG adds.",
         DX1 - 0.3, size=BASE, wrap=33, va="top", linespacing=1.22)

    Y_CUT = CY0 - 0.6                              # drop the empty strip below the bottom panels
    ax.set_ylim(Y_CUT, H)
    fig.set_size_inches(7.1, 7.1 * (H - Y_CUT) / W)
    return fig
