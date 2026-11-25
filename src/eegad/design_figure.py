"""Fig. 1 — study design (schematic). Drawn with matplotlib only (no image files), with the numbers of the run
(participants, epochs, channel correspondence) read from outputs/. Saved also as SVG with editable text, so it can be
retouched in Inkscape, PowerPoint (Insert > Pictures, then Convert to Shape) or similar."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle, Wedge

from .article_figures import GROUP, article_style

INK = "#222222"
SOFT = "#F4F4F4"
EDGE = "#8A8A8A"
RQ_COL = {"RQ1": "#0072B2", "RQ2": "#009E73", "RQ3": "#D55E00"}


# ---- drawing primitives ---------------------------------------------------------------------------------------------
def box(ax, x, y, w, h, fc="white", ec=INK, lw=0.9, r=1.2, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                                zorder=z))


def text(ax, x, y, s, size=6.3, weight="normal", ha="left", va="center", color=INK, z=5, **kw):
    ax.text(x, y, s, fontsize=size, fontweight=weight, ha=ha, va=va, color=color, zorder=z, **kw)


def arrow(ax, x0, y0, x1, y1, color=INK, lw=1.0, style="-|>", ms=7, z=2, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms, color=color, lw=lw,
                                 zorder=z, linestyle=ls, shrinkA=0, shrinkB=0))


def person(ax, x, y, h, color, alpha=1.0, z=4):
    """Bust icon: head and shoulders, bottom at y, height h."""
    ax.add_patch(Wedge((x, y), 0.42 * h, 0, 180, fc=color, ec="none", alpha=alpha, zorder=z))
    ax.add_patch(Circle((x, y + 0.66 * h), 0.21 * h, fc=color, ec="none", alpha=alpha, zorder=z))


def people(ax, x, y, h, color, n=3, alpha=1.0, crossed=False):
    """A small group: n busts side by side (front one larger)."""
    dx = 0.55 * h
    for k in range(n):
        xx = x + (k - (n - 1) / 2) * dx
        person(ax, xx, y + (0 if k == (n - 1) // 2 else 0.12 * h), h * (1.0 if k == (n - 1) // 2 else 0.82),
               color, alpha, z=4 if k == (n - 1) // 2 else 3)
    if crossed:
        w = (n - 1) * dx / 2 + 0.5 * h
        ax.plot([x - w, x + w], [y - 0.05 * h, y + 1.0 * h], color="#B03A2E", lw=1.0, zorder=6)


def number(ax, x, y, k, r=1.35, color=INK):
    ax.add_patch(Circle((x, y), r, fc=color, ec="none", zorder=6))
    text(ax, x, y, str(k), size=6.5, weight="bold", ha="center", color="white", z=7)


def flag_argentina(ax, x, y, w, h):
    for k, c in enumerate(("#74ACDF", "white", "#74ACDF")):
        ax.add_patch(Rectangle((x, y + h * (2 - k) / 3), w, h / 3, fc=c, ec="none", zorder=4))
    ax.add_patch(Circle((x + w / 2, y + h / 2), h * 0.11, fc="#F6B40E", ec="none", zorder=5))
    ax.add_patch(Rectangle((x, y), w, h, fc="none", ec="0.4", lw=0.4, zorder=6))


def flag_chile(ax, x, y, w, h):
    ax.add_patch(Rectangle((x, y), w, h / 2, fc="#D52B1E", ec="none", zorder=4))
    ax.add_patch(Rectangle((x, y + h / 2), w, h / 2, fc="white", ec="none", zorder=4))
    ax.add_patch(Rectangle((x, y + h / 2), h / 2, h / 2, fc="#0039A6", ec="none", zorder=5))
    ax.plot(x + h / 4, y + 3 * h / 4, marker="*", color="white", ms=2.2, zorder=6)
    ax.add_patch(Rectangle((x, y), w, h, fc="none", ec="0.4", lw=0.4, zorder=6))


def flag_greece(ax, x, y, w, h):
    s = h / 9
    for k in range(9):
        ax.add_patch(Rectangle((x, y + h - (k + 1) * s), w, s, fc="#0D5EAF" if k % 2 == 0 else "white", ec="none",
                               zorder=4))
    c = 5 * s
    ax.add_patch(Rectangle((x, y + h - c), c, c, fc="#0D5EAF", ec="none", zorder=5))
    ax.add_patch(Rectangle((x, y + h - c / 2 - s / 2), c, s, fc="white", ec="none", zorder=6))
    ax.add_patch(Rectangle((x + c / 2 - s / 2, y + h - c), s, c, fc="white", ec="none", zorder=6))
    ax.add_patch(Rectangle((x, y), w, h, fc="none", ec="0.4", lw=0.4, zorder=7))


def head(ax, x, y, r, dots_grey=None, dots_main=None, s_grey=0.6, s_main=3.2):
    """Head seen from above (nose up) with electrodes; dots in units of the head radius."""
    ax.add_patch(Ellipse((x - r, y), 0.22 * r, 0.42 * r, fc="white", ec=INK, lw=0.7, zorder=3))
    ax.add_patch(Ellipse((x + r, y), 0.22 * r, 0.42 * r, fc="white", ec=INK, lw=0.7, zorder=3))
    ax.add_patch(Polygon([(x - 0.16 * r, y + 0.97 * r), (x, y + 1.2 * r), (x + 0.16 * r, y + 0.97 * r)],
                         closed=True, fc="white", ec=INK, lw=0.7, zorder=3))
    ax.add_patch(Circle((x, y), r, fc="#FBFBFB", ec=INK, lw=0.8, zorder=4))
    if dots_grey is not None and len(dots_grey):
        ax.scatter(x + r * dots_grey[:, 0], y + r * dots_grey[:, 1], s=s_grey, color="0.6", lw=0, zorder=5)
    if dots_main is not None and len(dots_main):
        ax.scatter(x + r * dots_main[:, 0], y + r * dots_main[:, 1], s=s_main, color=INK, lw=0, zorder=6)


def project(U):
    """Azimuthal equidistant projection of unit vectors (x right, y nose, z up): vertex at the centre."""
    U = np.asarray(U, float)
    theta = np.arccos(np.clip(U[:, 2], -1, 1))
    phi = np.arctan2(U[:, 1], U[:, 0])
    rho = theta / (np.pi / 2)
    return np.column_stack([rho * np.cos(phi), rho * np.sin(phi)])


def electrode_layouts(channels, brainlat_selected):
    """2-D positions: the 128 BioSemi electrodes, the BioSemi electrodes used for BrainLat and the 19 standard 10-20
    positions, scaled so that the 10-20 ring sits inside the head outline."""
    import mne

    from .features import name_in_montage, standard_1020_positions, unit_vectors
    pos_1020, _ = standard_1020_positions(channels)
    n_st, U_st = unit_vectors(pos_1020)
    U19 = np.array([U_st[n_st.index(name_in_montage(c, pos_1020))] for c in channels])
    xy19 = project(U19)
    pos_bs = mne.channels.make_standard_montage("biosemi128").get_positions()["ch_pos"]
    n_bs, U_bs = unit_vectors(pos_bs)
    xy_bs = project(U_bs)
    scale = 0.86 / np.max(np.linalg.norm(xy19, axis=1))
    xy19, xy_bs = xy19 * scale, xy_bs * scale
    keep = np.linalg.norm(xy_bs, axis=1) <= 1.02
    sel = np.array([n in set(brainlat_selected) for n in n_bs])
    return {"biosemi_all": xy_bs[keep], "biosemi_selected": xy_bs[sel & keep], "1020": xy19}


def eeg_trace(ax, x, y, w, h, seed=3, epochs=4):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, 500)
    sig = (np.sin(2 * np.pi * 10 * t * epochs) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.7 * t))
           + 0.5 * np.sin(2 * np.pi * 5 * t * epochs + 1) + 0.35 * rng.standard_normal(len(t)))
    sig = np.convolve(sig, np.ones(5) / 5, mode="same")
    sig = sig / np.max(np.abs(sig))
    ax.plot(x + w * t, y + h / 2 + 0.42 * h * sig, color=INK, lw=0.5, zorder=5)
    for k in range(1, epochs):
        ax.plot([x + w * k / epochs] * 2, [y, y + h], color=RQ_COL["RQ3"], lw=0.6, ls=(0, (2, 1.5)), zorder=6)


def spectrum(ax, x, y, w, h):
    """Power spectrum icon (log frequency, 1-45 Hz) with the four bands of the relative-power features."""
    f = np.geomspace(1.0, 45.0, 300)
    psd = 1 / f ** 1.2 + 0.25 * np.exp(-0.5 * ((f - 9.5) / 1.4) ** 2)
    lp = np.log10(psd)
    lp = (lp - lp.min()) / (lp.max() - lp.min())
    fx = lambda v: x + w * np.log10(v) / np.log10(45.0)
    bands = (("θ", 4, 8, "#D6EAF8"), ("α", 8, 13, "#D5F5E3"), ("β", 13, 30, "#FDEBD0"), ("γ", 30, 45, "#F5EEF8"))
    for lab, lo, hi, c in bands:
        ax.add_patch(Rectangle((fx(lo), y), fx(hi) - fx(lo), h, fc=c, ec="none", zorder=3))
        text(ax, (fx(lo) + fx(hi)) / 2, y + h * 0.88, lab, size=5.5, ha="center", color="0.25", z=6)
    ax.plot(fx(f), y + 0.06 * h + 0.7 * h * lp, color=INK, lw=0.8, zorder=5)
    ax.plot([x, x + w], [y, y], color=INK, lw=0.5, zorder=5)
    ax.plot([x, x], [y, y + h], color=INK, lw=0.5, zorder=5)
    text(ax, x + w, y - 0.75, "Hz (log)", size=4.6, ha="right", color="0.35")


def tree(ax, x, y, h, color="#2E7D32"):
    ax.add_patch(Rectangle((x - 0.06 * h, y), 0.12 * h, 0.3 * h, fc="#6D4C41", ec="none", zorder=5))
    ax.add_patch(Polygon([(x - 0.35 * h, y + 0.25 * h), (x, y + h), (x + 0.35 * h, y + 0.25 * h)], closed=True,
                         fc=color, ec="none", zorder=5))


def loso_strip(ax, x, y, w, h, n=10, held=6):
    cw = w / n
    for k in range(n):
        held_out = k == held
        ax.add_patch(Rectangle((x + k * cw + 0.08 * cw, y), 0.84 * cw, h, fc=RQ_COL["RQ3"] if held_out else "0.82",
                               ec="none", zorder=5))


def lock(ax, x, y, s, color=INK):
    ax.add_patch(Rectangle((x - 0.5 * s, y - 0.5 * s), s, 0.75 * s, fc=color, ec="none", zorder=7))
    ax.add_patch(Wedge((x, y + 0.25 * s), 0.34 * s, 0, 180, width=0.12 * s, fc=color, ec="none", zorder=7))


# ---- the figure -----------------------------------------------------------------------------------------------------
def _lines(ax, x, y_top, lines, size=5.6, step=1.6, **kw):
    for k, ln in enumerate(lines):
        text(ax, x, y_top - k * step, ln, size=size, **kw)


def _cohort_card(ax, y0, h, flags, name, where, device, rows, notes, dots):
    box(ax, 1.0, y0, 30.0, h, fc="white", ec=EDGE, lw=0.8)
    fx = 2.4
    for fl in flags:
        fl(ax, fx, y0 + h - 4.3, 4.6, 3.1)
        fx += 5.4
    text(ax, fx + 0.4, y0 + h - 2.2, name, size=7.8, weight="bold")
    text(ax, fx + 0.4, y0 + h - 4.4, where, size=5.9, color="0.3")
    hy = y0 + h - 12.2
    head(ax, 8.2, hy, 4.4, dots_grey=dots[0], dots_main=dots[1])
    _lines(ax, 8.2, hy - 6.3, device, size=5.6, step=1.5, ha="center", color="0.25")
    yy = y0 + h - 8.4
    for lab, n, col, crossed in rows:
        people(ax, 18.2, yy - 1.2, 2.5, col, n=3, alpha=0.45 if crossed else 1.0, crossed=crossed)
        text(ax, 21.8, yy, lab, size=6.6, weight="bold", color=col if not crossed else "0.45")
        text(ax, 25.4, yy, f"n = {n}", size=6.2, color=INK if not crossed else "0.45")
        yy -= 4.5
    _lines(ax, 2.4, y0 + 5.6, notes[:2], size=5.8, step=1.75)
    text(ax, 2.4, y0 + 1.55, notes[2], size=5.4, color="0.4", style="italic")


def _rq_card(ax, y0, h, key, title, question, lines):
    box(ax, 68.0, y0, 31.0, h, fc="white", ec=RQ_COL[key], lw=1.0)
    ax.add_patch(Rectangle((68.0, y0 + 0.8), 1.1, h - 1.6, fc=RQ_COL[key], ec="none", zorder=3))
    text(ax, 70.4, y0 + h - 2.1, f"{key}   {title}", size=7.0, weight="bold", color=RQ_COL[key])
    _lines(ax, 70.4, y0 + h - 4.8, question, size=6.1, step=1.75, style="italic")
    _lines(ax, 70.4, y0 + h - 4.8 - 1.75 * len(question) - 1.4, lines, size=5.6, step=1.75)


def study_design_figure(info: dict):
    """info: n_bl_AD, n_bl_HC, bl_excluded_text, n_gr_AD, n_gr_HC, n_gr_FTD, epochs_bl, epochs_gr, sfreq_bl, sfreq_gr,
    angle_mean, angle_max, rf_trees, channels, brainlat_selected (BioSemi labels of the 19 electrodes used)."""
    article_style()
    fig = plt.figure(figsize=(7.1, 5.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 70.4)
    ax.set_aspect("equal")
    ax.axis("off")
    lay = electrode_layouts(info["channels"], info["brainlat_selected"])

    for x, letter, title in ((1.0, "a", "Two cohorts"), (33.4, "b", "One harmonised EEG pipeline"),
                             (68.0, "c", "Three questions")):
        text(ax, x, 68.3, letter, size=10, weight="bold")
        text(ax, x + 3.0, 68.3, title, size=8.2, weight="bold")

    # ---- a: cohorts ---------------------------------------------------------------------------------------------
    _cohort_card(ax, 36.6, 29.7, [flag_argentina, flag_chile], "BrainLat", "Argentina and Chile; multicentre",
                 ["BioSemi", f"128 channels · {info['sfreq_bl']:g} Hz"],
                 [("AD", info["n_bl_AD"], GROUP["AD"], False), ("HC", info["n_bl_HC"], GROUP["HC"], False)],
                 ["Demographics: age · sex · education", "Severity: MoCA, IFS",
                  info["bl_excluded_text"]],
                 (lay["biosemi_all"], lay["biosemi_selected"]))
    _cohort_card(ax, 5.0, 30.0, [flag_greece], "Greek cohort (ds004504)", "AHEPA hospital, Thessaloniki; one centre",
                 ["Nihon Kohden", f"19 channels · {info['sfreq_gr']:g} Hz"],
                 [("AD", info["n_gr_AD"], GROUP["AD"], False), ("HC", info["n_gr_HC"], GROUP["HC"], False),
                  ("FTD", info["n_gr_FTD"], GROUP["FTD"], True)],
                 ["Demographics: age · sex (education not recorded)", "Severity: MMSE",
                  "FTD excluded by design (the question is AD vs HC)"],
                 (None, lay["1020"]))

    # ---- b: pipeline (one container: the same steps for both cohorts) ------------------------------------------
    box(ax, 33.4, 5.0, 32.8, 61.3, fc="white", ec=EDGE, lw=0.8)
    steps_y, hs = [54.4, 42.8, 31.2, 19.6, 6.0], [10.6, 10.2, 10.2, 10.2, 12.2]
    for k, (y0, h) in enumerate(zip(steps_y, hs), start=1):
        box(ax, 34.2, y0, 31.2, h, fc=SOFT, ec="none", r=0.9)
        number(ax, 36.0, y0 + h - 1.8, k)
    for k in range(len(steps_y) - 1):
        arrow(ax, 36.0, steps_y[k] + 0.2, 36.0, steps_y[k + 1] + hs[k + 1] - 0.2, lw=0.8, ms=5)

    y0, h = steps_y[0] + 0.6, hs[0] - 0.6                         # 1 common montage
    text(ax, 38.2, y0 + h - 1.8, "Common montage: 19 electrodes (10-20)", size=6.5, weight="bold")
    head(ax, 40.6, y0 + 3.1, 2.4, dots_grey=lay["biosemi_all"], dots_main=lay["biosemi_selected"], s_grey=0.22,
         s_main=1.3)
    arrow(ax, 43.6, y0 + 3.1, 45.4, y0 + 3.1, lw=0.7, ms=5)
    head(ax, 48.0, y0 + 3.1, 2.4, dots_main=lay["1020"], s_main=1.3)
    _lines(ax, 51.6, y0 + 5.1, ["BrainLat: nearest of the", "128 BioSemi electrodes",
                                f"(mean {info['angle_mean']:.1f}°, max {info['angle_max']:.1f}°)"], step=1.55)

    y0, h = steps_y[1] + 0.6, hs[1] - 0.6                         # 2 pass band and reference
    text(ax, 38.2, y0 + h - 1.8, "Same pass band and reference", size=6.5, weight="bold")
    fx, fw, fh = 38.6, 8.4, 4.0
    ax.plot([fx, fx + fw], [y0 + 1.5, y0 + 1.5], color=INK, lw=0.5, zorder=5)
    ax.plot([fx + 0.4, fx + 1.1, fx + 6.2, fx + 7.3, fx + fw], [y0 + 1.5, y0 + 1.5 + fh, y0 + 1.5 + fh, y0 + 1.5,
                                                               y0 + 1.5], color=RQ_COL["RQ1"], lw=1.1, zorder=6)
    text(ax, fx + 0.6, y0 + 0.6, "0.5", size=5.0, ha="center", color="0.3")
    text(ax, fx + 6.8, y0 + 0.6, "40–45 Hz", size=5.0, ha="center", color="0.3")
    _lines(ax, 48.4, y0 + 6.0, ["0.5–45 Hz FIR in both cohorts;", "Greek data low-passed at", "40 Hz, as distributed in",
                                "BrainLat; average reference"], step=1.55)

    y0, h = steps_y[2] + 0.6, hs[2] - 0.6                         # 3 epochs
    text(ax, 38.2, y0 + h - 1.8, "Non-overlapping 4-s epochs", size=6.5, weight="bold")
    eeg_trace(ax, 38.4, y0 + 0.9, 10.6, 4.6)
    _lines(ax, 51.0, y0 + 5.3, [f"{info['epochs_bl']:,} epochs (BrainLat)", f"{info['epochs_gr']:,} epochs (Greek)",
                                "no epoch rejection"], step=1.6)

    y0, h = steps_y[3] + 0.6, hs[3] - 0.6                         # 4 features
    text(ax, 38.2, y0 + h - 1.8, "11 features per epoch", size=6.5, weight="bold")
    spectrum(ax, 38.6, y0 + 1.3, 9.4, 4.8)
    _lines(ax, 49.6, y0 + 5.6, ["relative θ, α, β, γ power;", "θ/α, θ/β; spectral entropy;",
                                "phase locking (PLV) θ, α, β;", "broadband correlation"], step=1.55)

    y0, h = steps_y[4] + 0.6, hs[4] - 0.6                         # 5 classifier and score
    text(ax, 38.2, y0 + h - 1.8, "EEG score of each participant", size=6.5, weight="bold")
    for k in range(3):
        tree(ax, 40.0 + 2.5 * k, y0 + 4.2, 3.1)
    loso_strip(ax, 38.6, y0 + 1.6, 9.6, 1.3)
    text(ax, 43.4, y0 + 0.7, "one participant left out", size=4.7, ha="center", color="0.35")
    _lines(ax, 49.8, y0 + 7.6, [f"Random Forest ({info['rf_trees']} trees),", "leave-one-subject-out: each",
                                "participant is scored by a", "model that never saw them;",
                                "P(AD) = mean over epochs"], step=1.6)

    arrow(ax, 31.2, 51.5, 33.3, 51.5, color=EDGE, lw=1.0, ms=6)   # a -> b
    arrow(ax, 31.2, 20.0, 33.3, 20.0, color=EDGE, lw=1.0, ms=6)

    # ---- c: questions -------------------------------------------------------------------------------------------
    _rq_card(ax, 47.4, 19.0, "RQ1", "Incremental value", ["Does the EEG add to age, sex", "and education?"],
             ["demographics vs demographics + EEG;", "likelihood-ratio test with p from a",
              "null bootstrap that re-runs the whole", "LOSO; check: the other cohort's model"])
    ax.add_patch(Rectangle((89.0, 61.8), 5.6, 1.5, fc="0.7", ec="none", zorder=4))
    ax.add_patch(Rectangle((89.0, 59.2), 5.6, 1.5, fc="0.7", ec="none", zorder=4))
    ax.add_patch(Rectangle((94.6, 59.2), 3.2, 1.5, fc=RQ_COL["RQ1"], ec="none", zorder=4))
    text(ax, 91.8, 64.2, "demographics", size=4.8, ha="center", color="0.35")
    text(ax, 96.2, 58.1, "+ EEG", size=4.8, ha="center", color=RQ_COL["RQ1"])

    _rq_card(ax, 26.8, 19.0, "RQ2", "Confounding by design", ["Does it still separate people", "who are alike?"],
             ["AD and HC compared only within", "groups made equal in education,",
              "age and sex: matched pairs,", "comparable pairs, overlap weights"])
    for yy in (40.6, 37.2):
        person(ax, 89.6, yy - 0.9, 2.3, GROUP["AD"])
        person(ax, 96.2, yy - 0.9, 2.3, GROUP["HC"])
        ax.plot([91.0, 94.8], [yy + 0.2, yy + 0.2], color=RQ_COL["RQ2"], lw=0.9, zorder=3)
    text(ax, 89.6, 35.0, "AD", size=4.8, ha="center", color=GROUP["AD"])
    text(ax, 96.2, 35.0, "HC", size=4.8, ha="center", color=GROUP["HC"])

    _rq_card(ax, 6.2, 19.0, "RQ3", "Transportability", ["Does a model from one cohort", "work in the other?"],
             ["trained on one cohort, frozen,", "applied once to the other:",
              "AUC and calibration; post hoc:", "local recalibration in the", "target cohort"])
    box(ax, 88.2, 17.4, 5.0, 2.4, fc="white", ec=INK, lw=0.6, r=0.5, z=4)
    text(ax, 90.7, 18.6, "BrainLat", size=4.9, ha="center", z=6)
    box(ax, 88.2, 11.6, 5.0, 2.4, fc="white", ec=INK, lw=0.6, r=0.5, z=4)
    text(ax, 90.7, 12.8, "Greek", size=4.9, ha="center", z=6)
    arrow(ax, 89.9, 17.3, 89.9, 14.1, lw=0.8, ms=5)
    arrow(ax, 91.5, 14.1, 91.5, 17.3, lw=0.8, ms=5)
    lock(ax, 95.2, 15.6, 1.3)
    text(ax, 95.2, 13.4, "frozen", size=4.8, ha="center", color="0.35")

    for yc in (56.9, 36.3, 15.7):                                  # b -> c
        arrow(ax, 66.3, yc, 67.9, yc, color=EDGE, lw=1.0, ms=6)

    box(ax, 1.0, 0.4, 98.0, 3.4, fc=SOFT, ec="none", r=0.8)
    text(ax, 50.0, 2.1, "Analysis plan and reading rules fixed in advance and frozen before the Greek diagnoses were "
                        "read · the same code and settings in both cohorts · open code", size=6.0, ha="center",
         color="0.25")
    return fig
