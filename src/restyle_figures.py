"""Re-render all paper figures in three aesthetic themes for comparison.

Outputs go to:
    tex/figures_calm/        - muted teal/coral/sage, serif (matches LaTeX body)
    tex/figures_minimal/     - Tufte-inspired, near-black + one accent
    tex/figures_bold/        - Okabe-Ito colorblind-safe, sans-serif, slightly bolder

Usage:
    uv run python src/restyle_figures.py            # all themes, all figures
    uv run python src/restyle_figures.py --theme calm
    uv run python src/restyle_figures.py --skip-umap --skip-novelty   # fast iteration
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


# =====================================================================
# Themes
# =====================================================================
THEMES: dict[str, dict] = {
    "calm": {
        # Muted, paper-friendly palette inspired by recent Nature/Cell figures.
        "palette": ["#2F6F8F", "#D7634D", "#6FA86A", "#D9A441", "#8B5E83",
                    "#4F6D7A", "#C39B77"],
        "neutral": "#2A2A2A",
        "muted":   "#7A7A7A",
        "grid":    "#DCDCDC",
        "bg":      "white",
        "font_family": ["DejaVu Serif", "Times New Roman", "serif"],
        "linewidth": 1.6,
        "marker_size": 5.0,
        "spine_top_right": False,   # remove top + right
        "grid_axes": "y",            # only horizontal grid lines
        "grid_alpha": 0.65,
        "grid_linestyle": "-",
        "grid_linewidth": 0.5,
        "label_in_data": False,
    },
    "minimal": {
        # Tufte-inspired: near-black + one accent. Hairline everything.
        "palette": ["#1B1B1B", "#B5253A", "#7C7C7C", "#3F6E8C", "#A38C00",
                    "#3A6F3A", "#7E4E7C"],
        "neutral": "#1B1B1B",
        "muted":   "#9E9E9E",
        "grid":    "#EAEAEA",
        "bg":      "white",
        "font_family": ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
        "linewidth": 1.1,
        "marker_size": 3.5,
        "spine_top_right": False,
        "grid_axes": "none",
        "grid_alpha": 0.0,
        "grid_linestyle": ":",
        "grid_linewidth": 0.4,
        "label_in_data": True,       # prefer direct data labels over legend boxes
    },
    "bold": {
        # Okabe-Ito (colorblind safe). Sans-serif, slightly heavier strokes.
        "palette": ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
                    "#56B4E9", "#F0E442"],
        "neutral": "#222222",
        "muted":   "#5A5A5A",
        "grid":    "#E2E2E2",
        "bg":      "white",
        "font_family": ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
        "linewidth": 2.1,
        "marker_size": 6.0,
        "spine_top_right": False,
        "grid_axes": "y",
        "grid_alpha": 0.55,
        "grid_linestyle": "-",
        "grid_linewidth": 0.6,
        "label_in_data": False,
    },
}

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


# =====================================================================
# Helpers
# =====================================================================
def out_dir(theme_name: str) -> Path:
    p = ROOT / "tex" / f"figures_{theme_name}"
    p.mkdir(parents=True, exist_ok=True)
    return p


@contextmanager
def styled(theme: dict):
    """rc_context that fully applies a theme."""
    rc = {
        "figure.facecolor": theme["bg"],
        "axes.facecolor":   theme["bg"],
        "savefig.facecolor": theme["bg"],
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
        "font.family": theme["font_family"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,           # only used if a title is set explicitly
        "axes.titleweight": "regular",
        "axes.titlepad": 8,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "axes.edgecolor": theme["neutral"],
        "axes.labelcolor": theme["neutral"],
        "xtick.color": theme["neutral"],
        "ytick.color": theme["neutral"],
        "text.color": theme["neutral"],
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "lines.linewidth": theme["linewidth"],
        "lines.markersize": theme["marker_size"],
        "patch.linewidth": 0.6,
        "axes.spines.top":   theme["spine_top_right"],
        "axes.spines.right": theme["spine_top_right"],
        "axes.grid": False,
    }
    with plt.rc_context(rc):
        yield


def style_axes(ax, theme):
    """Apply per-axes touch-ups (grid, spine colours)."""
    ax.tick_params(direction="out", length=3, width=0.8, color=theme["neutral"])
    for spine in ("top", "right"):
        if not theme["spine_top_right"]:
            ax.spines[spine].set_visible(False)
    if theme["grid_axes"] in ("y", "both"):
        ax.yaxis.grid(True, color=theme["grid"], linestyle=theme["grid_linestyle"],
                      linewidth=theme["grid_linewidth"], alpha=theme["grid_alpha"])
        ax.set_axisbelow(True)
    if theme["grid_axes"] in ("x", "both"):
        ax.xaxis.grid(True, color=theme["grid"], linestyle=theme["grid_linestyle"],
                      linewidth=theme["grid_linewidth"], alpha=theme["grid_alpha"])
        ax.set_axisbelow(True)


def knot_label(label: str) -> str:
    """Convert a knot type string like '3_1' or '3_1#3_1' into LaTeX."""
    parts = label.split("#")
    out = []
    for p in parts:
        m = re.match(r"^(\d+)_(\d+)$", p.strip())
        if m:
            out.append(f"${m.group(1)}_{{{m.group(2)}}}$")
        else:
            out.append(p)
    return "#".join(out)


def save(fig, outdir: Path, name: str):
    fig.savefig(outdir / f"{name}.pdf")
    fig.savefig(outdir / f"{name}.png")
    plt.close(fig)


# =====================================================================
# Data loaders (cached)
# =====================================================================
_cache: dict[str, object] = {}


def load_json(name: str):
    if name not in _cache:
        with open(RESULTS / name) as f:
            _cache[name] = json.load(f)
    return _cache[name]


# =====================================================================
# Figure functions
# =====================================================================
def fig_masking_curve(theme, outdir):
    data = load_json("masking_smoke.json")
    proteins = [r for r in data["results"] if r.get("levels") and not r.get("error")]
    levels = sorted(int(k) for k in proteins[0]["levels"].keys())

    means, stds = [], []
    for lev in levels:
        vals = np.array([r["levels"][str(lev)]["avg_knotted_p"] for r in proteins])
        means.append(vals.mean())
        stds.append(vals.std())

    means = np.array(means)
    stds = np.array(stds)
    # Mean ± 1 SD, but clipped to [0, 1] so the band doesn't spill past valid range
    lo = np.clip(means - stds, 0, 1)
    hi = np.clip(means + stds, 0, 1)

    with styled(theme):
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        c0 = theme["palette"][0]

        ax.fill_between(levels, lo, hi, color=c0, alpha=0.18, linewidth=0,
                        label="±1 SD")
        ax.plot(levels, means, "-o", color=c0, label="Mean knot probability",
                markersize=theme["marker_size"], linewidth=theme["linewidth"])

        ax.axhline(0.5, color=theme["muted"], linestyle="--",
                   linewidth=0.7, zorder=0)
        ax.text(11, 0.52, "knot lost", color=theme["muted"], fontsize=9,
                ha="left", va="bottom")

        ax.set_xlabel("Sequence masking (%)")
        ax.set_ylabel("Knot probability")
        ax.set_xlim(8, 92)
        ax.set_ylim(0, 1.02)
        ax.set_xticks([10, 20, 30, 40, 50, 60, 70, 80, 90])
        style_axes(ax, theme)
        if not theme["label_in_data"]:
            ax.legend(loc="lower left")
        else:
            # direct annotation
            ax.annotate("Mean", xy=(40, means[3]), xytext=(40, means[3] + 0.07),
                        color=c0, fontsize=10, ha="center")
        save(fig, outdir, "fig_masking_curve")


def fig_breaking_histogram(theme, outdir):
    data = load_json("masking_smoke.json")
    proteins = [r for r in data["results"] if r.get("levels") and not r.get("error")]
    levels = sorted(int(k) for k in proteins[0]["levels"].keys())

    bps = []
    for r in proteins:
        bp = max(levels) + 5
        for lev in levels:
            if r["levels"][str(lev)]["avg_knotted_p"] < 0.5:
                bp = lev
                break
        bps.append(bp)

    with styled(theme):
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        c0 = theme["palette"][0]
        c_med = theme["palette"][1]

        bins = np.arange(7.5, 100, 5)
        ax.hist(bps, bins=bins, color=c0, edgecolor="white", linewidth=0.8,
                alpha=0.9 if theme is not THEMES["minimal"] else 0.85)

        med = np.median(bps)
        ax.axvline(med, color=c_med, linestyle="--", linewidth=1.0)
        ax.text(med - 1, ax.get_ylim()[1] * 0.95,
                f"median = {med:.0f}%", color=c_med, ha="right", va="top",
                fontsize=10)

        ax.set_xlabel("Breaking point (sequence masking %)")
        ax.set_ylabel("Number of proteins")
        ax.set_xlim(5, 100)
        style_axes(ax, theme)
        save(fig, outdir, "fig_breaking_histogram")


def fig_rmsd_vs_knot(theme, outdir):
    """Two-panel plot: structural drift on top, topological persistence on bottom.
    Removes the previous confusing dual y-axis."""
    data = load_json("rmsd_analysis.json")
    valid = [r for r in data["results"] if r.get("levels") and not r.get("error")]
    levels = [5, 10, 20, 30, 40, 50, 60, 70, 80, 85, 90, 95]

    rmsd_med, rmsd_q25, rmsd_q75 = [], [], []
    knot_mean, sid_mean = [], []
    for pct in levels:
        rmsds = [r["levels"][str(pct)]["rmsd"] for r in valid
                 if str(pct) in r["levels"] and "rmsd" in r["levels"][str(pct)]]
        knots = [r["levels"][str(pct)]["knotted_p"] for r in valid
                 if str(pct) in r["levels"] and "knotted_p" in r["levels"][str(pct)]]
        sids = [r["levels"][str(pct)]["seq_identity"] for r in valid
                if str(pct) in r["levels"] and "seq_identity" in r["levels"][str(pct)]]
        rmsd_med.append(np.median(rmsds) if rmsds else np.nan)
        rmsd_q25.append(np.percentile(rmsds, 25) if rmsds else np.nan)
        rmsd_q75.append(np.percentile(rmsds, 75) if rmsds else np.nan)
        knot_mean.append(np.mean(knots) if knots else np.nan)
        sid_mean.append(np.mean(sids) if sids else np.nan)

    rmsd_med = np.array(rmsd_med)
    rmsd_q25 = np.array(rmsd_q25)
    rmsd_q75 = np.array(rmsd_q75)

    with styled(theme):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.6, 5.2),
                                       sharex=True, gridspec_kw={"height_ratios": [1, 1.05]})
        c_rmsd = theme["palette"][1]
        c_knot = theme["palette"][0]
        c_sid  = theme["palette"][2]

        ax1.fill_between(levels, rmsd_q25, rmsd_q75, color=c_rmsd, alpha=0.18,
                         linewidth=0)
        ax1.plot(levels, rmsd_med, "-o", color=c_rmsd, label="Median RMSD")
        ax1.set_ylabel("Structural drift\nC$\\alpha$ RMSD (Å)")
        style_axes(ax1, theme)

        ax2.plot(levels, knot_mean, "-o", color=c_knot, label="Mean knot probability")
        ax2.plot(levels, sid_mean, "--^", color=c_sid, label="Mean sequence identity",
                 markersize=theme["marker_size"] - 1)
        ax2.set_ylim(0, 1.02)
        ax2.set_ylabel("Topology / Identity")
        ax2.set_xlabel("Sequence masking (%)")
        style_axes(ax2, theme)

        # Direct labels in minimal theme, legend otherwise
        if theme["label_in_data"]:
            ax1.annotate("RMSD", xy=(60, rmsd_med[6]), color=c_rmsd, fontsize=10,
                         xytext=(60, rmsd_med[6] + 1.2), ha="center")
            ax2.annotate("knot prob.", xy=(60, knot_mean[6]),
                         xytext=(60, knot_mean[6] + 0.07), color=c_knot,
                         fontsize=10, ha="center")
            ax2.annotate("seq. identity", xy=(60, sid_mean[6]),
                         xytext=(60, sid_mean[6] - 0.10), color=c_sid,
                         fontsize=10, ha="center")
        else:
            ax2.legend(loc="lower left")

        ax1.set_xlim(0, 100)
        save(fig, outdir, "fig_rmsd_vs_knot")


def fig_knot_types(theme, outdir):
    data = load_json("guided_gen_combined.json")
    topos = Counter()
    n_tmc = 0
    for r in data["results"]:
        if r["is_knotted"]:
            for k, v in r.get("topology", {}).items():
                if k == "0_1":
                    continue
                if k == "TMC":
                    n_tmc += 1
                    continue
                topos[k] += 1

    top = topos.most_common(12)
    labels = [knot_label(t[0]) for t in top]
    counts = [t[1] for t in top]

    with styled(theme):
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        c0 = theme["palette"][0]
        # Single hue with intensity sequence; trefoil dominant gets full saturation,
        # rest a single calm color for clarity (no rainbow noise)
        colors = [c0] * len(labels)

        y = np.arange(len(labels))
        bars = ax.barh(y, counts, color=colors, edgecolor="white", linewidth=0.6)
        ax.invert_yaxis()
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Occurrences in generated proteins")

        # value labels at bar ends
        for bar, val in zip(bars, counts):
            ax.text(val + 0.4, bar.get_y() + bar.get_height() / 2,
                    str(val), va="center", ha="left", color=theme["muted"],
                    fontsize=9)

        ax.set_xlim(0, max(counts) * 1.12)
        if n_tmc:
            ax.text(0.98, 0.04,
                    f"({n_tmc} additional knotted proteins exceeded the\n"
                    f"Alexander-polynomial crossing limit)",
                    transform=ax.transAxes, ha="right", va="bottom",
                    color=theme["muted"], fontsize=8.5)
        style_axes(ax, theme)
        # for horizontal bar, prefer x-grid not y-grid
        ax.yaxis.grid(False)
        if theme["grid_axes"] in ("y", "both"):
            ax.xaxis.grid(True, color=theme["grid"], linestyle=theme["grid_linestyle"],
                          linewidth=theme["grid_linewidth"], alpha=theme["grid_alpha"])
            ax.set_axisbelow(True)
        save(fig, outdir, "fig_knot_types")


def fig_transition_curves(theme, outdir):
    data = load_json("transition_analysis.json")
    levels = data["levels"]
    proteins = data["transition_data"]
    avg = np.array([data["avg_curve"][str(l)] for l in levels])
    std = np.array([data["std_curve"][str(l)] for l in levels])

    with styled(theme):
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        c_line = theme["palette"][0]
        c_envel = theme["muted"]

        # Plot a small random subset of individual curves to suggest variability
        rng = np.random.default_rng(0)
        idxs = rng.choice(len(proteins), size=min(25, len(proteins)), replace=False)
        for i in idxs:
            ax.plot(levels, proteins[i]["curve"], color=c_envel, alpha=0.22,
                    linewidth=0.6)

        # IQR-clipped envelope from real per-protein values
        all_curves = np.array([p["curve"] for p in proteins])
        q25 = np.percentile(all_curves, 25, axis=0)
        q75 = np.percentile(all_curves, 75, axis=0)

        ax.fill_between(levels, q25, q75, color=c_line, alpha=0.18, linewidth=0,
                        label="25–75% across proteins")
        ax.plot(levels, avg, "-o", color=c_line, label=f"Mean (n={data['n_proteins']})",
                markersize=theme["marker_size"])
        ax.axhline(0.5, color=theme["muted"], linestyle=":", linewidth=0.6, zorder=0)

        ax.set_xlabel("Sequence masking (%)")
        ax.set_ylabel("Knot probability")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(min(levels) - 2, max(levels) + 2)
        style_axes(ax, theme)
        if not theme["label_in_data"]:
            ax.legend(loc="lower left")
        save(fig, outdir, "fig_transition_curves")


def fig_contiguous_vs_random(theme, outdir):
    data = load_json("contiguous_masking.json")
    results = data["results"]
    levels = [25, 50, 75, 85, 90]

    rand_means, rand_sd, cont_means, cont_sd = [], [], [], []
    for lev in levels:
        rand = [r["levels"][str(lev)]["avg_knotted_p"] for r in results
                if r.get("mode") == "random" and r.get("levels") and str(lev) in r["levels"]]
        cont = [r["levels"][str(lev)]["avg_knotted_p"] for r in results
                if r.get("mode") == "contiguous" and r.get("levels") and str(lev) in r["levels"]]
        rand_means.append(np.mean(rand) if rand else 0)
        rand_sd.append(np.std(rand) / max(np.sqrt(len(rand)), 1) if rand else 0)
        cont_means.append(np.mean(cont) if cont else 0)
        cont_sd.append(np.std(cont) / max(np.sqrt(len(cont)), 1) if cont else 0)

    with styled(theme):
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        c0, c1 = theme["palette"][0], theme["palette"][1]
        x = np.arange(len(levels))
        w = 0.36

        ax.bar(x - w / 2, rand_means, w, yerr=rand_sd, color=c0,
               edgecolor="white", linewidth=0.6, label="Random masking",
               error_kw=dict(ecolor=theme["muted"], elinewidth=0.7, capsize=2))
        ax.bar(x + w / 2, cont_means, w, yerr=cont_sd, color=c1,
               edgecolor="white", linewidth=0.6, label="Contiguous masking",
               error_kw=dict(ecolor=theme["muted"], elinewidth=0.7, capsize=2))

        ax.set_xticks(x)
        ax.set_xticklabels([f"{l}%" for l in levels])
        ax.set_xlabel("Sequence masking")
        ax.set_ylabel("Mean knot probability")
        ax.set_ylim(0, 1.05)
        style_axes(ax, theme)
        if theme["label_in_data"]:
            ax.text(x[-1] - w / 2, rand_means[-1] + 0.04, "random",
                    color=c0, ha="center", fontsize=10)
            ax.text(x[-1] + w / 2, cont_means[-1] + 0.04, "contiguous",
                    color=c1, ha="center", fontsize=10)
        else:
            ax.legend(loc="upper right")
        save(fig, outdir, "fig_contiguous_vs_random")


def fig_targeted_masking(theme, outdir):
    """Mean knot prob as a function of the fraction of total sequence masked,
    separately for masking inside the knot core, outside the core, and at random.
    Aggregates per (mode, level) → mean pct_of_total_seq, mean knot prob."""
    data = load_json("targeted_masking.json")
    levels = ["0.25", "0.5", "0.75", "1.0"]

    series = {}
    for mode in ("core_only", "noncore_only", "random"):
        xs, ys, ys_err = [], [], []
        for lv_key in levels:
            x_vals = [r["levels"][lv_key]["pct_of_total_seq"]
                      for r in data["results"]
                      if r["mode"] == mode and lv_key in r["levels"]]
            y_vals = [r["levels"][lv_key]["avg_knotted_p"]
                      for r in data["results"]
                      if r["mode"] == mode and lv_key in r["levels"]]
            if not y_vals:
                continue
            xs.append(np.mean(x_vals))
            ys.append(np.mean(y_vals))
            ys_err.append(np.std(y_vals) / max(np.sqrt(len(y_vals)), 1))
        series[mode] = (np.array(xs), np.array(ys), np.array(ys_err))

    with styled(theme):
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        c_core = theme["palette"][1]
        c_non  = theme["palette"][2]
        c_rand = theme["palette"][0]

        for mode, label, color, marker in [
            ("noncore_only", "Non-core only", c_non, "s"),
            ("core_only", "Core only", c_core, "o"),
            ("random", "Random (entire sequence)", c_rand, "^"),
        ]:
            x, y, e = series[mode]
            ax.errorbar(x, y, yerr=e, marker=marker, linestyle="-",
                        color=color, label=label, capsize=2,
                        elinewidth=0.7, ecolor=theme["muted"],
                        markersize=theme["marker_size"])

        ax.axhline(0.5, color=theme["muted"], linestyle=":", linewidth=0.6, zorder=0)
        ax.set_xlabel("Fraction of total sequence masked (%)")
        ax.set_ylabel("Mean knot probability")
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, 102)
        style_axes(ax, theme)
        ax.legend(loc="lower left", ncol=1)
        save(fig, outdir, "fig_targeted_masking")


def fig_length_gen(theme, outdir):
    data = load_json("length_gen.json")
    by_len = defaultdict(list)
    for r in data["results"]:
        by_len[r["length"]].append(r)
    lengths = sorted(by_len.keys())
    success = [100 * np.mean([x["is_knotted"] for x in by_len[l]]) for l in lengths]
    mean_score = [np.mean([x["knot_score"] for x in by_len[l]]) for l in lengths]
    n_per = len(by_len[lengths[0]])

    with styled(theme):
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        c0, c1 = theme["palette"][0], theme["palette"][1]

        x = np.arange(len(lengths))
        w = 0.55
        bars = ax.bar(x, success, w, color=c0, edgecolor="white", linewidth=0.6,
                      label="Success rate (%)")
        for b, val in zip(bars, success):
            ax.text(b.get_x() + b.get_width() / 2, val + 2, f"{val:.0f}",
                    color=theme["muted"], ha="center", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(lengths)
        ax.set_xlabel("Target protein length (residues)")
        ax.set_ylabel("Success rate (%)")
        ax.set_ylim(0, 110)

        # Add mean score as small markers above bars (single axis, percent scale)
        ax2 = ax.twinx()
        ax2.plot(x, mean_score, "o--", color=c1, markersize=theme["marker_size"],
                 label="Mean knot score")
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("Mean knot score", color=c1)
        ax2.tick_params(axis="y", labelcolor=c1, color=c1)
        ax2.spines["right"].set_color(c1)
        ax2.spines["top"].set_visible(False)
        ax2.spines["left"].set_visible(False)
        ax2.spines["right"].set_visible(True)
        ax2.grid(False)

        style_axes(ax, theme)
        ax.text(0.02, 0.98, f"n = {n_per} per length",
                transform=ax.transAxes, va="top", ha="left",
                color=theme["muted"], fontsize=9)
        save(fig, outdir, "fig_length_gen")


def fig_typed_gen(theme, outdir):
    data = load_json("typed_gen.json")
    by_target = defaultdict(list)
    for r in data["results"]:
        by_target[r["target_type"]].append(r)
    types = ["3_1", "4_1", "5_1", "5_2"]
    types = [t for t in types if t in by_target]
    any_knot = [100 * np.mean([x["any_knot_prob"] > 0.5 for x in by_target[t]])
                for t in types]
    on_target = [100 * np.mean([x["hit_target"] for x in by_target[t]]) for t in types]

    with styled(theme):
        fig, ax = plt.subplots(figsize=(6.0, 3.8))
        c0, c1 = theme["palette"][0], theme["palette"][1]
        x = np.arange(len(types))
        w = 0.36
        b1 = ax.bar(x - w / 2, any_knot, w, color=c0, edgecolor="white",
                    linewidth=0.6, label="Any knot generated")
        b2 = ax.bar(x + w / 2, on_target, w, color=c1, edgecolor="white",
                    linewidth=0.6, label="Target type achieved")
        for bars in (b1, b2):
            for b in bars:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5,
                        f"{b.get_height():.0f}", color=theme["muted"],
                        ha="center", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([knot_label(t) for t in types])
        ax.set_xlabel("Target knot type")
        ax.set_ylabel("Success rate (%)")
        ax.set_ylim(0, 130)
        style_axes(ax, theme)
        ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0),
                  frameon=False)
        save(fig, outdir, "fig_typed_gen")


def fig_identity_vs_knot(theme, outdir):
    """Per-protein scatter of seq identity (after masking+regen) vs knot probability."""
    data = load_json("rmsd_analysis.json")
    valid = [r for r in data["results"] if r.get("levels") and not r.get("error")]

    # Collect (seq_id, knot_p, masking_pct) triples
    rows = []
    for r in valid:
        for pct_str, lv in r["levels"].items():
            if "seq_identity" in lv and "knotted_p" in lv:
                rows.append((float(lv["seq_identity"]), float(lv["knotted_p"]),
                             int(pct_str)))
    if not rows:
        print("  fig_identity_vs_knot: no data")
        return
    sid = np.array([row[0] for row in rows])
    kp = np.array([row[1] for row in rows])
    mp = np.array([row[2] for row in rows])

    with styled(theme):
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        # Sequential colormap mapped to masking %; works in print and colorblind
        cmap = plt.get_cmap("viridis" if theme is THEMES["bold"] else "cividis")
        sc = ax.scatter(sid, kp, c=mp, cmap=cmap, s=18, alpha=0.75,
                        edgecolors="none")
        cbar = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label("Sequence masking (%)")
        cbar.outline.set_linewidth(0.5)

        # Soft quadrant guides
        ax.axhline(0.5, color=theme["muted"], linestyle=":", linewidth=0.6)
        ax.axvline(0.5, color=theme["muted"], linestyle=":", linewidth=0.6)

        # Subtle quadrant labels in low-density corners (top-right is dense, skip)
        ax.text(0.02, 0.97, "novel sequence,\nknot preserved",
                transform=ax.transAxes, color=theme["palette"][2], fontsize=8.5,
                ha="left", va="top", style="italic")
        ax.text(0.02, 0.03, "novel sequence,\nknot lost",
                transform=ax.transAxes, color=theme["palette"][1], fontsize=8.5,
                ha="left", va="bottom", style="italic")

        ax.set_xlabel("Sequence identity to original")
        ax.set_ylabel("Knot probability")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(-0.02, 1.05)
        style_axes(ax, theme)
        save(fig, outdir, "fig_identity_vs_knot")


def fig_sliding_window(theme, outdir):
    data = load_json("sliding_window.json")
    results = data["results"]

    # Re-bin onto a common normalized grid (0..1) using per-protein interpolation
    grid = np.linspace(0.0, 1.0, 21)
    interp_curves = []
    for r in results:
        v = r.get("vulnerability") or []
        if not v:
            continue
        x = np.array([p["norm_position"] for p in v])
        y = np.array([p["avg_knotted_p"] for p in v])
        order = np.argsort(x)
        if len(x) < 2:
            continue
        interp_curves.append(np.interp(grid, x[order], y[order]))
    arr = np.array(interp_curves)
    mean = arr.mean(axis=0)
    q25 = np.percentile(arr, 25, axis=0)
    q75 = np.percentile(arr, 75, axis=0)

    with styled(theme):
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        c0 = theme["palette"][0]

        ax.fill_between(grid, q25, q75, color=c0, alpha=0.18, linewidth=0,
                        label="25–75% across proteins")
        ax.plot(grid, mean, "-o", color=c0,
                label=f"Mean (n={arr.shape[0]})",
                markersize=theme["marker_size"] - 1)

        # Light annotation indicating typical knot core position
        try:
            kl = load_json("knot_location_analysis.json")
            core_center = kl["core_center_mean"]
            ax.axvline(core_center, color=theme["muted"], linestyle="--",
                       linewidth=0.6, zorder=0)
            ax.text(core_center, 1.0,
                    "  mean core\n  centre",
                    color=theme["muted"], fontsize=9,
                    ha="left", va="top")
        except Exception:
            pass

        ax.set_xlabel("Normalized position of masking window along sequence")
        ax.set_ylabel("Knot probability\n(after window masked)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0.55, 1.02)
        style_axes(ax, theme)
        if not theme["label_in_data"]:
            ax.legend(loc="lower left")
        save(fig, outdir, "fig_sliding_window")


def get_or_compute_umap(seed: int = 42):
    """Cache UMAP 2D coordinates so we compute them only once across themes."""
    cache = ROOT / "results" / "umap_cache.json"
    if cache.exists():
        with open(cache) as f:
            return json.load(f)

    print("  computing UMAP from 5000x1536 embeddings (~2-3 min)...")
    embs = load_json("embeddings_all.json")
    ids = [r["id"] for r in embs]
    labels = [r["label"] for r in embs]
    X = np.array([r["embedding"] for r in embs], dtype=np.float32)

    import umap  # noqa: WPS433
    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3,
                        random_state=seed)
    Y = reducer.fit_transform(X)

    out = {"ids": ids, "labels": labels,
           "x": [float(v) for v in Y[:, 0]],
           "y": [float(v) for v in Y[:, 1]]}
    with open(cache, "w") as f:
        json.dump(out, f)
    return out


def fig_umap_knotted(theme, outdir, *, skip: bool = False):
    if skip:
        return
    try:
        coords = get_or_compute_umap()
    except Exception as exc:
        print(f"  fig_umap_knotted: skipped ({exc})")
        return
    convertibles = set(load_json("convertible_analysis.json")["convertible_ids"])

    x = np.array(coords["x"])
    y = np.array(coords["y"])
    labels = np.array(coords["labels"])
    is_conv = np.array([cid in convertibles for cid in coords["ids"]])

    with styled(theme):
        fig, ax = plt.subplots(figsize=(5.4, 5.0))
        c_un, c_kn, c_co = theme["palette"][0], theme["palette"][1], theme["palette"][3]

        ax.scatter(x[labels == 0], y[labels == 0], s=4, alpha=0.32, color=c_un,
                   edgecolors="none", label=f"Unknotted (n={int((labels == 0).sum())})")
        ax.scatter(x[labels == 1], y[labels == 1], s=6, alpha=0.55, color=c_kn,
                   edgecolors="none", label=f"Knotted (n={int((labels == 1).sum())})")
        if is_conv.any():
            ax.scatter(x[is_conv], y[is_conv], s=70, marker="*",
                       color=c_co, edgecolors="white", linewidths=0.6,
                       label=f"Convertible (n={int(is_conv.sum())})", zorder=5)

        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(loc="upper right", fontsize=9, markerscale=1.2,
                  handletextpad=0.4)
        style_axes(ax, theme)
        save(fig, outdir, "fig_umap_knotted")


def _seq_identity_bytes(a: np.ndarray, b: np.ndarray) -> float:
    """Max ungapped sliding alignment identity for two uint8 byte arrays.
    Vectorized via sliding_window_view; ~1ms per pair for ~1000-aa sequences."""
    from numpy.lib.stride_tricks import sliding_window_view
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0.0
    if n > m:
        a, b = b, a
        n, m = m, n
    windows = sliding_window_view(b, n)  # (m-n+1, n)
    matches = (windows == a[None, :]).sum(axis=1, dtype=np.int32)
    return float(matches.max()) / m


def _max_identity_against(query_bytes: np.ndarray,
                          ref_bytes_list: list[np.ndarray]) -> float:
    """Max identity of a single query vs a list of references."""
    best = 0.0
    for r in ref_bytes_list:
        v = _seq_identity_bytes(query_bytes, r)
        if v > best:
            best = v
    return best


def fig_novelty(theme, outdir, *, skip: bool = False, n_random: int = 100):
    """Histogram of max identity of generated knotted sequences vs known knotted."""
    if skip:
        return
    cache = ROOT / "results" / "novelty_per_protein.json"
    if cache.exists():
        with open(cache) as f:
            blob = json.load(f)
        gen_ids = np.array(blob["generated_max_identity"])
        rand_ids = np.array(blob["random_max_identity"])
    else:
        try:
            from datasets import load_dataset
        except ImportError:
            print("  fig_novelty: skipped (datasets not installed)")
            return
        try:
            ds = load_dataset("EvaKlimentova/Diffusion-all_knots", split="train")
        except Exception as exc:
            print(f"  fig_novelty: skipped ({exc})")
            return

        def _to_bytes(s: str) -> np.ndarray:
            return np.frombuffer(s.encode("ascii", "replace"), dtype=np.uint8)

        known_knotted = [_to_bytes(r["Sequence"]) for r in ds
                         if r["Tool"] == "Real" and r["Label"] == 1]
        known_unknotted_seqs = [r["Sequence"] for r in ds
                                if r["Tool"] == "Real" and r["Label"] == 0]

        gen = load_json("guided_gen_combined.json")
        gen_seqs = [_to_bytes(r["sequence"]) for r in gen["results"]
                    if r["is_knotted"]]

        rng = np.random.default_rng(0)
        rand_idx = rng.choice(len(known_unknotted_seqs),
                              size=min(n_random, len(known_unknotted_seqs)),
                              replace=False)
        rand_seqs = [_to_bytes(known_unknotted_seqs[i]) for i in rand_idx]

        print(f"  fig_novelty: computing identities ({len(gen_seqs)} gen + "
              f"{len(rand_seqs)} rand vs {len(known_knotted)} known)...")
        import time
        t0 = time.time()
        gen_ids_list = [_max_identity_against(q, known_knotted) for q in gen_seqs]
        print(f"    generated done in {time.time() - t0:.1f}s")
        t0 = time.time()
        rand_ids_list = [_max_identity_against(q, known_knotted) for q in rand_seqs]
        print(f"    random done in {time.time() - t0:.1f}s")

        gen_ids = np.array(gen_ids_list)
        rand_ids = np.array(rand_ids_list)
        with open(cache, "w") as f:
            json.dump({"generated_max_identity": gen_ids.tolist(),
                       "random_max_identity": rand_ids.tolist()}, f)

    with styled(theme):
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        # Match the paper caption: generated = blue, random = orange/red baseline
        c_gen = theme["palette"][0]
        c_rand = theme["palette"][1]

        # Convert to percent
        gen_pct = 100 * gen_ids
        rand_pct = 100 * rand_ids
        lo = min(gen_pct.min(), rand_pct.min()) - 1
        hi = max(gen_pct.max(), rand_pct.max()) + 1
        bins = np.linspace(lo, hi, 25)

        ax.hist(rand_pct, bins=bins, color=c_rand, alpha=0.55, edgecolor="white",
                linewidth=0.5,
                label=f"Random unknotted (n={len(rand_pct)}, mean {rand_pct.mean():.1f}%)")
        ax.hist(gen_pct, bins=bins, color=c_gen, alpha=0.65, edgecolor="white",
                linewidth=0.5,
                label=f"Generated knotted (n={len(gen_pct)}, mean {gen_pct.mean():.1f}%)")

        ax.axvline(rand_pct.mean(), color=c_rand, linestyle=":", linewidth=0.9)
        ax.axvline(gen_pct.mean(), color=c_gen, linestyle=":", linewidth=0.9)

        ax.set_xlabel("Max sequence identity to nearest known knotted protein (%)")
        ax.set_ylabel("Number of proteins")
        ax.set_xlim(lo, max(hi, 30))
        style_axes(ax, theme)
        ax.legend(loc="upper right")
        save(fig, outdir, "fig_novelty")


# =====================================================================
# Driver
# =====================================================================
ALL_FIGS = [
    ("masking_curve",       fig_masking_curve),
    ("breaking_histogram",  fig_breaking_histogram),
    ("rmsd_vs_knot",        fig_rmsd_vs_knot),
    ("knot_types",          fig_knot_types),
    ("transition_curves",   fig_transition_curves),
    ("contiguous_vs_random", fig_contiguous_vs_random),
    ("targeted_masking",    fig_targeted_masking),
    ("length_gen",          fig_length_gen),
    ("typed_gen",           fig_typed_gen),
    ("identity_vs_knot",    fig_identity_vs_knot),
    ("sliding_window",      fig_sliding_window),
    ("umap_knotted",        fig_umap_knotted),
    ("novelty",             fig_novelty),
]


def run(themes: list[str], skip_umap: bool, skip_novelty: bool, only: list[str] | None):
    for tn in themes:
        if tn not in THEMES:
            raise SystemExit(f"unknown theme: {tn}")
        out = out_dir(tn)
        print(f"\n=== Theme: {tn} -> {out.relative_to(ROOT)} ===")
        for name, fn in ALL_FIGS:
            if only and name not in only:
                continue
            kwargs = {}
            if name == "umap_knotted":
                kwargs["skip"] = skip_umap
            if name == "novelty":
                kwargs["skip"] = skip_novelty
            try:
                fn(THEMES[tn], out, **kwargs)
                print(f"  fig_{name}")
            except Exception as exc:
                print(f"  fig_{name}: ERROR ({type(exc).__name__}: {exc})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theme", choices=list(THEMES.keys()) + ["all"], default="all")
    ap.add_argument("--skip-umap", action="store_true",
                    help="skip UMAP recomputation/figure (slow first run)")
    ap.add_argument("--skip-novelty", action="store_true",
                    help="skip novelty figure (requires HF dataset download)")
    ap.add_argument("--only", nargs="*", default=None,
                    help="only render these figure names (e.g. masking_curve rmsd_vs_knot)")
    args = ap.parse_args()
    themes = list(THEMES.keys()) if args.theme == "all" else [args.theme]
    run(themes, args.skip_umap, args.skip_novelty, args.only)


if __name__ == "__main__":
    main()
