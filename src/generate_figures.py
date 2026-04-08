"""Generate all publication-quality figures for the paper."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

OUT = Path('tex/figures')
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})


def fig1_masking_curve():
    """Aggregated masking curve with error bars."""
    with open('results/masking_smoke.json') as f:
        data = json.load(f)
    proteins = [r for r in data['results'] if r.get('levels') and not r.get('error')]
    levels = sorted([int(k) for k in proteins[0]['levels'].keys()])

    means, stds, medians = [], [], []
    for lev in levels:
        vals = [r['levels'][str(lev)]['avg_knotted_p'] for r in proteins]
        means.append(np.mean(vals))
        stds.append(np.std(vals))
        medians.append(np.median(vals))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(levels, np.array(means) - np.array(stds),
                     np.array(means) + np.array(stds), alpha=0.2, color='#2980b9')
    ax.plot(levels, means, 'o-', color='#2980b9', linewidth=2, markersize=6, label='Mean')
    ax.plot(levels, medians, 's--', color='#e74c3c', linewidth=1.5, markersize=5, label='Median')
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7, label='Breaking threshold')
    ax.set_xlabel('Sequence masking (%)')
    ax.set_ylabel('Knot probability')
    ax.set_title(f'Knot Stability Under Random Masking (n={len(proteins)})')
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(5, 95)
    ax.grid(True, alpha=0.3)
    fig.savefig(OUT / 'fig_masking_curve.pdf')
    fig.savefig(OUT / 'fig_masking_curve.png')
    plt.close()
    print(f"  fig_masking_curve.pdf")


def fig2_breaking_histogram():
    """Breaking point histogram."""
    with open('results/masking_smoke.json') as f:
        data = json.load(f)
    proteins = [r for r in data['results'] if r.get('levels') and not r.get('error')]
    levels = sorted([int(k) for k in proteins[0]['levels'].keys()])

    breaking_points = []
    for r in proteins:
        bp = 95
        for lev in levels:
            if r['levels'][str(lev)]['avg_knotted_p'] < 0.5:
                bp = lev
                break
        breaking_points.append(bp)

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = list(range(5, 100, 5))
    ax.hist(breaking_points, bins=bins, color='#2980b9', edgecolor='white', alpha=0.8)
    ax.axvline(np.median(breaking_points), color='#e74c3c', linestyle='--',
               linewidth=2, label=f'Median = {np.median(breaking_points):.0f}%')
    ax.set_xlabel('Breaking point (masking %)')
    ax.set_ylabel('Number of proteins')
    ax.set_title(f'Distribution of Knot Breaking Points (n={len(proteins)})')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    fig.savefig(OUT / 'fig_breaking_histogram.pdf')
    fig.savefig(OUT / 'fig_breaking_histogram.png')
    plt.close()
    print(f"  fig_breaking_histogram.pdf")


def fig3_rmsd_vs_knot():
    """RMSD vs knot probability dual-axis plot."""
    with open('results/rmsd_analysis.json') as f:
        data = json.load(f)
    valid = [r for r in data['results'] if r.get('levels') and not r.get('error')]
    levels = [5, 10, 20, 30, 40, 50, 60, 70, 80, 85, 90, 95]

    mean_rmsd, mean_knot, mean_seqid = [], [], []
    for pct in levels:
        rmsds = [r['levels'][str(pct)]['rmsd'] for r in valid
                 if str(pct) in r['levels'] and 'rmsd' in r['levels'][str(pct)]]
        knots = [r['levels'][str(pct)]['knotted_p'] for r in valid
                 if str(pct) in r['levels'] and 'knotted_p' in r['levels'][str(pct)]]
        seqids = [r['levels'][str(pct)]['seq_identity'] for r in valid
                  if str(pct) in r['levels'] and 'seq_identity' in r['levels'][str(pct)]]
        mean_rmsd.append(np.median(rmsds) if rmsds else 0)
        mean_knot.append(np.mean(knots) if knots else 0)
        mean_seqid.append(np.mean(seqids) if seqids else 0)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    l1, = ax1.plot(levels, mean_rmsd, 'o-', color='#e74c3c', linewidth=2, markersize=6, label='Median RMSD (Å)')
    l2, = ax2.plot(levels, mean_knot, 's-', color='#2980b9', linewidth=2, markersize=6, label='Mean knot probability')
    l3, = ax2.plot(levels, mean_seqid, '^--', color='#27ae60', linewidth=1.5, markersize=5, label='Sequence identity')

    ax1.set_xlabel('Sequence masking (%)')
    ax1.set_ylabel('Median RMSD (Å)', color='#e74c3c')
    ax2.set_ylabel('Probability / Identity', color='#2980b9')
    ax1.tick_params(axis='y', labelcolor='#e74c3c')
    ax2.tick_params(axis='y', labelcolor='#2980b9')

    lines = [l1, l2, l3]
    ax1.legend(lines, [l.get_label() for l in lines], loc='center left')
    ax1.set_title(f'Structural Drift vs Topological Persistence (n={len(valid)})')
    ax1.grid(True, alpha=0.3)
    fig.savefig(OUT / 'fig_rmsd_vs_knot.pdf')
    fig.savefig(OUT / 'fig_rmsd_vs_knot.png')
    plt.close()
    print(f"  fig_rmsd_vs_knot.pdf")


def fig4_contiguous_vs_random():
    """Contiguous vs random masking comparison."""
    with open('results/contiguous_masking.json') as f:
        data = json.load(f)
    results = data['results']
    levels = [25, 50, 75, 85, 90]

    rand_means, cont_means = [], []
    for lev in levels:
        rand = [r['levels'][str(lev)]['avg_knotted_p'] for r in results
                if r.get('mode') == 'random' and r.get('levels') and str(lev) in r['levels']]
        cont = [r['levels'][str(lev)]['avg_knotted_p'] for r in results
                if r.get('mode') == 'contiguous' and r.get('levels') and str(lev) in r['levels']]
        rand_means.append(np.mean(rand) if rand else 0)
        cont_means.append(np.mean(cont) if cont else 0)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(levels))
    w = 0.35
    ax.bar(x - w/2, rand_means, w, label='Random masking', color='#2980b9', alpha=0.8)
    ax.bar(x + w/2, cont_means, w, label='Contiguous masking', color='#e74c3c', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{l}%' for l in levels])
    ax.set_xlabel('Masking percentage')
    ax.set_ylabel('Mean knot probability')
    ax.set_title('Random vs Contiguous Masking')
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')
    fig.savefig(OUT / 'fig_contiguous_vs_random.pdf')
    fig.savefig(OUT / 'fig_contiguous_vs_random.png')
    plt.close()
    print(f"  fig_contiguous_vs_random.pdf")


def fig5_knot_types():
    """Knot type distribution from guided generation."""
    with open('results/guided_gen_combined.json') as f:
        data = json.load(f)

    topos = Counter()
    for r in data['results']:
        if r['is_knotted']:
            for k, v in r.get('topology', {}).items():
                if k != '0_1':
                    topos[k] += 1

    # Sort by frequency, keep top 10
    top = topos.most_common(10)
    labels = [t[0] for t in top]
    counts = [t[1] for t in top]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    ax.barh(range(len(labels)), counts, color=colors, edgecolor='white')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Occurrences')
    ax.set_title(f'Knot Types in Generated Proteins (n={data["n_knotted"]} knotted)')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    fig.savefig(OUT / 'fig_knot_types.pdf')
    fig.savefig(OUT / 'fig_knot_types.png')
    plt.close()
    print(f"  fig_knot_types.pdf")


def fig6_transition_sharpness():
    """Individual protein transition curves overlaid."""
    with open('results/transition_analysis.json') as f:
        data = json.load(f)

    levels = data['levels']
    proteins = data['transition_data']

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot individual curves with low alpha
    for p in proteins:
        ax.plot(levels, p['curve'], color='#2980b9', alpha=0.1, linewidth=0.8)

    # Overlay mean + std
    avg = [data['avg_curve'][str(l)] for l in levels]
    std = [data['std_curve'][str(l)] for l in levels]
    ax.fill_between(levels, np.array(avg) - np.array(std),
                     np.array(avg) + np.array(std), alpha=0.3, color='#e74c3c')
    ax.plot(levels, avg, 'o-', color='#e74c3c', linewidth=2.5, markersize=6, label='Mean ± std')
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7)

    ax.set_xlabel('Sequence masking (%)')
    ax.set_ylabel('Knot probability')
    ax.set_title(f'Individual Protein Transition Curves (n={data["n_proteins"]})')
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.savefig(OUT / 'fig_transition_curves.pdf')
    fig.savefig(OUT / 'fig_transition_curves.png')
    plt.close()
    print(f"  fig_transition_curves.pdf")


def fig7_knot_core_analysis():
    """Knot core size vs breaking point scatter."""
    with open('results/knot_location_analysis.json') as f:
        data = json.load(f)

    proteins = data['proteins']
    core_fracs = [p['core_fraction'] * 100 for p in proteins]
    bps = [p['breaking_point'] for p in proteins]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(core_fracs, bps, alpha=0.6, color='#2980b9', s=40, edgecolors='white', linewidth=0.5)

    # Add trend line
    z = np.polyfit(core_fracs, bps, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(core_fracs), max(core_fracs), 100)
    ax.plot(x_line, p(x_line), '--', color='#e74c3c', linewidth=1.5,
            label=f'r = {data["corr_core_size_vs_bp"]:.3f}')

    ax.set_xlabel('Knot core (% of sequence)')
    ax.set_ylabel('Breaking point (masking %)')
    ax.set_title(f'Knot Core Size vs Breaking Point (n={data["n_proteins"]})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(OUT / 'fig_knot_core_vs_bp.pdf')
    fig.savefig(OUT / 'fig_knot_core_vs_bp.png')
    plt.close()
    print(f"  fig_knot_core_vs_bp.pdf")


if __name__ == '__main__':
    print("Generating figures...")
    fig1_masking_curve()
    fig2_breaking_histogram()
    fig3_rmsd_vs_knot()
    fig4_contiguous_vs_random()
    fig5_knot_types()
    fig6_transition_sharpness()
    fig7_knot_core_analysis()
    print(f"\nAll figures saved to {OUT}/")
