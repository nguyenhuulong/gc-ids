import json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
N = json.load(open(sys.argv[1]))
out = sys.argv[2]
plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans'})
col = {'GAN': '#0072B2', 'VAE': '#D55E00'}
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
for ax, fam, lab, unit in ((axes[0], 'GAN', 'cGAN', 'iterations'), (axes[1], 'VAE', 'cVAE', 'epochs')):
    rows = N['dial'][fam]
    for r in rows:
        xs = list(r['per_seed_mmd2'].values()); ys = [r['per_seed_macro_f1'][s] for s in r['per_seed_mmd2']]
        ax.scatter(xs, ys, s=10, color=col[fam], alpha=0.35, linewidths=0)
    mx = [r['mmd2'][0] for r in rows]; my = [r['macro_f1'][0] for r in rows]
    ax.errorbar(mx, my, xerr=[r['mmd2'][1] for r in rows], yerr=[r['macro_f1'][1] for r in rows], fmt='o-', color=col[fam],
                capsize=2, lw=1.2, ms=4)
    offs = [(6, 6), (6, -12), (-22, 8), (6, 6)] if fam == 'VAE' else [(6, 6)] * 4
    if fam == 'VAE':
        offs = [(6, 6), (-30, -14), (8, -14), (-26, 8)]
    for (r, x, y), o in zip(zip(rows, mx, my), offs):
        ax.annotate(f"{r['ckpt']}", (x, y), textcoords='offset points', xytext=o, fontsize=8)
    c = [c for c in N['dial_corr'] if c['family'] == fam and c['outcome'] == 'macro_f1'][0]
    ax.set_xscale('log')
    ax.set_xlabel('Generator MMD² (log scale)')
    ax.set_title(f"{lab} (checkpoint = {unit})\nSpearman ρ = {c['spearman']:.2f}, p = {c['p']:.2f}", fontsize=8.5)
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    tk = [1e-3, 3e-3, 1e-2, 3e-2]
    ax.xaxis.set_major_locator(FixedLocator(tk)); ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v:g}'))
    ax.set_xlim(9e-4, 4e-2) if fam == 'GAN' else (ax.set_xlim(1.2e-2, 3.6e-2), ax.xaxis.set_major_locator(FixedLocator([0.015, 0.02, 0.03])))
    ax.set_ylim(0.50, 0.62)
    ax.grid(alpha=0.25, lw=0.5)
axes[0].set_ylabel('Macro-F1 of C2 (MLP probe)')
fig.tight_layout()
fig.savefig(out, dpi=300)
print('saved', out)
