"""UNSW-NB15 numbers of the manuscript from results_archive/commit3.
Usage (from the repository root):  python analysis/calc_unsw.py results_archive analysis/outputs/numbers_unsw.json
A deterministic function of the raw result files; nothing is tuned or selected here."""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[1]))   # repository root
from gcids import stats as S

SP = Path(sys.argv[1])           # results_archive
OUT = Path(sys.argv[2])
DS = 'unswnb15'


def sub(name, files):
    d = SP / '_work' / name
    if d.exists():
        shutil.rmtree(d)
    (d / 'raw').mkdir(parents=True)
    for f in files:
        shutil.copy(f, d / 'raw' / Path(f).name)
    return d


c3 = SP / 'commit3/raw'
main_dir = sub('main_unsw', [c3 / 'unswnb15_s0.jsonl', c3 / 'unswnb15_s1.jsonl', c3 / 'tabnet_unswnb15.jsonl'])
df = S.load_results(main_dir)
genq = S.load_genq(main_dir)

N = {}
CLASSES = [k[3:] for k in df.columns if k.startswith('f1:')]
N['classes'] = CLASSES
mlp = df[(df.probe == 'mlp')]
lin = df[(df.probe == 'linear')]
CLS = ['accuracy', 'macro_f1', 'g_mean', 'mcc', 'rare_recall', 'rare_f1']
EMB = ['silhouette', 'davies_bouldin', 'intra_inter', 'alignment', 'uniformity', 'knn_acc']
variants = list(dict.fromkeys(df.variant))
N['n_seeds'] = {v: int(mlp[mlp.variant == v].seed.nunique()) for v in variants}
N['seeds'] = sorted(set(mlp.seed))

# ---- classification / embedding means and stds
agg = S.aggregate(df, DS, CLS + ['balanced_accuracy'], 'mlp')
N['cls'] = {v: {m: [float(agg.loc[v, m]), float(agg.loc[v, m + '_std'])] for m in CLS} for v in agg.index}
aggl = S.aggregate(df, DS, CLS, 'linear')
N['cls_linear'] = {v: {m: [float(aggl.loc[v, m]), float(aggl.loc[v, m + '_std'])] for m in CLS} for v in aggl.index}
agge = S.aggregate(df, DS, EMB, 'mlp')
N['emb'] = {v: {m: [float(agge.loc[v, m]), float(agge.loc[v, m + '_std'])] for m in EMB} for v in agge.index if not np.isnan(agge.loc[v, 'silhouette'])}

# per-class F1 for rare classes
pc = {}
for v in variants:
    d = mlp[mlp.variant == v]
    pc[v] = {c: [float(d[f'f1:{c}'].mean()), float(d[f'f1:{c}'].std(ddof=1))] for c in CLASSES if f'f1:{c}' in d}
    pc[v].update({f'recall:{c}': [float(d[f'recall:{c}'].mean()), float(d[f'recall:{c}'].std(ddof=1))] for c in CLASSES if f'recall:{c}' in d})
N['perclass'] = pc

# ---- paired tests vs C2-GAN (family = all listed comparators, per metric)
ref = 'C2-GAN'
others = [v for v in variants if v != ref and N['n_seeds'][v] >= 2]
mets = ['macro_f1', 'g_mean', 'rare_f1', 'rare_recall', 'silhouette', 'davies_bouldin', 'alignment', 'accuracy', 'mcc']
pt = S.paired_table(df, DS, [(ref, o) for o in others], mets, 'mlp')
N['paired_vs_gan'] = pt.to_dict('records')

# paired among other pairs of interest (reading of controls)
pairs2 = [('C3-realpair', 'C1b-noise'), ('SupCon', 'C1b-noise'), ('C1', 'C1b-noise'), ('SCARF', 'C1b-noise'),
          ('C2-GAN', 'C2-VAE'), ('C3-realpair', 'C1'), ('SupCon', 'C1'), ('C1b-noise', 'S0'), ('C1', 'S0')]
pt2 = S.paired_table(df, DS, pairs2, mets, 'mlp')
N['paired_other'] = pt2.to_dict('records')

# ---- generator quality (commit 1, main generators, 6 seeds)
gq = genq[(genq.scope == 'overall') & genq.variant.isin(['GenQ-VAE', 'GenQ-GAN'])]
keys = ['mmd2', 'wasserstein', 'corr_diff', 'precision', 'recall', 'density', 'coverage', 'c2st_auc', 'memorisation', 'diversity_real', 'diversity_fake']
N['genq'] = {v: {k: [float(g[k].mean()), float(g[k].std(ddof=1))] for k in keys} for v, g in gq.groupby('variant')}
N['genq_n'] = {v: int(len(g)) for v, g in gq.groupby('variant')}
# per class MMD2 from raw records
pcm = {}
for p in (c3 / 'unswnb15_s0.jsonl', c3 / 'unswnb15_s1.jsonl'):
    for l in open(p):
        r = json.loads(l)
        if 'genq' in r and r['variant'] in ('GenQ-VAE', 'GenQ-GAN'):
            for c, m in r['genq']['per_class'].items():
                pcm.setdefault(r['variant'], {}).setdefault(c, []).append(m)
N['genq_perclass'] = {v: {c: {k: [float(np.nanmean([m[k] for m in ms])), float(np.nanstd([m[k] for m in ms], ddof=1))]
                              for k in ('mmd2', 'precision', 'recall', 'c2st_auc')}
                          | {'n': int(ms[0]['n'])} for c, ms in d.items()} for v, d in pcm.items()}
# paired GAN vs VAE on pooled mmd
a = gq[gq.variant == 'GenQ-GAN'].set_index('seed').mmd2
b = gq[gq.variant == 'GenQ-VAE'].set_index('seed').mmd2
s_ = sorted(set(a.index) & set(b.index))
N['genq_mmd_pair'] = {'n': len(s_), 'gan_lt_vae_all_seeds': bool(all(a[s] < b[s] for s in s_)),
                      't_p': float(sps.ttest_rel(a[s_], b[s_]).pvalue)}

# ---- cost (commit 1)
cost = {}
for v in ['C1b-noise', 'C2-VAE', 'C2-GAN', 'SupCon', 'LightGBM', 'FT-Transformer', 'TabNet', 'S0']:
    d = mlp[mlp.variant == v]
    g = lambda c: float(d[c].mean()) if c in d and not d[c].isna().all() else None
    cost[v] = {'gen': g('cost:generator_train:seconds'), 'pre': g('cost:pretrain:seconds'), 'probe': g('cost:probe_train:seconds'),
               'train': g('cost:train:seconds'), 'peak': g('cost:pretrain:peak_gpu_mb'), 'params': g('cost:pretrain:n_params'),
               'gen_peak': g('cost:generator_train:peak_gpu_mb'), 'train_peak': g('cost:train:peak_gpu_mb'),
               'train_params': g('cost:train:n_params')}
gp = {}
for p in (c3 / 'unswnb15_s0.jsonl', c3 / 'unswnb15_s1.jsonl'):
    for l in open(p):
        r = json.loads(l)
        if r['variant'] in ('C2-GAN', 'C2-VAE') and 'generator_train' in (r.get('cost') or {}):
            gp.setdefault(r['variant'], []).append(r['cost']['generator_train']['n_params'])
for v, x in gp.items():
    cost[v]['gen_params'] = float(np.mean(x))
N['cost'] = cost

json.dump(N, open(OUT, 'w'), indent=1, default=float)
print('ok', OUT)
