"""Ten-seed paired tests, longer-budget runs and C2-GAN-noAux on UNSW-NB15 (commit 4).
Usage (from the repository root):  python analysis/calc_ext.py results_archive analysis/outputs/numbers_ext.json"""
import json, shutil, sys
from pathlib import Path
import numpy as np, pandas as pd
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[1]))   # repository root
from gcids import stats as S
SP = Path(sys.argv[1]); OUT = Path(sys.argv[2])
c1, c3, c4 = SP/'commit1/raw', SP/'commit3/raw', SP/'commit4/raw'

def sub(name, files):
    d = SP/'_work'/name
    if d.exists(): shutil.rmtree(d)
    (d/'raw').mkdir(parents=True)
    for f in files: shutil.copy(f, d/'raw'/Path(f).name)
    return d

CORE = ['S0','C1','C1b-noise','SupCon','C3-realpair','SCARF','C2-VAE','C2-GAN','GC-mask-GAN']
mets = ['macro_f1','g_mean','mcc','rare_f1','rare_recall','silhouette','davies_bouldin']
N = {}
for ds, main, ext in [('nslkdd', [c1/'nslkdd_s0.jsonl', c1/'nslkdd_s1.jsonl'], [c4/'ext_nsl_a.jsonl', c4/'ext_nsl_b.jsonl']),
                      ('unswnb15', [c3/'unswnb15_s0.jsonl', c3/'unswnb15_s1.jsonl'], [c4/'ext_unsw_a.jsonl', c4/'ext_unsw_b.jsonl'])]:
    d = sub('ext10_'+ds, main+ext)
    df = S.load_results(d)
    df = df[df.variant.isin(CORE)]
    mlp = df[df.probe=='mlp']
    seeds = {v: sorted(mlp[mlp.variant==v].seed.unique().tolist()) for v in CORE}
    allseeds = sorted(set.intersection(*[set(s) for s in seeds.values()]))
    others = [v for v in CORE if v != 'C2-GAN']
    pt = S.paired_table(df, ds, [('C2-GAN', o) for o in others], mets, 'mlp')
    agg = S.aggregate(df, ds, mets, 'mlp')
    # restrict aggregate to common seeds
    dfc = df[df.seed.isin(allseeds)]
    aggc = S.aggregate(dfc, ds, mets, 'mlp')
    N[ds] = {'seeds': allseeds, 'n_per_variant': {v: len(s) for v, s in seeds.items()},
             'agg': {v: {m: [float(aggc.loc[v, m]), float(aggc.loc[v, m+'_std'])] for m in mets} for v in aggc.index},
             'paired': pt.to_dict('records')}
    # also other pairs of interest (controls)
    pairs2 = [('C3-realpair','C1b-noise'),('SupCon','C1b-noise'),('C1','C1b-noise'),('SCARF','C1b-noise'),('C2-GAN','C2-VAE'),('C2-GAN','C3-realpair'),('C2-GAN','SupCon'),('C2-GAN','GC-mask-GAN')]
    N[ds]['paired_other'] = S.paired_table(df, ds, pairs2, mets, 'mlp').to_dict('records')

# ---- long budget on UNSW (seeds 42,123,456 from unsw_long_noaux; 789,1011,2024 from unsw_long_b)
dl = sub('long_unsw', [c4/'unsw_long_noaux.jsonl', c4/'unsw_long_b.jsonl', c3/'unswnb15_s0.jsonl', c3/'unswnb15_s1.jsonl', c3/'tabnet_unswnb15.jsonl'])
dfl = S.load_results(dl)
dfl = dfl[dfl.dataset=='unswnb15']
mlp = dfl[dfl.probe=='mlp']

main_only = sub('main_unsw_only', [c3/'unswnb15_s0.jsonl', c3/'unswnb15_s1.jsonl', c3/'tabnet_unswnb15.jsonl'])
dm = S.load_results(main_only); dm = dm[dm.probe=='mlp']
lg = S.load_results(sub('long_only', [c4/'unsw_long_noaux.jsonl', c4/'unsw_long_b.jsonl'])); lg = lg[lg.probe=='mlp']
LM = ['accuracy','macro_f1','g_mean','mcc','rare_recall','rare_f1']
L = {}
def ms(df, v, m):
    x = df[df.variant==v][m]; return [float(x.mean()), float(x.std(ddof=1)) if len(x)>1 else None, int(len(x))]
for v, df in [('S0', dm), ('C2-GAN', dm), ('LightGBM', dm), ('S0-long', lg), ('C2-GAN-long', lg)]:
    L[v] = {m: ms(df, v, m) for m in LM}
# paired long vs short on same seeds (6)
both = pd.concat([dm.assign(src='main'), lg.assign(src='long')])
both = both.drop_duplicates(['seed','variant'], keep='last')
both['dataset'] = 'unswnb15'
pr = []
for A, B in [('S0-long','S0'),('C2-GAN-long','C2-GAN'),('C2-GAN-long','S0-long'),('C2-GAN','S0'),('C2-GAN-long','LightGBM'),('S0-long','LightGBM')]:
    pr.append(S.paired_table(both, 'unswnb15', [(A,B)], ['macro_f1','g_mean','mcc','rare_f1','rare_recall'], 'mlp'))
N['long'] = {'agg': L, 'paired': pd.concat(pr).to_dict('records')}
# noAux (seeds 42,123,456; same run)
na = lg[lg.variant.isin(['C2-GAN','C2-GAN-noAux'])].assign(dataset='unswnb15')
N['noaux'] = {'agg': {v: {m: ms(na, v, m) for m in LM} for v in ['C2-GAN','C2-GAN-noAux']},
              'paired': S.paired_table(na, 'unswnb15', [('C2-GAN-noAux','C2-GAN')], ['macro_f1','g_mean','mcc','rare_f1','rare_recall'], 'mlp').to_dict('records')}
# generator metrics noAux
gq = S.load_genq(sub('noaux_genq', [c4/'unsw_long_noaux.jsonl']))
gq = gq[gq.scope=='overall']
keys = ['mmd2','wasserstein','corr_diff','precision','recall','density','coverage','c2st_auc','memorisation']
N['noaux']['genq'] = {v: {k: [float(g[k].mean()), float(g[k].std(ddof=1)), int(len(g))] for k in keys} for v, g in gq.groupby('variant')}
# replicates C2-GAN seeds 42,123,456: main run vs w3 run
rep = {}
for v in ['C2-GAN']:
    a = dm[dm.variant==v].set_index('seed'); b = lg[lg.variant==v].set_index('seed')
    rep = {int(s): {m: [float(a.loc[s,m]), float(b.loc[s,m])] for m in ['macro_f1','rare_f1','mcc','g_mean']} for s in b.index}
N['replicate_c2gan_unsw'] = rep
json.dump(N, open(OUT,'w'), indent=1, default=float)
print('ok')
