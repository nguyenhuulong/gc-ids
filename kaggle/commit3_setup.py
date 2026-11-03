import os, shutil, subprocess, glob
import pandas as pd
src = '/kaggle/input/datasets/<kaggle-user>/gc-ids'
roots = [os.path.dirname(p) for p in glob.glob(src + '/**/gcids', recursive=True)]
assert roots, 'gcids not found in the gc-ids dataset'
shutil.copytree(roots[0], '/kaggle/working/gc-ids', dirs_exist_ok=True)
os.chdir('/kaggle/working/gc-ids')
subprocess.run('pip install -q lightgbm pyyaml', shell=True)
subprocess.run('pip install -q --no-deps pytorch-tabnet', shell=True)
r = subprocess.run('python scripts/link_kaggle_data.py', shell=True, capture_output=True, text=True)
print(r.stdout, r.stderr[-800:])
open('configs/fast.yaml', 'w').write(
    "pretrain: {epochs: 50}\n"
    "vae: {epochs: 100, snapshots: [10, 25, 50, 100]}\n"
    "gan: {iters: 8000, snapshots: [1000, 2000, 4000, 8000]}\n"
    "probe: {probe_epochs: 30}\n"
    "supervised: {epochs: 30}\n"
    "eval: {n_eval: 10000}\n")
open('configs/extra_baselines.yaml', 'w').write("variants:\n  TabNet: {method: TabNet}\nrun_list: [TabNet]\n")
for f in ['UNSW_NB15_training-set.csv', 'UNSW_NB15_testing-set.csv']:
    print(f, len(pd.read_csv('data/unswnb15/' + f, usecols=[0])))
code = open('gcids/data.py').read()
old = '    te = rd("UNSW_NB15_testing-set.csv")\n    if strict_counts:'
new = ('    te = rd("UNSW_NB15_testing-set.csv")\n'
       '    if len(tr) == 82332 and len(te) == 175341:\n'
       '        log.warning("UNSW-NB15 partition files are swapped; using train 175,341 / test 82,332")\n'
       '        tr, te = te, tr\n'
       '    if strict_counts:')
if 'files are swapped' not in code:
    assert old in code, 'patch target not found in data.py'
    open('gcids/data.py', 'w').write(code.replace(old, new))
print('patched:', 'files are swapped' in open('gcids/data.py').read())
r = subprocess.run('python -c "from gcids.data import load_unswnb15 as f; d=f(\'data/unswnb15\'); print(\'UNSW OK\', len(d.train), len(d.test))"',
                   shell=True, capture_output=True, text=True)
print(r.stdout, r.stderr[-800:])
