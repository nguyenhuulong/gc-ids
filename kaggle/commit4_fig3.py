# Fig. 3 (real versus generated samples): retrains cVAE and cGAN with seed 42 for each dataset and draws the figures.
# Run after cloning the repository on Kaggle (GPU on, Internet on) and linking the data (see README.md of this folder).
import os, subprocess
os.chdir('/kaggle/working/gc-ids')
os.makedirs('results/figures', exist_ok=True)
for ds in ['unswnb15', 'nslkdd']:
    cmd = ('python -u -m gcids.figures_gen --dataset %s --config configs/base.yaml configs/%s.yaml configs/fast.yaml '
           '--seed 42 --cache cache --results results --out results/figures --allow-train --device cuda 2>&1 | tail -15') % (ds, ds)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(ds, r.stdout, r.stderr[-500:])
print(subprocess.run('ls -la results/figures', shell=True, capture_output=True, text=True).stdout)
