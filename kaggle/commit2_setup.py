import os, shutil, subprocess, glob
src = '/kaggle/input/datasets/<kaggle-user>/gc-ids'
roots = [os.path.dirname(p) for p in glob.glob(src + '/**/gcids', recursive=True)]
assert roots, 'gcids not found in the gc-ids dataset'
shutil.copytree(roots[0], '/kaggle/working/gc-ids', dirs_exist_ok=True)
os.chdir('/kaggle/working/gc-ids')
r = subprocess.run('pip install -q lightgbm pytest pyyaml', shell=True, capture_output=True, text=True)
print(r.stderr[-300:])
r = subprocess.run('pip install -q --no-deps pytorch-tabnet', shell=True, capture_output=True, text=True)
print(r.stdout[-300:], r.stderr[-300:])
r = subprocess.run('python scripts/link_kaggle_data.py', shell=True, capture_output=True, text=True)
print(r.stdout, r.stderr[-1000:])
FAST = """pretrain: {epochs: 50}
vae: {epochs: 100, snapshots: [10, 25, 50, 100]}
gan: {iters: 8000, snapshots: [1000, 2000, 4000, 8000]}
probe: {probe_epochs: 30}
supervised: {epochs: 30}
eval: {n_eval: 10000}
"""
DIAL = """# Generator-quality dial for the REDUCED budget (use together with configs/fast.yaml, whose snapshots are these checkpoints).
# Analysis: Spearman(MMD, Macro-F1 / Rare-F1 / silhouette) via gcids.stats.dial_correlation. Use >= 3 seeds.
# Note: with pretrain_frac 0.5 the GAN checkpoints at <= 4000 iterations are still in phase 1 (majority classes only), so minority classes
# are poorly generated there; that is exactly the low-quality end of the dial.
variants:
  C2-GAN-it1000: {method: C2-GAN, checkpoint: 1000}
  GenQ-GAN-it1000: {method: GenQ-GAN, checkpoint: 1000}
  C2-GAN-it2000: {method: C2-GAN, checkpoint: 2000}
  GenQ-GAN-it2000: {method: GenQ-GAN, checkpoint: 2000}
  C2-GAN-it4000: {method: C2-GAN, checkpoint: 4000}
  GenQ-GAN-it4000: {method: GenQ-GAN, checkpoint: 4000}
  C2-GAN-it8000: {method: C2-GAN, checkpoint: 8000}
  GenQ-GAN-it8000: {method: GenQ-GAN, checkpoint: 8000}
  C2-VAE-ep10: {method: C2-VAE, checkpoint: 10}
  GenQ-VAE-ep10: {method: GenQ-VAE, checkpoint: 10}
  C2-VAE-ep25: {method: C2-VAE, checkpoint: 25}
  GenQ-VAE-ep25: {method: GenQ-VAE, checkpoint: 25}
  C2-VAE-ep50: {method: C2-VAE, checkpoint: 50}
  GenQ-VAE-ep50: {method: GenQ-VAE, checkpoint: 50}
  C2-VAE-ep100: {method: C2-VAE, checkpoint: 100}
  GenQ-VAE-ep100: {method: GenQ-VAE, checkpoint: 100}
run_list: [C2-GAN-it1000, GenQ-GAN-it1000, C2-GAN-it2000, GenQ-GAN-it2000, C2-GAN-it4000, GenQ-GAN-it4000, C2-GAN-it8000, GenQ-GAN-it8000, C2-VAE-ep10, GenQ-VAE-ep10, C2-VAE-ep25, GenQ-VAE-ep25, C2-VAE-ep50, GenQ-VAE-ep50, C2-VAE-ep100, GenQ-VAE-ep100]
"""
EXTRA = """variants:
  TabNet: {method: TabNet}
run_list: [TabNet]
"""
open('configs/fast.yaml', 'w').write(FAST)
open('configs/ablation_dial_fast.yaml', 'w').write(DIAL)
open('configs/extra_baselines.yaml', 'w').write(EXTRA)
print(sorted(os.listdir('configs')))
