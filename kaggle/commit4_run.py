import subprocess, time, os, glob
os.chdir('/kaggle/working/gc-ids')
os.makedirs('logs', exist_ok=True)
os.makedirs('results/raw', exist_ok=True)
CFG = lambda ds: 'python -u -m gcids.run --dataset %s --config configs/base.yaml configs/%s.yaml configs/fast.yaml configs/ext.yaml' % (ds, ds)
CORE = 'S0 C1 C1b-noise SupCon C3-realpair SCARF C2-VAE C2-GAN GC-mask-GAN'
LONG = 'S0-long C2-GAN-long C2-GAN C2-GAN-noAux GenQ-GAN GenQ-GAN-noAux'
FIG = ('python -u -m gcids.figures_gen --dataset %s --config configs/base.yaml configs/%s.yaml configs/fast.yaml '
       '--seed 42 --cache cache --results results --out results/figures --allow-train --device cuda')

def core(ds, seeds, out):
    return CFG(ds) + ' --seeds %s --variants %s --device cuda --out results/raw/%s.jsonl' % (seeds, CORE, out)

cmds = {
    # 4 processes, 2 per GPU
    'w1': 'export CUDA_VISIBLE_DEVICES=0; ' + core('unswnb15', '2025 3407', 'ext_unsw_a') + '; ' + core('nslkdd', '2025 3407', 'ext_nsl_a'),
    'w2': 'export CUDA_VISIBLE_DEVICES=1; ' + core('unswnb15', '7777 31337', 'ext_unsw_b') + '; ' + core('nslkdd', '7777 31337', 'ext_nsl_b'),
    'w3': ('export CUDA_VISIBLE_DEVICES=0; ' + CFG('unswnb15') + ' --seeds 42 123 456 --variants ' + LONG +
           ' --device cuda --out results/raw/unsw_long_noaux.jsonl'),
    'w4': ('export CUDA_VISIBLE_DEVICES=1; ' + CFG('unswnb15') + ' --seeds 789 1011 2024 --variants S0-long C2-GAN-long'
           ' --device cuda --out results/raw/unsw_long_b.jsonl; '
           + FIG % ('unswnb15', 'unswnb15') + '; ' + FIG % ('nslkdd', 'nslkdd')),
}
jobs = []
for name, cmd in cmds.items():
    jobs.append(subprocess.Popen(cmd, shell=True, executable='/bin/bash', stdout=open('logs/' + name + '.log', 'w'), stderr=subprocess.STDOUT))
t0 = time.time()
def count(f):
    t = open(f).read()
    return t.count(' DONE '), t.count('FAILED') + t.count('Traceback')
while any(p.poll() is None for p in jobs):
    time.sleep(300)
    s = {os.path.basename(f)[:-4]: count(f) for f in sorted(glob.glob('logs/*.log'))}
    print('%.0f min | done/failed: %s' % ((time.time() - t0) / 60, s), flush=True)
    subprocess.run('zip -rqu /kaggle/working/results_backup4.zip results logs', shell=True)
subprocess.run('zip -rq /kaggle/working/results_backup4.zip results logs', shell=True)
for f in sorted(glob.glob('logs/*.log')):
    t = open(f).read()
    print(f, 'DONE:', t.count(' DONE '), 'FAILED:', t.count('FAILED'), 'Traceback:', t.count('Traceback'))
    if 'Traceback' in t:
        print(t[t.find('Traceback'):][:1500])
print(subprocess.run('ls results/figures | head -40', shell=True, capture_output=True, text=True).stdout)
