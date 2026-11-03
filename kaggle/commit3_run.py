import subprocess, time, os, glob
os.chdir('/kaggle/working/gc-ids')
os.makedirs('logs', exist_ok=True)
SEEDS = '42 123 456 789 1011 2024'
BASE = 'S0 S1 C1 C1b-noise C1b-mixup C2-VAE C2-GAN SupCon C3-realpair GC-mask-GAN GC-mask-VAE SCARF LightGBM FT-Transformer GenQ-VAE GenQ-GAN'
UN = 'python -u -m gcids.run --dataset unswnb15 --config configs/base.yaml configs/unswnb15.yaml configs/fast.yaml'
NS = 'python -u -m gcids.run --dataset nslkdd --config configs/base.yaml configs/nslkdd.yaml configs/fast.yaml'
cmds = {}
for sh in range(2):
    cmds['unsw_s%d' % sh] = ('CUDA_VISIBLE_DEVICES=%d ' % sh + UN + ' --seeds ' + SEEDS + ' --variants ' + BASE +
                             ' --device cuda --shard %d/2 --out results/raw/unswnb15_s%d.jsonl' % (sh, sh))
cmds['nsl_s1'] = ('CUDA_VISIBLE_DEVICES=0 ' + NS + ' --seeds ' + SEEDS + ' --variants S1'
                  ' --device cuda --out results/raw/nslkdd_S1.jsonl')
cmds['tabnet_unsw'] = ('CUDA_VISIBLE_DEVICES=1 ' + UN.replace('configs/fast.yaml', 'configs/extra_baselines.yaml configs/fast.yaml')
                      + ' --seeds ' + SEEDS + ' --variants TabNet --device cuda --out results/raw/tabnet_unswnb15.jsonl')
jobs = []
for name, cmd in cmds.items():
    jobs.append(subprocess.Popen(cmd, shell=True, stdout=open('logs/' + name + '.log', 'w'), stderr=subprocess.STDOUT))
t0 = time.time()
while any(p.poll() is None for p in jobs):
    time.sleep(300)
    n = sum(open(f).read().count(' DONE ') for f in glob.glob('logs/*.log'))
    nf = sum(open(f).read().count('FAILED') + open(f).read().count('Traceback') for f in glob.glob('logs/*.log'))
    print('%.0f min: %d runs done, %d failed' % ((time.time() - t0) / 60, n, nf), flush=True)
    subprocess.run('zip -rqu /kaggle/working/results_backup3.zip results logs', shell=True)
subprocess.run('zip -rq /kaggle/working/results_backup3.zip results logs', shell=True)
for f in sorted(glob.glob('logs/*.log')):
    t = open(f).read()
    print(f, 'DONE:', t.count(' DONE '), 'FAILED:', t.count('FAILED'))
    if 'Traceback' in t:
        print(t[t.find('Traceback'):][:1500])
