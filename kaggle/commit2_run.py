import subprocess, time, os, glob
os.chdir('/kaggle/working/gc-ids')
os.makedirs('logs', exist_ok=True)
NS = 'python -u -m gcids.run --dataset nslkdd --config configs/base.yaml configs/nslkdd.yaml'
GAN_VARS = 'C2-GAN C2-GAN-noAux C2-GAN-minority GenQ-GAN GenQ-GAN-noAux GenQ-GAN-minority'
SEEDS6 = '42 123 456 789 1011 2024'
cmds = {}
cmds['dial'] = ('CUDA_VISIBLE_DEVICES=0 ' + NS + ' configs/ablation_dial_fast.yaml configs/fast.yaml'
                ' --seeds 42 123 456 --device cuda --out results/raw/dial_nslkdd.jsonl')
cmds['gan'] = ('CUDA_VISIBLE_DEVICES=1 ' + NS + ' configs/ablation_gan.yaml configs/fast.yaml'
               ' --seeds 456 123 42 --variants ' + GAN_VARS + ' --device cuda --out results/raw/gan_nslkdd.jsonl')
tab1 = ('CUDA_VISIBLE_DEVICES=0 python -u -m gcids.run --dataset nslkdd --config configs/base.yaml configs/nslkdd.yaml'
        ' configs/extra_baselines.yaml configs/fast.yaml --seeds ' + SEEDS6 + ' --variants TabNet --device cuda'
        ' --out results/raw/tabnet_nslkdd.jsonl')
tab2 = ('CUDA_VISIBLE_DEVICES=0 python -u -m gcids.run --dataset unswnb15 --config configs/base.yaml configs/unswnb15.yaml'
        ' configs/extra_baselines.yaml configs/fast.yaml --seeds ' + SEEDS6 + ' --variants TabNet --device cuda'
        ' --out results/raw/tabnet_unswnb15.jsonl')
cmds['tabnet'] = tab1 + ' ; ' + tab2
jobs = []
for name, cmd in cmds.items():
    jobs.append(subprocess.Popen(cmd, shell=True, stdout=open('logs/' + name + '.log', 'w'), stderr=subprocess.STDOUT))
t0 = time.time()
while any(p.poll() is None for p in jobs):
    time.sleep(300)
    n = sum(open(f).read().count(' DONE ') for f in glob.glob('logs/*.log'))
    nf = sum(open(f).read().count('FAILED') for f in glob.glob('logs/*.log'))
    print('%.0f min: %d runs done, %d failed' % ((time.time() - t0) / 60, n, nf), flush=True)
    subprocess.run('zip -rqu /kaggle/working/results_backup2.zip results', shell=True)
print('DONE')
