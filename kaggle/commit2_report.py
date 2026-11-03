import subprocess
r = subprocess.run('zip -rq /kaggle/working/results_backup2.zip results logs', shell=True, capture_output=True, text=True, cwd='/kaggle/working/gc-ids')
print(r.stdout[-500:], r.stderr[-500:])
for f in sorted(__import__('glob').glob('/kaggle/working/gc-ids/logs/*.log')):
    t = open(f).read()
    print(f.split('/')[-1], 'DONE:', t.count(' DONE '), 'FAILED:', t.count('FAILED'))
