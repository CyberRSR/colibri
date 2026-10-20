"""check_mimo_cuda.py's CUDA run alone, against a saved CPU run: probe.py ENGINE CPUDIR OUT [K=V ...]"""
import json, os, subprocess, sys
import numpy as np
engine, cpudir, out = sys.argv[1], sys.argv[2], sys.argv[3]
extra = dict(kv.partition('=')[::2] for kv in sys.argv[4:])
res = json.load(open(os.path.join(cpudir, 'result.json')))
cfg = dict(res['config']); cfg.update(extra)
prompt = open('C:/colibri/dev/docs/experiments/mimo-gpu-prompt-4090-2026-10-20/correctness/prompt.txt', encoding='utf-8').read()
os.makedirs(out, exist_ok=True)
env = {k: v for k, v in os.environ.items() if not k.startswith(('MIMO_', 'COLI_', 'CUDA_'))}
env.update(cfg); env['MIMO_CUDA'] = '1'; env['MIMO_LOGITS'] = os.path.join(out, 'cuda.f32')
with open(os.path.join(out, 'cuda.log'), 'wb') as log, open(os.path.join(out, 'cuda.txt'), 'wb') as rep:
    subprocess.run([engine, 'D:/AI/Models/MiMo-V2.6-Pro-MOPD', '--cap', '88', '--ngen', str(res['requested_tokens']), '--prompt', prompt],
                   env=env, stdout=rep, stderr=log, check=True, timeout=1800)
V = 152576
cpu = np.fromfile(os.path.join(cpudir, 'cpu.f32'), dtype='<f4').reshape(-1, V).astype(np.float64)
cuda = np.fromfile(os.path.join(out, 'cuda.f32'), dtype='<f4').reshape(-1, V).astype(np.float64)
m = int((cpu.argmax(1) == cuda.argmax(1)).sum())
lp = cpu - cpu.max(1, keepdims=True); lq = cuda - cuda.max(1, keepdims=True)
lp -= np.log(np.exp(lp).sum(1, keepdims=True)); lq -= np.log(np.exp(lq).sum(1, keepdims=True))
kl = (np.exp(lp) * (lp - lq)).sum(1)
bad = np.nonzero(kl > 1e-4)[0]
print(json.dumps({'env': extra, 'argmax': f'{m}/{cpu.shape[0]}', 'max_abs': float(np.abs(cpu - cuda).max()), 'max_kl': float(kl.max()),
                  'first_bad_positions': bad[:8].tolist(), 'bad_count': int(len(bad)),
                  'same_reply': open(os.path.join(cpudir, 'cpu.txt'), 'rb').read() == open(os.path.join(out, 'cuda.txt'), 'rb').read()}))
