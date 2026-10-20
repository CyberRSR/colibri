"""The [topk] lines of an engine log with each id's text: topk_show.py ENGINE_LOG"""
import json, sys
tok = json.load(open('D:/AI/Models/MiMo-V2.6-Pro-MOPD/tokenizer.json', encoding='utf-8'))
vocab = {i: t for t, i in tok['model']['vocab'].items()}
for a in tok.get('added_tokens', []): vocab[a['id']] = a['content']
bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('\xa1'), ord('\xac') + 1)) + list(range(ord('\xae'), ord('\xff') + 1))
cs = bs[:]
n = 0
for b in range(256):
    if b not in bs: bs.append(b); cs.append(256 + n); n += 1
back = {chr(c): b for b, c in zip(bs, cs)}
def text(i):
    t = vocab.get(i, '?')
    try: return bytes(back[ch] for ch in t).decode('utf-8', 'replace')
    except KeyError: return t
for line in open(sys.argv[1], encoding='utf-8', errors='replace'):
    if not line.startswith('[topk]'): continue
    f = line.split()[1:]
    ids, lg = [int(x) for x in f[0::2]], [float(x) for x in f[1::2]]
    print(f'{text(ids[0])!r:>14} {lg[0]:9.4f} | {text(ids[1])!r:>14} {lg[1]:9.4f} (gap {lg[0] - lg[1]:.4f}) | {text(ids[2])!r:>14} {lg[2]:9.4f}')
