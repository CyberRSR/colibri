"""spans.py <dump prefix> <anchors.npz>: the positions of the replayed transcripts that the model
itself wrote (inside <|open|>message role="assistant" ... <|end_of_msg|>, after the first inner
tag's <|sep|>), each with its message's last position and its channel (the innermost open tag:
think, response, a tool call's), and how often the head's first id at a position is the token the
record has next."""
import collections, sys
import numpy as np
from tokenizers import Tokenizer

DUMP, OUT = sys.argv[1], sys.argv[2]
OPEN, CLOSE, SEP, EOM = 163587, 163588, 163589, 163586
tok = Tokenizer.from_file('C:/AI/Kimi-K3.1-Pro/tokenizer.json')
steps = [tuple(map(int, l.split())) for l in open(DUMP + '.steps.txt') if l.strip()]
ids_all = np.fromfile(DUMP + '.ids.i32', dtype=np.int32)
top_all = np.fromfile(DUMP + '.top1.i32', dtype=np.int32)
trans, row = [], 0
for pos0, C in steps:
    if pos0 == 0: trans.append([row, 0])
    assert pos0 == trans[-1][1]
    trans[-1][1] += C; row += C

tid_l, pos_l, end_l, chan_l = [], [], [], []
names = collections.Counter()
for tid, (r0, N) in enumerate(trans):
    ids = ids_all[r0:r0 + N]
    N = min(N, len(top_all) - r0)
    top = top_all[r0:r0 + N]
    stack, i, in_asst, gen_from, msg_pos = [], 0, False, -1, []
    chan = np.full(N, '', dtype=object)
    gen = np.zeros(N, dtype=bool)
    while i < N:
        t = int(ids[i])
        if t == OPEN or t == CLOSE:
            j = i + 1
            while j < N and int(ids[j]) not in (SEP, OPEN, CLOSE, EOM): j += 1
            name = tok.decode([int(x) for x in ids[i + 1:j]]).strip()
            tag = name.split(' ')[0] if name else '?'
            if t == OPEN:
                stack.append(tag)
                if tag == 'message':
                    in_asst = 'role="assistant"' in name
                    gen_from = -1
                elif in_asst and gen_from < 0:
                    gen_from = j + 1          # after the first inner tag's <|sep|>
                names[(in_asst, tag)] += 1
            else:
                while stack and stack[-1] != tag: stack.pop()
                if stack: stack.pop()
        if in_asst and gen_from >= 0 and i >= gen_from:
            gen[i] = True
            chan[i] = next((s for s in reversed(stack) if s != 'message'), 'message')
            msg_pos.append(i)
        if t == EOM:
            if in_asst and msg_pos:
                for p in msg_pos:
                    tid_l.append(tid); pos_l.append(p); end_l.append(i); chan_l.append(chan[p])
            in_asst, gen_from, msg_pos, stack = False, -1, [], []
        i += 1
    g = np.nonzero(gen[:N - 1])[0]
    agree = (top[g] == ids[g + 1]).mean() if len(g) else float('nan')
    alln = (top[:N - 1] == ids[1:N]).mean()
    print(f'transcript {tid}: {N} positions, {int(gen.sum())} written by the model; the head\'s first id is the recorded next token at '
          f'{100 * agree:.1f}% of them ({100 * alln:.1f}% over every position)')
np.savez(OUT, tid=np.array(tid_l), pos=np.array(pos_l), end=np.array(end_l), chan=np.array(chan_l, dtype=str))
print('tags seen (inside an assistant message, name):', dict(names.most_common(14)))
print('channels of the model\'s positions:', dict(collections.Counter(chan_l).most_common(10)))
