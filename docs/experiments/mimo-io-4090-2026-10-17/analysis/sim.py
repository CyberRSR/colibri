#!/usr/bin/env python3
"""Replay an IOLOG's routing (each MoE step's union) and the prefetcher's ranked candidates
through the engine's pooled expert cache, with a per-layer bonus on the LRU stamp:
victim = least (used + bonus[layer]). Counts reads per layer in the measured generations.
Usage: sim.py LOG [--bonus L0-L1:B ...] [--topk-pf R] [--cap 104]"""
import argparse, struct, collections

SUBMIT, DEMAND, DEMAND_END, REQ, USED, HIT, PRED_RANK = 1, 5, 6, 13, 14, 21, 23
ALL = False


def load(path):
    raw = open(path, 'rb').read()
    out = []
    for i in range(len(raw) // 32):
        t, k, a, b, c, d, e = struct.unpack_from('<d6i', raw, i * 32)
        if k: out.append((t, k, a, b, c, d, e))
    out.sort(key=lambda x: x[0])
    return out


def steps_from(evs):
    """[(kind, ...)]: ('pred', li, cands[(row, rank, ex)]), ('dem', li, n_rows, union list, measured_gen)"""
    reqs, cur = [], None
    for x in evs:
        if x[1] == REQ:
            if x[2] == 0: cur = [x[0], None, None]
            elif x[2] == 1 and cur: cur[1] = x[0]
            elif x[2] == 2 and cur: cur[2] = x[0]; reqs.append(cur); cur = None
    gens = [(r[1], r[2]) for r in (reqs if ALL else reqs[1:])]
    out = []
    pend = {}
    lastt = {}
    curd = None
    for x in evs:
        t, k, a, b, c, d, e = x
        if k == PRED_RANK:
            if lastt.get(a) is None or t - lastt[a] > 2e-3:
                if a in pend and pend[a]: out.append(('pred', a, pend[a]))
                pend[a] = []
            lastt[a] = t
            pend[a].append((c // 100, c % 100, b))
        elif k == DEMAND:
            # flush a pending prediction for this layer first (it precedes the demand)
            if pend.get(a):
                out.append(('pred', a, pend[a])); pend[a] = []
            lastt[a] = None
            curd = (a, [], t)
        elif k in (USED, HIT) and curd:
            curd[1].append(c)
        elif k == SUBMIT and curd and d == 0:
            curd[1].append(c)
        elif k == DEMAND_END and curd:
            li, U, td = curd; curd = None
            meas = any(g0 <= td < g1 for g0, g1 in gens)
            out.append(('dem', li, U, meas))
    # predictions are emitted lazily: reorder so that each pred precedes its layer's demand.
    return out


class Cache:
    """The engine's pool: cap x layers buffers, a layer holds at most 2 cap; the victim is the
    least used + bonus[layer] slot that is not a standing prefetch (lazy-deletion heap)."""
    def __init__(self, nlayers, cap, bonus, reuse=0.0):
        import heapq
        self.h = heapq
        self.cap, self.max = cap, 2 * cap
        self.total = cap * nlayers
        self.bufs = 0
        self.slots = {}         # (li, e) -> [used, pref, ver]
        self.per = collections.Counter()
        self.bonus = bonus
        self.reuse = reuse
        self.heap = []
        self.lheap = collections.defaultdict(list)
        self.ver = 0

    def push(self, key):
        v = self.slots[key]
        self.ver += 1; v[2] = self.ver
        if not v[1]:
            sc = v[0] + self.bonus[key[0]] + (self.reuse if len(v) > 3 and v[3] >= 2 else 0) if v[0] else 0
            self.h.heappush(self.heap, (sc, self.ver, key))
            self.h.heappush(self.lheap[key[0]], (v[0], self.ver, key))

    def pop_valid(self, heap):
        while heap:
            sc, ver, key = self.h.heappop(heap)
            v = self.slots.get(key)
            if v and v[2] == ver and not v[1]:
                return key
        return None

    def take(self, li):
        if self.bufs < self.total and self.per[li] < self.max:
            self.bufs += 1
            return True
        key = self.pop_valid(self.lheap[li] if self.per[li] >= self.max else self.heap)
        if key is None: return False
        del self.slots[key]; self.per[key[0]] -= 1
        return True

    def put(self, li, ex, used, pref):
        self.slots[(li, ex)] = [used, pref, 0, 0 if pref else 1]; self.per[li] += 1
        self.push((li, ex))


def run(steps, nl, cap, bonus, pf_rank=8, reuse=0.0):
    c = Cache(nl, cap, bonus, reuse)
    reads = collections.Counter(); dem = collections.Counter(); waste = collections.Counter(); used = collections.Counter()
    clock = 0
    for st in steps:
        if st[0] == 'pred':
            li, cands = st[1], st[2]
            order, seen, votes = [], set(), collections.Counter()
            for row, rank, ex in cands:
                if rank >= pf_rank: continue
                if ex not in seen: seen.add(ex); order.append(ex)
                votes[ex] += 1
            order.sort(key=lambda e: -votes[e])
            room = cap // 2
            for ex in order:
                if room <= 0: break
                if (li, ex) in c.slots: continue
                if not c.take(li): break
                c.put(li, ex, 0, 1); room -= 1
        else:
            li, U, meas = st[1], st[2], st[3]
            clock += 1
            for ex in U:
                v = c.slots.get((li, ex))
                if v:
                    if v[1] and meas: used[li] += 1; reads[li] += 1
                    v[0] = clock; v[1] = 0; v[3] += 1; c.push((li, ex))
            for key in [k for k, v in c.slots.items() if k[0] == li and v[1]]:
                v = c.slots[key]; v[1] = 0; v[0] = 0; c.push(key)
                if meas: waste[li] += 1; reads[li] += 1
            for ex in U:
                if (li, ex) in c.slots: continue
                if not c.take(li): continue
                c.put(li, ex, clock, 0)
                if meas: dem[li] += 1; reads[li] += 1
    return reads, dem, waste, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log')
    ap.add_argument('--cap', type=int, default=104)
    ap.add_argument('--bonus', action='append', default=[], help='L0-L1:steps')
    ap.add_argument('--pf-rank', type=int, default=8)
    ap.add_argument('--all', action='store_true', help='count every request (varied prompts)')
    ap.add_argument('--reuse', type=float, default=0.0, help='steps of bonus for slots used twice or more')
    a = ap.parse_args()
    global ALL
    ALL = a.all
    evs = load(a.log)
    steps = steps_from(evs)
    nl, E = 70, 384
    moe_layers = sorted(set(s[1] for s in steps if s[0] == 'dem'))
    bonus = collections.defaultdict(int)
    for spec in a.bonus:
        rng, b = spec.split(':')
        l0, l1 = map(int, rng.split('-'))
        for li in range(l0, l1 + 1): bonus[li] = float(b) * 69
    reads, dem, waste, used = run(steps, len(moe_layers), a.cap, bonus, a.pf_rank, a.reuse * 69)
    early = [l for l in moe_layers if l <= 32]; late = [l for l in moe_layers if l > 32]
    for name, grp in (('early 1-32', early), ('late 33-69', late)):
        print(f"{name}: reads {sum(reads[l] for l in grp)} (demand {sum(dem[l] for l in grp)}, "
              f"prefetch used {sum(used[l] for l in grp)}, wasted {sum(waste[l] for l in grp)})")
    print(f"total reads {sum(reads.values())}")


if __name__ == '__main__':
    main()
