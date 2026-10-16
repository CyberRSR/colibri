#!/usr/bin/env python3
"""Replay a MiMo routing trace (MIMO_ROUTE_TRACE) through expert-cache policies.

The engine writes every MoE step's routing; this replays it offline, so a cache
policy or a prefetch rule can be compared without running the model again:

  - how often the experts a step chooses are the ones layer li's router ranks
    first from the residual at the two points the prefetcher can use (after the
    previous layer's attention, and after its MLP), for several prediction sizes;
  - the hit rate of each cache policy at a given capacity per layer, counted
    the way the engine counts (every expert of a step's union, once);
  - the reads a prefetch rule would add (wrong predictions) and save.

Nothing here is timing: it is the cache's and the predictor's arithmetic.
Usage: mimo_route_sim.py TRACE [--cap N ...] [--skip-steps N]
"""
import argparse
import struct
import sys
from collections import OrderedDict, defaultdict


def read_trace(path):
    data = open(path, "rb").read()
    magic, L, E, K, R = struct.unpack_from("<5i", data, 0)
    if magic != 0x3154524D:
        sys.exit("not a MiMo routing trace")
    off, steps = 20, []
    while off + 12 <= len(data):
        li, n, pos = struct.unpack_from("<3i", data, off); off += 12
        sel = struct.unpack_from(f"<{n * K}h", data, off); off += 2 * n * K
        pred = None
        if n == 1:
            have = struct.unpack_from("<2h", data, off); off += 4
            ra = struct.unpack_from(f"<{R}h", data, off); off += 2 * R
            rb = struct.unpack_from(f"<{R}h", data, off); off += 2 * R
            pred = (ra if have[0] else None, rb if have[1] else None)
        steps.append((li, n, pos, sel, pred))
    return (L, E, K, R), steps


def union(sel):
    seen, out = set(), []
    for e in sel:
        if e not in seen:
            seen.add(e); out.append(e)
    return out


def prediction_report(steps, K):
    ks = [8, 10, 12, 16, 24, 32]
    tot = 0
    hit = {("A", k): 0 for k in ks} | {("B", k): 0 for k in ks} | {("AB", k): 0 for k in ks}
    for li, n, pos, sel, pred in steps:
        if n != 1 or not pred or pred[0] is None or pred[1] is None:
            continue
        actual = set(sel)
        tot += len(actual)
        for k in ks:
            a, b = set(pred[0][:k]), set(pred[1][:k])
            hit[("A", k)] += len(actual & a)
            hit[("B", k)] += len(actual & b)
            hit[("AB", k)] += len(actual & (a | b))
    if not tot:
        print("no one-row steps with predictions")
        return
    print(f"prediction recall over {tot} decode choices (fraction of a step's experts in the top-k):")
    for src, name in (("A", "after the previous attention"), ("B", "after the previous MLP"), ("AB", "union of both")):
        print(f"  {name:30s} " + "  ".join(f"k={k}: {100 * hit[(src, k)] / tot:5.1f}%" for k in ks))


class EngineCache:
    """One layer's cache as the engine keeps it: every expert of a step gets the
    step's stamp; a miss takes a slot that is not in the step -- the least recently
    used (decay 0), or, with MIMO_CACHE_DECAY, the one whose expert has the lowest
    decayed use count (multiplied by the decay at each of the layer's steps), the
    least recently used among equals."""
    def __init__(self, cap, decay=0.0):
        self.cap, self.decay = cap, decay
        self.used = {}                      # resident expert -> stamp
        self.freq, self.ftime, self.t = defaultdict(float), defaultdict(int), 0
        self.clock = 0

    def f(self, e):
        return self.freq[e] * self.decay ** (self.t - self.ftime[e])

    def step(self, experts, li):
        self.clock += 1
        self.t += 1
        if self.decay > 0:
            for e in experts:
                self.freq[e] = self.f(e) + 1.0
                self.ftime[e] = self.t
        hits, misses = 0, []
        for e in experts:
            if e in self.used:
                hits += 1
                self.used[e] = self.clock
            else:
                misses.append(e)
        protect = set(experts)
        for e in misses:
            if len(self.used) >= self.cap:
                # a step larger than the cache: the engine reuses the slots of the
                # step's experts it has already computed
                cand = [v for v in self.used if v not in protect] or list(self.used)
                if self.decay > 0:
                    victim = min(cand, key=lambda v: (self.f(v), self.used[v]))
                else:
                    victim = min(cand, key=lambda v: self.used[v])
                del self.used[victim]
            self.used[e] = self.clock
        return hits


class GlobalLRU:
    """One LRU across all layers, cap * layers slots in all (layers share the RAM)."""
    def __init__(self, total): self.total, self.d = total, OrderedDict()
    def step(self, experts, li):
        hits = 0
        for e in experts:
            if (li, e) in self.d:
                hits += 1; self.d.move_to_end((li, e))
        protect = {(li, e) for e in experts}
        for e in experts:
            if (li, e) in self.d:
                continue
            if len(self.d) >= self.total:
                victim = next((v for v in self.d if v not in protect), next(iter(self.d)))
                del self.d[victim]
            self.d[(li, e)] = 1
        return hits


def belady(steps, cap, L):
    """The offline optimum per layer: evict the resident used furthest in the future."""
    future = defaultdict(lambda: defaultdict(list))
    seq = defaultdict(list)
    for idx, (li, n, pos, sel, pred) in enumerate(steps):
        seq[li].append(union(sel))
    total = hits = 0
    per = {}
    for li, s in seq.items():
        nxt = defaultdict(list)
        for t, ex in enumerate(s):
            for e in ex:
                nxt[e].append(t)
        ptr = defaultdict(int)
        res = set(); h = 0
        for t, ex in enumerate(s):
            for e in ex:
                ptr[e] += 1
            h += sum(1 for e in ex if e in res)
            protect = set(ex)
            for e in ex:
                if e in res:
                    continue
                if len(res) >= cap:
                    def next_use(v):
                        lst = nxt[v]; i = ptr[v]
                        return lst[i] if i < len(lst) else 1 << 30
                    victim = max([v for v in res if v not in protect] or list(res), key=next_use)
                    res.discard(victim)
                res.add(e)
            total += len(ex)
        hits += h
    return hits, total


def cache_report(steps, L, caps, skip, decays):
    print(f"cache hit rate per policy (steps after the first {skip} counted):")
    nmoe = len({s[0] for s in steps})
    for cap in caps:
        print(f"  cap {cap}:")
        policies = [("LRU (engine default)", lambda: EngineCache(cap))]
        policies += [(f"decayed count {d}", (lambda d: lambda: EngineCache(cap, d))(d)) for d in decays]
        shared = GlobalLRU(cap * nmoe)
        policies.append(("one LRU for all layers", lambda: shared))
        for name, make in policies:
            caches = defaultdict(make)
            h = tot = dh = dtot = 0
            for idx, (li, n, pos, sel, pred) in enumerate(steps):
                ex = union(sel)
                got = caches[li].step(ex, li)
                if idx >= skip:
                    h += got; tot += len(ex)
                    if n == 1:
                        dh += got; dtot += len(ex)
            print(f"    {name:24s} {100 * h / max(tot, 1):5.1f}% (decode steps {100 * dh / max(dtot, 1):5.1f}%)")
        bh, bt = belady(steps, cap, L)
        print(f"    {'Belady optimum':24s} {100 * bh / max(bt, 1):5.1f}% (every step, cold start)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace")
    ap.add_argument("--cap", type=int, action="append", default=[])
    ap.add_argument("--skip-steps", type=int, default=0, help="warm the caches on this many steps before counting")
    ap.add_argument("--decay", type=float, action="append", default=[], help="MIMO_CACHE_DECAY values to compare")
    a = ap.parse_args()
    (L, E, K, R), steps = read_trace(a.trace)
    print(f"{len(steps)} MoE steps, {L} layers, {E} experts, top-{K}")
    prediction_report(steps, K)
    cache_report(steps, L, a.cap or [104], a.skip_steps, a.decay or [0.9, 0.97, 0.99, 0.995])


if __name__ == "__main__":
    main()
