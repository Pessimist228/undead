#!/usr/bin/env python3
"""Make a realistic synthetic two-hour data set in the exact formats report_run.py reads.

    python3 make_fake.py OUTDIR [--seed 7]

Six cards, five of them load; a handful of preemptions (more on low bids); card 6 never loads; between
minutes 67 and 80 only one card is healthy, so the router sheds load with 503; streams cut by a stop are
resumed on another card after about 1.4 s. Instance ids are made up (no real hosts, no addresses).
"""
import argparse
import datetime as dt
import heapq
import json
import os
import random

CAP = 8          # streams per card the router accepts
QUEUE_S = 10     # a new request waits this long for a free slot, then gets 503
T0 = dt.datetime(2026, 11, 19, 13, 39, 28, 633000, tzinfo=dt.timezone.utc).timestamp()
DUR = 7200.0
MIN = 60.0


def iso(t, ms=True):
    d = dt.datetime.fromtimestamp(t, dt.timezone.utc)
    return d.strftime("%Y-%m-%dT%H:%M:%S") + (".%03dZ" % (d.microsecond // 1000) if ms else "Z")


# card plan: (id, premium, rent offset before load [s], load time [s] or None, outages [(stop, back or None)] in s after load start)
PLAN = [
    (61449544, 1.05, -8 * MIN, 6 * MIN, [(18 * MIN, 26 * MIN), (66 * MIN, 88 * MIN)]),
    (61449547, 1.05, -8 * MIN, 8 * MIN, [(34 * MIN, 38 * MIN), (66.5 * MIN, 85 * MIN)]),
    (61449551, 1.15, -8 * MIN, 5 * MIN, [(40 * MIN, 47 * MIN), (67 * MIN, 82 * MIN)]),
    (61449560, 1.15, -8 * MIN, 9 * MIN, [(68 * MIN, 79 * MIN)]),
    (61449566, 1.30, -8 * MIN, 7 * MIN, [(50 * MIN, 50 * MIN + 15), (101 * MIN, 102.5 * MIN)]),
    (61449571, 1.30, -8 * MIN, None, []),
]
DPH = {61449544: 0.286, 61449547: 0.301, 61449551: 0.327, 61449560: 0.322, 61449566: 0.364, 61449571: 0.355}
END = DUR + 40  # destroyed by us right after the load


class Card:
    def __init__(self, cid, prem, rent, load, outages):
        self.id, self.prem = cid, prem
        self.created = T0 + rent
        self.ready = T0 + rent + load if load is not None else None
        # load is counted from the start of the load, not the rent: ready = created + load
        self.outages = [(T0 + a, (T0 + b) if b else None) for a, b in outages]
        self.destroyed = T0 + END

    def at(self, t):
        """(status, hb) at time t."""
        if t < self.created:
            return None
        if t >= self.destroyed:
            return ("absent", "none")
        if self.ready is None or t < self.ready:
            return ("loading", "none") if t > self.created + 30 else ("unknown", "none")
        for a, b in self.outages:
            if a <= t < (b if b else 1e18):
                if t < a + 20:
                    return ("running", "bad")
                if b is None:
                    return ("exited", "none") if t < a + 600 else ("offline", "none")
                if b - a <= 30:       # a probe blip
                    return ("running", "bad")
                return ("exited", "none") if t < b - 110 else ("loading", "none")
        return ("running", "ok")

    def healthy(self, t):
        return self.at(t) == ("running", "ok")

    def next_stop(self, t):
        ns = [a for a, b in self.outages if a > t and (b is None or b - a > 30)]
        return min(ns) if ns else 1e18


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    os.makedirs(a.out, exist_ok=True)
    cards = [Card(*p) for p in PLAN]
    # PLAN gives the load time after the rent; fix: ready = created + load
    for c, p in zip(cards, PLAN):
        c.ready = c.created + p[3] if p[3] is not None else None

    # ---------- requests
    active = []   # (card index, start, end)
    run, journal = [], []

    def load(ci, t):
        return sum(1 for c, s, e in active if c == ci and s <= t < e)

    def pick(t, cap=True):
        hs = [i for i, c in enumerate(cards) if c.healthy(t) and (not cap or load(i, t) < CAP)]
        return min(hs, key=lambda i: (load(i, t), rng.random())) if hs else None

    heap = [(T0 + rng.uniform(0, 4), w) for w in range(24)]
    heapq.heapify(heap)
    counters = [0] * 24
    while heap:
        t, w = heapq.heappop(heap)
        if t >= T0 + DUR:
            continue
        counters[w] += 1
        rid = counters[w]
        ci, q = None, 0.0
        for q in range(0, QUEUE_S + 1):
            ci = pick(t + q)
            if ci is not None:
                break
        if ci is None:
            end = t + QUEUE_S
            run.append(dict(worker=w, request=rid, start=iso(t), ttfc_ms=None, chunks=0, duration_ms=round(QUEUE_S * 1000 + rng.uniform(5, 40), 1),
                            max_gap_ms=0.0, max_gap_end=iso(end), finish_reason=None, status=503, error="HTTPStatusError", chars=0))
            journal.append(dict(id="cmpl-%08x" % rng.getrandbits(32), start=iso(t), end=iso(end), model="fake-model", prompt_tokens=rng.randint(80, 400),
                                output_tokens=0, ttft_ms=None, duration_ms=QUEUE_S * 1000, queued_ms=QUEUE_S * 1000, replicas=[], resumes=[],
                                finish="rejected_503", status=503))
            heapq.heappush(heap, (end + rng.uniform(0.5, 2.0), w))
            continue
        ts0 = t + q
        n = int(min(1024, max(40, rng.lognormvariate(6.4, 0.55))))
        rate = rng.uniform(28, 42)
        ttfc = 0.35 + rng.random() * 0.7 + q
        gen_start = ts0 + ttfc - q
        cur, done, now = ci, 0, gen_start
        names, resumes, max_gap, gap_end = ["i%d" % cards[ci].id], [], rng.uniform(120, 330), None
        cut = False
        while True:
            rem = n - done
            finish = now + rem / rate
            die = cards[cur].next_stop(now)
            if finish <= die:
                active.append((cur, now, finish))
                done, now = n, finish
                break
            made = int((die - now) * rate)
            done += made
            active.append((cur, now, die))
            pause = max(0.6, rng.gauss(1.4, 0.22))
            if rng.random() < 0.04:
                pause += rng.uniform(1.0, 3.0)
            nxt = pick(die + pause, cap=False)
            if nxt is None or rng.random() < 0.03:      # nowhere to continue: the stream is cut
                cut = True
                now = die + pause
                break
            resumes.append(dict(pos=done, pause_ms=round(pause * 1000), **{"from": "i%d" % cards[cur].id}, to="i%d" % cards[nxt].id, reason="stream_error"))
            names.append("i%d" % cards[nxt].id)
            if pause * 1000 > max_gap:
                max_gap, gap_end = pause * 1000, die + pause
            cur, now = nxt, die + pause
        dur = now - ts0
        if cut:
            reason, err, status, jfin = None, "RemoteProtocolError", 200, "error"
        else:
            reason, err, status, jfin = ("length" if n >= 1024 else "stop"), None, 200, ("length" if n >= 1024 else "stop")
        run.append(dict(worker=w, request=rid, start=iso(ts0), ttfc_ms=round(ttfc * 1000, 1), chunks=done, duration_ms=round(dur * 1000, 1),
                        max_gap_ms=round(max_gap, 1), max_gap_end=iso(gap_end or now), finish_reason=reason, status=status, error=err,
                        chars=int(done * 4.1)))
        journal.append(dict(id="cmpl-%08x" % rng.getrandbits(32), start=iso(ts0 - q), end=iso(now), model="fake-model", prompt_tokens=rng.randint(80, 400),
                            output_tokens=done, ttft_ms=round(ttfc * 1000), duration_ms=round((now - ts0 + q) * 1000), queued_ms=round(q * 1000),
                            replicas=names, resumes=resumes, finish=jfin, status=status))
        heapq.heappush(heap, (now + rng.uniform(0.2, 1.0), w))

    run.sort(key=lambda r: r["start"])
    journal.sort(key=lambda r: r["end"])
    wl = lambda name, rows: open(os.path.join(a.out, name), "w").write("".join(json.dumps(r) + "\n" for r in rows))  # noqa: E731
    wl("run.jsonl", run)
    wl("journal.jsonl", journal)
    with open(os.path.join(a.out, "load-start.txt"), "w") as f:
        f.write(iso(T0, ms=False) + "\n")

    # ---------- samples, credit
    samples, credit_rows = [], []
    credit_start = 25.0
    t = min(c.created for c in cards)
    spent = 0.0
    while t <= T0 + END:
        for c in cards:
            s = c.at(t)
            if s is None:
                continue
            row = {"type": "sample", "t": iso(t, False), "ts": t, "id": c.id, "status": s[0], "hb": s[1],
                   "intended_status": "running", "min_bid": round(DPH[c.id] / c.prem, 4),
                   "dph_total": round(DPH[c.id] + rng.uniform(-0.002, 0.002), 4)}
            if rng.random() < 0.3:
                row["inet_down_billed"] = round(rng.uniform(0, 0.01), 4)
            samples.append(row)
            if s[0] in ("running", "loading", "unknown"):
                spent += row["dph_total"] * 60 / 3600
        if int((t - T0) // 60) % 10 == 0 and abs(((t - T0) % 600)) < 1:
            credit_rows.append({"type": "credit", "t": iso(t, False), "ts": t, "credit": round(credit_start - spent * 1.012, 4)})
        t += 60
    # one file per UTC date
    by_day = {}
    for r in samples + credit_rows:
        by_day.setdefault(iso(r["ts"], False)[:10], []).append(r)
    for day, rows in by_day.items():
        wl("samples-%s.jsonl" % day, rows)

    # ---------- fleet every 15 s
    fleet = []
    t = T0 - 60
    all_active = active
    while t <= T0 + DUR + 30:
        inst, healthy, router = {}, [], {}
        for c in cards:
            s = c.at(t)
            if s is None:
                continue
            h = c.healthy(t)
            inst[str(c.id)] = {"status": s[0], "intended": "running", "healthy": h}
            if h:
                healthy.append(c.id)
                router["i%d" % c.id] = {"in_flight": load(cards.index(c), t), "state": "active"}
        fleet.append({"t": iso(t), "ts": t, "healthy": healthy, "registered": ["i%d" % i for i in healthy], "instances": inst,
                      "router": router, "actions": [], "errors": []})
        t += 15
    wl("fleet.jsonl", fleet)

    # ---------- events
    ev = []
    for c in cards:
        ev.append({"event": "created", "id": c.id, "bid": round(DPH[c.id], 3), "t": iso(c.created)})
        prev = None
        tt = c.created
        while tt <= T0 + END:
            s = c.at(tt)
            if s and prev and s != prev:
                ev.append({"event": "change", "id": c.id, "old": {"status": prev[0], "hb": prev[1]},
                           "new": {"status": s[0], "hb": s[1]}, "t": iso(tt)})
            prev = s or prev
            tt += 15
        ev.append({"event": "destroyed", "id": c.id, "reason": "teardown", "t": iso(c.destroyed)})
    ev.append({"event": "bid_changed", "id": cards[4].id, "price": 0.37, "t": iso(T0 + 3600)})
    ev.sort(key=lambda e: e["t"])
    wl("events.jsonl", ev)

    state = {"started": T0 - 8 * MIN, "hours": 2.0, "budget": 5.0, "credit_start": credit_start,
             "last_credit": credit_rows[-1]["credit"],
             "instances": [{"id": c.id, "bid": round(DPH[c.id], 3), "min_bid": round(DPH[c.id] / c.prem, 4), "premium": c.prem,
                            "created": c.created, "destroyed": c.destroyed, "geolocation": "XX"} for c in cards],
        "rent_errors": [{"t": iso(T0 - 15 * MIN), "error": "no offers"}]}
    json.dump(state, open(os.path.join(a.out, "state.json"), "w"), indent=1)
    print("fake data in", a.out, "requests", len(run), "journal", len(journal))


if __name__ == "__main__":
    main()
