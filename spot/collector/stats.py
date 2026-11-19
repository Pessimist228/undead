"""Quick health and price check over the files written by collect.py (stdlib only)."""
import argparse
import glob
import json
import os
import statistics
import sys
import zlib
from datetime import datetime, timedelta

FMT = "%Y-%m-%dT%H:%M:%SZ"
KINDS = ("on-demand", "bid")


def read_members(path):
    """Yield decoded text of each gzip member; stop with a warning at a damaged one."""
    with open(path, "rb") as f:
        raw = f.read()
    while raw:
        d = zlib.decompressobj(31)
        try:
            text = d.decompress(raw)
            if not d.eof:
                raise EOFError("truncated member")
        except (zlib.error, EOFError) as e:
            print("WARNING: %s: skipping damaged trailing data (%s)" % (os.path.basename(path), e),
                  file=sys.stderr)
            return
        yield text.decode("utf-8", "replace")
        raw = d.unused_data


def load(data_dir):
    recs = []
    for path in sorted(glob.glob(os.path.join(data_dir, "vast-*.jsonl.gz"))):
        for text in read_members(path):
            for ln in text.splitlines():
                try:
                    recs.append(json.loads(ln))
                except ValueError:
                    print("WARNING: bad json line in %s" % os.path.basename(path), file=sys.stderr)
    return recs


def cheapest6(rec):
    """Mean dph_total of the 6 cheapest offers with reliability2 >= 0.98, else None."""
    ix = {c: i for i, c in enumerate(rec["cols"])}
    prices = sorted(r[ix["dph_total"]] for r in rec["rows"]
                    if r[ix["dph_total"]] is not None and (r[ix["reliability2"]] or 0) >= 0.98)
    return sum(prices[:6]) / 6 if len(prices) >= 6 else None


def machines(rec):
    i = rec["cols"].index("machine_id")
    return len({r[i] for r in rec["rows"] if r[i] is not None})


def fmt(v, spec="%.3f"):
    return "n/a" if v is None else spec % v


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=os.path.join(here, "data"))
    recs = load(ap.parse_args(argv).data)
    if not recs:
        print("no data found")
        return 1
    ticks = sorted({r["tick"] for r in recs})
    times = [datetime.strptime(t, FMT) for t in ticks]
    gaps = [b - a for a, b in zip(times, times[1:]) if b - a > timedelta(minutes=15)]
    failed = sum(1 for r in recs if not r.get("ok"))
    print("ticks: %d   first: %s   last: %s" % (len(ticks), ticks[0], ticks[-1]))
    print("gaps > 15 min: %d, total length %s   failed queries: %d"
          % (len(gaps), sum(gaps, timedelta()), failed))
    oks = [r for r in recs if r.get("ok")]
    gpus = sorted({r["gpu"] for r in recs})
    print("\nCOVERAGE (gpu: ticks with data | median offers on-demand / bid | truncated)")
    for g in gpus:
        mine = [r for r in oks if r["gpu"] == g]
        med = {k: [r["n"] for r in mine if r["kind"] == k] for k in KINDS}
        m = [fmt(statistics.median(med[k]) if med[k] else None, "%g") for k in KINDS]
        print("  %-16s %5d | %s / %s | %d" % (g, len({r["tick"] for r in mine}), m[0], m[1],
                                              sum(1 for r in mine if r.get("truncated"))))
    print("\nLATEST TICK %s (mean $/h of 6 cheapest, reliability2>=0.98 | discount | machines)" % ticks[-1])
    for g in gpus:
        cur = {k: next((r for r in oks if r["tick"] == ticks[-1] and r["gpu"] == g and r["kind"] == k), None)
               for k in KINDS}
        p = {k: cheapest6(cur[k]) if cur[k] else None for k in KINDS}
        disc = (1 - p["bid"] / p["on-demand"]) * 100 if p["bid"] is not None and p["on-demand"] else None
        mc = ["n/a" if cur[k] is None else str(machines(cur[k])) for k in KINDS]
        print("  %-16s od %s | bid %s | discount %s | machines %s / %s"
              % (g, fmt(p["on-demand"]), fmt(p["bid"]), fmt(disc, "%.1f%%"), mc[0], mc[1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
