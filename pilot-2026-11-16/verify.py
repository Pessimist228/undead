#!/usr/bin/env python3
"""Recompute step0 results from the raw events.jsonl journals, ignoring the summaries.

    tar -xzf raw/A-0a-20261116-144825.tar.gz
    python3 verify.py 0a-20261116-144825 [another-run-dir]

For each run: resumed answers compared with their control by token IDs and logprobs,
end-to-end results through the router, and pauses. With two run directories it also
compares the control answers between them (for example card A against card B).
"""
import collections
import json
import statistics
import sys


def load(run_dir):
    with open(f"{run_dir}/events.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def check(ev, name):
    last = {}
    cut_tot = collections.Counter()
    cut_ok = collections.Counter()
    toks = collections.Counter()
    lp_bad = collections.Counter()
    first = collections.defaultdict(list)
    rep_tot = collections.Counter()
    rep_ok = collections.Counter()
    disagree = 0
    for e in ev:
        kind = e.get("kind")
        key = (e.get("config"), e.get("prompt"), e.get("mode"))
        if kind == "control":
            last[key] = e
        elif kind == "cut":
            c = last[key]
            k = e["cut"]
            want = c["ids"][k:]
            ok = e["ids"] == want
            cut_tot[e["config"]] += 1
            cut_ok[e["config"]] += ok
            toks[e["config"]] += len(want)
            lp_bad[e["config"]] += (e.get("lps") != c.get("lps", [])[k:])
            disagree += (ok != bool(e.get("ids_match")))
            if not ok:
                d = next((i for i, (a, b) in enumerate(zip(e["ids"], want)) if a != b),
                         min(len(e["ids"]), len(want)))
                first[e["config"]].append(d)
        elif kind in ("aa", "ab"):
            rep_tot[(e["config"], kind)] += 1
            rep_ok[(e["config"], kind)] += bool(e.get("match"))
    print(f"== {name}")
    for cfg in cut_tot:
        line = (f"  {cfg}: resumed answers matching control {cut_ok[cfg]}/{cut_tot[cfg]} "
                f"({100 * cut_ok[cfg] / cut_tot[cfg]:.1f}%), tokens compared {toks[cfg]}, "
                f"logprobs differ in {lp_bad[cfg]}")
        if first[cfg]:
            line += f", median first divergence {statistics.median(first[cfg]):.0f} tokens after the cut"
        print(line)
    if disagree:
        print(f"  WARNING: recomputation disagrees with the journal in {disagree} records")
    for (cfg, kind), n in rep_tot.items():
        where = "same replica" if kind == "aa" else "other replica"
        print(f"  {cfg}: full reruns, {where}: {rep_ok[(cfg, kind)]}/{n}")
    subs = collections.defaultdict(list)
    for e in ev:
        if e.get("kind") == "e2e":
            subs[(e["config"], e["sub"])].append(e)
    for (cfg, sub), lst in subs.items():
        pauses = sorted(p for e in lst for p in (e.get("pauses") or []))
        errs = sum(1 for e in lst if e.get("error"))
        hit = sum(1 for e in lst if (e.get("attempts") or 1) > 1)
        line = (f"  {cfg} router/{sub}: matched {sum(1 for e in lst if e.get('match'))}/{len(lst)}, "
                f"failure hit the answer in {hit}, errors {errs}")
        if pauses:
            line += (f", pause median {statistics.median(pauses) * 1000:.0f} ms, "
                     f"max {pauses[-1] * 1000:.0f} ms ({len(pauses)} samples)")
        print(line)
    for e in ev:
        if e.get("kind") == "perf":
            print(f"  throughput: {e.get('conc')} stream(s), {e.get('tok_s')} tok/s, "
                  f"median TTFT {e.get('ttft_med')} s")
    return last


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    a = check(load(argv[1]), argv[1])
    if len(argv) > 2:
        b = check(load(argv[2]), argv[2])
        same = diff = lp = 0
        for key, e in a.items():
            if key in b:
                if e["ids"] == b[key]["ids"]:
                    same += 1
                    lp += (e.get("lps") == b[key].get("lps"))
                else:
                    diff += 1
        print(f"== across the two runs: controls equal by token IDs {same}, different {diff}, "
              f"logprobs bit-identical in {lp} of the equal ones")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
