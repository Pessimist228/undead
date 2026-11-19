#!/usr/bin/env python3
"""Turn the logs of a live test on interruptible GPU cards into numbers, charts and a time-lapse.

    python3 report_run.py --dir LOGS [--out LOGS/report] [--ondemand-price 0.5] [--no-video]

Everything is read from --dir; any input file may be missing. Section 1 loads files, section 2 computes
the numbers (no drawing), section 3 and 4 draw. Card ids never appear on the charts: cards are "card 1..n"
in the order they were rented.
"""
import argparse
import bisect
import datetime as dt
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
import textwrap

import numpy as np

# ----------------------------------------------------------------------------------------------
# 1. Loading
# ----------------------------------------------------------------------------------------------


def parse_time(v):
    """ISO string, epoch number or numeric string -> epoch seconds (UTC), or None."""
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if re.fullmatch(r"\d+(\.\d+)?", s):
        return float(s)
    try:
        s2 = s.replace("Z", "+00:00").replace(" ", "T", 1) if "T" not in s else s.replace("Z", "+00:00")
        d = dt.datetime.fromisoformat(s2)
    except ValueError:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=dt.timezone.utc)
    return d.timestamp()


def read_jsonl(path):
    out = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if isinstance(r, dict):
                    out.append(r)
    except OSError:
        pass
    return out


def read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        return r if isinstance(r, dict) else None
    except (OSError, ValueError):
        return None


def parse_load_start(text, ref_ts=None):
    """load-start.txt: ISO time, epoch, or a bare HH:MM[:SS] (UTC, date taken from ref_ts)."""
    if not text:
        return None
    text = text.strip().splitlines()[0].strip() if text.strip() else ""
    t = parse_time(text)
    if t is not None:
        return t
    m = re.fullmatch(r"(\d{1,2}):(\d{2})(?::(\d{2}(?:\.\d+)?))?\s*(?:UTC|Z)?", text)
    if m and ref_ts is not None:
        day = dt.datetime.fromtimestamp(ref_ts, dt.timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        return day.timestamp() + int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3) or 0)
    return None


def load_dir(d):
    """Read every input file that exists. Missing files give empty lists / None."""
    def p(n):
        return os.path.join(d, n)

    data = {"dir": d}
    data["run"] = read_jsonl(p("run.jsonl"))
    samples, credits = [], []
    for fn in sorted(glob.glob(p("samples-*.jsonl"))):
        for r in read_jsonl(fn):
            ts = r.get("ts")
            ts = float(ts) if isinstance(ts, (int, float)) else parse_time(r.get("t"))
            if ts is None:
                continue
            r["ts"] = ts
            if r.get("type") == "credit" or ("credit" in r and "status" not in r):
                if isinstance(r.get("credit"), (int, float)):
                    credits.append(r)
            elif "status" in r and r.get("id") is not None:
                samples.append(r)
    samples.sort(key=lambda r: r["ts"])
    credits.sort(key=lambda r: r["ts"])
    data["samples"], data["credits"] = samples, credits
    events = read_jsonl(p("events.jsonl"))
    for e in events:
        e["ts"] = parse_time(e.get("t"))
    data["events"] = events
    data["state"] = read_json(p("state.json"))
    fleet = read_jsonl(p("fleet.jsonl"))
    for r in fleet:
        ts = r.get("ts")
        r["ts"] = float(ts) if isinstance(ts, (int, float)) else parse_time(r.get("t"))
    data["fleet"] = sorted([r for r in fleet if r["ts"] is not None], key=lambda r: r["ts"])
    data["journal"] = read_jsonl(p("journal.jsonl"))
    try:
        with open(p("load-start.txt"), encoding="utf-8") as f:
            txt = f.read()
    except OSError:
        txt = ""
    data["load_start_text"] = txt
    return data


# ----------------------------------------------------------------------------------------------
# 2. Numbers
# ----------------------------------------------------------------------------------------------

DEFAULTS = {"min_down": 30.0, "grid": 15.0}


def pct(values, q):
    v = [x for x in values if x is not None]
    return float(np.percentile(v, q)) if v else None


def median(values):
    return pct(values, 50)


def prepare_requests(data):
    """Normalise run.jsonl: start/end epoch, failed flag and failure class."""
    out = []
    for r in data["run"]:
        st = parse_time(r.get("start"))
        if st is None:
            continue
        dur = r.get("duration_ms") or 0
        status = r.get("status")
        failed = status != 200 or bool(r.get("error")) or r.get("finish_reason") is None
        kind = None
        if failed:
            if status == 503:
                kind = "503"
            elif status == 200:
                kind = "cut"
            else:
                kind = "other"
        out.append({"start": st, "end": st + dur / 1000.0, "failed": failed, "kind": kind,
                    "ttfc": r.get("ttfc_ms"), "chunks": r.get("chunks") or 0,
                    "max_gap_ms": r.get("max_gap_ms"), "max_gap_end": parse_time(r.get("max_gap_end"))})
    out.sort(key=lambda r: r["end"])
    return out


def prepare_resumes(data, requests, gap_threshold_ms=1000.0):
    """Resumed streams: from the router journal if there is one, else estimated from client-side gaps.
    Returns (list of {"end": epoch, "pauses": [ms,...]}, source)."""
    out = []
    if data["journal"]:
        for j in data["journal"]:
            res = j.get("resumes") or []
            if not res:
                continue
            end = parse_time(j.get("end"))
            if end is None:
                st = parse_time(j.get("start")) or 0
                end = st + (j.get("duration_ms") or 0) / 1000.0
            pauses = [r.get("pause_ms") for r in res if isinstance(r, dict) and r.get("pause_ms") is not None]
            out.append({"end": end, "pauses": pauses, "n": len(res)})
        return sorted(out, key=lambda r: r["end"]), "journal"
    for r in requests:
        g = r["max_gap_ms"]
        if g is not None and g >= gap_threshold_ms and not r["failed"]:
            out.append({"end": r["max_gap_end"] or r["end"], "pauses": [g], "n": 1})
    return sorted(out, key=lambda r: r["end"]), ("client gaps" if requests else "none")


def classify_obs(h, seen):
    if h:
        return "healthy"
    return "down" if seen else "loading"


def build_segments(obs, created, destroyed, dt_typ, min_down):
    """obs: sorted [(ts, healthy)] -> [(a, b, state)] with states loading/healthy/down/destroyed."""
    segs = []
    seen = False
    for i, (ts, h) in enumerate(obs):
        nxt = obs[i + 1][0] if i + 1 < len(obs) else ts + dt_typ
        if destroyed is not None and ts >= destroyed:
            state = "destroyed"
            ts = max(ts, destroyed)
        else:
            state = classify_obs(h, seen)
            seen = seen or h
            if destroyed is not None:
                nxt = min(nxt, destroyed)
        if nxt > ts:
            segs.append([ts, nxt, state])
    if segs and created is not None and created < segs[0][0] and segs[0][2] == "loading":
        segs[0][0] = created  # only a card seen loading is known to have been loading since it was rented
    if segs and destroyed is not None and segs[-1][2] != "destroyed" and segs[-1][1] < destroyed:
        segs[-1][1] = destroyed
    # bridge short blips: a "down" shorter than min_down between two healthy stretches is a probe blip
    for i in range(1, len(segs) - 1):
        a, b, s = segs[i]
        if s == "down" and b - a < min_down and segs[i - 1][2] == "healthy" and segs[i + 1][2] == "healthy":
            segs[i][2] = "healthy"
    merged = []
    for a, b, s in segs:
        if merged and merged[-1][2] == s and abs(merged[-1][1] - a) < 1e-6:
            merged[-1][1] = b
        else:
            merged.append([a, b, s])
    return [tuple(m) for m in merged]


def rent_info(data):
    """Per card: created, destroyed, premium, in order of renting. Keys are str ids."""
    info = {}

    def get(cid):
        return info.setdefault(str(cid), {"id": str(cid), "created": None, "destroyed": None, "premium": None,
                                          "bid": None, "min_bid": None})
    st = data["state"] or {}
    for i in st.get("instances") or []:
        if i.get("id") is None:
            continue
        c = get(i["id"])
        c["created"] = i.get("created") if isinstance(i.get("created"), (int, float)) else parse_time(i.get("created"))
        c["destroyed"] = i.get("destroyed") if isinstance(i.get("destroyed"), (int, float)) else parse_time(i.get("destroyed"))
        c["bid"], c["min_bid"] = i.get("bid"), i.get("min_bid")
        prem = i.get("premium")
        if prem is None and i.get("bid") and i.get("min_bid"):
            prem = i["bid"] / i["min_bid"]  # a multiplier of the minimum bid, like state.json "premium" (1.15 = 15% over)
        c["premium"] = prem
    for e in data["events"]:
        if e.get("id") is None or e.get("ts") is None:
            continue
        if e.get("event") == "created":
            c = get(e["id"])
            c["created"] = c["created"] or e["ts"]
            c["bid"] = c["bid"] or e.get("bid")
        elif e.get("event") == "destroyed":
            c = get(e["id"])
            c["destroyed"] = c["destroyed"] or e["ts"]
    for r in data["samples"]:
        c = get(r["id"])
        if c["created"] is None or r["ts"] < c["created"]:
            c["created"] = r["ts"] if c["created"] is None else c["created"]
        if c["min_bid"] is None and r.get("min_bid"):
            c["min_bid"] = r["min_bid"]
    for r in data["fleet"]:
        for cid in (r.get("instances") or {}):
            c = get(cid)
            if c["created"] is None:
                c["created"] = r["ts"]
    return info


def card_observations(data):
    """{id: [(ts, healthy)]} from fleet.jsonl, else from the per-minute samples; plus the typical spacing."""
    obs = {}
    if data["fleet"]:
        for r in data["fleet"]:
            hl = {str(x) for x in (r.get("healthy") or [])}
            inst = r.get("instances") or {}
            for cid in set(inst) | hl:
                h = cid in hl or bool((inst.get(cid) or {}).get("healthy"))
                obs.setdefault(str(cid), []).append((r["ts"], h))
        tss = [r["ts"] for r in data["fleet"]]
    else:
        for r in data["samples"]:
            h = r.get("status") == "running" and r.get("hb") == "ok"
            obs.setdefault(str(r["id"]), []).append((r["ts"], h))
        tss = sorted({r["ts"] for r in data["samples"]})
    diffs = np.diff(tss) if len(tss) > 1 else []
    dt_typ = float(np.median(diffs)) if len(diffs) else 60.0
    return obs, dt_typ


DESTROY_EVENTS = ("destroyed", "dropped")  # events.jsonl lines where we removed a card (reason strings vary)


def stopped_by_us(data, card, ts):
    """True when we destroyed or dropped the card at this moment (state.json time or an event, -30 s..+60 s).
    Not used: intended_status "stopped" - the marketplace sets it itself when it preempts a card."""
    cid = card["id"]
    if card.get("destroyed") is not None and ts - 30 <= card["destroyed"] <= ts + 60:
        return True
    for e in data["events"]:
        if e.get("event") in DESTROY_EVENTS and str(e.get("id")) == cid and e.get("ts") is not None \
                and ts - 30 <= e["ts"] <= ts + 60:
            return True
    return False


def build_cards(data, opts):
    info = rent_info(data)
    obs, dt_typ = card_observations(data)
    cards = []
    for cid, c in info.items():
        o = sorted(obs.get(cid, []))
        c = dict(c)
        c["segs"] = build_segments(o, c["created"], c["destroyed"], dt_typ, opts["min_down"]) if o else []
        cards.append(c)
    cards.sort(key=lambda c: (c["created"] if c["created"] is not None else 1e18, c["id"]))
    for i, c in enumerate(cards):
        c["label"] = "card %d" % (i + 1)
    return cards, dt_typ


def clip_segs(segs, t0, t1):
    out = []
    for a, b, s in segs:
        a2, b2 = max(a, t0), min(b, t1)
        if b2 > a2:
            out.append((a2, b2, s))
    return out


def state_at(segs, t):
    starts = [s[0] for s in segs]
    i = bisect.bisect_right(starts, t) - 1
    if i < 0 or t >= segs[i][1] + 1e-9:
        return None
    return segs[i][2]


def card_stops(data, card, t0, t1):
    """Stops not caused by us inside [t0, t1]: healthy -> not healthy. With down time and whether it returned."""
    stops = []
    segs = card["segs"]
    for i, (a, b, s) in enumerate(segs):
        if s != "down" or i == 0 or segs[i - 1][2] != "healthy" or not (t0 <= a <= t1):
            continue
        if stopped_by_us(data, card, a):
            continue
        nxt = segs[i + 1] if i + 1 < len(segs) else None
        back = nxt is not None and nxt[2] == "healthy"
        stops.append({"card": card["id"], "label": card["label"], "ts": a, "premium": card["premium"],
                      "back": back, "down_s": b - a, "back_ts": b if back else None})
    return stops


def healthy_count_series(cards, t0, t1, grid):
    ts = np.arange(t0, t1, grid)
    cnt = np.zeros(len(ts), dtype=int)
    for c in cards:
        for a, b, s in c["segs"]:
            if s == "healthy":
                i0 = int(np.searchsorted(ts, a, side="left"))
                i1 = int(np.searchsorted(ts, b, side="left"))
                cnt[i0:i1] += 1
    return ts, cnt


def estimate_cost(data, cards, t_end):
    """Time-based cost estimate: sum of dph_total x time over each card's per-minute samples."""
    by = {}
    for r in data["samples"]:
        by.setdefault(str(r["id"]), []).append(r)
    pts = []  # (ts, dollars) increments
    for cid, rows in by.items():
        rows.sort(key=lambda r: r["ts"])
        for i, r in enumerate(rows):
            if r.get("status") == "absent" or not isinstance(r.get("dph_total"), (int, float)):
                continue
            nxt = rows[i + 1]["ts"] if i + 1 < len(rows) else r["ts"] + 60
            pts.append((r["ts"], r["dph_total"] * min(nxt - r["ts"], 600) / 3600.0))
    if pts:
        return sum(p[1] for p in pts), sorted(pts)
    # fallback: bid x time alive, from state.json
    info = rent_info(data)
    total, pts = 0.0, []
    for c in info.values():
        if c["created"] is None or not c["bid"]:
            continue
        end = c["destroyed"] or t_end
        v = c["bid"] * max(0.0, end - c["created"]) / 3600.0
        total += v
        pts.append((end, v))
    return (total, sorted(pts)) if pts else (None, [])


def money_curve(data, est_pts):
    """(ts array, dollars spent so far) for the time-lapse counter."""
    st = data["state"] or {}
    cs = st.get("credit_start")
    cr = data["credits"]
    if cs is None and cr:
        cs = cr[0]["credit"]
    if cs is not None and cr:
        t_start = st.get("started") if isinstance(st.get("started"), (int, float)) else cr[0]["ts"]
        ts = [min(t_start, cr[0]["ts"])] + [r["ts"] for r in cr]
        sp = [0.0] + [cs - r["credit"] for r in cr]
        return np.array(ts), np.array(sp)
    if est_pts:
        ts = [p[0] for p in est_pts]
        sp = np.cumsum([p[1] for p in est_pts])
        return np.array(ts), np.array(sp)
    return None, None


def compute_window(data, requests):
    first_start = min([r["start"] for r in requests], default=None)
    t0 = parse_load_start(data["load_start_text"], first_start)
    if t0 is None:
        t0 = first_start
    if t0 is None and data["fleet"]:
        t0 = data["fleet"][0]["ts"]
    if t0 is None and data["samples"]:
        t0 = data["samples"][0]["ts"]
    t1 = max([r["end"] for r in requests], default=None)
    if t1 is None and data["fleet"]:
        t1 = data["fleet"][-1]["ts"]
    if t1 is None and data["samples"]:
        t1 = data["samples"][-1]["ts"]
    if t0 is not None and t1 is not None and t1 <= t0:
        t1 = t0 + 1
    return t0, t1


def compute_metrics(data, opts=None, ondemand_price=None):
    """All numbers for summary.json. Pure function of the loaded data. Also returns the objects the charts need."""
    opts = dict(DEFAULTS, **(opts or {}))
    requests = prepare_requests(data)
    t0, t1 = compute_window(data, requests)
    m = {"load_start": iso(t0), "load_end": iso(t1),
         "load_duration_s": (t1 - t0) if t0 is not None and t1 is not None else None}
    cards, dt_typ = build_cards(data, opts)
    ctx = {"t0": t0, "t1": t1, "cards": cards, "requests": requests, "opts": opts}

    # ---- cards
    m["cards_rented"] = len(cards) if cards else None
    healthy_h, stops = 0.0, []
    if t0 is not None and t1 is not None:
        for c in cards:
            c["healthy_s"] = sum(b - a for a, b, s in clip_segs(c["segs"], t0, t1) if s == "healthy")
            healthy_h += c["healthy_s"] / 3600.0
            stops += card_stops(data, c, t0, t1)
        stops.sort(key=lambda s: s["ts"])
    have_cards = any(c["segs"] for c in cards)
    m["card_hours_healthy"] = healthy_h if have_cards else None
    m["stops_total"] = len(stops) if have_cards else None
    m["stops_per_card_hour"] = (len(stops) / healthy_h) if have_cards and healthy_h > 0 else None
    m["stops_by_premium"] = stops_by_premium(cards, stops) if have_cards else None
    back = [s for s in stops if s["back"]]
    m["stops_came_back"] = len(back) if have_cards else None
    m["stops_not_back"] = (len(stops) - len(back)) if have_cards else None
    m["down_median_s"] = median([s["down_s"] for s in back])
    m["down_max_s"] = max([s["down_s"] for s in back], default=None)
    # time from rent to first healthy per card
    t_ready = []
    sample_first = {}
    for r in data["samples"]:
        if r.get("hb") == "ok":
            sample_first.setdefault(str(r["id"]), r["ts"])
    raw_obs, _ = card_observations(data)
    served_ids = {cid for cid, o in raw_obs.items() if any(h for _, h in o)} | set(sample_first)
    seen_ids = set(raw_obs) | {str(r["id"]) for r in data["samples"]}
    for c in cards:
        fh = next((a for a, b, s in c["segs"] if s == "healthy"), None)
        sf = sample_first.get(c["id"])  # the per-minute samples start at renting: they see earlier than the router log
        fh = sf if sf is not None and (fh is None or sf < fh) else fh
        c["first_healthy"] = fh
        if fh is not None and c["created"] is not None:
            t_ready.append(max(0.0, fh - c["created"]))
        c["tracked"] = c["id"] in seen_ids
        c["served"] = c["id"] in served_ids
    m["rent_to_healthy_s"] = {c["label"]: (max(0.0, c["first_healthy"] - c["created"])
                                           if c["first_healthy"] is not None and c["created"] is not None
                                           else (None if c["tracked"] else "not tracked"))
                              for c in cards} if have_cards else None
    m["rent_to_healthy_median_s"] = median(t_ready)
    m["cards_tracked"] = sum(c["tracked"] for c in cards) if have_cards else None
    m["cards_served"] = sum(c["served"] for c in cards) if have_cards else None
    m["cards_never_served"] = sum(c["tracked"] and not c["served"] for c in cards) if have_cards else None
    m["cards_not_tracked"] = sum(not c["tracked"] for c in cards) if have_cards else None
    ctx["stops"] = stops

    # ---- fleet size
    if have_cards and t0 is not None:
        ts, cnt = healthy_count_series(cards, t0, t1, opts["grid"])
        ctx["count_ts"], ctx["count"] = ts, cnt
        m["healthy_min"] = int(cnt.min()) if len(cnt) else None
        m["healthy_median"] = float(np.median(cnt)) if len(cnt) else None
    else:
        m["healthy_min"] = m["healthy_median"] = None

    # ---- requests
    tot = len(requests)
    fail = [r for r in requests if r["failed"]]
    m["requests_total"] = tot if tot else None
    m["requests_finished"] = (tot - len(fail)) if tot else None
    m["requests_failed"] = len(fail) if tot else None
    m["percent_finished"] = (100.0 * (tot - len(fail)) / tot) if tot else None
    m["failed_rejected_503"] = sum(r["kind"] == "503" for r in fail) if tot else None
    m["failed_stream_cut"] = sum(r["kind"] == "cut" for r in fail) if tot else None
    m["failed_other"] = sum(r["kind"] == "other" for r in fail) if tot else None
    ttfc = [r["ttfc"] for r in requests if isinstance(r["ttfc"], (int, float))]
    m["ttfc_p50_ms"], m["ttfc_p95_ms"] = pct(ttfc, 50), pct(ttfc, 95)
    tokens = sum(r["chunks"] for r in requests)
    m["tokens_generated"] = tokens if tot else None

    # ---- resumes
    resumes, src = prepare_resumes(data, requests)
    ctx["resumes"] = resumes
    pauses = [p for r in resumes for p in r["pauses"]]
    ctx["pauses_ms"] = pauses
    m["streams_resumed"] = len(resumes) if (data["journal"] or tot) else None
    m["resume_events"] = sum(r["n"] for r in resumes) if (data["journal"] or tot) else None
    m["resume_source"] = src
    m["resume_pause_p50_ms"], m["resume_pause_p95_ms"] = pct(pauses, 50), pct(pauses, 95)
    m["resume_pause_max_ms"] = max(pauses, default=None)
    m["resume_pause_share_under_2s"] = (sum(p < 2000 for p in pauses) / len(pauses)) if pauses else None
    m["resume_pauses_over_10s"] = sum(p > 10000 for p in pauses) if pauses else None
    m["resume_pause_longest_s"] = (max(pauses) / 1000.0) if pauses else None

    # ---- money
    st = data["state"] or {}
    cs = st.get("credit_start")
    last = st.get("last_credit")
    if cs is None and data["credits"]:
        cs = data["credits"][0]["credit"]
    if last is None and data["credits"]:
        last = data["credits"][-1]["credit"]
    m["money_used"] = (cs - last) if cs is not None and last is not None else None
    t_end = t1 if t1 is not None else 0
    est, est_pts = estimate_cost(data, cards, t_end)
    m["money_estimate"] = est
    ctx["money_ts"], ctx["money_spent"] = money_curve(data, est_pts)
    if m["money_used"] is not None and not data["credits"] and ctx["money_spent"] is not None and ctx["money_spent"][-1] > 0:
        # no credit readings: shape of the curve from dph x time, scaled to end at the real credit drop
        ctx["money_spent"] = ctx["money_spent"] * (m["money_used"] / ctx["money_spent"][-1])
    spent = m["money_used"] if m["money_used"] is not None else est
    m["money_basis"] = "credit" if m["money_used"] is not None else ("estimate" if est is not None else None)
    m["dollars_per_million_tokens"] = (spent / (tokens / 1e6)) if spent is not None and tokens else None
    m["rented_card_hours"] = rented_card_hours(cards, st, t1) if have_cards else None
    m["cost_per_healthy_card_hour"] = (spent / healthy_h) if spent is not None and have_cards and healthy_h > 0 else None
    m["cost_per_rented_card_hour"] = (spent / m["rented_card_hours"]) if spent is not None and m["rented_card_hours"] else None
    if ondemand_price is not None:
        m["ondemand_price_per_hour"] = ondemand_price
        m["ondemand_cost_healthy_card_hours"] = ondemand_price * healthy_h if have_cards else None
        m["ondemand_cost_rented_card_hours"] = ondemand_price * m["rented_card_hours"] if have_cards else None
        cph = m["cost_per_healthy_card_hour"]
        m["cost_vs_ondemand"] = None if cph is None else (
            "Per healthy card-hour we paid $%.3f, which is %s the on-demand price of $%.3f (%.0f%% of it)." % (
                cph, "lower than" if cph < ondemand_price else "higher than", ondemand_price, 100 * cph / ondemand_price))
    m["cap_events"] = sum(1 for e in data["events"] if e.get("event") == "cap")
    m["rent_errors"] = len((st.get("rent_errors") or [])) if data["state"] else None
    return m, ctx


def rented_card_hours(cards, state, t_fallback):
    """Rent to destroy, summed over all cards, not clipped to the load. A card with no destroy time (lost
    from state.json) is counted until the end of the run (state finished_at, else the last destroy, else t_fallback)."""
    ends = [c["destroyed"] for c in cards if c["destroyed"] is not None]
    run_end = (state or {}).get("finished_at") or (max(ends) if ends else t_fallback)
    total = 0.0
    for c in cards:
        if c["created"] is None:
            continue
        total += max(0.0, (c["destroyed"] or run_end) - c["created"]) / 3600.0
    return total


def stops_by_premium(cards, stops):
    groups = {}
    for c in cards:
        if c.get("premium") is None:
            continue
        g = groups.setdefault(round(float(c["premium"]), 4), {"premium": round(float(c["premium"]), 4), "cards": 0,
                                                              "healthy_card_hours": 0.0, "stops": 0})
        g["cards"] += 1
        g["healthy_card_hours"] += c.get("healthy_s", 0.0) / 3600.0
    if not groups:
        return None
    pm = {c["id"]: round(float(c["premium"]), 4) for c in cards if c.get("premium") is not None}
    for s in stops:
        if s["card"] in pm:
            groups[pm[s["card"]]]["stops"] += 1
    out = []
    for k in sorted(groups):
        g = groups[k]
        g["stops_per_card_hour"] = g["stops"] / g["healthy_card_hours"] if g["healthy_card_hours"] > 0 else None
        out.append(g)
    return out


def iso(t):
    if t is None:
        return None
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def hhmm(t, sec=False):
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%H:%M:%S" if sec else "%H:%M")


def fmt_dur(s):
    if s is None:
        return "n/a"
    s = float(s)
    if s < 90:
        return "%d s" % round(s)
    if s < 5400:
        return "%.0f min" % (s / 60)
    return "%d h %02d min" % (s // 3600, round((s % 3600) / 60) % 60)


def fmt_num(v, nd=0, prefix="", suffix=""):
    if v is None:
        return "n/a"
    return "%s%s%s" % (prefix, format(v, ",.%df" % nd), suffix)


def fmt_premium(p):
    """state.json premium is a multiplier of the minimum bid (1.15); a value below 1 is read as a fraction over it."""
    return "x%.2f of minimum" % p if p >= 1 else "+%.0f%% over minimum" % (p * 100)


def fmt_ms(v):
    return "n/a" if v is None else ("%.2f s" % (v / 1000.0))


def summary_text(m):
    L = []
    add = L.append
    add("Load: %s to %s UTC, %s" % (m["load_start"] or "n/a", m["load_end"] or "n/a", fmt_dur(m["load_duration_s"])))
    add("")
    add("CARDS")
    add("  Cards rented: %s; served (healthy at least once): %s; never served: %s; not tracked (no data): %s" % (
        fmt_num(m["cards_rented"]), fmt_num(m["cards_served"]), fmt_num(m["cards_never_served"]), fmt_num(m["cards_not_tracked"])))
    add("  Card-hours with a healthy server: %s" % fmt_num(m["card_hours_healthy"], 2))
    add("  Stops not caused by us: %s (%s per healthy card-hour)" % (
        fmt_num(m["stops_total"]), fmt_num(m["stops_per_card_hour"], 2)))
    for g in m["stops_by_premium"] or []:
        add("    bid %s: %d cards, %.2f healthy card-h, %d stops, %s per card-hour" % (
            fmt_premium(g["premium"]), g["cards"], g["healthy_card_hours"], g["stops"], fmt_num(g["stops_per_card_hour"], 2)))
    add("  Came back: %s of %s; down time median %s, max %s (of those that came back)" % (
        fmt_num(m["stops_came_back"]), fmt_num(m["stops_total"]), fmt_dur(m["down_median_s"]), fmt_dur(m["down_max_s"])))
    add("  Rent to first healthy: median %s" % fmt_dur(m["rent_to_healthy_median_s"]))
    for k, v in (m["rent_to_healthy_s"] or {}).items():
        add("    %s: %s" % (k, v if isinstance(v, str) else (fmt_dur(v) if v is not None else "never")))
    add("  Healthy cards during load: min %s, median %s" % (fmt_num(m["healthy_min"]), fmt_num(m["healthy_median"], 1)))
    add("")
    add("REQUESTS")
    add("  Total %s, finished %s (%s%%), failed %s" % (fmt_num(m["requests_total"]), fmt_num(m["requests_finished"]),
                                                    fmt_num(m["percent_finished"], 2), fmt_num(m["requests_failed"])))
    add("    rejected with 503: %s, stream cut: %s, other: %s" % (
        fmt_num(m["failed_rejected_503"]), fmt_num(m["failed_stream_cut"]), fmt_num(m["failed_other"])))
    add("  Streams resumed: %s (%s resume events, source: %s)" % (fmt_num(m["streams_resumed"]),
                                                              fmt_num(m["resume_events"]), m["resume_source"]))
    add("    pause p50 %s, p95 %s, max %s; under 2 s: %s of pauses; over 10 s: %s" % (
        fmt_ms(m["resume_pause_p50_ms"]), fmt_ms(m["resume_pause_p95_ms"]), fmt_ms(m["resume_pause_max_ms"]),
        "n/a" if m["resume_pause_share_under_2s"] is None else "%.0f%%" % (100 * m["resume_pause_share_under_2s"]),
        fmt_num(m["resume_pauses_over_10s"])))
    add("  Time to first chunk: p50 %s, p95 %s" % (fmt_ms(m["ttfc_p50_ms"]), fmt_ms(m["ttfc_p95_ms"])))
    add("  Tokens generated: %s" % fmt_num(m["tokens_generated"]))
    add("")
    add("MONEY")
    add("  Used (credit start minus last reading): %s" % fmt_num(m["money_used"], 2, "$"))
    add("  Time-based estimate (dph x time): %s" % fmt_num(m["money_estimate"], 2, "$"))
    add("  Dollars per million tokens (%s): %s" % (m["money_basis"] or "n/a", fmt_num(m["dollars_per_million_tokens"], 3, "$")))
    add("  Rented card-hours (rent to destroy): %s; cost per rented card-hour: %s" % (
        fmt_num(m["rented_card_hours"], 2), fmt_num(m["cost_per_rented_card_hour"], 3, "$")))
    add("  Cost per healthy card-hour: %s" % fmt_num(m["cost_per_healthy_card_hour"], 3, "$"))
    if m.get("cost_vs_ondemand"):
        add("  " + m["cost_vs_ondemand"])
    if "ondemand_price_per_hour" in m:
        add("  Same card-hours at on-demand $%.3f/h: %s for healthy card-hours, %s for rented card-hours (%.2f h)" % (
            m["ondemand_price_per_hour"], fmt_num(m["ondemand_cost_healthy_card_hours"], 2, "$"),
            fmt_num(m["ondemand_cost_rented_card_hours"], 2, "$"), m["rented_card_hours"] or 0))
    return "\n".join(L) + "\n"


def caption_events(cards, stops, t0, t1):
    """One line per card taken / back, in time order, for the time-lapse."""
    ev = []
    for s in stops:
        ev.append((s["ts"], "%s UTC  %s taken by the marketplace" % (hhmm(s["ts"]), s["label"])))
        if s["back"] and s["back_ts"] is not None and s["back_ts"] <= t1:
            ev.append((s["back_ts"], "%s UTC  %s is back after %s" % (hhmm(s["back_ts"]), s["label"], fmt_dur(s["down_s"]))))
    return sorted(ev)


# ----------------------------------------------------------------------------------------------
# 3. Drawing
# ----------------------------------------------------------------------------------------------

# Palette: dataviz reference palette (light surface). Blue/orange/aqua validated (CVD dE >= 9, normal >= 19).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"     # healthy / finished
ORANGE = "#eb6834"   # down / failed
AQUA = "#1baf7a"     # resumed
STATE_COLORS = {"loading": "#b5b4ad", "healthy": BLUE, "down": ORANGE, "destroyed": GRID}
STATE_NAMES = {"loading": "loading", "healthy": "healthy", "down": "down", "destroyed": "destroyed"}
FONT = ["DejaVu Sans", "Arial", "sans-serif"]


def mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": FONT, "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "text.color": INK,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 1.0,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "axes.labelsize": 12.5,
    })
    return plt


def style_axes(ax, grid_axis="y"):
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(length=0)


def time_axis(ax, t0, t1, label=True):
    span = (t1 - t0) / 60.0
    step = 10 if span <= 45 else 20 if span <= 150 else 30 if span <= 300 else 60
    first = math.ceil(t0 / (step * 60)) * step * 60
    ticks = np.arange(first, t1 + 1, step * 60)
    ticks = ticks[ticks - t0 > 0.35 * step * 60]  # a label squeezed against the left edge collides with the y axis
    ax.set_xticks(ticks)
    ax.set_xticklabels([hhmm(t) for t in ticks])
    ax.set_xlim(t0, t1)
    if label:
        ax.set_xlabel("Time, UTC")


def legend_row(fig, handles, labels, y=0.935, x=0.02, size=12, ncol=None):
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(x, y), ncol=ncol or len(labels), frameon=False,
               fontsize=size, handlelength=1.3, handletextpad=0.5, columnspacing=1.4, labelcolor=INK2)


def lane_cards(ctx):
    """Cards that get a lane: tracked ones, plus cards with a rent time but no data (drawn as "not tracked")."""
    return [c for c in ctx["cards"] if c["segs"] or (c["created"] is not None and ctx["t1"] is not None and c["created"] < ctx["t1"])]


def untracked_span(c, t0, t1):
    a = max(c["created"], t0)
    b = min(c["destroyed"] if c["destroyed"] is not None else t1, t1)
    return (a, b) if b > a else None


def lane_artists(plt, cards=None, t0=None, t1=None, with_stops=True):
    """Legend handles for the states that actually occur (all of them when cards is None)."""
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    present = set(STATE_COLORS)
    untracked = False
    if cards is not None:
        present = {s for c in cards for a, b, s in clip_segs(c["segs"], t0, t1)}
        untracked = any(not c["segs"] for c in cards)
    keys = [k for k in ("healthy", "loading", "down", "destroyed") if k in present]
    hs = [Patch(facecolor=STATE_COLORS[k], linewidth=0) for k in keys]
    names = [STATE_NAMES[k] for k in keys]
    if untracked:
        hs.append(Patch(facecolor=SURFACE, edgecolor=MUTED, hatch="////", linewidth=0))
        names.append("not tracked")
    if with_stops:
        hs.append(Line2D([], [], marker="v", linestyle="", color=INK, markeredgecolor=SURFACE, markersize=9))
        names.append("taken")
    return hs, names


def hatch_rect(a, b, y, h):
    from matplotlib.patches import Rectangle
    return Rectangle((a, y - h / 2), b - a, h, facecolor=SURFACE, edgecolor=MUTED, hatch="////", linewidth=0)


def draw_lanes(ax, cards, stops, t0, t1, ticksize=12, show_stops=True, msize=9, draw_segments=True):
    """One horizontal lane per card, coloured by state; stop moments marked. The label size follows the lane pitch."""
    from matplotlib.patches import Rectangle
    n = len(cards)
    h = 0.62
    for i, c in enumerate(cards if draw_segments else []):
        y = n - 1 - i
        for a, b, s in clip_segs(c["segs"], t0, t1):
            ax.add_patch(Rectangle((a, y - h / 2), b - a, h, facecolor=STATE_COLORS[s], linewidth=0))
        if not c["segs"] and c["created"] is not None:
            span = untracked_span(c, t0, t1)
            if span:
                ax.add_patch(hatch_rect(span[0], span[1], y, h))
    if show_stops:
        idx = {c["id"]: n - 1 - i for i, c in enumerate(cards)}
        xs = [s["ts"] for s in stops if s["card"] in idx]
        ys = [idx[s["card"]] + h / 2 + 0.16 for s in stops if s["card"] in idx]
        ax.plot(xs, ys, linestyle="", marker="v", markersize=msize, color=INK, markeredgecolor=SURFACE,
                markeredgewidth=1.2, clip_on=False, zorder=5)
    ax.set_ylim(-0.6, n - 0.3)
    ax.set_yticks([n - 1 - i for i in range(n)])
    pitch_pt = ax.figure.get_figheight() * ax.get_position().height / (n + 0.3) * 72
    ax.set_yticklabels([c["label"] for c in cards], fontsize=min(ticksize, round(pitch_pt * 0.78, 1)))
    ax.tick_params(axis="x", labelbottom=False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(length=0)
    ax.set_xlim(t0, t1)


def draw_count(ax, ts, cnt, t0, t1, ticksize=12, label_min=True, xlabel=True):
    """Step line of healthy cards; the y axis ends at the highest count seen, not at the number of lanes."""
    ax.step(ts, cnt, where="post", color=BLUE, linewidth=1.8, solid_joinstyle="round")
    ax.fill_between(ts, cnt, step="post", color=BLUE, alpha=0.10, linewidth=0)
    top = max(int(cnt.max()) if len(cnt) else 1, 1)
    ax.set_ylim(0, top + (1.5 if label_min else 0.5))
    ax.set_yticks(list(range(0, top + 1, 1 if top <= 5 else 2)))
    style_axes(ax, "y")
    time_axis(ax, t0, t1, label=xlabel)
    ax.tick_params(axis="y", labelsize=ticksize)
    if label_min and len(cnt) and cnt.min() < np.median(cnt):
        i = int(np.argmax(cnt == cnt.min()))
        ax.plot([ts[i], ts[i]], [cnt[i], top + 0.15], color=MUTED, linewidth=1)
        left = ts[i] < t0 + 0.6 * (t1 - t0)
        ax.text(ts[i] + (t1 - t0) * (0.01 if left else -0.01), top + 0.3, "lowest: %d healthy" % cnt.min(), fontsize=ticksize,
                color=INK, ha="left" if left else "right", va="bottom")


def save(fig, path, dpi=200):
    fig.savefig(path, dpi=dpi)


def rect_in(fig, left, bottom_in, width, height_in):
    """Axes rectangle with the vertical position and size given in inches from the bottom of the figure."""
    H = fig.get_figheight()
    return [left, bottom_in / H, width, height_in / H]


def chart_fleet(m, ctx, path):
    plt = mpl()
    cards = lane_cards(ctx)
    if not cards or ctx["t0"] is None:
        return False
    n = len(cards)
    pitch = 0.2
    lanes_h, count_h = n * pitch, 1.0
    H = 0.6 + 0.4 + lanes_h + 0.3 + count_h + 0.65
    fig = plt.figure(figsize=(8, H))
    fig.text(0.02, 1 - 0.1 / H, "Cards over time", fontsize=18, fontweight="bold", va="top", color=INK)
    hs, ls = lane_artists(plt, cards, ctx["t0"], ctx["t1"], bool(ctx["stops"]))
    legend_row(fig, hs, ls, y=1 - 0.5 / H, size=11)
    ax2 = fig.add_axes(rect_in(fig, 0.14, 0.65, 0.83, count_h))
    ax1 = fig.add_axes(rect_in(fig, 0.14, 0.65 + count_h + 0.3, 0.83, lanes_h), sharex=ax2)
    draw_lanes(ax1, cards, ctx["stops"], ctx["t0"], ctx["t1"], ticksize=12, msize=7)
    fig.text(0.02, (0.65 + count_h + 0.1) / H, "Healthy cards", fontsize=12, color=INK2, va="bottom")
    if "count" in ctx:
        draw_count(ax2, ctx["count_ts"], ctx["count"], ctx["t0"], ctx["t1"])
    save(fig, path)
    plt.close(fig)
    return True


def per_minute(requests, resumes, t0, t1):
    n = max(1, int((t1 - t0) // 60))  # whole minutes only: a short last minute would look like a drop
    fin, fail, res = np.zeros(n), np.zeros(n), np.zeros(n)
    for r in requests:
        i = int((r["end"] - t0) // 60)
        if 0 <= i < n:
            (fail if r["failed"] else fin)[i] += 1
    for r in resumes:
        i = int((r["end"] - t0) // 60)
        if 0 <= i < n:
            res[i] += 1
    return fin, fail, res


def chart_requests(m, ctx, path):
    plt = mpl()
    if not ctx["requests"] or ctx["t0"] is None:
        return False
    t0, t1 = ctx["t0"], ctx["t1"]
    fin, fail, res = per_minute(ctx["requests"], ctx["resumes"], t0, t1)
    x = t0 + 60 * np.arange(len(fin)) + 30
    fig = plt.figure(figsize=(8, 5.4))
    fig.text(0.02, 0.965, "Requests per minute", fontsize=18, fontweight="bold", va="top")
    from matplotlib.patches import Patch
    legend_row(fig, [Patch(facecolor=BLUE, linewidth=0), Patch(facecolor=ORANGE, linewidth=0), Patch(facecolor=AQUA, linewidth=0)],
               ["finished", "failed", "stream resumed"], y=0.915, size=12)
    ax1 = fig.add_axes([0.12, 0.34, 0.85, 0.50])
    ax2 = fig.add_axes([0.12, 0.10, 0.85, 0.17], sharex=ax1)
    w = 60 * 0.82
    ax1.bar(x, fin, width=w, color=BLUE, linewidth=0)
    ax1.bar(x, fail, width=w, bottom=fin, color=ORANGE, linewidth=0)
    style_axes(ax1, "y")
    ax1.tick_params(axis="x", labelbottom=False)
    ax1.set_ylabel("requests finished\nor failed")
    ax1.set_xlim(t0, t1)
    ax2.bar(x, res, width=w, color=AQUA, linewidth=0)
    style_axes(ax2, "y")
    ax2.set_ylabel("resumed", fontsize=12)
    ax2.set_ylim(0, max(res.max() * 1.15, 2))
    ax2.yaxis.set_major_locator(plt.MaxNLocator(2, integer=True))
    time_axis(ax2, t0, t1)
    save(fig, path)
    plt.close(fig)
    return True


def chart_pauses(m, ctx, path):
    """Pause lengths on a log scale, so a group around a second and a group of tens of seconds both show."""
    plt = mpl()
    from matplotlib.ticker import FuncFormatter, NullFormatter
    pa = [p / 1000.0 for p in ctx["pauses_ms"] if p and p > 0]
    fig = plt.figure(figsize=(8, 4.8))
    fig.text(0.02, 0.965, "How long a resumed answer paused", fontsize=18, fontweight="bold", va="top")
    ax = fig.add_axes([0.11, 0.17, 0.86, 0.62])
    if not pa:
        ax.text(0.5, 0.5, "No resumed streams in this run", ha="center", va="center", fontsize=15, color=INK2, transform=ax.transAxes)
        ax.set_axis_off()
        save(fig, path)
        plt.close(fig)
        return True
    lo, hi = min(pa), max(pa)
    bins = np.logspace(math.log10(lo * 0.93), math.log10(hi * 1.08), max(8, int(24 * math.log10(hi / lo * 1.16)) + 1))
    cnts, _, _ = ax.hist(pa, bins=bins, color=BLUE, linewidth=0.0, rwidth=1.0)
    ax.set_xscale("log")
    style_axes(ax, "y")
    ticks = [t for t in (0.5, 1, 2, 5, 10, 20, 50, 100, 200) if lo * 0.8 <= t <= hi * 1.3]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "%g" % v))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlim(lo * 0.8, hi * 1.3)
    ax.set_xlabel("Pause before the answer continued, seconds (log scale)")
    ax.set_ylabel("streams resumed")
    ax.yaxis.set_major_locator(plt.MaxNLocator(5, integer=True))
    ymax = max(cnts.max(), 1)
    ax.set_ylim(0, ymax * 1.32)
    p50 = float(np.median(pa))
    ax.axvline(p50, color=INK, linewidth=1.5)
    ax.text(p50 * 1.06, ymax * 1.22, "median %.2f s" % p50, fontsize=13, ha="left", va="center", color=INK)
    short = [p for p in pa if p < 2]
    long_ = [p for p in pa if p > 10]
    if short and long_:
        ax.text(p50 * 0.93, ymax * 1.22, "%d under 2 s (%.0f%%)" % (len(short), 100 * len(short) / len(pa)),
                fontsize=12.5, ha="right", va="center", color=INK2)
        ax.text(hi * 1.22, cnts.max() * 0.0 + max(cnts[bins[:-1] >= min(long_) * 0.9].max() * 1.12, ymax * 0.2), "%d over 10 s,\nlongest %.0f s" % (len(long_), max(long_)),
                fontsize=12.5, ha="right", va="bottom", color=INK2, linespacing=1.2)
    else:
        ax.text(0.99, 0.97, "%d pauses, longest %.2f s" % (len(pa), hi), transform=ax.transAxes, ha="right", va="top",
                fontsize=12, color=INK2)
    save(fig, path)
    plt.close(fig)
    return True


def wrap_label(lab, width):
    """Wrap each explicit line separately, so a deliberate line break stays."""
    return "\n".join(textwrap.fill(l, width, break_on_hyphens=False) for l in lab.split("\n"))


def poster_numbers(m):
    """Six headline numbers in the order of the post: (value, label)."""
    out = []
    if m["percent_finished"] is not None:
        out.append(("%.1f%%" % m["percent_finished"], "of %s requests finished" % fmt_num(m["requests_total"])))
    if m["failed_stream_cut"] is not None:
        out.append((fmt_num(m["failed_stream_cut"]), "streams cut mid-answer"))
    if m["streams_resumed"] is not None:
        mp = m["resume_pause_p50_ms"]
        out.append((fmt_num(m["streams_resumed"]), "streams resumed" + ("" if mp is None else "\nmedian pause %.1f s" % (mp / 1000.0))))
    if m["stops_total"] is not None:
        out.append((fmt_num(m["stops_total"]), "cards taken by the marketplace"))
    if m["cards_never_served"] is not None:
        out.append(("%d of %d" % (m["cards_never_served"], m["cards_tracked"]), "tracked cards never started serving"))
    cph = m["cost_per_healthy_card_hour"]
    if cph is not None:
        lab = "per healthy card-hour"
        if "ondemand_price_per_hour" in m:
            lab += "; on-demand $%.2f" % m["ondemand_price_per_hour"]
        out.append(("$%.2f" % cph, lab))
    return out


def default_title(m):
    d = m["load_duration_s"]
    return "%d minutes of LLM serving on interruptible GPUs" % round(d / 60) if d else "LLM serving on interruptible GPUs"


def default_subtitle(m, ctx):
    day = ""
    if ctx["t0"] is not None:
        d = dt.datetime.fromtimestamp(ctx["t0"], dt.timezone.utc)
        day = "%d %s %d (UTC)" % (d.day, d.strftime("%b"), d.year)
    nw = ctx.get("n_workers")
    gpu = ctx.get("gpu")
    parts = ["%s streaming clients" % nw if nw else "streaming clients", "one router",
             "%s instances" % gpu if gpu else "rented instances"]
    return ", ".join(parts) + (", " + day if day else "")


def chart_poster(m, ctx, path):
    plt = mpl()
    fig = plt.figure(figsize=(8, 4.5))
    H = 4.5
    fig.text(0.03, 0.965, ctx.get("title") or default_title(m), fontsize=19.5, fontweight="bold", va="top")
    fig.text(0.03, 0.895, ctx.get("subtitle") or default_subtitle(m, ctx), fontsize=10.5, color=INK2, va="top")
    nums = poster_numbers(m)
    k = max(len(nums), 1)
    for i, (v, lab) in enumerate(nums):
        x = 0.03 + i * (0.94 / k)
        fig.text(x, 0.848, v, fontsize=20, fontweight="bold", va="top", color=INK)
        fig.text(x, 0.777, wrap_label(lab, 18), fontsize=8.4, va="top", color=INK2, linespacing=1.2)
    cards = lane_cards(ctx)
    if cards and ctx["t0"] is not None:
        n = len(cards)
        hs, ls = lane_artists(plt, cards, ctx["t0"], ctx["t1"], bool(ctx["stops"]))
        legend_row(fig, hs, ls, y=0.672, x=0.03, size=9, ncol=len(ls))
        lanes_h, count_h = 1.9, 0.36
        ax1 = fig.add_axes(rect_in(fig, 0.105, 0.46 + count_h + 0.08, 0.875, lanes_h))
        draw_lanes(ax1, cards, ctx["stops"], ctx["t0"], ctx["t1"], ticksize=11, msize=5.5)
        if "count" in ctx:
            ax2 = fig.add_axes(rect_in(fig, 0.105, 0.46, 0.875, count_h), sharex=ax1)
            draw_count(ax2, ctx["count_ts"], ctx["count"], ctx["t0"], ctx["t1"], ticksize=8.5, label_min=False, xlabel=False)
            ax2.tick_params(axis="x", labelsize=8.5)
            ax2.set_yticks([0, int(ctx["count"].max())])
            fig.text(0.03, (0.46 + count_h / 2) / H, "healthy", fontsize=8.5, color=INK2, va="center")
        foot = "Times in UTC."
        if m.get("cards_not_tracked"):
            foot = "%d of %d cards tracked; the hatched card has no data. Times in UTC." % (m["cards_tracked"], m["cards_rented"])
        fig.text(0.03, 0.06 / H, foot, fontsize=8, color=MUTED, va="bottom")
    save(fig, path)
    plt.close(fig)
    return True


# ----------------------------------------------------------------------------------------------
# 4. Time-lapse
# ----------------------------------------------------------------------------------------------

def have_h264():
    exe = shutil.which("ffmpeg")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return exe if "libx264" in out else None


class Timelapse:
    """Persistent figure; render(T) draws the state at test time T."""

    W, H, DPI = 1280, 720, 100

    def __init__(self, m, ctx):
        plt = mpl()
        from matplotlib.collections import PolyCollection
        self.plt, self.m, self.ctx = plt, m, ctx
        self.t0, self.t1 = ctx["t0"], ctx["t1"]
        self.cards = lane_cards(ctx)
        n = len(self.cards)
        self.n = n
        fig = plt.figure(figsize=(self.W / self.DPI, self.H / self.DPI), dpi=self.DPI)
        self.fig = fig
        fig.text(0.03, 0.95, ctx.get("title") or default_title(m), fontsize=22, fontweight="bold", va="top")
        self.clock = fig.text(0.97, 0.95, "", fontsize=24, ha="right", va="top", color=INK2)
        # counters
        self.tiles = []
        names = ["requests finished", "streams resumed", "requests failed", "money spent"]
        for i, nm in enumerate(names):
            x = 0.03 + i * 0.245
            v = fig.text(x, 0.835, "", fontsize=34, fontweight="bold", va="top")
            fig.text(x, 0.745, nm, fontsize=15, color=INK2, va="top")
            self.tiles.append(v)
        hs, ls = lane_artists(plt, self.cards, self.t0, self.t1, bool(ctx["stops"]))
        legend_row(fig, hs, ls, y=0.685, x=0.03, size=14)
        self.ax = fig.add_axes([0.11, 0.155, 0.865, 0.47])
        draw_lanes(self.ax, self.cards, [], self.t0, self.t1, ticksize=14, show_stops=False, draw_segments=False)
        self.ax.set_ylim(-0.6, n - 0.2)
        # time axis labels under the lanes
        self.ax.tick_params(axis="x", labelbottom=True, labelsize=14, length=0)
        time_axis(self.ax, self.t0, self.t1, label=False)
        self.ax.grid(axis="x", color=GRID, linewidth=0.8)
        self.ax.set_axisbelow(True)
        self.coll = PolyCollection([], linewidths=0)
        self.ax.add_collection(self.coll)
        self.hcoll = PolyCollection([], facecolors=SURFACE, edgecolors=MUTED, linewidths=0, hatch="////")
        self.ax.add_collection(self.hcoll)
        self.cursor = self.ax.axvline(self.t0, color=INK, linewidth=1.6, zorder=6)
        (self.marks,) = self.ax.plot([], [], linestyle="", marker="v", markersize=11, color=INK,
                                     markeredgecolor=SURFACE, markeredgewidth=1.2, clip_on=False, zorder=7)
        self.caption = fig.text(0.03, 0.045, "", fontsize=19, va="center", color=INK)
        self.final = self._final_panel(fig)
        h = 0.62
        self.h = h
        self.segs = [(n - 1 - i, clip_segs(c["segs"], self.t0, self.t1)) for i, c in enumerate(self.cards)]
        self.hspans = [(n - 1 - i, untracked_span(c, self.t0, self.t1)) for i, c in enumerate(self.cards)
                       if not c["segs"] and c["created"] is not None]
        idx = {c["id"]: n - 1 - i for i, c in enumerate(self.cards)}
        self.stop_pts = [(s["ts"], idx[s["card"]] + h / 2 + 0.16) for s in ctx["stops"] if s["card"] in idx]
        self.events = caption_events(self.cards, ctx["stops"], self.t0, self.t1)
        r = ctx["requests"]
        self.fin_ends = np.array(sorted(x["end"] for x in r if not x["failed"]))
        self.fail_ends = np.array(sorted(x["end"] for x in r if x["failed"]))
        self.res_ends = np.array(sorted(x["end"] for x in ctx["resumes"]))
        self.money_ts, self.money = ctx["money_ts"], ctx["money_spent"]

    def _final_panel(self, fig):
        """Six headline numbers drawn over the counter row in the last frames."""
        from matplotlib.patches import Rectangle
        arts = [fig.add_artist(Rectangle((0, 0.70), 1, 0.20, transform=fig.transFigure, facecolor=SURFACE, linewidth=0, zorder=10))]
        nums = poster_numbers(self.m)
        k = max(len(nums), 1)
        for i, (v, lab) in enumerate(nums):
            x = 0.03 + i * (0.94 / k)
            arts.append(fig.text(x, 0.885, v, fontsize=27, fontweight="bold", va="top", zorder=11))
            arts.append(fig.text(x, 0.815, wrap_label(lab, 19), fontsize=11, va="top", color=INK2, linespacing=1.15, zorder=11))
        for a in arts:
            a.set_visible(False)
        return arts

    def counts(self, T):
        fin = int(np.searchsorted(self.fin_ends, T, side="right"))
        fail = int(np.searchsorted(self.fail_ends, T, side="right"))
        res = int(np.searchsorted(self.res_ends, T, side="right"))
        money = float(np.interp(T, self.money_ts, self.money)) if self.money_ts is not None else None
        return fin, res, fail, money

    def render(self, T, hold_s, final=False):
        verts, cols = [], []
        for y, segs in self.segs:
            for a, b, s in segs:
                if a >= T:
                    continue
                b = min(b, T)
                verts.append([(a, y - self.h / 2), (b, y - self.h / 2), (b, y + self.h / 2), (a, y + self.h / 2)])
                cols.append(STATE_COLORS[s])
        self.coll.set_verts(verts)
        self.coll.set_facecolor(cols)
        hv = []
        for y, span in self.hspans:
            if span and span[0] < T:
                a, b = span[0], min(span[1], T)
                hv.append([(a, y - self.h / 2), (b, y - self.h / 2), (b, y + self.h / 2), (a, y + self.h / 2)])
        self.hcoll.set_verts(hv)
        for a in self.final:
            a.set_visible(final)
        self.cursor.set_xdata([T, T])
        pts = [p for p in self.stop_pts if p[0] <= T]
        self.marks.set_data([p[0] for p in pts], [p[1] for p in pts])
        self.clock.set_text("%s UTC" % hhmm(T, sec=False))
        fin, res, fail, money = self.counts(T)
        self.tiles[0].set_text(fmt_num(fin))
        self.tiles[1].set_text(fmt_num(res))
        self.tiles[2].set_text(fmt_num(fail))
        self.tiles[3].set_text("n/a" if money is None else "$%.2f" % money)
        cap = ""
        if final:
            cap = "Final numbers after %s, times in UTC" % fmt_dur(self.t1 - self.t0)
        else:
            k = bisect.bisect_right([e[0] for e in self.events], T) - 1
            if k >= 0 and T - self.events[k][0] <= hold_s:
                cap = self.events[k][1]
        self.caption.set_text(cap)
        self.fig.canvas.draw()
        return bytes(self.fig.canvas.buffer_rgba())


def make_timelapse(m, ctx, out_dir, seconds=34.0, hold=2.0, fps=30, log=print):
    cards = [c for c in ctx["cards"] if c["segs"]]  # no tracked card, nothing to replay
    if not cards or ctx["t0"] is None:
        log("timelapse: no card data, skipped")
        return None
    tl = Timelapse(m, ctx)
    n_play = int((seconds - hold) * fps)
    n_hold = int(hold * fps)
    hold_s = 2.6 * (tl.t1 - tl.t0) / (seconds - hold)  # a caption stays 2.6 s of video time
    ff = have_h264()
    mp4 = os.path.join(out_dir, "timelapse.mp4")
    if ff:
        proc = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", "1280x720",
                                 "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                                 "-preset", "medium", "-movflags", "+faststart", mp4], stdin=subprocess.PIPE)
        for i in range(n_play):
            T = tl.t0 + (tl.t1 - tl.t0) * i / max(n_play - 1, 1)
            proc.stdin.write(tl.render(T, hold_s))
        last = tl.render(tl.t1, hold_s, final=True)
        for _ in range(n_hold):
            proc.stdin.write(last)
        proc.stdin.close()
        proc.wait()
        return mp4 if proc.returncode == 0 else None
    # fallback: GIF with Pillow, 10 fps, half size
    from PIL import Image
    log("timelapse: no H.264 encoder, writing timelapse.gif")
    gif_fps = 10
    frames = []
    total = int(seconds * gif_fps)
    n_p = int((seconds - hold) * gif_fps)
    for i in range(total):
        final = i >= n_p
        T = tl.t1 if final else tl.t0 + (tl.t1 - tl.t0) * i / max(n_p - 1, 1)
        buf = tl.render(T, hold_s, final=final)
        im = Image.frombytes("RGBA", (1280, 720), buf).convert("RGB").resize((960, 540), Image.LANCZOS)
        frames.append(im.convert("P", palette=Image.ADAPTIVE, colors=64))
    path = os.path.join(out_dir, "timelapse.gif")
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / gif_fps), loop=0, optimize=True)
    return path


# ----------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------

def run(args, log=print):
    d = args.dir
    out = args.out or os.path.join(d, "report")
    os.makedirs(out, exist_ok=True)
    data = load_dir(d)
    m, ctx = compute_metrics(data, {"min_down": args.min_down}, args.ondemand_price)
    ctx["title"], ctx["subtitle"] = getattr(args, "title", None), getattr(args, "subtitle", None)
    ctx["n_workers"] = len({r.get("worker") for r in data["run"]}) or None
    ctx["gpu"] = (data["state"] or {}).get("gpu")
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary_text(m))
    made = ["summary.json", "summary.txt"]
    for name, fn in (("fleet.png", chart_fleet), ("requests.png", chart_requests), ("pauses.png", chart_pauses),
                     ("poster.png", chart_poster)):
        try:
            ok = fn(m, ctx, os.path.join(out, name))
        except Exception as e:  # one broken chart must not stop the others
            log("%s: failed (%s: %s)" % (name, type(e).__name__, e))
            ok = False
        made.append(name) if ok else log("%s: not enough data, skipped" % name)
    if not args.no_video:
        try:
            p = make_timelapse(m, ctx, out, log=log)
            if p:
                made.append(os.path.basename(p))
        except Exception as e:
            log("timelapse: failed (%s: %s)" % (type(e).__name__, e))
    log("wrote to %s: %s" % (out, ", ".join(made)))
    return m


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="folder with run.jsonl, samples-*.jsonl, events.jsonl, ...")
    ap.add_argument("--out", help="output folder (default: DIR/report)")
    ap.add_argument("--ondemand-price", type=float, help="on-demand dollars per card-hour, adds a cost comparison")
    ap.add_argument("--min-down", type=float, default=DEFAULTS["min_down"],
                    help="a loss of health shorter than this many seconds is a probe blip, not a stop (default 30)")
    ap.add_argument("--title", help="poster / time-lapse title (default: N minutes of LLM serving on interruptible GPUs)")
    ap.add_argument("--subtitle", help="poster subtitle (default is built from the data: clients, GPU model, date)")
    ap.add_argument("--no-video", action="store_true", help="skip timelapse.mp4")
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
