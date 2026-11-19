#!/usr/bin/env python3
"""Streaming load generator for an OpenAI-compatible endpoint (built for the undead router).

Keeps --streams concurrent streaming chat completions going for --duration seconds. Each worker takes the next
prompt from a built-in list and sends ONLY: model, messages, stream, max_tokens, temperature, seed.
One JSON line per request goes to --out (no prompt and no answer text), a summary goes to stdout.
A request still running when the duration ends may finish for up to 60 more seconds.
"""
import argparse
import asyncio
import datetime
import json
import math
import sys
import time

import aiohttp

GRACE = 60.0
GAP_REPORT_MS = 1000.0
TEMPERATURE = 0.7
SEED_BASE = 1000

PROMPTS = [
    "Write a detailed, long story about a lighthouse keeper who discovers that the sea has started to answer back.",
    "Explain how a modern CPU executes instructions, from fetch to retirement, in as much depth as you can.",
    "Write a long essay on the causes and consequences of the Industrial Revolution in Britain.",
    "Describe, step by step and in great detail, how to bake sourdough bread at home, including common mistakes.",
    "Write a long tutorial on building a REST API in Python, with explanations of every design decision.",
    "Tell a long tale about two rival merchants in a medieval port city who are forced to share a ship.",
    "Explain thoroughly how vaccines train the immune system, covering the main vaccine types.",
    "Write an extensive travel guide to a fictional island, covering history, food, villages and local customs.",
    "Discuss in depth how public-key cryptography works, from the math intuition to real protocols.",
    "Write a long, vivid account of a mountain expedition that goes wrong, told from three different perspectives.",
]


def utc_iso(ts=None):
    d = datetime.datetime.now(datetime.timezone.utc) if ts is None else \
        datetime.datetime.fromtimestamp(ts, datetime.timezone.utc)
    return d.strftime("%Y-%m-%dT%H:%M:%S.") + "%03dZ" % (d.microsecond // 1000)


def percentile(values, p):
    """Nearest-rank percentile; None for an empty list."""
    if not values:
        return None
    v = sorted(values)
    return v[max(0, math.ceil(p / 100.0 * len(v)) - 1)]


class Shared:
    def __init__(self, args):
        self.args = args
        self.counter = 0
        self.records = []
        self.gaps = []  # (utc_iso, worker, gap_ms) for every gap above the report threshold
        self.out = None

    def take(self):
        """The next (request number, prompt, seed). Called without awaiting, so workers never interleave here."""
        self.counter += 1
        n = self.counter
        return n, PROMPTS[(n - 1) % len(PROMPTS)], SEED_BASE + n

    def done_taking(self):
        return self.args.max_requests is not None and self.counter >= self.args.max_requests


async def one_request(sh, session, worker, n, prompt, seed, hard_deadline):
    a = sh.args
    body = {"model": a.model, "messages": [{"role": "user", "content": prompt}], "stream": True,
            "max_tokens": a.max_tokens, "temperature": TEMPERATURE, "seed": seed}
    rec = {"worker": worker, "request": n, "start": utc_iso(), "ttfc_ms": None, "chunks": 0, "duration_ms": None,
           "max_gap_ms": None, "max_gap_end": None, "finish_reason": None, "status": None, "error": None,
           "chars": 0}
    t0 = time.monotonic()
    last = None
    gaps = []

    async def stream():
        nonlocal last
        async with session.post(a.url.rstrip("/") + "/v1/chat/completions", json=body,
                                headers={"Accept": "text/event-stream"}) as resp:
            rec["status"] = resp.status
            if resp.status != 200:
                await resp.content.read(1000)  # drain a little; the text is not kept
                return
            async for raw in resp.content:
                line = raw.strip()
                if not line.startswith(b"data:"):
                    continue  # blank lines and ": keepalive" comments
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except ValueError:
                    continue
                choices = obj.get("choices") if isinstance(obj, dict) else None
                if not choices:
                    continue
                ch = choices[0]
                text = (ch.get("delta") or {}).get("content")
                if text:
                    now = time.monotonic()
                    wall = utc_iso()
                    if last is None:
                        rec["ttfc_ms"] = round((now - t0) * 1000, 1)
                    else:
                        gap = (now - last) * 1000
                        gaps.append((wall, gap))
                    last = now
                    rec["chunks"] += 1
                    rec["chars"] += len(text)
                if ch.get("finish_reason"):
                    rec["finish_reason"] = ch["finish_reason"]

    try:
        await asyncio.wait_for(stream(), max(1.0, hard_deadline - time.monotonic()))
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001 - any failure is a data point, not a crash
        rec["error"] = type(e).__name__
    rec["duration_ms"] = round((time.monotonic() - t0) * 1000, 1)
    if gaps:
        wall, gap = max(gaps, key=lambda g: g[1])
        rec["max_gap_ms"], rec["max_gap_end"] = round(gap, 1), wall
        sh.gaps.extend((w, worker, round(g, 1)) for w, g in gaps if g > GAP_REPORT_MS)
    rec["failed"] = rec["status"] != 200 or rec["error"] is not None or rec["finish_reason"] is None
    sh.records.append(rec)
    out = {k: v for k, v in rec.items() if k != "failed"}
    sh.out.write(json.dumps(out) + "\n")
    sh.out.flush()


async def worker_loop(sh, session, worker, end, hard_deadline):
    while time.monotonic() < end and not sh.done_taking():
        n, prompt, seed = sh.take()
        await one_request(sh, session, worker, n, prompt, seed, hard_deadline)


async def run(args):
    sh = Shared(args)
    t_start = time.monotonic()
    end, hard = t_start + args.duration, t_start + args.duration + GRACE
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=10)
    with open(args.out, "w") as f:
        sh.out = f
        async with aiohttp.ClientSession(timeout=timeout) as session:
            await asyncio.gather(*(worker_loop(sh, session, w, end, hard) for w in range(1, args.streams + 1)))
    return sh


def summary(records, gaps):
    ok = [r for r in records if not r["failed"]]
    bad = [r for r in records if r["failed"]]
    ttfc = [r["ttfc_ms"] for r in records if r["ttfc_ms"] is not None]
    rates = [r["chunks"] / (r["duration_ms"] / 1000.0) for r in records
             if r["chunks"] and r["duration_ms"]]

    def f(x, nd=1):
        return "n/a" if x is None else "%.*f" % (nd, x)
    lines = ["requests finished: %d, failed: %d" % (len(ok), len(bad)),
             "time to first chunk ms: p50 %s, p95 %s" % (f(percentile(ttfc, 50)), f(percentile(ttfc, 95))),
             "chunks per second per stream (chunks / total duration): p50 %s" % f(percentile(rates, 50), 2)]
    errs = {}
    for r in bad:
        k = "HTTP %s" % r["status"] if r["status"] not in (200, None) else (r["error"] or "no finish_reason")
        errs[k] = errs.get(k, 0) + 1
    if errs:
        lines.append("failures: " + ", ".join("%s x%d" % kv for kv in sorted(errs.items())))
    lines.append("gaps longer than %d ms (UTC time, worker, gap ms): %d" % (GAP_REPORT_MS, len(gaps)))
    lines.extend("%s, %s, %.0f" % g for g in sorted(gaps))
    return "\n".join(lines)


def parse(argv=None):
    p = argparse.ArgumentParser(description="streaming load for an OpenAI-compatible endpoint")
    p.add_argument("--url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--streams", type=int, default=8)
    p.add_argument("--duration", type=float, default=240.0)
    p.add_argument("--max-tokens", dest="max_tokens", type=int, default=500)
    p.add_argument("--out", required=True)
    p.add_argument("--max-requests", dest="max_requests", type=int, default=None, help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if a.streams < 1 or a.duration <= 0 or a.max_tokens < 1:
        p.error("streams, duration and max-tokens must be positive")
    return a


def main(argv=None):
    sh = asyncio.run(run(parse(argv)))
    print(summary(sh.records, sh.gaps))
    return 0


if __name__ == "__main__":
    sys.exit(main())
