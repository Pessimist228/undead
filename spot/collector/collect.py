"""Vast.ai public offer price collector (stdlib only, Python 3.9+).

One run = one tick: query single-GPU offers for every GPU in GPUS, for
on-demand and interruptible ("bid") pricing, and append the raw rows as one new
gzip member to <out>/vast-YYYY-MM-DD.jsonl.gz. Failures are recorded as error
lines so gaps stay visible. Run it from a scheduler every 10 minutes.
"""
import argparse
import gzip
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone

BASE = "https://console.vast.ai/api/v0"
UA = {"User-Agent": "Mozilla/5.0", "Accept": "application/json", "Content-Type": "application/json"}
COLS = ["id", "machine_id", "host_id", "gpu_name", "gpu_ram", "dph_total", "dph_base", "min_bid",
        "reliability2", "geolocation", "hosting_type", "vms_enabled", "driver_version",
        "cuda_max_good", "inet_down", "inet_up", "disk_space", "storage_cost", "inet_down_cost",
        "inet_up_cost", "is_bid", "duration", "direct_port_count", "compute_cap", "cpu_ram",
        "verification", "num_gpus", "rented"]
GPUS = ["RTX 3090", "RTX 4090", "RTX 5090", "RTX A6000", "L40S", "RTX PRO 6000 WS", "RTX 6000Ada",
        "A100 SXM4", "H100 SXM"]
KINDS = ("on-demand", "bid")
LIMIT = 200
FMT = "%Y-%m-%dT%H:%M:%SZ"
LOCK, LOG = "collector.lock", "collector.log"
LOCK_MAX_AGE = timedelta(minutes=8)
LOG_MAX_BYTES = 1024 * 1024
RUN_DEADLINE = 300  # seconds; past this the remaining queries are skipped so ticks never overlap


def query(gpu, kind):  # kind is "on-demand" or "bid"
    return {"verified": {"eq": True}, "external": {"eq": False}, "rentable": {"eq": True},
            "rented": {"eq": False}, "num_gpus": {"eq": 1}, "gpu_name": {"eq": gpu},
            "order": [["dph_total", "asc"]], "type": kind, "limit": LIMIT, "allocated_storage": 50}


def http_put_json(url, payload, timeout):
    """PUT a JSON body and return the decoded JSON reply (replaced by a fake in tests)."""
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=UA, method="PUT")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def now():
    return datetime.now(timezone.utc).replace(microsecond=0)


def fetch(gpu, kind):
    """Return the offers list. Two retries (3 s, then 8 s pause); the last error is raised."""
    body = {"select_cols": COLS, "q": query(gpu, kind)}
    for pause in (3, 8, None):
        try:
            offers = http_put_json(BASE + "/search/asks/", body, 30)["offers"]
            if not isinstance(offers, list):
                raise ValueError("'offers' is not a list")
            return offers
        except Exception:
            if pause is None:
                raise
            time.sleep(pause)


def take_lock(out, tick):
    """Create the lock file; False if a lock younger than 8 minutes exists."""
    path = os.path.join(out, LOCK)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            with open(path) as f:
                started = datetime.strptime(f.read().strip(), FMT).replace(tzinfo=timezone.utc)
            if now() - started < LOCK_MAX_AGE:
                return False
        except (OSError, ValueError):
            pass  # unreadable or garbled lock counts as stale
        fd = os.open(path, os.O_WRONLY | os.O_TRUNC)
    with os.fdopen(fd, "w") as f:
        f.write(tick)
    return True


def release_lock(out, tick):
    path = os.path.join(out, LOCK)
    try:
        with open(path) as f:
            mine = f.read().strip() == tick
        if mine:
            os.remove(path)
    except OSError:
        pass


def append_gz(path, lines):
    """Append all lines as ONE new gzip member, so earlier members stay intact."""
    data = "".join(json.dumps(x, separators=(",", ":")) + "\n" for x in lines)
    with gzip.open(path, "ab") as f:
        f.write(data.encode("utf-8"))


def write_log(out, text):
    path = os.path.join(out, LOG)
    if os.path.exists(path) and os.path.getsize(path) > LOG_MAX_BYTES:
        os.replace(path, path + ".1")
    with open(path, "a") as f:
        f.write(text + "\n")


def run_once(out, gpus, verbose=False):
    """One tick. Returns the process exit code."""
    os.makedirs(out, exist_ok=True)
    tick = now().strftime(FMT)
    if not take_lock(out, tick):
        return 0  # previous run still going
    t0 = time.monotonic()
    try:
        lines, ok, bad, offers, trunc = [], 0, 0, 0, 0
        for i, (gpu, kind) in enumerate((g, k) for g in gpus for k in KINDS):
            if i:
                time.sleep(0.5)
            line = {"tick": tick, "gpu": gpu, "kind": kind}
            try:
                if time.monotonic() - t0 > RUN_DEADLINE:
                    raise TimeoutError("skipped: run deadline")
                rows = fetch(gpu, kind)
                n = len(rows)
                line.update(ts=now().strftime(FMT), ok=True, n=n, truncated=n >= LIMIT, cols=COLS,
                            rows=[[o.get(c) for c in COLS] for o in rows])
                ok, offers, trunc = ok + 1, offers + n, trunc + (n >= LIMIT)
            except Exception as e:
                line.update(ts=now().strftime(FMT), ok=False, err=("%s: %s" % (type(e).__name__, e))[:200])
                bad += 1
            lines.append(line)
        append_gz(os.path.join(out, "vast-%s.jsonl.gz" % tick[:10]), lines)
        write_log(out, "%s ok=%d failed=%d offers=%d truncated=%d duration=%.1fs"
                  % (tick, ok, bad, offers, trunc, time.monotonic() - t0))
        if verbose:
            print("%s ok=%d failed=%d offers=%d truncated=%d" % (tick, ok, bad, offers, trunc))
        return 0 if ok else 1
    finally:
        release_lock(out, tick)


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join(here, "data"))
    ap.add_argument("--gpus", help='comma separated override, e.g. "RTX 3090,L40S"')
    ap.add_argument("--loop", type=float, metavar="SECONDS", help="collect every SECONDS (default: once)")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)
    gpus = [g.strip() for g in a.gpus.split(",") if g.strip()] if a.gpus else GPUS
    if not a.loop:
        return run_once(a.out, gpus, a.verbose)
    while True:
        t = time.monotonic()
        try:
            run_once(a.out, gpus, a.verbose)
        except Exception as e:
            print("run failed: %r" % e, file=sys.stderr)
        time.sleep(max(1.0, a.loop - (time.monotonic() - t)))


if __name__ == "__main__":
    sys.exit(main())
