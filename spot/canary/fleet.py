#!/usr/bin/env python3
"""fleet.py: keep the router's replica registry equal to the set of healthy canary instances.

Each pass: `canary.py --dir DIR urls --json`, GET /admin/replicas, then
  PUT    i<ID>  {"url", "max_streams"}  for a healthy instance that is missing or whose url changed,
  DELETE i<ID>  for a registered instance that was not healthy in TWO consecutive passes, or no longer exists.
Only names of the form i<digits> are ever touched. One JSON line per pass goes to DIR/fleet.jsonl.
The loop ends when DIR/state.json says the run is finished; before exiting every i<digits> replica is deleted.
Standard library only. The admin token is read from a file and never printed or written.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

OURS = re.compile(r"i[0-9]+")
BAD_PASSES = 2


def iso(ts):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def run_canary(cmd):
    """The one place that starts a process. Returns stdout text; raises RuntimeError with a short reason."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        raise RuntimeError("timeout")
    except OSError as e:
        raise RuntimeError(type(e).__name__)
    if p.returncode != 0:
        raise RuntimeError("rc %d" % p.returncode)
    return p.stdout


def real_http(method, url, token, body=None):
    """The one place that talks to the router. Returns (status, parsed json or None); raises OSError-like on transport errors."""
    data = None if body is None else json.dumps(body).encode()
    h = {"Authorization": "Bearer " + token, "Accept": "application/json"}
    if data is not None:
        h["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            code, raw = r.status, r.read(1 << 20)
    except urllib.error.HTTPError as e:
        code, raw = e.code, e.read(1 << 20)
    try:
        return code, json.loads(raw.decode("utf-8", "replace"))
    except ValueError:
        return code, None


class Fleet:
    def __init__(self, a, token, runner=run_canary, http=real_http, sleep=time.sleep, now=time.time):
        self.a, self.token, self.runner, self.http, self.sleep, self.now = a, token, runner, http, sleep, now
        self.bad = {}  # replica name -> consecutive passes without a healthy instance
        self.admin = a.router_admin.rstrip("/")
        self.cmd = [sys.executable, a.canary, "--dir", a.dir] + (["--key-file", a.key_file] if a.key_file else []) \
            + ["urls", "--json"]

    # ---- inputs ----
    def instances(self, errors):
        """{id: {"status", "intended", "healthy", "url"}} from canary, or None when the call failed."""
        try:
            obj = json.loads(self.runner(self.cmd))
            out = {}
            for i in obj["instances"]:
                out[int(i["id"])] = {"status": i.get("status"), "intended": i.get("intended"),
                                     "healthy": bool(i.get("healthy")), "url": i.get("url")}
            return out
        except RuntimeError as e:
            errors.append("canary: %s" % e)
        except (ValueError, KeyError, TypeError):
            errors.append("canary: bad output")
        return None

    def registry(self, errors):
        """{name: view} of the router, or None when the call failed."""
        try:
            code, obj = self.http("GET", self.admin + "/admin/replicas", self.token)
        except Exception as e:  # noqa: BLE001 - only the class name is kept
            errors.append("router list: %s" % type(e).__name__)
            return None
        if code != 200 or not isinstance(obj, list):
            errors.append("router list: HTTP %s" % code)
            return None
        return {r["name"]: r for r in obj if isinstance(r, dict) and isinstance(r.get("name"), str)}

    def call(self, op, name, body, actions, errors):
        method = "PUT" if op == "put" else "DELETE"
        try:
            code, _ = self.http(method, "%s/admin/replicas/%s" % (self.admin, name), self.token, body)
        except Exception as e:  # noqa: BLE001 - only the class name is kept
            errors.append("%s %s: %s" % (op, name, type(e).__name__))
            return False
        if code not in (200, 201):
            errors.append("%s %s: HTTP %s" % (op, name, code))
            return False
        actions.append({"op": op, "name": name})
        return True

    def finished(self):
        try:
            with open(os.path.join(self.a.dir, "state.json")) as f:
                return json.load(f).get("phase") == "finished"
        except (OSError, ValueError):
            return False

    # ---- one pass ----
    def one_pass(self, cleanup=False):
        errors, actions = [], []
        t = self.now()
        inst = None if cleanup else self.instances(errors)
        reg = self.registry(errors)
        if reg is not None:
            if cleanup:
                for name in sorted(n for n in reg if OURS.fullmatch(n)):
                    if self.call("delete", name, None, actions, errors):
                        reg.pop(name)
            elif inst is not None:
                self.reconcile(inst, reg, actions, errors)
        healthy = sorted(i for i, v in (inst or {}).items() if v["healthy"])
        rec = {"t": iso(t), "ts": t, "healthy": healthy,
               "registered": sorted(reg) if reg is not None else None,
               "instances": {str(i): {"status": v["status"], "intended": v["intended"], "healthy": v["healthy"]}
                             for i, v in sorted((inst or {}).items())},
               "router": {n: {"in_flight": r.get("in_flight"), "state": r.get("state")}
                          for n, r in sorted((reg or {}).items())},
               "actions": actions, "errors": errors}
        with open(os.path.join(self.a.dir, "fleet.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    def reconcile(self, inst, reg, actions, errors):
        for name in sorted(n for n in reg if OURS.fullmatch(n)):
            v = inst.get(int(name[1:]))
            if v is not None and v["healthy"]:
                self.bad[name] = 0
                continue
            self.bad[name] = self.bad.get(name, 0) + 1
            if v is None or self.bad[name] >= BAD_PASSES:
                if self.call("delete", name, None, actions, errors):
                    reg.pop(name)
                    self.bad.pop(name, None)
        for iid, v in sorted(inst.items()):
            name = "i%d" % iid
            if not v["healthy"] or not v["url"]:
                continue
            cur = reg.get(name)
            if cur is None or cur.get("url") != v["url"]:
                if self.call("put", name, {"url": v["url"], "max_streams": self.a.max_streams}, actions, errors):
                    reg[name] = {"name": name, "url": v["url"], "state": "active", "in_flight": 0}
                    self.bad[name] = 0
        for name in [n for n in self.bad if n not in reg]:
            del self.bad[name]

    def run(self):
        while True:
            if self.finished():
                self.one_pass(cleanup=True)
                return 0
            self.one_pass()
            if self.a.once:
                return 0
            self.sleep(self.a.interval)


def parse(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="keep the router's replica registry equal to the healthy instances")
    p.add_argument("--dir", required=True, help="canary state directory")
    p.add_argument("--router-admin", dest="router_admin", required=True, help="e.g. http://127.0.0.1:8089")
    p.add_argument("--token-file", dest="token_file", required=True)
    p.add_argument("--interval", type=float, default=15.0)
    p.add_argument("--max-streams", dest="max_streams", type=int, default=16)
    p.add_argument("--once", action="store_true")
    p.add_argument("--canary", default=os.path.join(here, "canary.py"))
    p.add_argument("--key-file", dest="key_file", default=None, help="passed to canary.py")
    a = p.parse_args(argv)
    if a.interval < 0 or not 0 <= a.max_streams <= 1024:
        p.error("interval must be >= 0 and max-streams 0..1024")
    return a


def main(argv=None, **kw):
    a = parse(argv)
    try:
        with open(a.token_file) as f:
            token = f.read().strip()
    except OSError:
        print("fleet: cannot read the token file", file=sys.stderr)
        return 2
    if not token:
        print("fleet: the token file is empty", file=sys.stderr)
        return 2
    return Fleet(a, token, **kw).run()


if __name__ == "__main__":
    sys.exit(main())
