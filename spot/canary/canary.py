#!/usr/bin/env python3
"""spot-canary: rent a few Vast.ai interruptible (bid) GPU instances, watch them with an
HTTP heartbeat, report how often they are taken away. Spends real money: read the safety notes.

Endpoint sources (vast-cli vast.py, github.com/vast-ai/vast-cli, fetched 2026-11-19):
  search offers  PUT    /api/v0/search/asks/            search__offers() "--new" branch (+ live observation)
  create         PUT    /api/v0/asks/{offer_id}/        create__instance()  -> {"success":true,"new_contract":ID}
  list           GET    /api/v1/instances/?select_filters=..&order_by=..&limit=25[&after_token=..]
                                                        _fetch_all_instances_v1() (v0 list is deprecated, 410)
  destroy        DELETE /api/v0/instances/{id}/ {}      destroy_instance()
  change bid     PUT    /api/v0/instances/bid_price/{id}/ {"client_id":"me","price":P}   change__bid()
  credit         GET    /api/v0/users/current/          show__user() (field "credit" seen live)
Port mapping: create() env {"-p 8080:8080": "1"} (parse_env); instance row gives public_ipaddr and
ports["8080/tcp"][0]["HostPort"] (machine tester code in vast.py).
UNVERIFIED (see README in final report): runtype "args" + args list semantics, cancel_unavail on bid
instances, per-GPU vs per-machine bid, credit update granularity, status values of an outbid instance,
v1 list row field names for cur_state/next_state/is_bid/min_bid, storage_cost/720 as hourly storage.
Safety: mutating calls only with --yes and only for ids in state.json; key never printed or stored.
Profiles (rent --profile FILE.json): image, args ("{KEY}" in an arg becomes a random instance secret), env, port,
disk, label, health_path, health_bearer. The profile and the secret live in state.json (mode 0600); the secret is
printed only by `urls --show-secret` and is sent only to instance addresses, never to console.vast.ai.
Unverified: how the image's entrypoint combines with runtype "args" for the vLLM image.
Running-state tools: rent --premium F (bid = ceil3(min_bid * F), F 1.0..1.5), rent --add N (more instances into an
active state, same profile/secret/disk/premium, same caps), drop --id X (destroy one instance, state stays active),
urls prints "ID URL" lines and with --healthy only those that pass the profile's health check right now.
--min-down MBPS (plan, rent; kept for --add) skips offers whose inet_down is missing or below it.
--max-traffic-cost X ($/GB, both directions) and --exclude-geo "CN,XX" filter offers the same way (stored, reused by
--add). create calls are paced (1.5 s apart, one retry after 6 s on HTTP 429). urls --json is the machine-readable view.
"""
import argparse
import datetime
import ipaddress
import fcntl
import json
import math
import os
import secrets
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

BASE = "https://console.vast.ai"
LABEL = "undead-canary"
OWN_LABELS = ("undead-canary", "undead-demo")
LOCK_WAIT, POLL_LOCK_WAIT = 90.0, 5.0
LOCKING = ("rent", "poll", "bid", "drop", "stop")
IMAGE = "python:3.12-alpine"
PORT = 8080
CMD_ARGS = ["python3", "-m", "http.server", str(PORT), "--bind", "0.0.0.0"]
MAX_COUNT, MAX_BUDGET, MIN_CREDIT, MIN_REL = 7, 50.0, 0.5, 0.98
CREATE_GAP, RETRY_429 = 1.5, 6.0  # seconds between create calls; wait before the one retry after HTTP 429
SPARE_OFFERS = 8  # offers vanish within seconds; rent walks down this many extra candidates
COLS = ["id", "machine_id", "host_id", "gpu_name", "dph_total", "min_bid", "reliability2", "geolocation",
        "storage_cost", "direct_port_count", "is_bid", "inet_down", "inet_down_cost", "inet_up_cost"]
SAMPLE_FIELDS = ("actual_status", "intended_status", "cur_state", "next_state", "status_msg",
                 "is_bid", "min_bid", "dph_total")
EXTRA_SAMPLE_FIELDS = ("inet_down_billed", "inet_up_billed", "storage_total_cost", "duration", "start_date")


class ApiError(Exception):
    def __init__(self, msg, status=None):
        Exception.__init__(self, msg)
        self.status = status


class Refuse(Exception):
    pass


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):  # never follow: the Authorization header must not travel
        return None


def real_http(method, url, key=None, body=None, timeout=30, headers=None):
    """The ONLY network function. Returns (status, parsed_json_or_None). Raises ApiError on transport errors.
    Invariant: the Vast API key goes only to console.vast.ai; `headers` (the instance secret) only to other hosts."""
    data = None if body is None else json.dumps(body).encode()
    h = {"Accept": "application/json", "User-Agent": "Mozilla/5.0"}
    if data is not None:
        h["Content-Type"] = "application/json"
    on_api = url.startswith(BASE + "/")
    if headers and on_api:
        raise ApiError("refusing to send extra headers to the API host")
    if key and on_api:
        h["Authorization"] = "Bearer " + key
    elif headers:
        h.update(headers)
    req = urllib.request.Request(url, data=data, method=method, headers=h)
    try:
        with urllib.request.build_opener(NoRedirect).open(req, timeout=timeout) as r:
            code, raw = r.status, r.read(1 << 20)
    except urllib.error.HTTPError as e:
        code, raw = e.code, e.read(1 << 20)
    except Exception as e:  # message deliberately reduced to the class name
        raise ApiError("%s %s failed: %s" % (method, urllib.parse.urlparse(url).path, type(e).__name__))
    try:
        return code, json.loads(raw.decode("utf-8", "replace"))
    except ValueError:
        return code, None


def iso(ts):
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def default_key_file():
    cd = os.environ.get("CREDENTIALS_DIRECTORY")
    return os.path.join(cd, "vast.key") if cd else "/etc/undead/vast.key"


def load_key(path):
    try:
        with open(path) as f:
            key = f.read().strip()
    except OSError:
        raise Refuse("cannot read key file %s" % path)
    if not key:
        raise Refuse("key file %s is empty" % path)
    return key


class Ctx:
    def __init__(self, d, key=None, http=real_http, now=time.time, sleep=time.sleep):
        self.dir, self.key, self.http, self.now, self.sleep = d, key, http, now, sleep
        self.secret = None

    def red(self, s):
        s = str(s)
        for v in (self.key, self.secret):
            if v:
                s = s.replace(v, "***")
        return s

    def say(self, s=""):
        print(self.red(s))

    def api(self, method, path, body=None):
        code, obj = self.http(method, BASE + path, self.key, body)
        if code >= 400:
            hint = obj.get("msg") or obj.get("error") or "" if isinstance(obj, dict) else ""
            raise ApiError(self.red("%s %s -> HTTP %s %s" % (method, path.split("?")[0], code, str(hint)[:120])), code)
        return obj if isinstance(obj, dict) else {}

    def credit(self):
        try:
            c = self.api("GET", "/api/v0/users/current/").get("credit")
        except ApiError:
            return None
        return float(c) if isinstance(c, (int, float)) and not isinstance(c, bool) else None

    def path(self, name):
        return os.path.join(self.dir, name)

    def append(self, name, rec):
        with open(self.path(name), "a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")

    def event(self, **kw):
        kw["t"] = iso(self.now())
        self.append("events.jsonl", kw)

    def load_state(self):
        try:
            with open(self.path("state.json")) as f:
                st = json.load(f)
        except FileNotFoundError:
            return None
        if isinstance(st, dict) and isinstance(st.get("secret"), str):
            self.secret = st["secret"]
        return st

    def save_state(self, st):
        os.makedirs(self.dir, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.dir, prefix=".state-", suffix=".tmp")
        try:
            os.fchmod(fd, 0o600)  # state.json may hold the instance secret
            with os.fdopen(fd, "w") as f:
                json.dump(st, f, indent=1, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path("state.json"))
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    # ---- the only three mutating calls; each insists the id is in state.json ----
    def _owned(self, st, iid):
        if st is None or iid not in [i["id"] for i in st["instances"]]:
            raise Refuse("instance %s is not in state.json; refusing to touch it" % iid)

    def destroy(self, st, iid):
        self._owned(st, iid)
        try:
            r = self.api("DELETE", "/api/v0/instances/%d/" % iid, {})
        except ApiError as e:
            if e.status == 404:
                return
            raise
        if r.get("success") is False:
            raise ApiError("destroy %s refused: %s" % (iid, str(r.get("msg"))[:100]))

    def set_bid(self, st, iid, price):
        self._owned(st, iid)
        self.api("PUT", "/api/v0/instances/bid_price/%d/" % iid, {"client_id": "me", "price": price})

    def create(self, offer_id, bid, disk, prof=None, secret=None):
        prof = prof or {}
        port = prof.get("port", PORT)
        args = [x.replace("{KEY}", secret or "") for x in prof["args"]] if "args" in prof else \
            ["python3", "-m", "http.server", str(port), "--bind", "0.0.0.0"]
        env = dict(prof.get("env") or {})
        env["-p %d:%d" % (port, port)] = "1"
        blob = {"client_id": "me", "image": prof.get("image", IMAGE), "env": env, "price": bid,
                "disk": disk, "label": prof.get("label", LABEL), "extra": None, "onstart": None, "image_login": None,
                "python_utf8": False, "lang_utf8": False, "use_jupyter_lab": False, "jupyter_dir": None,
                "force": False, "cancel_unavail": True, "template_hash_id": None, "user": None,
                "runtype": "args", "args": args}
        return self.api("PUT", "/api/v0/asks/%d/" % offer_id, blob)


# ---------------------------------------------------------------- profiles
PROFILE_KEYS = {"image", "args", "env", "port", "disk", "label", "health_path", "health_bearer"}


def load_profile(path):
    try:
        with open(path) as f:
            p = json.load(f)
    except (OSError, ValueError):
        raise Refuse("cannot read profile %s as JSON" % path)
    if not isinstance(p, dict):
        raise Refuse("profile must be a JSON object")
    bad = sorted(set(p) - PROFILE_KEYS)
    if bad:
        raise Refuse("unknown profile keys: %s" % ", ".join(bad))

    def need(key, ok, what):
        if key in p and not ok(p[key]):
            raise Refuse("profile key %s must be %s" % (key, what))
    need("image", lambda v: isinstance(v, str) and v.strip(), "a non-empty string")
    need("args", lambda v: isinstance(v, list) and all(isinstance(x, str) for x in v), "a list of strings")
    need("env", lambda v: isinstance(v, dict) and all(isinstance(k, str) and isinstance(x, str) for k, x in v.items()),
         "an object of strings")
    need("port", lambda v: isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= 65535, "an integer 1..65535")
    need("disk", lambda v: num(v) and v > 0, "a positive number")
    need("label", lambda v: isinstance(v, str) and v.strip(), "a non-empty string")
    need("health_path", lambda v: isinstance(v, str) and v.startswith("/") and not any(c.isspace() for c in v),
         "a path starting with /")
    need("health_bearer", lambda v: isinstance(v, bool), "true or false")
    return p


def new_secret():
    while True:
        s = secrets.token_urlsafe(24)
        if not s.startswith("-"):  # it goes into argv as a value: a leading dash would look like an option
            return s


# ---------------------------------------------------------------- offers and plan
def ceil3(x):
    return math.ceil(round(x * 1000, 6)) / 1000.0


def hourly_of(o, bid, disk):
    return bid + disk * (o.get("storage_cost") or 0.0) / 720.0  # storage_cost is $/GB/month


def fetch_offers(ctx, gpu, disk):
    q = {"verified": {"eq": True}, "external": {"eq": False}, "rentable": {"eq": True}, "rented": {"eq": False},
         "num_gpus": {"eq": 1}, "gpu_name": {"eq": gpu}, "order": [["dph_total", "asc"]], "type": "bid",
         "limit": 200, "allocated_storage": disk}
    return ctx.api("PUT", "/api/v0/search/asks/", {"select_cols": COLS, "q": q}).get("offers") or []


def num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def geo_code(o):
    """Two-letter country code at the end of the geolocation string ("Zhejiang, CN" gives "CN"), or None."""
    g = o.get("geolocation")
    code = g.rsplit(",", 1)[-1].strip().upper() if isinstance(g, str) else ""
    return code if len(code) == 2 and code.isalpha() else None


def select_offers(offers, count, max_bid, disk, premium=1.0, avoid_hosts=(), avoid_machines=(), min_down=0,
                  max_traffic=None, exclude_geo=()):
    ok = []
    for o in offers:
        if not (num(o.get("min_bid")) and num(o.get("reliability2")) and num(o.get("direct_port_count"))):
            continue
        if o["reliability2"] < MIN_REL or o["direct_port_count"] < 1 or ceil3(o["min_bid"] * premium) > max_bid:
            continue
        if o.get("id") is None or o.get("host_id") is None or o.get("machine_id") is None:
            continue
        if o["host_id"] in avoid_hosts or o["machine_id"] in avoid_machines:
            continue
        if min_down and not (num(o.get("inet_down")) and o["inet_down"] >= min_down):
            continue  # download speed missing or too low: a big model would take too long to arrive
        if max_traffic is not None and not (num(o.get("inet_down_cost")) and num(o.get("inet_up_cost"))
                                            and o["inet_down_cost"] <= max_traffic and o["inet_up_cost"] <= max_traffic):
            continue  # traffic is billed per GB on some hosts; a missing price counts as unknown, so skipped
        if exclude_geo and (geo_code(o) is None or geo_code(o) in exclude_geo):
            continue
        ok.append(o)
    ok.sort(key=lambda o: (hourly_of(o, ceil3(o["min_bid"] * premium), disk), o["id"]))
    out, hosts, machines = [], set(), set()
    for o in ok:
        if o["host_id"] in hosts or o["machine_id"] in machines:
            continue
        hosts.add(o["host_id"])
        machines.add(o["machine_id"])
        out.append(dict(o, bid=ceil3(o["min_bid"] * premium)))
        if len(out) == count:
            break
    return out


def make_plan(ctx, a, avoid_hosts=(), avoid_machines=()):
    pool = select_offers(fetch_offers(ctx, a.gpu, a.disk), a.count + SPARE_OFFERS, a.max_bid, a.disk,
                         getattr(a, "premium", None) or 1.0, avoid_hosts, avoid_machines,
                         getattr(a, "min_down", None) or 0, getattr(a, "max_traffic_cost", None),
                         getattr(a, "exclude_geo", None) or ())
    chosen = pool[:a.count]
    hourly = sum(hourly_of(o, o["bid"], a.disk) for o in chosen)
    return {"chosen": chosen, "pool": pool, "hourly": hourly, "projected": hourly * a.hours,
            "credit": ctx.credit()}


def show_plan(ctx, a, p):
    for o in p["chosen"]:
        ctx.say("offer %s machine %s host %s %s bid %.3f storage %.4f/h rel %.4f ports %s %s down %s Mbps traffic down %s up %s $/GB" % (
            o["id"], o["machine_id"], o["host_id"], o["gpu_name"], o["bid"],
            a.disk * (o.get("storage_cost") or 0) / 720.0, o["reliability2"], o["direct_port_count"],
            o.get("geolocation"), "n/a" if not num(o.get("inet_down")) else "%.0f" % o["inet_down"],
            "n/a" if not num(o.get("inet_down_cost")) else "%.4f" % o["inet_down_cost"],
            "n/a" if not num(o.get("inet_up_cost")) else "%.4f" % o["inet_up_cost"]) + (" (min_bid %.3f x premium %.2f)" % (o["min_bid"], a.premium)
                                     if (getattr(a, "premium", None) or 1.0) != 1.0 else ""))
    ctx.say("selected %d of %d wanted" % (len(p["chosen"]), a.count))
    ctx.say("hourly total %.4f USD, projected for %sh: %.2f USD, account credit: %s" % (
        p["hourly"], a.hours, p["projected"], "unknown" if p["credit"] is None else "%.2f" % p["credit"]))


# ---------------------------------------------------------------- commands
DEFAULTS = {"gpu": "RTX 4090", "count": 3, "hours": 48.0, "disk": 16.0, "max_bid": 0.40, "premium": 1.0, "min_down": 0.0,
            "max_traffic_cost": None, "exclude_geo": []}


def norm_geo(v):
    """A comma list "CN,xx" (or a stored list) as a sorted list of upper-case two-letter codes."""
    items = v.split(",") if isinstance(v, str) else list(v or [])
    out = sorted({str(x).strip().upper() for x in items if str(x).strip()})
    if any(len(x) != 2 or not x.isalpha() for x in out):
        raise Refuse("exclude-geo takes two-letter country codes, comma separated")
    return out


def fill_defaults(a, st=None, keep=()):
    """Command-line values stay; what was not given comes from the stored state (only for names in `keep`) or DEFAULTS."""
    for k, d in DEFAULTS.items():
        if getattr(a, k, None) is None:
            stored = (st or {}).get(k) if k in keep else None
            setattr(a, k, stored if stored is not None else d)
    if not num(a.premium) or not 1.0 <= a.premium <= 1.5:
        raise Refuse("premium must be between 1.0 and 1.5")
    if not num(a.min_down) or a.min_down < 0:
        raise Refuse("min-down must be 0 or more (Mbps)")
    a.exclude_geo = norm_geo(a.exclude_geo)
    if a.max_traffic_cost is not None and (not num(a.max_traffic_cost) or a.max_traffic_cost < 0):
        raise Refuse("max-traffic-cost must be 0 or more ($ per GB)")


def cmd_plan(ctx, a):
    fill_defaults(a)
    show_plan(ctx, a, make_plan(ctx, a))
    return 0


def create_call(ctx, o, disk, prof, secret):
    """One create call; after HTTP 429 wait and retry the same offer once (a second 429 reaches the caller)."""
    try:
        return ctx.create(o["id"], o["bid"], disk, prof, secret)
    except ApiError as e:
        if e.status != 429:
            raise
    ctx.sleep(RETRY_429)
    return ctx.create(o["id"], o["bid"], disk, prof, secret)


def create_loop(ctx, st, pool, want, disk, prof, secret, premium=1.0):
    """Walk down the candidate pool until `want` instances were created; returns how many were."""
    made, calls = 0, 0
    for o in pool:
        if made >= want:
            break
        if calls:
            ctx.sleep(CREATE_GAP)  # the marketplace answers HTTP 429 to rapid create calls
        calls += 1
        try:
            r = create_call(ctx, o, disk, prof, secret)
            iid = r.get("new_contract")
            if r.get("success") is False or not isinstance(iid, int):
                raise ApiError("create refused: %s" % str(r.get("msg") or r)[:100], 400)
        except ApiError as e:
            st["rent_errors"].append({"offer": o["id"], "error": ctx.red(e)})
            ctx.save_state(st)
            if e.status is None or e.status >= 500:  # outcome unknown: an instance may exist
                st["ambiguous"].append({"offer": o["id"], "note": "check console for label " + (prof or {}).get("label", LABEL)})
                ctx.save_state(st)
                break
            continue  # a definite refusal, usually "offer no longer available": try the next one
        st["instances"].append({"id": iid, "offer_id": o["id"], "machine_id": o["machine_id"],
                                "host_id": o["host_id"], "bid": o["bid"], "created": ctx.now(),
                                "hourly": hourly_of(o, o["bid"], disk), "min_bid": o["min_bid"], "premium": premium,
                                "geolocation": o.get("geolocation"), "inet_down": o.get("inet_down"),
                                "inet_down_cost": o.get("inet_down_cost"), "inet_up_cost": o.get("inet_up_cost"),
                                "destroyed": None, "last": None})
        made += 1
        ctx.save_state(st)
        ctx.event(event="created", id=iid, offer=o["id"], bid=o["bid"])
    return made


def cmd_rent(ctx, a):
    if a.add is not None:
        return cmd_add(ctx, a)
    if a.budget is None:
        raise Refuse("--budget is required")
    fill_defaults(a)
    if not 1 <= a.count <= MAX_COUNT:
        raise Refuse("count must be 1..%d" % MAX_COUNT)
    if not 0 < a.budget <= MAX_BUDGET:
        raise Refuse("budget must be >0 and <= %s" % MAX_BUDGET)
    if a.hours <= 0:
        raise Refuse("hours must be > 0")
    prof = load_profile(a.profile) if a.profile else None
    if prof and "disk" in prof:
        a.disk = float(prof["disk"])
    st = ctx.load_state()
    if st and st.get("phase") != "finished":
        raise Refuse("state.json has unfinished instances; run stop first")
    ctx.secret = None
    p = make_plan(ctx, a)
    show_plan(ctx, a, p)
    if len(p["chosen"]) < a.count:
        raise Refuse("only %d suitable offers, wanted %d" % (len(p["chosen"]), a.count))
    if any(o["bid"] > a.max_bid for o in p["chosen"]):
        raise Refuse("a bid exceeds --max-bid")
    if p["credit"] is None:
        raise Refuse("account credit unknown; refusing to spend")
    if p["credit"] < 2 * p["hourly"]:
        raise Refuse("credit %.2f is less than two hours of cost (%.2f)" % (p["credit"], 2 * p["hourly"]))
    if p["projected"] > a.budget:
        ctx.say("note: projected cost exceeds budget, the budget cap will end the run after about %.1f h"
                % (a.budget / p["hourly"]))
    if not a.yes:
        ctx.say("dry run: would create %d interruptible instances (label %s, image %s). Add --yes." % (
            len(p["chosen"]), (prof or {}).get("label", LABEL), (prof or {}).get("image", IMAGE)))
        return 0
    secret = new_secret() if prof else None
    ctx.secret = secret
    now = ctx.now()
    st = {"version": 1, "phase": "active", "started": now, "hours": a.hours, "budget": a.budget,
          "max_bid": a.max_bid, "credit_start": p["credit"], "last_credit": p["credit"], "poll_count": 0,
          "disk": a.disk, "gpu": a.gpu, "premium": a.premium, "min_down": a.min_down,
          "max_traffic_cost": a.max_traffic_cost, "exclude_geo": a.exclude_geo, "instances": [], "ambiguous": [], "rent_errors": []}
    if prof:
        st["profile"], st["secret"] = prof, secret
    ctx.save_state(st)  # written before the first create, so a crash leaves a trace
    made = create_loop(ctx, st, p["pool"], a.count, a.disk, prof, secret, a.premium)
    if made < a.count or st["ambiguous"]:
        if not st["instances"] and not st["ambiguous"]:
            st["phase"], st["finish_reason"] = "finished", "rent_failed"
            ctx.save_state(st)
        ctx.say("RENT INCOMPLETE: created %d of %d. %s Run: stop --yes" % (
            len(st["instances"]), a.count, "AMBIGUOUS create, check the console! " if st["ambiguous"] else ""))
        return 3
    ctx.say("rented %d instances: %s (offers skipped as gone: %d)" % (
        len(st["instances"]), [i["id"] for i in st["instances"]], len(st["rent_errors"])))
    return 0


def cmd_add(ctx, a):
    """rent --add N: more instances for an active run. Profile, secret, disk, premium, time cap and budget are the run's."""
    st = ctx.load_state()
    if not st or st.get("phase") != "active":
        raise Refuse("--add needs an active state (rent first)")
    fixed = [n for n in ("budget", "hours", "disk", "count", "profile") if getattr(a, n, None) is not None]
    if fixed:
        raise Refuse("with --add these come from the run and cannot be given: " + ", ".join("--" + n for n in fixed))
    n = a.add
    live = [i for i in st["instances"] if not i["destroyed"]]
    if n < 1:
        raise Refuse("--add must be at least 1")
    if n + len(live) > MAX_COUNT:
        raise Refuse("%d new + %d live instances exceed the maximum of %d" % (n, len(live), MAX_COUNT))
    now = ctx.now()
    if now - st["started"] >= st["hours"] * 3600:
        raise Refuse("the time cap of the run is reached; no new instances")
    prof, secret = st.get("profile"), st.get("secret")
    a.count, a.disk = n, st["disk"]
    a.hours = max(0.0, st["hours"] - (now - st["started"]) / 3600.0)
    fill_defaults(a, st, keep=("gpu", "max_bid", "premium", "min_down", "max_traffic_cost", "exclude_geo"))
    p = make_plan(ctx, a, {i["host_id"] for i in live}, {i["machine_id"] for i in live})
    show_plan(ctx, a, p)
    if len(p["chosen"]) < n:
        raise Refuse("only %d suitable offers, wanted %d" % (len(p["chosen"]), n))
    if any(o["bid"] > a.max_bid for o in p["chosen"]):
        raise Refuse("a bid exceeds --max-bid")
    if p["credit"] is None:
        raise Refuse("account credit unknown; refusing to spend")
    need = 2 * (sum(i.get("hourly", i["bid"]) for i in live) + p["hourly"])
    if p["credit"] < need:
        raise Refuse("credit %.2f is less than two hours of cost of the live and new instances (%.2f)" % (p["credit"], need))
    if est_spend(st, p["credit"], now) >= st["budget"]:
        raise Refuse("the budget of the run is used up; no new instances")
    if st.get("ambiguous"):
        ctx.say("WARNING: an earlier create had an unknown outcome; check the console for label %s" %
                (prof or {}).get("label", LABEL))
    if not a.yes:
        ctx.say("dry run: would add %d interruptible instances to the run. Add --yes." % len(p["chosen"]))
        return 0
    before = len(st["ambiguous"])
    first_new = len(st["instances"])
    made = create_loop(ctx, st, p["pool"], n, a.disk, prof, secret, a.premium)
    if made < n or len(st["ambiguous"]) > before:
        ctx.say("ADD INCOMPLETE: created %d of %d. %sRun: urls / drop / stop" % (
            made, n, "AMBIGUOUS create, check the console! " if len(st["ambiguous"]) > before else ""))
        return 3
    ctx.say("added %d instances: %s" % (made, [i["id"] for i in st["instances"][first_new:]]))
    return 0


def list_instances(ctx):
    rows, params = {}, {"select_filters": {}, "order_by": [{"col": "id", "dir": "asc"}], "limit": 25}
    for _ in range(40):
        q = "&".join("%s=%s" % (k, urllib.parse.quote_plus(v if isinstance(v, str) else json.dumps(v)))
                     for k, v in params.items())
        r = ctx.api("GET", "/api/v1/instances/?" + q)
        for row in r.get("instances") or []:
            rows[row.get("id")] = row
        if not r.get("next_token"):
            return rows
        params["after_token"] = r["next_token"]
    raise ApiError("instance list has too many pages")


def addr(row, port=PORT):
    m = (row.get("ports") or {}).get("%d/tcp" % port) or []
    try:
        hp = int(m[0]["HostPort"])
    except (IndexError, KeyError, TypeError, ValueError):
        return None
    ip = row.get("public_ipaddr")
    if not isinstance(ip, str):
        return None
    try:  # a literal IPv4 only: the instance secret must not follow a odd host name from the listing
        ip = str(ipaddress.IPv4Address(ip.strip()))
    except ValueError:
        return None
    return ip, hp


def heartbeat(ctx, row, st=None):
    prof = (st or {}).get("profile") or {}
    a = addr(row, prof.get("port", PORT))
    if a is None:
        return "bad", "noaddr"
    kw = {}
    if prof.get("health_bearer") and (st or {}).get("secret"):
        kw["headers"] = {"Authorization": "Bearer " + st["secret"]}
    try:
        code, _ = ctx.http("GET", "http://%s:%d%s" % (a + (prof.get("health_path", "/"),)), None, None, 5, **kw)
    except Exception as e:
        return "bad", "exc:" + type(e).__name__
    return ("ok", "200") if code == 200 else ("bad", "http:%s" % code)


def est_spend(st, credit, now):
    """The larger of two estimates, so a top-up during the run cannot hide spending."""
    by_time = sum(i["bid"] * ((i["destroyed"] or now) - i["created"]) / 3600.0 for i in st["instances"])
    if credit is not None and st.get("credit_start") is not None:
        return max(by_time, st["credit_start"] - credit)
    return by_time


def destroy_all(ctx, st, reason):
    for i in st["instances"]:
        if i["destroyed"]:
            continue
        try:
            ctx.destroy(st, i["id"])
        except (ApiError, Refuse) as e:
            ctx.event(event="destroy_failed", id=i["id"], error=str(e))
            continue
        i["destroyed"] = ctx.now()
        ctx.event(event="destroyed", id=i["id"], reason=reason)
        ctx.save_state(st)
    if all(i["destroyed"] for i in st["instances"]):
        st["phase"], st["finished_at"], st["finish_reason"] = "finished", ctx.now(), reason
    ctx.save_state(st)
    return st["phase"] == "finished"


def cmd_poll(ctx, a):
    st = ctx.load_state()
    if not st or st.get("phase") == "finished":
        ctx.say("nothing to poll")
        return 0
    st["poll_count"] = st.get("poll_count", 0) + 1
    now = ctx.now()
    rc = 0
    try:
        rows = list_instances(ctx)
    except ApiError as e:
        ctx.event(event="poll_failed", error=str(e))
        ctx.say("poll failed: %s" % e)
        rows, rc = None, 1  # no samples this pass, but the caps below still apply
    credit = ctx.credit()
    if credit is None:
        ctx.event(event="credit_unavailable")
    else:
        st["last_credit"] = credit
    day = iso(now)[:10]
    for i in (st["instances"] if rows is not None else []):
        if i["destroyed"]:
            continue
        row = rows.get(i["id"])
        status = (row.get("actual_status") or "unknown") if row else "absent"
        hb, detail = heartbeat(ctx, row, st) if row and status == "running" else ("none", "")
        s = {"type": "sample", "t": iso(now), "ts": now, "id": i["id"], "status": status, "hb": hb, "hb_detail": detail}
        for f in SAMPLE_FIELDS:
            s[f] = row.get(f) if row else None
        for f in EXTRA_SAMPLE_FIELDS:
            if row and f in row:
                s[f] = row[f]
        ctx.append("samples-%s.jsonl" % day, s)
        cur = {"status": status, "hb": hb}
        if cur != i["last"]:
            ctx.event(event="change", id=i["id"], old=i["last"], new=cur, detail=detail)
            i["last"] = cur
    if st["poll_count"] % 10 == 0:
        ctx.append("samples-%s.jsonl" % day, {"type": "credit", "t": iso(now), "ts": now, "credit": credit})
    reason = None
    if now - st["started"] >= st["hours"] * 3600:
        reason = "time_cap"
    elif est_spend(st, credit, now) >= st["budget"]:
        reason = "budget_cap"
    elif credit is not None and credit < MIN_CREDIT:
        reason = "low_credit"
    ctx.save_state(st)
    if reason:
        ctx.event(event="cap", reason=reason)
        destroy_all(ctx, st, reason)
    return rc


def cmd_orphans(ctx, a):
    """Read-only: account instances carrying one of our labels whose id is not in state.json. Destroys nothing."""
    st = ctx.load_state() or {}
    labels = set(OWN_LABELS)
    pl = (st.get("profile") or {}).get("label")
    if isinstance(pl, str) and pl:
        labels.add(pl)
    known = {i.get("id") for i in (st.get("instances") or []) if isinstance(i, dict)}
    rows = list_instances(ctx)
    orph = sorted((iid, r) for iid, r in rows.items() if r.get("label") in labels and iid not in known)
    ctx.say("orphans (our label, not in state.json): %d" % len(orph))
    for iid, r in orph:
        ctx.say("  %s label %s status %s/%s is_bid %s dph %s gpu %s" % (
            iid, r.get("label"), r.get("actual_status"), r.get("intended_status"), r.get("is_bid"),
            r.get("dph_total"), r.get("gpu_name")))
    return 0


def cmd_list(ctx, a):
    """Read-only: every instance in the account, safe fields only."""
    rows = list_instances(ctx)
    ctx.say("instances in the account: %d" % len(rows))
    for iid, r in sorted(rows.items()):
        ctx.say("  %s label %s status %s/%s is_bid %s dph %s gpu %s" % (
            iid, r.get("label"), r.get("actual_status"), r.get("intended_status"), r.get("is_bid"),
            r.get("dph_total"), r.get("gpu_name")))
    return 0


def urls_json(ctx, st, prof, rows):
    """One JSON object: every not-destroyed instance of the state; the running ones are probed. No secret."""
    out = []
    for i in (st["instances"] if st else []):
        if i["destroyed"]:
            continue
        row = rows.get(i["id"])
        status = (row.get("actual_status") or "unknown") if row else "absent"
        ad = addr(row, prof.get("port", PORT)) if row else None
        healthy = bool(ad and status == "running" and heartbeat(ctx, row, st)[0] == "ok")
        out.append({"id": i["id"], "status": status, "intended": row.get("intended_status") if row else None,
                    "url": "http://%s:%d" % ad if ad else None, "healthy": healthy})
    ctx.say(json.dumps({"t": iso(ctx.now()), "instances": out}))
    return 0


def cmd_urls(ctx, a):
    """Read-only: "ID http://IP:HOSTPORT" of every running instance of this state (with --healthy only those that
    pass the health check now), then the secret (hidden unless asked)."""
    st = ctx.load_state()
    prof = ((st or {}).get("profile") or {})
    if a.json and a.show_secret:
        raise Refuse("--json never carries the secret")
    rows = list_instances(ctx) if st else {}
    if a.json:
        return urls_json(ctx, st, prof, rows)
    found = []
    for i in (st["instances"] if st else []):
        row = rows.get(i["id"])
        ad = addr(row, prof.get("port", PORT)) if row and not i["destroyed"] else None
        if ad and row.get("actual_status") == "running":
            found.append((i["id"], "http://%s:%d" % ad, row))
    if not found:
        ctx.say("no running instances")
        return 0
    if a.healthy:  # probe now with the profile's health check
        found = [f for f in found if heartbeat(ctx, f[2], st)[0] == "ok"]
        if not found:
            ctx.say("no healthy instances")
            return 0
    for iid, u, _ in found:
        ctx.say("%d %s" % (iid, u))
    if not st.get("secret"):
        ctx.say("secret: (none)")
    elif a.show_secret:
        print("secret: " + st["secret"])  # the one place the secret is printed; bypasses the redaction on purpose
    else:
        ctx.say("secret: (hidden)")
    return 0


def cmd_drop(ctx, a):
    """Destroy ONE instance of this state. The run stays active even when it was the last one."""
    st = ctx.load_state()
    ctx._owned(st, a.id)
    inst = [i for i in st["instances"] if i["id"] == a.id][0]
    if inst["destroyed"]:
        raise Refuse("instance %s is already destroyed" % a.id)
    if not a.yes:
        ctx.say("dry run: would destroy instance %s. Add --yes." % a.id)
        return 0
    ctx.destroy(st, a.id)
    inst["destroyed"] = ctx.now()
    ctx.save_state(st)
    ctx.event(event="dropped", id=a.id)
    ctx.say("instance %s destroyed; the run stays %s" % (a.id, st["phase"]))
    return 0


def cmd_bid(ctx, a):
    st = ctx.load_state()
    ctx._owned(st, a.id)
    if not 0 < a.price <= a.max_bid:
        raise Refuse("price must be >0 and <= max-bid %s" % a.max_bid)
    if not a.yes:
        ctx.say("dry run: would set bid of instance %s to %.3f. Add --yes." % (a.id, a.price))
        return 0
    ctx.set_bid(st, a.id, a.price)
    for i in st["instances"]:
        if i["id"] == a.id:
            i["bid"] = a.price
    ctx.event(event="bid_changed", id=a.id, price=a.price)
    ctx.save_state(st)
    ctx.say("bid of %s is now %.3f" % (a.id, a.price))
    return 0


def cmd_stop(ctx, a):
    st = ctx.load_state()
    if not st or st.get("phase") == "finished":
        ctx.say("nothing to stop")
        return 0
    ids = [i["id"] for i in st["instances"] if not i["destroyed"]]
    if st.get("ambiguous"):
        ctx.say("WARNING: a create call had an unknown outcome; check the console for label " + LABEL)
    if not a.yes:
        ctx.say("dry run: would destroy %s. Add --yes." % ids)
        return 0
    done = destroy_all(ctx, st, "manual_stop")
    ctx.say("finished" if done else "some instances could not be destroyed; rerun stop")
    return 0 if done else 1


def read_samples(ctx):
    out = []
    names = sorted(n for n in os.listdir(ctx.dir) if n.startswith("samples-") and n.endswith(".jsonl")) \
        if os.path.isdir(ctx.dir) else []
    for n in names:
        with open(ctx.path(n)) as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass  # a torn last line must not break the report
    return out


def cmd_status(ctx, a):
    st = ctx.load_state()
    if not st:
        ctx.say("no state")
        return 0
    ctx.say("phase %s started %s hours %s budget %s credit_start %s last_credit %s polls %s %s" % (
        st["phase"], iso(st["started"]), st["hours"], st["budget"], st["credit_start"], st.get("last_credit"),
        st["poll_count"], st.get("finish_reason", "")))
    last = {}
    for s in read_samples(ctx):
        if s.get("type") == "sample":
            last[s["id"]] = s
    for i in st["instances"]:
        s = last.get(i["id"], {})
        ctx.say("  %s bid %.3f destroyed %s last %s status %s hb %s dph %s" % (
            i["id"], i["bid"], bool(i["destroyed"]), s.get("t"), s.get("status"), s.get("hb"), s.get("dph_total")))
    return 0


def report_for(st, samples):
    res = {}
    for i in st["instances"]:
        ss = sorted((s for s in samples if s.get("type") == "sample" and s["id"] == i["id"]), key=lambda s: s["ts"])
        up = [s["status"] == "running" and s["hb"] == "ok" for s in ss]
        r = {"samples": len(ss), "first_good_s": None, "good_share": None, "stops": 0, "outages": [], "hours": 0.0}
        seen, out_start = False, None
        good_t = 0.0
        for k, s in enumerate(ss):
            if k + 1 < len(ss) and up[k]:
                good_t += ss[k + 1]["ts"] - s["ts"]
            if up[k]:
                if r["first_good_s"] is None:
                    r["first_good_s"] = s["ts"] - i["created"]
                if out_start is not None:
                    r["outages"].append({"start": iso(out_start[0]), "seconds": s["ts"] - out_start[0], "cause": out_start[1]})
                    out_start = None
                seen = True
            elif seen and out_start is None:
                out_start = (s["ts"], "%s/%s" % (s["status"], s["hb"]))
                r["stops"] += 1
        if out_start is not None:
            r["outages"].append({"start": iso(out_start[0]), "seconds": ss[-1]["ts"] - out_start[0],
                                 "cause": out_start[1], "ongoing": True})
        if len(ss) > 1:
            span = ss[-1]["ts"] - ss[0]["ts"]
            r["hours"] = span / 3600.0
            r["good_share"] = good_t / span if span else None
        down = [s["dph_total"] for s in ss if s["status"] != "running" and num(s.get("dph_total"))]
        r["avg_dph_when_not_running"] = sum(down) / len(down) if down else None
        res[i["id"]] = r
    return res


def cmd_report(ctx, a):
    st = ctx.load_state()
    if not st:
        ctx.say("no state")
        return 0
    rep = report_for(st, read_samples(ctx))
    for iid, r in rep.items():
        ctx.say("instance %s: samples %d, rent->first good heartbeat %s s, good share %s, stops %d, observed %.2f h, "
                "avg $/h while not running %s" % (
                    iid, r["samples"], r["first_good_s"], "n/a" if r["good_share"] is None else "%.3f" % r["good_share"],
                    r["stops"], r["hours"], r["avg_dph_when_not_running"]))
        for o in r["outages"]:
            ctx.say("    outage from %s for %d s (%s)%s" % (o["start"], o["seconds"], o["cause"], " ONGOING" if o.get("ongoing") else ""))
    hrs = sum(r["hours"] for r in rep.values())
    stops = sum(r["stops"] for r in rep.values())
    used = None if st.get("last_credit") is None else st["credit_start"] - st["last_credit"]
    ctx.say("overall: %d stops in %.2f instance-hours = %s per instance-hour; credit used %s" % (
        stops, hrs, "n/a" if not hrs else "%.4f" % (stops / hrs), None if used is None else "%.4f" % used))
    return 0


# ---------------------------------------------------------------- cli
def build_parser():
    p = argparse.ArgumentParser(prog="canary.py")
    p.add_argument("--dir", default="/var/lib/undead-canary")
    p.add_argument("--key-file", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "rent"):
        s = sub.add_parser(name)
        s.add_argument("--gpu", default=None)
        s.add_argument("--count", type=int, default=None)
        s.add_argument("--hours", type=float, default=None)
        s.add_argument("--disk", type=float, default=None)
        s.add_argument("--max-bid", dest="max_bid", type=float, default=None)
        s.add_argument("--premium", type=float, default=None, help="bid = min_bid x premium, 1.0 to 1.5")
        s.add_argument("--min-down", dest="min_down", type=float, default=None,
                       help="skip offers with inet_down below this many Mbps (0 = no filter)")
        s.add_argument("--max-traffic-cost", dest="max_traffic_cost", type=float, default=None,
                       help="skip offers whose inet_down_cost or inet_up_cost ($/GB) is above this or missing")
        s.add_argument("--exclude-geo", dest="exclude_geo", default=None,
                       help='skip offers in these countries, e.g. "CN,XX" (two-letter codes)')
        if name == "rent":
            s.add_argument("--budget", type=float, default=None)
            s.add_argument("--yes", action="store_true")
            s.add_argument("--profile", default=None, help="JSON profile: image, args, env, port, ...")
            s.add_argument("--add", type=int, default=None, help="add N instances to the active run")
    s = sub.add_parser("bid")
    s.add_argument("--id", type=int, required=True)
    s.add_argument("--price", type=float, required=True)
    s.add_argument("--max-bid", dest="max_bid", type=float, default=0.40)
    s.add_argument("--yes", action="store_true")
    s = sub.add_parser("drop")
    s.add_argument("--id", type=int, required=True)
    s.add_argument("--yes", action="store_true")
    sub.add_parser("stop").add_argument("--yes", action="store_true")
    for name in ("poll", "status", "report", "list", "orphans"):
        sub.add_parser(name)
    s = sub.add_parser("urls")
    s.add_argument("--show-secret", dest="show_secret", action="store_true")
    s.add_argument("--healthy", action="store_true")
    s.add_argument("--json", action="store_true", help="one JSON object with every live instance, no secret")
    return p


CMDS = {"plan": cmd_plan, "rent": cmd_rent, "poll": cmd_poll, "bid": cmd_bid, "stop": cmd_stop,
        "status": cmd_status, "report": cmd_report, "list": cmd_list, "urls": cmd_urls,
        "drop": cmd_drop, "orphans": cmd_orphans}


class Busy(Exception):
    pass


def take_lock(d, wait):
    """Exclusive flock on DIR/state.lock; raises Busy after `wait` seconds. Returns the open file."""
    os.makedirs(d, exist_ok=True)
    f = open(os.path.join(d, "state.lock"), "a")
    end = time.monotonic() + wait
    while True:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return f
        except OSError:
            if time.monotonic() >= end:
                f.close()
                raise Busy()
            time.sleep(0.1)


def main(argv=None, http=real_http, now=time.time, sleep=time.sleep):
    a = build_parser().parse_args(argv)
    ctx = Ctx(a.dir, None, http, now, sleep)
    rc = 1
    lock = None
    try:
        if a.cmd in LOCKING:
            try:
                lock = take_lock(a.dir, POLL_LOCK_WAIT if a.cmd == "poll" else LOCK_WAIT)
            except Busy:
                if a.cmd == "poll":
                    ctx.say("busy, skipped")
                    return 0
                ctx.say("error: another canary command holds %s (waited %d s); nothing was done" % (
                    os.path.join(a.dir, "state.lock"), LOCK_WAIT))
                return 4
        try:
            ctx.key = load_key(a.key_file or default_key_file())
        except Refuse:
            if a.cmd not in ("plan", "status", "report"):
                raise
        rc = CMDS[a.cmd](ctx, a)
    except (Refuse, ApiError) as e:
        ctx.say("error: %s" % e)
        rc = 2 if isinstance(e, Refuse) else 1
    except Exception as e:
        ctx.say("error: unexpected %s: %s" % (type(e).__name__, e))
        rc = 1
    finally:
        if lock:
            lock.close()
    if a.cmd != "plan" or os.path.isdir(a.dir):
        try:
            os.makedirs(a.dir, exist_ok=True)
            ctx.append("canary.log", {"t": iso(now()), "cmd": a.cmd, "yes": bool(getattr(a, "yes", False)), "rc": rc})
        except OSError:
            pass
    return rc


if __name__ == "__main__":
    sys.exit(main())
