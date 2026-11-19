import contextlib
import io
import json
import os
import re
import shutil
import tempfile
import unittest
from unittest import mock

import canary

KEY = "SECRETKEY-123"
T0 = 1_800_000_000.0


def offer(oid, machine, host, min_bid=0.1, rel=0.99, ports=2, storage=0.2, down=500.0, geo="PL",
          down_cost=None, up_cost=None):
    o = {"inet_down": down, "id": oid, "machine_id": machine, "host_id": host, "gpu_name": "RTX 4090", "dph_total": min_bid + 0.01,
         "min_bid": min_bid, "reliability2": rel, "geolocation": geo, "storage_cost": storage,
         "direct_port_count": ports, "is_bid": True}
    if down_cost is not None:
        o["inet_down_cost"] = down_cost
    if up_cost is not None:
        o["inet_up_cost"] = up_cost
    return o


class FakeVast:
    """Simulates the Vast API surface used by canary.py. Never touches the network, never sleeps."""

    def __init__(self):
        self.t = T0
        self.credit = 100.0
        self.offers = [offer(1, 11, 21, 0.10), offer(2, 12, 22, 0.12), offer(3, 13, 23, 0.14), offer(4, 14, 24, 0.20)]
        self.rows = {}
        self.calls = []
        self.next_id = 5000
        self.fail_list = False
        self.credit_fails = False
        self.create_results = []      # per create call: None = ok, int = HTTP status, "net" = transport error
        self.page = 2
        self.hb_down = set()
        self.error_msg = None
        self.sent = []
        self.hb_paths = []
        self.add_foreign(777, "undead-canary")
        self.add_foreign(778, "mine")

    def add_foreign(self, iid, label):
        self.rows[iid] = {"id": iid, "label": label, "actual_status": "running", "ports": {}, "public_ipaddr": "9.9.9.9"}

    def now(self):
        return self.t

    def tick(self, dt):
        self.t += dt
        for r in self.rows.values():
            if r.get("is_bid") and r["actual_status"] == "running":
                self.credit -= r["min_bid"] * dt / 3600.0

    def set_status(self, iid, status):
        self.rows[iid]["actual_status"] = status

    def mutating(self):
        return [c for c in self.calls if c[0] in ("DELETE",) or (c[0] == "PUT" and "/asks/" in c[1] and "search" not in c[1])
                or "bid_price" in c[1]]

    def http(self, method, url, key=None, body=None, timeout=30, headers=None):
        path = url.split("://", 1)[1]
        self.calls.append((method, url, body))
        self.sent.append((url, key, headers))
        if url.startswith(canary.BASE + "/"):
            assert key == KEY, "api call without key"
            assert not headers, "instance headers sent to the API host"
        else:
            assert key is None, "key sent to a non-API host"
            hostport, _, rest = path.partition("/")
            ip, port = hostport.split(":")
            for r in self.rows.values():
                if r.get("public_ipaddr") == ip and any(
                        v[0].get("HostPort") == port for v in r["ports"].values()):
                    ok = r["actual_status"] == "running" and r["id"] not in self.hb_down
                    if not ok:
                        raise OSError("timed out")
                    self.hb_paths.append("/" + rest)
                    if r.get("_need") and (headers or {}).get("Authorization") != "Bearer " + r["_need"]:
                        return 401, None
                    return 200, None
            raise OSError("no route")
        p = url[len(canary.BASE):]
        if self.error_msg:
            return 401, {"msg": self.error_msg}
        if method == "PUT" and p == "/api/v0/search/asks/":
            return 200, {"offers": list(self.offers)}
        if method == "GET" and p == "/api/v0/users/current/":
            return (500, {}) if self.credit_fails else (200, {"credit": self.credit, "balance": 1})
        if method == "GET" and p.startswith("/api/v1/instances/"):
            if self.fail_list:
                return 503, {"error": "down"}
            allrows = [self.rows[k] for k in sorted(self.rows)]
            m = re.search(r"after_token=(\d+)", p)
            start = int(m.group(1)) if m else 0
            chunk = allrows[start:start + self.page]
            nxt = str(start + self.page) if start + self.page < len(allrows) else None
            return 200, {"instances": chunk, "next_token": nxt}
        m = re.match(r"/api/v0/asks/(\d+)/$", p)
        if method == "PUT" and m:
            res = self.create_results.pop(0) if self.create_results else None
            if res == "net":
                raise canary.ApiError("PUT failed: Timeout")
            if res:
                return res, {"success": False, "msg": "no"}
            o = [x for x in self.offers if x["id"] == int(m.group(1))][0]
            self.next_id += 1
            n = self.next_id
            cport = [int(k.split()[1].split(":")[0]) for k in body["env"] if k.startswith("-p ")][0]
            need = body["args"][body["args"].index("--api-key") + 1] if "--api-key" in body["args"] else None
            self.rows[n] = {"id": n, "label": body["label"], "actual_status": "loading", "intended_status": "running",
                            "cur_state": "running", "next_state": "running", "status_msg": "", "is_bid": True,
                            "min_bid": body["price"], "dph_total": body["price"] + 0.01, "public_ipaddr": "1.2.3.%d" % (n % 250),
                            "ports": {"%d/tcp" % cport: [{"HostPort": str(40000 + n)}]}, "_need": need}
            return 200, {"success": True, "new_contract": n}
        m = re.match(r"/api/v0/instances/(\d+)/$", p)
        if method == "DELETE" and m:
            if int(m.group(1)) not in self.rows:
                return 404, {}
            del self.rows[int(m.group(1))]
            return 200, {"success": True}
        m = re.match(r"/api/v0/instances/bid_price/(\d+)/$", p)
        if method == "PUT" and m:
            self.rows[int(m.group(1))]["min_bid"] = body["price"]
            return 200, {"success": True}
        return 404, {}


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.dir = os.path.join(self.tmp, "state")
        self.keyfile = os.path.join(self.tmp, "vast.key")
        with open(self.keyfile, "w") as f:
            f.write(KEY + "\n")
        self.fake = FakeVast()
        self.sleeps = []

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_cmd(self, *args):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = canary.main(["--dir", self.dir, "--key-file", self.keyfile] + list(args), self.fake.http, self.fake.now,
                             self.sleeps.append)
        self.out = buf.getvalue()
        return rc

    def rent(self, n=3, extra=()):
        return self.run_cmd("rent", "--count", str(n), "--hours", "48", "--budget", "20", "--yes", *extra)

    def state(self):
        with open(os.path.join(self.dir, "state.json")) as f:
            return json.load(f)

    def poll(self, dt=60):
        self.fake.tick(dt)
        return self.run_cmd("poll")

    def lines(self, name):
        with open(os.path.join(self.dir, name)) as f:
            return [json.loads(x) for x in f]

    def ids(self):
        return [i["id"] for i in self.state()["instances"]]


class OfferSelection(Base):
    def test_distinct_hosts_and_machines_cheapest_first(self):
        offers = [offer(1, 11, 21, 0.10), offer(2, 11, 22, 0.11), offer(3, 12, 21, 0.12), offer(4, 13, 23, 0.13), offer(5, 14, 24, 0.14)]
        got = canary.select_offers(offers, 3, 0.4, 16)
        self.assertEqual([o["id"] for o in got], [1, 4, 5])

    def test_filters_reliability_ports_maxbid(self):
        offers = [offer(1, 1, 1, 0.1, rel=0.97), offer(2, 2, 2, 0.1, ports=0), offer(3, 3, 3, 0.45), offer(4, 4, 4, 0.3)]
        got = canary.select_offers(offers, 3, 0.4, 16)
        self.assertEqual([o["id"] for o in got], [4])

    def test_bid_rounds_up_to_three_decimals(self):
        self.assertEqual(canary.ceil3(0.1231), 0.124)
        self.assertEqual(canary.ceil3(0.123), 0.123)
        self.assertEqual(canary.ceil3(0.1), 0.1)
        got = canary.select_offers([offer(1, 1, 1, 0.1001)], 1, 0.4, 16)
        self.assertEqual(got[0]["bid"], 0.101)

    def test_rounded_bid_above_max_is_dropped(self):
        self.assertEqual(canary.select_offers([offer(1, 1, 1, 0.4004)], 1, 0.4, 16), [])


class Plan(Base):
    def test_plan_is_read_only_and_prints_totals(self):
        rc = self.run_cmd("plan", "--count", "3", "--hours", "48")
        self.assertEqual(rc, 0)
        self.assertEqual(self.fake.mutating(), [])
        self.assertIn("projected for 48.0h", self.out)
        self.assertIn("account credit: 100.00", self.out)
        hourly = 0.10 + 0.12 + 0.14 + 3 * 16 * 0.2 / 720
        self.assertIn("%.4f" % hourly, self.out)
        self.assertFalse(os.path.exists(os.path.join(self.dir, "state.json")))


class RentRefusals(Base):
    def assertRefused(self, *args):
        rc = self.run_cmd(*args)
        self.assertEqual(rc, 2, self.out)
        self.assertEqual(self.fake.mutating(), [])

    def test_count_over_maximum(self):
        self.assertEqual(canary.MAX_COUNT, 7)
        self.assertRefused("rent", "--count", "8", "--budget", "10", "--yes")

    def test_budget_over_fifty(self):
        self.assertRefused("rent", "--count", "3", "--budget", "50.01", "--yes")

    def test_unfinished_state(self):
        self.assertEqual(self.rent(1), 0)
        self.fake.calls.clear()
        self.assertRefused("rent", "--count", "1", "--budget", "10", "--yes")

    def test_low_credit(self):
        self.fake.credit = 0.5
        self.assertRefused("rent", "--count", "3", "--budget", "10", "--yes")

    def test_credit_unknown(self):
        self.fake.credit_fails = True
        self.assertRefused("rent", "--count", "3", "--budget", "10", "--yes")

    def test_bid_above_max_bid(self):
        self.assertRefused("rent", "--count", "3", "--budget", "10", "--max-bid", "0.05", "--yes")

    def test_too_few_offers(self):
        self.fake.offers = self.fake.offers[:2]
        self.assertRefused("rent", "--count", "3", "--budget", "10", "--yes")

    def test_no_yes_prints_and_changes_nothing(self):
        rc = self.run_cmd("rent", "--count", "3", "--budget", "10")
        self.assertEqual(rc, 0)
        self.assertIn("dry run", self.out)
        self.assertEqual(self.fake.mutating(), [])
        self.assertFalse(os.path.exists(os.path.join(self.dir, "state.json")))


class RentRun(Base):
    def test_rent_creates_and_records(self):
        self.assertEqual(self.rent(3), 0)
        st = self.state()
        self.assertEqual(len(st["instances"]), 3)
        creates = [c for c in self.fake.calls if c[0] == "PUT" and "/asks/" in c[1] and "search" not in c[1]]
        self.assertEqual(len(creates), 3)
        b = creates[0][2]
        self.assertEqual(b["label"], "undead-canary")
        self.assertEqual(b["image"], canary.IMAGE)
        self.assertEqual(b["env"], {"-p 8080:8080": "1"})
        self.assertEqual(b["runtype"], "args")
        self.assertEqual(b["price"], 0.10)
        self.assertEqual(st["credit_start"], 100.0)
        self.assertEqual(st["phase"], "active")

    def test_gone_offer_is_skipped_and_next_candidate_used(self):
        self.fake.create_results = [None, 400]
        self.assertEqual(self.rent(3), 0)
        st = self.state()
        self.assertEqual(len(st["instances"]), 3)
        self.assertEqual(len(st["rent_errors"]), 1)
        self.assertEqual(st["ambiguous"], [])
        self.assertEqual(len({i["host_id"] for i in st["instances"]}), 3)
        self.assertNotIn(st["rent_errors"][0]["offer"], [i["offer_id"] for i in st["instances"]])

    def test_partial_failure_records_created_and_stop_removes(self):
        self.fake.create_results = [None, 400, 400, 400]
        self.assertEqual(self.rent(3), 3)
        st = self.state()
        self.assertEqual(len(st["instances"]), 1)
        self.assertEqual(st["phase"], "active")
        self.assertEqual(st["ambiguous"], [])
        made = st["instances"][0]["id"]
        self.assertIn(made, self.fake.rows)
        self.assertEqual(self.run_cmd("stop", "--yes"), 0)
        self.assertNotIn(made, self.fake.rows)
        self.assertEqual(self.state()["phase"], "finished")

    def test_unknown_outcome_stops_without_trying_more_offers(self):
        self.fake.create_results = ["net"]
        self.assertEqual(self.rent(2), 3)
        creates = [c for c in self.fake.calls if c[0] == "PUT" and "/asks/" in c[1] and "search" not in c[1]]
        self.assertEqual(len(creates), 1)

    def test_unknown_outcome_create_is_flagged_and_blocks_rerent(self):
        self.fake.create_results = ["net"]
        self.assertEqual(self.rent(1), 3)
        st = self.state()
        self.assertEqual(len(st["ambiguous"]), 1)
        self.assertEqual(st["phase"], "active")
        self.assertIn("AMBIGUOUS", self.out)
        self.assertEqual(self.rent(1), 2)

    def test_all_candidates_refused_leaves_clean_state(self):
        self.fake.create_results = [400, 400, 400, 400]
        self.assertEqual(self.rent(1), 3)
        self.assertEqual(self.state()["phase"], "finished")
        self.assertEqual(len(self.state()["rent_errors"]), 4)
        self.assertEqual(self.rent(1), 0)

    def test_state_saved_after_every_create(self):
        seen = []
        orig = self.fake.http

        def spy(method, url, key=None, body=None, timeout=30):
            if method == "PUT" and "/asks/" in url and "search" not in url:
                seen.append(len(self.state()["instances"]))
            return orig(method, url, key, body, timeout)
        self.fake.http = spy
        self.assertEqual(self.rent(3), 0)
        self.assertEqual(seen, [0, 1, 2])
        self.assertEqual(len(self.state()["instances"]), 3)


class SafetyRules(Base):
    def test_mutations_need_yes(self):
        self.rent(2)
        n = len(self.fake.rows)
        self.fake.calls.clear()
        self.assertEqual(self.run_cmd("stop"), 0)
        self.assertEqual(self.run_cmd("bid", "--id", str(self.ids()[0]), "--price", "0.2"), 0)
        self.assertIn("dry run", self.out)
        self.assertEqual(self.fake.mutating(), [])
        self.assertEqual(len(self.fake.rows), n)

    def test_only_state_ids_destroyed(self):
        self.rent(2)
        mine = self.ids()
        self.assertEqual(self.run_cmd("stop", "--yes"), 0)
        deleted = [c[1] for c in self.fake.calls if c[0] == "DELETE"]
        self.assertEqual(sorted(int(u.rstrip("/").split("/")[-1]) for u in deleted), sorted(mine))
        self.assertIn(777, self.fake.rows)
        self.assertIn(778, self.fake.rows)

    def test_foreign_destroy_and_bid_refused(self):
        self.rent(1)
        st = self.state()
        ctx = canary.Ctx(self.dir, KEY, self.fake.http, self.fake.now)
        with self.assertRaises(canary.Refuse):
            ctx.destroy(st, 777)
        with self.assertRaises(canary.Refuse):
            ctx.set_bid(st, 777, 0.1)
        self.assertEqual(self.run_cmd("bid", "--id", "777", "--price", "0.2", "--yes"), 2)
        self.assertEqual([c for c in self.fake.calls if c[0] == "DELETE" or "bid_price" in c[1]], [])

    def test_caps_never_touch_foreign_labelled_instance(self):
        self.rent(2)
        self.fake.t += 49 * 3600
        self.assertEqual(self.run_cmd("poll"), 0)
        self.assertEqual(self.state()["phase"], "finished")
        self.assertIn(777, self.fake.rows)

    def test_bid_change_for_own_instance(self):
        self.rent(1)
        iid = self.ids()[0]
        self.assertEqual(self.run_cmd("bid", "--id", str(iid), "--price", "0.25", "--yes"), 0)
        self.assertEqual(self.fake.rows[iid]["min_bid"], 0.25)
        self.assertEqual(self.state()["instances"][0]["bid"], 0.25)
        self.assertEqual(self.run_cmd("bid", "--id", str(iid), "--price", "0.9", "--yes"), 2)

    def test_key_never_leaks(self):
        self.fake.error_msg = "invalid key " + KEY
        self.assertEqual(self.run_cmd("plan"), 1)
        self.assertNotIn(KEY, self.out)
        self.assertIn("***", self.out)
        self.fake.error_msg = None
        self.rent(2)
        self.poll()
        self.run_cmd("status")
        self.run_cmd("report")
        self.fake.error_msg = "bad " + KEY
        self.poll()
        for name in os.listdir(self.dir):
            with open(os.path.join(self.dir, name)) as f:
                self.assertNotIn(KEY, f.read(), name)

    def test_default_key_file(self):
        with mock.patch.dict(os.environ, {"CREDENTIALS_DIRECTORY": "/run/cred"}):
            self.assertEqual(canary.default_key_file(), "/run/cred/vast.key")
        env = dict(os.environ)
        env.pop("CREDENTIALS_DIRECTORY", None)
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(canary.default_key_file(), "/etc/undead/vast.key")

    def test_missing_key_refuses_mutations(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = canary.main(["--dir", self.dir, "--key-file", "/nonexistent/k", "rent", "--budget", "5", "--yes"],
                             self.fake.http, self.fake.now)
        self.assertEqual(rc, 2)
        self.assertEqual(self.fake.calls, [])

    def test_atomic_state_write(self):
        self.rent(1)
        with open(os.path.join(self.dir, "state.json")) as f:
            before = f.read()
        ctx = canary.Ctx(self.dir, KEY, self.fake.http, self.fake.now)
        with mock.patch("canary.os.replace", side_effect=OSError("boom")):
            with self.assertRaises(OSError):
                ctx.save_state({"x": 1})
        with open(os.path.join(self.dir, "state.json")) as f:
            self.assertEqual(f.read(), before)
        self.assertEqual([n for n in os.listdir(self.dir) if n.endswith(".tmp")], [])

    def test_command_log_one_line_each(self):
        self.run_cmd("status")
        self.run_cmd("plan")
        self.rent(1)
        self.run_cmd("status")
        lines = self.lines("canary.log")
        self.assertEqual([l["cmd"] for l in lines], ["status", "plan", "rent", "status"])


class PollAndCaps(Base):
    def test_pagination_lists_all(self):
        self.rent(3)
        rows = canary.list_instances(canary.Ctx(self.dir, KEY, self.fake.http, self.fake.now))
        self.assertEqual(len(rows), 5)

    def test_poll_failure_before_cap_destroys_nothing(self):
        self.rent(2)
        self.fake.fail_list = True
        self.fake.t += 3600
        self.assertEqual(self.run_cmd("poll"), 1)
        self.assertEqual([c for c in self.fake.calls if c[0] == "DELETE"], [])
        self.assertEqual(self.state()["phase"], "active")
        self.assertTrue(any(e["event"] == "poll_failed" for e in self.lines("events.jsonl")))
        self.assertEqual([s for s in self.lines("samples-%s.jsonl" % canary.iso(self.fake.t)[:10])
                          if s.get("type") == "sample"] if os.path.exists(
                              os.path.join(self.dir, "samples-%s.jsonl" % canary.iso(self.fake.t)[:10])) else [], [])

    def test_time_cap_applies_even_when_listing_fails(self):
        self.rent(2)
        self.fake.fail_list = True
        self.fake.t += 49 * 3600
        self.assertEqual(self.run_cmd("poll"), 1)
        self.assertEqual(self.state()["phase"], "finished")
        self.assertEqual(self.state()["finish_reason"], "time_cap")
        self.assertEqual(sorted(k for k in self.fake.rows), [777, 778])

    def test_top_up_does_not_hide_spend(self):
        self.rent(2)
        st = self.state()
        st["budget"] = 0.5
        with open(os.path.join(self.dir, "state.json"), "w") as f:
            json.dump(st, f)
        self.fake.t += 3 * 3600  # two cards at their bids for three hours cost more than 0.5
        self.assertGreater(canary.est_spend(self.state(), st["credit_start"] + 100.0, self.fake.t), 0.5)

    def test_list_command_is_read_only(self):
        self.rent(1)
        before = len([c for c in self.fake.calls if c[0] in ("PUT", "DELETE")])
        self.assertEqual(self.run_cmd("list"), 0)
        self.assertEqual(len([c for c in self.fake.calls if c[0] in ("PUT", "DELETE")]), before)

    def test_time_cap(self):
        self.rent(2)
        self.fake.t += 47.9 * 3600
        self.assertEqual(self.run_cmd("poll"), 0)
        self.assertEqual(self.state()["phase"], "active")
        self.fake.t += 0.2 * 3600
        self.run_cmd("poll")
        self.assertEqual(self.state()["finish_reason"], "time_cap")

    def test_budget_cap_by_credit(self):
        self.rent(1, ())
        self.run_cmd("poll")
        self.fake.credit -= 19.99
        self.run_cmd("poll")
        self.assertEqual(self.state()["phase"], "active")
        self.fake.credit -= 0.02
        self.run_cmd("poll")
        self.assertEqual(self.state()["finish_reason"], "budget_cap")
        self.assertEqual(self.ids()[0] in self.fake.rows, False)

    def test_budget_cap_fallback_bid_times_time_when_credit_unknown(self):
        self.assertEqual(self.run_cmd("rent", "--count", "3", "--hours", "48", "--budget", "1", "--yes"), 0)
        self.fake.credit_fails = True
        self.fake.t += 1.5 * 3600   # 0.36 $/h * 1.5 h = 0.54
        self.run_cmd("poll")
        self.assertEqual(self.state()["phase"], "active")
        self.fake.t += 1.5 * 3600   # 1.08 total
        self.run_cmd("poll")
        self.assertEqual(self.state()["finish_reason"], "budget_cap")

    def test_low_credit_cap(self):
        self.fake.credit = 5.0
        self.rent(1)
        self.fake.credit = 0.49
        self.run_cmd("poll")
        self.assertEqual(self.state()["finish_reason"], "low_credit")

    def test_failed_destroy_retried_next_pass(self):
        self.rent(2)
        self.fake.t += 49 * 3600
        orig = self.fake.http
        first = self.ids()[0]

        def flaky(method, url, key=None, body=None, timeout=30):
            if method == "DELETE" and url.rstrip("/").endswith(str(first)):
                return 500, {}
            return orig(method, url, key, body, timeout)
        self.fake.http = flaky
        self.run_cmd("poll")
        self.assertEqual(self.state()["phase"], "active")
        self.fake.http = orig
        self.run_cmd("poll")
        self.assertEqual(self.state()["phase"], "finished")
        self.assertEqual(sorted(self.fake.rows), [777, 778])

    def test_credit_recorded_every_tenth_pass(self):
        self.rent(1)
        for _ in range(10):
            self.poll()
        names = [n for n in os.listdir(self.dir) if n.startswith("samples-")]
        recs = []
        for n in names:
            recs += self.lines(n)
        self.assertEqual(len([r for r in recs if r["type"] == "credit"]), 1)
        self.assertEqual(len([r for r in recs if r["type"] == "sample"]), 10)
        self.assertRegex(names[0], r"^samples-\d{4}-\d{2}-\d{2}\.jsonl$")

    def test_events_on_change_only_and_unreachable_heartbeat(self):
        self.rent(1)
        iid = self.ids()[0]
        self.poll()                       # loading
        self.poll()                       # loading again: no event
        self.fake.set_status(iid, "running")
        self.poll()                       # running + hb ok
        self.fake.hb_down.add(iid)
        self.poll()                       # running + hb bad
        ev = [e for e in self.lines("events.jsonl") if e["event"] == "change"]
        self.assertEqual([(e["new"]["status"], e["new"]["hb"]) for e in ev],
                         [("loading", "none"), ("running", "ok"), ("running", "bad")])

    def test_heartbeat_not_probed_while_not_running(self):
        self.rent(1)
        self.fake.calls.clear()
        self.poll()
        self.assertEqual([c for c in self.fake.calls if not c[1].startswith(canary.BASE)], [])

    def test_poll_without_state_is_noop(self):
        self.assertEqual(self.run_cmd("poll"), 0)
        self.assertEqual(self.fake.calls, [])


class LifecycleReport(Base):
    def test_scripted_outbid_lifecycle(self):
        self.assertEqual(self.rent(1), 0)
        iid = self.ids()[0]
        self.poll()                              # +60  loading
        self.fake.set_status(iid, "running")
        self.poll()                              # +120 up
        self.poll()                              # +180 up
        self.fake.set_status(iid, "exited")      # outbid
        self.poll()                              # +240 down
        self.poll()                              # +300 down
        self.fake.set_status(iid, "running")
        self.poll()                              # +360 up
        self.poll()                              # +420 up
        ev = [(e["new"]["status"], e["new"]["hb"]) for e in self.lines("events.jsonl") if e["event"] == "change"]
        self.assertEqual(ev, [("loading", "none"), ("running", "ok"), ("exited", "none"), ("running", "ok")])
        st = self.state()
        rep = canary.report_for(st, [s for n in os.listdir(self.dir) if n.startswith("samples-") for s in self.lines(n)])[iid]
        self.assertEqual(rep["first_good_s"], 120)
        self.assertEqual(rep["stops"], 1)
        self.assertEqual(len(rep["outages"]), 1)
        self.assertEqual(rep["outages"][0]["seconds"], 120)
        self.assertEqual(rep["outages"][0]["cause"], "exited/none")
        self.assertAlmostEqual(rep["good_share"], 0.5)
        self.assertAlmostEqual(rep["hours"], 0.1)
        self.assertEqual(self.run_cmd("report"), 0)
        self.assertIn("stops 1", self.out)
        self.assertIn("1 stops in 0.10 instance-hours = 10.0000 per instance-hour", self.out)
        self.assertIn("credit used", self.out)
        self.assertEqual(self.run_cmd("status"), 0)
        self.assertIn("status running", self.out)

    def test_heartbeat_loss_while_running_counts_and_ongoing_outage(self):
        self.rent(1)
        iid = self.ids()[0]
        self.fake.set_status(iid, "running")
        self.poll()
        self.poll()
        self.fake.hb_down.add(iid)
        self.poll()
        self.poll()
        self.run_cmd("report")
        self.assertIn("ONGOING", self.out)
        self.assertIn("running/bad", self.out)

    def test_never_up_is_not_a_stop(self):
        self.rent(1)
        self.poll()
        self.poll()
        rep = canary.report_for(self.state(), [s for n in os.listdir(self.dir) if n.startswith("samples-") for s in self.lines(n)])
        self.assertEqual(list(rep.values())[0]["stops"], 0)
        self.assertIsNone(list(rep.values())[0]["first_good_s"])

PROFILE = {"image": "vllm/vllm-openai:v0.33.0",
           "args": ["--model", "M", "--port", "8000", "--api-key", "{KEY}", "--tag", "x{KEY}y"],
           "env": {"FOO": "bar", "-p 8000:8000": "2"}, "port": 8000, "disk": 40, "label": "demo",
           "health_path": "/v1/models", "health_bearer": True}


class Profiles(Base):
    def write_profile(self, prof=None, raw=None):
        path = os.path.join(self.tmp, "prof.json")
        with open(path, "w") as f:
            f.write(raw if raw is not None else json.dumps(PROFILE if prof is None else prof))
        return path

    def rent_prof(self, n=2, prof=None, extra=()):
        return self.rent(n, ["--profile", self.write_profile(prof), *extra])

    def create_bodies(self):
        return [c[2] for c in self.fake.calls if c[0] == "PUT" and re.search(r"/api/v0/asks/\d+/$", c[1])]

    def make_running(self):
        for iid in self.ids():
            self.fake.set_status(iid, "running")

    def test_create_blob_from_profile(self):
        self.assertEqual(self.rent_prof(2), 0)
        st = self.state()
        secret = st["secret"]
        self.assertGreaterEqual(len(secret), 32)
        for b in self.create_bodies():
            self.assertEqual(b["image"], "vllm/vllm-openai:v0.33.0")
            self.assertEqual(b["args"], ["--model", "M", "--port", "8000", "--api-key", secret, "--tag", "x%sy" % secret])
            self.assertEqual(b["env"], {"FOO": "bar", "-p 8000:8000": "1"})
            self.assertEqual(b["label"], "demo")
            self.assertEqual(b["disk"], 40.0)
            self.assertEqual(b["runtype"], "args")
        self.assertEqual(st["profile"], PROFILE)
        self.assertEqual(st["disk"], 40.0)

    def test_profile_disk_overrides_flag_in_offer_search(self):
        self.rent_prof(1, extra=["--disk", "16"])
        q = [c[2] for c in self.fake.calls if c[1].endswith("/search/asks/")][0]["q"]
        self.assertEqual(q["allocated_storage"], 40.0)

    def test_without_profile_nothing_changes(self):
        self.rent(1)
        b = self.create_bodies()[0]
        self.assertEqual((b["image"], b["args"], b["label"], b["env"], b["disk"]),
                         (canary.IMAGE, canary.CMD_ARGS, canary.LABEL, {"-p 8080:8080": "1"}, 16.0))
        st = self.state()
        self.assertNotIn("secret", st)
        self.assertNotIn("profile", st)
        self.assertEqual(os.stat(os.path.join(self.dir, "state.json")).st_mode & 0o777, 0o600)
        self.poll()
        self.assertEqual(self.fake.hb_paths, [])
        self.make_running()
        self.poll()
        self.assertEqual(self.fake.hb_paths, ["/"])
        self.assertTrue(all(h is None for u, k, h in self.fake.sent))

    def test_profile_without_args_serves_on_its_port(self):
        self.rent_prof(1, {"port": 9000})
        self.assertEqual(self.create_bodies()[0]["args"], ["python3", "-m", "http.server", "9000", "--bind", "0.0.0.0"])

    def test_secret_fresh_per_rent_and_never_dash_first(self):
        self.rent_prof(1)
        first = self.state()["secret"]
        self.run_cmd("stop", "--yes")
        with mock.patch("canary.secrets.token_urlsafe", side_effect=["-abc", "ok-token"]):
            self.rent_prof(1)
        self.assertEqual(self.state()["secret"], "ok-token")
        self.assertNotEqual(first, "ok-token")

    def test_state_is_0600_on_every_rewrite(self):
        self.rent_prof(1)
        path = os.path.join(self.dir, "state.json")
        self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)
        for step in (lambda: self.poll(), lambda: self.run_cmd("bid", "--id", str(self.ids()[0]), "--price", "0.2", "--yes"),
                     lambda: self.run_cmd("stop", "--yes")):
            os.chmod(path, 0o644)
            step()
            self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)

    def test_secret_only_in_state_json(self):
        self.rent_prof(2)
        secret = self.state()["secret"]
        outs = [self.out]
        self.make_running()
        self.poll()
        outs.append(self.out)
        for cmd in (["status"], ["report"], ["list"], ["urls"], ["plan"], ["stop"]):
            self.run_cmd(*cmd)
            outs.append(self.out)
        self.fake.error_msg = "echo " + secret + " " + KEY
        self.poll()
        self.run_cmd("rent", "--budget", "5", "--yes", "--profile", self.write_profile())
        outs.append(self.out)
        for o in outs:
            self.assertNotIn(secret, o)
            self.assertNotIn(KEY, o)
        for name in os.listdir(self.dir):
            if name != "state.json":
                with open(os.path.join(self.dir, name)) as f:
                    self.assertNotIn(secret, f.read(), name)

    def test_invalid_profiles_refused_before_any_call(self):
        bad = [{"image": 5}, {"args": "x"}, {"args": [1]}, {"env": {"a": 1}}, {"port": "80"}, {"port": True},
               {"port": 0}, {"disk": 0}, {"disk": "40"}, {"label": ""}, {"health_path": "v1"},
               {"health_bearer": "yes"}, {"surprise": 1}, []]
        for prof in bad:
            self.assertEqual(self.rent_prof(1, prof), 2, prof)
        path = self.write_profile(raw="{not json")
        self.assertEqual(self.rent(1, ["--profile", path]), 2)
        self.assertEqual(self.rent(1, ["--profile", "/nonexistent/p.json"]), 2)
        self.assertEqual(self.fake.calls, [])

    def test_dry_run_with_profile_changes_nothing(self):
        self.assertEqual(self.run_cmd("rent", "--budget", "5", "--profile", self.write_profile()), 0)
        self.assertIn("dry run", self.out)
        self.assertEqual(self.fake.mutating(), [])
        self.assertFalse(os.path.exists(os.path.join(self.dir, "state.json")))

    def test_heartbeat_uses_port_path_and_bearer(self):
        self.rent_prof(2)
        secret = self.state()["secret"]
        self.make_running()
        self.poll()
        self.assertEqual(self.fake.hb_paths, ["/v1/models"] * 2)
        hb = [(u, k, h) for u, k, h in self.fake.sent if not u.startswith(canary.BASE)]
        self.assertEqual(len(hb), 2)
        for u, k, h in hb:
            self.assertIsNone(k)
            self.assertEqual(h, {"Authorization": "Bearer " + secret})
        self.assertEqual([s["hb"] for n in os.listdir(self.dir) if n.startswith("samples-") for s in self.lines(n)
                          if s["status"] == "running"], ["ok", "ok"])

    def test_bearer_off_sends_no_header_and_fake_rejects(self):
        self.rent_prof(1, dict(PROFILE, health_bearer=False))
        self.make_running()
        self.poll()
        hb = [h for u, k, h in self.fake.sent if not u.startswith(canary.BASE)]
        self.assertEqual(hb, [None])
        self.assertEqual(self.lines("events.jsonl")[-1]["detail"], "http:401")

    def test_credentials_go_only_where_they_belong(self):
        self.rent_prof(2)
        secret = self.state()["secret"]
        self.make_running()
        self.poll()
        self.run_cmd("urls", "--show-secret")
        self.run_cmd("list")
        ips = {r["public_ipaddr"] for r in self.fake.rows.values()}
        for url, key, headers in self.fake.sent:
            if url.startswith(canary.BASE + "/"):
                self.assertEqual(key, KEY)
                self.assertIsNone(headers)
            else:
                self.assertIsNone(key)
                self.assertIn(url.split("://")[1].split(":")[0], ips)
                self.assertIn(headers, (None, {"Authorization": "Bearer " + secret}))

    def test_non_ipv4_address_gets_no_heartbeat(self):
        self.rent_prof(1)
        iid = self.ids()[0]
        self.fake.set_status(iid, "running")
        self.fake.rows[iid]["public_ipaddr"] = "evil.example.com"
        before = len(self.fake.sent)
        self.poll()
        self.assertEqual([u for u, k, h in self.fake.sent[before:] if not u.startswith(canary.BASE)], [])
        self.assertEqual(self.lines("events.jsonl")[-1]["detail"], "noaddr")

    # ---- urls ----
    def test_urls_lists_running_instances_and_hides_secret(self):
        self.rent_prof(2)
        secret = self.state()["secret"]
        self.run_cmd("urls")
        self.assertEqual((self.out.strip()), "no running instances")
        self.make_running()
        self.fake.page = 100
        n_before, n_mut = len(self.fake.calls), len(self.fake.mutating())
        self.assertEqual(self.run_cmd("urls"), 0)
        calls = self.fake.calls[n_before:]
        self.assertEqual([c[0] for c in calls], ["GET"])
        self.assertIn("/api/v1/instances/", calls[0][1])
        lines = self.out.splitlines()
        want = ["%d http://%s:%s" % (i, self.fake.rows[i]["public_ipaddr"], self.fake.rows[i]["ports"]["8000/tcp"][0]["HostPort"])
                for i in self.ids()]
        self.assertEqual(lines, want + ["secret: (hidden)"])
        self.assertNotIn(secret, self.out)
        self.run_cmd("urls", "--show-secret")
        self.assertEqual(self.out.splitlines(), want + ["secret: " + secret])
        self.assertEqual(len(self.fake.mutating()), n_mut)

    def test_urls_skips_not_running_foreign_and_unaddressed(self):
        self.rent_prof(3)
        a, b, c = self.ids()
        self.fake.set_status(a, "running")
        self.fake.set_status(b, "exited")
        self.fake.set_status(c, "running")
        self.fake.rows[c]["public_ipaddr"] = None
        self.fake.rows[777]["actual_status"] = "running"
        self.run_cmd("urls")
        self.assertEqual(len(self.out.splitlines()), 2)
        self.assertIn("http://%s:" % self.fake.rows[a]["public_ipaddr"], self.out)
        self.assertNotIn("9.9.9.9", self.out)

    def test_urls_without_running_or_state_exit_zero(self):
        self.assertEqual(self.run_cmd("urls"), 0)
        self.assertEqual(self.out.strip(), "no running instances")
        self.rent_prof(1)
        self.assertEqual(self.run_cmd("urls", "--show-secret"), 0)
        self.assertEqual(self.out.strip(), "no running instances")
        self.assertNotIn(self.state()["secret"], self.out)

    def test_urls_without_profile_has_no_secret(self):
        self.rent(1)
        self.make_running()
        self.run_cmd("urls", "--show-secret")
        self.assertEqual(self.out.splitlines()[-1], "secret: (none)")
        self.assertIn(":%d" % (40000 + self.ids()[0]), self.out)

    def test_urls_needs_key(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = canary.main(["--dir", self.dir, "--key-file", "/nonexistent/k", "urls"], self.fake.http, self.fake.now)
        self.assertEqual(rc, 2)
        self.assertEqual(self.fake.calls, [])

class Premium(Base):
    def bodies(self):
        return [c[2] for c in self.fake.calls if c[0] == "PUT" and re.search(r"/api/v0/asks/\d+/$", c[1])]

    def test_select_offers_applies_premium_and_max_bid(self):
        got = canary.select_offers([offer(1, 1, 1, 0.1), offer(2, 2, 2, 0.1001), offer(3, 3, 3, 0.3)], 3, 0.4, 16, 1.5)
        self.assertEqual([(o["id"], o["bid"]) for o in got], [(1, 0.15), (2, 0.151)])
        self.assertEqual([o["id"] for o in canary.select_offers([offer(3, 3, 3, 0.3)], 1, 0.4, 16, 1.0)], [3])
        self.assertEqual(canary.select_offers([offer(1, 1, 1, 0.1001)], 1, 0.4, 16, 1.2)[0]["bid"], 0.121)

    def test_rent_with_premium_bids_and_stores_it(self):
        self.assertEqual(self.rent(3, ["--premium", "1.5"]), 0)
        self.assertEqual([b["price"] for b in self.bodies()], [0.15, 0.18, 0.21])
        st = self.state()
        self.assertEqual(st["premium"], 1.5)
        self.assertEqual([i["bid"] for i in st["instances"]], [0.15, 0.18, 0.21])

    def test_default_premium_is_one(self):
        self.rent(2)
        self.assertEqual(self.state()["premium"], 1.0)
        self.assertEqual([b["price"] for b in self.bodies()], [0.10, 0.12])

    def test_offers_over_max_bid_after_premium_are_skipped(self):
        self.assertEqual(self.rent(2, ["--premium", "1.5", "--max-bid", "0.19"]), 0)
        self.assertEqual([b["price"] for b in self.bodies()], [0.15, 0.18])
        self.assertEqual(self.rent(3, ["--premium", "1.5", "--max-bid", "0.19"]), 2)  # unfinished state first
        self.run_cmd("stop", "--yes")
        self.assertEqual(self.rent(3, ["--premium", "1.5", "--max-bid", "0.19"]), 2)
        self.assertIn("only 2 suitable offers", self.out)

    def test_plan_shows_the_bid(self):
        self.assertEqual(self.run_cmd("plan", "--count", "2", "--premium", "1.2"), 0)
        self.assertIn("bid 0.120", self.out)
        self.assertIn("bid 0.144", self.out)
        self.assertIn("premium 1.20", self.out)
        self.assertEqual(self.fake.mutating(), [])
        self.run_cmd("plan", "--count", "1")
        self.assertIn("bid 0.100", self.out)
        self.assertNotIn("premium", self.out)

    def test_premium_range_enforced(self):
        for v in ("0.99", "1.51", "-1"):
            self.assertEqual(self.run_cmd("plan", "--premium", v), 2, v)
            self.assertEqual(self.run_cmd("rent", "--budget", "5", "--yes", "--premium", v), 2, v)
        self.assertEqual(self.fake.calls, [])
        self.assertEqual(self.run_cmd("plan", "--premium", "1.0"), 0)
        self.assertEqual(self.run_cmd("plan", "--premium", "1.5"), 0)

    def test_rent_still_requires_budget(self):
        self.assertEqual(self.run_cmd("rent", "--yes"), 2)
        self.assertIn("--budget", self.out)
        self.assertEqual(self.fake.calls, [])


class AddInstances(Base):
    def bodies(self):
        return [c[2] for c in self.fake.calls if c[0] == "PUT" and re.search(r"/api/v0/asks/\d+/$", c[1])]

    def start(self, n=2, prof=True, extra=()):
        path = os.path.join(self.tmp, "prof.json")
        with open(path, "w") as f:
            json.dump(PROFILE, f)
        return self.rent(n, (["--profile", path] if prof else []) + list(extra))

    def add(self, n=1, *extra, yes=True):
        return self.run_cmd("rent", "--add", str(n), *(["--yes"] if yes else []), *extra)

    def test_add_reuses_profile_secret_disk_premium_and_keeps_caps(self):
        self.start(2, extra=["--premium", "1.5"])
        before = self.state()
        n0 = len(self.bodies())
        self.fake.tick(600)
        self.assertEqual(self.add(1), 0)
        st = self.state()
        self.assertEqual(len(st["instances"]), 3)
        b = self.bodies()[n0:]
        self.assertEqual(len(b), 1)
        self.assertEqual(b[0]["price"], 0.21)  # offer 3: 0.14 x 1.5
        self.assertEqual(b[0]["disk"], 40.0)
        self.assertEqual(b[0]["image"], PROFILE["image"])
        self.assertIn("--api-key", b[0]["args"])
        self.assertEqual(b[0]["args"][b[0]["args"].index("--api-key") + 1], before["secret"])
        for k in ("secret", "profile", "started", "hours", "budget", "credit_start", "premium", "disk", "max_bid"):
            self.assertEqual(st[k], before[k], k)
        self.assertEqual(st["instances"][2]["id"], self.fake.next_id)
        self.assertNotIn(before["secret"], self.out)
        self.assertEqual(os.stat(os.path.join(self.dir, "state.json")).st_mode & 0o777, 0o600)

    def test_premium_on_command_line_overrides(self):
        self.start(1, extra=["--premium", "1.5"])
        n0 = len(self.bodies())
        self.assertEqual(self.add(1, "--premium", "1.2"), 0)
        self.assertEqual(self.bodies()[n0]["price"], 0.144)  # offer 2: 0.12 x 1.2
        self.assertEqual(self.run_cmd("rent", "--add", "1", "--premium", "1.9", "--yes"), 2)

    def test_hosts_and_machines_of_live_instances_are_excluded(self):
        self.fake.offers = [offer(1, 11, 21, 0.10), offer(2, 11, 22, 0.11), offer(3, 13, 21, 0.12), offer(4, 14, 24, 0.20)]
        self.start(1)
        n0 = len(self.bodies())
        self.assertEqual(self.add(1), 0)
        self.assertEqual(self.bodies()[n0]["price"], 0.20)  # 2 shares the machine, 3 shares the host
        self.assertEqual([i["offer_id"] for i in self.state()["instances"]], [1, 4])

    def test_destroyed_instances_free_their_host(self):
        self.fake.offers = [offer(1, 11, 21, 0.10), offer(2, 12, 22, 0.20)]
        self.start(1)
        iid = self.ids()[0]
        self.assertEqual(self.run_cmd("drop", "--id", str(iid), "--yes"), 0)
        n0 = len(self.bodies())
        self.assertEqual(self.add(1), 0)
        self.assertEqual(self.bodies()[n0]["price"], 0.10)  # offer 1 is usable again

    def test_total_live_count_limited(self):
        self.fake.offers = [offer(i, 10 + i, 20 + i, 0.10 + 0.01 * i) for i in range(1, 13)]
        self.start(7)
        n0 = len(self.bodies())
        self.assertEqual(self.add(1), 2)
        self.assertEqual(len(self.bodies()), n0)
        a = self.ids()[0]
        self.run_cmd("drop", "--id", str(a), "--yes")
        self.assertEqual(self.add(2), 2)  # 6 live + 2 > 7
        self.assertEqual(self.add(0), 2)
        self.assertEqual(self.add(1), 0)
        self.assertEqual(len([i for i in self.state()["instances"] if not i["destroyed"]]), 7)

    def test_needs_active_state(self):
        self.assertEqual(self.add(1), 2)
        self.start(1)
        self.run_cmd("stop", "--yes")
        n0 = len(self.bodies())
        self.assertEqual(self.add(1), 2)
        self.assertEqual(len(self.bodies()), n0)

    def test_dry_run_creates_nothing(self):
        self.start(1)
        n_mut = len(self.fake.mutating())
        before = self.state()
        self.assertEqual(self.add(1, yes=False), 0)
        self.assertIn("dry run", self.out)
        self.assertEqual(len(self.fake.mutating()), n_mut)
        self.assertEqual(self.state(), before)

    def test_credit_covers_live_and_new_for_two_hours(self):
        self.fake.credit = 0.6  # enough for two new instances (2 x 2 h), not for 2 live + 1 new
        self.start(2)
        n0 = len(self.bodies())
        self.assertEqual(self.add(1), 2)
        self.assertIn("two hours", self.out)
        self.assertEqual(len(self.bodies()), n0)

    def test_credit_enough_adds(self):
        self.fake.credit = 1.0
        self.start(2)
        self.assertEqual(self.add(1), 0)

    def test_credit_unknown_refuses(self):
        self.start(1)
        self.fake.credit_fails = True
        n0 = len(self.bodies())
        self.assertEqual(self.add(1), 2)
        self.assertEqual(len(self.bodies()), n0)

    def test_time_cap_and_budget_checked_before_create(self):
        self.start(1)
        n0 = len(self.bodies())
        self.fake.tick(48 * 3600 + 1)
        self.assertEqual(self.add(1), 2)
        self.assertIn("time cap", self.out)
        self.assertEqual(len(self.bodies()), n0)

    def test_budget_used_up_refuses(self):
        self.assertEqual(self.run_cmd("rent", "--count", "1", "--hours", "48", "--budget", "0.05", "--yes"), 0)
        self.fake.set_status(self.ids()[0], "running")
        self.fake.tick(1800)
        n0 = len(self.bodies())
        self.assertEqual(self.add(1), 2)
        self.assertIn("budget", self.out)
        self.assertEqual(len(self.bodies()), n0)

    def test_gone_offer_walks_down_the_pool(self):
        self.start(1)
        self.fake.create_results = [409]
        self.assertEqual(self.add(1), 0)
        st = self.state()
        self.assertEqual([i["offer_id"] for i in st["instances"]], [1, 3])  # offer 2 was refused as gone, offer 3 taken
        self.assertEqual(len(st["rent_errors"]), 1)

    def test_unknown_outcome_stops_and_is_flagged(self):
        self.start(1)
        self.fake.create_results = ["net"]
        n0 = len(self.bodies())
        self.assertEqual(self.add(1), 3)
        self.assertEqual(len(self.bodies()), n0 + 1)
        self.assertTrue(self.state()["ambiguous"])
        self.assertIn("ADD INCOMPLETE", self.out)

    def test_run_fixing_options_are_refused(self):
        self.start(1)
        for opt in (["--budget", "99"], ["--hours", "10"], ["--disk", "20"], ["--count", "2"]):
            self.assertEqual(self.add(1, *opt), 2, opt)
        self.assertEqual(self.add(1, "--profile", os.path.join(self.tmp, "prof.json")), 2)

    def test_add_without_profile_state(self):
        self.start(1, prof=False)
        n0 = len(self.bodies())
        self.assertEqual(self.add(1), 0)
        b = self.bodies()[n0]
        self.assertEqual((b["image"], b["args"], b["label"]), (canary.IMAGE, canary.CMD_ARGS, canary.LABEL))
        self.assertNotIn("secret", self.state())

    def test_add_never_touches_foreign_instances(self):
        self.start(1)
        self.add(1)
        self.assertEqual([c for c in self.fake.calls if c[0] == "DELETE"], [])
        self.assertIn(777, self.fake.rows)


class DropInstance(Base):
    def setUp(self):
        Base.setUp(self)
        self.rent(2)

    def deletes(self):
        return [c[1] for c in self.fake.calls if c[0] == "DELETE"]

    def test_drop_destroys_one_and_keeps_run_active(self):
        a, b = self.ids()
        self.assertEqual(self.run_cmd("drop", "--id", str(a), "--yes"), 0)
        self.assertEqual(self.deletes(), [canary.BASE + "/api/v0/instances/%d/" % a])
        st = self.state()
        self.assertTrue(st["instances"][0]["destroyed"])
        self.assertIsNone(st["instances"][1]["destroyed"])
        self.assertEqual(st["phase"], "active")
        self.assertEqual([e for e in self.lines("events.jsonl") if e["event"] == "dropped"][0]["id"], a)
        self.assertNotIn(a, self.fake.rows)
        self.assertIn(b, self.fake.rows)

    def test_needs_yes(self):
        a = self.ids()[0]
        self.assertEqual(self.run_cmd("drop", "--id", str(a)), 0)
        self.assertIn("dry run", self.out)
        self.assertEqual(self.deletes(), [])
        self.assertIsNone(self.state()["instances"][0]["destroyed"])

    def test_foreign_unknown_and_destroyed_refused(self):
        self.assertEqual(self.run_cmd("drop", "--id", "777", "--yes"), 2)
        self.assertEqual(self.run_cmd("drop", "--id", "1", "--yes"), 2)
        self.assertEqual(self.deletes(), [])
        a = self.ids()[0]
        self.run_cmd("drop", "--id", str(a), "--yes")
        n = len(self.deletes())
        self.assertEqual(self.run_cmd("drop", "--id", str(a), "--yes"), 2)
        self.assertIn("already destroyed", self.out)
        self.assertEqual(len(self.deletes()), n)

    def test_dropping_the_last_instance_leaves_the_run_active(self):
        for i in self.ids():
            self.assertEqual(self.run_cmd("drop", "--id", str(i), "--yes"), 0)
        self.assertEqual(self.state()["phase"], "active")
        self.assertEqual(self.run_cmd("drop", "--id", str(self.ids()[0]), "--yes"), 2)

    def test_poll_with_no_live_instance_is_harmless_then_time_cap_finishes(self):
        for i in self.ids():
            self.run_cmd("drop", "--id", str(i), "--yes")
        n_del = len(self.deletes())
        self.assertEqual(self.poll(60), 0)
        self.assertEqual(self.state()["phase"], "active")
        self.assertEqual(len(self.deletes()), n_del)
        self.assertEqual(self.state()["poll_count"], 1)
        self.assertEqual(self.poll(48 * 3600), 0)
        st = self.state()
        self.assertEqual((st["phase"], st["finish_reason"]), ("finished", "time_cap"))
        self.assertEqual(len(self.deletes()), n_del)
        self.assertIn(777, self.fake.rows)

    def test_budget_cap_with_no_live_instance_finishes(self):
        a, b = self.ids()
        self.fake.set_status(a, "running")
        self.fake.set_status(b, "running")
        self.fake.tick(60)
        self.run_cmd("drop", "--id", str(a), "--yes")
        self.run_cmd("drop", "--id", str(b), "--yes")
        st = self.state()
        st["budget"] = 0.0001
        canary.Ctx(self.dir, KEY).save_state(st)
        self.poll(60)
        self.assertEqual(self.state()["finish_reason"], "budget_cap")

    def test_add_after_all_dropped_and_stop_after(self):
        for i in self.ids():
            self.run_cmd("drop", "--id", str(i), "--yes")
        self.assertEqual(self.run_cmd("rent", "--add", "2", "--yes"), 0)
        self.assertEqual(len([i for i in self.state()["instances"] if not i["destroyed"]]), 2)
        self.assertEqual(self.run_cmd("stop", "--yes"), 0)
        self.assertEqual(self.state()["phase"], "finished")

    def test_stop_on_run_with_nothing_live_finishes(self):
        for i in self.ids():
            self.run_cmd("drop", "--id", str(i), "--yes")
        n = len(self.deletes())
        self.assertEqual(self.run_cmd("stop", "--yes"), 0)
        self.assertEqual(self.state()["phase"], "finished")
        self.assertEqual(len(self.deletes()), n)


class UrlsHealthy(Base):
    def setUp(self):
        Base.setUp(self)
        path = os.path.join(self.tmp, "prof.json")
        with open(path, "w") as f:
            json.dump(PROFILE, f)
        self.rent(3, ["--profile", path])
        for i in self.ids():
            self.fake.set_status(i, "running")

    def test_plain_urls_have_id_first_and_do_not_probe(self):
        self.run_cmd("urls")
        lines = self.out.splitlines()
        self.assertEqual([l.split()[0] for l in lines[:3]], [str(i) for i in self.ids()])
        self.assertTrue(all(l.split()[1].startswith("http://") for l in lines[:3]))
        self.assertEqual(self.fake.hb_paths, [])

    def test_healthy_lists_only_passing_instances(self):
        a, b, c = self.ids()
        self.fake.hb_down.add(b)
        self.assertEqual(self.run_cmd("urls", "--healthy"), 0)
        lines = self.out.splitlines()
        self.assertEqual([l.split()[0] for l in lines[:-1]], [str(a), str(c)])
        self.assertEqual(lines[-1], "secret: (hidden)")
        self.assertEqual(self.fake.hb_paths, ["/v1/models"] * 2)
        secret = self.state()["secret"]
        probes = [h for u, k, h in self.fake.sent if not u.startswith(canary.BASE)]
        self.assertEqual(probes, [{"Authorization": "Bearer " + secret}] * 3)

    def test_healthy_show_secret(self):
        self.run_cmd("urls", "--healthy", "--show-secret")
        self.assertEqual(self.out.splitlines()[-1], "secret: " + self.state()["secret"])
        self.assertEqual(len(self.out.splitlines()), 4)

    def test_none_healthy_exit_zero_no_secret(self):
        self.fake.hb_down.update(self.ids())
        self.assertEqual(self.run_cmd("urls", "--healthy", "--show-secret"), 0)
        self.assertEqual(self.out.strip(), "no healthy instances")
        self.assertNotIn(self.state()["secret"], self.out)

    def test_wrong_bearer_means_unhealthy(self):
        st = self.state()
        st["secret"] = "other-secret"
        canary.Ctx(self.dir, KEY).save_state(st)  # the instances were started with the old secret
        self.run_cmd("urls", "--healthy")
        self.assertEqual(self.out.strip(), "no healthy instances")

    def test_only_running_instances_are_probed(self):
        a, b, c = self.ids()
        self.fake.set_status(b, "exited")
        self.run_cmd("urls", "--healthy")
        self.assertEqual(len(self.fake.hb_paths), 2)
        self.assertEqual([l.split()[0] for l in self.out.splitlines()[:-1]], [str(a), str(c)])

    def test_healthy_works_without_profile(self):
        self.run_cmd("stop", "--yes")
        self.rent(1)
        for i in self.ids():
            self.fake.set_status(i, "running")
        self.fake.hb_paths.clear()
        self.run_cmd("urls", "--healthy")
        self.assertEqual(self.fake.hb_paths, ["/"])
        self.assertEqual(self.out.splitlines()[-1], "secret: (none)")


class MinDown(Base):
    def bodies(self):
        return [c[2] for c in self.fake.calls if c[0] == "PUT" and re.search(r"/api/v0/asks/\d+/$", c[1])]

    def setUp(self):
        Base.setUp(self)
        self.fake.offers = [offer(1, 11, 21, 0.10, down=80.0), offer(2, 12, 22, 0.12, down=None),
                            offer(3, 13, 23, 0.14, down=300.0), offer(4, 14, 24, 0.20, down=900.0),
                            offer(5, 15, 25, 0.22, down=250.0)]

    def test_select_offers_filters_missing_and_slow(self):
        got = canary.select_offers(self.fake.offers, 5, 0.4, 16, min_down=250)
        self.assertEqual([o["id"] for o in got], [3, 4, 5])  # 250 itself passes, missing and 80 do not
        self.assertEqual([o["id"] for o in canary.select_offers(self.fake.offers, 5, 0.4, 16)], [1, 2, 3, 4, 5])
        self.assertEqual([o["id"] for o in canary.select_offers(self.fake.offers, 5, 0.4, 16, min_down=0)], [1, 2, 3, 4, 5])

    def test_search_requests_the_column(self):
        self.run_cmd("plan", "--count", "1")
        body = [c[2] for c in self.fake.calls if c[1].endswith("/search/asks/")][0]
        self.assertIn("inet_down", body["select_cols"])

    def test_plan_shows_inet_down_and_filters(self):
        self.assertEqual(self.run_cmd("plan", "--count", "2", "--min-down", "200"), 0)
        self.assertIn("offer 3 ", self.out)
        self.assertIn("down 300 Mbps", self.out)
        self.assertIn("down 900 Mbps", self.out)
        self.assertNotIn("offer 1 ", self.out)
        self.assertNotIn("offer 2 ", self.out)
        self.run_cmd("plan", "--count", "2")
        self.assertIn("down 80 Mbps", self.out)
        self.assertIn("down n/a Mbps", self.out)

    def test_rent_uses_filter_and_stores_it(self):
        self.assertEqual(self.rent(2, ["--min-down", "200"]), 0)
        self.assertEqual([b["price"] for b in self.bodies()], [0.14, 0.20])
        self.assertEqual(self.state()["min_down"], 200.0)

    def test_default_is_no_filter(self):
        self.rent(2)
        self.assertEqual(self.state()["min_down"], 0.0)
        self.assertEqual([b["price"] for b in self.bodies()], [0.10, 0.12])

    def test_too_few_fast_offers_refuses_without_creating(self):
        self.assertEqual(self.rent(3, ["--min-down", "800"]), 2)
        self.assertIn("only 1 suitable offers", self.out)
        self.assertEqual(self.bodies(), [])

    def test_add_reuses_stored_min_down_and_cli_overrides(self):
        self.assertEqual(self.rent(1, ["--min-down", "200"]), 0)
        n0 = len(self.bodies())
        self.assertEqual(self.run_cmd("rent", "--add", "1", "--yes"), 0)
        self.assertEqual(self.bodies()[n0]["price"], 0.20)  # offers 4 and 5 are fast; offer 3 is the used host
        self.assertEqual(self.run_cmd("rent", "--add", "1", "--min-down", "900", "--yes"), 2)  # none left that fast
        self.assertEqual(self.run_cmd("rent", "--add", "1", "--min-down", "100", "--yes"), 0)
        self.assertEqual(len(self.bodies()), n0 + 2)

    def test_negative_min_down_refused(self):
        self.assertEqual(self.run_cmd("plan", "--min-down", "-1"), 2)
        self.assertEqual(self.fake.calls, [])


class TrafficAndGeo(Base):
    def bodies(self):
        return [c[2] for c in self.fake.calls if c[0] == "PUT" and re.search(r"/api/v0/asks/\d+/$", c[1])]

    def setUp(self):
        Base.setUp(self)
        self.fake.offers = [
            offer(1, 11, 21, 0.10, geo="Zhejiang, CN", down_cost=0.001, up_cost=0.001),
            offer(2, 12, 22, 0.11, geo="Poland, PL", down_cost=0.052, up_cost=0.001),
            offer(3, 13, 23, 0.12, geo="Berlin, DE", down_cost=0.002, up_cost=0.003),
            offer(4, 14, 24, 0.13, geo="Berlin, DE", down_cost=0.002),  # up price missing
            offer(5, 15, 25, 0.14, geo=None, down_cost=0.0, up_cost=0.0),
            offer(6, 16, 26, 0.15, geo="Texas, us", down_cost=0.0, up_cost=0.01)]

    def ids_of(self, **kw):
        return [o["id"] for o in canary.select_offers(self.fake.offers, 10, 0.4, 16, **kw)]

    def test_geo_code(self):
        self.assertEqual(canary.geo_code({"geolocation": "Zhejiang, CN"}), "CN")
        self.assertEqual(canary.geo_code({"geolocation": "Texas, us"}), "US")
        self.assertEqual(canary.geo_code({"geolocation": "PL"}), "PL")
        self.assertIsNone(canary.geo_code({"geolocation": "Nowhere"}))
        self.assertIsNone(canary.geo_code({}))

    def test_traffic_filter_skips_expensive_and_missing(self):
        self.assertEqual(self.ids_of(max_traffic=0.003), [1, 3, 5])  # 4 lacks the up price, 6 up 0.01, 2 down 0.052
        self.assertEqual(self.ids_of(max_traffic=0.0), [5])
        self.assertEqual(self.ids_of(), [1, 2, 3, 4, 5, 6])

    def test_geo_filter_compares_the_trailing_code(self):
        self.assertEqual(self.ids_of(exclude_geo=["CN"]), [2, 3, 4, 6])  # 5 has no geolocation: unknown, skipped
        self.assertEqual(self.ids_of(exclude_geo=["CN", "US", "DE"]), [2])

    def test_plan_shows_prices_and_applies_filters(self):
        self.assertEqual(self.run_cmd("plan", "--count", "3", "--max-traffic-cost", "0.01", "--exclude-geo", "cn, xx"), 0)
        self.assertIn("traffic down 0.0020 up 0.0030 $/GB", self.out)
        self.assertNotIn("offer 1 ", self.out)
        self.assertNotIn("offer 2 ", self.out)
        body = [c[2] for c in self.fake.calls if c[1].endswith("/search/asks/")][0]
        self.assertIn("inet_down_cost", body["select_cols"])
        self.assertIn("inet_up_cost", body["select_cols"])
        self.run_cmd("plan", "--count", "6")
        self.assertIn("traffic down 0.0020 up n/a $/GB", self.out)

    def test_rent_stores_filters_and_instance_facts(self):
        self.assertEqual(self.rent(2, ["--max-traffic-cost", "0.01", "--exclude-geo", "xx,cn", "--premium", "1.2"]), 0)
        st = self.state()
        self.assertEqual((st["max_traffic_cost"], st["exclude_geo"]), (0.01, ["CN", "XX"]))
        i = st["instances"][0]
        self.assertEqual((i["id"], i["offer_id"]), (self.ids()[0], 3))
        self.assertEqual((i["min_bid"], i["premium"], i["geolocation"], i["inet_down"]), (0.12, 1.2, "Berlin, DE", 500.0))
        self.assertEqual((i["inet_down_cost"], i["inet_up_cost"]), (0.002, 0.003))
        self.assertEqual([x["offer_id"] for x in st["instances"]], [3, 6])  # 4 lacks a price, 5 has no geo; 6 is US

    def test_no_filters_by_default(self):
        self.rent(1)
        st = self.state()
        self.assertEqual((st["max_traffic_cost"], st["exclude_geo"]), (None, []))
        self.assertEqual(st["instances"][0]["offer_id"], 1)

    def test_invalid_values_refused_before_any_call(self):
        for opt in (["--exclude-geo", "CHN"], ["--exclude-geo", "C1"], ["--max-traffic-cost", "-0.1"]):
            self.assertEqual(self.run_cmd("plan", *opt), 2, opt)
            self.assertEqual(self.rent(1, opt), 2, opt)
        self.assertEqual(self.fake.calls, [])

    def test_add_reuses_filters_and_cli_overrides(self):
        self.fake.offers += [offer(7, 17, 27, 0.16, geo="Hunan, CN", down_cost=0.001, up_cost=0.001),
                             offer(8, 18, 28, 0.17, geo="Paris, FR", down_cost=0.0025, up_cost=0.0025),
                             offer(9, 19, 29, 0.18, geo="Rome, IT", down_cost=0.05, up_cost=0.05)]
        self.assertEqual(self.rent(1, ["--max-traffic-cost", "0.003", "--exclude-geo", "CN"]), 0)
        self.assertEqual([i["offer_id"] for i in self.state()["instances"]], [3])
        self.assertEqual(self.run_cmd("rent", "--add", "1", "--yes"), 0)  # stored filters: only offer 8 is left
        self.assertEqual([i["offer_id"] for i in self.state()["instances"]], [3, 8])
        # command line overrides: no geo filter, tighter traffic limit; the cheapest CN host is offer 1
        self.assertEqual(self.run_cmd("rent", "--add", "1", "--exclude-geo", "", "--max-traffic-cost", "0.001", "--yes"), 0)
        self.assertEqual([i["offer_id"] for i in self.state()["instances"]], [3, 8, 1])
        st = self.state()
        self.assertEqual((st["max_traffic_cost"], st["exclude_geo"]), (0.003, ["CN"]))  # the run's own values stay


class Pacing(Base):
    def bodies(self):
        return [c for c in self.fake.calls if c[0] == "PUT" and re.search(r"/api/v0/asks/\d+/$", c[1])]

    def test_gap_between_create_calls_only(self):
        self.rent(1)
        self.assertEqual(self.sleeps, [])
        self.run_cmd("stop", "--yes")
        self.sleeps.clear()
        self.rent(3)
        self.assertEqual(self.sleeps, [1.5, 1.5])

    def test_gap_also_between_walk_down_attempts(self):
        self.fake.create_results = [409]
        self.rent(1)
        self.assertEqual(self.sleeps, [1.5])

    def test_add_is_paced_too(self):
        self.rent(1)
        self.sleeps.clear()
        self.run_cmd("rent", "--add", "2", "--yes")
        self.assertEqual(self.sleeps, [1.5])

    def test_429_waits_and_retries_the_same_offer_once(self):
        self.fake.create_results = [429]
        self.assertEqual(self.rent(1), 0)
        self.assertEqual(self.sleeps, [6.0])
        urls = [c[1] for c in self.bodies()]
        self.assertEqual(urls, [canary.BASE + "/api/v0/asks/1/"] * 2)
        st = self.state()
        self.assertEqual(st["instances"][0]["offer_id"], 1)
        self.assertEqual(st["rent_errors"], [])

    def test_second_429_is_a_definite_refusal_and_the_walk_goes_on(self):
        self.fake.create_results = [429, 429]
        self.assertEqual(self.rent(1), 0)
        self.assertEqual(self.sleeps, [6.0, 1.5])
        self.assertEqual([c[1][-8:] for c in self.bodies()], ["/asks/1/", "/asks/1/", "/asks/2/"])
        st = self.state()
        self.assertEqual(st["instances"][0]["offer_id"], 2)
        self.assertEqual(len(st["rent_errors"]), 1)
        self.assertFalse(st["ambiguous"])

    def test_other_errors_are_not_retried(self):
        self.fake.create_results = [503]
        self.assertEqual(self.rent(1), 3)
        self.assertEqual(len(self.bodies()), 1)
        self.assertEqual(self.sleeps, [])


class ListingFieldsAndJson(Base):
    def start(self, n=3):
        path = os.path.join(self.tmp, "prof.json")
        with open(path, "w") as f:
            json.dump(PROFILE, f)
        self.rent(n, ["--profile", path])

    def samples(self):
        return [s for n in os.listdir(self.dir) if n.startswith("samples-") for s in self.lines(n) if s["type"] == "sample"]

    def test_billing_fields_are_stored_when_present(self):
        self.start(2)
        a, b = self.ids()
        extra = {"inet_down_billed": 1.5, "inet_up_billed": 0.25, "storage_total_cost": 0.07, "duration": 612.5,
                 "start_date": 1800000000.0}
        self.fake.rows[a].update(extra)
        self.poll()
        by = {s["id"]: s for s in self.samples()}
        for k, v in extra.items():
            self.assertEqual(by[a][k], v)
            self.assertNotIn(k, by[b])

    def test_json_covers_every_live_instance(self):
        self.start(4)
        a, b, c, d = self.ids()
        self.fake.set_status(a, "running")
        self.fake.set_status(b, "running")
        self.fake.hb_down.add(b)
        self.fake.rows[c]["actual_status"] = "loading"
        del self.fake.rows[d]
        self.fake.hb_paths.clear()
        self.assertEqual(self.run_cmd("urls", "--json"), 0)
        self.assertEqual(len(self.out.strip().splitlines()), 1)
        obj = json.loads(self.out)
        self.assertEqual(set(obj), {"t", "instances"})
        self.assertRegex(obj["t"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
        by = {i["id"]: i for i in obj["instances"]}
        self.assertEqual(list(by), [a, b, c, d])
        for i in by.values():
            self.assertEqual(set(i), {"id", "status", "intended", "url", "healthy"})
        self.assertEqual((by[a]["status"], by[a]["healthy"]), ("running", True))
        self.assertEqual(by[a]["url"], "http://%s:%s" % (self.fake.rows[a]["public_ipaddr"], 40000 + a))
        self.assertEqual((by[a]["intended"]), "running")
        self.assertEqual((by[b]["status"], by[b]["healthy"]), ("running", False))
        self.assertEqual((by[c]["status"], by[c]["healthy"]), ("loading", False))
        self.assertEqual((by[d]["status"], by[d]["intended"], by[d]["url"], by[d]["healthy"]), ("absent", None, None, False))
        self.assertEqual(len(self.fake.hb_paths), 1)  # only running instances are probed (b is down: no path logged)
        self.assertNotIn(self.state()["secret"], self.out)

    def test_json_skips_destroyed_and_handles_no_state(self):
        self.assertEqual(self.run_cmd("urls", "--json"), 0)
        self.assertEqual(json.loads(self.out)["instances"], [])
        self.start(2)
        a, b = self.ids()
        self.run_cmd("drop", "--id", str(a), "--yes")
        self.run_cmd("urls", "--json")
        self.assertEqual([i["id"] for i in json.loads(self.out)["instances"]], [b])

    def test_json_never_shows_the_secret(self):
        self.start(1)
        self.assertEqual(self.run_cmd("urls", "--json", "--show-secret"), 2)
        self.assertNotIn(self.state()["secret"], self.out)
        self.run_cmd("urls", "--json")
        self.assertNotIn(self.state()["secret"], self.out)


class StateLock(Base):
    def interleave(self, locked):
        """poll loads state, then (inside its own HTTP call) `rent --add 1` runs on the same directory."""
        self.assertEqual(self.rent(2), 0)
        before = self.ids()
        inner = {}
        fired = []
        real = self.fake.http

        def hook(method, url, *a, **kw):
            if method == "GET" and "/api/v1/instances/" in url and not fired:
                fired.append(1)
                inner["rc"] = self.run_cmd("rent", "--add", "1", "--yes")
                inner["out"] = self.out
            return real(method, url, *a, **kw)

        buf = io.StringIO()
        patches = [mock.patch.object(canary, "LOCK_WAIT", 0.3), mock.patch.object(canary, "POLL_LOCK_WAIT", 0.3)]
        if not locked:
            patches.append(mock.patch.object(canary, "take_lock", lambda d, w: None))
        self.fake.tick(60)
        with contextlib.ExitStack() as es:
            for p in patches:
                es.enter_context(p)
            with contextlib.redirect_stdout(buf):
                rc = canary.main(["--dir", self.dir, "--key-file", self.keyfile, "poll"], hook, self.fake.now,
                                 self.sleeps.append)
        return before, inner, rc

    def test_lost_update_happens_without_the_lock(self):
        before, inner, rc = self.interleave(locked=False)
        self.assertEqual(inner["rc"], 0)
        created = [i for i in self.fake.rows if i not in (777, 778)]
        self.assertEqual(len(created), 3)
        self.assertEqual(len(self.ids()), 2)  # the third instance is untracked: the bug

    def test_lock_prevents_the_lost_update(self):
        before, inner, rc = self.interleave(locked=True)
        self.assertEqual(inner["rc"], 4)
        self.assertIn("nothing was done", inner["out"])
        self.assertEqual(rc, 0)
        self.assertEqual(self.ids(), before)
        self.assertEqual(len([i for i in self.fake.rows if i not in (777, 778)]), 2)  # inner add created nothing

    def test_lock_released_after_command(self):
        self.assertEqual(self.rent(2), 0)
        self.assertEqual(self.run_cmd("rent", "--add", "1", "--yes"), 0)
        self.assertEqual(len(self.ids()), 3)

    def test_poll_skips_when_busy(self):
        self.assertEqual(self.rent(2), 0)
        f = canary.take_lock(self.dir, 1)
        try:
            with mock.patch.object(canary, "POLL_LOCK_WAIT", 0.2):
                self.assertEqual(self.run_cmd("poll"), 0)
            self.assertIn("busy, skipped", self.out)
        finally:
            f.close()

    def test_other_mutators_exit_4_when_busy_and_do_nothing(self):
        self.assertEqual(self.rent(2), 0)
        calls = len(self.fake.calls)
        iid = self.ids()[0]
        f = canary.take_lock(self.dir, 1)
        try:
            with mock.patch.object(canary, "LOCK_WAIT", 0.2):
                for args in (("drop", "--id", str(iid), "--yes"), ("stop", "--yes"),
                             ("bid", "--id", str(iid), "--price", "0.2", "--yes"), ("rent", "--count", "1", "--yes")):
                    self.assertEqual(self.run_cmd(*args), 4, args)
        finally:
            f.close()
        self.assertEqual(len(self.fake.calls), calls)
        self.assertEqual(len(self.ids()), 2)

    def test_read_only_commands_ignore_the_lock(self):
        self.assertEqual(self.rent(2), 0)
        f = canary.take_lock(self.dir, 1)
        try:
            with mock.patch.object(canary, "LOCK_WAIT", 0.2):
                for args in (("status",), ("report",), ("list",), ("urls",), ("orphans",), ("plan",)):
                    self.assertNotEqual(self.run_cmd(*args), 4, args)
        finally:
            f.close()


class Orphans(Base):
    def test_lists_own_labels_not_in_state_and_destroys_nothing(self):
        self.assertEqual(self.rent(2), 0)
        tracked = self.ids()
        self.fake.add_foreign(900, "undead-demo")
        self.fake.add_foreign(901, "custom-label")
        self.fake.add_foreign(902, "other")
        before = len(self.fake.calls)
        self.assertEqual(self.run_cmd("orphans"), 0)
        self.assertIn("orphans (our label, not in state.json): 2", self.out)
        for i in (777, 900):
            self.assertIn("  %d label" % i, self.out)
        self.assertNotIn("  778 label", self.out)
        self.assertNotIn("  902 label", self.out)
        self.assertNotIn("  901 label", self.out)
        for i in tracked:
            self.assertNotIn("  %d label" % i, self.out)
        self.assertTrue(all(c[0] == "GET" for c in self.fake.calls[before:]))
        self.assertNotIn(KEY, self.out)

    def test_profile_label_counts(self):
        prof = os.path.join(self.tmp, "p.json")
        with open(prof, "w") as f:
            json.dump({"image": "img/x:1", "args": ["--api-key", "{KEY}"], "label": "custom-label"}, f)
        self.assertEqual(self.rent(1, ("--profile", prof)), 0)
        self.fake.add_foreign(901, "custom-label")
        self.assertEqual(self.run_cmd("orphans"), 0)
        self.assertIn("  901 label custom-label", self.out)


class RealHttpInvariant(unittest.TestCase):
    """real_http is the only network function: check what it puts on the wire, with a stand-in for the opener."""

    def capture(self, url, key=None, headers=None):
        seen = []

        class Resp:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self, n):
                return b"{}"

        class Opener:
            def open(self, req, timeout=None):
                seen.append(dict((k.lower(), v) for k, v in req.header_items()))
                return Resp()

        with mock.patch("canary.urllib.request.build_opener", return_value=Opener()):
            canary.real_http("GET", url, key, None, 5, headers)
        return seen[0]

    def test_key_only_to_api_host(self):
        self.assertEqual(self.capture(canary.BASE + "/api/v0/x/", "K")["authorization"], "Bearer K")
        self.assertNotIn("authorization", self.capture("http://1.2.3.4:80/", "K"))
        self.assertNotIn("authorization", self.capture("https://console.vast.ai.evil.com/", "K"))

    def test_instance_headers_only_to_non_api_host(self):
        h = self.capture("http://1.2.3.4:8000/v1/models", None, {"Authorization": "Bearer S"})
        self.assertEqual(h["authorization"], "Bearer S")
        with self.assertRaises(canary.ApiError):
            self.capture(canary.BASE + "/api/v0/x/", "K", {"Authorization": "Bearer S"})
        h = self.capture("http://1.2.3.4:8000/", "K", {"Authorization": "Bearer S"})
        self.assertEqual(h["authorization"], "Bearer S")  # the API key never rides along



if __name__ == "__main__":
    unittest.main()
