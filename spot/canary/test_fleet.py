import io
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

import fleet

TOKEN = "TOKEN-abc-123"


class FakeAdmin:
    """The router's admin API in memory: http(method, url, token, body) -> (status, json)."""

    def __init__(self):
        self.reg = {}
        self.calls = []
        self.tokens = []
        self.fail = {}  # (method, name or "list") -> status or "net"

    def http(self, method, url, token, body=None):
        self.tokens.append(token)
        path = url.split("/admin/replicas", 1)[1].lstrip("/")
        name = path or "list"
        self.calls.append((method, name, body))
        f = self.fail.get((method, name))
        if f == "net":
            raise ConnectionResetError("boom")
        if f:
            return f, {"error": "x"}
        if token != TOKEN:
            return 401, {"error": "token"}
        if method == "GET":
            return 200, [dict(v, name=k) for k, v in self.reg.items()]
        if method == "PUT":
            new = name not in self.reg
            self.reg.setdefault(name, {"state": "active", "in_flight": 0, "max_streams": 0}).update(body)
            return (201 if new else 200), {}
        if method == "DELETE":
            return 200, {"removed": self.reg.pop(name, None) is not None}
        return 405, None

    def puts(self):
        return [(n, b) for m, n, b in self.calls if m == "PUT"]

    def deletes(self):
        return [n for m, n, b in self.calls if m == "DELETE"]


def inst(iid, healthy=True, status="running", url=None):
    return {"id": iid, "status": status, "intended": "running",
            "url": url if url is not None else "http://10.0.0.%d:4%04d" % (iid % 250, iid), "healthy": healthy}


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.dir = os.path.join(self.tmp, "state")
        os.makedirs(self.dir)
        self.token_file = os.path.join(self.tmp, "admin.token")
        with open(self.token_file, "w") as f:
            f.write(TOKEN + "\n")
        self.admin = FakeAdmin()
        self.instances = []
        self.canary_error = None
        self.cmds = []
        self.sleeps = []
        self.t = 1_800_000_000.0

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def runner(self, cmd):
        self.cmds.append(cmd)
        if self.canary_error:
            raise RuntimeError(self.canary_error) if self.canary_error != "garbage" else ValueError()
        return json.dumps({"t": "x", "instances": self.instances})

    def make(self, *extra):
        a = fleet.parse(["--dir", self.dir, "--router-admin", "http://127.0.0.1:8089", "--token-file", self.token_file,
                         "--canary", "/x/canary.py", *extra])
        return fleet.Fleet(a, TOKEN, self.runner, self.admin.http, self.sleeps.append, lambda: self.t)

    def passes(self, fl, n=1):
        recs = []
        for _ in range(n):
            self.t += 15
            recs.append(fl.one_pass())
        return recs

    def log(self):
        with open(os.path.join(self.dir, "fleet.jsonl")) as f:
            return [json.loads(x) for x in f]

    def set_state(self, phase):
        with open(os.path.join(self.dir, "state.json"), "w") as f:
            json.dump({"phase": phase}, f)


class Reconcile(Base):
    def test_healthy_instances_are_added_with_url_and_max_streams(self):
        self.instances = [inst(1), inst(2), inst(3, healthy=False, status="loading")]
        fl = self.make("--max-streams", "12")
        rec = self.passes(fl)[0]
        self.assertEqual(self.admin.puts(), [("i1", {"url": "http://10.0.0.1:40001", "max_streams": 12}),
                                             ("i2", {"url": "http://10.0.0.2:40002", "max_streams": 12})])
        self.assertEqual(rec["healthy"], [1, 2])
        self.assertEqual(rec["registered"], ["i1", "i2"])
        self.assertEqual(rec["actions"], [{"op": "put", "name": "i1"}, {"op": "put", "name": "i2"}])
        self.assertEqual(rec["instances"]["3"], {"status": "loading", "intended": "running", "healthy": False})
        self.assertEqual(rec["router"], {"i1": {"in_flight": 0, "state": "active"}, "i2": {"in_flight": 0, "state": "active"}})
        self.assertEqual(rec["errors"], [])

    def test_default_max_streams_and_command(self):
        self.instances = [inst(1)]
        fl = self.make()
        self.passes(fl)
        self.assertEqual(self.admin.puts()[0][1]["max_streams"], 16)
        self.assertEqual(self.cmds[0][1:], ["/x/canary.py", "--dir", self.dir, "urls", "--json"])

    def test_nothing_is_repeated_and_a_changed_url_is_put_again(self):
        self.instances = [inst(1)]
        fl = self.make()
        self.passes(fl, 3)
        self.assertEqual(len(self.admin.puts()), 1)
        self.instances = [inst(1, url="http://99.9.9.9:5000")]
        rec = self.passes(fl)[0]
        self.assertEqual(self.admin.puts()[1], ("i1", {"url": "http://99.9.9.9:5000", "max_streams": 16}))
        self.assertEqual(rec["actions"], [{"op": "put", "name": "i1"}])
        self.assertEqual(self.admin.deletes(), [])

    def test_removed_only_after_two_unhealthy_passes(self):
        self.instances = [inst(1), inst(2)]
        fl = self.make()
        self.passes(fl)
        self.instances = [inst(1, healthy=False, status="exited"), inst(2)]
        r1 = self.passes(fl)[0]
        self.assertEqual((r1["actions"], r1["registered"]), ([], ["i1", "i2"]))
        r2 = self.passes(fl)[0]
        self.assertEqual(r2["actions"], [{"op": "delete", "name": "i1"}])
        self.assertEqual(r2["registered"], ["i2"])
        self.assertEqual(r2["healthy"], [2])

    def test_flap_that_recovers_after_one_bad_pass_stays(self):
        self.instances = [inst(1)]
        fl = self.make()
        self.passes(fl)
        for _ in range(3):
            self.instances = [inst(1, healthy=False)]
            self.passes(fl)
            self.instances = [inst(1)]
            self.passes(fl)
        self.assertEqual(self.admin.deletes(), [])
        self.assertEqual(len(self.admin.puts()), 1)

    def test_vanished_instance_is_removed_at_once(self):
        self.instances = [inst(1), inst(2)]
        fl = self.make()
        self.passes(fl)
        self.instances = [inst(2)]
        rec = self.passes(fl)[0]
        self.assertEqual(rec["actions"], [{"op": "delete", "name": "i1"}])

    def test_back_after_removal_is_registered_again(self):
        self.instances = [inst(1)]
        fl = self.make()
        self.passes(fl)
        self.instances = [inst(1, healthy=False)]
        self.passes(fl, 2)
        self.assertNotIn("i1", self.admin.reg)
        self.instances = [inst(1)]
        self.passes(fl)
        self.assertIn("i1", self.admin.reg)

    def test_foreign_names_are_never_touched(self):
        foreign = ["main", "static-1", "idle", "ix1", "i", "i-2", "x12", "I3"]
        for n in foreign:
            self.admin.reg[n] = {"url": "http://1.1.1.1:1", "state": "active", "in_flight": 0}
        self.instances = [inst(1), inst(2, healthy=False)]
        fl = self.make()
        self.passes(fl, 4)
        self.assertEqual([n for n in self.admin.reg if n in foreign], foreign)
        for m, n, b in self.admin.calls:
            if m != "GET":
                self.assertRegex(n, r"^i[0-9]+$")
        self.set_state("finished")
        fl.run()
        self.assertEqual([n for n in self.admin.reg if n in foreign], foreign)
        self.assertEqual([n for n in self.admin.reg if n not in foreign], [])

    def test_unregistered_unhealthy_instance_never_triggers_a_delete(self):
        self.instances = [inst(1, healthy=False)]
        self.passes(self.make(), 3)
        self.assertEqual(self.admin.calls.count(("DELETE", "i1", None)), 0)


class Errors(Base):
    def test_canary_failures_change_nothing_and_do_not_count(self):
        self.instances = [inst(1)]
        fl = self.make()
        self.passes(fl)
        self.instances = [inst(1, healthy=False)]
        self.passes(fl)  # first bad pass
        for err in ("rc 1", "timeout", "garbage"):
            self.canary_error = err
            rec = self.passes(fl)[0]
            self.assertEqual(rec["actions"], [])
            self.assertTrue(rec["errors"] and rec["errors"][0].startswith("canary:"))
            self.assertEqual(rec["registered"], ["i1"])
        self.assertEqual(self.admin.deletes(), [])
        self.canary_error = None
        rec = self.passes(fl)[0]  # the second bad pass counts only now
        self.assertEqual(rec["actions"], [{"op": "delete", "name": "i1"}])

    def test_router_list_errors_are_recorded_and_the_loop_goes_on(self):
        self.instances = [inst(1)]
        fl = self.make()
        self.admin.fail[("GET", "list")] = "net"
        rec = self.passes(fl)[0]
        self.assertEqual(rec["errors"], ["router list: ConnectionResetError"])
        self.assertIsNone(rec["registered"])
        self.assertEqual(self.admin.puts(), [])
        self.admin.fail[("GET", "list")] = 500
        self.assertEqual(self.passes(fl)[0]["errors"], ["router list: HTTP 500"])
        del self.admin.fail[("GET", "list")]
        self.assertEqual(self.passes(fl)[0]["errors"], [])
        self.assertIn("i1", self.admin.reg)

    def test_failed_put_is_retried_next_pass(self):
        self.instances = [inst(1)]
        fl = self.make()
        self.admin.fail[("PUT", "i1")] = 400
        rec = self.passes(fl)[0]
        self.assertEqual((rec["errors"], rec["actions"], rec["registered"]), (["put i1: HTTP 400"], [], []))
        del self.admin.fail[("PUT", "i1")]
        rec = self.passes(fl)[0]
        self.assertEqual((rec["errors"], rec["registered"]), ([], ["i1"]))

    def test_failed_delete_is_retried(self):
        self.instances = [inst(1)]
        fl = self.make()
        self.passes(fl)
        self.instances = []
        self.admin.fail[("DELETE", "i1")] = "net"
        rec = self.passes(fl)[0]
        self.assertEqual(rec["errors"], ["delete i1: ConnectionResetError"])
        self.assertEqual(rec["registered"], ["i1"])
        del self.admin.fail[("DELETE", "i1")]
        self.assertEqual(self.passes(fl)[0]["actions"], [{"op": "delete", "name": "i1"}])

    def test_wrong_token_is_an_error_not_a_crash(self):
        self.instances = [inst(1)]
        fl = self.make()
        fl.token = "wrong"
        rec = self.passes(fl)[0]
        self.assertEqual(rec["errors"], ["router list: HTTP 401"])

    def test_token_is_sent_but_never_written(self):
        self.instances = [inst(1)]
        fl = self.make()
        self.admin.fail[("PUT", "i1")] = 500
        self.passes(fl, 2)
        self.assertEqual(set(self.admin.tokens), {TOKEN})
        for name in os.listdir(self.dir):
            with open(os.path.join(self.dir, name)) as f:
                self.assertNotIn(TOKEN, f.read())


class Loop(Base):
    def test_log_line_has_all_fields(self):
        self.instances = [inst(1)]
        self.passes(self.make())
        rec = self.log()[0]
        self.assertEqual(set(rec), {"t", "ts", "healthy", "registered", "instances", "router", "actions", "errors"})
        self.assertRegex(rec["t"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")

    def test_once_runs_a_single_pass(self):
        self.instances = [inst(1)]
        fl = self.make("--once")
        self.assertEqual(fl.run(), 0)
        self.assertEqual(len(self.log()), 1)
        self.assertEqual(self.sleeps, [])
        self.assertIn("i1", self.admin.reg)

    def test_loops_with_interval_until_the_run_is_finished_then_cleans_up(self):
        self.instances = [inst(1), inst(2)]
        fl = self.make("--interval", "7")
        count = []

        def sleep(s):
            self.sleeps.append(s)
            count.append(1)
            if len(count) == 3:
                self.set_state("finished")
        fl.sleep = sleep
        self.set_state("active")
        self.assertEqual(fl.run(), 0)
        self.assertEqual(self.sleeps, [7.0, 7.0, 7.0])
        recs = self.log()
        self.assertEqual(len(recs), 4)  # three normal passes and the cleanup pass
        self.assertEqual(recs[2]["registered"], ["i1", "i2"])
        last = recs[-1]
        self.assertEqual(last["actions"], [{"op": "delete", "name": "i1"}, {"op": "delete", "name": "i2"}])
        self.assertEqual(last["registered"], [])
        self.assertEqual(self.admin.reg, {})
        self.assertEqual(len(self.cmds), 3)  # no canary call for the cleanup

    def test_finished_state_at_start_cleans_up_at_once(self):
        self.admin.reg = {"i5": {"url": "http://1.2.3.4:5"}, "main": {"url": "http://9.9.9.9:9"}}
        self.set_state("finished")
        self.assertEqual(self.make().run(), 0)
        self.assertEqual(list(self.admin.reg), ["main"])
        self.assertEqual(self.cmds, [])

    def test_missing_or_broken_state_json_is_not_finished(self):
        fl = self.make("--once")
        self.assertFalse(fl.finished())
        with open(os.path.join(self.dir, "state.json"), "w") as f:
            f.write("{broken")
        self.assertFalse(fl.finished())
        self.set_state("active")
        self.assertFalse(fl.finished())

    def test_cleanup_with_router_down_records_error_and_exits(self):
        self.admin.reg = {"i5": {"url": "http://1.2.3.4:5"}}
        self.set_state("finished")
        self.admin.fail[("GET", "list")] = "net"
        self.assertEqual(self.make().run(), 0)
        self.assertEqual(self.log()[-1]["errors"], ["router list: ConnectionResetError"])


class Cli(Base):
    def test_token_file_problems(self):
        err = io.StringIO()
        old = sys.stderr
        sys.stderr = err
        try:
            self.assertEqual(fleet.main(["--dir", self.dir, "--router-admin", "http://x", "--token-file", "/nonexistent"]), 2)
            with open(self.token_file, "w") as f:
                f.write("\n")
            self.assertEqual(fleet.main(["--dir", self.dir, "--router-admin", "http://x", "--token-file", self.token_file]), 2)
        finally:
            sys.stderr = old
        self.assertNotIn(TOKEN, err.getvalue())

    def test_default_canary_path_is_next_to_fleet_py(self):
        a = fleet.parse(["--dir", "d", "--router-admin", "http://x", "--token-file", "t"])
        self.assertEqual(a.canary, os.path.join(os.path.dirname(os.path.abspath(fleet.__file__)), "canary.py"))
        self.assertEqual((a.interval, a.max_streams, a.once), (15.0, 16, False))


class RealSockets(Base):
    """fleet.py end to end: a real subprocess as canary and a real HTTP server as the router admin API."""

    def test_once_against_real_http_and_a_fake_canary_process(self):
        reg = {"main": {"url": "http://1.1.1.1:1"}}
        seen = []

        class H(BaseHTTPRequestHandler):
            def _send(self, code, obj):
                raw = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def _auth(self):
                seen.append(self.headers.get("Authorization"))
                return self.headers.get("Authorization") == "Bearer " + TOKEN

            def do_GET(self):
                if not self._auth():
                    return self._send(401, {})
                self._send(200, [dict(v, name=k, in_flight=0, state="active") for k, v in reg.items()])

            def do_PUT(self):
                if not self._auth():
                    return self._send(401, {})
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                reg[self.path.rsplit("/", 1)[1]] = body
                self._send(201, {})

            def do_DELETE(self):
                self._auth()
                reg.pop(self.path.rsplit("/", 1)[1], None)
                self._send(200, {})

            def log_message(self, *a):
                pass

        srv = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        fake = os.path.join(self.tmp, "canary.py")
        with open(fake, "w") as f:
            f.write("import json, sys\nassert sys.argv[1:] == ['--dir', %r, 'urls', '--json'], sys.argv\n"
                    "print(json.dumps({'t': 'x', 'instances': [{'id': 7, 'status': 'running', 'intended': 'running', "
                    "'url': 'http://5.6.7.8:9', 'healthy': True}]}))\n" % self.dir)
        try:
            rc = fleet.main(["--dir", self.dir, "--router-admin", "http://127.0.0.1:%d" % srv.server_address[1],
                             "--token-file", self.token_file, "--canary", fake, "--once"])
        finally:
            srv.shutdown()
            srv.server_close()
        self.assertEqual(rc, 0)
        self.assertEqual(reg["i7"], {"url": "http://5.6.7.8:9", "max_streams": 16})
        self.assertIn("main", reg)
        self.assertEqual(set(seen), {"Bearer " + TOKEN})
        rec = self.log()[0]
        self.assertEqual((rec["errors"], rec["healthy"], rec["actions"]), ([], [7], [{"op": "put", "name": "i7"}]))

    def test_canary_process_failure_is_an_error_line(self):
        fake = os.path.join(self.tmp, "bad.py")
        with open(fake, "w") as f:
            f.write("import sys\nsys.exit(3)\n")
        a = fleet.parse(["--dir", self.dir, "--router-admin", "http://127.0.0.1:1", "--token-file", self.token_file,
                         "--canary", fake, "--once"])
        rec = fleet.Fleet(a, TOKEN).one_pass()
        self.assertIn("canary: rc 3", rec["errors"])
        self.assertTrue(any(e.startswith("router list:") for e in rec["errors"]))


if __name__ == "__main__":
    unittest.main()
