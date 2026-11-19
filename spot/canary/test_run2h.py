"""run2h.sh against a fake router (two ports), the real fleet.py, a fake canary and a fake load.py, in a temporary
root (the script's /opt/undead is rewritten, the ports and the 25 minute wait are shortened, sleep is shimmed)."""
import json
import os
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SECRET = "SEKRET-xyz"

FAKE_CANARY = '''import json, os, sys
a = sys.argv[1:]
root = os.path.dirname(os.path.abspath(__file__))
if a[-1] == "--json":
    sys.stdout.write(open(os.path.join(root, "json.txt")).read())
elif "urls" in a and "--show-secret" in a:
    sys.stdout.write(open(os.path.join(root, "urls.txt")).read())
else:
    sys.exit(9)
'''
FAKE_ROUTER = '''import json, os, stat, sys, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.exists(os.path.join(root, "router-dies")):
    sys.exit(4)
av = sys.argv[1:]
tokfile = av[av.index("--admin-token-file") + 1]
tok = open(tokfile).read().strip()
open(os.path.join(root, "routercmd.json"), "w").write(json.dumps({
    "argv": av, "key": os.environ.get("UNDEAD_REPLICA_API_KEY"), "cwd": os.getcwd(), "pid": os.getpid(),
    "token_mode": stat.S_IMODE(os.stat(tokfile).st_mode)}))
reg = {}
class Main(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'{"answers": 3}')
    def log_message(self, *a): pass
class Admin(BaseHTTPRequestHandler):
    def ok(self):
        return self.headers.get("Authorization") == "Bearer " + tok
    def send(self, code, obj):
        raw = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw)
    def do_GET(self):
        if not self.ok(): return self.send(401, {})
        self.send(200, [dict(v, name=k, in_flight=0, state="active") for k, v in reg.items()])
    def do_PUT(self):
        if not self.ok(): return self.send(401, {})
        reg[self.path.rsplit("/", 1)[1]] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.send(201, {})
    def do_DELETE(self):
        if not self.ok(): return self.send(401, {})
        reg.pop(self.path.rsplit("/", 1)[1], None); self.send(200, {})
    def log_message(self, *a): pass
m = HTTPServer(("127.0.0.1", av and int(av[av.index("--port") + 1])), Main)
ad = HTTPServer(("127.0.0.1", int(av[av.index("--admin-port") + 1])), Admin)
threading.Thread(target=ad.serve_forever, daemon=True).start()
m.serve_forever()
'''
FAKE_LOAD = '''import os, sys, time
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "loadcmd.txt"), "w").write(" ".join(sys.argv[1:]))
time.sleep(1.0)
print("fake summary")
'''


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def slurp(path):
    with open(path) as f:
        return f.read()


def inst(iid, healthy=True):
    return {"id": iid, "status": "running" if healthy else "loading", "intended": "running",
            "url": "http://10.0.0.%d:4%04d" % (iid, iid), "healthy": healthy}


class Run2h(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.root = os.path.join(self.tmp, "undead")
        for d in ("canary", "router-registry", "venv/bin"):
            os.makedirs(os.path.join(self.root, d))
        for path, text in (("canary/canary.py", FAKE_CANARY), ("router-registry/router.py", FAKE_ROUTER),
                           ("canary/load.py", FAKE_LOAD)):
            with open(os.path.join(self.root, path), "w") as f:
                f.write(text)
        shutil.copy(os.path.join(HERE, "fleet.py"), os.path.join(self.root, "canary/fleet.py"))
        os.symlink(sys.executable, os.path.join(self.root, "venv/bin/python"))
        self.rport, self.aport = free_port(), free_port()
        script = slurp(os.path.join(HERE, "run2h.sh"))
        for old, new in (("/opt/undead", self.root), ("ROUTER_PORT=8088", "ROUTER_PORT=%d" % self.rport),
                         ("ADMIN_PORT=8089", "ADMIN_PORT=%d" % self.aport), ("WAIT_TRIES=300", "WAIT_TRIES=4")):
            self.assertIn(old, script)
            script = script.replace(old, new)
        self.script = os.path.join(self.tmp, "run2h.sh")
        with open(self.script, "w") as f:
            f.write(script)
        self.shim = os.path.join(self.tmp, "shim")
        os.makedirs(self.shim)
        with open(os.path.join(self.shim, "sleep"), "w") as f:
            f.write("#!/bin/sh\nexec /bin/sleep 0.3\n")
        os.chmod(os.path.join(self.shim, "sleep"), 0o755)
        self.state = os.path.join(self.tmp, "state")
        os.makedirs(self.state)
        with open(os.path.join(self.state, "state.json"), "w") as f:
            json.dump({"phase": "active", "secret": SECRET, "instances": []}, f)
        self.set_canary([inst(1), inst(2)])

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def set_canary(self, instances, urls=True):
        with open(os.path.join(self.root, "canary/json.txt"), "w") as f:
            f.write(json.dumps({"t": "x", "instances": instances}) + "\n")
        with open(os.path.join(self.root, "canary/urls.txt"), "w") as f:
            f.write("1 http://10.0.0.1:1\nsecret: %s\n" % SECRET if urls else "no running instances\n")

    def run_script(self, *args, timeout=60):
        env = dict(os.environ, PATH=self.shim + os.pathsep + os.environ["PATH"])
        return subprocess.run(["sh", self.script, self.state, *args], capture_output=True, text=True, env=env,
                              timeout=timeout)

    def leftovers(self):
        out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
        return [l for l in out.splitlines() if self.state in l and ("router.py" in l or "fleet.py" in l or "load.py" in l)]

    def check_quiet(self, p):
        for text in (p.stdout, p.stderr):
            self.assertNotIn(SECRET, text)
        token = slurp(os.path.join(self.state, "admin.token")).strip()
        self.assertTrue(token)
        for name in os.listdir(self.state):
            if name in ("state.json", "admin.token"):
                continue
            self.assertNotIn(SECRET, slurp(os.path.join(self.state, name)), name)
            self.assertNotIn(token, slurp(os.path.join(self.state, name)), name)
        self.assertNotIn(token, p.stdout + p.stderr)

    def test_full_run(self):
        p = self.run_script()
        self.assertEqual(p.returncode, 0, p.stderr)
        cmd = json.loads(slurp(os.path.join(self.root, "routercmd.json")))
        self.assertEqual(cmd["argv"], [
            "--model", "Qwen/Qwen2.5-7B-Instruct", "--host", "127.0.0.1", "--port", str(self.rport),
            "--admin-host", "127.0.0.1", "--admin-port", str(self.aport),
            "--admin-token-file", os.path.join(self.state, "admin.token"),
            "--journal", os.path.join(self.state, "journal.jsonl"), "--queue-timeout", "10"])
        self.assertEqual(cmd["key"], SECRET)
        self.assertEqual(cmd["cwd"], os.path.join(self.root, "router-registry"))
        self.assertEqual(cmd["token_mode"], 0o600)
        self.assertEqual(stat.S_IMODE(os.stat(os.path.join(self.state, "admin.token")).st_mode), 0o600)
        self.assertEqual(slurp(os.path.join(self.root, "canary/loadcmd.txt")),
                         "--url http://127.0.0.1:%d --model Qwen/Qwen2.5-7B-Instruct --streams 24 --duration 7200 "
                         "--max-tokens 500 --out %s/run.jsonl" % (self.rport, self.state))
        self.assertRegex(slurp(os.path.join(self.state, "load-start.txt")), r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z\n$")
        self.assertEqual(json.loads(slurp(os.path.join(self.state, "metrics-after.json"))), {"answers": 3})
        reps = json.loads(slurp(os.path.join(self.state, "replicas-after.json")))
        self.assertEqual(sorted(r["name"] for r in reps), ["i1", "i2"])
        self.assertEqual(reps[0]["url"], "http://10.0.0.1:40001")
        self.assertTrue(os.path.exists(os.path.join(self.state, "fleet.jsonl")))
        self.assertIn("fake summary", p.stdout)
        self.assertIn("load-start.txt", p.stdout)
        self.assertEqual(self.leftovers(), [])
        with self.assertRaises(ProcessLookupError):
            os.kill(cmd["pid"], 0)
        self.check_quiet(p)

    def test_streams_and_seconds_arguments(self):
        p = self.run_script("5", "9")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("--streams 5 --duration 9 ", slurp(os.path.join(self.root, "canary/loadcmd.txt")))

    def test_secret_comes_from_state_json_before_any_instance_runs(self):
        self.set_canary([inst(1), inst(2)], urls=False)
        p = self.run_script()
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(json.loads(slurp(os.path.join(self.root, "routercmd.json")))["key"], SECRET)
        self.check_quiet(p)

    def test_no_secret_anywhere_stops_before_the_router(self):
        self.set_canary([inst(1), inst(2)], urls=False)
        with open(os.path.join(self.state, "state.json"), "w") as f:
            json.dump({"phase": "active", "instances": []}, f)
        p = self.run_script()
        self.assertEqual(p.returncode, 1)
        self.assertIn("no instance secret", p.stderr)
        self.assertFalse(os.path.exists(os.path.join(self.root, "routercmd.json")))

    def test_fewer_than_two_replicas_exits_3_and_stops_everything(self):
        self.set_canary([inst(1), inst(2, healthy=False)])
        p = self.run_script()
        self.assertEqual(p.returncode, 3, p.stderr)
        self.assertIn("NOT destroyed", p.stderr)
        self.assertFalse(os.path.exists(os.path.join(self.root, "canary/loadcmd.txt")))
        self.assertEqual(self.leftovers(), [])
        cmd = json.loads(slurp(os.path.join(self.root, "routercmd.json")))
        with self.assertRaises(ProcessLookupError):
            os.kill(cmd["pid"], 0)
        self.check_quiet(p)

    def test_router_that_exits_stops_the_run(self):
        open(os.path.join(self.root, "router-dies"), "w").close()
        p = self.run_script()
        self.assertEqual(p.returncode, 1)
        self.assertIn("router exited", p.stderr)
        self.assertEqual(self.leftovers(), [])

    def test_bad_arguments(self):
        env = dict(os.environ)
        for args in ([], [self.state, "x"], [self.state, "0"], [self.state, "4", "z"], ["/nonexistent"], [self.state, "1", "2", "3"]):
            p = subprocess.run(["sh", self.script, *args], capture_output=True, text=True, env=env, timeout=20)
            self.assertEqual(p.returncode, 2, args)


if __name__ == "__main__":
    unittest.main()
