"""demo_preempt.sh against fake canary / router / load in a temporary root (the script's /opt/undead is rewritten)."""
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SECRET = "SEKRET-xyz"

FAKE_CANARY = '''import os, sys
a = sys.argv[1:]
root = os.path.dirname(os.path.abspath(__file__))
if "urls" in a:
    assert "--healthy" in a and "--show-secret" in a
    sys.stdout.write(open(os.path.join(root, "urls.txt")).read())
elif "bid" in a:
    open(os.path.join(root, "bidcall.txt"), "w").write(" ".join(a))
'''
FAKE_ROUTER = '''import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
open(os.path.join(root, "routercmd.json"), "w").write(json.dumps({"argv": sys.argv[1:], "key": os.environ.get("UNDEAD_REPLICA_API_KEY")}))
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'{"answers": 3}')
    def log_message(self, *a): pass
HERE_PORT = int(sys.argv[sys.argv.index("--port") + 1])
HTTPServer(("127.0.0.1", HERE_PORT), H).serve_forever()
'''
FAKE_LOAD = '''import os, sys, time
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "loadcmd.txt"), "w").write(" ".join(sys.argv[1:]))
time.sleep(1.5)
print("fake summary")
'''


def slurp(path):
    with open(path) as f:
        return f.read()


class DemoScript(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.root = os.path.join(self.tmp, "undead")
        for d in ("canary", "router", "venv/bin"):
            os.makedirs(os.path.join(self.root, d))
        for path, text in (("canary/canary.py", FAKE_CANARY), ("router/router.py", FAKE_ROUTER), ("canary/load.py", FAKE_LOAD)):
            with open(os.path.join(self.root, path), "w") as f:
                f.write(text)
        os.symlink(sys.executable, os.path.join(self.root, "venv/bin/python"))
        with open(os.path.join(HERE, "demo_preempt.sh")) as f:
            script = f.read()
        self.assertIn("/opt/undead/canary/canary.py", script)
        self.script = os.path.join(self.tmp, "demo.sh")
        with open(self.script, "w") as f:
            f.write(script.replace("/opt/undead", self.root))
        self.shim = os.path.join(self.tmp, "shim")
        os.makedirs(self.shim)
        with open(os.path.join(self.shim, "sleep"), "w") as f:
            f.write("#!/bin/sh\nexec /bin/sleep 0.2\n")
        os.chmod(os.path.join(self.shim, "sleep"), 0o755)
        self.state = os.path.join(self.tmp, "state")
        os.makedirs(self.state)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_script(self, urls_txt, *args):
        with open(os.path.join(self.root, "canary/urls.txt"), "w") as f:
            f.write(urls_txt)
        env = dict(os.environ, PATH=self.shim + os.pathsep + os.environ["PATH"])
        p = subprocess.run(["sh", self.script, self.state, *args], capture_output=True, text=True, env=env, timeout=60)
        return p

    def read(self, rel):
        with open(os.path.join(self.root, rel)) as f:
            return f.read()

    URLS3 = "11 http://1.1.1.1:111\n12 http://2.2.2.2:222\n13 http://3.3.3.3:333\nsecret: " + SECRET + "\n"

    def test_all_healthy_go_to_the_router_and_first_is_killed(self):
        p = self.run_script(self.URLS3)
        self.assertEqual(p.returncode, 0, p.stderr)
        import json
        cmd = json.loads(self.read("routercmd.json"))
        self.assertEqual(cmd["argv"], ["--replicas", "http://1.1.1.1:111,http://2.2.2.2:222,http://3.3.3.3:333",
                                       "--model", "Qwen/Qwen2.5-7B-Instruct", "--host", "127.0.0.1", "--port", "8088"])
        self.assertEqual(cmd["key"], SECRET)
        self.assertEqual(self.read("canary/bidcall.txt"), "--dir %s bid --id 11 --price 0.05 --yes" % self.state)
        kill = slurp(os.path.join(self.state, "kill.txt")).split()
        self.assertEqual(len(kill), 2)
        self.assertRegex(kill[0], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z$")
        self.assertEqual(kill[1], "11")
        self.assertIn("--streams 8 --duration 240 --max-tokens 500", self.read("canary/loadcmd.txt"))
        self.assertIn('{"answers": 3}', slurp(os.path.join(self.state, "metrics-after.json")))
        self.assertIn("fake summary", p.stdout)
        self.assertIn("11", p.stdout)
        self.assertNotIn(SECRET, p.stdout + p.stderr)
        for n in os.listdir(self.state):
            self.assertNotIn(SECRET, slurp(os.path.join(self.state, n)), n)

    def test_kill_id_given_and_healthy(self):
        p = self.run_script(self.URLS3, "13")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("--id 13 ", self.read("canary/bidcall.txt"))
        self.assertEqual(slurp(os.path.join(self.state, "kill.txt")).split()[1], "13")

    def test_kill_id_not_healthy_stops_before_anything_starts(self):
        p = self.run_script(self.URLS3, "99")
        self.assertEqual(p.returncode, 1)
        self.assertIn("not one of the healthy instances", p.stderr)
        self.assertFalse(os.path.exists(os.path.join(self.root, "routercmd.json")))
        self.assertFalse(os.path.exists(os.path.join(self.root, "canary/bidcall.txt")))
        self.assertNotIn(SECRET, p.stdout + p.stderr)

    def test_fewer_than_two_healthy_stops(self):
        p = self.run_script("11 http://1.1.1.1:111\nsecret: " + SECRET + "\n")
        self.assertEqual(p.returncode, 1)
        self.assertIn("at least two healthy replicas, found 1", p.stderr)
        p = self.run_script("no healthy instances\n")
        self.assertEqual(p.returncode, 1)
        self.assertIn("found 0", p.stderr)
        self.assertFalse(os.path.exists(os.path.join(self.root, "routercmd.json")))
        self.assertNotIn(SECRET, p.stdout + p.stderr)

    def test_bad_arguments(self):
        env = dict(os.environ)
        for args in ([], ["a", "1", "2"], [self.state, "x1"]):
            p = subprocess.run(["sh", self.script, *args], capture_output=True, text=True, env=env, timeout=20)
            self.assertEqual(p.returncode, 2, args)

    def test_router_stopped_when_the_bid_call_fails(self):
        with open(os.path.join(self.root, "canary/canary.py"), "a") as f:
            f.write('if "bid" in a:\n    sys.exit(1)\n')
        p = self.run_script(self.URLS3)
        self.assertEqual(p.returncode, 1)
        self.assertIn("bid change failed", p.stderr)
        r = subprocess.run(["sh", "-c", "ps -eo args | grep '[r]outer.py --replicas http://1.1.1.1:111'"],
                           capture_output=True, text=True)
        self.assertEqual(r.stdout, "")


if __name__ == "__main__":
    unittest.main()
