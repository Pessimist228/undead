import asyncio
import contextlib
import io
import json
import os
import shutil
import tempfile
import threading
import time
import unittest

from aiohttp import web

import load

ALLOWED = {"model", "messages", "stream", "max_tokens", "temperature", "seed"}
SECRET_TEXT = "ZEBRA"


def chunk(content=None, finish=None, role=False):
    delta = {"role": "assistant"} if role else ({"content": content} if content is not None else {})
    return b"data: " + json.dumps({"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}).encode() + b"\n\n"


class Server:
    """A tiny OpenAI-style streaming server. Behaviour is picked by the seed of the request:
    1002 pauses 1.5 s in the middle, 1003 ends without finish_reason, 1004 answers HTTP 500."""

    def __init__(self):
        self.bodies = []
        self.port = None
        self._ready = threading.Event()

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        assert self._ready.wait(5)

    def _run(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        app = web.Application()
        app.router.add_post("/v1/chat/completions", self.handle)
        self.runner = web.AppRunner(app)
        self.loop.run_until_complete(self.runner.setup())
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        self.loop.run_until_complete(site.start())
        self.port = site._server.sockets[0].getsockname()[1]
        self._ready.set()
        self.loop.run_forever()

    def stop(self):
        async def down():
            await self.runner.cleanup()
        asyncio.run_coroutine_threadsafe(down(), self.loop).result(5)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(5)
        self.loop.close()

    async def handle(self, request):
        body = await request.json()
        self.bodies.append(body)
        seed = body["seed"]
        if seed == 1004:
            return web.Response(status=500, text="boom " + SECRET_TEXT)
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        await resp.write(chunk(role=True))
        for i in range(5):
            await resp.write(b": keepalive\n\n")
            await resp.write(chunk(SECRET_TEXT + "xy"))  # 7 chars
            if seed == 1002 and i == 2:
                await asyncio.sleep(1.5)
            await asyncio.sleep(0.01)
        if seed != 1003:
            await resp.write(chunk(finish="length"))
            await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp


class LoadTests(unittest.TestCase):
    def setUp(self):
        self.srv = Server()
        self.srv.start()
        self.tmp = tempfile.mkdtemp()
        self.out = os.path.join(self.tmp, "run.jsonl")

    def tearDown(self):
        self.srv.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_load(self, *extra):
        buf = io.StringIO()
        t = time.monotonic()
        with contextlib.redirect_stdout(buf):
            rc = load.main(["--url", "http://127.0.0.1:%d" % self.srv.port, "--model", "m", "--streams", "3",
                            "--duration", "30", "--max-tokens", "77", "--out", self.out, *extra])
        self.elapsed = time.monotonic() - t
        with open(self.out) as f:
            self.recs = [json.loads(x) for x in f]
        return rc, buf.getvalue()

    def test_lines_and_summary(self):
        rc, text = self.run_load("--max-requests", "6")
        self.assertEqual(rc, 0)
        self.assertLess(self.elapsed, 8)
        self.assertEqual(len(self.recs), 6)
        by = {r["request"]: r for r in self.recs}
        self.assertEqual(sorted(by), [1, 2, 3, 4, 5, 6])
        for r in self.recs:
            self.assertEqual(set(r), {"worker", "request", "start", "ttfc_ms", "chunks", "duration_ms", "max_gap_ms",
                                      "max_gap_end", "finish_reason", "status", "error", "chars"})
            self.assertIn(r["worker"], (1, 2, 3))
            self.assertRegex(r["start"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z$")
        ok = by[1]
        self.assertEqual((ok["chunks"], ok["chars"], ok["finish_reason"], ok["status"], ok["error"]),
                         (5, 35, "length", 200, None))
        self.assertGreaterEqual(ok["ttfc_ms"], 0)
        self.assertLess(ok["max_gap_ms"], 500)
        self.assertRegex(ok["max_gap_end"], r"Z$")
        pause = by[2]
        self.assertGreaterEqual(pause["max_gap_ms"], 1500)
        self.assertLess(pause["max_gap_ms"], 2000)
        self.assertEqual(pause["finish_reason"], "length")
        self.assertGreaterEqual(pause["duration_ms"], 1500)
        nofin = by[3]
        self.assertIsNone(nofin["finish_reason"])
        self.assertEqual((nofin["chunks"], nofin["status"], nofin["error"]), (5, 200, None))
        bad = by[4]
        self.assertEqual((bad["status"], bad["chunks"], bad["ttfc_ms"], bad["max_gap_ms"], bad["max_gap_end"]),
                         (500, 0, None, None, None))
        # summary
        lines = text.splitlines()
        self.assertEqual(lines[0], "requests finished: 4, failed: 2")
        self.assertIn("failures: HTTP 500 x1, no finish_reason x1", text)
        self.assertRegex(text, r"time to first chunk ms: p50 \d+\.\d, p95 \d+\.\d")
        self.assertRegex(text, r"chunks per second per stream \(chunks / total duration\): p50 \d+\.\d\d")
        gap_lines = lines[lines.index([l for l in lines if l.startswith("gaps longer")][0]):]
        self.assertEqual(gap_lines[0], "gaps longer than 1000 ms (UTC time, worker, gap ms): 1")
        self.assertEqual(len(gap_lines), 2)
        t, w, g = [x.strip() for x in gap_lines[1].split(",")]
        self.assertEqual((t, int(w)), (pause["max_gap_end"], pause["worker"]))
        self.assertAlmostEqual(float(g), pause["max_gap_ms"], delta=1)

    def test_request_body_has_only_allowed_fields_and_rising_seeds(self):
        self.run_load("--max-requests", "12")
        bodies = sorted(self.srv.bodies, key=lambda b: b["seed"])
        self.assertEqual(len(bodies), 12)
        for b in bodies:
            self.assertEqual(set(b), ALLOWED)
            self.assertEqual((b["model"], b["stream"], b["max_tokens"], b["temperature"]), ("m", True, 77, 0.7))
            self.assertEqual(b["messages"][0]["role"], "user")
        self.assertEqual([b["seed"] for b in bodies], list(range(1001, 1013)))
        self.assertGreaterEqual(len({b["messages"][0]["content"] for b in bodies}), 10)

    def test_file_has_no_prompt_or_answer_text(self):
        self.run_load("--max-requests", "4")
        with open(self.out) as f:
            raw = f.read()
        self.assertNotIn(SECRET_TEXT, raw)
        for p in load.PROMPTS:
            self.assertNotIn(p[:30], raw)
        self.assertGreaterEqual(len(load.PROMPTS), 10)

    def test_connection_failure_is_a_failed_request(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            load.main(["--url", "http://127.0.0.1:1", "--model", "m", "--streams", "1", "--duration", "5",
                       "--out", self.out, "--max-requests", "1"])
        with open(self.out) as f:
            rec = json.loads(f.readline())
        self.assertEqual((rec["status"], rec["finish_reason"]), (None, None))
        self.assertEqual(rec["error"], "ClientConnectorError")
        self.assertIn("requests finished: 0, failed: 1", buf.getvalue())

    def test_duration_stops_new_requests(self):
        t = time.monotonic()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            load.main(["--url", "http://127.0.0.1:%d" % self.srv.port, "--model", "m", "--streams", "1",
                       "--duration", "0.5", "--out", self.out])
        self.assertLess(time.monotonic() - t, 6)
        with open(self.out) as f:
            recs = [json.loads(x) for x in f]
        # request 1 is quick; request 2 (the 1.5 s pause) starts inside the 0.5 s and is allowed to finish after it
        self.assertEqual([r["request"] for r in recs], [1, 2])
        self.assertEqual(recs[1]["finish_reason"], "length")
        self.assertGreaterEqual(recs[1]["duration_ms"], 1500)


class Helpers(unittest.TestCase):
    def test_percentile_nearest_rank(self):
        self.assertEqual(load.percentile([40, 10, 30, 20], 50), 20)
        self.assertEqual(load.percentile([40, 10, 30, 20], 95), 40)
        self.assertEqual(load.percentile([7], 50), 7)
        self.assertIsNone(load.percentile([], 50))

    def test_summary_numbers(self):
        recs = [{"failed": False, "ttfc_ms": 100.0, "chunks": 10, "duration_ms": 2000.0, "status": 200, "error": None},
                {"failed": False, "ttfc_ms": 300.0, "chunks": 40, "duration_ms": 2000.0, "status": 200, "error": None},
                {"failed": True, "ttfc_ms": None, "chunks": 0, "duration_ms": 10.0, "status": 503, "error": None}]
        text = load.summary(recs, [("2026-01-01T00:00:02.000Z", 2, 1500.0), ("2026-01-01T00:00:01.000Z", 5, 1200.0)])
        self.assertIn("requests finished: 2, failed: 1", text)
        self.assertIn("p50 100.0, p95 300.0", text)
        self.assertIn("p50 5.00", text)
        self.assertTrue(text.endswith("2026-01-01T00:00:01.000Z, 5, 1200\n2026-01-01T00:00:02.000Z, 2, 1500"))


if __name__ == "__main__":
    unittest.main()
