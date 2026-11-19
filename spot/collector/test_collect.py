import contextlib
import gzip
import io
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

import collect
import stats


def dt(s):
    return datetime.strptime(s, collect.FMT).replace(tzinfo=timezone.utc)


def offer(i, price, machine=None, rel=0.99):
    return {"id": i, "machine_id": machine or i, "dph_total": price, "reliability2": rel}


def fake_ok(url, payload, timeout):
    return {"offers": [offer(1, 0.3), offer(2, 0.4)]}


def read_lines(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(x) for x in f]


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = self.tmp.name
        self.addCleanup(self.tmp.cleanup)
        for target, val in (("collect.time.sleep", None), ("collect.http_put_json", fake_ok)):
            p = mock.patch(target, val) if val else mock.patch(target)
            p.start()
            self.addCleanup(p.stop)
        self.sleep = collect.time.sleep

    def run_at(self, when, gpus=("RTX 3090",)):
        with mock.patch("collect.now", return_value=dt(when)):
            return collect.run_once(self.out, list(gpus))

    def path(self, day):
        return os.path.join(self.out, "vast-%s.jsonl.gz" % day)


class CollectTests(Base):
    def test_file_name_uses_utc_date_after_midnight(self):
        self.assertEqual(self.run_at("2026-03-01T23:55:00Z"), 0)
        self.run_at("2026-03-02T00:00:03Z")
        self.assertTrue(os.path.exists(self.path("2026-03-01")))
        self.assertTrue(os.path.exists(self.path("2026-03-02")))

    def test_two_runs_two_members(self):
        self.run_at("2026-03-01T10:00:00Z")
        self.run_at("2026-03-01T10:10:00Z")
        p = self.path("2026-03-01")
        with open(p, "rb") as f:
            self.assertEqual(f.read().count(b"\x1f\x8b\x08"), 2)
        lines = read_lines(p)
        self.assertEqual(len(lines), 4)  # 2 runs x 1 gpu x 2 kinds
        self.assertEqual({x["tick"] for x in lines}, {"2026-03-01T10:00:00Z", "2026-03-01T10:10:00Z"})

    def test_row_order_and_nulls(self):
        self.run_at("2026-03-01T10:00:00Z")
        line = read_lines(self.path("2026-03-01"))[0]
        self.assertEqual(line["cols"], collect.COLS)
        self.assertTrue(line["ok"])
        row = line["rows"][0]
        self.assertEqual(len(row), len(collect.COLS))
        self.assertEqual(row[0], 1)
        self.assertEqual(row[collect.COLS.index("dph_total")], 0.3)
        self.assertIsNone(row[collect.COLS.index("gpu_name")])
        self.assertEqual(line["n"], 2)
        self.assertFalse(line["truncated"])

    def test_failure_isolated(self):
        def flaky(url, payload, timeout):
            if payload["q"]["gpu_name"]["eq"] == "L40S":
                raise OSError("boom")
            return fake_ok(url, payload, timeout)
        with mock.patch("collect.http_put_json", flaky):
            code = self.run_at("2026-03-01T10:00:00Z", ["RTX 3090", "L40S"])
        self.assertEqual(code, 0)
        lines = read_lines(self.path("2026-03-01"))
        self.assertEqual(len(lines), 4)
        bad = [x for x in lines if not x["ok"]]
        self.assertEqual({x["gpu"] for x in bad}, {"L40S"})
        self.assertIn("boom", bad[0]["err"])
        self.assertNotIn("rows", bad[0])
        self.assertEqual(sum(x["ok"] for x in lines), 2)

    def test_run_deadline_skips_remaining_queries(self):
        calls = []

        def counting(url, payload, timeout):
            calls.append(1)
            return fake_ok(url, payload, timeout)

        with mock.patch("collect.RUN_DEADLINE", -1), mock.patch("collect.http_put_json", counting):
            self.assertEqual(self.run_at("2026-03-01T10:00:00Z"), 1)
        lines = read_lines(self.path("2026-03-01"))
        self.assertEqual(calls, [])
        self.assertEqual([x["ok"] for x in lines], [False, False])
        self.assertIn("run deadline", lines[0]["err"])
        self.assertFalse(os.path.exists(os.path.join(self.out, collect.LOCK)))

    def test_all_fail_exit_code_1(self):
        with mock.patch("collect.http_put_json", side_effect=OSError("down")):
            self.assertEqual(self.run_at("2026-03-01T10:00:00Z"), 1)

    def test_retries_then_success(self):
        calls = []

        def fail_twice(url, payload, timeout):
            calls.append(timeout)
            if len(calls) <= 2:
                raise OSError("temp")
            return fake_ok(url, payload, timeout)
        with mock.patch("collect.http_put_json", fail_twice):
            self.assertEqual(len(collect.fetch("RTX 3090", "bid")), 2)
        self.assertEqual(calls, [30, 30, 30])
        self.assertEqual([c.args[0] for c in self.sleep.call_args_list], [3, 8])

    def test_truncated_flag(self):
        full = {"offers": [offer(i, 0.1) for i in range(collect.LIMIT)]}
        with mock.patch("collect.http_put_json", return_value=full):
            self.run_at("2026-03-01T10:00:00Z")
        lines = read_lines(self.path("2026-03-01"))
        self.assertTrue(all(x["truncated"] and x["n"] == collect.LIMIT for x in lines))

    def test_fresh_lock_skips(self):
        os.makedirs(self.out, exist_ok=True)
        with open(os.path.join(self.out, collect.LOCK), "w") as f:
            f.write("2026-03-01T09:55:00Z")
        with mock.patch("collect.http_put_json") as h:
            self.assertEqual(self.run_at("2026-03-01T10:00:00Z"), 0)
            h.assert_not_called()
        self.assertFalse(os.path.exists(self.path("2026-03-01")))
        self.assertTrue(os.path.exists(os.path.join(self.out, collect.LOCK)))  # not ours: kept

    def test_stale_lock_replaced_and_removed(self):
        lock = os.path.join(self.out, collect.LOCK)
        with open(lock, "w") as f:
            f.write("2026-03-01T09:00:00Z")
        self.run_at("2026-03-01T10:00:00Z")
        self.assertTrue(os.path.exists(self.path("2026-03-01")))
        self.assertFalse(os.path.exists(lock))

    def test_lock_removed_after_exception(self):
        with mock.patch("collect.append_gz", side_effect=RuntimeError("disk")):
            with self.assertRaises(RuntimeError):
                self.run_at("2026-03-01T10:00:00Z")
        self.assertFalse(os.path.exists(os.path.join(self.out, collect.LOCK)))

    def test_log_line_and_rotation(self):
        self.run_at("2026-03-01T10:00:00Z")
        log = os.path.join(self.out, collect.LOG)
        text = open(log).read()
        self.assertIn("2026-03-01T10:00:00Z ok=2 failed=0 offers=4 truncated=0", text)
        with open(log, "w") as f:
            f.write("x" * (collect.LOG_MAX_BYTES + 10))
        self.run_at("2026-03-01T10:10:00Z")
        self.assertEqual(os.path.getsize(log + ".1"), collect.LOG_MAX_BYTES + 10)
        self.assertEqual(len(open(log).read().splitlines()), 1)


def rec(tick, gpu, kind, prices=None, ok=True, machines=None, rel=0.99):
    if not ok:
        return {"tick": tick, "ts": tick, "gpu": gpu, "kind": kind, "ok": False, "err": "x"}
    rows = [[None] * len(collect.COLS) for _ in prices]
    for i, (r, p) in enumerate(zip(rows, prices)):
        r[collect.COLS.index("dph_total")] = p
        r[collect.COLS.index("reliability2")] = rel
        r[collect.COLS.index("machine_id")] = machines[i] if machines else i
    return {"tick": tick, "ts": tick, "gpu": gpu, "kind": kind, "ok": True, "n": len(rows),
            "truncated": False, "cols": collect.COLS, "rows": rows}


def write_member(path, recs):
    with gzip.open(path, "ab") as f:
        f.write("".join(json.dumps(r) + "\n" for r in recs).encode())


class StatsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "vast-2026-03-01.jsonl.gz")

    def report(self):
        buf, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
            stats.main(["--data", self.tmp.name])
        return buf.getvalue(), err.getvalue()

    def synth(self):
        g = "RTX 3090"
        for t in ("T00:00:00Z", "T00:10:00Z", "T00:20:00Z", "T01:00:00Z"):
            tick = "2026-03-01" + t
            recs = [rec(tick, g, "on-demand", [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9], machines=[1, 1, 2, 3, 4, 5, 6])]
            recs.append(rec(tick, g, "bid", [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]))
            if t == "T00:10:00Z":
                recs[1] = rec(tick, g, "bid", ok=False)
            write_member(self.path, recs)

    def test_report(self):
        self.synth()
        out, _ = self.report()
        self.assertIn("ticks: 4", out)
        self.assertIn("gaps > 15 min: 1, total length 0:40:00", out)
        self.assertIn("failed queries: 1", out)
        # od mean(0.2..0.7)=0.45, bid mean=0.225 -> 50% discount; od machines 6 of 7 rows
        self.assertIn("od 0.450 | bid 0.225 | discount 50.0% | machines 6 / 6", out)
        self.assertLessEqual(len(out.splitlines()), 40)

    def test_na_with_fewer_than_six(self):
        write_member(self.path, [rec("2026-03-01T00:00:00Z", "L40S", "on-demand", [0.2, 0.3]),
                                 rec("2026-03-01T00:00:00Z", "L40S", "bid", [0.1] * 6, rel=0.5)])
        out, _ = self.report()
        self.assertIn("od n/a | bid n/a | discount n/a", out)

    def test_damaged_trailing_member(self):
        self.synth()
        extra = gzip.compress(b'{"tick": "2026-03-01T02:00:00Z"}\n' * 50)
        with open(self.path, "ab") as f:
            f.write(extra[:-12])  # truncated member
        out, err = self.report()
        self.assertIn("ticks: 4", out)
        self.assertIn("WARNING", err)


if __name__ == "__main__":
    unittest.main()
