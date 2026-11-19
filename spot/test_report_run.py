"""Unit tests for the numbers in report_run.py, on a tiny hand-made data set.  Run: python3 -m pytest -q"""
import datetime as dt
import json
import os
import tempfile
import unittest

import report_run as rr

T0 = dt.datetime(2026, 11, 19, 12, 0, 0, tzinfo=dt.timezone.utc).timestamp()


def iso(t):
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def write(d, name, rows):
    with open(os.path.join(d, name), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def req(w, i, start, dur_s, ttfc, chunks, finish, status=200, error=None, gap=250.0):
    return {"worker": w, "request": i, "start": iso(T0 + start), "ttfc_ms": ttfc, "chunks": chunks,
            "duration_ms": dur_s * 1000.0, "max_gap_ms": gap, "max_gap_end": iso(T0 + start + dur_s),
            "finish_reason": finish, "status": status, "error": error, "chars": chunks * 4}


def make_tiny(d):
    # --- fleet every 60 s for 10 minutes. card 1: healthy, down 300..420 (back at 480), card 2: loading until 120.
    fleet = []
    for k in range(11):
        t = T0 + 60 * k
        h1 = not (300 <= 60 * k < 480)
        h2 = 120 <= 60 * k < 600
        healthy = [x for x, h in ((1, h1), (2, h2)) if h]
        inst = {"1": {"status": "running" if h1 else "exited", "intended": "running", "healthy": h1}}
        if k < 10:
            inst["2"] = {"status": "running" if h2 else "loading", "intended": "running", "healthy": h2}
        fleet.append({"t": iso(t), "ts": t, "healthy": healthy, "registered": [], "instances": inst, "router": {},
                      "actions": [], "errors": []})
    write(d, "fleet.jsonl", fleet)
    write(d, "events.jsonl", [
        {"event": "created", "id": 1, "bid": 0.4, "t": iso(T0 - 60)},
        {"event": "created", "id": 2, "bid": 0.6, "t": iso(T0 - 60)},
        {"event": "destroyed", "id": 2, "reason": "teardown", "t": iso(T0 + 600)},
        {"event": "cap", "reason": "budget", "t": iso(T0 + 10)},
    ])
    with open(os.path.join(d, "state.json"), "w") as f:
        json.dump({"started": T0 - 60, "hours": 2, "budget": 5, "credit_start": 10.0, "last_credit": 9.5,
                   "instances": [
                       {"id": 1, "bid": 0.4, "min_bid": 0.36, "premium": 1.1, "created": T0 - 60, "destroyed": None},
                       {"id": 2, "bid": 0.6, "min_bid": 0.5, "premium": 1.3, "created": T0 - 60, "destroyed": T0 + 600}],
                   "rent_errors": [{"e": 1}]}, f)
    rows = [{"type": "sample", "t": iso(T0 + 60 * k), "ts": T0 + 60 * k, "id": cid, "status": "running", "hb": "ok",
             "intended_status": "running", "min_bid": 0.3, "dph_total": dph}
            for k in range(10) for cid, dph in ((1, 0.36), (2, 0.72))]
    for r in rows:
        if r["id"] == 2 and r["ts"] < T0 + 120:
            r["status"], r["hb"] = "loading", "none"
    write(d, "samples-2026-11-19.jsonl", rows)
    write(d, "samples-2026-11-20.jsonl", [{"type": "credit", "t": iso(T0 + 700), "ts": T0 + 700, "credit": 9.4}])
    write(d, "run.jsonl", [
        req(0, 1, 0, 50, 100.0, 100, "length"), req(0, 2, 60, 50, 200.0, 100, "stop"),
        req(1, 1, 120, 50, 300.0, 100, "length"), req(1, 2, 180, 50, 400.0, 100, "length"),
        req(2, 1, 240, 50, 500.0, 100, "length"), req(2, 2, 500, 100, 600.0, 100, "length"),
        req(3, 1, 300, 10, None, 0, None, status=503, error="HTTPStatusError", gap=0),
        req(3, 2, 320, 10, None, 0, None, status=503, error="HTTPStatusError", gap=0),
        req(4, 1, 400, 30, 250.0, 40, None, error="ReadError"),
        req(4, 2, 450, 5, None, 0, None, status=500, error="X"),
    ])
    write(d, "journal.jsonl", [
        {"id": "a", "start": iso(T0), "end": iso(T0 + 50), "resumes": [{"pos": 5, "pause_ms": 1400}, {"pos": 9, "pause_ms": 1600}],
         "finish": "length", "status": 200},
        {"id": "b", "start": iso(T0), "end": iso(T0 + 60), "resumes": [{"pos": 5, "pause_ms": 1200}], "finish": "stop", "status": 200},
        {"id": "c", "start": iso(T0), "end": iso(T0 + 70), "resumes": [], "finish": "stop", "status": 200},
    ])
    with open(os.path.join(d, "load-start.txt"), "w") as f:
        f.write("2026-11-19T12:00:00Z\n")


class Tiny(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        make_tiny(cls.tmp.name)
        cls.data = rr.load_dir(cls.tmp.name)
        cls.m, cls.ctx = rr.compute_metrics(cls.data, ondemand_price=1.0)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_window(self):
        self.assertEqual(self.m["load_start"], "2026-11-19T12:00:00Z")
        self.assertAlmostEqual(self.m["load_duration_s"], 600.0)

    def test_cards(self):
        m = self.m
        self.assertEqual(m["cards_rented"], 2)
        self.assertAlmostEqual(m["card_hours_healthy"], 900 / 3600)
        self.assertEqual(m["stops_total"], 1)
        self.assertAlmostEqual(m["stops_per_card_hour"], 4.0)
        self.assertEqual((m["stops_came_back"], m["stops_not_back"]), (1, 0))
        self.assertAlmostEqual(m["down_median_s"], 180)
        self.assertAlmostEqual(m["down_max_s"], 180)
        self.assertEqual((m["cards_served"], m["cards_never_served"], m["cards_not_tracked"]), (2, 0, 0))

    def test_premium_split(self):
        g = {x["premium"]: x for x in self.m["stops_by_premium"]}
        self.assertEqual(g[1.1]["stops"], 1)
        self.assertAlmostEqual(g[1.1]["stops_per_card_hour"], 1 / (420 / 3600))
        self.assertEqual(g[1.3]["stops"], 0)

    def test_rent_to_healthy(self):
        self.assertEqual(self.m["rent_to_healthy_s"], {"card 1": 60.0, "card 2": 180.0})
        self.assertAlmostEqual(self.m["rent_to_healthy_median_s"], 120.0)

    def test_healthy_count(self):
        self.assertEqual(self.m["healthy_min"], 1)
        self.assertAlmostEqual(self.m["healthy_median"], 1.5)

    def test_requests(self):
        m = self.m
        self.assertEqual((m["requests_total"], m["requests_finished"], m["requests_failed"]), (10, 6, 4))
        self.assertAlmostEqual(m["percent_finished"], 60.0)
        self.assertEqual((m["failed_rejected_503"], m["failed_stream_cut"], m["failed_other"]), (2, 1, 1))
        self.assertAlmostEqual(m["ttfc_p50_ms"], 300.0)
        self.assertAlmostEqual(m["ttfc_p95_ms"], 570.0)
        self.assertEqual(m["tokens_generated"], 640)

    def test_resumes(self):
        m = self.m
        self.assertEqual(m["resume_source"], "journal")
        self.assertEqual(m["streams_resumed"], 2)
        self.assertEqual(m["resume_events"], 3)
        self.assertAlmostEqual(m["resume_pause_p50_ms"], 1400.0)
        self.assertAlmostEqual(m["resume_pause_p95_ms"], 1580.0)
        self.assertAlmostEqual(m["resume_pause_max_ms"], 1600.0)

    def test_money(self):
        m = self.m
        self.assertAlmostEqual(m["money_used"], 0.5)
        self.assertAlmostEqual(m["money_estimate"], 0.36 * 600 / 3600 + 0.72 * 600 / 3600)
        self.assertAlmostEqual(m["dollars_per_million_tokens"], 0.5 / 640e-6)
        self.assertEqual(m["money_basis"], "credit")
        self.assertAlmostEqual(m["ondemand_cost_healthy_card_hours"], 0.25)
        self.assertEqual(m["cap_events"], 1)

    def test_text_and_json_roundtrip(self):
        txt = rr.summary_text(self.m)
        self.assertIn("60.00%", txt)
        self.assertIn("rejected with 503: 2", txt)
        json.dumps(self.m)
        # no card ids leak into the text
        self.assertNotIn("61449", txt)

    def test_captions(self):
        ev = rr.caption_events(self.ctx["cards"], self.ctx["stops"], self.ctx["t0"], self.ctx["t1"])
        self.assertEqual(len(ev), 2)
        self.assertIn("card 1 taken", ev[0][1])
        self.assertIn("back after 3 min", ev[1][1])


class Pieces(unittest.TestCase):
    def test_blip_is_bridged_but_two_bad_probes_are_a_stop(self):
        obs = [(0, True), (15, True), (30, False), (45, True), (60, True)]
        segs = rr.build_segments(obs, 0, None, 15, 30)
        self.assertEqual([s[2] for s in segs], ["healthy"])
        obs = [(0, True), (15, False), (30, False), (45, True), (60, True)]
        segs = rr.build_segments(obs, 0, None, 15, 30)
        self.assertEqual([s[2] for s in segs], ["healthy", "down", "healthy"])
        self.assertEqual(segs[1][:2], (15, 45))

    def test_loading_before_first_healthy_and_never_loaded(self):
        segs = rr.build_segments([(0, False), (15, False), (30, False)], -10, None, 15, 30)
        self.assertEqual([s[2] for s in segs], ["loading"])
        self.assertEqual(segs[0][:2], (-10, 45))

    def test_destroyed_state(self):
        segs = rr.build_segments([(0, True), (15, True), (30, False), (45, False)], 0, 20, 15, 30)
        self.assertEqual([s[2] for s in segs], ["healthy", "destroyed"])
        self.assertEqual(segs[0][:2], (0, 20))

    def test_load_start_formats(self):
        ref = T0 + 5
        self.assertEqual(rr.parse_load_start("2026-11-19T12:00:00Z\n", ref), T0)
        self.assertEqual(rr.parse_load_start("12:00:30", ref), T0 + 30)
        self.assertEqual(rr.parse_load_start("%d" % T0, ref), T0)
        self.assertIsNone(rr.parse_load_start("garbage", ref))

    def test_stop_we_asked_for_is_not_counted(self):
        with tempfile.TemporaryDirectory() as d:
            write(d, "fleet.jsonl", [{"t": iso(T0 + 15 * k), "ts": T0 + 15 * k, "healthy": [1] if k < 4 else [],
                                      "instances": {"1": {"status": "running", "healthy": k < 4}}} for k in range(10)])
            write(d, "events.jsonl", [{"event": "dropped", "id": 1, "t": iso(T0 + 70)}])
            write(d, "run.jsonl", [req(0, 1, 0, 140, 100.0, 10, "length")])
            m, _ = rr.compute_metrics(rr.load_dir(d))
            self.assertEqual(m["stops_total"], 0)

    def test_marketplace_intended_stopped_still_counts(self):
        """On Vast the marketplace itself sets intended_status "stopped" on a preempted card: not "caused by us"."""
        with tempfile.TemporaryDirectory() as d:
            write(d, "fleet.jsonl", [{"t": iso(T0 + 15 * k), "ts": T0 + 15 * k, "healthy": [1] if k < 4 else [],
                                      "instances": {"1": {"status": "running", "healthy": k < 4}}} for k in range(10)])
            write(d, "samples-2026-11-19.jsonl", [{"type": "sample", "t": iso(T0 + 60), "ts": T0 + 60, "id": 1, "status": "exited",
                                                   "hb": "none", "intended_status": "stopped", "dph_total": 0.3}])
            write(d, "run.jsonl", [req(0, 1, 0, 140, 100.0, 10, "length")])
            m, _ = rr.compute_metrics(rr.load_dir(d))
            self.assertEqual(m["stops_total"], 1)

    def test_card_without_data_is_untracked_and_rented_hours_count_it(self):
        with tempfile.TemporaryDirectory() as d:
            write(d, "events.jsonl", [{"event": "created", "id": 7, "bid": 0.3, "t": iso(T0)}])
            with open(os.path.join(d, "state.json"), "w") as f:
                json.dump({"credit_start": 5.0, "last_credit": 4.0, "finished_at": T0 + 3600, "instances": []}, f)
            write(d, "run.jsonl", [req(0, 1, 0, 100, 100.0, 10, "length")])
            data = rr.load_dir(d)
            cards, _ = rr.build_cards(data, rr.DEFAULTS)
            self.assertEqual(len(cards), 1)
            self.assertEqual(cards[0]["segs"], [])      # no data: nothing to draw but a hatched lane
            self.assertAlmostEqual(rr.rented_card_hours(cards, data["state"], T0 + 10), 1.0)

    def test_pause_shares(self):
        with tempfile.TemporaryDirectory() as d:
            write(d, "run.jsonl", [req(0, 1, 0, 100, 100.0, 10, "length")])
            write(d, "journal.jsonl", [{"start": iso(T0), "end": iso(T0 + 5), "resumes": [{"pause_ms": p}]} for p in (900, 1100, 1900, 47000)])
            m, _ = rr.compute_metrics(rr.load_dir(d))
            self.assertAlmostEqual(m["resume_pause_share_under_2s"], 0.75)
            self.assertEqual(m["resume_pauses_over_10s"], 1)
            self.assertAlmostEqual(m["resume_pause_p50_ms"], 1500.0)

    def test_only_run_file(self):
        with tempfile.TemporaryDirectory() as d:
            write(d, "run.jsonl", [req(0, 1, 0, 5, 100.0, 10, "length"), req(0, 2, 6, 1.5, 100.0, 10, "length", gap=2300.0)])
            m, ctx = rr.compute_metrics(rr.load_dir(d))
            self.assertEqual(m["requests_total"], 2)
            self.assertIsNone(m["cards_rented"])
            self.assertIsNone(m["money_used"])
            self.assertEqual(m["resume_source"], "client gaps")
            self.assertEqual(m["streams_resumed"], 1)
            self.assertIn("n/a", rr.summary_text(m))

    def test_empty_dir_runs_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            class A:
                pass
            a = A()
            a.dir, a.out, a.ondemand_price, a.min_down, a.no_video = d, None, None, 30.0, True
            rr.run(a, log=lambda *x: None)
            self.assertTrue(os.path.exists(os.path.join(d, "report", "summary.json")))


if __name__ == "__main__":
    unittest.main()
