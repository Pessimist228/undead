#!/usr/bin/env python3
"""Тесты роутера на фейковых репликах: убиваем, замораживаем, рвём клиента.

Запуск:  python tests/run_tests.py            (30 промптов)
         python tests/run_tests.py --n 100    (как в спеке)
GPU и vLLM не нужны. Скрипт сам поднимает фейковые реплики и роутеры.
"""
import argparse
import asyncio
import json
import os
import random
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime

import aiohttp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROUTER_PORT = 18080
REPLICA_PORTS = [19001, 19002]
ROUTER = f"http://127.0.0.1:{ROUTER_PORT}"
LOGDIR = os.path.join(ROOT, "tests", "logs")


class Proc:
    def __init__(self, name, cmd):
        self.name, self.cmd, self.popen, self.starts = name, cmd, None, 0

    def start(self):
        os.makedirs(LOGDIR, exist_ok=True)
        log = open(os.path.join(LOGDIR, f"{self.name}.log"), "ab")
        self.popen = subprocess.Popen(self.cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        self.starts += 1

    def alive(self):
        return self.popen is not None and self.popen.poll() is None

    def sig(self, s):
        if self.alive():
            os.kill(self.popen.pid, s)

    def stop(self):
        if self.alive():
            self.sig(signal.SIGCONT)
            self.popen.kill()
            self.popen.wait()


class Env:
    def __init__(self, args):
        self.args = args
        extra = [] if args.eos_id else ["--no-eos-id"]
        self.replicas = {
            f"http://127.0.0.1:{port}": Proc(
                f"replica-{port}",
                [sys.executable, "fake_replica.py", "--port", str(port),
                 "--name", str(port), "--tps", str(args.tps), *extra],
            )
            for port in REPLICA_PORTS
        }
        self.router = Proc(
            "router",
            [sys.executable, "router.py", "--tokenizer", "bytes", "--model", "fake-bytes",
             "--port", str(ROUTER_PORT), "--replicas", ",".join(self.replicas),
             "--deadline", "3", "--cooldown", "1", "--max-attempts", "3",
             "--first-token-timeout", "2"],
        )
        self.supervise = True
        self.session = None
        self.failures = []
        self._sup = None

    async def up(self):
        self.session = aiohttp.ClientSession()
        for p in self.replicas.values():
            p.start()
        self.router.start()
        for url in [*self.replicas, ROUTER]:
            await self.wait_health(url)
        self._sup = asyncio.create_task(self._supervisor())

    async def down(self):
        self._sup.cancel()
        await self.session.close()
        for p in [*self.replicas.values(), self.router]:
            p.stop()

    async def _supervisor(self):
        """Поднимает мёртвые реплики, как это сделал бы Kubernetes."""
        while True:
            if self.supervise:
                for p in self.replicas.values():
                    if not p.alive():
                        p.start()
            await asyncio.sleep(0.2)

    async def wait_health(self, url, timeout=15):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            try:
                async with self.session.get(url + "/health", timeout=aiohttp.ClientTimeout(total=1)) as r:
                    if r.status == 200:
                        return
            except Exception:  # noqa: BLE001
                pass
            await asyncio.sleep(0.1)
        raise RuntimeError(f"{url} не поднялся")

    async def all_up(self):
        for url in self.replicas:
            await self.wait_health(url)
        end = time.monotonic() + 10
        while time.monotonic() < end:
            async with self.session.get(ROUTER + "/metrics") as r:
                m = await r.json()
            if all(v == "up" for v in m["replicas"].values()):
                return
            await asyncio.sleep(0.1)

    async def info(self, aid):
        async with self.session.get(f"{ROUTER}/v1/answers/{aid}") as r:
            return await r.json()

    def check(self, ok, name, detail=""):
        if not ok:
            self.failures.append(f"{name}: {detail}")
        return ok


class Result:
    def __init__(self):
        self.answer_id = None
        self.text = ""
        self.positions = []  # позиции событий с текстом
        self.last_pos = 0
        self.finish = None
        self.error = None
        self.error_type = None
        self.done = False
        self.dropped = False


async def read_sse(resp, res, hook=None):
    pos = res.last_pos
    async for raw in resp.content:
        line = raw.decode("utf-8").rstrip("\r\n")
        if line.startswith("id:"):
            pos = int(line.rsplit(":", 1)[-1])
        elif line.startswith("data:"):
            payload = line[5:].strip()
            if payload == "[DONE]":
                res.done = True
                return
            obj = json.loads(payload)
            if "error" in obj:
                res.error = obj["error"]["message"]
                res.error_type = obj["error"].get("type")
                return
            ch = obj["choices"][0]
            delta = ch["delta"].get("content", "")
            if delta:
                res.text += delta
                res.positions.append(pos)
            res.last_pos = pos
            if ch["finish_reason"]:
                res.finish = ch["finish_reason"]
            if hook and await hook(res) == "drop":
                res.dropped = True
                return


async def ask(env, prompt, seed, hook=None, base=ROUTER, max_tokens=4000):
    res = Result()
    body = {"messages": [{"role": "user", "content": prompt}], "stream": True,
            "seed": seed, "max_tokens": max_tokens, "temperature": 0.7}
    async with env.session.post(base + "/v1/chat/completions", json=body) as r:
        res.answer_id = r.headers.get("X-Answer-Id")
        await read_sse(r, res, hook)
    return res


def verify(env, name, res, control, info, need_failover=True):
    ok = True
    ok &= env.check(res.error is None, name, f"ошибка: {res.error}")
    ok &= env.check(res.done and res.finish == "stop", name, f"нет финала, finish={res.finish}")
    ok &= env.check(res.text == control["text"], name, "текст не совпал с контрольным")
    ok &= env.check("�" not in res.text and "#" not in res.text, name, "битые символы")
    ok &= env.check(all(a < b for a, b in zip(res.positions, res.positions[1:])), name,
                    "повтор позиций у клиента")
    ok &= env.check(info["ids_sha256"] == control["sha"], name, "хеш ID токенов не совпал")
    if need_failover:
        ok &= env.check(info["attempts"] >= 2, name, "сбой не попал в середину ответа")
    return ok


# ---------- реестр реплик, админ-API, очередь, журнал ----------

RG_PORTS = [19011, 19012, 19013, 19014, 19015]
RG, RG_ADMIN = "http://127.0.0.1:18091", "http://127.0.0.1:18191"
RG_TOKEN = "adm-7f3a9c1e5b2d4086"
RG_SECRET = "SECRET-KEY-XYZ-4242"
RG_MARK = "ZEBRA-QUARTZ-7731"  # особая строка в каждом промпте: в журнале её быть не должно
RG_SEED = 5
JOURNAL_KEYS = {"id", "start", "end", "model", "prompt_tokens", "output_tokens", "ttft_ms",
                "duration_ms", "queued_ms", "replicas", "resumes", "finish", "status"}


def rg_url(port):
    return f"http://127.0.0.1:{port}"


def rg_data(prompt):
    """Ответ фейковой реплики: то же, что получит роутер, считаем здесь же."""
    sys.path.insert(0, ROOT)
    from fake_replica import split_sequence, text_for
    from toks import ByteTokenizer

    head, _ = split_sequence(ByteTokenizer().encode_chat([{"role": "user", "content": prompt}]))
    return text_for(head, RG_SEED)


def rg_expected(prompt, cut=None):
    data = rg_data(prompt)
    if cut is None:
        return data.decode("utf-8")
    return data[:cut].decode("utf-8", errors="replace").rstrip("�")


def rg_prompt(tag, lo, hi):
    """Промпт, ответ на который длится от lo до hi байт (токенов)."""
    for i in range(2000):
        p = f"{RG_MARK} {tag} {i}: расскажи про ноду"
        if lo <= len(rg_data(p)) <= hi:
            return p
    raise RuntimeError("нет промпта нужной длины")


async def registry_tests(env, report, args):
    tmp = tempfile.mkdtemp(prefix="undead-rg-")
    jpath, tpath = os.path.join(tmp, "journal.jsonl"), os.path.join(tmp, "token")
    with open(tpath, "w") as f:
        f.write(RG_TOKEN + "\n")
    reps = {p: Proc(f"rg-replica-{p}", [sys.executable, "fake_replica.py", "--port", str(p),
                                         "--name", str(p), "--tps", "200",
                                         *(["--api-key", "secret-k1"] if p == 19014 else []),
                                         *(["--http-error", "500"] if p == 19015 else [])])
            for p in RG_PORTS}
    router = Proc("rg-router", [
        sys.executable, "router.py", "--tokenizer", "bytes", "--model", "fake-bytes",
        "--port", "18091", "--replicas", f"{rg_url(19011)},{rg_url(19012)}",
        "--deadline", "3", "--cooldown", "1", "--max-attempts", "3",
        "--first-token-timeout", "2", "--queue-timeout", "2", "--journal", jpath,
        "--admin-port", "18191", "--admin-token-file", tpath])
    s = env.session

    def check(ok, name, detail=""):
        return env.check(bool(ok), name, detail)

    async def adm(method, path, token=RG_TOKEN, headers=None, **kw):
        h = dict(headers or {})
        if token is not None:
            h["Authorization"] = f"Bearer {token}"
        async with s.request(method, RG_ADMIN + path, headers=h, **kw) as r:
            text = await r.text()
            try:
                return r.status, json.loads(text), text, r.headers
            except ValueError:
                return r.status, None, text, r.headers

    async def listing():
        return (await adm("GET", "/admin/replicas"))[1]

    async def by_name(name):
        return next((r for r in await listing() if r["name"] == name), None)

    async def name_of(url):
        return next(r["name"] for r in await listing() if r["url"] == url)

    async def rinfo(aid):
        async with s.get(f"{RG}/v1/answers/{aid}") as r:
            return await r.json()

    async def settle():
        """Ждём, пока все ответы кончатся и кулдауны пройдут."""
        for _ in range(300):
            if all(r["in_flight"] == 0 and r["cooldown_seconds_left"] == 0
                   for r in await listing()):
                return
            await asyncio.sleep(0.1)
        check(False, "registry", f"реестр не успокоился: {await listing()}")

    async def reset(wanted):
        """Привести реестр к заданному: {имя: тело PUT}."""
        await settle()
        for r in await listing():
            if r["name"] not in wanted:
                await adm("DELETE", f"/admin/replicas/{r['name']}")
        for name, body in wanted.items():
            st, data, text, _ = await adm("PUT", f"/admin/replicas/{name}", json=body)
            check(st in (200, 201), "registry", f"PUT {name}: {st} {text}")
        await settle()

    def spec(port, **kw):
        return {"url": rg_url(port), "max_streams": 16, "state": "active", **kw}

    async def journal(aid=None, wait=6.0):
        """Строки журнала; с aid ждём строку этого запроса."""
        end = time.monotonic() + wait
        while True:
            with open(jpath, encoding="utf-8") as f:
                lines = [json.loads(x) for x in f.read().splitlines() if x]
            if aid is None:
                return lines
            hit = next((x for x in lines if x["id"] == aid), None)
            if hit or time.monotonic() > end:
                return hit
            await asyncio.sleep(0.05)

    def same(_env, name, res, prompt, cut=None):
        want = rg_expected(prompt, cut)
        ok = check(res.error is None and res.done, name, f"error={res.error}")
        ok &= check(res.text == want, name, "текст не совпал: потеря или повтор токенов")
        ok &= check(all(a < b for a, b in zip(res.positions, res.positions[1:])), name,
                         "повтор позиций у клиента")
        return ok

    def msg(prompt, **kw):
        return {"messages": [{"role": "user", "content": prompt}], "seed": RG_SEED,
                "temperature": 0.7, "max_tokens": 4000, **kw}

    long_a = rg_prompt("long-a", 850, 1300)
    long_b = rg_prompt("long-b", 850, 1300)
    mid_a = rg_prompt("mid-a", 450, 700)
    mid_b = rg_prompt("mid-b", 450, 700)
    short = rg_prompt("short", 300, 700)
    log_text = ""
    try:
        for p in reps.values():
            p.start()
        router.start()
        for url in [*(rg_url(p) for p in RG_PORTS), RG]:
            await env.wait_health(url)
        for _ in range(100):
            try:
                if (await adm("GET", "/admin/replicas"))[0] == 200:
                    break
            except aiohttp.ClientError:
                await asyncio.sleep(0.1)
        r0, r1 = (await listing())[:2]
        check([r0["name"], r1["name"]] == ["r0", "r1"] and r0["url"] == rg_url(19011)
                  and r0["max_streams"] == 0 and r1["max_streams"] == 0 and r0["state"] == "active",
                  "registry",
                  f"начальный список из --replicas: {await listing()}")

        # --- 0. без новых флагов лимита нет: 40 одновременных потоков на одной реплике
        await adm("DELETE", "/admin/replicas/r1")
        peak, stop = 0, []

        async def watch():
            nonlocal peak
            while not stop:
                row = await by_name("r0")
                peak = max(peak, row["in_flight"])
                await asyncio.sleep(0.02)

        mon = asyncio.create_task(watch())
        results = await asyncio.gather(*(ask(env, short, RG_SEED, base=RG, max_tokens=60)
                                         for _ in range(40)))
        stop.append(1)
        await mon
        good = check(all(r.error is None and r.finish == "length" for r in results), "nolimit",
                     f"не все 40 обслужены: {[r.error for r in results if r.error][:3]}")
        good &= check(all(same(env, "nolimit", r, short, cut=60) for r in results[:1]), "nolimit", "")
        good &= check(peak > 16, "nolimit", f"одновременно было не больше {peak} потоков")
        report["без флагов лимита нет"] = (f"40 потоков на одной реплике обслужены, пик {peak}"
                                           if good else "ПРОВАЛ")

        # --- 1. добавленная через админку реплика начинает получать запросы
        await reset({"r0": spec(19011)})
        st, data, _, _ = await adm("PUT", "/admin/replicas/r2", json={"url": rg_url(19013)})
        good = check(st == 201 and data["max_streams"] == 0 and data["state"] == "active",
                         "add", f"PUT: {st} {data}")
        used = []
        for i in range(4):
            res = await ask(env, short, RG_SEED, base=RG, max_tokens=60)
            good &= same(env, "add", res, short, cut=60)
            used += (await rinfo(res.answer_id))["replicas"]
        row = await by_name("r2")
        good &= check(rg_url(19013) in used and row["started"] >= 1, "add",
                          f"новая реплика не получила запросов: {used}, {row}")
        async with s.get(RG + "/metrics") as r:
            keys = list((await r.json())["replicas"])
        good &= check(rg_url(19011) in keys and "r2" in keys and rg_url(19013) not in keys,
                      "metrics", f"/metrics выдаёт адрес добавленной реплики: {keys}")
        report["реплика через админку"] = (
            f"из 4 запросов новой достались {used.count(rg_url(19013))}" if good else "ПРОВАЛ")

        # --- 2. draining: новых не получает, начатый ответ дочитывает
        await reset({"r0": spec(19011), "r1": spec(19012, state="draining")})
        idle_used = []
        for i in range(4):  # у простаивающей draining реплики нет причин проигрывать, кроме состояния
            r = await ask(env, short, RG_SEED, base=RG, max_tokens=30)
            idle_used += (await rinfo(r.answer_id))["replicas"]
        good = check(idle_used == [rg_url(19011)] * 4, "drain",
                     f"простаивающая draining реплика получила запросы: {idle_used}")
        await adm("PUT", "/admin/replicas/r1", json={"state": "active"})
        seen = {}

        async def drain_hook(res):
            if seen or res.last_pos < 20:
                return
            inf = await rinfo(res.answer_id)
            seen["name"] = name = await name_of(inf["replica"])
            seen["url"] = inf["replica"]
            st, data, _, _ = await adm("PUT", f"/admin/replicas/{name}", json={"state": "draining"})
            seen["put"] = (st, data["state"], data["max_streams"])
            seen["others"] = []
            for i in range(3):
                r = await ask(env, short, RG_SEED, base=RG, max_tokens=40)
                seen["others"] += (await rinfo(r.answer_id))["replicas"]
            seen["row"] = await by_name(name)

        res = await ask(env, long_a, RG_SEED, hook=drain_hook, base=RG)
        info = await rinfo(res.answer_id)
        good &= same(env, "drain", res, long_a)
        good &= check(seen["put"] == (200, "draining", 16), "drain", f"PUT: {seen['put']}")
        good &= check(seen["url"] not in seen["others"] and len(seen["others"]) == 3,
                          "drain", f"draining реплика получила новые запросы: {seen['others']}")
        good &= check(seen["row"]["in_flight"] == 1 and seen["row"]["state"] == "draining",
                          "drain", f"в GET: {seen['row']}")
        good &= check(info["replicas"] == [seen["url"]] and info["attempts"] == 1, "drain",
                          f"начатый ответ ушёл со своей реплики: {info['replicas']}")
        await settle()
        good &= check((await by_name(seen["name"]))["in_flight"] == 0, "drain",
                          "после ответа in_flight не 0")
        report["draining"] = ("новых не получила, начатый ответ дочитан на ней"
                              if good else "ПРОВАЛ")

        # --- 3. удаление реплики посреди ответа: ответ целый на другой реплике
        await reset({"r0": spec(19011), "r1": spec(19012)})
        seen = {}

        async def delete_hook(res):
            if seen or res.last_pos < 30:
                return
            inf = await rinfo(res.answer_id)
            seen["name"], seen["url"] = await name_of(inf["replica"]), inf["replica"]
            st, data, _, _ = await adm("DELETE", f"/admin/replicas/{seen['name']}")
            seen["del"] = (st, data)

        res = await ask(env, mid_a, RG_SEED, hook=delete_hook, base=RG)
        info = await rinfo(res.answer_id)
        line = await journal(res.answer_id)
        good = same(env, "delete", res, mid_a)
        good &= check(res.finish == "stop", "delete", f"finish={res.finish}")
        good &= check(seen["del"][0] == 200 and seen["del"][1]["removed"]
                          and seen["del"][1]["interrupted_streams"] == 1, "delete",
                          f"DELETE: {seen['del']}")
        good &= check(len(info["replicas"]) == 2 and info["replicas"][0] == seen["url"]
                          and info["replicas"][1] != seen["url"], "delete",
                          f"ответ не перешёл на другую реплику: {info['replicas']}")
        good &= check(seen["name"] not in [r["name"] for r in await listing()], "delete",
                          "удалённая реплика осталась в списке")
        other = (info["replica_names"] + [None, None])[1]
        good &= check(line and len(line["resumes"]) == 1
                          and line["resumes"][0]["from"] == seen["name"]
                          and line["resumes"][0]["to"] == other
                          and line["resumes"][0]["reason"] == "replica_removed"
                          and 0 < line["resumes"][0]["pos"] < line["output_tokens"],
                          "delete", f"журнал: {line and line['resumes']}")
        row = await by_name(other)
        good &= check(row and row["resumed_in"] == 1, "delete", f"resumed_in={row}")
        report["удаление посреди ответа"] = (
            f"ответ целый, {seen['name']} -> {other}, пауза "
            f"{line['resumes'][0]['pause_ms']} мс" if good else "ПРОВАЛ")

        # --- 4. max_streams: второй запрос ждёт и получает слот, когда первый кончил
        await reset({"r0": spec(19011, max_streams=1)})
        marks = {}

        async def timed(prompt, tag):
            async def hook(res):
                marks.setdefault(tag + "_first", time.monotonic())
            res = await ask(env, prompt, RG_SEED, hook=hook, base=RG, max_tokens=200)
            marks[tag + "_end"] = time.monotonic()
            return res

        task_a = asyncio.create_task(timed(mid_a, "a"))
        await asyncio.sleep(0.1)
        task_b = asyncio.create_task(timed(mid_b, "b"))
        await asyncio.sleep(0.15)
        row = await by_name("r0")
        good = check(row["in_flight"] == 1 and not task_b.done(), "maxstreams",
                         f"второй запрос не ждёт: {row}")
        res_a, res_b = await task_a, await task_b
        good &= same(env, "maxstreams-a", res_a, mid_a, cut=200)
        good &= same(env, "maxstreams-b", res_b, mid_b, cut=200)
        good &= check(marks["b_first"] >= marks["a_end"] - 0.15, "maxstreams",
                          "второй ответ пошёл раньше конца первого")
        lb = await journal(res_b.answer_id)
        good &= check(lb and 50 <= lb["queued_ms"] <= 2600 and lb["status"] == 200,
                          "maxstreams", f"queued_ms второго: {lb and lb['queued_ms']}")
        report["max_streams и очередь"] = (
            f"второй ждал {lb['queued_ms']} мс и получил слот" if good else "ПРОВАЛ")

        # --- 5. очередь не дождалась: 503 с Retry-After
        task_a = asyncio.create_task(ask(env, long_a, RG_SEED, base=RG))
        await wait_tokens_by_metrics(s, RG, 5)
        t = time.monotonic()
        async with s.post(RG + "/v1/chat/completions", json=msg(mid_b)) as r:
            st, hdr, data = r.status, r.headers, await r.json()
        took = time.monotonic() - t
        good = check(st == 503 and hdr.get("Retry-After") == "1", "queue503",
                         f"status={st} Retry-After={hdr.get('Retry-After')}")
        good &= check(set(data) == {"error"} and data["error"]["message"]
                          and data["error"]["type"], "queue503", f"тело: {data}")
        good &= check(1.9 <= took < 12, "queue503", f"503 через {took:.1f} с при таймауте 2 с")
        res_a = await task_a
        good &= same(env, "queue503-a", res_a, long_a)
        await settle()
        lines = await journal()
        l503 = next((x for x in lines if x["finish"] == "rejected_503"), None)
        good &= check(l503 and l503["status"] == 503 and l503["replicas"] == []
                          and l503["output_tokens"] == 0 and l503["ttft_ms"] is None
                          and l503["queued_ms"] >= 1900 and l503["prompt_tokens"] > 0,
                          "queue503", f"строка журнала: {l503}")
        report["таймаут очереди"] = f"503 с Retry-After через {took:.1f} с" if good else "ПРОВАЛ"

        # --- 6. продолженный ответ принимает реплика, у которой max_streams уже занят
        await reset({"r0": spec(19011, max_streams=1), "r1": spec(19012, max_streams=1)})
        task_a = asyncio.create_task(ask(env, long_a, RG_SEED, base=RG))
        await wait_tokens_by_metrics(s, RG, 5)
        task_b = asyncio.create_task(ask(env, long_b, RG_SEED, base=RG))
        for _ in range(200):
            act = await metrics_active(s, RG)
            if len(act) == 2 and all(x["tokens"] >= 20 for x in act):
                break
            await asyncio.sleep(0.02)
        # убиваем реплику любого из двух ответов: второй ответ идёт на занятой реплике
        port_x = int(act[0]["replica"].rsplit(":", 1)[1])
        port_y = int(act[1]["replica"].rsplit(":", 1)[1])
        good = check(port_x != port_y, "resume-full", f"оба ответа на одной реплике: {act}")
        reps[port_x].sig(signal.SIGKILL)
        peak = 0
        while not (task_a.done() and task_b.done()):
            row = await by_name(await name_of(rg_url(port_y)))
            peak = max(peak, row["in_flight"] if row else 0)
            await asyncio.sleep(0.02)
        res_a, res_b = await task_a, await task_b
        good &= same(env, "resume-full-a", res_a, long_a)
        good &= same(env, "resume-full-b", res_b, long_b)
        infos = [await rinfo(res_a.answer_id), await rinfo(res_b.answer_id)]
        moved = [i for i in infos if i["attempts"] == 2]
        good &= check(len(moved) == 1 and moved[0]["replicas"] == [rg_url(port_x), rg_url(port_y)],
                           "resume-full", f"ответ не перешёл на занятую реплику: {infos}")
        good &= check(peak == 2, "resume-full", f"in_flight на занятой реплике достигал {peak}, ждали 2")
        line = await journal(moved[0]["id"]) if moved else None
        good &= check(line and len(line["resumes"]) == 1 and line["resumes"][0]["pos"] > 0
                          and line["resumes"][0]["reason"] in ("connection_closed", "connection_error")
                          and line["resumes"][0]["from"] != line["resumes"][0]["to"]
                          and line["finish"] == "stop" and len(line["replicas"]) == 2, "resume-full", f"журнал: {line}")
        reps[port_x].start()
        await env.wait_health(rg_url(port_x))
        report["продолжение на занятой реплике"] = (
            f"принят при max_streams=1, одновременно {peak} потока" if good else "ПРОВАЛ")

        # --- 7. журнал: обычный запрос и обрыв клиента
        await reset({"r0": spec(19011), "r1": spec(19012)})
        res = await ask(env, mid_a, RG_SEED, base=RG)
        line = await journal(res.answer_id)
        sys.path.insert(0, ROOT)
        from toks import ByteTokenizer
        n_prompt = len(ByteTokenizer().encode_chat([{"role": "user", "content": mid_a}]))
        n_out = len(rg_data(mid_a))
        good = same(env, "journal-normal", res, mid_a)
        good &= check(line and set(line) == JOURNAL_KEYS, "journal-normal", f"ключи: {line}")
        good &= check(line["finish"] == "stop" and line["status"] == 200
                          and line["model"] == "fake-bytes" and line["resumes"] == []
                          and line["prompt_tokens"] == n_prompt
                          and n_out <= line["output_tokens"] <= n_out + 1
                          and len(line["replicas"]) == 1 and line["replicas"][0] in ("r0", "r1")
                          and 0 <= line["ttft_ms"] < line["duration_ms"]
                          and line["duration_ms"] >= n_out / 200 * 1000 * 0.8
                          and line["queued_ms"] < 2000,
                          "journal-normal", f"строка: {line}")
        t0 = datetime.fromisoformat(line["start"].replace("Z", "+00:00"))
        t1 = datetime.fromisoformat(line["end"].replace("Z", "+00:00"))
        good &= check(line["start"].endswith("Z") and 0 < (t1 - t0).total_seconds() * 1000
                          and abs((t1 - t0).total_seconds() * 1000 - line["duration_ms"]) < 5,
                          "journal-normal", f"время: {line['start']} {line['end']}")

        async def drop_hook(res):
            return "drop" if res.last_pos >= 30 else None

        res = await ask(env, mid_b, RG_SEED, hook=drop_hook, base=RG)
        good &= check(res.dropped, "journal-drop", "не успели оборвать клиента")
        line_d = await journal(res.answer_id)
        n_out_b = len(rg_data(mid_b))
        good &= check(line_d and line_d["finish"] == "client_disconnect"
                          and line_d["status"] == 200 and line_d["resumes"] == []
                          and n_out_b <= line_d["output_tokens"] <= n_out_b + 1,
                          "journal-drop", f"строка: {line_d}")
        report["журнал: обычный и обрыв клиента"] = (
            f"stop за {line['duration_ms']} мс, client_disconnect после {line_d['output_tokens']} токенов"
            if good else "ПРОВАЛ")

        # --- 8. админ-API: доступ
        await reset({"r0": spec(19011), "r1": spec(19012)})
        before = await listing()
        good = True
        for name, kw in {
            "без токена": {"token": None},
            "чужой токен": {"token": RG_TOKEN + "x"},
            "пустой токен": {"token": ""},
            "не Bearer": {"token": None, "headers": {"Authorization": f"Basic {RG_TOKEN}"}},
        }.items():
            for method, path, body in (("GET", "/admin/replicas", None),
                                       ("PUT", "/admin/replicas/zz", {"url": rg_url(19013)}),
                                       ("DELETE", "/admin/replicas/r1", None)):
                extra = {"json": body} if body else {}
                st, data, text, hdr = await adm(method, path, **kw, **extra)
                good &= check(st == 401 and data["error"]["type"] == "unauthorized"
                                  and hdr.get("WWW-Authenticate") == "Bearer", "admin-auth",
                                  f"«{name}» {method}: {st}")
        good &= check(await listing() == before, "admin-auth", "запрос без права изменил реестр")
        for name, headers in {
            "чужой Origin": {"Origin": "https://evil.example"},
            "Origin null": {"Origin": "null"},
            "чужой Host": {"Host": "evil.example:18191"},
        }.items():
            for token in (RG_TOKEN, None):
                st, data, _, _ = await adm("GET", "/admin/replicas", token=token, headers=headers)
                good &= check(st == 403 and data["error"]["type"] == "forbidden", "admin-guard",
                                  f"«{name}» token={bool(token)}: {st}")
        st, *_ = await adm("PUT", "/admin/replicas/zz", headers={"Origin": "https://evil.example"},
                           json={"url": rg_url(19013)})
        good &= check(st == 403 and await by_name("zz") is None, "admin-guard",
                          f"PUT с чужим Origin: {st}")
        for name, headers in {"свой Origin": {"Origin": "http://127.0.0.1:18191"},
                              "страница с localhost": {"Origin": "http://localhost:3000"},
                              "Host localhost": {"Host": "localhost:18191"}}.items():
            st, *_ = await adm("GET", "/admin/replicas", headers=headers)
            good &= check(st == 200, "admin-guard", f"«{name}» отклонён: {st}")
        async with s.get(RG + "/admin/replicas") as r:
            good &= check(r.status == 404, "admin-main", f"админка на основном порту: {r.status}")
        report["админка: токен и Origin"] = (
            "401 без токена и с чужим, 403 для веб-страниц, на основном порту админки нет"
            if good else "ПРОВАЛ")

        # --- 9. админ-API: проверка тел, частичные обновления, идемпотентность, ключи
        good = True
        await adm("PUT", "/admin/replicas/r9", json={"url": rg_url(19013), "state": "draining",
                                                     "api_key": RG_SECRET, "max_streams": 4})
        before = await listing()
        bad_bodies = {
            "не JSON": {"data": b"{oops"}, "массив": {"json": [1]}, "строка": {"json": "x"},
            "лишнее поле": {"json": {"max_stream": 3}},
            "url не строка": {"json": {"url": 5}}, "url пустой": {"json": {"url": ""}},
            "ftp": {"json": {"url": "ftp://127.0.0.1:21"}},
            "без схемы": {"json": {"url": "127.0.0.1:8000"}},
            "без хоста": {"json": {"url": "http://"}},
            "без хоста с портом": {"json": {"url": "http://:8000"}},
            "пароль в url": {"json": {"url": "http://user:pw@127.0.0.1:8000"}},
            "логин в url": {"json": {"url": "http://user@127.0.0.1:8000"}},
            "пробел в url": {"json": {"url": "http://127.0.0.1:8000/a b"}},
            "query в url": {"json": {"url": "http://127.0.0.1:8000/?a=1"}},
            "плохой порт": {"json": {"url": "http://127.0.0.1:99999"}},
            "max_streams -1": {"json": {"max_streams": -1}},
            "max_streams 1025": {"json": {"max_streams": 1025}},
            "max_streams строка": {"json": {"max_streams": "5"}},
            "max_streams дробное": {"json": {"max_streams": 2.5}},
            "max_streams 16.0": {"json": {"max_streams": 16.0}},
            "max_streams bool": {"json": {"max_streams": True}},
            "state чужой": {"json": {"state": "paused"}},
            "state None": {"json": {"state": None}},
            "api_key пустой": {"json": {"api_key": ""}},
            "api_key число": {"json": {"api_key": 123}},
            "api_key с пробелом": {"json": {"api_key": "a b"}},
            "половина верная": {"json": {"max_streams": 7, "state": "bogus"}},
        }
        for name, kw in bad_bodies.items():
            st, data, _, _ = await adm("PUT", "/admin/replicas/r9", **kw)
            good &= check(st == 400 and data and data["error"]["message"], "admin-bad",
                              f"«{name}» существующая: {st}")
        for name, kw in {"новая без url": {"json": {"max_streams": 3}},
                         "новая пустое тело": {"json": {}},
                         "новая с плохим url": {"json": {"url": "ftp://h"}}}.items():
            st, data, _, _ = await adm("PUT", "/admin/replicas/brand-new", **kw)
            good &= check(st == 400 and data, "admin-bad", f"«{name}»: {st}")
        for name in ("R0", "-a", "a_b", "a.b", "x" * 64, "a%20b", "%D0%B0", "a%0A"):
            st, data, _, _ = await adm("PUT", f"/admin/replicas/{name}", json={"url": rg_url(19013)})
            good &= check(st == 400 and data, "admin-name", f"имя {name!r}: {st}")
            st, data, _, _ = await adm("DELETE", f"/admin/replicas/{name}")
            good &= check(st == 400, "admin-name", f"DELETE имя {name!r}: {st}")
        for name in ("a", "0", "x" * 63, "a-b-"):
            st, *_ = await adm("PUT", f"/admin/replicas/{name}", json={"url": rg_url(19013)})
            good &= check(st == 201, "admin-name", f"имя {name!r} должно быть можно: {st}")
            await adm("DELETE", f"/admin/replicas/{name}")
        good &= check(await listing() == before, "admin-bad",
                          "плохой запрос изменил реестр или создал реплику")

        st, data, text, _ = await adm("GET", "/admin/replicas")
        row = next(r for r in data if r["name"] == "r9")
        good &= check(RG_SECRET not in text and "api_key" not in text.replace("api_key_set", "")
                          and row["api_key_set"] is True, "admin-key", "GET показал ключ")
        good &= check({"name", "url", "state", "max_streams", "in_flight",
                           "cooldown_seconds_left", "started", "resumed_in",
                           "last_error"} <= set(row), "admin-get", f"поля: {sorted(row)}")
        # частичное обновление и идемпотентность
        body = {"url": rg_url(19013) + "/", "max_streams": 3, "state": "active"}
        st1, d1, t1, _ = await adm("PUT", "/admin/replicas/r9", json=body)
        a1 = await listing()
        st2, d2, t2, _ = await adm("PUT", "/admin/replicas/r9", json=body)
        a2 = await listing()
        good &= check(st1 == st2 == 200 and a1 == a2 and d1 == d2 and RG_SECRET not in t1 + t2,
                          "admin-idem", f"повторный PUT: {st1} {st2}")
        now_row = next(r for r in a2 if r["name"] == "r9")
        good &= check(now_row["url"] == rg_url(19013) and now_row["max_streams"] == 3
                          and now_row["state"] == "active" and now_row["api_key_set"],
                          "admin-partial", f"{now_row}")
        st, d, _, _ = await adm("PUT", "/admin/replicas/r9", json={"state": "draining"})
        good &= check(st == 200 and d["state"] == "draining" and d["max_streams"] == 3
                          and d["url"] == rg_url(19013) and d["api_key_set"], "admin-partial", f"{d}")
        st, d, _, _ = await adm("PUT", "/admin/replicas/r9", json={"api_key": None})
        good &= check(st == 200 and d["api_key_set"] is False and d["state"] == "draining",
                          "admin-partial", f"api_key null: {d}")
        st1, d1, _, _ = await adm("DELETE", "/admin/replicas/r9")
        st2, d2, _, _ = await adm("DELETE", "/admin/replicas/r9")
        good &= check(st1 == st2 == 200 and d1["removed"] is True and d2["removed"] is False
                          and await by_name("r9") is None, "admin-idem", f"DELETE: {st1} {st2} {d1} {d2}")
        report["админка: тела, идемпотентность"] = (
            f"{len(bad_bodies) + 3} плохих тел и 8 плохих имён дали 400, повторы ничего не меняют"
            if good else "ПРОВАЛ")

        # --- 10. ключ реплики: свой на реплику, не виден в GET
        await reset({"k1": {"url": rg_url(19014), "api_key": "secret-k1"}})
        good = True
        res = await ask(env, short, RG_SEED, base=RG, max_tokens=60)
        good &= same(env, "replica-key", res, short, cut=60)
        t = time.monotonic()
        await adm("PUT", "/admin/replicas/k1", json={"api_key": "wrong-key"})
        res = await ask(env, short, RG_SEED, base=RG, max_tokens=60)
        good &= check(res.error is not None and "API key" in res.error
                          and time.monotonic() - t < 2, "replica-key",
                          f"с неверным ключом: {res.error!r}")
        row = await by_name("k1")
        good &= check(row["last_error"] and "HTTP 401" in row["last_error"]
                          and "wrong-key" not in json.dumps(row), "replica-key",
                          f"last_error: {row['last_error']!r}")
        await adm("PUT", "/admin/replicas/k1", json={"api_key": "secret-k1"})
        await settle()
        res = await ask(env, short, RG_SEED, base=RG, max_tokens=60)
        good &= same(env, "replica-key", res, short, cut=60)
        report["ключ на реплику"] = "свой ключ работает, неверный даёт ошибку, в GET его нет" if good else "ПРОВАЛ"

        # --- 10a. клиент ушёл из очереди: билет снимается, ответ для него не генерируется
        await reset({"r0": spec(19011, max_streams=1)})
        started0 = (await by_name("r0"))["started"]
        async with s.get(RG + "/metrics") as r:
            answers0 = (await r.json())["answers"]
        # A короче таймаута очереди: призрачный B успел бы получить слот, когда A кончит
        task_a = asyncio.create_task(ask(env, mid_a, RG_SEED, base=RG, max_tokens=200))
        await wait_tokens_by_metrics(s, RG, 5)
        try:  # клиент B закрывает соединение, пока стоит в очереди
            await asyncio.wait_for(s.post(RG + "/v1/chat/completions", json=msg(mid_b)), 0.3)
        except asyncio.TimeoutError:
            pass
        res_a = await task_a
        await asyncio.sleep(0.6)
        row = await by_name("r0")
        async with s.get(RG + "/metrics") as r:
            answers1 = (await r.json())["answers"]
        good = same(env, "queue-gone", res_a, mid_a, cut=200)
        good &= check(row["in_flight"] == 0 and row["started"] == started0 + 1 and
                      answers1 == answers0 + 1, "queue-gone",
                      f"для ушедшего клиента запущена генерация: {row}, ответов {answers0}->{answers1}")
        gone = [x for x in await journal() if x["replicas"] == [] and x["finish"] == "client_disconnect"]
        good &= check(len(gone) == 1 and gone[0]["output_tokens"] == 0 and gone[0]["queued_ms"] >= 100,
                      "queue-gone", f"журнал: {gone}")
        report["клиент ушёл из очереди"] = "второй генерации нет, in_flight 0, строка client_disconnect" if good else "ПРОВАЛ"

        # --- 10b. запрос, упавший до первого токена, не обходит лимит и draining
        await reset({"r0": spec(19011, max_streams=1), "r1": spec(19015)})
        task_a = asyncio.create_task(ask(env, long_a, RG_SEED, base=RG))
        await wait_tokens_by_metrics(s, RG, 5)
        peak, stop = 0, []

        async def watch0():
            nonlocal peak
            while not stop:
                row = await by_name("r0")
                peak = max(peak, row["in_flight"])
                await asyncio.sleep(0.02)

        mon = asyncio.create_task(watch0())
        t = time.monotonic()
        async with s.post(RG + "/v1/chat/completions", json=msg(mid_b)) as r:
            st, hdr = r.status, r.headers
            await r.read()
        took = time.monotonic() - t
        res_a = await task_a
        stop.append(1)
        await mon
        good = check(st == 503 and hdr.get("Retry-After") == "1" and took >= 1.9, "pretoken-limit",
                     f"status={st} через {took:.1f} с")
        good &= check(peak <= 1, "pretoken-limit", f"r0 доходила до {peak} потоков при max_streams 1")
        good &= same(env, "pretoken-limit-a", res_a, long_a)
        good &= check((await by_name("r1"))["started"] == 0 and (await by_name("r1"))["last_error"]
                      == "HTTP 500", "pretoken-limit", f"r1: {await by_name('r1')}")
        await reset({"r0": spec(19015), "r1": spec(19012, state="draining")})
        async with s.post(RG + "/v1/chat/completions", json=msg(mid_b, stream=True)) as r:
            st = r.status
            await r.read()
        row = await by_name("r1")
        good &= check(st == 503 and row["started"] == 0 and row["in_flight"] == 0, "pretoken-drain",
                      f"draining реплика получила запрос без токена: {st} {row}")
        report["до первого токена: лимит и draining"] = (
            f"503 вместо обхода лимита, r0 не выше {peak}; draining не тронута" if good else "ПРОВАЛ")

        # --- 10c. отказ по ключу считается только для реплик, которые есть и не менялись
        import io
        import contextlib
        sys.path.insert(0, ROOT)
        import router as R

        class Req:
            def __init__(self, name, body=None):
                self.match_info, self._b = {"name": name}, body

            async def json(self):
                return self._b

        rt_ = R.Router(R.parse_args(["--replicas", "http://h1:1,http://h2:2",
                                      "--tokenizer", "bytes", "--model", "m"]))
        ans = R.Answer(id="x", prompt_ids=[], max_tokens=1, temperature=1, top_p=1, seed=1)

        def refuse_all():
            for n, rp in rt_.registry.items():
                ans.auth_refused[n] = rp.epoch

        refuse_all()
        good = check(rt_._refused(ans) == 2, "auth-epoch", "два отказа не посчитаны")
        await rt_.admin_put(Req("r0", {"api_key": "new-key"}))
        good &= check(rt_._refused(ans) == 1, "auth-epoch", "смена ключа не сняла отказ")
        refuse_all()
        await rt_.admin_put(Req("r0", {"api_key": "new-key"}))  # тот же ключ: ничего не меняется
        good &= check(rt_._refused(ans) == 2, "auth-epoch", "повторный PUT снял отказ")
        await rt_.admin_put(Req("r1", {"url": "http://h3:3"}))
        good &= check(rt_._refused(ans) == 1, "auth-epoch", "смена url не сняла отказ")
        refuse_all()
        await rt_.admin_delete(Req("r1"))
        good &= check(rt_._refused(ans) == 1, "auth-epoch", "удалённая реплика считается")
        await rt_.admin_put(Req("r1", {"url": "http://h2:2"}))  # то же имя, новая реплика
        good &= check(rt_._refused(ans) == 1 and len(rt_.registry) == 2, "auth-epoch",
                      "новая реплика с тем же именем унаследовала отказ")
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            R.parse_args(["--replicas", rg_url(19011), "--admin-token-file", tpath])
            quiet = buf.getvalue()
            R.parse_args(["--replicas", rg_url(19011), "--admin-token-file", tpath,
                          "--admin-port", "18199"])
        good &= check("ignored without --admin-port" in quiet and RG_TOKEN not in buf.getvalue()
                      and buf.getvalue().count("ignored") == 1, "admin-warn",
                      f"предупреждение: {buf.getvalue()!r}")
        report["отказ по ключу и предупреждение"] = "считаются только актуальные реплики; предупреждение одно" if good else "ПРОВАЛ"

        # --- 10d. короткие потоки не помечаются как обрыв клиента; поздний GET журнал не меняет
        await reset({"r0": spec(19011), "r1": spec(19012)})
        finishes, last = [], None
        for i in range(30):
            res = await ask(env, short, RG_SEED, base=RG, max_tokens=2)
            line = await journal(res.answer_id)
            finishes.append(line and line["finish"])
            last = (res.answer_id, line)
        good = check(finishes == ["length"] * 30, "journal-short", f"finish: {set(finishes)}")
        before_lines = await journal()
        late = Result()
        async with s.get(f"{RG}/v1/streams/{last[0]}?from=0") as r:
            await read_sse(r, late)
        await asyncio.sleep(0.2)
        after_lines = await journal()
        good &= check(late.done and after_lines == before_lines
                      and sum(x["id"] == last[0] for x in after_lines) == 1, "journal-late",
                      "поздний GET /v1/streams изменил журнал")
        report["журнал: короткие потоки, поздний GET"] = "30 коротких без ложного client_disconnect, поздний GET ничего не менял" if good else "ПРОВАЛ"

        # --- 11. журнал целиком: набор полей, нет промптов, текста и ключей
        await reset({"r0": spec(19011), "r1": spec(19012)})
        await asyncio.sleep(0.3)
        with open(jpath, encoding="utf-8") as f:
            raw = f.read()
        lines = [json.loads(x) for x in raw.splitlines()]
        good = check(len(lines) >= 10 and all(set(x) == JOURNAL_KEYS for x in lines), "journal-all",
                         f"{len(lines)} строк, лишние или пропавшие поля")
        good &= check(RG_MARK not in raw and "расскажи про ноду" not in raw
                          and "<|user|>" not in raw and "Вопрос" not in raw, "journal-all",
                          "в журнале есть текст промпта")
        from fake_replica import SENTENCES
        good &= check(raw.isascii() and not any(sn[:12] in raw for sn in SENTENCES)
                          and "token" not in raw.replace("_tokens", ""),
                          "journal-all", "в журнале есть текст ответа или ID токенов")
        good &= check(all(isinstance(v, (int, str, type(None), list)) for x in lines
                              for v in x.values())
                          and all(len(x["resumes"]) <= 3 and len(x["replicas"]) <= 4 for x in lines),
                          "journal-all", "в журнале подозрительные значения")
        kinds = sorted({x["finish"] for x in lines})
        good &= check({"stop", "length", "client_disconnect", "rejected_503", "error"} >= set(kinds)
                          and {"stop", "length", "client_disconnect", "rejected_503"} <= set(kinds),
                          "journal-all", f"виды finish: {kinds}")
        report["журнал: приватность"] = (
            f"{len(lines)} строк, finish: {', '.join(kinds)}; промпта и текста ответа нет"
            if good else "ПРОВАЛ")
    finally:
        router.stop()
        for p in reps.values():
            p.stop()
        for name in ("rg-router", *(f"rg-replica-{p}" for p in RG_PORTS)):
            try:
                with open(os.path.join(LOGDIR, f"{name}.log"), encoding="utf-8", errors="replace") as f:
                    log_text += f.read()
            except OSError:
                pass

    # ключи и токен не попали ни в логи процессов, ни в журнал
    with open(jpath, encoding="utf-8") as f:
        raw = f.read()
    leaked = [w for w in (RG_TOKEN, RG_SECRET, "secret-k1", "wrong-key")
              if w in raw or w in log_text]
    check(not leaked, "secrets", f"утекло в логи или журнал: {leaked}")
    report["секреты"] = "токен админки и ключи не попали в журнал и логи" if not leaked else "ПРОВАЛ"

    # --- 12. отказ запуска без пригодного файла токена
    empty, blank = os.path.join(tmp, "empty"), os.path.join(tmp, "blank")
    open(empty, "w").close()
    with open(blank, "w") as f:
        f.write(" \n\t\n")
    base = [sys.executable, "router.py", "--tokenizer", "bytes", "--model", "fake-bytes",
            "--port", "18092", "--replicas", rg_url(19011), "--admin-port", "18199"]
    cases = {
        "без --admin-token-file": base,
        "файла нет": [*base, "--admin-token-file", os.path.join(tmp, "nope")],
        "пустой файл": [*base, "--admin-token-file", empty],
        "только пробелы": [*base, "--admin-token-file", blank],
        "каталог вместо файла": [*base, "--admin-token-file", tmp],
        "пароль в --replicas": [*base[:-2], "--replicas", "http://u:p@127.0.0.1:19011"],
        "нет --replicas и нет админки": [sys.executable, "router.py", "--tokenizer", "bytes",
                                         "--model", "fake-bytes", "--port", "18092"],
        "пустой --replicas и нет админки": [sys.executable, "router.py", "--tokenizer", "bytes",
                                            "--model", "fake-bytes", "--port", "18092",
                                            "--replicas", ""],
    }
    good = True
    for name, cmd in cases.items():
        try:
            out = await asyncio.to_thread(
                subprocess.run, cmd, cwd=ROOT, capture_output=True, timeout=15)
            good &= check(out.returncode != 0 and b"Traceback" not in out.stderr, "admin-start",
                              f"«{name}»: код {out.returncode}, {out.stderr[-200:]!r}")
        except subprocess.TimeoutExpired:
            good &= check(False, "admin-start", f"«{name}»: роутер запустился")
    try:
        await asyncio.wait_for(asyncio.open_connection("127.0.0.1", 18199), 1)
        good &= check(False, "admin-start", "порт админки слушается")
    except (OSError, asyncio.TimeoutError):
        pass
    # --- без --replicas, но с админкой: пустой реестр, запрос ждёт и получает 503, реплика через PUT
    ok_tok = os.path.join(tmp, "tok")
    with open(ok_tok, "w") as f:
        f.write(RG_TOKEN)
    empty_router = Proc("rg-empty-router", [
        sys.executable, "router.py", "--tokenizer", "bytes", "--model", "fake-bytes",
        "--port", "18093", "--queue-timeout", "1", "--admin-port", "18193",
        "--admin-token-file", ok_tok])
    rep19011 = Proc("rg-replica-19011b", [sys.executable, "fake_replica.py", "--port", "19011",
                                          "--name", "b", "--tps", "200"])
    rep19011.start()
    empty_router.start()
    try:
        base_e, adm_e = "http://127.0.0.1:18093", "http://127.0.0.1:18193"
        await env.wait_health(base_e)
        await env.wait_health(rg_url(19011))
        hdr = {"Authorization": f"Bearer {RG_TOKEN}"}
        async with s.get(adm_e + "/admin/replicas", headers=hdr) as r:
            listed = await r.json()
        t = time.monotonic()
        async with s.post(base_e + "/v1/chat/completions", json=msg(short, max_tokens=20)) as r:
            st, ra = r.status, r.headers.get("Retry-After")
        good &= check(listed == [] and st == 503 and ra == "1" and time.monotonic() - t >= 0.9,
                      "empty-registry", f"список {listed}, статус {st}")
        async with s.put(adm_e + "/admin/replicas/a", headers=hdr,
                         json={"url": rg_url(19011), "max_streams": 16}) as r:
            good &= check(r.status == 201, "empty-registry", f"PUT: {r.status}")
        res = await ask(env, short, RG_SEED, base=base_e, max_tokens=60)
        good &= same(env, "empty-registry", res, short, cut=60)
    finally:
        empty_router.stop()
        rep19011.stop()
    report["отказ запуска без токена"] = f"{len(cases)} случаев, роутер не стартует" if good else "ПРОВАЛ"
    shutil.rmtree(tmp, ignore_errors=True)


async def metrics_active(session, base):
    async with session.get(base + "/metrics") as r:
        return (await r.json())["active"]


async def wait_tokens_by_metrics(session, base, n):
    for _ in range(400):
        act = await metrics_active(session, base)
        if act and max(x["tokens"] for x in act) >= n:
            return
        await asyncio.sleep(0.01)
    raise RuntimeError("ответ не пошёл")


async def main(args):
    random.seed(args.seed)
    env = Env(args)
    await env.up()
    report = {}
    try:
        prompts = [f"Вопрос №{i}: что будет, если нода {i} умрёт посреди ответа?"
                   for i in range(args.n)]

        # 1. Контрольные прогоны без сбоев, по 10 одновременно
        t = time.monotonic()
        control = {}
        sem = asyncio.Semaphore(10)

        async def one_control(i):
            async with sem:
                res = await ask(env, prompts[i], i)
                info = await env.info(res.answer_id)
                env.check(res.done and res.error is None and info["attempts"] == 1,
                          f"control[{i}]", f"error={res.error} attempts={info['attempts']}")
                control[i] = {"text": res.text, "sha": info["ids_sha256"], "tokens": info["tokens"]}

        await asyncio.gather(*(one_control(i) for i in range(args.n)))
        report["контроль"] = f"{len(control)} ответов за {time.monotonic() - t:.1f} с"

        # 2. kill -9 посреди ответа
        ok_count, pauses = 0, []
        for i in range(args.n):
            await env.all_up()
            kill_at = random.randint(5, min(150, control[i]["tokens"] - 30))
            killed = []

            async def hook(res, kill_at=kill_at, killed=killed):
                if not killed and res.last_pos >= kill_at:
                    inf = await env.info(res.answer_id)
                    env.replicas[inf["replica"]].sig(signal.SIGKILL)
                    killed.append(inf["replica"])

            res = await ask(env, prompts[i], i, hook)
            info = await env.info(res.answer_id)
            if verify(env, f"kill9[{i}]", res, control[i], info):
                ok_count += 1
                pauses += info["pauses"]
        report["kill -9"] = f"{ok_count}/{args.n} ответов совпали с контрольными"
        if pauses:
            report["пауза после kill -9"] = (
                f"медиана {statistics.median(pauses) * 1000:.0f} мс, "
                f"максимум {max(pauses) * 1000:.0f} мс")

        # 3. Тихая смерть: SIGSTOP, потом SIGCONT
        ok_count, pauses = 0, []
        for i in range(min(5, args.n)):
            await env.all_up()
            frozen = []

            async def hook(res, frozen=frozen):
                if not frozen and res.last_pos >= 20:
                    inf = await env.info(res.answer_id)
                    env.replicas[inf["replica"]].sig(signal.SIGSTOP)
                    frozen.append(inf["replica"])

            res = await ask(env, prompts[i], i, hook)
            info = await env.info(res.answer_id)
            good = verify(env, f"sigstop[{i}]", res, control[i], info)
            env.replicas[frozen[0]].sig(signal.SIGCONT)
            await asyncio.sleep(0.7)
            late = await env.info(res.answer_id)
            good &= env.check(late["ids_sha256"] == control[i]["sha"], f"sigstop[{i}]",
                              "поздние токены старой попытки попали в лог")
            ok_count += good
            pauses += info["pauses"]
        report["SIGSTOP"] = f"{ok_count}/{min(5, args.n)} ответов совпали, поздние токены выброшены"
        if pauses:
            report["пауза после SIGSTOP"] = (
                f"медиана {statistics.median(pauses):.2f} с, максимум {max(pauses):.2f} с")

        # 4. Клиент отвалился и вернулся
        ok_count = 0
        m = min(10, args.n)
        for i in range(m):
            await env.all_up()

            async def hook(res):
                return "drop" if res.last_pos >= 30 else None

            res = await ask(env, prompts[i], i, hook)
            env.check(res.dropped, f"resume[{i}]", "не успели оборвать клиента")
            await asyncio.sleep(0.2)
            url = f"{ROUTER}/v1/streams/{res.answer_id}?from={res.last_pos}"
            async with env.session.get(url) as r:
                await read_sse(r, res)
            info = await env.info(res.answer_id)
            ok_count += verify(env, f"resume[{i}]", res, control[i], info, need_failover=False)
        report["обрыв клиента"] = f"{ok_count}/{m} ответов дочитаны с нужной позиции"

        # 5. Двадцать ответов разом, одну реплику убиваем
        await env.all_up()
        m = min(20, args.n)
        victim = next(iter(env.replicas.values()))

        async def killer():
            await asyncio.sleep(0.15)
            victim.sig(signal.SIGKILL)

        results, _ = await asyncio.gather(
            asyncio.gather(*(ask(env, prompts[i], i) for i in range(m))), killer())
        ok_count, moved = 0, 0
        for i, res in enumerate(results):
            info = await env.info(res.answer_id)
            ok_count += verify(env, f"batch[{i}]", res, control[i], info, need_failover=False)
            moved += info["attempts"] >= 2
        env.check(moved > 0, "batch", "убийство не задело ни один ответ")
        report["20 ответов разом"] = f"{ok_count}/{m} совпали, {moved} пережили смерть реплики"

        # 6. Запрос без стрима
        await env.all_up()
        body = {"messages": [{"role": "user", "content": prompts[0]}], "seed": 0,
                "max_tokens": 4000}

        async def nonstream_killer():
            for _ in range(200):
                async with env.session.get(ROUTER + "/metrics") as r:
                    active = (await r.json())["active"]
                if active and active[0]["tokens"] >= 20:
                    env.replicas[active[0]["replica"]].sig(signal.SIGKILL)
                    return
                await asyncio.sleep(0.01)

        async def nonstream():
            async with env.session.post(ROUTER + "/v1/chat/completions", json=body) as r:
                return r.status, await r.json(), r.headers.get("X-Answer-Id")

        (status, data, aid), _ = await asyncio.gather(nonstream(), nonstream_killer())
        info = await env.info(aid)
        good = env.check(status == 200 and data["choices"][0]["message"]["content"] == control[0]["text"],
                         "nonstream", f"status={status}")
        good &= env.check(info["attempts"] >= 2, "nonstream", "сбой не попал в середину ответа")
        report["без стрима"] = "ответ целый после kill -9" if good else "ПРОВАЛ"

        # 7. Ядовитый запрос: лимит попыток
        await env.all_up()
        before = sum(p.starts for p in env.replicas.values())
        res = await ask(env, "POISON этот запрос убивает реплику", 1)
        await asyncio.sleep(0.5)
        deaths = sum(p.starts for p in env.replicas.values()) - before
        info = await env.info(res.answer_id)
        good = env.check(res.error is not None and "attempt limit" in res.error, "poison",
                         f"ждали ошибку лимита, получили {res.error!r}")
        good &= env.check(info["attempts"] == 3 and deaths == 3, "poison",
                          f"attempts={info['attempts']} deaths={deaths}")
        report["ядовитый запрос"] = (
            f"остановлен после {info['attempts']} попыток, убито реплик: {deaths}")

        # 8. Все реплики мертвы: честная ошибка
        await env.all_up()
        env.supervise = False
        for p in env.replicas.values():
            p.sig(signal.SIGKILL)
        await asyncio.sleep(0.1)
        t = time.monotonic()
        res = await ask(env, prompts[0], 0)
        took = time.monotonic() - t
        env.check(res.error is not None and "no live replicas" in res.error, "alldead",
                  f"ждали ошибку, получили {res.error!r}")
        env.check(2.5 < took < 6, "alldead", f"ошибка пришла через {took:.1f} с при дедлайне 3 с")
        report["все реплики мертвы"] = f"ошибка через {took:.1f} с при дедлайне 3 с"
        env.supervise = True
        await env.all_up()

        # 9. Реплика заморожена ДО запроса: новые ответы не должны зависнуть
        await env.all_up()
        m = min(4, args.n)
        frozen_url, frozen = next(iter(env.replicas.items()))
        t = time.monotonic()
        try:
            frozen.sig(signal.SIGSTOP)
            results = await asyncio.wait_for(
                asyncio.gather(*(ask(env, prompts[i], i) for i in range(m))), 30)
        except asyncio.TimeoutError:
            results = []
            env.check(False, "frozen-new", "ответы не пришли за 30 с: роутер завис на замороженной реплике")
        finally:
            frozen.sig(signal.SIGCONT)
        took = time.monotonic() - t
        # граница: first-token-timeout 2 с + самый длинный ответ при tps + 3 с запаса на нагрузку
        bound = 2 + max(control[i]["tokens"] for i in range(m)) / args.tps + 3
        ok_count = 0
        for i, res in enumerate(results):
            info = await env.info(res.answer_id)
            ok_count += verify(env, f"frozen-new[{i}]", res, control[i], info, need_failover=False)
            env.check(frozen_url not in info["replicas"], f"frozen-new[{i}]",
                      f"замороженная реплика числится принявшей: {info['replicas']}")
        if results:
            env.check(took < bound, "frozen-new", f"{m} ответов за {took:.1f} с, граница {bound:.1f} с")
            env.check(took > 1.8, "frozen-new",
                      f"за {took:.1f} с: ни один запрос не попал на замороженную")
        report["заморожена до запроса"] = f"{ok_count}/{m} совпали за {took:.1f} с (граница {bound:.1f} с)"
        await env.all_up()

        # 10. Реплика отклонила запрос (HTTP 400): честная ошибка сразу, без кулдауна
        await env.all_up()
        async with env.session.get(ROUTER + "/metrics") as r:
            failovers = (await r.json())["failovers"]
        t = time.monotonic()
        res = await ask(env, "TOOLONG слишком длинный запрос", 5)
        took = time.monotonic() - t
        async with env.session.get(ROUTER + "/metrics") as r:
            mt = await r.json()
        info = await env.info(res.answer_id)
        good = env.check(res.error is not None and "maximum context length" in res.error
                         and res.error.startswith("replica rejected the request: HTTP 400: "),
                         "rejected", f"сообщение: {res.error!r}")
        good &= env.check(res.error_type == "invalid_request_error", "rejected",
                          f"тип ошибки: {res.error_type!r}")
        good &= env.check(took < 1, "rejected", f"ошибка пришла через {took:.1f} с")
        good &= env.check(info["attempts"] == 0, "rejected", f"attempts={info['attempts']}")
        good &= env.check(all(v == "up" for v in mt["replicas"].values()), "rejected",
                          f"реплики в кулдауне: {mt['replicas']}")
        good &= env.check(mt["failovers"] == failovers, "rejected", "засчитано переключение")
        body = {"messages": [{"role": "user", "content": "TOOLONG слишком длинный запрос"}],
                "seed": 5, "max_tokens": 100}
        async with env.session.post(ROUTER + "/v1/chat/completions", json=body) as r:
            status, data, aid = r.status, await r.json(), r.headers.get("X-Answer-Id")
        good &= env.check(status == 400 and data["error"]["type"] == "invalid_request_error"
                          and "maximum context length" in data["error"]["message"],
                          "rejected-nonstream", f"status={status} data={data}")
        good &= env.check(bool(aid), "rejected-nonstream", "нет X-Answer-Id")
        res = await ask(env, prompts[0], 0)
        info = await env.info(res.answer_id)
        good &= verify(env, "after-rejected", res, control[0], info, need_failover=False)
        good &= env.check(info["attempts"] == 1, "after-rejected", f"attempts={info['attempts']}")
        report["HTTP 400 от реплики"] = (
            "ошибка сразу, реплики живы, без стрима 400" if good else "ПРОВАЛ")

        # 11. finish_reason abort от реплики: ответ не считается законченным
        await env.all_up()
        res = await ask(env, "ABORTME прервись на пятом токене", 7)
        info = await env.info(res.answer_id)
        good = env.check(res.error is None and res.done and res.finish == "stop", "abort",
                         f"error={res.error} finish={res.finish}")
        good &= env.check(info["attempts"] == 2, "abort", f"attempts={info['attempts']}")
        good &= env.check(info["tokens"] > 5, "abort", f"tokens={info['tokens']}")
        good &= env.check("#" not in res.text and "�" not in res.text, "abort", "битые символы")
        report["finish_reason abort"] = (
            f"продолжен на другой реплике, {info['tokens']} токенов" if good else "ПРОВАЛ")
        await env.all_up()

        # 12. finish_reason repetition от реплики: это финал, а не сбой
        await env.all_up()
        async with env.session.get(ROUTER + "/metrics") as r:
            failovers = (await r.json())["failovers"]
        res = await ask(env, "REPEATME зациклись на восьмом токене", 3)
        info = await env.info(res.answer_id)
        async with env.session.get(ROUTER + "/metrics") as r:
            mt = await r.json()
        good = env.check(res.error is None and res.done and res.finish == "repetition",
                         "repetition", f"error={res.error} finish={res.finish}")
        good &= env.check(info["attempts"] == 1 and info["tokens"] == 8, "repetition",
                          f"attempts={info['attempts']} tokens={info['tokens']}")
        good &= env.check(mt["failovers"] == failovers
                          and all(v == "up" for v in mt["replicas"].values()), "repetition",
                          f"переключения или кулдаун: {mt['failovers']} {mt['replicas']}")
        report["finish_reason repetition"] = (
            "принят как финал, без переключений" if good else "ПРОВАЛ")

        # 14. max_tokens режет ответ посреди буквы: обрезанная буква клиенту не уходит
        await env.all_up()
        raw = control[0]["text"].encode("utf-8")
        cut = next(p for p in range(20, len(raw)) if raw[p - 1] in (0xD0, 0xD1))
        whole = raw[:cut - 1].decode("utf-8")
        res = await ask(env, prompts[0], 0, max_tokens=cut)
        info = await env.info(res.answer_id)
        good = env.check(res.error is None and res.done and res.finish == "length", "midchar",
                         f"error={res.error} finish={res.finish}")
        good &= env.check(info["tokens"] == cut, "midchar", f"tokens={info['tokens']}, ждали {cut}")
        good &= env.check(res.text == whole, "midchar", f"текст {res.text[-12:]!r}, ждали {whole[-12:]!r}")
        good &= env.check(all(a < b for a, b in zip(res.positions, res.positions[1:])), "midchar",
                          "текст пришёл дважды на одной позиции")
        tail = Result()
        tail.last_pos = res.last_pos
        async with env.session.get(f"{ROUTER}/v1/streams/{res.answer_id}?from={res.last_pos}") as r:
            await read_sse(r, tail)
        good &= env.check(tail.text == "" and tail.finish == "length", "midchar",
                          f"при дочитывании с конца пришёл текст {tail.text!r}")
        body = {"messages": [{"role": "user", "content": prompts[0]}], "seed": 0,
                "max_tokens": cut, "temperature": 0.7}
        async with env.session.post(ROUTER + "/v1/chat/completions", json=body) as r:
            data = await r.json()
        good &= env.check(data["choices"][0]["message"]["content"] == whole, "midchar",
                          "ответ без стрима отличается от стрима")
        report["обрез посреди буквы"] = (
            f"на {cut}-м токене, клиент получил целые буквы" if good else "ПРОВАЛ")

        # 13. Реплика требует ключ API: без ключа быстрая ошибка, с ключом обычный ответ
        key_url = "http://127.0.0.1:19003"
        nokey_router, key_router = "http://127.0.0.1:18081", "http://127.0.0.1:18082"
        common = ["--tokenizer", "bytes", "--model", "fake-bytes", "--replicas", key_url,
                  "--deadline", "3", "--cooldown", "1", "--max-attempts", "3",
                  "--first-token-timeout", "2"]
        extra = [
            Proc("replica-19003", [sys.executable, "fake_replica.py", "--port", "19003",
                                   "--name", "19003", "--tps", str(args.tps),
                                   "--api-key", "secret-123"]),
            Proc("router-nokey", [sys.executable, "router.py", "--port", "18081", *common]),
            Proc("router-key", [sys.executable, "router.py", "--port", "18082",
                                "--replica-api-key", "secret-123", *common]),
        ]
        for p in extra:
            p.start()
        try:
            for url in (key_url, nokey_router, key_router):
                await env.wait_health(url)
            t = time.monotonic()
            res = await ask(env, prompts[0], 0, base=nokey_router)
            took = time.monotonic() - t
            good = env.check(res.error is not None and "API key" in res.error, "apikey",
                             f"ждали ошибку про ключ, получили {res.error!r}")
            good &= env.check(took < 1, "apikey",
                              f"ошибка пришла через {took:.1f} с, а дедлайн ждать незачем")
            res = await ask(env, prompts[0], 0, base=key_router)
            good &= env.check(res.error is None and res.done
                              and res.text == control[0]["text"], "apikey",
                              f"с ключом ответ не совпал: error={res.error}")
            report["ключ API реплик"] = (
                f"без ключа ошибка за {took:.1f} с, с ключом ответ целый" if good else "ПРОВАЛ")
        finally:
            for p in extra:
                p.stop()

        # 15. Все реплики заморожены ДО запроса: роутер обязан сдаться по дедлайну, а не висеть.
        #     Попытка на замороженной реплике длится дольше кулдауна, поэтому «выбрать некого»
        #     не наступает никогда: дедлайн должен срабатывать и без этого.
        await env.all_up()
        t = time.monotonic()
        try:
            for p in env.replicas.values():
                p.sig(signal.SIGSTOP)
            res = await asyncio.wait_for(ask(env, prompts[0], 0), 20)
        except asyncio.TimeoutError:
            res = None
        finally:
            for p in env.replicas.values():
                p.sig(signal.SIGCONT)
        took = time.monotonic() - t
        good = env.check(res is not None, "allfrozen",
                         "ответа нет за 20 с при дедлайне 3 с: роутер завис")
        if res is not None:
            good &= env.check(res.error is not None and "no live replicas" in res.error,
                              "allfrozen", f"ждали ошибку, получили {res.error!r}")
            good &= env.check(2.5 < took < 8, "allfrozen",
                              f"ошибка пришла через {took:.1f} с при дедлайне 3 с")
        report["все заморожены до запроса"] = (
            f"ошибка через {took:.1f} с при дедлайне 3 с" if good else "ПРОВАЛ")
        await env.all_up()

        # 16. Генерация встала, а соединение шлёт keep-alive без токенов: тишина считается
        #     от последнего токена, роутер обязан уйти на другую реплику
        stall_url, stall_router = "http://127.0.0.1:19004", "http://127.0.0.1:18083"
        live_url = next(iter(env.replicas))
        extra = [
            Proc("replica-19004", [sys.executable, "fake_replica.py", "--port", "19004",
                                   "--name", "19004", "--tps", str(args.tps),
                                   "--stall-after", "30",
                                   *([] if args.eos_id else ["--no-eos-id"])]),
            Proc("router-stall", [sys.executable, "router.py", "--port", "18083",
                                  "--tokenizer", "bytes", "--model", "fake-bytes",
                                  "--replicas", f"{stall_url},{live_url}",
                                  "--deadline", "3", "--cooldown", "1", "--max-attempts", "3",
                                  "--first-token-timeout", "2"]),
        ]
        for p in extra:
            p.start()
        try:
            for url in (stall_url, stall_router):
                await env.wait_health(url)
            t = time.monotonic()
            try:
                res = await asyncio.wait_for(ask(env, prompts[0], 0, base=stall_router), 20)
            except asyncio.TimeoutError:
                res = None
            took = time.monotonic() - t
            good = env.check(res is not None and control[0]["tokens"] > 60, "keepalive",
                             "ответ не закончился за 20 с: keep-alive сбрасывает таймер тишины")
            if res is not None:
                async with env.session.get(f"{stall_router}/v1/answers/{res.answer_id}") as r:
                    info = await r.json()
                good &= env.check(res.error is None and res.done
                                  and res.text == control[0]["text"], "keepalive",
                                  f"ответ не совпал: error={res.error}")
                good &= env.check(info["ids_sha256"] == control[0]["sha"], "keepalive",
                                  "хеш ID токенов не совпал")
                good &= env.check(info["replicas"] == [stall_url, live_url], "keepalive",
                                  f"ждали переход с зависшей реплики, было {info['replicas']}")
                good &= env.check(took < 8, "keepalive", f"ответ занял {took:.1f} с")
            report["keep-alive без токенов"] = (
                f"ушёл с зависшей реплики, ответ целый за {took:.1f} с" if good else "ПРОВАЛ")
        finally:
            for p in extra:
                p.stop()

        # 17. Курсоры дочитывания: с любой границы события хвост склеивается в тот же текст,
        #     а чужой, кривой и несуществующий курсор получают 400
        await env.all_up()
        k = 1 % args.n
        cum = {}

        async def remember(r):
            cum[r.last_pos] = r.text

        res = await ask(env, prompts[k], k, hook=remember)
        full = res.text
        marks = sorted(cum)
        good = env.check(res.done and full == control[k]["text"] and len(marks) > 3, "cursor",
                         "исходный ответ не годится для проверки")

        async def tail_from(cursor, header=False):
            tail = Result()
            kw = {"headers": {"Last-Event-ID": cursor}} if header else {"params": {"from": cursor}}
            async with env.session.get(f"{ROUTER}/v1/streams/{res.answer_id}", **kw) as r:
                if r.status != 200:
                    return r.status, tail
                await read_sse(r, tail)
            return 200, tail

        checked = 0
        for pos in marks[::max(1, len(marks) // 25)]:
            status, tail = await tail_from(str(pos))
            good &= env.check(status == 200 and tail.done and cum[pos] + tail.text == full,
                              "cursor", f"с позиции {pos} текст не склеился")
            checked += 1
        status, tail = await tail_from(f"{res.answer_id}:{marks[1]}", header=True)
        good &= env.check(status == 200 and cum[marks[1]] + tail.text == full, "cursor",
                          "Last-Event-ID с номером ответа не сработал")
        inside = next((q for q in range(1, marks[-1]) if q not in cum), marks[-1] + 5)
        bad = {"чужой ответ": f"ans-000000000000:{marks[1]}", "не число": "abc", "минус": "-1",
               "не граница события": str(inside), "дальше конца": str(marks[-1] + 1000)}
        for name, cursor in bad.items():
            status, _ = await tail_from(cursor)
            good &= env.check(status == 400, "cursor",
                              f"курсор «{name}» ({cursor}) принят, статус {status}")
        report["курсоры дочитывания"] = (
            f"{checked} границ склеились, {len(bad)} плохих курсоров отклонены"
            if good else "ПРОВАЛ")

        # 18. Поля, которые роутер не передаёт репликам: выставленное значение получает 400,
        #     значение «не задано» проходит и ответ не меняет
        base = {"messages": [{"role": "user", "content": prompts[0]}], "seed": 0,
                "max_tokens": 4000, "temperature": 0.7}
        unset = {"presence_penalty": 0, "frequency_penalty": 0.0, "repetition_penalty": 1.0,
                 "top_k": -1, "min_p": 0, "n": 1, "logprobs": False, "stop": None,
                 "tool_choice": "auto", "response_format": {"type": "text"}, "logit_bias": {}}
        async with env.session.post(ROUTER + "/v1/chat/completions", json={**base, **unset}) as r:
            data = await r.json()
            good = env.check(r.status == 200
                             and data["choices"][0]["message"]["content"] == control[0]["text"],
                             "fields", f"значения «не задано» не прошли, статус {r.status}")
        really_set = {"presence_penalty": 0.5, "frequency_penalty": 0.2, "repetition_penalty": 1.1,
                      "top_k": 40, "min_p": 0.05, "logit_bias": {"5": 1}, "min_tokens": 4, "n": 2,
                      "logprobs": True, "top_logprobs": 3, "stop": ["\n"],
                      "tools": [{"type": "function"}], "response_format": {"type": "json_object"},
                      "structured_outputs": {"regex": "a+"},
                      "chat_template_kwargs": {"enable_thinking": False}}
        for key, value in really_set.items():
            async with env.session.post(ROUTER + "/v1/chat/completions",
                                        json={**base, key: value}) as r:
                data = await r.json()
                good &= env.check(r.status == 400 and key in data["error"]["message"], "fields",
                                  f"{key}={value!r} принят, статус {r.status}")
        report["неподдержанные поля"] = (
            f"{len(really_set)} выставленных отклонены, «не задано» проходит"
            if good else "ПРОВАЛ")

        # 19. Запросы, которые могла прислать только веб-страница: чужой Origin и чужой Host
        #     получают 403, свой адрес и localhost проходят
        async def knock(method, path, headers, body=None):
            async with env.session.request(method, ROUTER + path, headers=headers,
                                           json=body) as r:
                return r.status

        chat = "/v1/chat/completions"
        body = {"messages": [{"role": "user", "content": prompts[0]}], "seed": 0, "max_tokens": 8}
        refused = {
            "чужой Origin": ("POST", chat, {"Origin": "https://evil.example"}, body),
            "Origin null": ("POST", chat, {"Origin": "null"}, body),
            "Origin похож на localhost": ("POST", chat,
                                          {"Origin": "http://localhost.evil.example"}, body),
            "чужой Host": ("GET", "/metrics", {"Host": f"evil.example:{ROUTER_PORT}"}, None),
            "Host 0.0.0.0": ("GET", "/metrics", {"Host": f"0.0.0.0:{ROUTER_PORT}"}, None),
            "чужой Host со своим Origin": ("POST", chat,
                                           {"Host": f"evil.example:{ROUTER_PORT}",
                                            "Origin": f"http://evil.example:{ROUTER_PORT}"}, body),
        }
        passed = {
            "без Origin": ("POST", chat, {}, body),
            "свой Origin": ("POST", chat, {"Origin": f"http://127.0.0.1:{ROUTER_PORT}"}, body),
            "страница с localhost": ("POST", chat, {"Origin": "http://localhost:3000"}, body),
            "Host localhost": ("GET", "/metrics", {"Host": f"localhost:{ROUTER_PORT}"}, None),
            "Host [::1]": ("GET", "/metrics", {"Host": f"[::1]:{ROUTER_PORT}"}, None),
        }
        good = True
        for name, call in refused.items():
            status = await knock(*call)
            good &= env.check(status == 403, "guard", f"«{name}» прошёл, статус {status}")
        for name, call in passed.items():
            status = await knock(*call)
            good &= env.check(status == 200, "guard", f"«{name}» отклонён, статус {status}")
        # те же правила с флагами запуска, без процессов: (страж, Host, Origin, пропустить ли)
        sys.path.insert(0, ROOT)
        from router import BrowserGuard
        table = [
            (BrowserGuard("0.0.0.0"), "api.example:8080", None, True),
            (BrowserGuard("0.0.0.0"), "api.example:8080", "https://evil.example", False),
            (BrowserGuard("0.0.0.0"), "api.example:8080", "https://api.example:8080", True),
            (BrowserGuard("0.0.0.0", "api.example"), "other.example", None, False),
            (BrowserGuard("0.0.0.0", "api.example"), "127.0.0.1:8080", None, True),
            (BrowserGuard("127.0.0.1", "api.example"), "API.example.:443", None, True),
            (BrowserGuard("127.0.0.1", "*"), "evil.example", None, True),
            (BrowserGuard("127.0.0.1", None, "https://ui.example"), "127.0.0.1:8080",
             "https://ui.example/", True),
            (BrowserGuard("127.0.0.1", None, "*"), "127.0.0.1:8080", "https://evil.example", True),
            (BrowserGuard("127.0.0.1"), None, None, True),
        ]
        for i, (guard, host, origin, ok) in enumerate(table):
            good &= env.check((guard.reason(host, origin) is None) == ok, "guard",
                              f"строка таблицы {i}: Host={host} Origin={origin}")
        report["Origin и Host"] = (
            f"{len(refused)} чужих отклонены, {len(passed)} своих прошли, "
            f"{len(table)} строк таблицы сошлись" if good else "ПРОВАЛ")

        # реестр реплик, админ-API, очередь и журнал: свои реплики и роутер на портах 19011-19014, 18091, 18191
        await registry_tests(env, report, args)

        # после всего роутер жив и отвечает как раньше
        res = await ask(env, prompts[0], 0)
        info = await env.info(res.answer_id)
        verify(env, "after-all", res, control[0], info, need_failover=False)

        async with env.session.get(ROUTER + "/metrics") as r:
            m = await r.json()
        report["счётчики роутера"] = (
            f"ответов {m['answers']}, с финалом {m['finished']}, "
            f"с ошибкой {m['failed']}, переключений {m['failovers']}")
    finally:
        await env.down()

    print()
    for k, v in report.items():
        print(f"  {k:24} {v}")
    print()
    if env.failures:
        print(f"ПРОВАЛЕНО проверок: {len(env.failures)}")
        for f in env.failures[:30]:
            print("  -", f)
        return 1
    print("Все проверки прошли.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="сколько промптов")
    ap.add_argument("--tps", type=float, default=400, help="скорость фейковой реплики")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-eos-id", dest="eos_id", action="store_false",
                    help="реплика не кладёт ID конца ответа в token_ids")
    sys.exit(asyncio.run(main(ap.parse_args())))
