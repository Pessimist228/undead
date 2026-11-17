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
import signal
import statistics
import subprocess
import sys
import time

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
