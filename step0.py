#!/usr/bin/env python3
"""Стенд шага 0: совпадает ли продолжение на другой реплике с непрерванным ответом.

    python step0.py run 0a                      одна карта, батч-инвариантность включена
    python step0.py run 0c                      та же карта, настройка vLLM по умолчанию
    python step0.py run 0d --model <NVFP4>      та же проверка на квантованной модели
    python step0.py serve                       одна реплика с флагами этапа (для двух хостов)
    python step0.py run 0b --replica-urls A,B   контроль на одном хосте, продолжение на другом
    python step0.py report results              четыре числа для поста по всем прогонам
    python step0.py run 0a --dry                проверка самого стенда на фейковых репликах

Что делает прогон:
  1. Контроль: ответ целиком на первой реплике, с ID токенов и logprobs.
  2. Разрез: «промпт + первые k токенов контроля» уходит на вторую реплику.
     Хвост обязан совпасть с контролем по ID токенов, logprobs сверяются побитово.
  3. То же под фоновой нагрузкой на второй реплике.
  4. Через роутер: kill -9 и SIGSTOP по группе процессов реплики посреди стрима.
Каждый запрос и каждое сравнение пишутся строкой в events.jsonl, итог в summary.md.

Коды выхода: 0 всё сошлось или этап был замером, 2 стенд не смог работать,
3 в этапе, где совпадение ожидалось, есть расхождения.
"""
import argparse
import asyncio
import atexit
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp
from aiohttp import ClientTimeout

from prompts import LOAD_PROMPTS, PROMPTS
from toks import make_tokenizer

ROOT = Path(__file__).resolve().parent
KEY_ENV = "UNDEAD_REPLICA_API_KEY"


class Abort(Exception):
    """Стенд не может продолжать: формат стрима не тот, реплика не поднялась."""


# ---------- конфигурации этапов ----------


@dataclass(frozen=True)
class Config:
    name: str
    bi: bool  # VLLM_BATCH_INVARIANT
    prefix_cache: bool
    perf: bool = False  # замер скорости на первой реплике
    e2e: bool = False  # прогон через роутер с убийством и заморозкой
    expect_match: bool = True  # расхождение считается провалом этапа


STAGES = {
    "0a": [
        Config("bi-cache-off", True, False, perf=True, e2e=True),
        Config("bi-cache-on", True, True),
    ],
    "0b": [Config("two-hosts", True, False)],
    "0c": [Config("default", False, True, perf=True, e2e=True, expect_match=False)],
    "0d": [Config("bi-cache-off", True, False, perf=True, e2e=True, expect_match=False)],
}
SERVE_CONFIGS = {c.name: c for cs in STAGES.values() for c in cs if c.name != "two-hosts"}


@dataclass(frozen=True)
class Mode:
    name: str
    temperature: float
    top_p: float
    seed: int


def modes_for(i):
    return [Mode("greedy", 0.0, 1.0, 1000 + i), Mode("seed", 0.7, 0.95, 1000 + i)]


# ---------- процессы ----------

_PROCS = []


class Proc:
    """Процесс в собственной группе: сигнал получают и сервер, и его дочерние процессы."""

    def __init__(self, name, cmd, logdir, env=None):
        self.name, self.cmd, self.env = name, cmd, env
        self.log = Path(logdir) / f"{name}.log"
        self.popen = None
        self.starts = 0
        self.was_healthy = False
        self.bad_starts = 0
        _PROCS.append(self)

    def start(self):
        self.log.parent.mkdir(parents=True, exist_ok=True)
        out = open(self.log, "ab")
        out.write(f"\n===== запуск {self.starts + 1}: {time.strftime('%H:%M:%S')} =====\n".encode())
        out.flush()
        self.popen = subprocess.Popen(
            self.cmd, cwd=ROOT, env=self.env, stdout=out, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.starts += 1
        self.was_healthy = False

    def alive(self):
        return self.popen is not None and self.popen.poll() is None

    def sig(self, s):
        if self.popen is None:
            return
        try:
            os.killpg(self.popen.pid, s)
        except (ProcessLookupError, PermissionError):
            pass

    def stop(self):
        self.sig(signal.SIGCONT)
        self.sig(signal.SIGKILL)
        if self.popen is not None:
            try:
                self.popen.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass

    def tail(self, lines=40):
        try:
            data = self.log.read_text(errors="replace").splitlines()
        except OSError:
            return "(журнала нет)"
        return "\n".join(data[-lines:])

    def grep(self, needles, limit=30):
        try:
            data = self.log.read_text(errors="replace").splitlines()
        except OSError:
            return []
        out = [ln.strip()[:300] for ln in data if any(n in ln for n in needles)]
        return out[:limit]


@atexit.register
def _kill_all():
    for p in _PROCS:
        p.stop()


def port_busy(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def serve_command(args, cfg, port, host="127.0.0.1", gpu_frac=None):
    """Команда и окружение одной реплики vLLM под конфигурацию этапа."""
    frac = gpu_frac if gpu_frac is not None else args.gpu_frac
    cmd = [
        "vllm", "serve", args.model, "--host", host, "--port", str(port),
        "--generation-config", "vllm",
        "--gpu-memory-utilization", str(frac),
        "--max-model-len", str(args.max_model_len),
        "--max-num-seqs", str(args.max_num_seqs),
        "--enable-prefix-caching" if cfg.prefix_cache else "--no-enable-prefix-caching",
    ]
    if args.revision:
        cmd += ["--revision", args.revision]
    backend = args.attention_backend
    if backend is None:
        # тест vLLM «декод против префилла» идёт на этом бэкенде
        backend = "FLASH_ATTN" if cfg.bi else ""
    if backend:
        cmd += ["--attention-backend", backend]
    cmd += shlex.split(args.extra_serve_args or "")
    env = dict(os.environ)
    env["VLLM_USE_V2_MODEL_RUNNER"] = "1"  # без тихого отката на V1
    env["VLLM_BATCH_INVARIANT"] = "1" if cfg.bi else "0"
    return cmd, env


def fake_command(args, port, drift=None):
    cmd = [sys.executable, "fake_replica.py", "--port", str(port), "--name", str(port),
           "--tps", str(args.dry_tps)]
    if drift == "token":
        cmd += ["--drift-token", "97"]
    elif drift == "logprob":
        cmd += ["--drift-logprob", "5"]
    return cmd, None


# ---------- сведения о машине ----------


def _pkg(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _file_sha(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
    except OSError:
        return None


def fingerprint(args):
    gpus = []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        )
        for line in out.stdout.strip().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 3:
                gpus.append({"name": parts[0], "driver": parts[1], "memory": parts[2]})
    except (OSError, subprocess.SubprocessError):
        pass
    return {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {n: _pkg(n) for n in (
            "vllm", "torch", "transformers", "tokenizers", "triton", "flashinfer-python",
            "aiohttp", "numpy")},
        "gpus": gpus,
        "vllm_env": {k: v for k, v in sorted(os.environ.items())
                     if k.startswith("VLLM_") and "KEY" not in k},
        "model": getattr(args, "model", None),
        "revision": getattr(args, "revision", None),
        "code": {f: _file_sha(ROOT / f) for f in (
            "step0.py", "router.py", "toks.py", "prompts.py", "fake_replica.py")},
    }


def resolve_revision(model):
    """Ревизия модели на Hugging Face, чтобы обе реплики и оба хоста взяли одни веса."""
    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(model).sha
    except Exception as e:  # noqa: BLE001
        print(f"  ревизию модели узнать не удалось ({type(e).__name__}), идём без неё")
        return None


# ---------- журнал ----------


class Journal:
    def __init__(self, path, stage):
        self.stage = stage
        self.events = []
        self.f = open(path, "a", encoding="utf-8")

    def add(self, **ev):
        ev = {"t": round(time.time(), 3), "stage": self.stage, **ev}
        self.events.append(ev)
        self.f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        self.f.flush()
        return ev

    def note(self, text, config=None):
        print(f"  ! {text}")
        self.add(kind="note", config=config, text=text)


# ---------- стенд: реплики и роутер ----------


class Stand:
    def __init__(self, args, cfg, rundir, journal):
        self.args, self.cfg, self.j = args, cfg, journal
        self.logdir = Path(rundir) / "logs" / cfg.name
        self.remote = bool(args.replica_urls)
        self.replicas = {}
        if self.remote:
            self.urls = [u.strip().rstrip("/") for u in args.replica_urls.split(",") if u.strip()]
        else:
            self.urls = [f"http://127.0.0.1:{p}" for p in args.ports]
            for n, (url, port) in enumerate(zip(self.urls, args.ports)):
                if args.dry:
                    cmd, env = fake_command(args, port, args.dry_drift if n == 1 else None)
                else:
                    cmd, env = serve_command(args, cfg, port)
                self.replicas[url] = Proc(f"replica-{port}", cmd, self.logdir, env)
        if len(self.urls) != 2:
            raise Abort("нужны ровно две реплики")
        self.router = None
        self.router_url = f"http://127.0.0.1:{args.router_port}"
        self.http = None
        self.supervise = True
        self.fatal = None
        self._sup = None

    async def up(self):
        headers = {"Authorization": f"Bearer {self.args.api_key}"} if self.args.api_key else None
        # Каждый запрос идёт своим соединением: реплика закрывает простаивающее через 5 с,
        # и POST по такому соединению падает с ServerDisconnectedError (поймано на 0b).
        self.http = aiohttp.ClientSession(read_bufsize=2**22, headers=headers,
                                          connector=aiohttp.TCPConnector(force_close=True))
        if not self.remote:
            ports = list(self.args.ports) + ([self.args.router_port] if self.cfg.e2e else [])
            busy = [p for p in ports if port_busy(p)]
            if busy:
                raise Abort(f"порты {busy} заняты. Похоже, остались процессы прошлого прогона: "
                            f"проверь `ps aux | grep -E 'vllm serve|router.py|fake_replica'` "
                            f"и заверши их")
        for url in self.urls:
            proc = self.replicas.get(url)
            if proc:
                print(f"  запускаю {proc.name}: {' '.join(proc.cmd)}")
                flags = {k: (proc.env or {}).get(k)
                         for k in ("VLLM_BATCH_INVARIANT", "VLLM_USE_V2_MODEL_RUNNER")}
                self.j.add(kind="replica_cmd", config=self.cfg.name, replica=proc.name,
                           cmd=proc.cmd, env=flags)
                proc.start()
            await self.wait_health(url, self.args.startup_timeout, proc, fail_fast=True)
        if self.cfg.e2e and not self.remote:
            a = self.args
            cmd = [sys.executable, "router.py", "--replicas", ",".join(self.urls),
                   "--model", a.model, "--tokenizer", a.tokenizer, "--host", "127.0.0.1",
                   "--port", str(a.router_port), "--max-attempts", "3",
                   "--first-token-timeout", "2" if a.dry else "15",
                   "--cooldown", "1" if a.dry else "3",
                   "--deadline", "5" if a.dry else "180"]
            env = dict(os.environ)
            if a.api_key:
                env[KEY_ENV] = a.api_key
            self.router = Proc("router", cmd, self.logdir, env)
            self.router.start()
            await self.wait_health(self.router_url, 120, self.router, fail_fast=True)
        self._sup = asyncio.create_task(self._supervisor())

    async def down(self):
        if self._sup:
            self._sup.cancel()
        if self.http:
            await self.http.close()
        for p in [self.router, *self.replicas.values()]:
            if p:
                p.stop()
        if self.replicas:
            await asyncio.sleep(self.args.restart_delay)  # карта освобождает память

    async def _supervisor(self):
        """Поднимает убитые реплики, как это сделал бы Kubernetes."""
        while True:
            if self.supervise:
                for p in self.replicas.values():
                    if p.alive():
                        continue
                    p.bad_starts = 0 if p.was_healthy else p.bad_starts + 1
                    if p.bad_starts >= 4:
                        self.fatal = f"{p.name} четыре раза подряд не поднялся:\n{p.tail()}"
                        return
                    p.sig(signal.SIGKILL)  # добить остатки группы, чтобы карта освободилась
                    await asyncio.sleep(self.args.restart_delay)
                    p.start()
            await asyncio.sleep(0.2)

    async def wait_health(self, url, timeout, proc=None, fail_fast=False):
        t0 = time.monotonic()
        shown = t0
        while time.monotonic() - t0 < timeout:
            if self.fatal:
                raise Abort(self.fatal)
            if fail_fast and proc is not None and not proc.alive():
                raise Abort(f"{proc.name} завершился при запуске. Хвост журнала:\n{proc.tail()}")
            try:
                async with self.http.get(url + "/health", timeout=ClientTimeout(total=3)) as r:
                    if r.status == 200:
                        if proc:
                            proc.was_healthy = True
                        return
            except Exception:  # noqa: BLE001
                pass
            if time.monotonic() - shown > 30:
                shown = time.monotonic()
                print(f"  жду {url}: {shown - t0:.0f} с")
            await asyncio.sleep(0.2 if self.args.dry else 1.0)
        tail = f"\nХвост журнала:\n{proc.tail()}" if proc else ""
        raise Abort(f"{url} не поднялся за {timeout:g} с{tail}")

    async def all_up(self):
        for url in self.urls:
            await self.wait_health(url, self.args.startup_timeout, self.replicas.get(url))
        if self.router:
            end = time.monotonic() + 60
            while time.monotonic() < end:
                if not self.router.alive():
                    raise Abort(f"роутер завершился. Хвост журнала:\n{self.router.tail()}")
                try:
                    async with self.http.get(self.router_url + "/metrics") as r:
                        m = await r.json()
                    if all(v == "up" for v in m["replicas"].values()):
                        return
                except aiohttp.ClientError:
                    pass
                await asyncio.sleep(0.1)
            raise Abort("роутер минуту не видит обе реплики живыми")


# ---------- запросы и сравнение ----------


@dataclass
class Gen:
    ids: list = field(default_factory=list)
    lps: list = field(default_factory=list)
    finish: str | None = None
    ttft: float | None = None
    dur: float = 0.0
    error: str | None = None
    saw_prompt_ids: bool = False
    chunks_without_ids: int = 0
    lp_misaligned: int = 0


def sha_ids(ids):
    return hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest()


def compare(exp_ids, exp_lps, got_ids, got_lps):
    """Сверка хвоста: первый несовпавший токен и logprobs до него."""
    n = min(len(exp_ids), len(got_ids))
    first = next((j for j in range(n) if exp_ids[j] != got_ids[j]), None)
    if first is None and len(exp_ids) != len(got_ids):
        first = n
    upto = n if first is None else first
    lp_cmp = lp_equal = 0
    max_d, lp_first = 0.0, None
    for j in range(upto):
        a = exp_lps[j] if j < len(exp_lps) else None
        b = got_lps[j] if j < len(got_lps) else None
        if a is None or b is None:
            continue
        lp_cmp += 1
        if a == b:
            lp_equal += 1
        else:
            max_d = max(max_d, abs(a - b))
            if lp_first is None:
                lp_first = j
    out = {"ids_match": first is None, "first_diff": first, "n_cmp": upto, "lp_cmp": lp_cmp,
           "lp_equal": lp_equal, "max_dlp": max_d, "lp_first_diff": lp_first}
    if first is not None:
        out["diff"] = {
            "exp_id": exp_ids[first] if first < len(exp_ids) else None,
            "got_id": got_ids[first] if first < len(got_ids) else None,
            "exp_lp": exp_lps[first] if first < len(exp_lps) else None,
            "got_lp": got_lps[first] if first < len(got_lps) else None,
        }
    return out


class Result:
    """Что увидел клиент роутера."""

    def __init__(self):
        self.answer_id = None
        self.text = ""
        self.positions = []
        self.last_pos = 0
        self.finish = None
        self.error = None
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


class Load:
    """Фоновая нагрузка: conc запросов крутятся на реплике, пока идёт сверка."""

    def __init__(self, bench, url, conc):
        self.bench, self.url, self.conc = bench, url, conc
        self.stop = False
        self.done = self.errors = self.started = 0
        self.tasks = []

    async def _worker(self, w):
        b = self.bench
        n = 0
        while not self.stop:
            k = w * 1000 + n
            n += 1
            ids = b.load_ids[k % len(b.load_ids)]
            self.started += 1
            g = await b.generate(self.url, b.body(ids, 128, Mode("seed", 0.8, 0.9, 5000 + k),
                                                  logprobs=False))
            if g.error:
                self.errors += 1
                await asyncio.sleep(0.2)
            else:
                self.done += 1

    async def __aenter__(self):
        self.tasks = [asyncio.create_task(self._worker(w)) for w in range(self.conc)]
        end = time.monotonic() + 5
        while self.started < self.conc and time.monotonic() < end:
            await asyncio.sleep(0.05)
        await asyncio.sleep(0.3)  # дать запросам войти в батч
        return self

    async def __aexit__(self, *exc):
        self.stop = True
        for t in self.tasks:
            t.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)


class Bench:
    def __init__(self, args, cfg, stand, journal):
        self.args, self.cfg, self.stand, self.j = args, cfg, stand, journal
        self.tok = make_tokenizer(args.tokenizer, args.model)
        self.stop_ids = list(self.tok.stop_ids)
        self._pids = {}
        self.load_ids = [self.tok.encode_chat([{"role": "user", "content": p}])
                         for p in LOAD_PROMPTS]
        self.controls = {}
        self.seen_prompt, self.seen_answer = set(), set()

    def ev(self, **kw):
        return self.j.add(config=self.cfg.name, **kw)

    def pids(self, i):
        if i not in self._pids:
            self._pids[i] = self.tok.encode_chat([{"role": "user", "content": PROMPTS[i]}])
        return self._pids[i]

    def body(self, prompt_ids, max_tokens, mode, logprobs=True):
        b = {"model": self.args.model, "prompt": prompt_ids, "stream": True,
             "return_token_ids": True, "add_special_tokens": False,
             "max_tokens": max_tokens, "stop_token_ids": self.stop_ids,
             "temperature": mode.temperature, "top_p": mode.top_p, "seed": mode.seed}
        if logprobs:
            b["logprobs"] = 1
        return b

    async def generate(self, url, body):
        g = Gen()
        t0 = time.monotonic()
        try:
            timeout = ClientTimeout(total=self.args.request_timeout)
            async with self.stand.http.post(url + "/v1/completions", json=body,
                                            timeout=timeout) as r:
                if r.status != 200:
                    g.error = f"HTTP {r.status}: {(await r.text())[:300]}"
                    return g
                async for raw in r.content:
                    line = raw.strip()
                    if not line.startswith(b"data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == b"[DONE]":
                        break
                    obj = json.loads(payload)
                    if "error" in obj:
                        g.error = "ошибка в стриме: " + json.dumps(
                            obj["error"], ensure_ascii=False)[:300]
                        break
                    for ch in obj.get("choices") or []:
                        ids = ch.get("token_ids")
                        if ids is None:
                            g.chunks_without_ids += 1
                            ids = []
                        if ids and g.ttft is None:
                            g.ttft = time.monotonic() - t0
                        g.ids += [int(x) for x in ids]
                        lp = (ch.get("logprobs") or {}).get("token_logprobs")
                        if lp is None:
                            lp = [None] * len(ids)
                        elif len(lp) != len(ids):
                            g.lp_misaligned += 1
                            lp = (list(lp) + [None] * len(ids))[:len(ids)]
                        g.lps += lp
                        if ch.get("prompt_token_ids"):
                            g.saw_prompt_ids = True
                        if ch.get("finish_reason"):
                            g.finish = ch["finish_reason"]
            if g.error is None and g.finish is None:
                g.error = "стрим закончился без finish_reason"
        except Exception as e:  # noqa: BLE001
            g.error = f"{type(e).__name__}: {e}"
        finally:
            g.dur = time.monotonic() - t0
        return g

    # ----- метки кэша -----

    def cache_label(self, url, i, mode):
        if not self.cfg.prefix_cache:
            return "off"
        if (url, i, mode) in self.seen_answer:
            return "warm-answer"  # реплика уже считала этот ответ
        if (url, i) in self.seen_prompt:
            return "warm-prompt"  # реплика видела только промпт
        return "cold"

    def mark(self, url, i, mode):
        self.seen_prompt.add((url, i))
        self.seen_answer.add((url, i, mode))

    # ----- проверка формата -----

    async def preflight(self):
        print(f"[{self.cfg.name}] проверка формата стрима")
        for url in self.stand.urls:
            g = await self.generate(url, self.body(self.load_ids[0], 24, Mode("greedy", 0.0, 1.0, 1)))
            if g.error:
                raise Abort(f"{url}: пробный запрос не прошёл: {g.error}")
            if not g.ids:
                raise Abort(f"{url}: в чанках нет token_ids, на такой реплике стенд и роутер "
                            f"работать не могут")
            has_lp = any(x is not None for x in g.lps)
            self.ev(kind="preflight", replica=url, tokens=len(g.ids), finish=g.finish,
                    has_logprobs=has_lp, lp_misaligned=g.lp_misaligned,
                    prompt_ids_echo=g.saw_prompt_ids,
                    stop_id_in_stream=bool(g.ids and g.ids[-1] in self.stop_ids),
                    ttft=round(g.ttft or 0, 4))
            if not has_lp:
                self.j.note(f"{url} не отдал logprobs: сверка пойдёт только по ID токенов",
                            self.cfg.name)
            if g.lp_misaligned:
                self.j.note(f"{url}: число logprobs в чанке не равно числу token_ids",
                            self.cfg.name)
        for p in self.stand.replicas.values():
            lines = p.grep(["model runner", "ttention backend", "atch invarian", "refix cach"])
            if lines:
                self.ev(kind="replica_log", replica=p.name, lines=lines)
            if any("V1 model runner" in ln for ln in lines):
                self.j.note(f"{p.name}: в журнале упомянут раннер V1, шум от сида там "
                            f"считается иначе. Смотри logs/{self.cfg.name}/{p.name}.log",
                            self.cfg.name)

    # ----- замер скорости -----

    async def perf(self):
        url = self.stand.urls[0]
        greedy = Mode("greedy", 0.0, 1.0, 1)
        for conc in (1, 16):
            count = 4 if conc == 1 else 16
            bodies = [self.body(self.load_ids[k % len(self.load_ids)], 256, greedy, logprobs=False)
                      for k in range(count)]
            t0 = time.monotonic()
            if conc == 1:
                gens = [await self.generate(url, b) for b in bodies]
                wall = sum(g.dur for g in gens)
            else:
                gens = await asyncio.gather(*(self.generate(url, b) for b in bodies))
                wall = time.monotonic() - t0
            tokens = sum(len(g.ids) for g in gens)
            ttfts = [g.ttft for g in gens if g.ttft is not None]
            self.ev(kind="perf", conc=conc, requests=count, tokens=tokens,
                    tok_s=round(tokens / wall, 1) if wall else None,
                    ttft_med=round(statistics.median(ttfts), 4) if ttfts else None,
                    errors=sum(1 for g in gens if g.error))
            print(f"[{self.cfg.name}] скорость, {conc} поток(ов): "
                  f"{tokens / wall:.0f} ток/с" if wall else "нет данных")

    # ----- контроль и разрезы -----

    async def control(self, i, mode, url):
        g = await self.generate(url, self.body(self.pids(i), self.args.max_tokens, mode))
        cache = self.cache_label(url, i, mode.name)
        self.mark(url, i, mode.name)
        if g.error:
            self.ev(kind="control", prompt=i, mode=mode.name, seed=mode.seed, replica=url,
                    error=g.error)
            return None
        content = len(g.ids) - (1 if g.ids and g.ids[-1] in self.stop_ids else 0)
        ctl = {"ids": g.ids, "lps": g.lps, "finish": g.finish, "content": content, "replica": url}
        self.controls[(i, mode.name)] = ctl
        self.ev(kind="control", prompt=i, mode=mode.name, seed=mode.seed, replica=url, cache=cache,
                prompt_tokens=len(self.pids(i)), n=len(g.ids), finish=g.finish,
                sha=sha_ids(g.ids), ttft=round(g.ttft or 0, 4), dur=round(g.dur, 3),
                ids=g.ids, lps=g.lps)
        return ctl

    def cut_point(self, i, mode, which, ctl):
        n = ctl["content"]
        if n < 6:
            return None
        rng = random.Random(f"{i}:{mode.name}:{which}")
        if which == 1:
            return rng.randint(1, max(1, n // 4))
        if which == 2:
            return rng.randint(n // 2, n - 1)
        return rng.randint(max(1, n // 4), max(1, 3 * n // 4))

    async def rerun(self, kind, i, mode, ctl, url, k=0, load=0, tag=None):
        """Запрос с префиксом из k токенов контроля и сверка хвоста. k=0: повтор целиком."""
        cache = self.cache_label(url, i, mode.name)
        body = self.body(self.pids(i) + ctl["ids"][:k], self.args.max_tokens - k, mode)
        g = await self.generate(url, body)
        self.mark(url, i, mode.name)
        ev = {"kind": kind, "tag": tag, "prompt": i, "mode": mode.name, "seed": mode.seed,
              "cut": k, "cache": cache, "load": load, "replica": url,
              "control_replica": ctl["replica"]}
        if g.error:
            ev["error"] = g.error
        else:
            ev.update(compare(ctl["ids"][k:], ctl["lps"][k:], g.ids, g.lps))
            ev["finish_match"] = g.finish == ctl["finish"]
            ev["match"] = ev["ids_match"] and ev["finish_match"]
            ev.update(n=len(g.ids), finish=g.finish, ttft=round(g.ttft or 0, 4),
                      dur=round(g.dur, 3), ids=g.ids, lps=g.lps)
        return self.ev(**ev)

    def route(self, i):
        """Где контроль и где продолжение. На двух хостах направления чередуются."""
        a, b = self.stand.urls
        if self.stand.remote and i % 2:
            return b, a
        return a, b

    async def matrix(self):
        n = self.args.n
        bad = 0
        t0 = time.monotonic()
        for i in range(n):
            src, dst = self.route(i)
            modes = modes_for(i)
            if i % 2:
                modes.reverse()  # чтобы холодный кэш достался обоим режимам
            for mode in modes:
                ctl = await self.control(i, mode, src)
                if ctl is None:
                    continue
                for which in (1, 2):
                    k = self.cut_point(i, mode, which, ctl)
                    if k is None:
                        continue
                    ev = await self.rerun("cut", i, mode, ctl, dst, k=k, tag=f"cut{which}")
                    bad += 0 if ev.get("match") else 1
            if (i + 1) % 5 == 0 or i + 1 == n:
                print(f"[{self.cfg.name}] разрезы: промптов {i + 1}/{n}, не сошлось {bad}, "
                      f"{time.monotonic() - t0:.0f} с")
            if i + 1 == 5 and bad >= 10 and self.cfg.expect_match:
                self.j.note("больше половины первых продолжений разошлись: похоже на системную "
                            "причину (раннер V1, не тот бэкенд внимания, разные флаги реплик), "
                            "а не на редкие ничьи в логитах", self.cfg.name)
        both = [i for i in range(n)
                if (i, "greedy") in self.controls and (i, "seed") in self.controls]
        if both and all(self.controls[(i, "greedy")]["ids"] == self.controls[(i, "seed")]["ids"]
                        for i in both):
            self.j.note("ответы с сидом совпали с жадными на всех промптах: сэмплинг не сработал, "
                        "режим seed в этом прогоне ничего не доказывает", self.cfg.name)
        if self.args.load <= 0:
            return
        bad = 0
        for target in dict.fromkeys(self.route(i)[1] for i in range(n)):
            async with Load(self, target, self.args.load) as load:
                for i in range(n):
                    if self.route(i)[1] != target:
                        continue
                    for mode in modes_for(i):
                        ctl = self.controls.get((i, mode.name))
                        k = self.cut_point(i, mode, 3, ctl) if ctl else None
                        if k is None:
                            continue
                        ev = await self.rerun("cut", i, mode, ctl, target, k=k,
                                              load=self.args.load, tag="cut-load")
                        bad += 0 if ev.get("match") else 1
            self.ev(kind="load", replica=target, conc=self.args.load, done=load.done,
                    errors=load.errors)
            if load.done == 0:
                self.j.note(f"фоновая нагрузка на {target} не выполнила ни одного запроса",
                            self.cfg.name)
        print(f"[{self.cfg.name}] разрезы под нагрузкой: не сошлось {bad}")

    async def repeats(self):
        """Повтор целиком: на той же реплике (пол шума) и на другой."""
        bad = 0
        for i in range(min(self.args.repeats, self.args.n)):
            for mode in modes_for(i):
                ctl = self.controls.get((i, mode.name))
                if not ctl:
                    continue
                other = next(u for u in self.stand.urls if u != ctl["replica"])
                for kind, url in (("aa", ctl["replica"]), ("ab", other)):
                    ev = await self.rerun(kind, i, mode, ctl, url)
                    bad += 0 if ev.get("match") else 1
        print(f"[{self.cfg.name}] повторы целиком: не сошлось {bad}")

    # ----- через роутер -----

    async def info(self, aid):
        async with self.stand.http.get(f"{self.stand.router_url}/v1/answers/{aid}") as r:
            return await r.json()

    async def ask(self, i, hook=None):
        mode = modes_for(i)[1]
        res = Result()
        body = {"messages": [{"role": "user", "content": PROMPTS[i]}], "stream": True,
                "seed": mode.seed, "max_tokens": self.args.max_tokens,
                "temperature": mode.temperature, "top_p": mode.top_p}
        try:
            async with self.stand.http.post(self.stand.router_url + "/v1/chat/completions",
                                            json=body) as r:
                res.answer_id = r.headers.get("X-Answer-Id")
                await read_sse(r, res, hook)
        except Exception as e:  # noqa: BLE001
            res.error = res.error or f"{type(e).__name__}: {e}"
        return res

    async def e2e_event(self, sub, i, res, **extra):
        ctl = self.controls.get((i, "seed"))
        info = await self.info(res.answer_id) if res.answer_id else {}
        ev = {"kind": "e2e", "sub": sub, "prompt": i, "answer": res.answer_id,
              "error": res.error, "finish": res.finish, "tokens": info.get("tokens"),
              "attempts": info.get("attempts"), "replicas": info.get("replicas"),
              "pauses": info.get("pauses"), "sha": info.get("ids_sha256")}
        if ctl is None:
            ev["error"] = ev["error"] or "нет контрольного ответа"
            ev["match"] = False
        else:
            ev["sha_ok"] = info.get("ids_sha256") == sha_ids(ctl["ids"])
            # роутер не отдаёт букву, обрезанную по max_tokens
            ev["text_ok"] = res.text == self.tok.decode(ctl["ids"]).rstrip("\ufffd")
            ev["order_ok"] = all(a < b for a, b in zip(res.positions, res.positions[1:]))
            ev["match"] = bool(res.error is None and res.done and res.finish == ctl["finish"]
                               and ev["sha_ok"] and ev["text_ok"] and ev["order_ok"])
        ev.update(extra)
        return self.ev(**ev), info

    async def e2e(self):
        a, stand = self.args, self.stand
        idx = [i for i in range(min(a.e2e_prompts, a.n)) if (i, "seed") in self.controls]
        if not idx:
            self.j.note("для прогона через роутер нет контрольных ответов", self.cfg.name)
            return
        name = self.cfg.name

        bad = 0
        for i in idx:
            ev, _ = await self.e2e_event("control", i, await self.ask(i))
            bad += 0 if ev["match"] else 1
        print(f"[{name}] роутер без сбоев: {len(idx) - bad}/{len(idx)} совпали с прямым контролем")

        ok = hit = 0
        for j in range(a.kills):
            i = idx[j % len(idx)]
            await stand.all_up()
            n = self.controls[(i, "seed")]["content"]
            kill_at = random.Random(f"kill:{j}").randint(max(2, n // 10), max(3, n - n // 4))
            killed = []

            async def hook(res, kill_at=kill_at, killed=killed):
                if not killed and res.last_pos >= kill_at:
                    inf = await self.info(res.answer_id)
                    proc = stand.replicas.get(inf.get("replica"))
                    if proc:
                        proc.sig(signal.SIGKILL)
                        killed.append(inf["replica"])

            ev, info = await self.e2e_event("kill", i, await self.ask(i, hook),
                                            kill_at=kill_at, killed=killed)
            landed = (info.get("attempts") or 0) >= 2
            hit += landed
            ok += bool(ev["match"] and landed)
            if (j + 1) % 5 == 0 or j + 1 == a.kills:
                print(f"[{name}] kill -9: {j + 1}/{a.kills}, сбой попал в ответ {hit}, совпало {ok}")

        ok = hit = 0
        for j in range(a.stops):
            i = idx[(j + a.kills) % len(idx)]
            await stand.all_up()
            n = self.controls[(i, "seed")]["content"]
            stop_at = max(2, min(20, n // 3))
            frozen = []

            async def hook(res, stop_at=stop_at, frozen=frozen):
                if not frozen and res.last_pos >= stop_at:
                    inf = await self.info(res.answer_id)
                    proc = stand.replicas.get(inf.get("replica"))
                    if proc:
                        proc.sig(signal.SIGSTOP)
                        frozen.append(inf["replica"])

            res = await self.ask(i, hook)
            for url in frozen:
                stand.replicas[url].sig(signal.SIGCONT)
            await asyncio.sleep(0.7 if a.dry else 1.5)  # поздние токены старой попытки
            ev, info = await self.e2e_event("stop", i, res, stop_at=stop_at, frozen=frozen)
            landed = (info.get("attempts") or 0) >= 2
            hit += landed
            ok += bool(ev["match"] and landed)
        if a.stops:
            print(f"[{name}] SIGSTOP: сбой попал в ответ {hit}/{a.stops}, совпало {ok}")

        ok = 0
        for j in range(a.drops):
            i = idx[(j + a.kills + a.stops) % len(idx)]
            await stand.all_up()
            n = self.controls[(i, "seed")]["content"]
            drop_at = max(2, min(30, n // 3))

            async def hook(res, drop_at=drop_at):
                return "drop" if res.last_pos >= drop_at else None

            res = await self.ask(i, hook)
            dropped_at = res.last_pos
            if res.dropped:
                await asyncio.sleep(0.2)
                url = f"{stand.router_url}/v1/streams/{res.answer_id}?from={res.last_pos}"
                try:
                    async with stand.http.get(url) as r:
                        await read_sse(r, res)
                except Exception as e:  # noqa: BLE001
                    res.error = res.error or f"{type(e).__name__}: {e}"
            ev, _ = await self.e2e_event("drop", i, res, dropped=res.dropped, dropped_at=dropped_at)
            ok += bool(ev["match"] and res.dropped)
        if a.drops:
            print(f"[{name}] обрыв клиента: {ok}/{a.drops} дочитаны и совпали")

        if a.batch:
            await stand.all_up()
            group = idx[:a.batch]
            victim = stand.replicas[stand.urls[0]]

            async def killer():
                await asyncio.sleep(a.batch_kill_delay)
                victim.sig(signal.SIGKILL)

            results, _ = await asyncio.gather(
                asyncio.gather(*(self.ask(i) for i in group)), killer())
            ok = moved = 0
            for i, res in zip(group, results):
                ev, info = await self.e2e_event("batch", i, res)
                ok += bool(ev["match"])
                moved += (info.get("attempts") or 0) >= 2
            print(f"[{name}] {len(group)} ответов разом: совпало {ok}, пережили смерть реплики {moved}")
        await stand.all_up()


# ---------- сводка ----------


def _pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "нет данных"


def summarize(events):
    out = {"stage": events[0]["stage"] if events else None, "configs": {}, "notes": []}
    for e in events:
        if e["kind"] == "note":
            out["notes"].append(e["text"])
    for cfg in dict.fromkeys(e.get("config") for e in events if e.get("config")):
        evs = [e for e in events if e.get("config") == cfg]
        c = {"reruns": {}, "e2e": {}, "perf": {}, "errors": 0, "pauses": {}, "load": []}
        for e in evs:
            if e["kind"] in ("cut", "aa", "ab", "control") and e.get("error"):
                c["errors"] += 1
            if e["kind"] in ("cut", "aa", "ab") and not e.get("error"):
                if e["kind"] == "cut":
                    label = "разрез, " + ("под нагрузкой" if e["load"] else "без нагрузки")
                else:
                    label = "повтор на той же реплике" if e["kind"] == "aa" else \
                        "повтор на другой реплике"
                key = f"{label} | кэш {e['cache']} | {e['mode']}"
                g = c["reruns"].setdefault(key, {
                    "n": 0, "matched": 0, "tokens": 0, "lp_checked": 0, "lp_bitwise": 0,
                    "max_dlp": 0.0, "first_diffs": []})
                g["n"] += 1
                g["matched"] += bool(e["match"])
                g["tokens"] += e["n_cmp"]
                if e["lp_cmp"]:
                    g["lp_checked"] += 1
                    g["lp_bitwise"] += e["lp_cmp"] == e["lp_equal"]
                g["max_dlp"] = max(g["max_dlp"], e["max_dlp"])
                if e["first_diff"] is not None:
                    g["first_diffs"].append(e["first_diff"])
            if e["kind"] == "e2e":
                g = c["e2e"].setdefault(e["sub"], {
                    "n": 0, "matched": 0, "landed": 0, "errors": 0, "mismatched": 0})
                landed = (e.get("attempts") or 0) >= 2
                g["n"] += 1
                g["landed"] += landed
                g["errors"] += bool(e.get("error"))
                g["mismatched"] += not e["match"]  # считаем и те, куда сбой не попал
                if e["sub"] in ("kill", "stop"):
                    g["matched"] += bool(e["match"] and landed)
                    if landed and e.get("pauses"):
                        c["pauses"].setdefault(e["sub"], []).extend(e["pauses"])
                else:
                    g["matched"] += bool(e["match"])
            if e["kind"] == "perf":
                c["perf"][str(e["conc"])] = {"tok_s": e["tok_s"], "ttft_med": e["ttft_med"]}
            if e["kind"] == "load":
                c["load"].append({"replica": e["replica"], "conc": e["conc"],
                                  "done": e["done"], "errors": e["errors"]})
        cuts = [g for k, g in c["reruns"].items() if k.startswith("разрез")]
        c["cuts_total"] = sum(g["n"] for g in cuts)
        c["cuts_matched"] = sum(g["matched"] for g in cuts)
        c["cut_tokens"] = sum(g["tokens"] for g in cuts)
        c["cuts_lp_checked"] = sum(g["lp_checked"] for g in cuts)
        c["cuts_lp_bitwise"] = sum(g["lp_bitwise"] for g in cuts)
        for sub, vals in list(c["pauses"].items()):
            vals = sorted(vals)
            c["pauses"][sub] = {
                "n": len(vals), "median": round(statistics.median(vals), 3),
                "p95": round(vals[min(len(vals) - 1, int(0.95 * len(vals)))], 3),
                "max": round(vals[-1], 3)}
        out["configs"][cfg] = c
    return out


def render(summary, env=None):
    lines = [f"# Этап {summary['stage']}", ""]
    if env:
        def cards(e):
            return ", ".join(f"{g['name']} (драйвер {g['driver']})" for g in e.get("gpus", []))

        if env.get("hosts"):
            for n, h in enumerate(env["hosts"]):
                lines.append(
                    f"Хост {'AB'[n] if n < 2 else n}: `{h.get('model')}`, ревизия "
                    f"`{h.get('revision')}`, vLLM {h.get('packages', {}).get('vllm')}, "
                    f"карта: {cards(h) or 'нет данных'}.")
            lines.append("")
        else:
            lines += [f"Модель: `{env.get('model')}`, ревизия `{env.get('revision')}`. "
                      f"vLLM {env.get('packages', {}).get('vllm')}. "
                      f"Карта: {cards(env) or 'нет данных'}.", ""]
    for name, c in summary["configs"].items():
        lines += [f"## Конфигурация {name}", ""]
        if c["cuts_total"]:
            lines += [
                f"Продолжений совпало: **{c['cuts_matched']} из {c['cuts_total']}** "
                f"({_pct(c['cuts_matched'], c['cuts_total'])}), сверено токенов: {c['cut_tokens']}. "
                f"Logprobs побитово равны в {c['cuts_lp_bitwise']} из {c['cuts_lp_checked']}.", ""]
        if c["reruns"]:
            lines += ["| Проверка | Кэш | Режим | Совпало | Токенов | Logprobs побитово | "
                      "Макс. разница logprob | Медиана позиции расхождения |",
                      "| --- | --- | --- | --- | --- | --- | --- | --- |"]
            for key, g in c["reruns"].items():
                label, cache, mode = [x.strip() for x in key.split("|")]
                med = statistics.median(g["first_diffs"]) if g["first_diffs"] else "нет"
                lines.append(
                    f"| {label} | {cache.replace('кэш ', '')} | {mode} | {g['matched']}/{g['n']} | "
                    f"{g['tokens']} | {g['lp_bitwise']}/{g['lp_checked']} | {g['max_dlp']:.3g} | {med} |")
            lines.append("")
        if c["e2e"]:
            names = {"control": "без сбоев", "kill": "kill -9", "stop": "SIGSTOP",
                     "drop": "обрыв клиента", "batch": "ответы разом, одна реплика убита"}
            lines += ["| Через роутер | Совпало | Сбой попал в ответ | Не совпало всего | Ошибок |",
                      "| --- | --- | --- | --- | --- |"]
            for sub, g in c["e2e"].items():
                base = g["landed"] if sub in ("kill", "stop") else g["n"]
                landed = f"{g['landed']}/{g['n']}" if sub in ("kill", "stop", "batch") else "нет"
                lines.append(f"| {names.get(sub, sub)} | {g['matched']}/{base} | {landed} | "
                             f"{g['mismatched']} | {g['errors']} |")
            lines.append("")
        for sub, p in c["pauses"].items():
            label = "kill -9" if sub == "kill" else "SIGSTOP"
            lines.append(f"Пауза после {label}: медиана {p['median'] * 1000:.0f} мс, "
                         f"95-й перцентиль {p['p95'] * 1000:.0f} мс, максимум {p['max'] * 1000:.0f} мс "
                         f"({p['n']} замеров).")
        if c["perf"]:
            parts = [f"{conc} поток(ов): {v['tok_s']} ток/с" for conc, v in c["perf"].items()]
            lines.append("Скорость первой реплики: " + ", ".join(parts) + ".")
        for ld in c["load"]:
            lines.append(f"Фоновая нагрузка на {ld['replica']}: {ld['conc']} потоков, выполнено "
                         f"запросов {ld['done']}, ошибок {ld['errors']}.")
        if c["errors"]:
            lines.append(f"Запросов с ошибкой: {c['errors']}.")
        lines.append("")
    if summary["notes"]:
        lines += ["## Замечания", ""] + [f"- {n}" for n in summary["notes"]] + [""]
    return "\n".join(lines)


def stage_failed(summary, stage):
    """Есть ли расхождения там, где совпадение ожидалось."""
    for cfg in STAGES[stage]:
        c = summary["configs"].get(cfg.name)
        if not c or not cfg.expect_match:
            continue
        for g in c["reruns"].values():
            if g["matched"] != g["n"] or g["lp_bitwise"] != g["lp_checked"]:
                return True
        if any(g["mismatched"] for g in c["e2e"].values()):
            return True
        if c["errors"]:
            return True
    return False


def report(root):
    """Четыре числа для поста по всем прогонам в папке."""
    runs = []
    for path in sorted(Path(root).glob("*/summary.json")):
        runs.append((path.parent.name, json.loads(path.read_text(encoding="utf-8"))))
    if not runs:
        print(f"в {root} нет прогонов")
        return 1
    lines = ["# Шаг 0: сводка по прогонам", ""]
    bi_perf = default_perf = None
    for name, s in runs:
        for cfg, c in s["configs"].items():
            if not c["cuts_total"]:
                continue
            lines.append(
                f"- `{name}`, {cfg}: продолжений совпало {c['cuts_matched']} из {c['cuts_total']} "
                f"({_pct(c['cuts_matched'], c['cuts_total'])}), logprobs побитово "
                f"{c['cuts_lp_bitwise']} из {c['cuts_lp_checked']}, токенов {c['cut_tokens']}")
            for sub, g in c["e2e"].items():
                if sub in ("kill", "stop"):
                    lines.append(f"    - через роутер, {sub}: {g['matched']} из {g['landed']}")
            for sub, p in c["pauses"].items():
                lines.append(f"    - пауза после {sub}: медиана {p['median'] * 1000:.0f} мс, "
                             f"максимум {p['max'] * 1000:.0f} мс")
            if c["perf"]:
                if s["stage"] == "0a" and cfg == "bi-cache-off":
                    bi_perf = c["perf"]
                if s["stage"] == "0c":
                    default_perf = c["perf"]
    if bi_perf and default_perf:
        lines += ["", "Цена батч-инвариантности: скорость первой реплики в 0a (bi-cache-off) "
                      "против 0c (default):"]
        for conc in bi_perf:
            a, b = bi_perf[conc]["tok_s"], default_perf.get(conc, {}).get("tok_s")
            if a and b:
                lines.append(f"- {conc} поток(ов): {a} ток/с с ней и {b} ток/с без неё, "
                             f"разница {100 * (a / b - 1):+.0f}%")
    text = "\n".join(lines) + "\n"
    (Path(root) / "REPORT.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


# ---------- запуск этапа ----------


async def run_stage(args):
    stage = args.stage
    rundir = Path(args.out) / f"{stage}-{time.strftime('%Y%m%d-%H%M%S')}"
    rundir.mkdir(parents=True, exist_ok=True)
    journal = Journal(rundir / "events.jsonl", stage)
    if not args.dry and not args.revision and not args.replica_urls:
        args.revision = resolve_revision(args.model)
    env = fingerprint(args)
    env["stage"] = stage
    env["args"] = {k: v for k, v in vars(args).items() if k not in ("api_key", "func")}
    for path in args.fingerprint_files or []:
        try:
            env.setdefault("hosts", []).append(json.loads(Path(path).read_text(encoding="utf-8")))
        except (OSError, ValueError) as e:
            journal.note(f"не прочитал {path}: {e}")
    hosts = env.get("hosts", [])
    if len(hosts) == 2:
        for label, pick in (
            ("карта", lambda h: [g["name"] for g in h.get("gpus", [])]),
            ("драйвер", lambda h: [g["driver"] for g in h.get("gpus", [])]),
            ("vLLM", lambda h: h.get("packages", {}).get("vllm")),
            ("torch", lambda h: h.get("packages", {}).get("torch")),
            ("ревизия модели", lambda h: h.get("revision")),
            ("флаги запуска", lambda h: h.get("serve_cmd")),
        ):
            x, y = pick(hosts[0]), pick(hosts[1])
            if x != y:
                journal.note(f"хосты различаются, {label}: {x} и {y}")
    (rundir / "env.json").write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Этап {stage}, журнал: {rundir}")

    started = time.monotonic()
    code = 0
    try:
        for cfg in STAGES[stage]:
            if args.only_config and cfg.name != args.only_config:
                continue
            stand = Stand(args, cfg, rundir, journal)
            try:
                print(f"[{cfg.name}] поднимаю реплики")
                await stand.up()
                bench = Bench(args, cfg, stand, journal)
                await bench.preflight()
                if cfg.perf and not stand.remote and "perf" in args.parts:
                    await bench.perf()
                if "matrix" in args.parts:
                    await bench.matrix()
                    await bench.repeats()
                if cfg.e2e and not stand.remote and "e2e" in args.parts:
                    await bench.e2e()
            finally:
                await stand.down()
    except Abort as e:
        journal.note(f"СТЕНД ОСТАНОВЛЕН: {e}")
        code = 2
    finally:
        summary = summarize(journal.events)
        (rundir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        text = render(summary, env)
        (rundir / "summary.md").write_text(text, encoding="utf-8")
        journal.f.close()
        bundle = rundir.with_suffix(".tar.gz")
        with tarfile.open(bundle, "w:gz") as tar:
            tar.add(rundir, arcname=rundir.name)
        print("\n" + text)
        print(f"Прогон занял {(time.monotonic() - started) / 60:.1f} мин. "
              f"Архив для отправки: {bundle}")
    if code == 0 and stage_failed(summary, stage):
        print("ИТОГ: есть расхождения или ошибки запросов там, где ждали совпадение. "
              "Дальше по этапам не идти.")
        code = 3
    elif code == 0:
        print("ИТОГ: этап пройден." if any(c.expect_match for c in STAGES[stage])
              else "ИТОГ: замер снят.")
    return code


def cmd_run(args):
    if args.dry:
        args.model, args.tokenizer = "fake-bytes", "bytes"
        if args.ports == [8001, 8002]:
            args.ports = [19011, 19012]
        if args.router_port == 8080:
            args.router_port = 18090
        args.restart_delay = 0.0
    if args.stage == "0b" and not args.replica_urls:
        print("для 0b нужны адреса двух реплик: --replica-urls http://A:порт,http://B:порт")
        return 2
    if args.stage == "0d" and args.model == DEFAULT_MODEL:
        print("для 0d укажи модель в NVFP4: --model <репозиторий>")
        return 2
    return asyncio.run(run_stage(args))


def cmd_serve(args):
    cfg = SERVE_CONFIGS[args.config]
    if not args.revision:
        args.revision = resolve_revision(args.model)
    cmd, env = serve_command(args, cfg, args.port, host=args.host, gpu_frac=args.gpu_frac)
    if args.api_key:
        env["VLLM_API_KEY"] = args.api_key  # ключ идёт через окружение, а не через командную строку
    fp = fingerprint(args)
    fp["serve_config"] = cfg.name
    fp["serve_cmd"] = cmd
    path = Path(args.fingerprint_out or f"serve-{platform.node()}.json")
    path.write_text(json.dumps(fp, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"сведения о хосте: {path}")
    print("запускаю:", " ".join(cmd))
    sys.stdout.flush()
    os.execvpe(cmd[0], cmd, env)


def cmd_fingerprint(args):
    print(json.dumps(fingerprint(args), ensure_ascii=False, indent=2))
    return 0


def cmd_summary(args):
    rundir = Path(args.rundir)
    events = [json.loads(line) for line in (rundir / "events.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    env = None
    if (rundir / "env.json").exists():
        env = json.loads((rundir / "env.json").read_text(encoding="utf-8"))
    print(render(summarize(events), env))
    return 0


DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Стенд шага 0")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--model", default=DEFAULT_MODEL)
        p.add_argument("--revision", default=None, help="ревизия модели; по умолчанию текущая")
        p.add_argument("--api-key", default=os.environ.get(KEY_ENV),
                       help=f"ключ реплик; по умолчанию из переменной {KEY_ENV}")
        p.add_argument("--max-model-len", type=int, default=8192)
        p.add_argument("--max-num-seqs", type=int, default=32)
        p.add_argument("--attention-backend", default=None,
                       help="по умолчанию FLASH_ATTN при батч-инвариантности; пустая строка "
                            "оставляет выбор vLLM")
        p.add_argument("--extra-serve-args", default="", help="добавка к команде vllm serve")

    r = sub.add_parser("run", help="прогнать этап")
    r.add_argument("stage", choices=sorted(STAGES))
    common(r)
    r.add_argument("--out", default="results")
    r.add_argument("--dry", action="store_true", help="фейковые реплики вместо vLLM")
    r.add_argument("--dry-drift", choices=["token", "logprob"], default=None,
                   help="вторая фейковая реплика нарочно портит продолжение")
    r.add_argument("--dry-tps", type=float, default=600.0)
    r.add_argument("--replica-urls", default=None,
                   help="две уже запущенные реплики через запятую (этап 0b)")
    r.add_argument("--fingerprint-files", nargs="*", default=None,
                   help="файлы serve-*.json с обоих хостов")
    r.add_argument("--ports", type=lambda s: [int(x) for x in s.split(",")], default=[8001, 8002])
    r.add_argument("--router-port", type=int, default=8080)
    r.add_argument("--tokenizer", choices=["hf", "bytes"], default="hf")
    r.add_argument("--gpu-frac", type=float, default=0.42,
                   help="доля памяти карты на одну реплику")
    r.add_argument("--n", type=int, default=50, help="сколько промптов")
    r.add_argument("--max-tokens", type=int, default=256)
    r.add_argument("--load", type=int, default=8, help="фоновых запросов; 0 выключает")
    r.add_argument("--repeats", type=int, default=10, help="промптов для повтора целиком")
    r.add_argument("--e2e-prompts", type=int, default=30)
    r.add_argument("--kills", type=int, default=None)
    r.add_argument("--stops", type=int, default=None)
    r.add_argument("--drops", type=int, default=5)
    r.add_argument("--batch", type=int, default=16, help="ответов разом; 0 выключает")
    r.add_argument("--batch-kill-delay", type=float, default=None)
    r.add_argument("--parts", type=lambda s: s.split(","), default=["perf", "matrix", "e2e"],
                   help="части прогона: perf, matrix, e2e")
    r.add_argument("--only-config", default=None)
    r.add_argument("--startup-timeout", type=float, default=1200.0)
    r.add_argument("--request-timeout", type=float, default=300.0)
    r.add_argument("--restart-delay", type=float, default=2.0,
                   help="пауза перед перезапуском убитой реплики, чтобы карта освободила память")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("serve", help="запустить одну реплику с флагами этапа")
    common(s)
    s.add_argument("--config", choices=sorted(SERVE_CONFIGS), default="bi-cache-off")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8001)
    s.add_argument("--gpu-frac", type=float, default=0.90)
    s.add_argument("--fingerprint-out", default=None)
    s.set_defaults(func=cmd_serve)

    f = sub.add_parser("fingerprint", help="сведения о машине в JSON")
    f.add_argument("--model", default=None)
    f.add_argument("--revision", default=None)
    f.set_defaults(func=cmd_fingerprint)

    m = sub.add_parser("summary", help="пересобрать сводку одного прогона")
    m.add_argument("rundir")
    m.set_defaults(func=cmd_summary)

    p = sub.add_parser("report", help="сводка по всем прогонам в папке")
    p.add_argument("root", nargs="?", default="results")
    p.set_defaults(func=lambda a: report(a.root))

    args = ap.parse_args(argv)
    if args.cmd == "run":
        live = args.stage in ("0a", "0d")
        if args.kills is None:
            args.kills = (4 if args.dry else 30) if live else (3 if args.dry else 10)
        if args.stops is None:
            args.stops = (2 if args.dry else 10) if live else (1 if args.dry else 3)
        if args.batch_kill_delay is None:
            args.batch_kill_delay = 0.15 if args.dry else 0.5
    return args


def _on_signal(signum, _frame):
    sys.exit(128 + signum)  # дальше сработают finally и atexit, группы процессов будут убиты


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGHUP, _on_signal)
    a = parse_args()
    sys.exit(a.func(a) or 0)
