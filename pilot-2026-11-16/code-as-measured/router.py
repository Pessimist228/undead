#!/usr/bin/env python3
"""Роутер: вся память об ответе живёт здесь, реплика только считает токены.

Клиент шлёт обычный POST /v1/chat/completions. Роутер токенизирует сам,
отправляет на реплику ID токенов и пишет каждый новый ID в лог ответа.
Если реплика умерла или замолчала, роутер шлёт prompt_ids + output_ids
на другую реплику, и клиент видит только паузу.
"""
import argparse
import asyncio
import hashlib
import json
import os
import random
import statistics
import time
import uuid
from collections import deque
from dataclasses import dataclass, field

from aiohttp import ClientSession, ClientTimeout, web

from toks import make_tokenizer

# Причины финала, после которых ответ считается законченным. У vLLM их пять:
# stop, length, abort, error, repetition. abort и error означают, что ответ
# оборван, и роутер продолжает его на другой реплике.
FINAL_REASONS = ("stop", "length", "repetition")


@dataclass
class Event:
    pos: int  # сколько токенов ответа покрыто после этого события
    delta: str
    finish: str | None = None


@dataclass
class Answer:
    id: str
    prompt_ids: list
    max_tokens: int
    temperature: float
    top_p: float
    seed: int
    t0: float = field(default_factory=time.monotonic)
    created: int = field(default_factory=lambda: int(time.time()))
    output_ids: list = field(default_factory=list)
    events: list = field(default_factory=list)
    emitted_chars: int = 0
    attempts: int = 0  # сколько запросов реплики приняли
    replicas: list = field(default_factory=list)
    pauses: list = field(default_factory=list)
    finish_reason: str | None = None
    error: str | None = None
    error_type: str = "upstream_failure"
    finished_at: float | None = None
    last_token_at: float | None = None
    pause_from: float | None = None
    auth_refused: set = field(default_factory=set)  # реплики, ответившие 401 или 403
    cond: asyncio.Condition = field(default_factory=asyncio.Condition)

    @property
    def done(self):
        return self.finish_reason is not None or self.error is not None

    def debug(self, tok):
        return {
            "id": self.id,
            "done": self.done,
            "finish_reason": self.finish_reason,
            "error": self.error,
            "tokens": len(self.output_ids),
            "ids_sha256": hashlib.sha256(
                json.dumps(self.output_ids).encode()
            ).hexdigest(),
            "attempts": self.attempts,
            "replicas": self.replicas,
            "replica": self.replicas[-1] if self.replicas else None,
            "pauses": [round(p, 3) for p in self.pauses],
            "seed": self.seed,
            "text": tok.decode(self.output_ids),
        }


class Router:
    def __init__(self, args):
        self.args = args
        self.tok = make_tokenizer(args.tokenizer, args.model)
        self.stop_ids = list(self.tok.stop_ids)
        self.replicas = [u.strip().rstrip("/") for u in args.replicas.split(",") if u.strip()]
        self.down_until = {u: 0.0 for u in self.replicas}
        self.rr = 0
        self.answers = {}
        self.tasks = set()
        self.http = None
        self.stats = {"answers": 0, "finished": 0, "failed": 0, "failovers": 0}

    # ---------- выбор реплики ----------

    def pick(self, avoid=None):
        now = time.monotonic()
        n = len(self.replicas)
        alive = [
            self.replicas[(self.rr + i) % n]
            for i in range(n)
            if self.down_until[self.replicas[(self.rr + i) % n]] <= now
        ]
        if not alive:
            return None
        choice = next((u for u in alive if u != avoid), alive[0])
        self.rr = (self.replicas.index(choice) + 1) % n
        return choice

    # ---------- лог ответа ----------

    async def _notify(self, a):
        async with a.cond:
            a.cond.notify_all()

    def _emit(self, a, final=None):
        # Токен мог разрезать букву: хвост придержим до следующего токена. Если ответ
        # на этом кончился (max_tokens посреди буквы), обрезанную букву не отдаём:
        # иначе финальное событие несло бы текст на уже отданной позиции, и клиент,
        # дочитывающий с неё, этот хвост терял бы.
        text = self.tok.decode(a.output_ids).rstrip("�")
        delta = text[a.emitted_chars:] if len(text) > a.emitted_chars else ""
        if delta or final:
            a.events.append(Event(len(a.output_ids), delta, final))
            a.emitted_chars = max(a.emitted_chars, len(text))

    async def _finish(self, a, reason):
        self._emit(a, final=reason)
        a.finish_reason = reason
        a.finished_at = time.monotonic()
        self.stats["finished"] += 1
        await self._notify(a)

    async def _fail(self, a, message):
        a.error = message
        a.finished_at = time.monotonic()
        self.stats["failed"] += 1
        await self._notify(a)

    # ---------- генерация ----------

    async def drive(self, a):
        waiting_since = None
        avoid = None
        try:
            while True:
                if a.output_ids and a.output_ids[-1] in self.stop_ids:
                    return await self._finish(a, "stop")
                if len(a.output_ids) >= a.max_tokens:
                    return await self._finish(a, "length")
                if a.attempts >= self.args.max_attempts:
                    return await self._fail(
                        a, f"лимит попыток: {a.attempts} реплик приняли запрос и упали"
                    )
                replica = self.pick(avoid)
                if replica is None:
                    waiting_since = waiting_since or time.monotonic()
                    if time.monotonic() - waiting_since > self.args.deadline:
                        note = ""
                        if a.auth_refused:
                            note = (
                                f"; ключ API отклонили {len(a.auth_refused)} "
                                f"из {len(self.replicas)} реплик"
                            )
                        return await self._fail(
                            a, f"нет живых реплик дольше {self.args.deadline:g} с{note}"
                        )
                    await asyncio.sleep(0.1)
                    continue
                result = await self._attempt(a, replica)
                if result in ("finished", "rejected"):
                    return
                if result == "unauthorized":
                    # дело в настройке, а не в запросе: пробуем другие реплики,
                    # а если ключ не подошёл ни одной, ждать дедлайн незачем
                    a.auth_refused.add(replica)
                    if len(a.auth_refused) >= len(self.replicas):
                        return await self._fail(
                            a,
                            "все реплики отклонили ключ API (HTTP 401 или 403): "
                            "проверь --replica-api-key",
                        )
                self.down_until[replica] = time.monotonic() + self.args.cooldown
                avoid = replica
                if result == "failed":
                    waiting_since = None
                    self.stats["failovers"] += 1
                    if a.pause_from is None:
                        a.pause_from = a.last_token_at or a.t0
                else:
                    waiting_since = waiting_since or time.monotonic()
        except Exception as e:  # noqa: BLE001 - ответ не должен зависнуть молча
            if not a.done:
                await self._fail(a, f"внутренняя ошибка роутера: {e!r}")

    def _gap_timeout(self, intervals):
        if len(intervals) < 5:
            return self.args.warm_gap
        return max(self.args.min_gap, 10 * statistics.median(intervals))

    def _upstream_body(self, a):
        body = {
            "model": self.args.model,
            "prompt": a.prompt_ids + a.output_ids,
            "stream": True,
            "return_token_ids": True,
            "add_special_tokens": False,
            "max_tokens": a.max_tokens - len(a.output_ids),
            "stop_token_ids": self.stop_ids,
            "request_id": f"{a.id}-attempt-{a.attempts + 1}",
        }
        if self.args.det_sampler:
            # жребий считает логит-процессор det_sampler.py от сида и позиции
            body["temperature"] = 0
            body["vllm_xargs"] = {
                "det_seed": a.seed,
                "det_temperature": a.temperature,
                "det_top_p": a.top_p,
                "det_offset": len(a.output_ids),
            }
        else:
            body["temperature"] = a.temperature
            body["top_p"] = a.top_p
            body["seed"] = a.seed
        return body

    async def _attempt(self, a, replica):
        """Один запрос к реплике.

        Возвращает finished, failed, unavailable, rejected или unauthorized.
        """
        accepted = False
        resp = None
        # бюджет первого токена идёт с отправки запроса: коннект + заголовки + токен
        t_sent = time.monotonic()
        try:
            resp = await asyncio.wait_for(
                self.http.post(
                    replica + "/v1/completions",
                    json=self._upstream_body(a),
                    timeout=ClientTimeout(total=None, sock_connect=self.args.connect_timeout),
                ),
                self.args.first_token_timeout,
            )
            if resp.status != 200:
                if resp.status in (401, 403):
                    return "unauthorized"
                if 400 <= resp.status < 500 and resp.status not in (408, 429):
                    # запрос плохой сам по себе: реплика здорова, повторять нет смысла
                    raw = b""
                    while len(raw) < 2000:
                        part = await asyncio.wait_for(
                            resp.content.read(2000 - len(raw)),
                            max(0.0, self.args.first_token_timeout - (time.monotonic() - t_sent)),
                        )
                        if not part:
                            break
                        raw += part
                    text = raw.decode("utf-8", errors="replace").strip()
                    msg = text
                    try:
                        obj = json.loads(raw)
                        err = obj.get("error") if isinstance(obj, dict) else None
                        if isinstance(err, dict) and err.get("message"):
                            msg = str(err["message"])
                        elif isinstance(obj, dict) and obj.get("message"):
                            msg = str(obj["message"])
                    except ValueError:
                        pass
                    a.error_type = "invalid_request_error"
                    await self._fail(a, f"реплика отклонила запрос: HTTP {resp.status}: {msg}")
                    return "rejected"
                return "unavailable"
            accepted = True
            a.attempts += 1
            a.replicas.append(replica)
            intervals = deque(maxlen=50)
            got_first = False
            last = None
            while True:
                timeout = (
                    self._gap_timeout(intervals)
                    if got_first
                    else max(0.0, self.args.first_token_timeout - (time.monotonic() - t_sent))
                )
                line = await asyncio.wait_for(resp.content.readline(), timeout)
                if not line:
                    return "failed"  # соединение закрылось без финала
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    return "failed"  # финального чанка не было
                obj = json.loads(payload)
                if "error" in obj:
                    return "failed"
                choices = obj.get("choices") or []
                if not choices:
                    continue
                ch = choices[0]
                ids = ch.get("token_ids") or []
                ids = ids[: a.max_tokens - len(a.output_ids)]
                if ids:
                    now = time.monotonic()
                    if a.pause_from is not None:
                        a.pauses.append(now - a.pause_from)
                        a.pause_from = None
                    if last is not None:
                        intervals.append(now - last)
                    last = now
                    got_first = True
                    a.last_token_at = now
                    a.output_ids.extend(int(i) for i in ids)
                    self._emit(a)
                    await self._notify(a)
                if ch.get("finish_reason"):
                    if ch["finish_reason"] not in FINAL_REASONS:
                        return "failed"  # abort и error: ответ не закончен, продолжим на другой реплике
                    await self._finish(a, ch["finish_reason"])
                    return "finished"
        except (asyncio.TimeoutError, OSError, ValueError, Exception):  # noqa: BLE001
            return "failed" if accepted else "unavailable"
        finally:
            if resp is not None:
                resp.close()  # поздние токены старой попытки сюда уже не придут

    # ---------- HTTP ----------

    def _chunk(self, a, ev, first):
        delta = {"content": ev.delta}
        if first:
            delta["role"] = "assistant"
        obj = {
            "id": a.id,
            "object": "chat.completion.chunk",
            "created": a.created,
            "model": self.args.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": ev.finish}],
        }
        return f"id: {a.id}:{ev.pos}\ndata: {json.dumps(obj, ensure_ascii=False)}\n\n".encode()

    async def _sse(self, request, a, from_pos):
        resp = web.StreamResponse(
            headers={
                "Content-Type": "text/event-stream; charset=utf-8",
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Answer-Id": a.id,
            }
        )
        await resp.prepare(request)
        idx = 0
        first = True
        try:
            while True:
                while idx < len(a.events):
                    ev = a.events[idx]
                    idx += 1
                    if ev.pos > from_pos or ev.finish:
                        if ev.pos <= from_pos:
                            ev = Event(ev.pos, "", ev.finish)
                        await resp.write(self._chunk(a, ev, first))
                        first = False
                if a.done:
                    break
                async with a.cond:
                    try:
                        await asyncio.wait_for(
                            a.cond.wait_for(lambda: idx < len(a.events) or a.done),
                            self.args.keepalive,
                        )
                    except asyncio.TimeoutError:
                        await resp.write(b": keep-alive\n\n")
            if a.error:
                err = {"error": {"message": a.error, "type": a.error_type}}
                await resp.write(
                    f"event: error\ndata: {json.dumps(err, ensure_ascii=False)}\n\n".encode()
                )
            else:
                await resp.write(b"data: [DONE]\n\n")
            await resp.write_eof()
        except (ConnectionError, asyncio.CancelledError, RuntimeError):
            pass  # клиент отвалился, генерация идёт дальше, он может вернуться
        return resp

    async def chat(self, request):
        try:
            body = await request.json()
            messages = body["messages"]
            assert isinstance(messages, list) and messages
        except Exception:  # noqa: BLE001
            return web.json_response(
                {"error": {"message": "нужен JSON с полем messages"}}, status=400
            )
        for key in ("stop", "tools", "response_format"):
            if body.get(key):
                return web.json_response(
                    {"error": {"message": f"поле {key} в первой версии не поддерживается"}},
                    status=400,
                )
        seed = body.get("seed")
        a = Answer(
            id="ans-" + uuid.uuid4().hex[:12],
            prompt_ids=self.tok.encode_chat(messages),
            max_tokens=int(
                body.get("max_tokens")
                or body.get("max_completion_tokens")
                or self.args.default_max_tokens
            ),
            temperature=float(body.get("temperature", 1.0)),
            top_p=float(body.get("top_p", 1.0)),
            seed=int(seed) if seed is not None else random.getrandbits(31),
        )
        self.answers[a.id] = a
        self.stats["answers"] += 1
        task = asyncio.create_task(self.drive(a))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

        if body.get("stream"):
            return await self._sse(request, a, 0)

        async with a.cond:
            await a.cond.wait_for(lambda: a.done)
        if a.error:
            return web.json_response(
                {"error": {"message": a.error, "type": a.error_type}},
                status=400 if a.error_type == "invalid_request_error" else 502,
                headers={"X-Answer-Id": a.id},
            )
        return web.json_response(
            {
                "id": a.id,
                "object": "chat.completion",
                "created": a.created,
                "model": self.args.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": self.tok.decode(a.output_ids).rstrip("�"),
                        },
                        "finish_reason": a.finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": len(a.prompt_ids),
                    "completion_tokens": len(a.output_ids),
                    "total_tokens": len(a.prompt_ids) + len(a.output_ids),
                },
            },
            headers={"X-Answer-Id": a.id},
        )

    async def resume(self, request):
        a = self.answers.get(request.match_info["id"])
        if a is None:
            return web.json_response(
                {"error": {"message": "ответ не найден или лог уже удалён"}}, status=404
            )
        raw = request.query.get("from") or request.headers.get("Last-Event-ID", "0")
        try:
            from_pos = int(str(raw).rsplit(":", 1)[-1])
        except ValueError:
            from_pos = 0
        return await self._sse(request, a, from_pos)

    async def answer_debug(self, request):
        a = self.answers.get(request.match_info["id"])
        if a is None:
            return web.json_response({"error": {"message": "ответ не найден"}}, status=404)
        return web.json_response(a.debug(self.tok))

    async def metrics(self, request):
        now = time.monotonic()
        return web.json_response(
            {
                **self.stats,
                "replicas": {u: ("down" if t > now else "up") for u, t in self.down_until.items()},
                "active": [
                    {
                        "id": a.id,
                        "tokens": len(a.output_ids),
                        "replica": a.replicas[-1] if a.replicas else None,
                    }
                    for a in self.answers.values()
                    if not a.done
                ],
            }
        )

    async def health(self, request):
        return web.json_response({"status": "ok"})

    async def _gc(self):
        while True:
            await asyncio.sleep(5)
            now = time.monotonic()
            for aid in [
                k
                for k, a in self.answers.items()
                if a.finished_at and now - a.finished_at > self.args.log_ttl
            ]:
                del self.answers[aid]

    async def on_startup(self, app):
        headers = None
        if self.args.replica_api_key:
            headers = {"Authorization": f"Bearer {self.args.replica_api_key}"}
        self.http = ClientSession(read_bufsize=2**22, headers=headers)
        app["gc"] = asyncio.create_task(self._gc())

    async def on_cleanup(self, app):
        app["gc"].cancel()
        await self.http.close()

    def app(self):
        app = web.Application()
        app.on_startup.append(self.on_startup)
        app.on_cleanup.append(self.on_cleanup)
        app.router.add_post("/v1/chat/completions", self.chat)
        app.router.add_get("/v1/streams/{id}", self.resume)
        app.router.add_get("/v1/answers/{id}", self.answer_debug)
        app.router.add_get("/metrics", self.metrics)
        app.router.add_get("/health", self.health)
        return app


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Роутер, который продолжает ответ на другой реплике")
    p.add_argument("--replicas", required=True, help="адреса реплик через запятую")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--tokenizer", choices=["hf", "bytes"], default="hf")
    p.add_argument("--host", default="127.0.0.1",
                   help="авторизации клиентов нет, поэтому по умолчанию только локальный адрес")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--det-sampler", action="store_true",
                   help="реплики запущены с det_sampler.py, жребий считается от сида и позиции")
    p.add_argument("--first-token-timeout", type=float, default=15.0)
    p.add_argument("--min-gap", type=float, default=1.0,
                   help="нижняя граница таймера тишины, секунды")
    p.add_argument("--warm-gap", type=float, default=5.0,
                   help="таймер тишины, пока замерено меньше пяти интервалов")
    p.add_argument("--connect-timeout", type=float, default=2.0)
    p.add_argument("--deadline", type=float, default=120.0,
                   help="сколько ждать живую реплику, секунды")
    p.add_argument("--cooldown", type=float, default=3.0,
                   help="сколько не трогать реплику после сбоя, секунды")
    p.add_argument("--max-attempts", type=int, default=3,
                   help="первая попытка плюс два продолжения")
    p.add_argument("--log-ttl", type=float, default=60.0,
                   help="сколько хранить лог после финала, секунды")
    p.add_argument("--keepalive", type=float, default=2.0)
    p.add_argument("--default-max-tokens", type=int, default=512)
    p.add_argument("--replica-api-key", default=os.environ.get("UNDEAD_REPLICA_API_KEY"),
                   help="ключ для реплик, запущенных с --api-key; "
                        "по умолчанию из переменной UNDEAD_REPLICA_API_KEY")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    web.run_app(Router(args).app(), host=args.host, port=args.port, print=None)
