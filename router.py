#!/usr/bin/env python3
"""Router: all state of an answer lives here, a replica only computes tokens.

The client sends a normal POST /v1/chat/completions. The router tokenizes the prompt,
sends token IDs to a replica and appends every new ID to the answer log.
If the replica dies or goes silent, the router sends prompt_ids + output_ids
to another replica, and the client only sees a pause.
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

# Finish reasons that mean the answer is complete. vLLM has five:
# stop, length, abort, error, repetition. abort and error mean the answer was
# cut short, and the router resumes it on another replica.
FINAL_REASONS = ("stop", "length", "repetition")

# Request fields the router does not forward to replicas. Dropping them silently would
# change the answer behind the client's back, so a field that is really set gets HTTP 400.
# The values listed here mean "not set" and pass.
NOT_FORWARDED = {
    "stop": (None, [], ""),
    "tools": (None, []),
    "tool_choice": (None, "none", "auto"),
    "functions": (None, []),
    "response_format": (None, {"type": "text"}),
    "structured_outputs": (None,),
    "guided_json": (None,),
    "guided_regex": (None,),
    "guided_choice": (None,),
    "guided_grammar": (None,),
    "presence_penalty": (None, 0),
    "frequency_penalty": (None, 0),
    "repetition_penalty": (None, 1),
    "top_k": (None, 0, -1),
    "min_p": (None, 0),
    "logit_bias": (None, {}),
    "min_tokens": (None, 0),
    "n": (None, 1),
    "logprobs": (None, False),
    "top_logprobs": (None, 0),
    "chat_template_kwargs": (None, {}),
}


@dataclass
class Event:
    pos: int  # how many answer tokens are covered after this event
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
    attempts: int = 0  # how many requests replicas accepted
    replicas: list = field(default_factory=list)
    pauses: list = field(default_factory=list)
    finish_reason: str | None = None
    error: str | None = None
    error_type: str = "upstream_failure"
    finished_at: float | None = None
    last_token_at: float | None = None
    pause_from: float | None = None
    auth_refused: set = field(default_factory=set)  # replicas that answered 401 or 403
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

    # ---------- replica selection ----------

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

    # ---------- answer log ----------

    async def _notify(self, a):
        async with a.cond:
            a.cond.notify_all()

    def _emit(self, a, final=None):
        # A token may end mid-character: hold the tail until the next token. If the answer
        # ends right there (max_tokens mid-character), the broken character is dropped:
        # otherwise the final event would carry text at an already delivered position, and
        # a client resuming from that position would lose it.
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

    # ---------- generation ----------

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
                        a, f"attempt limit: {a.attempts} replicas accepted the request and failed"
                    )
                # The deadline is checked on every turn, not only when there is nobody to pick.
                # Replicas that fail slowly (a frozen process, a connect timeout) leave their
                # cooldown before the next turn, so "nobody to pick" may never happen.
                if (
                    waiting_since is not None
                    and time.monotonic() - waiting_since > self.args.deadline
                ):
                    note = ""
                    if a.auth_refused:
                        note = (
                            f"; API key refused by {len(a.auth_refused)} "
                            f"of {len(self.replicas)} replicas"
                        )
                    return await self._fail(
                        a, f"no live replicas for more than {self.args.deadline:g} s{note}"
                    )
                replica = self.pick(avoid)
                if replica is None:
                    waiting_since = waiting_since or time.monotonic()
                    await asyncio.sleep(0.1)
                    continue
                t_try = time.monotonic()
                result = await self._attempt(a, replica)
                if result in ("finished", "rejected"):
                    return
                if result == "unauthorized":
                    # a setup problem, not a request problem: try the other replicas,
                    # and if no replica accepts the key there is no point waiting for the deadline
                    a.auth_refused.add(replica)
                    if len(a.auth_refused) >= len(self.replicas):
                        return await self._fail(
                            a,
                            "all replicas refused the API key (HTTP 401 or 403): "
                            "check --replica-api-key",
                        )
                self.down_until[replica] = time.monotonic() + self.args.cooldown
                avoid = replica
                if result == "failed":
                    waiting_since = None
                    self.stats["failovers"] += 1
                    if a.pause_from is None:
                        a.pause_from = a.last_token_at or a.t0
                else:
                    waiting_since = waiting_since or t_try
        except Exception as e:  # noqa: BLE001 - an answer must never hang silently
            if not a.done:
                await self._fail(a, f"internal router error: {e!r}")

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
        # vLLM 0.33.0 derives sampler noise from (seed, position, token id), so a request
        # that carries the already emitted tokens in the prompt draws the same noise.
        body["temperature"] = a.temperature
        body["top_p"] = a.top_p
        body["seed"] = a.seed
        return body

    async def _attempt(self, a, replica):
        """One request to a replica.

        Returns finished, failed, unavailable, rejected or unauthorized.
        """
        accepted = False
        resp = None
        # the first-token budget starts when the request is sent: connect + headers + token
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
                    # the request itself is bad: the replica is healthy, retrying is pointless
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
                    await self._fail(a, f"replica rejected the request: HTTP {resp.status}: {msg}")
                    return "rejected"
                return "unavailable"
            accepted = True
            a.attempts += 1
            a.replicas.append(replica)
            intervals = deque(maxlen=50)
            got_first = False
            last = None
            while True:
                # Silence is measured from the last token, not from the last line: a keep-alive
                # comment or a chunk without tokens must not restart the timer.
                if got_first:
                    budget, since = self._gap_timeout(intervals), last
                else:
                    budget, since = self.args.first_token_timeout, t_sent
                timeout = max(0.0, budget - (time.monotonic() - since))
                line = await asyncio.wait_for(resp.content.readline(), timeout)
                if not line:
                    return "failed"  # connection closed without a final chunk
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    return "failed"  # no final chunk
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
                        return "failed"  # abort and error: the answer is not finished, resume on another replica
                    await self._finish(a, ch["finish_reason"])
                    return "finished"
        except (asyncio.TimeoutError, OSError, ValueError, Exception):  # noqa: BLE001
            return "failed" if accepted else "unavailable"
        finally:
            if resp is not None:
                resp.close()  # late tokens of the old attempt will not arrive here

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
            pass  # the client dropped, generation goes on, the client may come back
        return resp

    async def chat(self, request):
        try:
            body = await request.json()
            messages = body["messages"]
            assert isinstance(messages, list) and messages
        except Exception:  # noqa: BLE001
            return web.json_response(
                {"error": {"message": "expected JSON with a messages field"}}, status=400
            )
        for key, unset in NOT_FORWARDED.items():
            if body.get(key) not in unset:
                return web.json_response(
                    {"error": {"message": f"field {key} is not supported in this version",
                               "type": "invalid_request_error"}},
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
                {"error": {"message": "answer not found or its log has expired"}}, status=404
            )
        # A cursor is the id of an event the client received: "<answer id>:<position>" or
        # the bare position. Anything else is a client bug, and guessing would lose or
        # repeat text, so it gets HTTP 400.
        raw = str(request.query.get("from") or request.headers.get("Last-Event-ID", "0"))
        owner, _, num = raw.rpartition(":")
        from_pos = int(num) if num.isascii() and num.isdigit() else -1
        known = from_pos == 0 or any(ev.pos == from_pos for ev in a.events)
        if from_pos < 0 or (owner and owner != a.id) or not known:
            return web.json_response(
                {"error": {"message": "bad cursor: expected the id of an event of this answer",
                           "type": "invalid_request_error"}},
                status=400,
            )
        return await self._sse(request, a, from_pos)

    async def answer_debug(self, request):
        a = self.answers.get(request.match_info["id"])
        if a is None:
            return web.json_response({"error": {"message": "answer not found"}}, status=404)
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
    p = argparse.ArgumentParser(description="Router that resumes an answer on another replica")
    p.add_argument("--replicas", required=True, help="replica URLs, comma-separated")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--tokenizer", choices=["hf", "bytes"], default="hf")
    p.add_argument("--host", default="127.0.0.1",
                   help="there is no client auth, so the default is localhost only")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--first-token-timeout", type=float, default=15.0)
    p.add_argument("--min-gap", type=float, default=1.0,
                   help="lower bound of the silence timer, seconds")
    p.add_argument("--warm-gap", type=float, default=5.0,
                   help="silence timer until five token intervals have been measured")
    p.add_argument("--connect-timeout", type=float, default=2.0)
    p.add_argument("--deadline", type=float, default=120.0,
                   help="how long to wait for a live replica, seconds")
    p.add_argument("--cooldown", type=float, default=3.0,
                   help="how long to leave a replica alone after a failure, seconds")
    p.add_argument("--max-attempts", type=int, default=3,
                   help="first attempt plus two resumes")
    p.add_argument("--log-ttl", type=float, default=60.0,
                   help="how long to keep the log after the answer ends, seconds")
    p.add_argument("--keepalive", type=float, default=2.0)
    p.add_argument("--default-max-tokens", type=int, default=512)
    p.add_argument("--replica-api-key", default=os.environ.get("UNDEAD_REPLICA_API_KEY"),
                   help="key for replicas started with --api-key; "
                        "defaults to the UNDEAD_REPLICA_API_KEY variable")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    web.run_app(Router(args).app(), host=args.host, port=args.port, print=None)
