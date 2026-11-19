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
import hmac
import ipaddress
import json
import os
import random
import re
import statistics
import sys
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import urlsplit

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


LOOPBACK_NAMES = ("127.0.0.1", "localhost", "::1")


def host_name(value):
    """The name from a Host header or an origin: Example.com:8080 gives example.com."""
    v = value.strip().lower()
    if v.startswith("["):  # [::1]:8080
        return v[1:].partition("]")[0]
    if v.count(":") == 1:
        v = v.partition(":")[0]
    return v.rstrip(".")


def is_loopback(host):
    host = (host or "").lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def name_list(value):
    return [x.strip().lower().rstrip("/") for x in (value or "").split(",") if x.strip()]


NAME_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,62}")
REPLICA_FIELDS = ("url", "max_streams", "state", "api_key")


def check_replica_url(value):
    """(normalized url, None) or (None, why not). No userinfo: the URL is shown by the admin API."""
    if not isinstance(value, str) or not value or len(value) > 2048 or any(
        c <= " " or c == "\x7f" for c in value
    ):
        return None, "url must be a string without spaces"
    if "?" in value or "#" in value:
        return None, "url must not have a query or a fragment"
    try:
        u = urlsplit(value)
        u.port  # noqa: B018 - raises ValueError on a bad port
    except ValueError:
        return None, "url is not valid"
    if u.scheme not in ("http", "https") or not u.hostname:
        return None, "url must be http or https with a host"
    if "@" in u.netloc:
        return None, "url must not contain user or password"
    return value.rstrip("/"), None


def check_max_streams(value):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 1024:
        return "max_streams must be an integer from 0 (no limit) to 1024"
    return None


def check_api_key(value):
    if not isinstance(value, str) or not 1 <= len(value) <= 512 or any(
        not "!" <= c <= "~" for c in value
    ):
        return "api_key must be a string of 1 to 512 visible ASCII characters, or null"
    return None


class BrowserGuard:
    """Turns away requests that a web page sent, not a client the operator runs.

    The router has no client auth and listens on localhost by default. A page open in a
    browser on the same machine can reach localhost as well, in two ways. A cross-site
    request carries an Origin header that names the page. With DNS rebinding the page
    points its own name at 127.0.0.1, and then the Host header carries that name.
    """

    def __init__(self, bind_host, allowed_hosts=None, allowed_origins=None):
        hosts = name_list(allowed_hosts)
        # the Host check is on for a router on localhost and whenever names are listed
        self.any_host = "*" in hosts or not (hosts or is_loopback(bind_host))
        self.hosts = {*LOOPBACK_NAMES, *(host_name(h) for h in hosts)}
        self.origins = set(name_list(allowed_origins))
        self.any_origin = "*" in self.origins

    def reason(self, host, origin):
        """Why the request is refused, or None when it may pass."""
        if host is not None and not self.any_host and host_name(host) not in self.hosts:
            return (f"Host {host!r} is not a localhost name. Start the router with "
                    "--allowed-hosts to accept it.")
        if origin is not None and not self.any_origin:
            o = origin.strip().lower().rstrip("/")
            where = o.partition("://")[2]
            same = host is not None and where == host.strip().lower()
            if not (same or host_name(where) in LOOPBACK_NAMES or o in self.origins):
                return (f"request from the web page {origin!r} is refused. Start the router "
                        "with --allowed-origins to accept it.")
        return None


@dataclass
class Replica:
    name: str
    url: str
    max_streams: int = 0  # 0: no limit
    state: str = "active"  # active takes new requests, draining only finishes what it has
    api_key: str | None = None  # None: use --replica-api-key
    in_flight: int = 0
    down_until: float = 0.0  # cooldown, time.monotonic()
    started: int = 0  # streams that began on it
    resumed_in: int = 0  # streams that were resumed onto it
    last_error: str | None = None
    removed: bool = False
    epoch: int = 0  # changes with the key and the url: a refusal of the old key says nothing now
    static_url: str | None = None  # the url it got from --replicas, if any
    live: dict = field(default_factory=dict)  # answer id -> (answer, response) of open streams

    def view(self, now):
        return {
            "name": self.name,
            "url": self.url,
            "state": self.state,
            "max_streams": self.max_streams,  # 0: no limit
            "in_flight": self.in_flight,
            "cooldown_seconds_left": round(max(0.0, self.down_until - now), 3),
            "api_key_set": self.api_key is not None,
            "started": self.started,
            "resumed_in": self.resumed_in,
            "last_error": self.last_error,
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
    auth_refused: dict = field(default_factory=dict)  # name -> epoch of replicas that said 401/403
    # journal
    req_t0: float = field(default_factory=time.monotonic)  # request arrival, before the queue
    started_wall: float = field(default_factory=time.time)
    queued: float = 0.0
    stream: bool = False
    first_token_at: float | None = None
    replica_names: list = field(default_factory=list)
    resumes: list = field(default_factory=list)
    fail_reason: str | None = None  # why the current attempt failed
    resume_from: str | None = None  # name of the replica that failed last, with its reason
    resume_reason: str | None = None
    readers: set = field(default_factory=set)  # tokens of the requests reading this answer now
    end_delivered: bool = False
    journaled: bool = False
    shed: bool = False  # no slot within the queue budget: the client gets HTTP 503
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
            "replica_names": self.replica_names,
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
        urls = [u.strip().rstrip("/") for u in args.replicas.split(",") if u.strip()]
        self.registry = {
            f"r{i}": Replica(f"r{i}", u, max_streams=args.max_streams, epoch=i + 1, static_url=u)
            for i, u in enumerate(urls)
        }
        self.epochs = iter(range(len(urls) + 1, 2**62))
        self.rr = 0
        self.slot_event = asyncio.Event()  # replaced and set whenever a slot may have opened
        self.queue = deque()  # tickets of requests waiting for a slot, first in first out
        self.journal = open(args.journal, "a", encoding="utf-8") if args.journal else None
        self.admin_token = args.admin_token.encode() if getattr(args, "admin_token", None) else None
        self.admin_guard = BrowserGuard(args.admin_host, args.allowed_hosts, None)
        self.admin_runner = None
        self.answers = {}
        self.tasks = set()
        self.http = None
        self.stats = {"answers": 0, "finished": 0, "failed": 0, "failovers": 0}
        self.guard = BrowserGuard(args.host, args.allowed_hosts, args.allowed_origins)

    # ---------- replica selection ----------

    def _in_order(self):
        """Replicas in round-robin order, starting at self.rr."""
        reps = list(self.registry.values())
        k = self.rr % len(reps) if reps else 0
        return reps[k:] + reps[:k]

    def _take(self, rep):
        """Reserve a stream on rep. Every _take is matched by one _release."""
        rep.in_flight += 1
        self.rr = (list(self.registry.values()).index(rep) + 1) % len(self.registry)

    def _release(self, rep):
        rep.in_flight = max(0, rep.in_flight - 1)
        self._wake()

    def _wake(self):
        ev, self.slot_event = self.slot_event, asyncio.Event()
        ev.set()

    def pick_new(self, avoid=None):
        """A replica for a new request: active, not cooling down, below max_streams, the
        fewest streams in flight (round-robin order breaks ties). Reserves a stream."""
        now = time.monotonic()
        fit = [r for r in self._in_order()
               if r.state == "active" and r.down_until <= now
               and (not r.max_streams or r.in_flight < r.max_streams)]
        if not fit:
            return None
        choice = min(fit, key=lambda r: (r is avoid, r.in_flight))
        self._take(choice)
        return choice

    def pick(self, avoid=None):
        """A replica for a stream that has started: never waits, may exceed max_streams.
        Active replicas first, then draining ones; the replica that just failed comes last.
        Reserves a stream."""
        now = time.monotonic()
        alive = [r for r in self._in_order() if r.down_until <= now]
        if not alive:
            return None
        choice = min(alive, key=lambda r: ((r is avoid) * 2 + (r.state != "active"), r.in_flight))
        self._take(choice)
        return choice

    async def _slot_wait(self, limit):
        try:  # a cooldown ends without any event, so the caller looks again after limit
            await asyncio.wait_for(self.slot_event.wait(), limit)
        except asyncio.TimeoutError:
            pass

    @staticmethod
    def _gone(request):
        tr = request.transport
        return tr is None or tr.is_closing()

    async def acquire(self, request):
        """Wait up to --queue-timeout for a slot for a new request, first come first served.
        Returns None on timeout, and also when the client has left: nobody wants that answer."""
        ticket = object()
        self.queue.append(ticket)
        end = time.monotonic() + max(0.0, self.args.queue_timeout)
        try:
            while True:
                if self._gone(request):
                    return None
                if self.queue[0] is ticket:
                    rep = self.pick_new()
                    if rep is not None:
                        return rep
                left = end - time.monotonic()
                if left <= 0:
                    return None
                await self._slot_wait(min(left, 0.1))
        finally:
            self.queue.remove(ticket)
            self._wake()  # the next in line may fit as well

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
        self._maybe_journal(a)

    async def _fail(self, a, message):
        a.error = message
        a.finished_at = time.monotonic()
        self.stats["failed"] += 1
        await self._notify(a)
        self._maybe_journal(a)

    # ---------- journal ----------

    def _journal_write(self, rec):
        if self.journal is None:
            return
        try:
            self.journal.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.journal.flush()
        except (OSError, ValueError) as e:
            print(f"journal: cannot write: {e}", file=sys.stderr)

    @staticmethod
    def _iso(t):
        return datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="milliseconds").replace(
            "+00:00", "Z")

    def _journal_record(self, rid, started_wall, duration, queued, finish, status, a=None,
                        prompt_tokens=None):
        """One line. Only counts, times and replica names: no prompt, no text, no token IDs."""
        ms = lambda x: round(x * 1000)  # noqa: E731
        return {
            "id": rid,
            "start": self._iso(started_wall),
            "end": self._iso(started_wall + duration),
            "model": self.args.model,
            "prompt_tokens": len(a.prompt_ids) if a else prompt_tokens,
            "output_tokens": len(a.output_ids) if a else 0,
            "ttft_ms": ms(a.first_token_at - a.req_t0) if a and a.first_token_at else None,
            "duration_ms": ms(duration),
            "queued_ms": ms(queued),
            "replicas": list(a.replica_names) if a else [],
            "resumes": list(a.resumes) if a else [],
            "finish": finish,
            "status": status,
        }

    def _maybe_journal(self, a):
        """Writes the line once the answer has ended and nobody is reading it any more: the
        last reader may still be sending the end of the stream when the answer ends."""
        if not a.done or a.journaled or a.readers:
            return
        a.journaled = True
        if a.shed:
            finish = "rejected_503"
        elif a.error:
            finish = "error"
        elif not a.end_delivered:
            finish = "client_disconnect"  # the client left and did not get the end
        else:
            finish = "stop" if a.finish_reason == "repetition" else a.finish_reason
        if a.shed:
            status = 503
        elif a.stream or not a.error:
            status = 200
        else:
            status = 400 if a.error_type == "invalid_request_error" else 502
        self._journal_write(self._journal_record(
            a.id, a.started_wall, a.finished_at - a.req_t0, a.queued, finish, status, a))

    # ---------- generation ----------

    async def drive(self, a, held=None):
        """held is the replica reserved for the first attempt by acquire()."""
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
                    if self._refused(a):
                        note = (
                            f"; API key refused by {self._refused(a)} "
                            f"of {len(self.registry)} replicas"
                        )
                    return await self._fail(
                        a, f"no live replicas for more than {self.args.deadline:g} s{note}"
                    )
                # until a token is out the request is still a new one: it takes a free slot of an
                # active replica or waits for one, and when the queue budget is gone it is shed.
                # Only a stream that has emitted tokens uses the resume rules.
                if held is not None:
                    replica, held = held, None
                elif a.output_ids:
                    replica = self.pick(avoid)
                else:
                    replica = self.pick_new(avoid)
                if replica is None:
                    if not a.output_ids and a.queued >= self.args.queue_timeout:
                        a.shed, a.error_type = True, "service_unavailable"
                        return await self._fail(
                            a, f"all replicas are busy or down: no free slot after "
                               f"{self.args.queue_timeout:g} s")
                    waiting_since = waiting_since or time.monotonic()
                    t_wait = time.monotonic()
                    await self._slot_wait(0.1)
                    if not a.output_ids:
                        a.queued += time.monotonic() - t_wait
                    continue
                t_try = time.monotonic()
                result = await self._attempt(a, replica)
                if result in ("finished", "rejected"):
                    return
                if result == "unauthorized":
                    # a setup problem, not a request problem: try the other replicas,
                    # and if no replica accepts the key there is no point waiting for the deadline
                    a.auth_refused[replica.name] = replica.epoch
                    if self._refused(a) >= len(self.registry):
                        return await self._fail(
                            a,
                            "all replicas refused the API key (HTTP 401 or 403): "
                            "check --replica-api-key",
                        )
                replica.down_until = time.monotonic() + self.args.cooldown
                avoid = replica
                if result == "failed":
                    a.resume_from = replica.name
                    a.resume_reason = a.fail_reason or "failed"
                    waiting_since = None
                    self.stats["failovers"] += 1
                    if a.pause_from is None:
                        a.pause_from = a.last_token_at or a.t0
                else:
                    waiting_since = waiting_since or t_try
        except Exception as e:  # noqa: BLE001 - an answer must never hang silently
            if not a.done:
                await self._fail(a, f"internal router error: {e!r}")
        finally:
            if held is not None:  # the answer ended before the reserved replica was used
                self._release(held)

    def _refused(self, a):
        """How many replicas in the registry now still refused the key they have now."""
        return sum(1 for n, e in a.auth_refused.items()
                   if n in self.registry and self.registry[n].epoch == e)

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

    def _auth(self, rep):
        key = rep.api_key or self.args.replica_api_key
        return {"Authorization": f"Bearer {key}"} if key else None

    def _note_error(self, rep, text):
        """Short text for the admin API. A key never belongs in it, so cut it out."""
        for key in (rep.api_key, self.args.replica_api_key):
            if key:
                text = text.replace(key, "***")
        rep.last_error = text[:200]

    def _failed(self, a, rep, reason, detail=None):
        """The stream broke after the replica accepted it. The reason is kept if the replica was
        deleted: that sets it first, and the connection error that follows is only its effect."""
        a.fail_reason = a.fail_reason or reason
        self._note_error(rep, detail or a.fail_reason)
        return "failed"

    async def _attempt(self, a, rep):
        """One request to a replica. The stream reserved on rep is released here.

        Returns finished, failed, unavailable, rejected or unauthorized.
        """
        replica = rep.url
        accepted = False
        resp = None
        # the first-token budget starts when the request is sent: connect + headers + token
        t_sent = time.monotonic()
        try:
            resp = await asyncio.wait_for(
                self.http.post(
                    replica + "/v1/completions",
                    json=self._upstream_body(a),
                    headers=self._auth(rep),
                    timeout=ClientTimeout(total=None, sock_connect=self.args.connect_timeout),
                ),
                self.args.first_token_timeout,
            )
            if rep.removed:  # deleted while the request was on its way: try another one
                return "unavailable"
            if resp.status != 200:
                self._note_error(rep, f"HTTP {resp.status}")
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
            a.replica_names.append(rep.name)
            a.fail_reason = None
            rep.live[a.id] = (a, resp)
            if a.output_ids:
                rep.resumed_in += 1
            else:
                rep.started += 1
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
                    return self._failed(a, rep, "connection_closed")  # no final chunk
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    return self._failed(a, rep, "replica_error")  # no final chunk
                obj = json.loads(payload)
                if "error" in obj:
                    return self._failed(a, rep, "replica_error")
                choices = obj.get("choices") or []
                if not choices:
                    continue
                ch = choices[0]
                ids = ch.get("token_ids") or []
                ids = ids[: a.max_tokens - len(a.output_ids)]
                if ids:
                    now = time.monotonic()
                    if a.first_token_at is None:
                        a.first_token_at = now
                    if a.pause_from is not None:
                        a.pauses.append(now - a.pause_from)
                        a.resumes.append({
                            "pos": len(a.output_ids), "pause_ms": round((now - a.pause_from) * 1000),
                            "from": a.resume_from, "to": rep.name, "reason": a.resume_reason,
                        })
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
                        # abort and error: the answer is not finished, resume on another replica
                        return self._failed(a, rep, "replica_error")
                    await self._finish(a, ch["finish_reason"])
                    return "finished"
        except (asyncio.TimeoutError, OSError, ValueError, Exception) as e:  # noqa: BLE001
            if not accepted:
                self._note_error(rep, f"{type(e).__name__}: {e}")
                return "unavailable"
            return self._failed(a, rep, "silence_timeout" if isinstance(e, asyncio.TimeoutError)
                                else "connection_error", f"{type(e).__name__}: {e}")
        finally:
            rep.live.pop(a.id, None)
            self._release(rep)
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

    async def _sse(self, request, a, from_pos, me=None):
        """me: the reader token, registered by the caller when it was created before this point
        (the journal waits for the last reader, so a stream must not look unread while it starts)."""
        me = me or object()
        a.readers.add(me)
        try:
            return await self._sse_body(request, a, from_pos)
        finally:
            a.readers.discard(me)
            self._maybe_journal(a)

    async def _sse_body(self, request, a, from_pos):
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
            a.end_delivered = True
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
        rid = "ans-" + uuid.uuid4().hex[:12]
        t_req, wall_req = time.monotonic(), time.time()
        prompt_ids = self.tok.encode_chat(messages)
        max_tokens = int(
            body.get("max_tokens") or body.get("max_completion_tokens") or self.args.default_max_tokens
        )
        temperature = float(body.get("temperature", 1.0))
        top_p = float(body.get("top_p", 1.0))
        seed = int(seed) if seed is not None else random.getrandbits(31)
        t_wait = time.monotonic()
        rep = await self.acquire(request)
        queued = time.monotonic() - t_wait
        if rep is not None and self._gone(request):
            self._release(rep)  # the client left while the slot was being handed over
            rep = None
        if rep is None:
            gone = self._gone(request)
            self._journal_write(self._journal_record(
                rid, wall_req, time.monotonic() - t_req, queued,
                "client_disconnect" if gone else "rejected_503", 499 if gone else 503,
                prompt_tokens=len(prompt_ids)))
            return self._shed_response(f"{self.args.queue_timeout:g}")
        a = Answer(
            id=rid, prompt_ids=prompt_ids, max_tokens=max_tokens, temperature=temperature,
            top_p=top_p, seed=seed, req_t0=t_req, started_wall=wall_req, queued=queued,
            stream=bool(body.get("stream")),
        )
        me = object()  # registered before anything can await: the journal waits for readers
        a.readers.add(me)
        self.answers[a.id] = a
        self.stats["answers"] += 1
        task = asyncio.create_task(self.drive(a, rep))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

        try:
            async with a.cond:
                # a request that fails before its first token may still be answered with 503,
                # so a stream starts only once there is something to send
                await a.cond.wait_for(lambda: a.done or (a.stream and a.events))
            if a.shed:
                return self._shed_response(f"{self.args.queue_timeout:g}")
            if a.stream:
                return await self._sse(request, a, 0, me)
            tr = request.transport  # None or closing: the client has gone
            a.end_delivered = tr is not None and not tr.is_closing()
        finally:
            a.readers.discard(me)
            self._maybe_journal(a)
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

    @staticmethod
    def _shed_response(seconds):
        return web.json_response(
            {"error": {"message": f"all replicas are busy or down: no free slot after {seconds} s",
                       "type": "service_unavailable"}},
            status=503, headers={"Retry-After": "1"},
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
                # the unauthenticated endpoint names only the urls given on the command line
                "replicas": {(r.url if r.url == r.static_url else r.name):
                             ("down" if r.down_until > now else "up")
                             for r in self.registry.values()},
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

    # ---------- admin API ----------

    @staticmethod
    def _bad(message, status=400):
        return web.json_response({"error": {"message": message, "type": "invalid_request_error"}},
                                 status=status)

    def admin_app(self):
        @web.middleware
        async def gate(request, handler):
            why = self.admin_guard.reason(request.headers.get("Host"), request.headers.get("Origin"))
            if why:
                return web.json_response(
                    {"error": {"message": why, "type": "forbidden"}}, status=403)
            scheme, _, token = request.headers.get("Authorization", "").partition(" ")
            if scheme.lower() != "bearer" or not hmac.compare_digest(
                    token.strip().encode(), self.admin_token):
                return web.json_response(
                    {"error": {"message": "missing or wrong admin token", "type": "unauthorized"}},
                    status=401, headers={"WWW-Authenticate": "Bearer"})
            return await handler(request)

        app = web.Application(middlewares=[gate])
        app.router.add_get("/admin/replicas", self.admin_list)
        app.router.add_put("/admin/replicas/{name}", self.admin_put)
        app.router.add_delete("/admin/replicas/{name}", self.admin_delete)
        return app

    async def admin_list(self, request):
        now = time.monotonic()
        return web.json_response([r.view(now) for r in self.registry.values()])

    async def admin_put(self, request):
        name = request.match_info["name"]
        if not NAME_RE.fullmatch(name):
            return self._bad("name must match [a-z0-9][a-z0-9-]{0,62}")
        try:
            body = await request.json()
        except ValueError:
            return self._bad("expected a JSON object")
        if not isinstance(body, dict):
            return self._bad("expected a JSON object")
        extra = sorted(set(body) - set(REPLICA_FIELDS))
        if extra:
            return self._bad(f"unknown fields: {', '.join(map(str, extra))}")
        rep = self.registry.get(name)
        if rep is None and "url" not in body:
            return self._bad("url is required for a replica that does not exist")
        new = {}
        if "url" in body:
            new["url"], why = check_replica_url(body["url"])
            if why:
                return self._bad(why)
        if "max_streams" in body:
            why = check_max_streams(body["max_streams"])
            if why:
                return self._bad(why)
            new["max_streams"] = body["max_streams"]
        if "state" in body:
            if body["state"] not in ("active", "draining"):
                return self._bad('state must be "active" or "draining"')
            new["state"] = body["state"]
        if "api_key" in body:  # null goes back to --replica-api-key
            why = None if body["api_key"] is None else check_api_key(body["api_key"])
            if why:
                return self._bad(why)
            new["api_key"] = body["api_key"]
        created = rep is None
        if created:
            rep = self.registry[name] = Replica(name, epoch=next(self.epochs), **new)
        else:
            if new.get("url", rep.url) != rep.url:
                rep.down_until = 0.0  # another host: the old cooldown says nothing about it
            if new.get("url", rep.url) != rep.url or new.get("api_key", rep.api_key) != rep.api_key:
                rep.epoch = next(self.epochs)
            for k, v in new.items():
                setattr(rep, k, v)
        self._wake()
        return web.json_response(rep.view(time.monotonic()), status=201 if created else 200)

    async def admin_delete(self, request):
        name = request.match_info["name"]
        if not NAME_RE.fullmatch(name):
            return self._bad("name must match [a-z0-9][a-z0-9-]{0,62}")
        rep = self.registry.pop(name, None)
        cut = 0
        if rep is not None:
            rep.removed = True
            # the streams on it go the way of a dead replica: their connection closes, and the
            # answer resumes on another replica
            for a, resp in list(rep.live.values()):
                a.fail_reason = "replica_removed"
                resp.close()
                cut += 1
        self._wake()
        return web.json_response({"name": name, "removed": rep is not None,
                                  "interrupted_streams": cut})

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
        self.http = ClientSession(read_bufsize=2**22)  # replica keys go per request
        app["gc"] = asyncio.create_task(self._gc())
        if self.args.admin_port:
            self.admin_runner = web.AppRunner(self.admin_app())
            await self.admin_runner.setup()
            await web.TCPSite(self.admin_runner, self.args.admin_host, self.args.admin_port).start()

    async def on_cleanup(self, app):
        app["gc"].cancel()
        if self.admin_runner:
            await self.admin_runner.cleanup()
        await self.http.close()
        if self.journal:
            self.journal.close()

    def app(self):
        @web.middleware
        async def guard(request, handler):
            why = self.guard.reason(request.headers.get("Host"), request.headers.get("Origin"))
            if why:
                return web.json_response(
                    {"error": {"message": why, "type": "forbidden"}}, status=403
                )
            return await handler(request)

        app = web.Application(middlewares=[guard])
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
    p.add_argument("--replicas", default="",
                   help="replica URLs, comma-separated; may be left out only with --admin-port")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--tokenizer", choices=["hf", "bytes"], default="hf")
    p.add_argument("--host", default="127.0.0.1",
                   help="there is no client auth, so the default is localhost only")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--allowed-hosts", default=None,
                   help="extra names to accept in the Host header, comma-separated, * for any; "
                        "a router on localhost accepts only localhost names by default")
    p.add_argument("--allowed-origins", default=None,
                   help="web page origins that may call the router, comma-separated, * for any; "
                        "by default only the router's own address and localhost pages may")
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
    p.add_argument("--max-streams", type=int, default=0,
                   help="streams per replica from --replicas that new requests may use, "
                        "0 means no limit")
    p.add_argument("--queue-timeout", type=float, default=10.0,
                   help="how long a new request waits for a free slot before HTTP 503, seconds")
    p.add_argument("--journal", default=None,
                   help="append one JSON line per finished request to this file")
    p.add_argument("--admin-port", type=int, default=None,
                   help="port of the admin API; off unless given, needs --admin-token-file")
    p.add_argument("--admin-host", default="127.0.0.1")
    p.add_argument("--admin-token-file", default=None,
                   help="file with the bearer token of the admin API, read once at start")
    args = p.parse_args(argv)
    args.admin_token = None
    if not args.admin_port and not any(u.strip() for u in args.replicas.split(",")):
        p.error("--replicas is required unless --admin-port is given")
    if not 0 <= args.max_streams <= 1024:
        p.error("--max-streams must be from 0 (no limit) to 1024")
    for i, url in enumerate(u.strip() for u in args.replicas.split(",") if u.strip()):
        if check_replica_url(url)[1]:
            p.error(f"--replicas entry {i + 1} is not an http(s) URL with a host and no password")
    if args.admin_token_file and not args.admin_port:
        print("router: --admin-token-file is ignored without --admin-port", file=sys.stderr)
    if args.admin_port:
        if not args.admin_token_file:
            p.error("--admin-port needs --admin-token-file")
        try:
            with open(args.admin_token_file, encoding="utf-8") as f:
                args.admin_token = f.read().strip()
        except (OSError, ValueError) as e:
            p.error(f"cannot read --admin-token-file: {e.__class__.__name__}")
        if not args.admin_token:
            p.error("--admin-token-file is empty")
    return args


if __name__ == "__main__":
    args = parse_args()
    try:
        router = Router(args)
    except OSError as e:  # the journal file
        sys.exit(f"router: {e}")
    web.run_app(router.app(), host=args.host, port=args.port, print=None)
