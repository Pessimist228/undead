#!/usr/bin/env python3
"""Фейковая реплика с API как у vLLM: POST /v1/completions с prompt из ID токенов.

Работает в паре с токенизатором bytes. Следующий токен зависит только от всей
последовательности, как у настоящей модели: реплика находит последнюю метку
ассистента, по хешу промпта и сида выбирает текст и отдаёт его следующий байт.
Если роутер потеряет или продублирует хоть один токен, вместо текста пойдут
решётки, и тест это увидит.

Для стенда шага 0 реплика умеет ещё три вещи:
  logprobs       в чанках, как у vLLM: число зависит только от промпта, сида и позиции
  --api-key      запросы к /v1 без верного ключа получают 401
  --drift-*      нарочно портит продолжение, чтобы проверить, что стенд это ловит
"""
import argparse
import asyncio
import hashlib
import json
import os
import random

from aiohttp import web

from toks import ASSISTANT_MARK, ByteTokenizer

EOS = ByteTokenizer.EOS
MARK = ASSISTANT_MARK.encode()
SENTENCES = [
    "Нода умерла, а ответ продолжается.",
    "Роутер помнит каждый отданный токен.",
    "Реплика только считает и ничего не хранит.",
    "Дешёвая карта может исчезнуть в любую секунду.",
    "Клиент видит паузу, но не видит аварию.",
    "Крышка ноутбука захлопнулась посреди слова.",
    "Продолжение идёт с того же места, без дублей.",
    "Кириллица занимает два байта на букву.",
    "Хеш контрольного прогона должен совпасть.",
    "Планировщик считает, окупается ли переезд.",
    "Оператор поднимает новую реплику и гасит старую.",
    "Таймер тишины ловит зависшую ноду.",
]


def text_for(prompt: bytes, seed) -> bytes:
    digest = hashlib.sha256(prompt + b"|" + str(seed).encode()).digest()
    rng = random.Random(digest)
    return " ".join(rng.choice(SENTENCES) for _ in range(rng.randint(6, 14))).encode()


def _h(prompt: bytes, seed, pos: int) -> int:
    digest = hashlib.sha256(prompt + b"|" + str(seed).encode() + b"|" + str(pos).encode()).digest()
    return int.from_bytes(digest[:4], "big")


def logprob_for(prompt: bytes, seed, pos: int) -> float:
    """Число от -5 до 0, одинаковое для одной и той же точки ответа."""
    return -(_h(prompt, seed, pos) % 500000) / 100000.0


def split_sequence(ids):
    data = bytes(i for i in ids if i < 256)
    cut = data.rfind(MARK)
    if cut < 0:
        return data, b""
    cut += len(MARK)
    return data[:cut], data[cut:]


async def completions(request):
    args = request.app["args"]
    body = await request.json()
    prompt = body.get("prompt")
    if not isinstance(prompt, list) or not all(isinstance(i, int) for i in prompt):
        return web.json_response({"error": {"message": "prompt must be token ids"}}, status=400)
    xargs = body.get("vllm_xargs") or {}
    if "det_seed" in xargs:
        seed = xargs["det_seed"]
    elif body.get("temperature") == 0:
        seed = "greedy"  # жадный ответ от сида не зависит
    else:
        seed = body.get("seed", 0)
    max_tokens = int(body.get("max_tokens", 16))
    want_logprobs = body.get("logprobs") is not None
    head, done = split_sequence(prompt)
    text = text_for(head, seed)
    intact = text.startswith(done)
    poison = b"POISON" in head
    abort = b"ABORTME" in head and len(done) == 0
    repeat = b"REPEATME" in head
    resumed = len(done) > 0
    if b"TOOLONG" in head:
        return web.json_response(
            {"error": {"message": "This model's maximum context length is 32768 tokens. "
                                  "However, your request has 99999 input tokens.",
                       "type": "BadRequestError", "code": 400}},
            status=400,
        )

    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    rid = body.get("request_id", "cmpl-fake")
    pos = len(done)
    sent = 0
    first = True
    rng = random.Random(pos)
    try:
        if args.first_token_delay:
            await asyncio.sleep(args.first_token_delay)
        while True:
            ids = []
            lps = []
            finish = None
            for _ in range(rng.randint(1, 3)):
                if abort and sent >= 5:
                    finish = "abort"  # как сервер при остановке: 5 токенов и обрыв
                    break
                if repeat and pos >= 8:
                    finish = "repetition"  # vLLM сам остановил зациклившийся ответ
                    break
                if sent >= max_tokens:
                    finish = "length"
                    break
                if not intact:
                    tok = ord("#")
                elif pos >= len(text):
                    tok = EOS
                else:
                    tok = text[pos]
                    if (resumed and args.drift_token
                            and _h(head, seed, pos) % args.drift_token == 0):
                        tok = ord("~")  # нарочно другой токен в продолжении
                lp = logprob_for(head, seed, pos)
                if resumed and args.drift_logprob and pos % args.drift_logprob == 0:
                    lp += 1e-6  # тот же токен, но логит уже не побитово равен
                if tok == EOS:
                    finish = "stop"
                    if args.emit_eos_id:
                        ids.append(tok)
                        lps.append(lp)
                    break
                ids.append(tok)
                lps.append(lp)
                pos += 1
                sent += 1
                if not intact and sent >= 40:
                    finish = "stop"
                    break
            choice = {
                "index": 0,
                "text": bytes(i for i in ids if i < 256).decode("utf-8", errors="replace"),
                "token_ids": ids,
                "finish_reason": finish,
                "stop_reason": None,
            }
            if want_logprobs:
                choice["logprobs"] = {
                    "text_offset": [0] * len(ids),
                    "token_logprobs": lps,
                    "tokens": [f"token_id:{i}" for i in ids],
                    "top_logprobs": [{f"token_id:{i}": lp} for i, lp in zip(ids, lps)],
                }
            if first:
                choice["prompt_token_ids"] = prompt
                first = False
            chunk = {"id": rid, "object": "text_completion", "choices": [choice]}
            await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())
            if poison and sent >= 3:
                os._exit(1)  # ядовитый запрос убивает реплику
            if finish:
                break
            await asyncio.sleep(len(ids) / args.tps)
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
    except (ConnectionError, asyncio.CancelledError, RuntimeError):
        pass
    return resp


async def health(request):
    return web.json_response({"status": "ok", "name": request.app["args"].name})


async def models(request):
    return web.json_response({"data": [{"id": "fake-bytes"}]})


@web.middleware
async def auth(request, handler):
    """Как у vLLM: ключ нужен только для путей /v1, /health открыт."""
    key = request.app["args"].api_key
    if key and request.path.startswith("/v1"):
        if request.headers.get("Authorization") != f"Bearer {key}":
            return web.json_response({"error": "Unauthorized"}, status=401)
    return await handler(request)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--name", default="fake")
    p.add_argument("--tps", type=float, default=50.0, help="токенов в секунду")
    p.add_argument("--first-token-delay", type=float, default=0.0)
    p.add_argument("--no-eos-id", dest="emit_eos_id", action="store_false",
                   help="не класть ID конца ответа в token_ids")
    p.add_argument("--api-key", default=None, help="требовать этот ключ для /v1")
    p.add_argument("--drift-token", type=int, default=0,
                   help="в продолжении подменять примерно каждый N-й токен")
    p.add_argument("--drift-logprob", type=int, default=0,
                   help="в продолжении сдвигать logprob на каждой N-й позиции")
    args = p.parse_args()
    app = web.Application(middlewares=[auth])
    app["args"] = args
    app.router.add_post("/v1/completions", completions)
    app.router.add_get("/health", health)
    app.router.add_get("/v1/models", models)
    web.run_app(app, host="127.0.0.1", port=args.port, print=None)
