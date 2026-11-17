#!/usr/bin/env python3
"""Быстрая проверка живой реплики vLLM: две минуты до большого прогона step0.py.

    python check_vllm.py --url http://127.0.0.1:8001
    python check_vllm.py --url http://A:8001 --url2 http://B:8001 --api-key KEY

Что проверяется:
  1. prompt из списка ID токенов, token_ids и logprobs в чанках стрима
  2. продолжение с середины при жадном выборе даёт те же ID, что и контроль
  3. то же при выборе с обычным seed: в vLLM 0.33.0 шум сэмплера считается
     от сида, позиции и ID токена, поэтому продолжение обязано совпасть
  4. ответ с сидом отличается от жадного, то есть сэмплинг вообще работает
С --url2 контроль идёт на первую реплику, продолжение на вторую.
Для проверки самого скрипта без GPU: --tokenizer bytes против fake_replica.py.
"""
import argparse
import http.client
import json
import os
import sys
from urllib.parse import urlparse

from toks import make_tokenizer


def stream(base, body, api_key=None, timeout=300):
    u = urlparse(base)
    cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
    conn = cls(u.hostname, u.port or (443 if u.scheme == "https" else 80), timeout=timeout)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    conn.request("POST", "/v1/completions", json.dumps(body), headers)
    resp = conn.getresponse()
    if resp.status != 200:
        raise RuntimeError(f"HTTP {resp.status}: {resp.read()[:400].decode(errors='replace')}")
    chunks = []
    for raw in resp:
        line = raw.strip()
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if payload == b"[DONE]":
            break
        chunks.append(json.loads(payload))
    conn.close()
    return chunks


def unpack(chunks):
    """ID токенов, logprobs и причина финала из чанков стрима."""
    ids, lps, finish = [], [], None
    for c in chunks:
        if "error" in c:
            raise RuntimeError(f"ошибка в стриме: {json.dumps(c['error'], ensure_ascii=False)[:300]}")
        for ch in c.get("choices") or []:
            part = ch.get("token_ids") or []
            lp = (ch.get("logprobs") or {}).get("token_logprobs")
            if lp is None or len(lp) != len(part):
                lp = [None] * len(part)
            ids += part
            lps += lp
            finish = ch.get("finish_reason") or finish
    return ids, lps, finish


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8001")
    ap.add_argument("--url2", default=None, help="реплика для продолжения; по умолчанию та же")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--tokenizer", choices=["hf", "bytes"], default="hf")
    ap.add_argument("--api-key", default=os.environ.get("UNDEAD_REPLICA_API_KEY"))
    ap.add_argument("--tokens", type=int, default=96)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    url2 = args.url2 or args.url

    tok = make_tokenizer(args.tokenizer, args.model)
    prompt = tok.encode_chat(
        [{"role": "user", "content": "Расскажи в трёх предложениях, зачем нужен Kubernetes."}])
    stop = list(tok.stop_ids)
    failed = []

    def run(url, prefix, max_tokens, sampling):
        body = {"model": args.model, "stream": True, "return_token_ids": True,
                "add_special_tokens": False, "stop_token_ids": stop, "logprobs": 1,
                "prompt": prompt + prefix, "max_tokens": max_tokens, **sampling}
        return unpack(stream(url, body, args.api_key))

    def split_check(name, sampling):
        """Контроль одним запросом против продолжения с разрезом посередине."""
        n = args.tokens
        control, control_lp, _ = run(args.url, [], n, sampling)
        again, _, _ = run(args.url, [], n, sampling)
        content = len(control) - (1 if control and control[-1] in stop else 0)
        k = max(1, content // 3)
        head = control[:k]
        tail, tail_lp, _ = run(url2, head, n - k, sampling)
        joined = head + tail
        print(f"\n[{name}]")
        print(f"  два контрольных прогона совпали: {'да' if control == again else 'НЕТ'}")
        # logprobs сверяем только до первого разошедшегося токена: дальше тексты разные
        same_upto = next((i for i, (x, y) in enumerate(zip(tail, control[k:])) if x != y),
                         min(len(tail), len(control) - k))
        pairs = [(a, b) for a, b in zip(control_lp[k:k + same_upto], tail_lp[:same_upto])
                 if a is not None and b is not None]
        if pairs:
            same = sum(1 for a, b in pairs if a == b)
            worst = max((abs(a - b) for a, b in pairs), default=0.0)
            print(f"  logprobs продолжения побитово равны контролю: {same} из {len(pairs)}, "
                  f"наибольшая разница {worst:.3g}")
        elif all(x is None for x in control_lp) or all(x is None for x in tail_lp):
            print("  logprobs реплика не отдала, сверка только по ID токенов")
        if joined == control:
            print(f"  разрез на токене {k}: продолжение совпало с контролем, "
                  f"{len(control)} токенов")
            if pairs and any(a != b for a, b in pairs):
                print("  ID совпали, но логиты не побитово равны: совпадение держится на удаче")
                failed.append(name + " (logprobs)")
            return control
        diff = next((i for i, (x, y) in enumerate(zip(joined, control)) if x != y),
                    min(len(joined), len(control)))
        print(f"  разрез на токене {k}: РАСХОЖДЕНИЕ на токене {diff} из {len(control)}")
        print(f"    контроль:     ...{tok.decode(control[max(0, diff - 8):diff + 8])!r}")
        print(f"    продолжение:  ...{tok.decode(joined[max(0, diff - 8):diff + 8])!r}")
        if diff == k:
            print("    разошлись на первом же токене продолжения: так выглядит шум, который "
                  "не привязан к позиции (раннер V1) или разные флаги реплик")
        failed.append(name)
        return control

    # 1. формат стрима
    where = args.url if url2 == args.url else f"{args.url}, продолжение на {url2}"
    print(f"Реплика: {where}, модель: {args.model}, промпт: {len(prompt)} токенов")
    try:
        body = {"model": args.model, "stream": True, "return_token_ids": True,
                "add_special_tokens": False, "stop_token_ids": stop, "logprobs": 1,
                "prompt": prompt, "max_tokens": 24, "temperature": 0}
        chunks = stream(args.url, body, args.api_key)
    except Exception as e:  # noqa: BLE001
        print(f"\nЗапрос с prompt из ID токенов не прошёл: {e}")
        return 1
    choices = [ch for c in chunks for ch in (c.get("choices") or [])]
    with_ids = [ch for ch in choices if ch.get("token_ids")]
    ids, lps, finish = unpack(chunks)
    print("\n[формат стрима]")
    print(f"  чанков: {len(chunks)}, с token_ids: {len(with_ids)}, токенов всего: {len(ids)}")
    print(f"  prompt_token_ids в первом чанке: "
          f"{'да' if choices and choices[0].get('prompt_token_ids') else 'нет'}")
    print(f"  больше одного токена в чанке: "
          f"{'бывает' if any(len(ch['token_ids']) > 1 for ch in with_ids) else 'нет'}")
    print(f"  logprobs в чанках: {'да' if any(x is not None for x in lps) else 'НЕТ'}")
    print(f"  finish_reason: {finish}, стоп-токен попал в token_ids: "
          f"{'да' if ids and ids[-1] in stop else 'нет'}")
    print(f"  текст: {tok.decode(ids)!r}")
    if not with_ids:
        print("  НЕТ token_ids в чанках: роутер на такой реплике работать не сможет")
        failed.append("формат стрима")

    # 2. жадный выбор
    greedy = split_check("жадный выбор", {"temperature": 0})

    # 3. выбор с обычным сидом
    sampled = split_check(f"сид {args.seed}, temperature 0.8, top_p 0.95",
                          {"temperature": 0.8, "top_p": 0.95, "seed": args.seed})

    # 4. сэмплинг вообще работает
    if greedy == sampled:
        print("\n  ответ с сидом равен жадному: сэмплинг не сработал, проверка 3 ничего не значит")
        failed.append("сэмплинг")

    print()
    if failed:
        print("Не прошло:", ", ".join(failed))
        return 1
    print("Всё прошло.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
