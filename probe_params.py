#!/usr/bin/env python3
"""Какие параметры запроса переживают продолжение с середины, а какие нет.

    python probe_params.py --url http://127.0.0.1:8001 --url2 http://127.0.0.1:8002 --model ПУТЬ

Для каждого набора параметров: контроль одним запросом на первой реплике, потом
продолжение с двух точек разреза на второй. Продолжение получает prompt + уже выданные
ID токенов и те же параметры. Сравниваются ID токенов и причина финала.
Это замер, а не тест: для части строк расхождение и есть ожидаемый результат.
Для проверки самого скрипта без GPU: --tokenizer bytes против fake_replica.py.
"""
import argparse
import collections
import json
import os
import sys

from check_vllm import stream, unpack
from prompts import PROMPTS
from toks import make_tokenizer

SAMPLE = {"temperature": 0.8, "top_p": 0.95}
REGEX = r"(\d\. [A-Za-z ,]{20,60}\.\n){6}"


def same(params, k, total):
    return params


def rebase_min_tokens(params, k, total):
    return {**params, "min_tokens": max(0, params["min_tokens"] - k)}


def cases(seed, total, banned):
    s = {**SAMPLE, "seed": seed}
    return [
        ("greedy", {"temperature": 0}, same),
        ("seed + top_p", s, same),
        ("seed + top_k", {"temperature": 0.8, "top_k": 40, "seed": seed}, same),
        ("seed + min_p", {"temperature": 0.8, "min_p": 0.05, "seed": seed}, same),
        ("repetition_penalty", {**s, "repetition_penalty": 1.15}, same),
        ("presence_penalty", {**s, "presence_penalty": 0.8}, same),
        ("frequency_penalty", {**s, "frequency_penalty": 0.8}, same),
        ("logit_bias", {**s, "logit_bias": {str(t): -100 for t in banned}}, same),
        ("stop-строка", {**s, "stop": ["\n\n"]}, same),
        ("min_tokens с пересчётом", {**s, "min_tokens": total}, rebase_min_tokens),
        ("min_tokens без пересчёта", {**s, "min_tokens": total}, same),
        ("structured_outputs regex", {**s, "structured_outputs": {"regex": REGEX}}, same),
        ("без seed", dict(SAMPLE), same),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8001")
    ap.add_argument("--url2", default="http://127.0.0.1:8002")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--tokenizer", choices=["hf", "bytes"], default="hf")
    ap.add_argument("--api-key", default=os.environ.get("UNDEAD_REPLICA_API_KEY"))
    ap.add_argument("--tokens", type=int, default=160)
    ap.add_argument("--prompts", default="0,21,41", help="номера промптов из prompts.PROMPTS")
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--out", default="probe_params.json")
    args = ap.parse_args()

    tok = make_tokenizer(args.tokenizer, args.model)
    stop = list(tok.stop_ids)
    total = args.tokens

    def run(url, prompt, prefix, max_tokens, params):
        body = {"model": args.model, "stream": True, "return_token_ids": True,
                "add_special_tokens": False, "stop_token_ids": stop, "logprobs": 1,
                "prompt": prompt + prefix, "max_tokens": max_tokens, **params}
        return unpack(stream(url, body, args.api_key))

    rows = []
    summary = collections.OrderedDict()
    for pi, pnum in enumerate(int(x) for x in args.prompts.split(",")):
        prompt = tok.encode_chat([{"role": "user", "content": PROMPTS[pnum]}])
        seed = args.seed + pi
        # Токены для logit_bias берём из обычного ответа: три самых частых.
        base, _, _ = run(args.url, prompt, [], total, {**SAMPLE, "seed": seed})
        banned = [t for t, _ in collections.Counter(base).most_common(3)]
        for name, params, cont_params in cases(seed, total, banned):
            st = summary.setdefault(name, {"ok": 0, "all": 0, "first": None, "lp_same": 0,
                                           "lp_all": 0, "errors": []})
            try:
                ctl, ctl_lp, ctl_fin = run(args.url, prompt, [], total, params)
            except RuntimeError as e:
                st["errors"].append(f"контроль: {str(e)[:160]}")
                continue
            content = len(ctl) - (1 if ctl and ctl[-1] in stop else 0)
            for k in sorted({max(1, content // 4), max(2, content * 2 // 3)}):
                if k >= len(ctl):
                    continue
                st["all"] += 1
                row = {"case": name, "prompt": pnum, "seed": seed, "cut": k,
                       "control_len": len(ctl), "control_finish": ctl_fin}
                try:
                    tail, tail_lp, fin = run(args.url2, prompt, ctl[:k], total - k,
                                             cont_params(params, k, total))
                except RuntimeError as e:
                    row["error"] = str(e)[:300]
                    st["errors"].append(f"продолжение: {str(e)[:160]}")
                    rows.append(row)
                    continue
                want = ctl[k:]
                div = next((i for i, (a, b) in enumerate(zip(tail, want)) if a != b), None)
                if div is None and len(tail) != len(want):
                    div = min(len(tail), len(want))
                upto = div if div is not None else len(want)
                pairs = [(a, b) for a, b in zip(ctl_lp[k:k + upto], tail_lp[:upto])
                         if a is not None and b is not None]
                st["lp_all"] += len(pairs)
                st["lp_same"] += sum(1 for a, b in pairs if a == b)
                ok = div is None and fin == ctl_fin
                st["ok"] += ok
                if div is not None:
                    at = k + div
                    st["first"] = at if st["first"] is None else min(st["first"], at)
                row.update(match=ok, first_divergence=None if div is None else k + div,
                           finish=fin, tail_len=len(tail))
                rows.append(row)

    print(f"{'параметры':28} {'совпало':>8}  {'logprobs побитово':>18}  примечание")
    for name, st in summary.items():
        note = ""
        if st["first"] is not None:
            note = f"первое расхождение на токене {st['first']}"
        if st["errors"]:
            note = (note + "; " if note else "") + f"ошибок {len(st['errors'])}: {st['errors'][0]}"
        lp = f"{st['lp_same']}/{st['lp_all']}" if st["lp_all"] else "нет данных"
        print(f"{name:28} {st['ok']:>3}/{st['all']:<4}  {lp:>18}  {note}")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "summary": summary, "rows": rows}, f, ensure_ascii=False, indent=1)
    print(f"\nПодробности: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
