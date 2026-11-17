# Pilot run, 16 November 2026

Every number below was recomputed from the raw `events.jsonl` journals with `verify.py`, not copied from the harness summaries. The journals are in `raw/`.

## Setup

- Two RTX 4090 (the 48 GB modded kind) on two physical machines of one provider, rented on Vast.ai. Driver 595.104.02 on both.
- vLLM 0.33.0 from PyPI, torch 2.14.0+cu130, Python 3.12.3.
- Models in bf16: Qwen2.5-1.5B-Instruct, Qwen2.5-7B-Instruct, Qwen3-1.7B. Weights were checked by hash against Hugging Face, see `model_hashes.txt`.
- Every replica runs with `VLLM_USE_V2_MODEL_RUNNER=1` and `--generation-config vllm`.
- Runs marked "on" add `VLLM_BATCH_INVARIANT=1`, `--attention-backend FLASH_ATTN` and prefix caching off (0a also has a pass with prefix caching on).
- Runs marked "off" set `VLLM_BATCH_INVARIANT=0`, keep prefix caching on and leave the attention backend to vLLM. The replica logs show it picked FLASH_ATTN there as well.
- Every request carries an explicit seed. Two sampling modes: greedy, and seeded with temperature 0.7 and top_p 0.95.
- 50 prompts: 20 Russian, 20 English, 4 code, 6 long ones of 1.5k to 1.9k tokens. Answers are 256 tokens unless noted.

## What is compared

A control answer is generated in one request on one replica. The same request is then resumed on the other replica from a cut point: the prompt is the original prompt IDs plus the first k token IDs of the control answer. A cut matches when every remaining token ID equals the control. Logprobs are compared as well.

Each prompt and mode gets three cut points. One of them runs while the resuming replica serves 8 background streams.

The end-to-end tests go through the router: the replica serving an answer gets `kill -9` or `SIGSTOP` in the middle of the stream, and the answer the client received is compared with a control.

## Runs

| Run | Card | Model | Batch invariance | Cuts matching control | kill -9 matching control | Archive in `raw/` |
|---|---|---|---|---|---|---|
| 0a | A | Qwen2.5-1.5B | on | 600/600 | 30/30 | A-0a-20261116-144825 |
| 0a | B | Qwen2.5-1.5B | on | 600/600 | 30/30 | B-0a-20261116-144157 |
| 0b, two machines | A+B | Qwen2.5-1.5B | on | 300/300 | not run | AB-0b-20261116-164302 |
| answers up to 1024 tokens | A | Qwen2.5-1.5B | on | 120/120 | not run | A-long1024-0a-20261116-161218 |
| 24 background streams | B | Qwen2.5-1.5B | on | 120/120 | 6/6 | B-heavy-0a-20261116-180126 |
| 0a | A | Qwen2.5-7B | on | 96/96 | 8/8 | A-7b-0a-20261116-173411 |
| 0a | B | Qwen2.5-7B | on | 96/96 | 8/8 | B-7b-0a-20261116-171325 |
| 0a | A | Qwen3-1.7B | on | 96/96 | 6/6 | A-qwen3-0a-20261116-180341 |
| 0a | B | Qwen3-1.7B | on | 96/96 | 6/6 | B-qwen3-0a-20261116-183121 |
| 0c | B | Qwen2.5-1.5B | off | 134/300 | 2/10 | B-0c-20261116-160748 |
| 0c | A | Qwen2.5-1.5B | off | 132/300 | 1/10 | A-0c-20261116-183710 |
| 0c | B | Qwen2.5-7B | off | 27/60 | 0/6 | B-7b-0c-20261116-174252 |
| 0c | A | Qwen3-1.7B | off | 25/72 | 2/4 | A-qwen3-0c-20261116-182942 |

With batch invariance on: 2124 cuts out of 2124 match, 316,350 tokens compared, logprobs bit-identical in all of them. Through the router: `kill -9` 94/94, `SIGSTOP` 34/34, client disconnect and resume 20/20, a batch of answers with one replica killed 84/84.

In 0b the two replicas ran on different machines. 150 cuts went from A to B and 150 from B to A. The replicas listened on localhost only and were joined by SSH port forwards through a third machine.

Control answers were also compared between the two cards: 200 of 200 on Qwen2.5-1.5B, 32 of 32 on Qwen2.5-7B and 32 of 32 on Qwen3-1.7B are equal by token IDs, with bit-identical logprobs.

## Pause seen by the client

Median pause in the stream, batch invariance on.

| Failure | Qwen2.5-1.5B | Qwen2.5-7B | Qwen3-1.7B |
|---|---|---|---|
| `kill -9` | 32 to 42 ms | 55 to 58 ms | 48 to 74 ms |
| replica frozen (`SIGSTOP`) | 1.03 s | 1.04 s | 1.02 s |
| batch of answers, one replica killed | 68 to 94 ms | 135 to 161 ms | 57 to 61 ms |

The largest single pause after `kill -9` was 78 ms. The one second for a frozen replica is the router's silence timeout.

The pause grows with context length, because the second replica has to read the prompt and the emitted tokens again. On Qwen2.5-1.5B the resumed request returned its first token in 28 ms with up to 300 tokens of context and in 54 ms with about 1.7k tokens. The `kill -9` tests used short prompts only.

## Cost of batch invariance

Throughput of one replica, tokens per second, without and with batch invariance.

| Model | 1 stream | 16 streams | Slowdown |
|---|---|---|---|
| Qwen2.5-1.5B | 232 / 64 | 3167 / 964 | 3.6x and 3.3x |
| Qwen2.5-7B | 64 / 29 | 978 / 460 | 2.25x and 2.1x |
| Qwen3-1.7B | 225 / 204 | 3001 / 1127 | 1.1x and 2.7x |

Each number is one short series: 4 requests in a row for the 1-stream column and 16 at once for the 16-stream column, greedy, 256 tokens each. The two configurations use the same attention backend and model runner. They differ in batch invariance and in prefix caching, and the prompts of this probe are under 100 characters, so caching hardly matters. The Qwen2.5-1.5B row was repeated once on another machine on 17 November with the same result. None of this was run under a real load.

This is one GPU model and three language models. I would not read a trend into it yet.

## Without batch invariance

- 55 to 65% of resumed answers differ from the control: 166 and 168 of 300 on Qwen2.5-1.5B (two machines), 33 of 60 on Qwen2.5-7B, 47 of 72 on Qwen3-1.7B.
- The router still finishes every answer. 185 end-to-end runs, zero errors. The text after the failure is just not the text the dead replica would have produced.
- Logprobs differ from the control in 299 of 300 cuts on Qwen2.5-1.5B. Where token IDs match, they match by luck.
- A plain rerun of the same request matched the control in 12 of 20 cases on the same replica and in 4 of 20 on the other one. There is no background load in that check, only the prefix cache state changes.
- A single request with nothing else going on is reproducible even here: 100 of 100 control answers were bit-identical between cards A and B.

## Which request parameters survive a resume

`probe_params.py`, Qwen2.5-1.5B, batch invariance on, 3 prompts, 2 cut points, 160-token answers. Details in `probe_params-A.json`.

| Parameters | Cuts matching | Note |
|---|---|---|
| greedy | 6/6 | |
| seed + top_p | 6/6 | |
| seed + top_k | 6/6 | |
| seed + min_p | 6/6 | |
| repetition_penalty | 6/6 | |
| logit_bias | 6/6 | |
| stop string | 6/6 | finish reason matches too |
| min_tokens, reduced by the tokens already emitted | 6/6 | |
| min_tokens, passed through unchanged | 0/6 | vLLM answers HTTP 400 |
| presence_penalty | 0/6 | diverges 0 to 8 tokens after the cut |
| frequency_penalty | 0/6 | same |
| structured_outputs (regex) | 0/6 | diverges at the cut |
| no seed | 0/6 | diverges at once |

vLLM counts presence and frequency penalties over generated tokens only, and in a resumed request the earlier tokens are part of the prompt. A structured-output grammar starts from its initial state. Without a seed vLLM picks a random one per request.

The router itself forwards only `temperature`, `top_p`, `seed` and `max_tokens`, and assigns a seed when the client sends none. Every other field of this table gets HTTP 400 from the router when it is set.

## Demo recording, 17 November 2026

A third machine: one RTX 4090 with 24 GB, driver 595.91.07, vLLM 0.33.0, two Qwen2.5-1.5B replicas with batch invariance on. The weights came from Hugging Face directly and matched the hashes in `model_hashes.txt`.

- Four recorded takes. In each one the answer that lost its replica to `kill -9` had the same sha256 of token IDs as the control request after it, and as four uninterrupted runs of the same request on both replicas.
- The pause was 169 to 189 ms, against 32 to 42 ms on the pilot machines for the same model.
- A separate probe showed why. After `kill -9`, a client reading from the killed replica saw the connection close 139 ms later. The other replica was not slowed down: it returned its first token in 23 to 46 ms both before and after the kill.
- So the pause is the time the OS takes to close the dead process's connection, plus the first token of the resumed request. The first part depends on the machine.

The take in `demo/demo.gif` is the fourth one: 321 tokens, pause 174 ms. An earlier attempt on a machine with driver 565.77 (CUDA 12.7) did not start at all, because the vLLM 0.33.0 wheel is built for CUDA 13.

## Changed router on a fourth machine, 17 November 2026

The router changed after a review (the last section says how), so it was run on real vLLM again. A fourth machine: one RTX 4090 with 24 GB, driver 590.48.01, vLLM 0.33.0, two Qwen2.5-1.5B replicas on one card, 12 prompts. The weights came from Hugging Face directly and matched `model_hashes.txt`. These runs used the `router.py` at the top of the repository.

- Batch invariance on (`F-0a-20261117-081040`): 72 cuts out of 72 match the control, 9,852 tokens compared, logprobs bit-identical. Through the router: `kill -9` 10/10 with a median pause of 31 ms, `SIGSTOP` 4/4, client disconnect and resume 4/4, a batch of answers with one replica killed 8/8.
- Batch invariance off (`F-0c-20261117-083406`): 37 cuts out of 72 differ from the control. All 34 runs through the router finished without errors, 8 of them with `kill -9` in the middle and 3 with `SIGSTOP`.
- The control answers of the first run are the control answers of card A from the pilot: 24 out of 24 equal by token IDs, logprobs bit-identical. Card A has driver 595.104.02 and 48 GB, this machine has driver 590.48.01 and 24 GB. The request from the demo recording also gave the same sha256 of token IDs here as on the demo machine with driver 595.91.07. So a different driver version did not change a single token.
- Both replicas frozen with `SIGSTOP` before the request, router deadline 20 s: the client got HTTP 502 "no live replicas for more than 20 s" after 24 s. The router measured in the pilot waits forever in this case. After `SIGCONT` the same request returned the same text as before the freeze. The log is `raw/F-frozen-20261117.txt`. Item 2 in it is a mistake in my test script, which sent broken JSON.
- Request fields the router does not forward got HTTP 400: `presence_penalty`, `frequency_penalty`, `top_k`, `stop`, `tools`, `n`, `response_format` (`raw/F-fields-20261117.txt`). A resume with a bad cursor got HTTP 400 too.
- Throughput repeated the pilot: 64 and 968 tokens per second with batch invariance at 1 and 16 streams, 231 and 3229 without.
- vLLM 0.34.0 on the same machine (`raw/F-vllm034-20261117.txt`): `check_vllm.py` passes, and the demo request with `kill -9` in the middle gave the same sha256 of token IDs as its control run and as vLLM 0.33.0. Nothing else was measured on 0.34.0.

The scripts for the last four checks are in `code-as-measured/2026-11-17/`.

## Not tested

1. Replicas on different GPU models. vLLM documents reproducibility for the same hardware and version only. A different driver version alone changed nothing in the 17 November check.
2. Quantized weights.
3. `kill -9` on a long context in the end-to-end tests.
4. Throughput overhead of the router itself.
5. A real preemption by a cloud provider. The tests use `kill -9` and `SIGSTOP`.
6. Replicas that span more than one GPU (tensor parallelism).
7. More than one failure during the same answer on real vLLM.
8. Throughput under a real load. The cost table above is short series only.

## Checking the numbers yourself

```bash
cd pilot-2026-11-16
mkdir run && tar -xzf raw/A-0a-20261116-144825.tar.gz -C run
mkdir run2 && tar -xzf raw/B-0a-20261116-144157.tar.gz -C run2
python3 verify.py run/0a-20261116-144825 run2/0a-20261116-144157
```

Each archive holds the event journal with all token IDs and logprobs, the environment fingerprint, replica logs and the harness summary. The harness prints in Russian.

`env.json` in every archive stores the first 16 hex digits of the sha256 of the code that produced it. Those exact files are in `code-as-measured/`: `router.py` is `6283d7e3ecc0aeff` in all runs, `step0.py` is `72ec467b0b7665e2`, and `step0.after-fix.py` is `2d539e74317fdb71` (a connection-handling fix made before run 0b, used on card A from then on; card B kept the earlier file). The `router.py` at the top of the repository is not the measured file. Its messages are translated to English and an unused legacy option is removed. After a review on 17 November it also got four changes. The deadline for finding a live replica is now checked on every turn: before that, replicas frozen before the request could keep an answer waiting forever. The silence timer counts from the last token. A bad resume cursor gets HTTP 400. Request fields the router does not forward are rejected instead of ignored. These four changes were run on the fake replicas first and then on real vLLM, see the 17 November check above. A fifth change came after that check: the router refuses requests with a foreign `Origin` header and, when it listens on localhost, a `Host` header that is not a localhost name. It was run on the fake replicas only.
