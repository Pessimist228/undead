#!/usr/bin/env bash
# frozen_real.sh : две настоящие реплики vLLM, новый роутер. Проверяем то, чего нет в step0.py:
#   обычный ответ; 400 на поле, которое роутер не пересылает; 400 на плохой курсор;
#   обе реплики заморожены ДО запроса: роутер обязан сдаться по дедлайну; после разморозки снова отвечает.
set -uo pipefail
W=/workspace; PROJ=$W/undead; M=/dev/shm/models/Qwen2.5-1.5B-Instruct
export PATH=$W/venv/bin:$PATH HF_HOME=/dev/shm/hf
R=http://127.0.0.1:8090
cd "$PROJ" || exit 1
for p in 8001 8002; do
  VLLM_USE_V2_MODEL_RUNNER=1 VLLM_BATCH_INVARIANT=1 setsid nohup vllm serve "$M" --host 127.0.0.1 --port $p \
    --generation-config vllm --gpu-memory-utilization 0.42 --max-model-len 8192 --max-num-seqs 32 \
    --no-enable-prefix-caching --attention-backend FLASH_ATTN > "$W/runs/serve-$p.log" 2>&1 < /dev/null &
done
t0=$(date +%s)
for i in $(seq 1 240); do
  curl -sf -o /dev/null localhost:8001/health && curl -sf -o /dev/null localhost:8002/health && break
  sleep 5
done
curl -sf -o /dev/null localhost:8001/health && curl -sf -o /dev/null localhost:8002/health || { echo "реплики не поднялись"; tail -5 "$W"/runs/serve-800*.log; exit 3; }
echo "реплики поднялись за $(( $(date +%s) - t0 )) с"
setsid nohup python router.py --replicas http://127.0.0.1:8001,http://127.0.0.1:8002 --model "$M" --port 8090 \
  --deadline 20 --first-token-timeout 8 --cooldown 3 > "$W/runs/router-frozen.log" 2>&1 < /dev/null &
for i in $(seq 1 60); do curl -sf -o /dev/null $R/health && break; sleep 1; done
echo "0. демо-запрос, сид 42: $(UNDEAD_ROUTER=$R python demo/client.py --seed 42 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep -a 'sha256' | tail -1)"
echo "   на машине из ролика (драйвер 595.91.07) было: 6a7ceb51eebd48aeca8f6175ff9df2c5"
ask() {  # ask <файл> <доп. поля JSON>
  curl -s -m 120 -D /tmp/h.txt -o "$1" -w '%{http_code} за %{time_total} с' -H 'Content-Type: application/json' \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Write two sentences about the sea.\"}],\"max_tokens\":60,\"seed\":7,\"temperature\":0.7$2}" \
    $R/v1/chat/completions
}
show() { python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print("   ", (d.get("error") or {}).get("message") or d["choices"][0]["message"]["content"][:90].replace("\n"," "))' "$1"; }
echo "1. обычный ответ: $(ask /tmp/a1.json '')"; show /tmp/a1.json
AID=$(grep -i '^x-answer-id' /tmp/h.txt | tr -d '\r' | cut -d' ' -f2)
echo "2. presence_penalty=0.5: $(ask /tmp/a2.json ',\"presence_penalty\":0.5')"; show /tmp/a2.json
echo "3. курсор abc: HTTP $(curl -s -m 20 -o /dev/null -w '%{http_code}' "$R/v1/streams/$AID?from=abc"); чужой ответ: HTTP $(curl -s -m 20 -o /dev/null -w '%{http_code}' "$R/v1/streams/$AID?from=ans-000000000000:3"); from=0: HTTP $(curl -s -m 20 -o /dev/null -w '%{http_code}' "$R/v1/streams/$AID?from=0")"
PG=""
for p in 8001 8002; do
  for pid in $(pgrep -f -- "--port $p"); do PG="$PG $(ps -o pgid= -p "$pid" | tr -d ' ')"; done
done
PG=$(echo $PG | tr ' ' '\n' | sort -u | tr '\n' ' ')
echo "группы процессов реплик: $PG"
for g in $PG; do kill -STOP -- "-$g"; done
sleep 1
echo "4. обе реплики заморожены, дедлайн 20 с: $(ask /tmp/a3.json '')"; show /tmp/a3.json
for g in $PG; do kill -CONT -- "-$g"; done
sleep 6
echo "5. после разморозки: $(ask /tmp/a4.json '')"; show /tmp/a4.json
cmp -s <(python3 -c 'import json;print(json.load(open("/tmp/a1.json"))["choices"][0]["message"]["content"])') <(python3 -c 'import json;print(json.load(open("/tmp/a4.json"))["choices"][0]["message"]["content"])') && echo "   тот же текст, что в п. 1 (тот же сид)" || echo "   ТЕКСТ ОТЛИЧАЕТСЯ от п. 1"
curl -s $R/metrics | python3 -c 'import json,sys; m=json.load(sys.stdin); print("metrics:", {k: m[k] for k in ("answers","finished","failed","failovers")}, m["replicas"])'
pkill -f '[r]outer.py --replicas'
for g in $PG; do kill -9 -- "-$g" 2>/dev/null; done
echo "готово"
