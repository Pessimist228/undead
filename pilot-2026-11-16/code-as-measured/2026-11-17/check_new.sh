#!/usr/bin/env bash
# check_new.sh : две реплики на свежем vLLM с инвариантностью. check_vllm.py, потом демо-сценарий через роутер:
#   ответ с kill -9 посередине и контрольный ответ без сбоя должны дать один хеш ID токенов.
set -uo pipefail
W=/workspace; V=$W/venv-new; M=/dev/shm/models/Qwen2.5-1.5B-Instruct
export PATH=$V/bin:$PATH HF_HOME=/dev/shm/hf UNDEAD_ROUTER=http://127.0.0.1:8090
cd "$W/undead" || exit 1
python -c 'import vllm, torch; print("vllm", vllm.__version__, "torch", torch.__version__)'
for p in 8001 8002; do
  VLLM_USE_V2_MODEL_RUNNER=1 VLLM_BATCH_INVARIANT=1 setsid nohup vllm serve "$M" --host 127.0.0.1 --port $p \
    --generation-config vllm --gpu-memory-utilization 0.42 --max-model-len 8192 --max-num-seqs 32 \
    --no-enable-prefix-caching --attention-backend FLASH_ATTN > "$W/runs/new-serve-$p.log" 2>&1 < /dev/null &
done
t0=$(date +%s)
for i in $(seq 1 240); do
  curl -sf -o /dev/null localhost:8001/health && curl -sf -o /dev/null localhost:8002/health && break
  pgrep -f -- '--port 800[12]' >/dev/null || break
  sleep 5
done
if ! { curl -sf -o /dev/null localhost:8001/health && curl -sf -o /dev/null localhost:8002/health; }; then
  echo "реплики на свежем vLLM не поднялись"; tail -n 12 "$W/runs/new-serve-8001.log"; exit 3
fi
echo "реплики поднялись за $(( $(date +%s) - t0 )) с"
grep -ahoE 'Using [A-Z_]+ attention backend[^.]*|Using AttentionBackendEnum\.[A-Z_]+ backend' "$W/runs/new-serve-8001.log" | sort -u | head -2
echo "== check_vllm.py"
python check_vllm.py --url http://127.0.0.1:8001 --url2 http://127.0.0.1:8002 --model "$M"; echo "check_vllm rc=$?"
setsid nohup python router.py --replicas http://127.0.0.1:8001,http://127.0.0.1:8002 --model "$M" --port 8090 > "$W/runs/new-router.log" 2>&1 < /dev/null &
for i in $(seq 1 60); do curl -sf -o /dev/null $UNDEAD_ROUTER/health && break; sleep 1; done
strip() { sed 's/\x1b\[[0-9;]*m//g' | grep -aE 'tokens, served|Pause|sha256|interrupted|identical|DIFFER|differ|no failure' ; }
echo "== ответ с kill -9 посередине"
( sleep 2.5; bash demo/kill_serving.sh > "$W/runs/new-kill.txt" 2>&1 ) &
python demo/client.py --seed 42 --save-hash /tmp/hnew 2>&1 | strip
wait
cat "$W/runs/new-kill.txt"
sleep 1
echo "== контроль без сбоя"
python demo/client.py --seed 42 --compare-hash /tmp/hnew 2>&1 | strip
echo "   на vLLM 0.33.0 этот запрос даёт 6a7ceb51eebd48aeca8f6175ff9df2c5"
pkill -f '[r]outer.py --replicas'
for p in 8001 8002; do for pid in $(pgrep -f -- "--port $p"); do kill -9 -- "-$(ps -o pgid= -p "$pid" | tr -d ' ')" 2>/dev/null; done; done
echo "готово"
