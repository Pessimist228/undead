#!/usr/bin/env bash
# field_check.sh : роутер на живой карте отвечает 400 на поля, которые он не пересылает (до реплик дело не доходит).
export PATH=/workspace/venv/bin:$PATH
cd /workspace/undead || exit 1
M=/dev/shm/models/Qwen2.5-1.5B-Instruct
setsid nohup python router.py --replicas http://127.0.0.1:8001,http://127.0.0.1:8002 --model "$M" --port 8091 > /workspace/runs/router-fields.log 2>&1 < /dev/null &
for i in $(seq 1 90); do curl -sf -o /dev/null localhost:8091/health && break; sleep 1; done
for extra in '"presence_penalty":0.5' '"frequency_penalty":0.3' '"top_k":40' '"stop":["x"]' '"tools":[{"type":"function"}]' '"n":2' '"response_format":{"type":"json_object"}'; do
  code=$(curl -s -m 20 -o /tmp/f.json -w '%{http_code}' -H 'Content-Type: application/json' \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8,$extra}" localhost:8091/v1/chat/completions)
  echo "$extra -> HTTP $code: $(python3 -c 'import json;d=json.load(open("/tmp/f.json"));print((d.get("error") or {}).get("message","ok")[:70])')"
done
pkill -f '[r]outer.py --replicas http://127.0.0.1:8001,http://127.0.0.1:8002 --model /dev/shm/models/Qwen2.5-1.5B-Instruct --port 8091'
echo "готово"
