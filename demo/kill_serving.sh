#!/usr/bin/env bash
# Kill the replica that is serving the active answer right now.
ROUTER=${1:-${UNDEAD_ROUTER:-http://127.0.0.1:8090}}
url=$(curl -s "$ROUTER/metrics" | python3 -c 'import json,sys; a=json.load(sys.stdin)["active"]; print(a[0]["replica"] if a else "")')
if [ -z "$url" ]; then echo "no active answer"; exit 1; fi
port=${url##*:}
pid=$(pgrep -f -- "--port $port" | head -1)
pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
echo "answer is served by replica :$port"
echo "process group $pgid"
echo "\$ kill -9 -$pgid"
kill -9 -- "-$pgid"
sleep 0.4
if curl -s -m 2 -o /dev/null "http://127.0.0.1:$port/health"; then
  echo "replica :$port is still alive"
else
  echo "replica :$port is dead"
fi
