#!/bin/sh
# Preemption demo: two vLLM replicas rented by canary.py, the published undead router (commit 3443445) in front,
# load.py streaming through it, and the bid of the first instance dropped in the middle of the answers.
# Run as root on the Ubuntu server:  demo_preempt.sh STATE_DIR [KILL_ID]
# The router gets every healthy replica (at least two). KILL_ID, when given, must be one of the healthy ones;
# otherwise the first healthy instance of state.json loses its bid.
# It never destroys instances and never prints the instance secret. The router and the load generator are
# stopped on any exit (trap). The rented instances stay: run `canary.py --dir STATE_DIR stop --yes` yourself.
set -u

CANARY=/opt/undead/canary/canary.py
LOAD=/opt/undead/canary/load.py
ROUTER_DIR=/opt/undead/router
PY=/opt/undead/venv/bin/python
MODEL=Qwen/Qwen2.5-7B-Instruct
ROUTER_PORT=8088
KILL_PRICE=0.05

[ $# -ge 1 ] && [ $# -le 2 ] || { echo "usage: $0 STATE_DIR [KILL_ID]" >&2; exit 2; }
DIR=$1
KILL_ID=${2:-}
case "$KILL_ID" in
    *[!0-9]*) echo "KILL_ID must be an instance number" >&2; exit 2 ;;
esac
[ -d "$DIR" ] || { echo "STATE_DIR $DIR does not exist" >&2; exit 2; }

ROUTER_PID=""
LOAD_PID=""
cleanup() {
    trap - EXIT
    if [ -n "$LOAD_PID" ]; then kill "$LOAD_PID" 2>/dev/null; fi
    if [ -n "$ROUTER_PID" ]; then
        kill "$ROUTER_PID" 2>/dev/null
        wait "$ROUTER_PID" 2>/dev/null
        echo "router stopped"
    fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

# a proxy in the environment must not capture calls to localhost
no_proxy=127.0.0.1,localhost
export no_proxy

http_get() {
    "$PY" -c 'import sys, urllib.request
sys.stdout.write(urllib.request.urlopen(sys.argv[1], timeout=5).read().decode())' "$1"
}

# (1) healthy replicas ("ID URL" lines) and the secret; the raw output is never shown because it carries the secret
OUT=$(python3 "$CANARY" --dir "$DIR" urls --healthy --show-secret) || {
    echo "canary.py urls failed:" >&2
    printf '%s\n' "$OUT" | grep -v '^secret:' >&2
    exit 1
}
LINES=$(printf '%s\n' "$OUT" | grep -E '^[0-9]+ http://')
SECRET=$(printf '%s\n' "$OUT" | sed -n 's/^secret: //p')
OUT=""
N=$(printf '%s\n' "$LINES" | grep -c -E '^[0-9]+ http://')
if [ "$N" -lt 2 ]; then
    echo "need at least two healthy replicas, found $N" >&2
    printf '%s\n' "$LINES" | grep -E '^[0-9]+ http://' >&2
    exit 1
fi
case "$SECRET" in
    ""|"(none)"|"(hidden)") echo "no instance secret in state (rent with a profile that uses {KEY})" >&2; exit 1 ;;
esac
IDS=$(printf '%s\n' "$LINES" | cut -d' ' -f1)
REPLICAS=$(printf '%s\n' "$LINES" | cut -d' ' -f2 | paste -s -d, -)

# the instance whose bid is dropped: KILL_ID if given (it must be healthy), else the first healthy one
if [ -z "$KILL_ID" ]; then
    KILL_ID=$(printf '%s\n' "$IDS" | head -n 1)
elif ! printf '%s\n' "$IDS" | grep -qx "$KILL_ID"; then
    echo "KILL_ID $KILL_ID is not one of the healthy instances: $(printf '%s' "$IDS" | paste -s -d' ' -)" >&2
    exit 1
fi
echo "replicas: $REPLICAS"
echo "instance to outbid: $KILL_ID"

# (2) the router; the replica key goes in the environment, not on the command line (ps shows arguments)
(
    cd "$ROUTER_DIR" || exit 1
    UNDEAD_REPLICA_API_KEY=$SECRET
    export UNDEAD_REPLICA_API_KEY
    exec "$PY" router.py --replicas "$REPLICAS" --model "$MODEL" --host 127.0.0.1 --port "$ROUTER_PORT"
) >"$DIR/router.log" 2>&1 &
ROUTER_PID=$!
SECRET=""

T=0
until http_get "http://127.0.0.1:$ROUTER_PORT/health" >/dev/null 2>&1; do
    if ! kill -0 "$ROUTER_PID" 2>/dev/null; then
        echo "router exited, see $DIR/router.log" >&2
        ROUTER_PID=""
        exit 1
    fi
    T=$((T + 1))
    if [ "$T" -gt 120 ]; then echo "router did not answer within 120 s, see $DIR/router.log" >&2; exit 1; fi
    sleep 1
done
echo "router is up"

# (3) the load
"$PY" "$LOAD" --url "http://127.0.0.1:$ROUTER_PORT" --model "$MODEL" --streams 8 --duration 240 \
    --max-tokens 500 --out "$DIR/run.jsonl" >"$DIR/load.out" 2>&1 &
LOAD_PID=$!

# (4) in the middle of the answers: drop the bid of the first instance
sleep 75
if ! kill -0 "$LOAD_PID" 2>/dev/null; then echo "load.py ended before the bid drop, see $DIR/load.out" >&2; exit 1; fi
echo "$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ) $KILL_ID" >"$DIR/kill.txt"
python3 "$CANARY" --dir "$DIR" bid --id "$KILL_ID" --price "$KILL_PRICE" --yes || {
    echo "bid change failed, the run is not valid" >&2
    exit 1
}

# (5) the load runs out, then the router's counters, then stop the router
wait "$LOAD_PID"
LOAD_RC=$?
LOAD_PID=""
http_get "http://127.0.0.1:$ROUTER_PORT/metrics" >"$DIR/metrics-after.json" || echo "could not read /metrics" >&2
kill "$ROUTER_PID" 2>/dev/null
wait "$ROUTER_PID" 2>/dev/null
ROUTER_PID=""
echo "router stopped (load.py exit code $LOAD_RC)"

# (6)
echo "--- kill.txt (UTC time of the bid drop, instance id)"
cat "$DIR/kill.txt"
echo "--- load.out"
cat "$DIR/load.out"
