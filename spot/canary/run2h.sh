#!/bin/sh
# Long run: the registry router, fleet.py keeping its replica list equal to the healthy canary instances, and
# constant streaming load, while the marketplace takes cards away and gives them back.
# Run as root on the Ubuntu server:  run2h.sh STATE_DIR [STREAMS=24] [SECONDS=7200]
# Never prints the instance secret or the admin token, never destroys instances. Router, fleet.py and load.py are
# stopped on any exit (trap). Exit 3: fewer than two replicas registered within 25 minutes.
set -u

CANARY=/opt/undead/canary/canary.py
FLEET=/opt/undead/canary/fleet.py
LOAD=/opt/undead/canary/load.py
ROUTER_DIR=/opt/undead/router-registry
PY=/opt/undead/venv/bin/python
MODEL=Qwen/Qwen2.5-7B-Instruct
ROUTER_PORT=8088
ADMIN_PORT=8089
WAIT_TRIES=300   # x 5 s = 25 minutes for the first two replicas

[ $# -ge 1 ] && [ $# -le 3 ] || { echo "usage: $0 STATE_DIR [STREAMS=24] [SECONDS=7200]" >&2; exit 2; }
DIR=$1
STREAMS=${2:-24}
DURATION=${3:-7200}
case "$STREAMS$DURATION" in
    ""|*[!0-9]*) echo "STREAMS and SECONDS must be whole numbers" >&2; exit 2 ;;
esac
[ "$STREAMS" -ge 1 ] && [ "$DURATION" -ge 1 ] || { echo "STREAMS and SECONDS must be at least 1" >&2; exit 2; }
[ -d "$DIR" ] || { echo "STATE_DIR $DIR does not exist" >&2; exit 2; }

ROUTER_PID=""
FLEET_PID=""
LOAD_PID=""
cleanup() {
    trap - EXIT
    if [ -n "$LOAD_PID" ]; then kill "$LOAD_PID" 2>/dev/null; fi
    if [ -n "$FLEET_PID" ]; then kill "$FLEET_PID" 2>/dev/null; wait "$FLEET_PID" 2>/dev/null; fi
    if [ -n "$ROUTER_PID" ]; then kill "$ROUTER_PID" 2>/dev/null; wait "$ROUTER_PID" 2>/dev/null; echo "router stopped"; fi
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
# the token is read from its file by python, so it is never in a command line
admin_get() {
    "$PY" -c 'import sys, urllib.request
tok = open(sys.argv[2]).read().strip()
req = urllib.request.Request(sys.argv[1], headers={"Authorization": "Bearer " + tok})
sys.stdout.write(urllib.request.urlopen(req, timeout=5).read().decode())' "http://127.0.0.1:$ADMIN_PORT/admin/replicas" "$DIR/admin.token"
}
replica_count() {
    admin_get 2>/dev/null | "$PY" -c 'import json, sys
print(len(json.load(sys.stdin)))' 2>/dev/null || echo 0
}

# the replica key: urls --show-secret prints it once an instance runs; before that it is only in state.json
OUT=$(python3 "$CANARY" --dir "$DIR" urls --show-secret 2>/dev/null) || OUT=""
SECRET=$(printf '%s\n' "$OUT" | sed -n 's/^secret: //p')
OUT=""
case "$SECRET" in
    ""|"(none)"|"(hidden)")
        SECRET=$(python3 -c 'import json, sys
print(json.load(open(sys.argv[1] + "/state.json")).get("secret") or "")' "$DIR" 2>/dev/null) || SECRET="" ;;
esac
[ -n "$SECRET" ] || { echo "no instance secret in $DIR (rent with a profile that uses {KEY})" >&2; exit 1; }

# admin token: new for every run, readable by root only
( umask 077; "$PY" -c 'import secrets
print(secrets.token_urlsafe(32))' >"$DIR/admin.token" ) || { echo "cannot write the admin token" >&2; exit 1; }
chmod 600 "$DIR/admin.token"

# the router starts with no replicas; fleet.py fills the registry through the admin API
(
    cd "$ROUTER_DIR" || exit 1
    UNDEAD_REPLICA_API_KEY=$SECRET
    export UNDEAD_REPLICA_API_KEY
    exec "$PY" router.py --model "$MODEL" --host 127.0.0.1 --port "$ROUTER_PORT" \
        --admin-host 127.0.0.1 --admin-port "$ADMIN_PORT" --admin-token-file "$DIR/admin.token" \
        --journal "$DIR/journal.jsonl" --queue-timeout 10
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

"$PY" "$FLEET" --dir "$DIR" --router-admin "http://127.0.0.1:$ADMIN_PORT" --token-file "$DIR/admin.token" \
    >"$DIR/fleet.out" 2>&1 &
FLEET_PID=$!

# wait for two replicas in the registry
T=0
while [ "$(replica_count)" -lt 2 ]; do
    if ! kill -0 "$ROUTER_PID" 2>/dev/null; then echo "router died, see $DIR/router.log" >&2; ROUTER_PID=""; exit 1; fi
    if ! kill -0 "$FLEET_PID" 2>/dev/null; then echo "fleet.py died, see $DIR/fleet.out" >&2; FLEET_PID=""; exit 1; fi
    T=$((T + 1))
    if [ "$T" -gt "$WAIT_TRIES" ]; then
        echo "fewer than two replicas after 25 minutes; stopping (instances are NOT destroyed)" >&2
        exit 3
    fi
    sleep 5
done
echo "two replicas are registered, starting the load"

date -u +%Y-%m-%dT%H:%M:%S.%3NZ >"$DIR/load-start.txt"
"$PY" "$LOAD" --url "http://127.0.0.1:$ROUTER_PORT" --model "$MODEL" --streams "$STREAMS" --duration "$DURATION" \
    --max-tokens 500 --out "$DIR/run.jsonl" >"$DIR/load.out" 2>&1 &
LOAD_PID=$!
wait "$LOAD_PID"
LOAD_RC=$?
LOAD_PID=""

http_get "http://127.0.0.1:$ROUTER_PORT/metrics" >"$DIR/metrics-after.json" || echo "could not read /metrics" >&2
admin_get >"$DIR/replicas-after.json" || echo "could not read /admin/replicas" >&2
kill "$FLEET_PID" 2>/dev/null
wait "$FLEET_PID" 2>/dev/null
FLEET_PID=""
kill "$ROUTER_PID" 2>/dev/null
wait "$ROUTER_PID" 2>/dev/null
ROUTER_PID=""
echo "router and fleet.py stopped (load.py exit code $LOAD_RC)"

echo "--- load-start.txt"
cat "$DIR/load-start.txt"
echo "--- load.out"
cat "$DIR/load.out"
exit "$LOAD_RC"
