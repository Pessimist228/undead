#!/usr/bin/env bash
# Runs and records the demo. Needs two replicas and the router already up.
#   left pane:  an answer streamed through the router, then the same request again as a control
#   right pane: kill -9 of the replica that is serving the answer
# Usage: demo/run_demo.sh OUT.cast [router-url] [seconds-before-kill]
OUT=${1:-demo.cast}
ROUTER=${2:-http://127.0.0.1:8090}
KILL_AFTER=${3:-2.5}
PY=${PY:-python3}
S=undead-demo
COLS=118; ROWS=42
# A private tmux server with a UTF-8 locale, so pane borders are drawn the same everywhere.
export LC_ALL=C.UTF-8 LANG=C.UTF-8
T="tmux -L undead-demo -u"
cd "$(dirname "$0")/.." || exit 1

active() { curl -s "$ROUTER/metrics" | $PY -c 'import json,sys; print(len(json.load(sys.stdin)["active"]))'; }
wait_done() { sleep 0.5; while [ "$(active)" != "0" ]; do sleep 0.2; done; }

$T kill-session -t $S 2>/dev/null
$T new-session -d -s $S -x $COLS -y $ROWS
$T set-option -t $S status off
$T split-window -h -t $S -l 38
$T set-option -t $S pane-border-style fg=colour240
$T set-option -t $S pane-active-border-style fg=colour240
$T set-option -t $S pane-border-status top
$T set-option -t $S pane-border-format " #{pane_title} "
$T select-pane -t $S:0.0 -T "client -> router -> two vLLM replicas"
$T select-pane -t $S:0.1 -T "operator"
# Some images start every shell in another directory and put the host name into the pane title.
ROOT=$(pwd)
for p in 0 1; do
  $T send-keys -t $S:0.$p "cd $ROOT; unset PROMPT_COMMAND; PS1='\$ '; export UNDEAD_ROUTER=$ROUTER; clear" Enter
done
sleep 1.5
$T select-pane -t $S:0.0 -T "client -> router -> two vLLM replicas"
$T select-pane -t $S:0.1 -T "operator"
$T select-pane -t $S:0.0
sleep 0.5
$PY demo/rec.py "$OUT" $COLS $ROWS -- tmux -L undead-demo -u attach -t $S &
REC=$!
sleep 1.5
$T send-keys -t $S:0.0 "$PY demo/client.py --seed 42 --save-hash /tmp/h" Enter
sleep "$KILL_AFTER"
$T send-keys -t $S:0.1 "demo/kill_serving.sh" Enter
wait_done
sleep 4
$T send-keys -t $S:0.0 "clear; $PY demo/client.py --seed 42 --compare-hash /tmp/h" Enter
wait_done
sleep 5
$T kill-session -t $S
wait $REC
echo "recorded: $OUT"
