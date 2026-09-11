#!/usr/bin/env bash
# Pod-side sweep driver. Runs every "planned" run in a manifest sequentially, then
# self-terminates the pod. Designed so the sweep completes and the pod dies even if the
# Claude session that started it disappears.
#
#   ./scripts/sweep_driver.sh [manifest]     # default spec/runs/lr-sweep.json
#
# Started detached by the agent:
#   setsid ./scripts/sweep_driver.sh > sweep_driver.log 2>&1 < /dev/null &
#
# Termination happens when ANY of these is true:
#   * all runs finished AND ./outputs_pulled exists (agent pulled results back)
#   * all runs finished AND GRACE_MIN minutes elapsed (agent session died)
#   * pod alive longer than MAX_HOURS (hard budget stop, even mid-run)
set -u

REPO="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST="${1:-spec/runs/lr-sweep.json}"
cd "$REPO" || exit 1

LOG="$REPO/sweep_driver.log"
START=$(date +%s)
GRACE_MIN=${GRACE_MIN:-45}
MAX_HOURS=${MAX_HOURS:-$(python3 -c "import json;print(json.load(open('$MANIFEST'))['budget'].get('max_pod_hours',4))" 2>/dev/null || echo 4)}

log() { echo "$(date -u +%FT%TZ) [driver] $*" | tee -a "$LOG" >&2; }

terminate() {
  local why="$1"
  log "TERMINATING pod ${RUNPOD_POD_ID:-unknown}: $why"
  sync
  local id="${RUNPOD_POD_ID:-}"
  if [ -z "$id" ]; then log "ERROR: RUNPOD_POD_ID unset; cannot self-terminate"; exit 1; fi

  if command -v runpodctl >/dev/null 2>&1; then
    runpodctl remove pod "$id"  >>"$LOG" 2>&1 && { log "terminated via runpodctl remove"; exit 0; }
    runpodctl pod delete "$id"  >>"$LOG" 2>&1 && { log "terminated via runpodctl pod delete"; exit 0; }
  fi
  local key="${RUNPOD_API_KEY:-${RUNPOD_AI_API_KEY:-}}"
  if [ -n "$key" ]; then
    curl -fsS -X DELETE "https://rest.runpod.io/v1/pods/$id" \
      -H "Authorization: Bearer $key" >>"$LOG" 2>&1 && { log "terminated via REST v2"; exit 0; }
    curl -fsS -X POST "https://api.runpod.io/graphql" \
      -H "Content-Type: application/json" -H "Authorization: Bearer $key" \
      -d "{\"query\":\"mutation { podTerminate(input: {podId: \\\"$id\\\"}) }\"}" >>"$LOG" 2>&1 \
      && { log "terminated via GraphQL"; exit 0; }
  fi
  log "ERROR: all termination methods failed. Pod is STILL BILLING - delete it manually."
  touch "$REPO/TERMINATION_FAILED"
  exit 1
}

# --- hard budget stop, runs in parallel with the sweep ---
(
  while true; do
    sleep 300
    if [ $(( ($(date +%s) - START) / 3600 )) -ge "$MAX_HOURS" ]; then
      log "MAX_HOURS=$MAX_HOURS reached while runs in flight"
      terminate "MAX_HOURS=$MAX_HOURS budget cap"
    fi
  done
) &
KILLER=$!
trap 'kill $KILLER 2>/dev/null' EXIT

planned=$(python3 -c "
import json
m = json.load(open('$MANIFEST'))
print(' '.join(r['name'] for r in m['runs'] if r.get('status') == 'planned'))
")
log "manifest=$MANIFEST planned runs: ${planned:-<none>}  max_hours=$MAX_HOURS grace=${GRACE_MIN}m"
[ -z "$planned" ] && { log "nothing planned; terminating"; terminate "no planned runs"; }

for run in $planned; do
  meta="$REPO/train_${run}.meta.json"
  if [ -f "$REPO/done_${run}" ]; then log "$run already marked done; skipping"; continue; fi

  log "launching $run"
  if ! python3 scripts/launch_run.py "$run" --manifest "$MANIFEST" >>"$LOG" 2>&1; then
    log "LAUNCH REFUSED for $run - see log above. Halting sweep (failure policy: no retry)."
    break
  fi
  pid=$(python3 -c "import json;print(json.load(open('$meta'))['pid'])" 2>/dev/null)
  log "$run running as pid=$pid"

  while ps -p "$pid" >/dev/null 2>&1; do sleep 30; done

  if grep -q "TEST SET RESULTS" "$REPO/train_${run}.log" 2>/dev/null; then
    touch "$REPO/done_${run}"
    log "$run finished (test-eval block present in log)"
  else
    log "$run EXITED WITHOUT test results - treating as failure. Halting sweep (no retry)."
    touch "$REPO/failed_${run}"
    break
  fi
done

log "sweep finished; waiting for outputs_pulled marker (grace ${GRACE_MIN}m)"
WAIT_START=$(date +%s)
while true; do
  [ -f "$REPO/outputs_pulled" ] && terminate "outputs_pulled marker present"
  elapsed_min=$(( ($(date +%s) - WAIT_START) / 60 ))
  [ "$elapsed_min" -ge "$GRACE_MIN" ] && terminate "grace period ${GRACE_MIN}m elapsed with no marker"
  sleep 60
done
