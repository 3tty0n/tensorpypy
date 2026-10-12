#!/bin/bash
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/config.sh"
APP="$HERE/../applevel"
SUITES="$HERE/../suites"

P=${SUITES_PYPY:-$PYPY}
JIT=${SUITES_JIT:-"--jit threshold=3,function_threshold=3,trace_eagerness=2,trace_limit=200000"}
EXPORTS=${SUITES_EXPORTS:-$HOME/.cache/metatensor-suites}
MODES=${SPLIT_MODES:-"0 1 3"}
PHASES=${SPLIT_PHASES:-"micro control spans"}
PYPY_SHA=$(binary_sha "$P")
MEMMAX=${SUITES_MEMMAX-40G}
capped() {
  if [ -n "$MEMMAX" ] && command -v systemd-run >/dev/null; then
    systemd-run --user --scope --quiet --collect -p MemoryMax="$MEMMAX" \
      -p MemorySwapMax=0 -- "$@"
  else
    "$@"
  fi
}

do_micro() {
  local tsv="$OUT/split_micro.tsv"
  tsv_init "$tsv" "variant\tk\tn\tsplit\tround\tsteady_us\tlaunches\tacc\tbinary"
  for round in $(seq "$ROUNDS"); do
    for v in 0 1 2 3 4 5; do
      for m in $MODES; do
        progress_step "micro v$v split=$m round $round"
        line=$(METATENSOR_SPLIT=$m "$P" $JIT "$APP/micro.py" app "$v" 4 1000000 "$ITERS" "$WARMUP" 2>/dev/null | tail -1)
        set -- $line
        echo -e "$v\t4\t1000000\t$m\t$round\t$7\t${11}\t$9\t$PYPY_SHA" >> "$tsv"
      done
    done
  done
}

do_control() {
  local tsv="$OUT/split_control.tsv"
  tsv_init "$tsv" "regime\tsplit\tround\ttotal_ms\tp50_us\tp95_us\tloops\tbridges\tkernels\texit_hash\tpass\tbinary"
  for round in $(seq "$ROUNDS"); do
    for regime in stable varying; do
      for m in $MODES; do
        progress_step "control $regime split=$m round $round"
        row=$(cd "$APP" && METATENSOR_SPLIT=$m MODEL_TOL=$(tolerance_for distilgpt2) \
              "$P" $JIT "$APP/earlyexit.py" "$WEIGHTS/distilgpt2" "$regime" "$ITERS" "$WARMUP" 2>/dev/null | grep '^control ' | tail -1)
        echo -e "$regime\t$m\t$round$(for f in total_ms p50_us p95_us loops bridges kernels exit_hash pass; do printf '\t%s' "$(field_of "$row" $f)"; done)\t$PYPY_SHA" >> "$tsv"
      done
    done
  done
}

do_spans() {
  local tsv="$OUT/split_spans.tsv"
  tsv_init "$tsv" "suite\tmodel\tkind\tguard\tspans\tbinary"
  for X in "$EXPORTS"/*/index.json; do
    X=$(dirname "$X")
    local sm
    sm=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['suite'], d['model'])" "$X/index.json")
    set -- $sm
    progress_step "spans $1/$2"
    local log="$OUT/.spans.log"
    rm -f "$log"
    PYPYLOG=jit-tensor-span:$log capped "$P" $JIT "$SUITES/run_port.py" "$X" --repeat 3 >/dev/null 2>&1 || true
    [ -f "$log" ] || continue
    grep '^span ' "$log" | awk -v s="$1" -v m="$2" -v b="$PYPY_SHA" \
      '{c[$2"\t"$3]++} END {for (k in c) print s"\t"m"\t"k"\t"c[k]"\t"b}' >> "$tsv"
    rm -f "$log"
  done
}

n=0
for p in $PHASES; do
  case $p in
    micro) n=$((n + ROUNDS * 6 * $(echo $MODES | wc -w)));;
    control) n=$((n + ROUNDS * 2 * $(echo $MODES | wc -w)));;
    spans) n=$((n + $(ls -d "$EXPORTS"/*/index.json | wc -l)));;
  esac
done
progress_init split "$n"
for p in $PHASES; do "do_$p"; done
progress_done
