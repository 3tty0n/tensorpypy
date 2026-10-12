#!/bin/bash
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/config.sh"
SUITES="$HERE/../suites"

SP=${SUITES_PYTHON:-$HOME/.venvs/torchbench/bin/python}
P=${SUITES_PYPY:-$PYPY}
EXPORTS=${SUITES_EXPORTS:-$HOME/.cache/metatensor-suites}
RUNS=${SUITES_RUNS:-3}
REPEAT=${SUITES_REPEAT:-30}
JIT=${SUITES_JIT:-"--jit threshold=3,function_threshold=3,trace_eagerness=2,trace_limit=200000"}
NOFUSE="enable_opts=intbounds:rewrite:virtualize:string:pure:earlyforce:heap:unroll"
MODELS=${ABL_MODELS:-"torchbench/resnet50 torchbench/mobilenet_v2 torchbench/hf_Bert torchbench/hf_GPT2 torchbench/timm_vision_transformer torchbench/pyhpc_equation_of_state torchbench/densenet121"}
ARMS=${ABL_ARMS:-"fused nofuse nojit deferred mi4 mi6 mi8"}
TIMEOUT=${ABL_TIMEOUT:-3600}
MEMMAX=${SUITES_MEMMAX-40G}
[ -x "$P" ] || { echo "run_suites_ablation.sh: no interpreter at $P" >&2; exit 1; }
capped() {
  if [ -n "$MEMMAX" ] && command -v systemd-run >/dev/null; then
    systemd-run --user --scope --quiet --collect -p MemoryMax="$MEMMAX" \
      -p MemorySwapMax=0 -- "$@"
  else
    "$@"
  fi
}

arm_cmd() {
  case $1 in
    fused)    echo "$P $JIT";;
    nofuse)   echo "$P $JIT,$NOFUSE";;
    nojit)    echo "$P --jit off";;
    deferred) echo "METATENSOR_LAZY=1 $P $JIT,$NOFUSE";;
    mi*)      echo "RTENSOR_MAX_INPUTS=${1#mi} $P $JIT";;
    split-branch) echo "METATENSOR_SPLIT=1 $P $JIT";;
    split-all)    echo "METATENSOR_SPLIT=3 $P $JIT";;
  esac
}

TSV="$OUT/ablation_suites.tsv"
COLS="batch median_ms min_ms first_ms launches cpu_fallback gc_ms lazy_nodes loops bridges kernels compiles"
tsv_init "$TSV" "suite\tmodel\tarm\trun\t$(echo $COLS | tr ' ' '\t')\tpass\trmse\tbinary"
PYPY_SHA=$(binary_sha "$P")
STATUS=0

progress_init ablation_suites $(( $(echo $MODELS | wc -w) * RUNS * $(echo $ARMS | wc -w) ))
for sm in $MODELS; do
  suite=${sm%%/*} model=${sm#*/}
  X="$EXPORTS/${suite}_${model//\//_}"
  [ -f "$X/index.json" ] || { echo "run_suites_ablation.sh: no export at $X (run_suites.sh)" >&2; STATUS=1; continue; }
  for run in $(seq "$RUNS"); do
    for arm in $ARMS; do
      progress_step "$sm $arm run $run/$RUNS"
      dumparg="" pass="" rmse=""
      [ "$run" = 1 ] && rm -rf "$X.abl" && dumparg="--dump $X.abl"
      stats=1
      [ "$arm" = nojit ] && stats=
      line=$(RTENSOR_LAZY_STATS=$stats capped timeout "$TIMEOUT" env $(arm_cmd "$arm") \
             "$SUITES/run_port.py" "$X" --repeat "$REPEAT" $dumparg \
             2>"$X.$arm.err" | grep '^port ' | tail -1) || true
      if [ "$run" = 1 ] && [ -n "$line" ]; then
        check=$(capped "$SP" "$HERE/suite_check.py" "$suite" "$model" "$X.abl" \
                --export "$X" 2>"$X.$arm.check.err" | grep '^check ' | tail -1) || true
        pass=$(field_of "$check" pass) rmse=$(field_of "$check" rmse)
        [ -n "$pass" ] || { pass=0; echo "run_suites_ablation.sh: $sm $arm check: $(tail -1 "$X.$arm.check.err")" >&2; }
        rm -rf "$X.abl"
      fi
      if [ -z "$line" ]; then
        echo "run_suites_ablation.sh: $sm $arm run $run: $(grep -m1 -E 'Error|error' "$X.$arm.err" || tail -1 "$X.$arm.err")" >&2
        STATUS=1
      fi
      vals=""
      for c in $COLS; do vals+="\t$(field_of "$line" "$c")"; done
      echo -e "$suite\t$model\t$arm\t$run$vals\t$pass\t$rmse\t$PYPY_SHA" >> "$TSV"
    done
  done
done
progress_done
echo "wrote $TSV"
exit $STATUS
