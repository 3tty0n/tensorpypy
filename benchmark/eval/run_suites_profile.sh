#!/bin/bash
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/config.sh"
SUITES="$HERE/../suites"

SP=${SUITES_PYTHON:-$HOME/.venvs/torchbench/bin/python}
P=${SUITES_PYPY:-$PYPY}
EXPORTS=${SUITES_EXPORTS:-$HOME/.cache/metatensor-suites}
REPEAT=${SUITES_REPEAT:-30}
JIT=${SUITES_JIT:-"--jit threshold=3,function_threshold=3,trace_eagerness=2,trace_limit=200000"}
NSYS=${NSYS:-/usr/local/cuda/bin/nsys}
SYSTEMS=${PROF_SYSTEMS:-"ours inductor inductor-cg"}
WORK=${PROF_WORK:-$HOME/.cache/metatensor-prof}
MEMMAX=${SUITES_MEMMAX-40G}
capped() {
  if [ -n "$MEMMAX" ] && command -v systemd-run >/dev/null; then
    systemd-run --user --scope --quiet --collect -p MemoryMax="$MEMMAX" \
      -p MemorySwapMax=0 -- "$@"
  else
    "$@"
  fi
}
MODELS=${SUITES_MODELS:-$("$P" -c "
import sys; sys.argv = ['x']
sys.path.insert(0, '$SUITES')
import run_port
print(' '.join('%s/%s' % k for k in sorted(run_port.MODELS)))" 2>/dev/null)}

COLS="iters wall_ms busy_ms host_ms events matmul_ms conv_ms attention_ms reduce_ms pointwise_ms layout_ms other_ms matmul_n conv_n attention_n reduce_n pointwise_n layout_n other_n"
TSV="$OUT/profile.tsv"
tsv_init "$TSV" "suite\tmodel\tsystem\tmedian_ms\t$(echo $COLS | tr ' ' '\t')\tbinary"
PYPY_SHA=$(binary_sha "$P")
TORCH_VER=$("$SP" -c "import torch; print(torch.__version__)" 2>/dev/null | tail -1)
mkdir -p "$WORK"
NSYS_ARGS="profile -t cuda --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite true"

progress_init profile $(( $(echo $MODELS | wc -w) * $(echo $SYSTEMS | wc -w) ))
for sm in $MODELS; do
  suite=${sm%%/*} model=${sm#*/}
  name=${suite}_${model//\//_}
  X="$EXPORTS/$name"
  [ -f "$X/index.json" ] || continue
  for system in $SYSTEMS; do
    progress_step "$sm $system"
    rep="$WORK/$name.$system"
    if [ -n "$PROF_REUSE" ] && [ -f "$rep.sqlite" ]; then
      kdir=-
      [ "$system" = ours ] && kdir="$WORK/$name.kernels"
      line="median_ms=$(grep -h -o 'median_ms=[0-9.]*' "$rep.line" 2>/dev/null | cut -d= -f2)"
      bin=$(cat "$rep.bin" 2>/dev/null)
      prof=$(python3 "$HERE/profile_summary.py" "$rep.sqlite" "$kdir" "$REPEAT" 2>>"$rep.err" | grep '^profile ' | tail -1)
      vals=""
      for c in $COLS; do vals+="\t$(field_of "$prof" "$c")"; done
      echo -e "$suite\t$model\t$system\t$(field_of "$line" median_ms)$vals\t$bin" >> "$TSV"
      continue
    fi
    rm -f "$rep.nsys-rep" "$rep.sqlite"
    if [ "$system" = ours ]; then
      kdir="$WORK/$name.kernels"
      rm -rf "$kdir"
      mkdir -p "$kdir"
      cp ${TMPDIR:-/tmp}/rtensor-tune-* "$kdir"/ 2>/dev/null || true
      line=$(TMPDIR=$kdir capped "$NSYS" $NSYS_ARGS -o "$rep" "$P" $JIT "$SUITES/run_port.py" "$X" \
             --repeat "$REPEAT" --profile 2>"$rep.err" | grep '^port ' | tail -1) || true
      bin=$PYPY_SHA
    else
      kdir=-
      line=$(capped "$NSYS" $NSYS_ARGS -o "$rep" "$SP" "$SUITES/run_torch.py" "$suite" "$model" \
             --backends "$system" --repeat "$REPEAT" --profile 2>"$rep.err" | grep '^torch ' | tail -1) || true
      bin=$TORCH_VER
    fi
    if [ -z "$line" ] || [ ! -f "$rep.nsys-rep" ]; then
      echo "run_suites_profile.sh: $sm $system: no run" >&2
      continue
    fi
    "$NSYS" export --type sqlite --force-overwrite true -o "$rep.sqlite" "$rep.nsys-rep" >/dev/null 2>&1 || true
    echo "$line" > "$rep.line"
    echo "$bin" > "$rep.bin"
    prof=$(python3 "$HERE/profile_summary.py" "$rep.sqlite" "$kdir" "$REPEAT" 2>>"$rep.err" | grep '^profile ' | tail -1)
    vals=""
    for c in $COLS; do vals+="\t$(field_of "$prof" "$c")"; done
    echo -e "$suite\t$model\t$system\t$(field_of "$line" median_ms)$vals\t$bin" >> "$TSV"
    rm -f "$rep.nsys-rep"
  done
done
progress_done
echo "wrote $TSV"
