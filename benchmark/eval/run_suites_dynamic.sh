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
MODELS=${DYN_MODELS:-"torchbench/pyhpc_equation_of_state:batch torchbench/mobilenet_v2:batch torchbench/hf_Bert:seq torchbench/hf_GPT2:seq torchbench/resnet50:batch torchbench/timm_vision_transformer:batch torchbench/densenet121:batch"}
ARMS=${DYN_ARMS:-"ours ours-fixed ours-spec eager compile-static compile-default compile-dynamic"}
SCRATCH=${DYN_SCRATCH:-$HOME/.cache/dyn-scratch}
MEMMAX=${SUITES_MEMMAX-40G}
capped() {
  if [ -n "$MEMMAX" ] && command -v systemd-run >/dev/null; then
    systemd-run --user --scope --quiet --collect -p MemoryMax="$MEMMAX" \
      -p MemorySwapMax=0 -- "$@"
  else
    "$@"
  fi
}

TSV="$OUT/dynamic_suites.tsv"
COLS="median_ms min_ms first_ms loops bridges kernels compiles launches cpu_fallback graphs"
tsv_init "$TSV" "suite\tmodel\tarm\tcache\trun\tphase\tsize\t$(echo $COLS | tr ' ' '\t')\tbinary"
CTSV="$OUT/dynamic_check.tsv"
tsv_init "$CTSV" "suite\tmodel\tarm\tpoint\tpass"
PYPY_SHA=$(binary_sha "$P")
TORCH_VER=$("$SP" -c "import torch; print(torch.__version__)" 2>/dev/null | tail -1)
mkdir -p "$SCRATCH"

sizes_of() {
  python3 - "$1" "$2" <<'EOF'
import json, math, sys
idx = json.load(open(sys.argv[1] + "/index.json"))
axis = {"batch": 0, "seq": 1}[sys.argv[2]]
e = (idx["inputs"] + list(idx["kwargs"].values()))[0]
s0 = e[1][axis]
def r(f):
    v = math.ceil(f * s0)
    return int(math.ceil(v / 8.0) * 8) if sys.argv[2] == "seq" else int(v)
first = [r(f) for f in (1, 3/4., 1/2., 3/8., 1/4.)]
unseen = [r(f) for f in (7/8., 5/8., 7/16., 5/16., 3/16.)]
f, u = ",".join(map(str, first)), ",".join(map(str, unseen))
print("%s:first=%s;revisit=%s;unseen=%s %s" % (sys.argv[2], f, f, u, " ".join(map(str, first + unseen))))
EOF
}

word_of() { echo "$1" | grep -o "$2=[^ ]*" | head -1 | cut -d= -f2; }

emit() {
  local suite=$1 model=$2 arm=$3 cache=$4 run=$5 bin=$6 out=$7
  echo "$out" | grep '^port_size ' | while read -r line; do
    vals=""
    for c in $COLS; do vals+="\t$(field_of "$line" "$c")"; done
    echo -e "$suite\t$model\t$arm\t$cache\t$run\t$(word_of "$line" phase)\t$(field_of "$line" size)$vals\t$bin" >> "$TSV"
  done
  [ -n "$(echo "$out" | grep '^port_size ')" ] || echo "run_suites_dynamic.sh: $suite/$model $arm $cache run $run: no rows" >&2
}

ours() {
  local cache=$1 dir=$2 spec=$3; shift 3
  TMPDIR=$dir TRITON_CACHE_DIR=$dir/triton capped env "$@" $P $JIT "$SUITES/run_port.py" "$X" \
    --repeat "$REPEAT" --sizes "$spec" $dumparg 2>"$X.dyn.err"
}

check() {
  local arm=$1 dump=$2
  capped "$SP" "$HERE/dyn_check.py" "$suite" "$model" "$dump" "$X.dyn-eager" 2>>"$X.dyn.check.err" \
    | grep '^dyncheck ' | while read -r line; do
      echo -e "$suite\t$model\t$arm\t$(word_of "$line" point)\t$(field_of "$line" pass)" >> "$CTSV"
    done
}

nsteps=0
for sm in $MODELS; do nsteps=$((nsteps + RUNS * 5 + 10 * (RUNS + 1) + 10 * ${DYN_SPEC_RUNS:-1})); done
progress_init dynamic "$nsteps"
for sm in $MODELS; do
  axis=${sm##*:} sm=${sm%:*}
  suite=${sm%%/*} model=${sm#*/}
  X="$EXPORTS/${suite}_${model//\//_}"
  read -r spec distinct < <(sizes_of "$X" "$axis")
  rm -rf "$X.dyn-eager" "$X.dyn-ours" "$X.dyn-spec"
  for run in $(seq "$RUNS"); do
    for arm in eager compile-static compile-default compile-dynamic; do
      [[ " $ARMS " == *" $arm "* ]] || continue
      progress_step "$sm $arm run $run"
      d=$(mktemp -d -p "$SCRATCH")
      dumparg=""
      [ "$arm" = eager ] && [ "$run" = 1 ] && dumparg="--dump $X.dyn-eager"
      out=$(TORCHINDUCTOR_CACHE_DIR=$d/inductor TRITON_CACHE_DIR=$d/triton capped "$SP" \
            "$SUITES/run_torch_dynamic.py" "$suite" "$model" "$arm" --sizes "$spec" \
            --repeat "$REPEAT" $dumparg 2>"$X.dyn.err") || true
      emit "$suite" "$model" "$arm" cold "$run" "$TORCH_VER" "$out"
      rm -rf "$d"
    done
    if [[ " $ARMS " == *" ours "* ]]; then
      progress_step "$sm ours run $run"
      d=$(mktemp -d -p "$SCRATCH")
      dumparg=""
      [ "$run" = 1 ] && dumparg="--dump $X.dyn-ours"
      emit "$suite" "$model" ours cold "$run" "$PYPY_SHA" "$(ours cold "$d" "$spec" || true)"
      rm -rf "$d"
      [ "$run" = 1 ] && [ -d "$X.dyn-ours" ] && check ours "$X.dyn-ours"
    fi
  done
  for size in $distinct; do
    if [[ " $ARMS " == *" ours-fixed "* ]]; then
      warm=$(mktemp -d -p "$SCRATCH")
      dumparg=""
      progress_step "$sm ours-fixed $size cold"
      emit "$suite" "$model" ours-fixed cold 1 "$PYPY_SHA" "$(ours cold "$warm" "$axis:fixed=$size" || true)"
      for run in $(seq "$RUNS"); do
        progress_step "$sm ours-fixed $size run $run"
        emit "$suite" "$model" ours-fixed warm "$run" "$PYPY_SHA" "$(ours warm "$warm" "$axis:fixed=$size" || true)"
      done
      rm -rf "$warm"
    fi
    if [[ " $ARMS " == *" ours-spec "* ]]; then
      warm=$(mktemp -d -p "$SCRATCH")
      for run in $(seq "${DYN_SPEC_RUNS:-1}"); do
        progress_step "$sm ours-spec $size run $run"
        dumparg=""
        [ "$run" = 1 ] && dumparg="--dump $X.dyn-spec"
        emit "$suite" "$model" ours-spec warm "$run" "$PYPY_SHA" \
          "$(ours warm "$warm" "$axis:fixed=$size" RTENSOR_STATIC_SIZES=1000000 || true)"
      done
      rm -rf "$warm"
    fi
  done
  if [ -d "$X.dyn-spec" ]; then
    for p in "$X.dyn-spec"/fixed-*; do
      [ -d "$p" ] || continue
      s=${p##*-}
      for ph in first unseen; do
        [ -d "$X.dyn-eager/$ph-$s" ] && mv "$p" "$X.dyn-spec/$ph-$s" && break
      done
    done
    check ours-spec "$X.dyn-spec"
  fi
  rm -rf "$X.dyn-eager" "$X.dyn-ours" "$X.dyn-spec"
done
progress_done
echo "wrote $TSV"
