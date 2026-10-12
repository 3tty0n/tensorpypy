#!/bin/bash
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)

if [ -f "$HERE/env.sh" ]; then
  source "$HERE/env.sh"
fi

PYPY=${PYPY:-$HERE/build/pypy-c}
BENCH=${BENCH:-$HERE/build/metatensor-bench}
TORCH_PYTHON=${TORCH_PYTHON:-${RTENSOR_PYTHON:-}}
RTENSOR_PYTHON=${RTENSOR_PYTHON:-$TORCH_PYTHON}
export RTENSOR_PYTHON TORCH_PYTHON
# Optional JAX/XLA + IREE venv (setup.sh writes it into env.sh). Unset or
# missing means those rows are skipped, nothing else changes.
JAX_PYTHON=${JAX_PYTHON:-}
if [ -n "$JAX_PYTHON" ] && ! [ -x "$JAX_PYTHON" ]; then
  echo "config.sh: JAX_PYTHON=$JAX_PYTHON is not executable; jax/iree rows skipped" >&2
  JAX_PYTHON=""
fi
export JAX_PYTHON
# Optional Torch-TensorRT venv (CPython 3.13: torch-tensorrt has no 3.14 wheel).
TRT_PYTHON=${TRT_PYTHON:-}
if [ -n "$TRT_PYTHON" ] && ! [ -x "$TRT_PYTHON" ]; then
  echo "config.sh: TRT_PYTHON=$TRT_PYTHON is not executable; tensorrt rows skipped" >&2
  TRT_PYTHON=""
fi
export TRT_PYTHON

if [ -z "$RTENSOR_CUBLAS" ] && [ -n "$RTENSOR_PYTHON" ]; then
  for f in "$(dirname "$RTENSOR_PYTHON")"/../lib/python3*/site-packages/nvidia/cu*/lib/libcublas.so.*; do
    [ -e "$f" ] && export RTENSOR_CUBLAS="$f" && break
  done
fi

WEIGHTS=${WEIGHTS:-$HERE/weights}
ITERS=${ITERS:-200}
WARMUP=${WARMUP:-30}
ROUNDS=${ROUNDS:-3}
# Correctness tolerance per (workload class, dtype), as a fraction of the
# reference output's largest absolute value. run_models.sh exports it as
# MODEL_TOL; each system's compare() multiplies it by the model's reference
# magnitude and prints "tol=<t> pass=<0|1>" next to maxabsdiff.
# Relative because output scales differ (gpt2 logits reach 332, ViT class
# scores 12). Each fraction leaves at least 2x margin over every recorded diff.
#   float32 models             2e-5
#   conv models under TF32     2e-3  (resnet18-*: cuDNN runs its convs in TF32)
#   float16                    1e-3
#   float64                    1e-8
# Micro accumulators are not covered here; they keep check.sh's close_enough.
tolerance_for() {
  local model=$1 dt=${RTENSOR_DTYPE:-float32}
  case "$dt" in
    float16) echo 1e-3; return ;;
    float64) echo 1e-8; return ;;
  esac
  case "$model" in
    resnet18-*) echo 2e-3; return ;;
  esac
  echo 2e-5
}
JIT_FLAGS=${JIT_FLAGS:---jit threshold=3,function_threshold=3,trace_eagerness=2,trace_limit=60000}
# Results are filed under the machine and accelerator: results/<host>-<gpu>/grid-<date>.
gpu_name() {
  # nvidia-smi prints NVML errors on stdout with exit status 0.
  local name
  name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
  case "$name" in
    ""|*NVML*|*[Ff]ailed*|*[Ee]rror*) return 1 ;;
  esac
  echo "$name"
}
gpu_slug() {
  local name
  if ! name=$(gpu_name); then
    echo "config.sh: nvidia-smi cannot see the GPU ($(nvidia-smi 2>&1 | head -1));" \
         "a driver/NVML version mismatch needs a reboot or a matching driver package" >&2
    echo "nogpu"
    return
  fi
  echo "$name" | tr 'A-Z' 'a-z' | sed -e 's/nvidia //' -e 's/geforce //' \
                                     -e 's/[^a-z0-9]//g'
}
# The effective fusion input cap, resolved the same way core.max_inputs()
# resolves it, so machine.txt records what the run actually used.
max_inputs_effective() {
  local n=${RTENSOR_MAX_INPUTS:-16}
  case "$n" in ''|*[!0-9]*) n=16 ;; esac
  [ "$n" -lt 4 ] && n=4
  [ "$n" -gt 16 ] && n=16
  echo "$n"
}
# RTENSOR_CC=auto means triton_compile.py asks the GPU; record what it answers,
# not the word "auto".
resolved_cc() {
  local cc=${RTENSOR_CC:-}
  if [ "$cc" = auto ] && [ -n "${RTENSOR_PYTHON:-}" ] && [ -x "$RTENSOR_PYTHON" ]; then
    cc=$("$RTENSOR_PYTHON" "$REPO/rpython/metatensor/triton_compile.py" --cc 2>/dev/null | tail -1)
  fi
  echo "${cc:-unknown}"
}
# A translated pypy-c is a 15 KB launcher that is byte-identical across
# builds; the code is the libpypy-c.so next to it, so that is what identifies
# the binary.
binary_sha() {
  local f=$1
  [ -f "$(dirname "$f")/libpypy-c.so" ] && f="$(dirname "$f")/libpypy-c.so"
  [ -f "$f" ] || { echo unknown; return; }
  sha256sum "$f" | cut -c1-12
}
sha256_of() {
  [ -f "$1" ] || { echo unknown; return; }
  sha256sum "$1" | cut -d' ' -f1
}
RUN_HOST=${RUN_HOST:-$(hostname)}
RUN_GPU=${RUN_GPU:-$(gpu_slug)}
# Exported so every stage of one run shares the first stage's directory; otherwise
# a run crossing midnight splits its tsv files over two dates.
OUT=${OUT:-$HERE/../results/$RUN_HOST-$RUN_GPU/grid-$(date +%F)}
export OUT
export RTENSOR_BUDGET_MB=${RTENSOR_BUDGET_MB:-8}

mkdir -p "$OUT" "$WEIGHTS"

# The directory name carries host and GPU; this carries driver, toolkit, wheels, CPU.
write_machine_txt() {
  local f="$OUT/machine.txt"
  # Never overwrite another machine's provenance: summarize/plot/compare are often
  # pointed at a result directory measured elsewhere.
  if [ -f "$f" ] && ! grep -qx "host          $RUN_HOST" "$f"; then
    return
  fi
  {
    echo "host          $RUN_HOST"
    echo "date          $(date -Iseconds)"
    echo "gpu           $(gpu_name || echo unknown)"
    echo "gpu_memory    $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)"
    echo "compute_cap   ${RTENSOR_CC:-unknown}"
    echo "driver        $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
    echo "cuda_home     ${CUDA_HOME:-unset}"
    echo "cpu           $(sed -n 's/^model name[ \t]*: //p' /proc/cpuinfo | head -1)"
    echo "cpu_threads   $(nproc)"
    echo "kernel        $(uname -sr)"
    if [ -n "$RTENSOR_PYTHON" ] && [ -x "$RTENSOR_PYTHON" ]; then
      "$RTENSOR_PYTHON" - <<'PY' 2>/dev/null || true
import importlib
for mod in ("torch", "triton", "transformers"):
    try:
        print("%-13s %s" % (mod, importlib.import_module(mod).__version__))
    except Exception:
        print("%-13s -" % mod)
PY
    fi
    if [ -n "$JAX_PYTHON" ]; then
      "$JAX_PYTHON" - <<'PY' 2>/dev/null || true
import importlib
for mod, attr in (("jax", "__version__"), ("jaxlib", "__version__"),
                  ("iree.compiler", "version"), ("iree.runtime", "version")):
    try:
        m = importlib.import_module(mod)
        v = getattr(m, attr, None)
        v = getattr(v, "VERSION", v) if v is not None else None
        if v is None:
            from importlib.metadata import version
            v = version(mod.replace("iree.", "iree-base-"))
        print("%-13s %s" % (mod, v))
    except Exception:
        print("%-13s -" % mod)
PY
    else
      echo "jax           - (JAX_PYTHON unset)"
    fi
    if [ -n "$TRT_PYTHON" ]; then
      "$TRT_PYTHON" - <<'PY' 2>/dev/null || true
import importlib
for mod in ("torch_tensorrt", "tensorrt"):
    try:
        print("%-13s %s" % (mod, importlib.import_module(mod).__version__))
    except Exception:
        print("%-13s -" % mod)
print("trt_torch     %s" % importlib.import_module("torch").__version__)
PY
    fi
    echo "iters         ${ITERS}"
    echo "rounds        ${ROUNDS}"
    echo "warmup_models ${WARMUP}"
    echo "warmup_micro  ${WARMUP}"
    echo "max_inputs    $(max_inputs_effective)"
    echo "compute_cap_r $(resolved_cc)"
    if [ -n "$TORCH_PYTHON" ] && [ -x "$TORCH_PYTHON" ]; then
      "$TORCH_PYTHON" - <<'PY' 2>/dev/null || true
import torch
print("torch_tf32_matmul %d" % int(torch.backends.cuda.matmul.allow_tf32))
print("torch_tf32_cudnn  %d" % int(torch.backends.cudnn.allow_tf32))
PY
    else
      echo "torch_tf32_matmul unknown"
      echo "torch_tf32_cudnn  unknown"
    fi
    echo "jax_matmul_precision highest"
    echo "trt_tf32      0"
    echo "pypy_c_sha256 $(sha256_of "$PYPY")"
    echo "bench_sha256  $(sha256_of "$BENCH")"
    echo "budget_mb     ${RTENSOR_BUDGET_MB}"
    echo "jit_flags     ${JIT_FLAGS}"
    echo "commit        $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  } > "$f"
}
write_machine_txt

if [ -z "$PYPY" ]; then
  echo "config.sh: PYPY not set (translated pypy-c with _metatensor)" >&2
fi
if [ -z "$BENCH" ]; then
  echo "config.sh: BENCH not set (translated metatensor-bench)" >&2
fi

# Every measurement also lands in $OUT/results.jsonl as a named-field record.
bench_record() {
  OUT="$OUT" "${RTENSOR_PYTHON:-python3}" "$HERE/record.py" "$@" || true
}

# Ours emits 12 fields and torch 13 (no launch counter, adds graph/break counts),
# so the tail is read by field count. torch, jax, iree and triton append
# compile_ms and first_run_ms (15).
record_micro_line() {
  local line=$1
  # shellcheck disable=SC2086
  set -- $line
  local mode=$1 variant=$2 k=$3 n=$4 iters=$5 warm=$6 steady=$7 kernels=$8
  local acc=$9
  shift 9
  local compiled=$1
  shift
  local extra=("$@")
  local binary; binary=$(binary_for_mode "$mode")
  if [ ${#extra[@]} -ge 5 ]; then
    bench_record micro mode="$mode" variant="$variant" k="$k" n="$n" \
      iters="$iters" warm_s="$warm" steady_us="$steady" kernels="$kernels" \
      acc="$acc" compiled_in_timed="$compiled" warmup="$WARMUP" \
      graphs="${extra[0]}" \
      breaks="${extra[1]}" dtype="${extra[2]}" compile_ms="${extra[3]}" \
      first_run_ms="${extra[4]}" binary="$binary"
  elif [ ${#extra[@]} -ge 3 ]; then
    bench_record micro mode="$mode" variant="$variant" k="$k" n="$n" \
      iters="$iters" warm_s="$warm" steady_us="$steady" kernels="$kernels" \
      acc="$acc" compiled_in_timed="$compiled" warmup="$WARMUP" \
      graphs="${extra[0]}" \
      breaks="${extra[1]}" dtype="${extra[2]}" binary="$binary"
  else
    bench_record micro mode="$mode" variant="$variant" k="$k" n="$n" \
      iters="$iters" warm_s="$warm" steady_us="$steady" kernels="$kernels" \
      acc="$acc" compiled_in_timed="$compiled" warmup="$WARMUP" \
      launches_per_iter="${extra[0]}" dtype="${extra[1]}" binary="$binary"
  fi
}
# Append the same value as micro.tsv's trailing column; mode is the line's first field.
binary_for_line() { binary_for_mode "$(echo "$1" | awk '{print $1}')"; }
# Rows are ragged (ours/app 12 fields, baselines 15) and readers locate dtype by
# column count (micro_values() in plot.py/summarize.py), so pad to 15 first so
# binary always lands in column 16.
tsv_pad_row() {
  local -a f
  IFS=$'\t' read -r -a f <<< "$1"
  while [ "${#f[@]}" -lt 15 ]; do f+=(""); done
  local IFS=$'\t'
  echo "${f[*]}"
}

tsv_init() {
  local path=$1 header=$2 have want
  [ -f "$path" ] || { echo -e "$header" > "$path"; return; }
  # Rows written before a column was added stay readable by name: if the header
  # is a prefix of the current one, update it.
  have=$(head -1 "$path")
  want=$(echo -e "$header")
  if [ "$have" != "$want" ] && [ "${want#"$have"}" != "$want" ]; then
    { echo "$want"; tail -n +2 "$path"; } > "$path.tmp" && mv "$path.tmp" "$path"
  fi
}

# Per-row provenance for micro.tsv/results.jsonl: the binary or package that
# produced the line. Computed once, not per row.
MICRO_BENCH_SHA=$(sha256_of "$BENCH" | cut -c1-12)
MICRO_PYPY_SHA=$(sha256_of "$PYPY" | cut -c1-12)
_mtxt_ver() { awk -v n="$1" '$1==n{print $2; exit}' "$OUT/machine.txt" 2>/dev/null; }
MICRO_TORCH_VER="torch-$(_mtxt_ver torch)"
MICRO_TRITON_VER="triton-$(_mtxt_ver triton)"
MICRO_JAX_VER="jax-$(_mtxt_ver jax)"
MICRO_IREE_VER="iree-$(_mtxt_ver iree.compiler)"
MICRO_TRT_VER="torch_tensorrt-$(_mtxt_ver torch_tensorrt)"
binary_for_mode() {
  case "$1" in
    fused|eager|nojit) echo "${MICRO_BENCH_SHA:-unknown}" ;;
    app) echo "${MICRO_PYPY_SHA:-unknown}" ;;
    torch-tensorrt) echo "${MICRO_TRT_VER:-unknown}" ;;
    torch-*) echo "${MICRO_TORCH_VER:-unknown}" ;;
    jax) echo "${MICRO_JAX_VER:-unknown}" ;;
    iree) echo "${MICRO_IREE_VER:-unknown}" ;;
    triton) echo "${MICRO_TRITON_VER:-unknown}" ;;
    *) echo unknown ;;
  esac
}

steady_of() { echo "$1" | grep -o 'steady_us=[0-9.]*' | head -1 | cut -d= -f2; }
field_of() { echo "$1" | grep -o "$2=[0-9.eE+-]*" | head -1 | cut -d= -f2; }
diff_of() { echo "$1" | grep -o 'maxabsdiff=[0-9.eE+-]*' | head -1 | cut -d= -f2; }

# --- running one measurement command -----------------------------------------
# Shared here because each duplicate grew its own bug (a stdout-tail row picker
# turned TensorRT warnings into garbage rows).

# The measurement row out of a command's stdout. Not `tail -1`: the backends
# print warnings there too. A row starts with the mode and has numeric
# variant/k fields.
bench_row() { awk 'NF>=12 && $2 ~ /^[0-9]+$/ && $3 ~ /^[0-9]+$/' | tail -1; }

# "<name>-<version>" for a module in some venv, or "unknown".
pkg_version() {
  local python=$1 module=$2 name=$3
  [ -n "$python" ] && [ -x "$python" ] || { echo unknown; return; }
  # tail -1: importing torch_tensorrt prints a plugin warning on stdout.
  "$python" -c "import $module; print('$name-' + $module.__version__)" 2>/dev/null |
    tail -1 | grep . || echo unknown
}

# Which optional baselines to measure; a caller narrows it, e.g. BASELINES=tensorrt.
BASELINES=${BASELINES:-"triton tensorrt compile-ro compile-mat jax iree"}
has_baseline() { case " $BASELINES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

# Run one measurement and echo its row, or echo nothing and say on stderr why.
# A baseline that cannot run must not be silently absent or reported as a fallback.
run_measurement() {
  local what=$1; shift
  local err row
  err=$(mktemp)
  row=$("$@" 2>"$err" | bench_row)
  if [ -z "$row" ]; then
    echo "$what: no row: $(grep -m1 -E 'Error|error:|Exception|refusing' "$err" || tail -1 "$err")" >&2
  fi
  rm -f "$err"
  [ -n "$row" ] && echo "$row"
  # A refused or failed baseline is one absent row, not the end of the stage:
  # returning non-zero here kills the caller under `set -e`.
  return 0
}

# The pid keeps one stage's cleanup from deleting the link a concurrently
# running stage is still executing.
EVAL_PYPY_LINK="$REPO/pypy-c-eval.$$"
eval_setup_pypy() {
  if [ -n "$PYPY" ] && [ "$(dirname "$PYPY")" != "$REPO" ]; then
    ln -sf "$PYPY" "$EVAL_PYPY_LINK"
    RUN_PYPY="$EVAL_PYPY_LINK"
  else
    RUN_PYPY="$PYPY"
  fi
}
eval_cleanup_pypy() {
  rm -f "$EVAL_PYPY_LINK"
}

# Progress reporting. bench.sh all redirects stdout+stderr into tee and saves
# the original stderr on fd 3, so the bar keeps its terminal.
PROG_WIDTH=24

_prog_fd() {
  if [ -t 3 ]; then echo 3; else echo 2; fi
}

_prog_hms() {
  printf '%d:%02d:%02d' $(($1 / 3600)) $(($1 % 3600 / 60)) $(($1 % 60))
}

progress_init() {
  PROG_LABEL=$1
  PROG_TOTAL=$2
  PROG_N=0
  PROG_T0=$(date +%s)
}

progress_step() {
  local what=$1 finished=$PROG_N fd el eta bar='' i filled
  [ "${PROG_TOTAL:-0}" -gt 0 ] || return 0
  PROG_N=$((PROG_N + 1))
  fd=$(_prog_fd)
  el=$(( $(date +%s) - PROG_T0 ))
  if [ "$finished" -gt 0 ]; then
    eta=$(_prog_hms $(( el * (PROG_TOTAL - finished) / finished )))
  else
    eta='--:--:--'
  fi
  filled=$(( PROG_WIDTH * finished / PROG_TOTAL ))
  i=0
  while [ $i -lt $PROG_WIDTH ]; do
    if [ $i -lt $filled ]; then bar="$bar#"; else bar="$bar-"; fi
    i=$((i + 1))
  done
  if [ -t "$fd" ]; then
    printf '\r\033[K[%s] %s %d/%d  %s  eta %s' \
      "$bar" "$PROG_LABEL" "$PROG_N" "$PROG_TOTAL" "$what" "$eta" >&"$fd"
  else
    printf '[%s %d/%d] %s  elapsed %s  eta %s\n' \
      "$PROG_LABEL" "$PROG_N" "$PROG_TOTAL" "$what" "$(_prog_hms $el)" "$eta" >&"$fd"
  fi
}

progress_done() {
  local fd el
  [ "${PROG_TOTAL:-0}" -gt 0 ] || return 0
  fd=$(_prog_fd)
  el=$(( $(date +%s) - PROG_T0 ))
  [ -t "$fd" ] && printf '\r\033[K' >&"$fd"
  printf '%s: %d/%d done in %s\n' \
    "$PROG_LABEL" "$PROG_N" "$PROG_TOTAL" "$(_prog_hms $el)" >&"$fd"
}
