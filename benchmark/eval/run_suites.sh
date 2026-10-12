#!/bin/bash
# The PyTorch 2 benchmark suites (TorchBench, HuggingFace, TIMM), as the
# inductor dashboard runs them: its runners, its default inference batch
# sizes, float32, TF32 matmuls, its timed() and its accuracy check.
#
#   ours          benchmark/suites/run_port.py: the model's upstream source
#                 through benchmark/suites/port.py, on the compat torch
#   torch-eager   benchmark/suites/run_torch.py through common.timed()
#   inductor      torch.compile, as the dashboard's --inductor run
#   inductor-cg   the same with inductor's CUDA graphs
#
# Each model is exported once (benchmark/eval/suite_export.py: state_dict,
# inputs, config) into SUITES_EXPORTS. The first run of ours dumps its outputs
# and suite_check.py judges them the way check_accuracy() judges a compiled
# model, with the verdict from torch._dynamo.utils.same(). A run in which any
# op fell back to the CPU is recorded as failed, not timed.
#
#   SUITES_MODELS    "suite/model ..." (default: every model with a port)
#   SUITES_RUNS      processes per system and model (3)
#   SUITES_REPEAT    timed forwards per process, the dashboard's --repeat (30)
#   SUITES_EXPORTS   where exports live (~/.cache/metatensor-suites)
#   SUITES_PYTHON    the suite venv's python (~/.venvs/torchbench/bin/python)
#   SUITES_PYPY      interpreter, default $PYPY
#   SUITES_JIT       JIT flags for ours (the harness's, with a trace limit a
#                    whole large forward fits in)
#   SUITES_MEMMAX    memory cap of each process (40G; empty for none), in its
#                    own systemd scope so an over-limit process is killed alone
#                    and recorded as failed, not by a host-wide OOM kill
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/config.sh"
SUITES="$HERE/../suites"

SP=${SUITES_PYTHON:-$HOME/.venvs/torchbench/bin/python}
P=${SUITES_PYPY:-$PYPY}
EXPORTS=${SUITES_EXPORTS:-$HOME/.cache/metatensor-suites}
RUNS=${SUITES_RUNS:-3}
SYSTEMS=" ${SUITES_SYSTEMS:-ours torch jax-xla} "
JP=${SUITES_JAX_PYTHON:-}
if [[ $SYSTEMS == *" jax-xla "* ]] && [ -z "$JP" ]; then
  echo "run_suites.sh: SUITES_JAX_PYTHON unset, jax-xla skipped" >&2
  SYSTEMS=${SYSTEMS/ jax-xla / }
fi
REPEAT=${SUITES_REPEAT:-30}
JIT=${SUITES_JIT:-"--jit threshold=3,function_threshold=3,trace_eagerness=2,trace_limit=200000"}
[ -x "$P" ] || { echo "run_suites.sh: no interpreter at $P" >&2; exit 1; }
MEMMAX=${SUITES_MEMMAX-40G}
capped() {
  if [ -n "$MEMMAX" ] && command -v systemd-run >/dev/null; then
    systemd-run --user --scope --quiet --collect -p MemoryMax="$MEMMAX" \
      -p MemorySwapMax=0 -- "$@"
  else
    "$@"
  fi
}
[ -x "$SP" ] || { echo "run_suites.sh: no suite venv at $SP (setup_suites.sh)" >&2; exit 1; }
MODELS=${SUITES_MODELS:-$("$P" -c "
import sys; sys.argv = ['x']
sys.path.insert(0, '$SUITES')
import run_port
print(' '.join('%s/%s' % k for k in sorted(run_port.MODELS)))" 2>/dev/null)}

TSV="$OUT/suites.tsv"
tsv_init "$TSV" "suite\tmodel\tsystem\trun\tbatch\tmedian_ms\tmin_ms\tfirst_ms\tlaunches\tpass\trmse\ttorch_rmse\tcpu_fallback\tbinary"
PYPY_SHA=$(binary_sha "$P")
TORCH_VER=$("$SP" -c "import torch; print(torch.__version__)" 2>/dev/null | tail -1)
[ -z "$JP" ] || JAX_VER=$("$JP" -c "import jax, torchax; print('jax-%s+torchax-%s' % (jax.__version__, torchax.__version__))" 2>/dev/null | tail -1)
STATUS=0
mkdir -p "$EXPORTS"

progress_init suites $(( $(echo $MODELS | wc -w) * RUNS ))
for sm in $MODELS; do
  suite=${sm%%/*} model=${sm#*/}
  X="$EXPORTS/${suite}_${model//\//_}"
  if [ ! -f "$X/index.json" ]; then
    "$SP" "$HERE/suite_export.py" "$suite" "$model" "$X" > "$X.log" 2>&1 || {
        echo "run_suites.sh: export of $sm failed: $(tail -1 "$X.log")" >&2
      STATUS=1; continue; }
  fi
  pass= rmse= trmse=
  for run in $(seq "$RUNS"); do
    progress_step "$sm run $run/$RUNS"
    if [[ $SYSTEMS == *" ours "* ]]; then
      dumparg=""
      [ "$run" = 1 ] && rm -rf "$X.dump" && dumparg="--dump $X.dump"
      line=$(capped "$P" $JIT "$SUITES/run_port.py" "$X" --repeat "$REPEAT" \
             $dumparg 2>"$X.err" | grep '^port ' | tail -1) || true
      if [ "$run" = 1 ] && [ -n "$line" ]; then
        check=$(capped "$SP" "$HERE/suite_check.py" "$suite" "$model" "$X.dump" \
                --export "$X" 2>"$X.check.err" | grep '^check ' | tail -1) || true
        pass=$(field_of "$check" pass) rmse=$(field_of "$check" rmse)
        trmse=$(field_of "$check" torch_rmse)
        [ -n "$pass" ] || { pass=0; echo "run_suites.sh: $sm check: $(tail -1 "$X.check.err")" >&2; }
        rm -rf "$X.dump"
      fi
      if [ -z "$line" ]; then
        echo "run_suites.sh: $sm ours run $run: $(grep -m1 -E 'Error|error' "$X.err" || tail -1 "$X.err")" >&2
        echo -e "$suite\t$model\tours\t$run\t\t\t\t\t\t0\t\t\t\t$PYPY_SHA" >> "$TSV"
        STATUS=1
      else
        echo -e "$suite\t$model\tours\t$run\t$(field_of "$line" batch)\t$(field_of "$line" median_ms)\t$(field_of "$line" min_ms)\t$(field_of "$line" first_ms)\t$(field_of "$line" launches)\t$pass\t$rmse\t$trmse\t$(field_of "$line" cpu_fallback)\t$PYPY_SHA" >> "$TSV"
      fi
    fi
    if [[ $SYSTEMS == *" torch "* ]]; then
      capped "$SP" "$SUITES/run_torch.py" "$suite" "$model" --repeat "$REPEAT" \
          2>/dev/null | grep '^torch ' | while read -r tline; do
        echo -e "$suite\t$model\ttorch-$(echo "$tline" | grep -o 'backend=[a-z-]*' | cut -d= -f2)\t$run\t$(field_of "$tline" batch)\t$(field_of "$tline" median_ms)\t$(field_of "$tline" min_ms)\t$(field_of "$tline" first_ms)\t\t\t\t\t\t$TORCH_VER" >> "$TSV"
      done
    fi
    if [[ $SYSTEMS == *" jax-xla "* ]]; then
      dumparg=""
      [ "$run" = 1 ] && rm -rf "$X.jaxdump" && dumparg="--dump $X.jaxdump" && jpass= jrmse= jtrmse=
      line=$(capped "$JP" "$SUITES/run_jax.py" "$X" --repeat "$REPEAT" \
             $dumparg 2>"$X.jax.err" | grep '^torch ' | tail -1) || true
      if [ "$run" = 1 ] && [ -n "$line" ]; then
        check=$(capped "$SP" "$HERE/suite_check.py" "$suite" "$model" "$X.jaxdump" \
                --export "$X" 2>"$X.jaxcheck.err" | grep '^check ' | tail -1) || true
        jpass=$(field_of "$check" pass) jrmse=$(field_of "$check" rmse)
        jtrmse=$(field_of "$check" torch_rmse)
        [ -n "$jpass" ] || { jpass=0; echo "run_suites.sh: $sm jax-xla check: $(tail -1 "$X.jaxcheck.err")" >&2; }
        rm -rf "$X.jaxdump"
      fi
      if [ -z "$line" ]; then
        echo "run_suites.sh: $sm jax-xla run $run: $(grep -E 'Error|error' "$X.jax.err" | tail -1 | grep . || tail -1 "$X.jax.err")" >&2
        echo -e "$suite\t$model\tjax-xla\t$run\t\t\t\t\t\t0\t\t\t\t$JAX_VER" >> "$TSV"
        STATUS=1
      else
        echo -e "$suite\t$model\tjax-xla\t$run\t$(field_of "$line" batch)\t$(field_of "$line" median_ms)\t$(field_of "$line" min_ms)\t$(field_of "$line" first_ms)\t\t$jpass\t$jrmse\t$jtrmse\t\t$JAX_VER" >> "$TSV"
      fi
    fi
  done
done
progress_done
echo "wrote $TSV"
exit $STATUS
