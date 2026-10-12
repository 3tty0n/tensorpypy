#!/bin/bash
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/config.sh"
SUITES="$HERE/../suites"

SP=${SUITES_PYTHON:-$HOME/.venvs/torchbench/bin/python}
P=${SUITES_PYPY:-$PYPY}
EXPORTS=${SUITES_EXPORTS:-$HOME/.cache/metatensor-suites}
JIT=${SUITES_JIT:-"--jit threshold=3,function_threshold=3,trace_eagerness=2,trace_limit=200000"}
NCU=${NCU:-/usr/local/cuda/bin/ncu}
WORK=${PROF_WORK:-$HOME/.cache/metatensor-prof}
NCU_REPEAT=${NCU_REPEAT:-30}
MODELS=${NCU_MODELS:-"torchbench/resnet50 torchbench/mobilenet_v2 torchbench/hf_Bert torchbench/timm_vision_transformer torchbench/pyhpc_equation_of_state huggingface/BertForMaskedLM timm/convnextv2_nano.fcmae_ft_in22k_in1k"}
METRICS="gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum"
NCU_ARGS="--profile-from-start off --metrics $METRICS --csv --page raw"
mkdir -p "$WORK"

TSV="$OUT/ncu.tsv"
tsv_init "$TSV" "suite\tmodel\tsystem\tclass\tkernels\ttime_ms\tread_mb\twrite_mb"
TFL="$OUT/flops.tsv"
tsv_init "$TFL" "suite\tmodel\tbatch\ttotal\tmatmul\tconv\tattention\tother"

progress_init ncu $(( $(echo $MODELS | wc -w) * 3 ))
for sm in $MODELS; do
  suite=${sm%%/*} model=${sm#*/}
  name=${suite}_${model//\//_}
  X="$EXPORTS/$name"
  progress_step "$sm flops"
  fl=$("$SP" "$HERE/suite_flops.py" "$suite" "$model" 2>/dev/null | grep '^flops ' | tail -1) || true
  [ -n "$fl" ] && echo -e "$suite\t$model\t$(field_of "$fl" batch)\t$(field_of "$fl" total)\t$(field_of "$fl" matmul)\t$(field_of "$fl" conv)\t$(field_of "$fl" attention)\t$(field_of "$fl" other)" >> "$TFL"
  for system in ours inductor; do
    progress_step "$sm $system"
    csv="$WORK/$name.$system.ncu.csv"
    if [ "$system" = ours ]; then
      kdir="$WORK/$name.ncukernels"
      rm -rf "$kdir"; mkdir -p "$kdir"
      cp ${TMPDIR:-/tmp}/rtensor-tune-* "$kdir"/ 2>/dev/null || true
      TMPDIR=$kdir "$NCU" $NCU_ARGS --range-filter yes:$NCU_REPEAT: --log-file "$csv" "$P" $JIT "$SUITES/run_port.py" "$X" --repeat $NCU_REPEAT --profile-each >/dev/null 2>"$csv.err" || true
    else
      kdir=-
      "$NCU" $NCU_ARGS --log-file "$csv" "$SP" "$SUITES/run_torch.py" "$suite" "$model" --backends inductor --repeat 1 --profile >/dev/null 2>"$csv.err" || true
    fi
    python3 "$HERE/ncu_summary.py" "$csv" "$kdir" | while read -r cls n t r w; do
      echo -e "$suite\t$model\t$system\t$cls\t$n\t$t\t$r\t$w" >> "$TSV"
    done
  done
done
progress_done
echo "wrote $TSV"
