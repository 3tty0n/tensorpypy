# MetaTensor evaluation harness

## Requirements

- NVIDIA GPU and driver. The default CUDA 13 torch and Triton wheels need driver >= 580.
  With an older driver set `TORCH_INDEX=https://download.pytorch.org/whl/cu128` (or `cu121`).
- `gcc`, `make` and `$CUDA_HOME/include/cuda.h` (default `CUDA_HOME=/usr/local/cuda`).
- `python2` or `pypy2` on `PATH` for the RPython toolchain. If neither exists, `setup.sh`
  downloads PyPy2.7 into `benchmark/eval/.toolchain/`.
- `python3` to create the venv.

## Reproduce

    benchmark/eval/bench.sh setup
    benchmark/eval/bench.sh all

- `bench.sh setup` (`setup.sh`) is idempotent. It
  - creates the venv at `${VENV:-$HOME/.venvs/metatensor}`,
  - detects the GPU compute capability and writes it and all derived paths to
    `benchmark/eval/env.sh`,
  - translates `metatensor-bench` and `pypy-c` into `${BUILD:-benchmark/eval/build}` unless
    already built,
  - exports the checkpoints into `${WEIGHTS:-benchmark/eval/weights}` (`SKIP_WEIGHTS=1` skips).
- `bench.sh` sources `config.sh`, which sources `env.sh`. No manual exports are needed.
- Python dependencies are declared in `pyproject.toml` and pinned in `requirements.lock`.
  `setup.sh` installs them with `uv` (fetched into `.toolchain/` if absent) or pip
  (`NO_UV=1`). A torch or Triton version that does not match the driver fails silently, so
  keep the pins. To move a pin:

      uv pip compile benchmark/eval/pyproject.toml -o benchmark/eval/requirements.lock

- Durations: `pypy-c` translation 15 to 60 minutes, `metatensor-bench` about 3 minutes,
  `bench.sh all` about 2 hours.
- `bench.sh all` runs setup, export, micro, models, ablation, dynamic, summarize and plot. It
  clears `$OUT/*.tsv` first. `SKIP_SETUP=1` reuses an existing build.
- Overrides for setup: `VENV`, `BUILD`, `WEIGHTS`, `TORCH_INDEX`, `MAKE_JOBS`.
- Overrides for `bench.sh`: `ITERS` (200), `WARMUP` (30), `ROUNDS` (3), `OUT`, `JIT_FLAGS`,
  `SKIP_EXPORT`, `SKIP_MICRO`, `SKIP_MODELS`, `SKIP_ABLATION`, `SKIP_DYNAMIC`,
  `SKIP_SUMMARIZE`, `SKIP_PLOT`.

## Commands

- `bench.sh help` prints the full list. `bench.sh list` prints model and ablation names.
- Each command appends to `$OUT/*.tsv` and writes a header only for a new file, so several
  runs followed by `bench.sh summarize` form one result set.
- Commands chain on a literal `--`:

      benchmark/eval/bench.sh micro --applevel -- models distilgpt2 -- summarize -- plot

- A failing command does not stop the chain. A bad argument stops it before anything runs.

| command | purpose |
|---|---|
| `setup`, `export` | build and install, export checkpoints |
| `micro [V K N]` | microbenchmarks, full grid or one point |
| `models [NAME...]` | end-to-end models |
| `ablation [EXP...]` | one runtime knob at a time |
| `dynamic` | sequence-length sweep |
| `batch [MODEL...]` | batch-size sweep |
| `deopt [PATTERN...]` | cost of a guard failure |
| `correctness [NAME...]` | each system against MetaTensor on five derived inputs |
| `inventory [NAME...]` | parameters, FLOPs, call and operator counts |
| `fusion [MODEL...]` | fusion-region statistics |
| `gap [MODEL...]` | launches and GPU utilisation per system (needs `nsys`) |
| `lazy`, `control`, `decode`, `suites` | see `bench.sh help` |
| `transition`, `motion`, `warmup`, `explain` | see `bench.sh help` |
| `check [GROUP...]` | smallest run exercising every mode |
| `summarize`, `plot`, `compare DIR...`, `size` | reports and figures |

Examples:

    benchmark/eval/bench.sh micro 8 1 256000
    benchmark/eval/bench.sh models distilgpt2 resnet18-b1
    benchmark/eval/bench.sh ablation fusion
    benchmark/eval/bench.sh batch bert-mini

## Microbenchmarks through pypy-c

    benchmark/eval/bench.sh micro --applevel

- The default driver is `build/metatensor-bench`, a translated RPython program without
  bytecode dispatch. `torch_bench.py` pays for CPython, so this comparison favours MetaTensor.
- `--applevel` runs the same variants through `build/pypy-c`, the path the models use. It
  adds `app` rows only, at float64 only, so run it in addition to a normal micro run.
- `summarize` then fills the `app-level (ours)` column and `interp tax` = app / fused.
- `benchmarks.toml` holds the grid and figure labels. A new point is one entry there, with
  implementations in `benchmark/bench/` and `benchmark/torch_bench.py`.

## Figures

- `bench.sh plot` writes vector PDFs and a `.tex` table per figure into `$OUT/figures`.
- Marks are medians over `$ROUNDS` rounds with whiskers to the minimum and maximum. Speedup
  whiskers span the ratio of opposite range ends, because rounds are not paired.
- Figures: `micro_speedup`, `fusion`, `models`, `dynamic`, `precision`, `ablation`, `gap`.
- Flags: `--only NAME` (repeatable), `--format pdf,png`, `--column single|double`,
  `--texture` (hatching for grayscale), `--titles`, `--iree` (include IREE).

### Cross-machine comparison

    benchmark/eval/bench.sh compare ../results/<host>-<gpu>/grid-<date>

- Compares `$OUT` with one or two other result directories (at most three machines).
- Series are named by accelerator, from `machine.txt`, or else from the `<gpu>` part of the
  directory name. Runs on the same accelerator model are separated by date.
- Figures: `compare_speedup`, `compare_models`, `compare_dynamic`. They compare ratios, not
  microseconds. Points not measured at the same size on both machines are dropped and named
  on stderr.

## Baselines

- PyTorch eager and `torch.compile` (mode `compile`), plus:
  - JAX/XLA (`jax.jit`): `benchmark/jax_bench.py`, `benchmark/applevel/jax_models.py`,
    system `jax`. All micro variants and the end-to-end models.
  - IREE (StableHLO from the same JAX code, CUDA HAL): same scripts, mode `iree`. All micro
    variants except 9 (CNN). All models except resnet18 and smollm2-135m.
  - Handwritten Triton: `benchmark/triton_bench.py`, mode `triton`. Micro variants 0-6 and
    11-13.
  - Torch-TensorRT 2.9 (`torch.compile(backend="tensorrt")`): `*_torch.py` and
    `torch_bench.py`, mode `tensorrt`, system `torch-tensorrt`.
  - `torch.compile` with `mode="reduce-overhead"` (CUDA graphs): mode `compile-ro`.
  - `torch.compile` with `mode="max-autotune"`: mode `compile-mat`. Compilation can take
    minutes per point.

### Installing

- JAX and IREE use a second venv, `${JAX_VENV:-$VENV-jax}`, from `requirements-jax.lock`, so
  their CUDA wheels do not move the torch pins. `setup.sh` creates it (`WITH_JAX=0` skips)
  and writes `JAX_PYTHON` into `env.sh`. Without it the jax and iree rows are absent. Manual
  setup:

      uv venv --python 3.14 ~/.venvs/metatensor-jax
      VIRTUAL_ENV=~/.venvs/metatensor-jax uv pip sync benchmark/eval/requirements-jax.lock
      export JAX_PYTHON=~/.venvs/metatensor-jax/bin/python

- Torch-TensorRT needs CPython <= 3.13 and torch 2.9, so it uses a third venv,
  `${TRT_VENV:-$VENV-trt}`, from `requirements-trt.lock` (`WITH_TRT=0` skips, `TRT_PYTHON`
  in `env.sh`). Its torch differs from the eager and compile columns. The model code is the
  same.
- Triton needs nothing beyond the torch venv.

### Running

- `bench.sh micro` and `bench.sh models` add the baseline rows automatically.
- `BASELINES` selects a subset of `triton tensorrt compile-ro compile-mat jax iree` (default
  all). With `--baselines-only` only those rows are added to an existing result set:

      BASELINES=iree benchmark/eval/bench.sh micro --baselines-only
      BASELINES="compile-ro compile-mat" benchmark/eval/bench.sh models --baselines-only

- `bench.sh check baselines` runs one point per baseline and compares its accumulator with
  torch eager.
- `summarize.md` gains a column per baseline, an `ours/Triton` ratio and a compilation
  overhead table. `plot` writes `micro_baselines.*` and `compile_overhead.*`.
- IREE is left out of figures by default because it is far slower. `plot --iree` writes
  `models_iree.*`, `micro_baselines_iree.*` and `compile_overhead_iree.*`.
- `compile-ro` in the micro variants needs `cudagraph_mark_step_begin()` and an output
  `clone()` inside the timed loop, because each iteration feeds the next. That cost is
  measured. In the models the single `logits.clone()` runs after the timed loop.

### Methodology

- All systems share shapes, dtypes, seeds, iteration counts, rounds and GPU, and synchronise
  before and after the timed loop.
- Compilation is excluded from `steady_us` and reported as `compile_ms` and `first_run_ms` in
  `results.jsonl`, the last two micro fields and the model line.
  - JAX: `compile_ms` is `jit(f).lower(...).compile()`. Variants 1-5 keep branches in Python.
    `jax_default_matmul_precision` is `highest` (no TF32). Weights are jit arguments.
  - IREE: exported with `jax.export`, compiled with `--iree-cuda-target=sm_<cc>` and
    `--iree-input-demote-f64-to-f32=false`. Runs through the Python runtime API without
    cuBLAS, untuned.
  - Triton: `compile_ms` is the first launch of each kernel. Fixed tiles (64x64x32 GEMM,
    1024-element blocks, 32x32 fp64 attention tiles), no autotuning.
  - Torch-TensorRT: `enabled_precisions={dtype}`, `disable_tf32=True`. HF models load with
    `attn_implementation="eager"` in this mode only, because 2.9 lowers masked
    `scaled_dot_product_attention` incorrectly. `compile_ms` is -1.
  - `torch.compile`: `compile_ms` is -1, `first_run_ms` is compile plus one iteration.
  - MetaTensor: `first_run_ms` is the first forward, including tracing and kernel compilation.
- Break-even in iterations against eager:
  `(first_run(system) - first_run(eager)) / (steady(eager) - steady(system))`.
- Every driver takes `WARMUP` (default 30) and runs that many iterations before timing.
  `run_micro.sh` passes the same value to all drivers.
- Model rows carry `launches_per_iter`. MetaTensor reads `_metatensor.launch_count()` over
  the timed loop. PyTorch counts kernels with `torch.profiler` over five iterations after the
  timed loop. JAX leaves it empty.

### Correctness tolerance

- Micro lines carry an accumulator compared with torch (`close_enough`, relative 1e-6).
- Model lines carry `maxabsdiff` against the stored MetaTensor logits, `tol=<t>` and
  `pass=<0|1>`, recorded in `models.tsv`, `results.jsonl` and `summary.md`.
- Tolerances come from `tolerance_for` in `config.sh`, as a fraction of the reference
  output's largest absolute value:

| workload class | dtype | relative tolerance |
|---|---|---|
| models | float32 | 2e-5 |
| conv models under TF32 (`resnet18-*`) | float32 | 2e-3 |
| any | float16 | 1e-3 |
| any | float64 | 1e-8 |

### Comparability limits

- The Triton GEMM is a Triton kernel. The MetaTensor GEMM is cuBLAS with a fused epilogue.
- Triton and `torch.compile` reuse on-disk kernel caches. The JAX persistent cache is off.
- IREE rejects the conv2d lowering from `jax.export`, so variant 9 and resnet18 are absent.
  smollm2-135m is absent because plain `jax` already exceeds the tolerance on it.
- `kernel_count` is reported only for MetaTensor and Triton.

## Launch gap analysis

    benchmark/eval/bench.sh gap bert-mini

- Steady-state wall time from a plain run, plus launch counts and GPU busy time per forward
  from two `nsys` traces at different iteration counts, differenced to cancel setup and
  compilation. `--cuda-graph-trace=node` is required.
- `gap.tsv` columns `gemm`, `splitk`, `gen`, `copy`, `other` sum to `launches_per_iter`. A
  row that does not sum says `CLASS-SUM-MISMATCH`.
- `gap_kernels.tsv` has one row per distinct kernel per (model, system) with launches and GPU
  microseconds per forward. The `bmm` column of `gap.tsv` is a lower bound.
- `launches` counts every kernel `nsys` sees, including cuBLAS, so it exceeds
  `_metatensor.launch_count()`. `util` is GPU busy time over wall time.
- The gap harness runs systems back to back. Quote `models.tsv` for latency.
- `RTENSOR_NO_CUBLASLT=1` disables the per-shape cuBLASLt algorithm selection for batched
  products (`rt_lt_bmm` in `rpython/metatensor/cuda.c`).

## Correctness on several inputs

    benchmark/eval/bench.sh correctness
    benchmark/eval/bench.sh correctness bert-mini
    SEEDS="1 2" benchmark/eval/bench.sh correctness

- Runs the model check on derived inputs `INPUT_SEED=1..5` and writes
  `$OUT/correctness.tsv` (`model system seed maxabsdiff argmax_match tol pass`).
- Inputs are derived by integer arithmetic in `benchmark/applevel/inputs.py`, so all sides
  get bit-identical inputs. `INPUT_SEED=0` is the stored input and the default.
  - Text: `ids[i] = (tokens[(i + k) % seq] + k) % vocab`.
  - Vision: `pixel[i] += 0.05 * (2*u(i, k) - 1)`, `u` an integer hash of index and `k`.
- MetaTensor runs first and writes `logits_pypy_seed<k>.bin`. The others are compared with it
  using `tolerance_for`. `ITERS` is 5.
- `argmax_match` is per-position argmax agreement for text and top-1 agreement for vision.
- `summarize.py` and `plot.py --only correctness` report the worst seed, not the mean.

## Model and operator inventory

    benchmark/eval/bench.sh inventory [NAME...]

- Writes `$OUT/model_inventory.tsv`, `$OUT/op_inventory.tsv` and
  `$OUT/op_inventory_notes.txt`. `plot.py --only model_inventory --only op_inventory` renders
  `figures/model_inventory.tex` and `figures/op_inventory.tex`.
- MetaTensor side: analytic from `index.json` and `lib_pypy/tensorpypy/models.py`, no GPU.
  `--no-torch` writes only these rows. `--selfcheck` checks that the two walkers agree.
- PyTorch side: one eager forward with `FlopCounterMode`, `torch.profiler` op counts and
  `p.numel()`.
- `params_only` counts learned weights. `params_plus_buffers` adds non-learned trained
  tensors (batch-norm running statistics). Inputs and tables rebuilt from the config are
  excluded on both sides.
- Operators are mapped to one vocabulary: `matmul`, `bmm`, `conv`, `layernorm`, `rmsnorm`,
  `softmax`, `gelu`, `silu`, `relu`, `add`, `mul`, `residual_add`, `embedding_gather`,
  `rotary`, `reshape_transpose`, `reduce`, `other`. The mapping is printed in the notes.
  - Only one dispatch level of each `aten::` op is counted.
  - One SDPA call counts as at least 2 `bmm`, 1 `softmax` and 1 `mul`.
  - PyTorch residual adds are counted under `add`.
- Every differing cell is explained in the notes, or marked `NOT EXPLAINED`.
- Generated-IR histograms come from the `fusion_stats.py` dump (`--ttir`, default
  `$TMPDIR/fusion-stats`) and are absent unless `bench.sh fusion` has run.

## Dynamic sequence length

    benchmark/eval/bench.sh dynamic

- Visits lengths `[32, 48, 64, 96, 128]` twice in fixed order. Pass 1 is the first visit,
  pass 2 the revisit.
- `step_us` (plotted) is the forward plus `logits.sum()` to a host scalar. `build_us` is input
  construction, reported separately. `total_us` is their sum.
- Compile counters are per-length deltas:
  - MetaTensor: `pypyjit.get_stats_snapshot()` (loops, bridges, `compile_ms`) and
    `_metatensor.kernel_count()` / `launch_count()`. `cache_hits` = launches minus new
    kernels.
  - PyTorch: `torch._dynamo.utils.counters['stats']['unique_graphs']` as `graphs` and
    `recompiles`. `compile_ms` is -1.
- A recompile appears in the counters, not in the median.

## Batch size

    benchmark/eval/bench.sh batch
    benchmark/eval/bench.sh batch bert-mini
    BATCHES="1 8 64" benchmark/eval/bench.sh batch

- Defaults: `BATCHES="1 2 4 8 16 32"`, `BATCH_MODELS="bert-base distilgpt2"`,
  `BATCH_SYSTEMS="ours torch-eager torch-compile torch-compile-ro jax"`. Output
  `$OUT/batch.tsv`.
- A batch is B copies of the stored sequence. `batch_rows_identical` checks that every block
  equals the first. Block 0 is also compared with `logits_pypy.bin`.
- MetaTensor folds the batch into rows (`[B*seq, d]`). PyTorch uses `[B, seq]`. JAX uses
  `jax.vmap`.
- `steady_us` is per forward. `per_seq_us` = `steady_us / B` is plotted. A batch that does not
  fit is recorded as `status=failed`.

## Guard failure cost

    benchmark/eval/bench.sh deopt
    benchmark/eval/bench.sh deopt alternate

- `benchmark/applevel/deopt_probe.py` runs micro variant 3 under the schedules in `PATTERNS`:
  - `never`: the guard never fails.
  - `alternate`: stable during warm-up, then flipping every iteration.
  - `both-hot`: flipping during warm-up too, so both paths are compiled before timing.
  - `fresh`: a new value every iteration, so the guard always fails.
- Columns: `steady_us` (median of the last quarter), `first_fail_us`, `after_fail_us` (median
  of the next five), `peak_us` (most expensive iteration, where the bridge is compiled).
- The same driver runs PyTorch (`--mode eager`, `compile`, `compile-ro`).
- Also runs `recovery_probe.py --times` and records rows `probe-a` and `probe-e`.
- These are single iterations and sensitive to other GPU users. A `peak_us` near one second
  indicates contention.

## Setup check

    benchmark/eval/bench.sh check
    benchmark/eval/bench.sh check micro dtypes

- Groups: `prereq micro dtypes torch baselines models ablation dynamic report`. A full check
  takes about 100 seconds and exits non-zero on the first error. Run it before a grid and
  after toolchain changes.
- It checks `launches_per_iter` (a kernel that fails to load falls back to the CPU without an
  error) and compares accumulators across fused, eager, nojit and torch.

## Fusion statistics

    benchmark/eval/bench.sh fusion [MODEL...]

- `fusion_stats.py` writes `$OUT/fusion.tsv`:
  - `kernels`: distinct fused kernels compiled for the forward.
  - `launches_per_iter`: launches per steady-state forward.
  - `nodes_min/median/max`: DAG nodes per kernel, from the TTIR.
  - `extra_outputs_total`: values leaving a region as an extra output.
  - `forced_*`: why each region ended, from the optimised trace.
- Runs with `PYPYLOG=jit-log-opt`, `RTENSOR_KERNEL_CACHE=0`, `RTENSOR_STATS=1` and a fresh
  `TMPDIR`. Logs and IR go to `/tmp/fusion-stats` (`--tmp` moves them).
- Force reasons, by the first consumer of a launch result:
  - `forced_library`: `matmul`, `bmm`, conv or `im2col`, head split or merge, gathers.
  - `forced_item`: a host read such as `.item()`.
  - `forced_loop`: stored to the heap, passed to the next iteration, or live at trace end.
  - `forced_leafcap`: chain cut at `RTENSOR_MAX_INPUTS` leaves.
  - `forced_assign`: `tensor.assign`, which forces every live chain.
  - `forced_other`: anything else, with consumer names printed on stderr.
- Counts are static (one per launch site per trace). Read them as proportions.
- `bench.sh plot --only fusion_stats` renders `figures/fusion_stats.tex` and a stacked bar.

## Implementation size

- `bench.sh size` counts non-blank, non-comment lines per component and writes
  `$OUT/figures/impl_size.tex`.

## Result files

- Default location: `benchmark/results/<host>-<gpu>/grid-<date>/`. `RUN_HOST` and `RUN_GPU`
  override the components, `OUT` the whole path. Result directories are not part of this
  repository.
- `$OUT/machine.txt` records driver, toolkit, wheel versions, CPU and knobs.
- Each stage appends to a positional `.tsv`. Every measurement is also written to
  `$OUT/results.jsonl` with named fields. Use the jsonl from new code.
- Row data can be packed into one reproducible `data.tar.gz` per run. The readers extract it
  on first use.

      benchmark/eval/archive.sh pack   RESULT_DIR
      benchmark/eval/archive.sh unpack RESULT_DIR
      benchmark/eval/archive.sh check  RESULT_DIR

- Runtime knobs are listed in `benchmark/README.md`.
- When comparing with another run, diff `$OUT/summary.md`. Absolute times differ by GPU, but the
  ordering of `fused`, `torch.compile` and `eager` should hold.

## Docker

    docker build -t metatensor-eval -f benchmark/eval/Dockerfile .
    docker run --gpus all metatensor-eval

- Base image `nvidia/cuda:13.0.1-devel-ubuntu24.04` with PyPy2.7. `setup.sh` runs at build
  time with `SKIP_WEIGHTS=1`. The entry point is `run_all.sh` (`bench.sh all`).
- Not tested to completion on GPU-backed Docker.

## Troubleshooting

- `cuMemAlloc ... failed`, a kernel falling back to the CPU, or no cuBLAS: the run continues
  on the CPU path and the numbers are wrong. Check `nvidia-smi`, `$CUDA_HOME`, and that
  `RTENSOR_CUBLAS` in `env.sh` points to a `libcublas.so.*` in the venv
  (`nvidia/cu*/lib/`).
- `Failed to initialize NVML: Driver/library version mismatch`: affects `nvidia-smi` only.
  A reboot or a matching NVML package clears it.
- Blackwell (sm_120): needs Triton >= 3.3. `setup.sh` sets `RTENSOR_CC=120`. If `RTENSOR_CC`
  is unset, `triton_compile.py` queries torch on each compile. Set it only to pin a target.
