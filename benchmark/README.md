# Benchmarks

## Layout

- `metatensor_bench.py`, `bench/`: RPython microbenchmark binary and its variants
  (`chain.py`, `mlp.py`, `transformer.py`, `cnn.py`, `reduce.py`, shared setup in `common.py`).
- `torch_bench.py`, `jax_bench.py`, `triton_bench.py`, `iree_adapter.py`: baselines for the
  same variants.
- `applevel/`: Python scripts run on a translated `pypy-c` (models, probes, PyTorch twins).
- `suites/`: ports of the PyTorch 2 benchmark suites (TorchBench, HuggingFace, TIMM).
- `motion/`: the leave rule check described below.
- `eval/`: the full evaluation harness, documented in `eval/README.md`.
- `benchmarks.toml`: microbenchmark grid and figure labels.

## Leave rule check (no PyPy build needed)

### What it is

- MOTION handles a change in a running program with a short change rule written against the
  interpreter, instead of a hand-written runtime adapter.
- The example is "leave". Two workers add gradient contributions into a step that is not yet
  committed, and worker B leaves. The application declares, per leave reason, whether B's
  uncommitted contribution is kept (`retain`, e.g. preemption) or removed (`revoke`, e.g. a
  fault).
- `pypy/module/_metatensor/rule_leave.py` is the rule (7 code lines). It drops the member and,
  on `revoke`, excludes that member's uncommitted contributions.
- `pypy/module/_metatensor/motion_iface.py` is the only vocabulary the rule may use: membership,
  the declared action and uncommitted contributions. It names no kernel, fused region, guard
  or resume data. `pypy/module/_metatensor/interp_group.py` implements it and calls the rule.
- `eval/audit_rules.py` checks this mechanically. A rule fails if it imports a module outside
  an allow list, names any part of the target representation or the recovery machinery, or
  uses a tile or block size constant. The audit prints the imports and names it found.
- `motion/rule_leave_leaky.py` is a negative control. It reaches into the aggregate
  representation and the resume data, so the audit must reject it. It is never imported.
- `motion/adapter_leave.py` is the hand-written comparison: a contributor log, deduplication
  and the transition.
- `motion/count_lines.py` counts hand-written code lines on each side, excluding blanks,
  comments, docstrings and the shared notification harness. The MOTION side counts the rule,
  the interface, `interp_group.py` and the policy lines.
- `motion/rule_leave.provenance` records the audit commands and their expected outcome.

### Commands

    python3 benchmark/eval/audit_rules.py pypy/module/_metatensor/rule_leave.py
    python3 benchmark/eval/audit_rules.py benchmark/motion/rule_leave_leaky.py
    python3 benchmark/motion/count_lines.py

- The first prints `all rules pass the audit` and exits with status 0.
- The second prints one `FAIL` line per violation and `AUDIT FAILED`, and exits with status 1.
- The third prints a table per side and ends with `N_rule=246 N_adapter=47`. The rule itself
  is 7 lines. Most of the MOTION total is the 224 lines of `interp_group.py`.

### End-to-end run (needs `pypy-c`)

    LEAVE_PYPY=/path/to/pypy-c bash benchmark/eval/run_leave.sh

- Runs both audits, then the histories `h1 h2 h3 h2post none` five ways: interpreter only
  (`ref`, `--jit off`), MOTION, the adapter, and two mutants (`MOTION_ERASE=1`,
  `MOTION_ORDER=late`).
- `motion/check_leave.py` checks every output against values it computes independently.
  MOTION and the adapter must match `ref` field by field. Both mutants must be caught on `h2`.
- Writes `$OUT/leave/` and `$OUT/leave.tsv` (`OUT` as in `eval/README.md`).
- Options: `LEAVE_STEPS` (200), `LEAVE_N` (64), `LEAVE_HISTORIES`, `LEAVE_PERF=1` for timing
  with `LEAVE_RUNS` (10), `LEAVE_PERF_N` (65536), `LEAVE_SYNC=1` and `LEAVE_PERF_ONLY=1`.
- `bench.sh motion` (`eval/run_motion.sh`, needs `pypy-c`) audits `motion/axis_rule.py` or
  `motion/layout_rule.py` (`MOTION_CASE=axis|layout`, control `motion/layout_rule_leaky.py`)
  and measures the rule against the hand-written guard-recovery path.

## Standalone microbenchmark binary

### Setup

    python3 -m venv .venv
    .venv/bin/pip install triton torch --index-url https://download.pytorch.org/whl/cu130
    export RTENSOR_PYTHON=$PWD/.venv/bin/python   # compiles .ttir to PTX
    benchmark/build.sh                            # -> benchmark/metatensor-bench

- `build.sh` needs `python2` and takes about 10 minutes. `PYTHON2` and `MAKE_JOBS` (default 4)
  override the interpreter and the parallelism.
- `benchmark/eval/bench.sh setup` performs the full setup, including `pypy-c`.

### Run

    benchmark/metatensor-bench MODE VARIANT K N ITERS [WARMUP]
    $RTENSOR_PYTHON benchmark/torch_bench.py TORCH_MODE VARIANT K N ITERS [WARMUP]

- `MODE`: `fused` (MetaTensor), `eager` (tensor optimisation off, one kernel per op), `nojit`.
- `TORCH_MODE`: `eager`, `compile`, `compile-ro`, `compile-mat`, `tensorrt`.
- `K`: chain length. `N`: tensor size or row count. `WARMUP`: 30 by default, for every system.
- Output columns: `mode variant k n iters warm_s steady_us kernels acc compiled_in_timed
  launches_per_iter dtype`.
- `launches_per_iter` must be 1.0 for the chain (variant 0) in `fused` mode.

### Variants

| variant | workload |
|---|---|
| 0 | elementwise chain of K ops |
| 1-5 | chain plus a guard, a materialisation, an `.item()` branch, `try/except`, a host write |
| 6 / 7 | 3-layer MLP forward / training step |
| 8 / 10 | Transformer block forward / 2-block training step |
| 9 | small CNN forward |
| 11 | row reduction, carried |
| 12 | one matmul layer, carried: `x = relu(matmul(x, W) + b)` |
| 13 | multi-head self-attention, carried, D=256, 8 heads |

- Variant 2 is variant 1 plus one real materialisation per iteration, in each system's own
  form: `.force()`, `torch.cuda.synchronize()`, `torch._dynamo.graph_break()`,
  `block_until_ready()`.

### Environment variables

| variable | meaning |
|---|---|
| `RTENSOR_DTYPE` | `float64` (default), `float32`, `float16` |
| `RTENSOR_CPU=1` | run without a GPU |
| `RTENSOR_BUDGET_MB` | device GC byte threshold (default 8) |
| `RTENSOR_FLAT_BLOCK` | elements per block of elementwise and gather kernels (default 4096) |
| `RTENSOR_BLOCK` | row-kernel tile width |
| `RTENSOR_MAX_INPUTS` | leaves one fused kernel may take, 4..16 (default 16), no retranslation |
| `RTENSOR_NO_CUBLASLT` | plain `cublasSgemmStridedBatched` instead of per-shape tuning |
| `RTENSOR_CUBLAS` | path to `libcublas.so` (default: dynamic loader) |
| `RTENSOR_DEBUG_KNOBS=1` | `metatensor-bench` prints the resolved knob values |
| `RTENSOR_PROFILE=1`, `CUDA_HOME` | profiling, CUDA location |

## Application-level scripts

- Scripts in `applevel/` run on a PyPy translated with `--withmod-_metatensor`:

      ./pypy-c -S --jit threshold=3,function_threshold=3,trace_eagerness=2 \
          benchmark/applevel/transformer.py

- Real checkpoints: `applevel/{gpt2,llama,bert,vit,mixer,resnet}{_export,,_torch}.py`. The
  export script writes the weights, the plain script runs them on PyPy, and the `_torch`
  script runs the reference model on the same input and reports `maxabsdiff`.
- The harness in `eval/` drives these scripts.

## Pitfalls

- The default JIT loop threshold is 1039. Pass `threshold=3` as above.
- Setting `RTENSOR_BLOCK` below the widest row routes row kernels to the CPU (about 10x
  slower).
- Tensor ops are elidable, so a chain over loop-invariant inputs is hoisted. Carry the result
  across iterations.
- Device buffers are recycled by GC finalisers. A GC policy change requires re-measuring the
  chain, the CNN and both training sizes.
- Work falls back to the CPU evaluator without an error when `libcublas` is missing (matmul
  only), when `rpython/metatensor/triton_compile.py` is missing, or when `cuMemAlloc` fails.
  A run about 50x slower than expected has one of these causes.
- The host is noisy. Interleave MetaTensor and PyTorch runs when comparing.
