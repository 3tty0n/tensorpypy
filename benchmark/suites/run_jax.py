#!/usr/bin/env python3
"""Time one suite model under JAX/XLA via torchax, with weights and inputs from EXPORT_DIR.

    run_jax.py EXPORT_DIR [--repeat 30] [--warmup 10] [--dump DIR]
Prints the run_torch.py line with backend=jax-xla. Env: SUITES_JAX_PYTHON (benchmark/eval/run_suites.sh).
"""
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "eval"))
import suite_census  # noqa: E402


def read(blob, entry):
    import numpy as np
    import torch
    off, shape, dtype = entry
    blob.seek(off)
    if dtype == "bfloat16":
        a = np.fromfile(blob, np.int16, count=int(np.prod(shape)))
        return torch.from_numpy(a).view(torch.bfloat16).reshape(shape)
    a = np.fromfile(blob, np.dtype(dtype), count=int(np.prod(shape)))
    return torch.from_numpy(a).reshape(shape)


def dump(outputs, path):
    import numpy as np
    os.makedirs(path, exist_ok=True)
    index = []
    with open(os.path.join(path, "data.bin"), "wb") as blob:
        for o in outputs:
            a = np.asarray(o, dtype=np.float32)
            index.append([blob.tell(), list(a.shape), "float32"])
            a.tofile(blob)
    json.dump({"outputs": index}, open(os.path.join(path, "index.json"), "w"))


def main(argv):
    d = os.path.abspath(argv[1])
    repeat = int(argv[argv.index("--repeat") + 1]) if "--repeat" in argv \
        else 30
    warmup = int(argv[argv.index("--warmup") + 1]) if "--warmup" in argv \
        else 10
    dump_dir = os.path.abspath(argv[argv.index("--dump") + 1]) \
        if "--dump" in argv else None
    idx = json.load(open(os.path.join(d, "index.json")))
    suite, name = idx["suite"], idx["model"]

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_default_matmul_precision", "tensorfloat32")
    import torch
    import torchax
    from torch.utils._pytree import tree_flatten, tree_leaves, tree_unflatten

    runner, args = suite_census.make_runner(suite, name)
    _, _, model, inputs, batch = runner.load_model("cpu", name,
                                                   batch_size=idx["batch"])
    model, inputs = runner.cast_based_on_args(model, inputs)
    model.eval()

    blob = open(os.path.join(d, "data.bin"), "rb")
    sd = model.state_dict()
    if set(sd) != set(idx["params"]):
        raise SystemExit("run_jax: state_dict keys differ from the export's")
    for k, e in idx["params"].items():
        sd[k] = read(blob, e)
    model.load_state_dict(sd)
    if isinstance(inputs, dict):
        pos, kw = [], dict(inputs)
        for k, e in idx["kwargs"].items():
            kw[k] = read(blob, e)
    else:
        pos, kw = list(inputs), {}
        slots = [i for i, v in enumerate(pos) if isinstance(v, torch.Tensor)]
        if len(slots) != len(idx["inputs"]):
            raise SystemExit("run_jax: %d tensor inputs, export has %d"
                             % (len(slots), len(idx["inputs"])))
        for i, e in zip(slots, idx["inputs"]):
            pos[i] = read(blob, e)

    leaves, spec = tree_flatten((pos, kw))
    tpos = [i for i, v in enumerate(leaves) if isinstance(v, torch.Tensor)]
    env = torchax.default_env()
    states, jf = torchax.extract_jax(model)
    xs = env.t2j_copy([leaves[i] for i in tpos])

    def forward(states, xs):
        ls = list(leaves)
        for i, x in zip(tpos, xs):
            ls[i] = x
        a, k = tree_unflatten(ls, spec)
        out = jf(states, a, k)
        return [o for o in tree_leaves(out) if isinstance(o, jax.Array)]

    fn = jax.jit(forward)
    t0 = time.perf_counter()
    out = jax.block_until_ready(fn(states, xs))
    first_ms = (time.perf_counter() - t0) * 1e3
    for _ in range(warmup):
        out = jax.block_until_ready(fn(states, xs))
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = jax.block_until_ready(fn(states, xs))
        times.append((time.perf_counter() - t0) * 1e3)
    if dump_dir:
        dump([o for o in out if jnp.issubdtype(o.dtype, jnp.floating)],
             dump_dir)
    print("torch suite=%s model=%s batch=%s backend=jax-xla median_ms=%.3f "
          "min_ms=%.3f first_ms=%.1f jax=%s torchax=%s"
          % (suite, name, idx["batch"], statistics.median(times), min(times),
             first_ms, jax.__version__, torchax.__version__), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
