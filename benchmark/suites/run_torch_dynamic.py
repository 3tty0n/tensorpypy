#!/usr/bin/env python3
import copy
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "eval"))
import suite_census  # noqa: E402


def main(argv):
    import torch
    from torch._dynamo.utils import counters
    from torch.utils._pytree import tree_flatten, tree_map
    suite, name, arm = argv[1], argv[2], argv[3]
    spec = argv[argv.index("--sizes") + 1]
    repeat = int(argv[argv.index("--repeat") + 1]) if "--repeat" in argv else 30
    warmup = int(argv[argv.index("--warmup") + 1]) if "--warmup" in argv else 10
    dump_dir = argv[argv.index("--dump") + 1] if "--dump" in argv else None
    torch._dynamo.config.cache_size_limit = 64
    torch._dynamo.config.accumulated_cache_size_limit = 512
    runner, args = suite_census.make_runner(suite, name)
    import common
    common.synchronize = torch.cuda.synchronize
    _, _, model, inputs, batch = runner.load_model("cuda", name, batch_size=None)
    model, inputs = runner.cast_based_on_args(model, inputs)
    fn = runner.forward_pass
    if arm != "eager":
        dyn = {"compile-static": False, "compile-default": None,
               "compile-dynamic": True}[arm]
        fn = torch.compile(fn, backend="inductor", dynamic=dyn)

    axis_name, phases = spec.split(":", 1)
    axis = {"batch": 0, "seq": 1}[axis_name]
    first = [t for t in tree_flatten(inputs)[0] if isinstance(t, torch.Tensor)][0]
    s0 = first.shape[axis]

    def cut(size):
        return tree_map(lambda t: t.narrow(axis, 0, size).contiguous()
                        if isinstance(t, torch.Tensor) and t.dim() > axis and
                        t.shape[axis] == s0 else t, inputs)

    def save(d, out):
        os.makedirs(d, exist_ok=True)
        outs = [t for t in tree_flatten(out)[0]
                if isinstance(t, torch.Tensor) and t.is_floating_point()]
        index, off = [], 0
        with open(os.path.join(d, "data.bin"), "wb") as f:
            for t in outs:
                a = t.detach().float().contiguous().cpu().numpy()
                index.append([off, list(a.shape), "float32"])
                f.write(a.tobytes())
                off += a.nbytes
        json.dump({"outputs": index}, open(os.path.join(d, "index.json"), "w"))

    def golden(x):
        try:
            m64, x64 = common.cast_to_fp64(copy.deepcopy(model), x)
            return runner.forward_pass(m64, x64)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            on = lambda t: t.cpu() if isinstance(t, torch.Tensor) else t
            m64, x64 = common.cast_to_fp64(copy.deepcopy(model).cpu(), tree_map(on, x))
            return runner.forward_pass(m64, x64)

    def call(x):
        out = fn(model, x)
        torch.cuda.synchronize()
        return out

    with torch.no_grad():
        for part in phases.split(";"):
            phase, sizes = part.split("=")
            for size in [int(v) for v in sizes.split(",")]:
                x = cut(size)
                g0 = counters["stats"]["unique_graphs"]
                t0 = time.perf_counter()
                out = call(x)
                first_ms = (time.perf_counter() - t0) * 1e3
                for _ in range(warmup):
                    call(x)
                times = []
                for _ in range(repeat):
                    t0 = time.perf_counter()
                    out = call(x)
                    times.append((time.perf_counter() - t0) * 1e3)
                if dump_dir:
                    d = os.path.join(dump_dir, "%s-%d" % (phase, size))
                    save(d, out)
                    if not os.path.isdir(os.path.join(dump_dir, "fp64-%d" % size)):
                        save(os.path.join(dump_dir, "fp64-%d" % size), golden(x))
                print("port_size suite=%s model=%s arm=%s phase=%s size=%d "
                      "median_ms=%.3f min_ms=%.3f first_ms=%.1f graphs=%d"
                      % (suite, name, arm, phase, size, statistics.median(times),
                         min(times), first_ms,
                         counters["stats"]["unique_graphs"] - g0), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
