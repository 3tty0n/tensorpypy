#!/usr/bin/env python3
"""Check a port's outputs with the dashboard's own accuracy check.

    suite_check.py SUITE MODEL DUMP_DIR [--export EXPORT_DIR]

DUMP_DIR is what run_port.py --dump wrote (outputs flattened in pytree order).
Mirrors check_accuracy() in benchmarks/dynamo/common.py with the port in the
compiled model's place: float64 golden from cast_to_fp64 and run_n_iterations,
eager from run_n_iterations under reset_rng_state, verdict from
torch._dynamo.utils.same() with the runner's tolerance, cosine flag and
multipliers. With --export, the runner's inputs are first checked to be
byte-identical to the port's.

Prints one `check ...` line: pass, and the RMSE against float64 of the port and
of eager torch over the float outputs finite in all three (diagnostic only).
A float64 reference that does not fit on the GPU is computed on the CPU, and
eager runs on the model itself when a copy does not fit. If eager gives NaN
(pyhpc isoneutral mixing in float32 overflows to inf * 0), the verdict is taken
with equal_nan and the line says eager_nan=1.
"""
import copy
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import suite_census  # noqa: E402


def load_dump(path):
    import numpy as np
    import torch
    idx = json.load(open(os.path.join(path, "index.json")))
    blob = open(os.path.join(path, "data.bin"), "rb")
    out = []
    for off, shape, dtype in idx["outputs"]:
        blob.seek(off)
        n = 1
        for d in shape:
            n *= d
        a = np.frombuffer(blob.read(n * 4), dtype=np.float32).copy()
        out.append(torch.from_numpy(a).reshape(shape))
    return out


def digest(t):
    return hashlib.sha256(t.detach().contiguous().cpu().numpy()
                          .tobytes()).hexdigest()


def main(argv):
    import torch
    from torch._dynamo.utils import clone_inputs, same
    from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten
    suite, name, dump = argv[1], argv[2], os.path.abspath(argv[3])
    export = os.path.abspath(argv[argv.index("--export") + 1]) \
        if "--export" in argv else None
    runner, args = suite_census.make_runner(suite, name)
    import common
    _, _, model, inputs, batch = runner.load_model("cuda", name,
                                                   batch_size=None)
    model, inputs = runner.cast_based_on_args(model, inputs)

    if export:
        idx = json.load(open(os.path.join(export, "index.json")))
        mine = [v for v in (inputs.values() if isinstance(inputs, dict)
                            else inputs) if isinstance(v, torch.Tensor)]
        blob = open(os.path.join(export, "data.bin"), "rb")
        theirs = idx["inputs"] + [idx["kwargs"][k] for k in
                                  (inputs.keys() if isinstance(inputs, dict)
                                   else [])]
        for t, (off, shape, dtype) in zip(mine, theirs):
            blob.seek(off)
            raw = blob.read(t.numel() * t.element_size())
            if hashlib.sha256(raw).hexdigest() != digest(t):
                raise SystemExit("suite_check: the runner's inputs are not "
                                 "the ones the port ran on")

    def golden(dev):
        on = lambda x: x.to(dev) if isinstance(x, torch.Tensor) else x
        m64, i64 = common.cast_to_fp64(copy.deepcopy(model).to(dev),
                                       tree_map(on, clone_inputs(inputs)))
        out = runner.run_n_iterations(m64, i64, runner.model_iter_fn)
        return tree_map(lambda x: x.to("cuda", torch.float64)
                        if isinstance(x, torch.Tensor) and
                        x.is_floating_point() else x, out)

    with torch.no_grad():
        try:
            fp64 = golden("cuda")
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            model.cpu()
            fp64 = golden("cpu")
            model.cuda()
        common.reset_rng_state()
        try:
            correct = runner.run_n_iterations(copy.deepcopy(model),
                                              clone_inputs(inputs),
                                              runner.model_iter_fn)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            common.reset_rng_state()
            correct = runner.run_n_iterations(model, clone_inputs(inputs),
                                              runner.model_iter_fn)
    tol, cosine = runner.get_tolerance_and_cosine_flag(False, "cuda", name)

    leaves, spec = tree_flatten(correct)
    ours = load_dump(dump)
    slots = [i for i, v in enumerate(leaves) if isinstance(v, torch.Tensor)
             and v.is_floating_point()]
    if len(slots) != len(ours):
        raise SystemExit("suite_check: the port gave %d float outputs, eager "
                         "gives %d" % (len(ours), len(slots)))
    flat_new = list(leaves)
    for i, t in zip(slots, ours):
        if tuple(t.shape) != tuple(leaves[i].shape):
            raise SystemExit("suite_check: output %d is %s, eager's %s"
                             % (i, list(t.shape), list(leaves[i].shape)))
        flat_new[i] = t.to(leaves[i].device, leaves[i].dtype)
    new = tree_unflatten(flat_new, spec)

    eager_nan = any(bool(torch.isnan(leaves[i]).any()) for i in slots)
    ok = same(correct, new, fp64, equal_nan=runner.equal_nan or eager_nan,
              use_larger_multiplier_for_smaller_tensor=runner
              .use_larger_multiplier_for_smaller_tensor(name),
              cos_similarity=cosine, tol=tol, force_max_multiplier=False,
              use_iou_for_bool=runner.use_iou_for_bool_accuracy(name),
              iou_threshold=runner.get_iou_threshold(name))

    f64 = [v for v in tree_flatten(fp64)[0] if isinstance(v, torch.Tensor)
           and v.is_floating_point()]
    num_r = num_t = cnt = 0.0
    for i, g in zip(slots, f64):
        rf, tf, gf = (x.reshape(-1) for x in (flat_new[i], leaves[i], g))
        for s in range(0, gf.numel(), 1 << 24):
            r, t, gc = (x[s:s + (1 << 24)].double() for x in (rf, tf, gf))
            m = torch.isfinite(r) & torch.isfinite(t) & torch.isfinite(gc)
            num_r += ((r - gc)[m] ** 2).sum().item()
            num_t += ((t - gc)[m] ** 2).sum().item()
            cnt += m.sum().item()
    print("check suite=%s model=%s batch=%s pass=%d tol=%g cosine=%d "
          "rmse=%.4g torch_rmse=%.4g eager_nan=%d" % (
              suite, name, batch, int(bool(ok)), tol, int(bool(cosine)),
              (num_r / cnt) ** 0.5, (num_t / cnt) ** 0.5, int(eager_nan)),
          flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
