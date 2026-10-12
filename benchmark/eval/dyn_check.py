#!/usr/bin/env python3
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import suite_census  # noqa: E402
from suite_check import load_dump  # noqa: E402


def main(argv):
    import torch
    from torch._dynamo.utils import same
    suite, name, ours, eager = argv[1], argv[2], argv[3], argv[4]
    runner, _ = suite_census.make_runner(suite, name)
    tol, cosine = runner.get_tolerance_and_cosine_flag(False, "cuda", name)
    status = 0
    for d in sorted(os.listdir(ours)):
        gold = os.path.join(eager, "fp64-" + d.split("-")[-1])
        if not os.path.isdir(os.path.join(eager, d)) or not os.path.isdir(gold):
            print("dyncheck suite=%s model=%s point=%s pass=0 missing=1" % (suite, name, d))
            status = 1
            continue
        ref = load_dump(os.path.join(eager, d))
        res = load_dump(os.path.join(ours, d))
        fp64 = [t.double() for t in load_dump(gold)]
        nan = any(bool(torch.isnan(t).any()) for t in ref)
        ok = len(ref) == len(res) and all(
            a.shape == b.shape for a, b in zip(ref, res)) and same(
            ref, res, fp64, tol=tol, cos_similarity=cosine, equal_nan=nan,
            use_larger_multiplier_for_smaller_tensor=runner
            .use_larger_multiplier_for_smaller_tensor(name),
            force_max_multiplier=False)
        print("dyncheck suite=%s model=%s point=%s pass=%d" % (suite, name, d, int(bool(ok))))
        status |= not ok
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv))
