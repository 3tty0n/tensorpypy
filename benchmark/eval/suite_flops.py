#!/usr/bin/env python3
import collections
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import suite_census  # noqa: E402

KIND = {"mm": "matmul", "addmm": "matmul", "bmm": "matmul", "baddbmm": "matmul",
        "convolution": "conv", "_convolution": "conv",
        "_scaled_dot_product_efficient_attention": "attention",
        "_scaled_dot_product_flash_attention": "attention",
        "_scaled_dot_product_cudnn_attention": "attention"}


def main(argv):
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    suite, name = argv[1], argv[2]
    runner, _ = suite_census.make_runner(suite, name)
    _, _, model, inputs, batch = runner.load_model("cuda", name, batch_size=None)
    model, inputs = runner.cast_based_on_args(model, inputs)
    fc = FlopCounterMode(display=False)
    with torch.no_grad(), fc:
        runner.forward_pass(model, inputs)
    by = collections.Counter()
    for op, n in fc.get_flop_counts()["Global"].items():
        by[KIND.get(str(op).split(".")[1] if "." in str(op) else str(op), "other")] += n
    print("flops suite=%s model=%s batch=%s total=%d matmul=%d conv=%d attention=%d other=%d"
          % (suite, name, batch, fc.get_total_flops(), by["matmul"], by["conv"],
             by["attention"], by["other"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
