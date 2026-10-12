#!/usr/bin/env python3
import csv
import math
import sys

PARTS = [("gemm", ["matmul_ms", "conv_ms"]), ("attention", ["attention_ms"]),
         ("pointwise", ["pointwise_ms", "reduce_ms", "other_ms"]), ("layout", ["layout_ms"])]


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def geomean(xs):
    xs = [x for x in xs if x > 0]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else 0.0


def main(argv):
    prof, flops, out = argv[1], argv[2], argv[3]
    base = argv[4] if len(argv) > 4 else "inductor"
    P = dict(((r["suite"], r["model"], r["system"]), r)
             for r in csv.DictReader(open(prof), delimiter="\t"))
    F = dict(((r["suite"], r["model"]), r) for r in csv.DictReader(open(flops), delimiter="\t"))
    with open(out, "w") as f:
        f.write("suite\tmodels\tspeedup\t" + "\t".join("d_" + k for k, _ in PARTS) +
                "\td_host\tgemm_speed\tattention_speed\tgemm_tflops_ours\tgemm_tflops_base"
                "\tattention_tflops_ours\tattention_tflops_base\n")
        for suite in ("torchbench", "huggingface", "timm"):
            keys = [k[:2] for k in P if k[0] == suite and k[2] == "ours" and
                    num(P[k]["wall_ms"]) and num(P.get(k[:2] + (base,), {}).get("wall_ms"))]
            d = dict((k, 0.0) for k, _ in PARTS + [("host", None)])
            sp, gs, as_, gt, at = [], [], [], [[], []], [[], []]
            for k in keys:
                o, b = P[k + ("ours",)], P[k + (base,)]
                w = num(b["wall_ms"])
                sp.append(w / num(o["wall_ms"]))
                for name, cols in PARTS:
                    d[name] += sum(num(b[c]) - num(o[c]) for c in cols) / w / len(keys)
                d["host"] += ((w - num(b["busy_ms"])) -
                              (num(o["wall_ms"]) - num(o["busy_ms"]))) / w / len(keys)
                fl = F.get(k)
                if not fl:
                    continue
                g = num(fl["matmul"]) + num(fl["conv"])
                a = num(fl["attention"])
                tg = [num(r["matmul_ms"]) + num(r["conv_ms"]) for r in (o, b)]
                ta = [num(r["attention_ms"]) for r in (o, b)]
                if g > 1e9 and all(tg):
                    gs.append(tg[1] / tg[0])
                    for j in (0, 1):
                        gt[j].append(g / tg[j] / 1e9)
                if a > 0 and all(ta):
                    as_.append(ta[1] / ta[0])
                    for j in (0, 1):
                        at[j].append(a / ta[j] / 1e9)
            f.write("%s\t%d\t%.3f\t%s\t%.3f\t%.3f\t%.3f\t%.1f\t%.1f\t%.1f\t%.1f\n" % (
                suite, len(keys), geomean(sp), "\t".join("%.3f" % d[k] for k, _ in PARTS),
                d["host"], geomean(gs), geomean(as_), geomean(gt[0]), geomean(gt[1]),
                geomean(at[0]), geomean(at[1])))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
