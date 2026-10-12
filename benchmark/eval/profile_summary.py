#!/usr/bin/env python3
import csv
import glob
import os
import re
import sys

CATS = ["matmul", "conv", "attention", "reduce", "pointwise", "layout", "other"]
ATTN = re.compile(r"flash|fmha|attention|sdpa|efficient", re.I)
CONV = re.compile(r"conv|fprop|dgrad|implicit|cudnn", re.I)
MATMUL = re.compile(r"gemm|cutlass|xmma|triton_tem|_mm|bmm|cublas|matmul", re.I)
CONVERT = re.compile(r"convertTensor|nchwToNhwc|nhwcToNchw|CUDA mem|memcpy|memset", re.I)
LAYOUT = re.compile(r"transpose|permute|clone|_cat|copy|im2col|gather|scatter|index", re.I)
REDUCE = re.compile(r"triton_red|triton_per|reduce|softmax|norm|splitK", re.I)
POINT = re.compile(r"triton_poi|elementwise|vectorized|unrolled", re.I)
ARITH = re.compile(r"arith\.(addf|subf|mulf|divf|maxnumf|minnumf|maximumf|minimumf)|math\.|tt\.dot|tt\.reduce|tt\.extern")


def ours_kinds(ttir_dir):
    kinds = {}
    for f in glob.glob(os.path.join(ttir_dir, "*.ttir")):
        src = open(f).read()
        m = re.search(r"tt\.func public @(\w+)\(", src)
        if not m:
            continue
        if "tt.dot" in src:
            if "math.exp" in src and "tt.reduce" in src:
                k = "attention"
            elif "divsi" in src:
                k = "conv"
            else:
                k = "matmul"
        elif "tt.reduce" in src:
            k = "reduce"
        elif not ARITH.search(src):
            k = "layout"
        else:
            k = "pointwise"
        kinds[m.group(1)] = k
    return kinds


def kind_of(name, kinds):
    base = name.split("(")[0].strip()
    if base in kinds:
        return kinds[base]
    if base.startswith("triton_poi"):
        return "layout" if LAYOUT.search(base) and not re.search(r"add|mul|relu|norm|sub|div|exp|tanh|gelu|silu|sigmoid|mean|sum|max", base) else "pointwise"
    if base.startswith("triton_red") or base.startswith("triton_per"):
        return "reduce"
    if CONVERT.search(name):
        return "layout"
    if ATTN.search(name):
        return "attention"
    if base.startswith("triton_tem") or MATMUL.search(name):
        return "conv" if CONV.search(name) else "matmul"
    if CONV.search(name):
        return "conv"
    if REDUCE.search(name):
        return "reduce"
    if LAYOUT.search(name):
        return "layout"
    if POINT.search(name):
        return "pointwise"
    return "other"


BOUNDARY = ("cuMemcpyDtoH", "cudaDeviceSynchronize", "cuCtxSynchronize",
            "cudaStreamSynchronize", "cuStreamSynchronize")


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return 0.0 if not n else (xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2]))


def table(c, name):
    return c.execute("select count(*) from sqlite_master where name=?", (name,)).fetchone()[0]


def windows(path, kdir, repeat):
    import sqlite3
    db = sqlite3.connect(path)
    kinds = ours_kinds(kdir) if kdir and kdir != "-" else {}
    names = dict(db.execute("select id, value from StringIds"))
    api = sorted((s, e, c, names[n]) for s, e, c, n in
                 db.execute("select start, end, correlationId, nameId from CUPTI_ACTIVITY_KIND_RUNTIME"))
    bounds = [e for s, e, c, n in api if n.startswith(BOUNDARY)]
    loads = [s for s, e, c, n in api if n.startswith("cuModuleLoad")]
    launch_at = dict((c, (s, e, n)) for s, e, c, n in api)
    gpu = []
    for s, e, c, n in db.execute("select start, end, correlationId, demangledName from CUPTI_ACTIVITY_KIND_KERNEL"):
        ls, le, an = launch_at.get(c, (s, s, ""))
        gpu.append((ls, s, e, kind_of(names[n], kinds), names[n], le - ls, c, an))
    for t in ("CUPTI_ACTIVITY_KIND_MEMSET", "CUPTI_ACTIVITY_KIND_MEMCPY"):
        if table(db, t):
            for s, e, c in db.execute("select start, end, correlationId from %s" % t):
                ls, le, an = launch_at.get(c, (s, s, ""))
                gpu.append((ls, s, e, "layout", t, le - ls, c, an))
    gpu.sort()
    if repeat and len(bounds) > 1.5 * (repeat + 1):
        k = int(round((len(bounds) - 1) / float(repeat)))
        bounds = bounds[::k]
    out = []
    j = 0
    for k in range(1, len(bounds)):
        lo, hi = bounds[k - 1], bounds[k]
        while j < len(gpu) and gpu[j][0] < lo:
            j += 1
        mine = []
        while j < len(gpu) and gpu[j][0] < hi:
            mine.append(gpu[j])
            j += 1
        if mine and not any(lo <= t < hi for t in loads):
            out.append((lo, hi, mine))
    return out


def busy_of(mine):
    busy, cs, ce = 0, None, None
    for g in sorted(mine, key=lambda x: x[1]):
        s, e = g[1], g[2]
        if ce is None or s > ce:
            if ce is not None:
                busy += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return busy + (ce - cs)


def main(argv):
    rows = []
    kdir = argv[2] if len(argv) > 2 else "-"
    repeat = int(argv[3]) if len(argv) > 3 else 0
    for lo, hi, mine in windows(argv[1], kdir, repeat):
        r = {"wall": hi - lo, "busy": busy_of(mine), "events": len(mine)}
        for c in CATS:
            r[c + "_ms"] = sum(g[2] - g[1] for g in mine if g[3] == c)
            r[c + "_n"] = sum(1 for g in mine if g[3] == c)
        rows.append(r)
    m = lambda key: median([r[key] for r in rows])
    out = ["iters=%d" % len(rows), "wall_ms=%.3f" % (m("wall") / 1e6),
           "busy_ms=%.3f" % (m("busy") / 1e6),
           "host_ms=%.3f" % (median([r["wall"] - r["busy"] for r in rows]) / 1e6),
           "events=%.1f" % m("events")]
    out += ["%s_ms=%.3f" % (c, m(c + "_ms") / 1e6) for c in CATS]
    out += ["%s_n=%.1f" % (c, m(c + "_n")) for c in CATS]
    print("profile " + " ".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
