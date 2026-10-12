#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from profile_summary import CATS, busy_of, median, windows  # noqa: E402

FIELDS = ["n_small", "t_small", "n_mid", "t_mid", "n_large", "t_large",
          "api_ms", "api_us_per_launch", "host_span_ms", "busy_ms",
          "bubbles", "bubble_ms", "lead_ms", "tail_ms"] + \
         ["avg_us_" + c for c in CATS]


def one(lo, hi, mine):
    r = {}
    d = [(g[2] - g[1]) / 1e3 for g in mine]
    for name, a, b in (("small", 0, 10), ("mid", 10, 100), ("large", 100, 1e18)):
        sel = [x for x in d if a <= x < b]
        r["n_" + name] = len(sel)
        r["t_" + name] = sum(sel) / 1e3
    calls = dict((g[6], g[5]) for g in mine if "Launch" in g[7])
    api = list(calls.values()) or [0]
    r["api_ms"] = sum(api) / 1e6
    r["api_us_per_launch"] = sum(api) / 1e3 / len(api)
    r["host_span_ms"] = (max(g[0] for g in mine) - lo) / 1e6
    r["busy_ms"] = busy_of(mine) / 1e6
    by_start = sorted(mine, key=lambda g: g[1])
    gaps = []
    end = by_start[0][2]
    for g in by_start[1:]:
        if g[1] > end:
            gaps.append(g[1] - end)
        end = max(end, g[2])
    gaps = [x for x in gaps if x > 1000]
    r["bubbles"] = len(gaps)
    r["bubble_ms"] = sum(gaps) / 1e6
    r["lead_ms"] = (by_start[0][1] - lo) / 1e6
    r["tail_ms"] = (hi - end) / 1e6
    for c in CATS:
        xs = [g[2] - g[1] for g in mine if g[3] == c]
        r["avg_us_" + c] = sum(xs) / 1e3 / len(xs) if xs else 0.0
    return r


def main(argv):
    work, out = argv[1], argv[2]
    repeat = int(argv[3]) if len(argv) > 3 else 30
    with open(out, "w") as f:
        f.write("name\tsystem\t" + "\t".join(FIELDS) + "\n")
        for path in sorted(os.listdir(work)):
            if not path.endswith(".sqlite"):
                continue
            stem = path[:-len(".sqlite")]
            name, system = stem.rsplit(".", 1)
            kdir = os.path.join(work, name + ".kernels") if system == "ours" else "-"
            ws = windows(os.path.join(work, path), kdir, repeat)
            if not ws:
                continue
            rows = [one(lo, hi, mine) for lo, hi, mine in ws]
            f.write("%s\t%s\t%s\n" % (name, system, "\t".join(
                "%.4g" % median([r[k] for r in rows]) for k in FIELDS)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
