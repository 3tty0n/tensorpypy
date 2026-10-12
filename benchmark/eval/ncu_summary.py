#!/usr/bin/env python3
import csv
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from profile_summary import CATS, kind_of, ours_kinds  # noqa: E402

UNIT = {"byte": 1.0, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9,
        "ns": 1e-6, "us": 1e-3, "ms": 1.0, "s": 1e3}


def main(argv):
    path, kdir = argv[1], argv[2]
    kinds = ours_kinds(kdir) if kdir != "-" else {}
    lines = [l for l in open(path) if l.startswith('"')]
    if len(lines) < 3:
        return 0
    rows = list(csv.reader(lines))
    head, units = rows[0], rows[1]
    col = dict((h, i) for i, h in enumerate(head))
    acc = collections.defaultdict(lambda: [0, 0.0, 0.0, 0.0])
    for r in rows[2:]:
        k = kind_of(r[col["Kernel Name"]], kinds)
        a = acc[k]
        a[0] += 1
        for j, m, scale in ((1, "gpu__time_duration.sum", 1.0), (2, "dram__bytes_read.sum", 1e-6),
                            (3, "dram__bytes_write.sum", 1e-6)):
            v = float(r[col[m]].replace(",", "") or 0)
            a[j] += v * UNIT.get(units[col[m]], 1.0) * scale
    for k in CATS:
        if k in acc:
            n, t, rd, wr = acc[k]
            print("%s %d %.4f %.2f %.2f" % (k, n, t, rd, wr))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
