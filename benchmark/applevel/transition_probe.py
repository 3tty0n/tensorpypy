"""What it costs to carry a fused value across a guard, two ways.
direct keeps the region's descriptor and recompiles its kernel with one more output. drain (METATENSOR_DRAIN=1) re-records the value from the leaves.

    transition_probe.py [ITERS] [SWITCH] [--series] [--rows N] [--cols N]
"""

import sys
import time

import _metatensor

try:
    import pypyjit
except ImportError:
    pypyjit = None

ROWS, COLS = 256, 64
SETTLE = 8          # iterations after the transition that may still compile


def counters():
    loops = bridges = 0
    if pypyjit is not None:
        try:
            c = pypyjit.get_stats_snapshot().counters
            loops = c['TOTAL_COMPILED_LOOPS']
            bridges = c['TOTAL_COMPILED_BRIDGES']
        except Exception:
            pass
    return (loops, bridges, _metatensor.launch_count(),
            _metatensor.kernel_compile_count(), _metatensor.kernel_count())


def median(xs):
    s = sorted(xs)
    n = len(s)
    if not n:
        return 0.0
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def leaf_values(rows, cols):
    return [((i * 37 + 11) % 97) * 0.01 - 0.5 for i in range(rows * cols)]


def weight_values(cols):
    return [((i * 23 + 5) % 61) * 0.02 - 0.6 for i in range(cols * cols)]


def reference(rows, cols):
    """The same arithmetic in Python, to catch a wrong arm.

    root is sum(relu(x*x + 1)). interior is sum(A @ W) with A = x*x, which equals
    sum_k (column sum of A at k) * (row sum of W at k), so no O(rows*cols*cols) loop."""
    vals = leaf_values(rows, cols)
    wvals = weight_values(cols)
    root = 0.0
    acol = [0.0] * cols
    for i, v in enumerate(vals):
        a = v * v
        acol[i % cols] += a
        b = a + 1.0
        root += b if b > 0.0 else 0.0
    interior = 0.0
    for k in range(cols):
        wrow = 0.0
        for j in range(cols):
            wrow += wvals[k * cols + j]
        interior += acol[k] * wrow
    return root, interior


def run(iters, switch, rows, cols, series):
    leaf = _metatensor.tensor(leaf_values(rows, cols), [rows, cols], False,
                              'float64')
    weight = _metatensor.tensor(weight_values(cols), [cols, cols], False,
                                'float64')
    one = _metatensor.scalar(1.0, 'float64')
    us = []
    roots = []
    interiors = []

    c0 = counters()
    t_all = time.time()
    for i in range(iters):
        t0 = time.time()
        a = leaf.mul(leaf)          # interior of the region
        b = a.add(one)
        y = b.relu()                # root of the region
        roots.append(y.sum().item())
        if i >= switch:
            # The interior value, needed by a call the region cannot absorb.
            # Not taken before SWITCH, so the first iteration here fails a guard.
            interiors.append(a.matmul(weight).sum().item())
        us.append((time.time() - t0) * 1e6)
    total_ms = (time.time() - t_all) * 1e3
    c1 = counters()

    if series:
        for i, v in enumerate(us):
            print('series\t%d\t%.1f' % (i, v))
    return us, total_ms, tuple(b - a for a, b in zip(c0, c1)), roots, interiors


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith('--')]
    series = '--series' in sys.argv
    rows, cols = ROWS, COLS
    for flag, name in (('--rows', 'rows'), ('--cols', 'cols')):
        if flag in sys.argv:
            v = int(sys.argv[sys.argv.index(flag) + 1])
            if name == 'rows':
                rows = v
            else:
                cols = v
    iters = int(argv[0]) if argv else 400
    switch = int(argv[1]) if len(argv) > 1 else iters // 2

    us, total_ms, d, roots, interiors = run(iters, switch, rows, cols, series)

    want_root, want_interior = reference(rows, cols)
    # Each value against its own scale. The interior sum cancels to a few units while the root is in the tens of thousands.
    ok = 1
    if roots and abs(roots[-1] - want_root) > 1e-9 * max(1.0, abs(want_root)):
        sys.stderr.write('transition_probe: root %r, expected %r\n'
                         % (roots[-1], want_root))
        ok = 0
    if interiors and abs(interiors[-1] - want_interior) > \
            1e-9 * max(1.0, abs(want_interior)):
        sys.stderr.write('transition_probe: interior %r, expected %r\n'
                         % (interiors[-1], want_interior))
        ok = 0
    if len(interiors) != iters - switch:
        sys.stderr.write('transition_probe: %d interior reads, expected %d\n'
                         % (len(interiors), iters - switch))
        ok = 0

    before = median(us[switch // 2:switch])
    at = us[switch] if switch < len(us) else 0.0
    settling = median(us[switch + 1:switch + 1 + SETTLE])
    after = median(us[switch + 1 + SETTLE:]) or 0.0
    loops, bridges, launches, compiles, kernels = d

    print('transition drain=%d iters=%d switch=%d rows=%d cols=%d '
          'before_us=%.1f at_us=%.1f settling_us=%.1f after_us=%.1f '
          'total_ms=%.1f loops=%d bridges=%d launches=%d compiles=%d '
          'kernels=%d pass=%d'
          % (1 if _drain_on() else 0,
             iters, switch, rows, cols, before, at, settling, after,
             total_ms, loops, bridges, launches, compiles, kernels, ok))
    return 0 if ok else 1


def _drain_on():
    import os
    v = os.environ.get('METATENSOR_DRAIN')
    return v is not None and v != '' and v != '0'


if __name__ == '__main__':
    sys.exit(main())
