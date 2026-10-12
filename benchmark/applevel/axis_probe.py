"""The axis change rule, derived against hand-written guard recovery.
The rule is benchmark/motion/axis_rule.py: the same vector addressed along the other axis of a square matrix.

    axis_probe.py ARM STEPS SWITCH [--n N] [--scoped]
    ARM: derived | handwritten | direct | drain
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'motion'))

import _metatensor
import axis_rule

N = 192
ARMS = ('derived', 'handwritten', 'direct', 'drain')


def median(xs):
    s = sorted(xs)
    k = len(s)
    return 0.0 if not k else (s[k // 2] if k % 2
                              else 0.5 * (s[k // 2 - 1] + s[k // 2]))


def mvalues(n):
    return [((i * 31 + 5) % 83) * 0.01 - 0.4 for i in range(n * n)]


def vvalues(n):
    return [((i * 17 + 3) % 53) * 0.02 - 0.5 for i in range(n)]


def run(arm, steps, switch, n, scoped=False):
    mv, vv = mvalues(n), vvalues(n)
    base = _metatensor.tensor(mv, [n, n], False, 'float64')
    one = _metatensor.scalar(1.0, 'float64')
    along = {}
    for axis in ('columns', 'rows'):
        along[axis] = _metatensor.tensor(vv, axis_rule.vector_shape(axis, n),
                                         False, 'float64')
    us = []
    got = {}
    anchor = None
    l0, k0 = _metatensor.launch_count(), _metatensor.kernel_count()
    lsw = ksw = 0
    for i in range(steps):
        if i == switch:
            lsw, ksw = _metatensor.launch_count(), _metatensor.kernel_count()
        t0 = time.time()
        region = base.mul(base).add(one).relu()
        axis = 'columns' if i < switch else 'rows'
        if arm == 'derived':
            out = axis_rule.apply(region, along[axis])
        elif arm == 'direct' or arm == 'drain':
            anchor = region.sum().force()
            out = axis_rule.apply(region, along[axis])
        else:
            out = axis_rule.apply(region.force(), along[axis])
        got[axis] = out.sum().item()
        if scoped:
            del region, out
        us.append((time.time() - t0) * 1e6)
    if anchor is not None:
        got['anchor'] = anchor.item()
    nafter = steps - switch
    counters = ((lsw - l0) / float(switch or 1),
                (_metatensor.launch_count() - lsw) / float(nafter or 1),
                ksw - k0, _metatensor.kernel_count() - ksw)
    retained = -1
    if hasattr(_metatensor, 'live_bytes'):
        retained = _metatensor.live_bytes()
    return us, got, retained, counters


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith('--')]
    arm = argv[0] if argv else 'derived'
    steps = int(argv[1]) if len(argv) > 1 else 200
    switch = int(argv[2]) if len(argv) > 2 else steps // 2
    n = N
    if '--n' in sys.argv:
        n = int(sys.argv[sys.argv.index('--n') + 1])
    if arm not in ARMS:
        sys.stderr.write('axis_probe: unknown arm %r\n' % arm)
        return 2
    drain = os.environ.get('METATENSOR_DRAIN', '') not in ('', '0')
    if drain != (arm == 'drain'):
        sys.stderr.write('axis_probe: arm %s needs METATENSOR_DRAIN %s\n'
                         % (arm, 'set' if arm == 'drain' else 'unset'))
        return 2

    scoped = '--scoped' in sys.argv
    us, got, retained, c = run(arm, steps, switch, n, scoped)

    mv, vv = mvalues(n), vvalues(n)
    squared = [v * v + 1.0 for v in mv]
    squared = [x if x > 0.0 else 0.0 for x in squared]
    ok = 1
    checks = [(axis, axis_rule.reference(squared, vv, n, axis))
              for axis in ('columns', 'rows')]
    checks.append(('anchor', sum(squared)))
    for axis, want in checks:
        if axis not in got:
            continue
        if abs(got[axis] - want) > 1e-9 * max(1.0, abs(want)):
            sys.stderr.write('axis_probe: %s %r, the rule says %r\n'
                             % (axis, got[axis], want))
            ok = 0

    after = median(us[switch + 8:]) if steps > switch + 8 else median(us)
    print('axis arm=%s scoped=%d steps=%d switch=%d n=%d step_us=%.1f '
          'at_us=%.1f '
          'retained_bytes=%d launches_before=%.2f launches_after=%.2f '
          'kernels_before=%d kernels_after=%d pass=%d'
          % (arm, int(scoped), steps, switch, n, after,
             us[switch] if switch < len(us) else 0.0, retained,
             c[0], c[1], c[2], c[3], ok))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
