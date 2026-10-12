"""Two workers, an uncommitted gradient aggregate, and a worker that leaves.

    leave_probe.py HISTORY [--impl motion|adapter] [--steps S] [--at K]
                           [--n N] [--save FILE] [--nostats] [--times]
                           [--policy retain-only] [--sync]

Worker B leaves at step K, with HISTORY h1, h2, h3, h2post or none. Inputs are dyadic and lr = 1/8, so sums are exact and compared bitwise.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'motion'))

import _metatensor as mt
import adapter_leave

try:
    import pypyjit
except ImportError:
    pypyjit = None
if '--nostats' in sys.argv:
    pypyjit = None

HISTORIES = {
    'h1': ('B', 'preempt', 3),
    'h2': ('B', 'fault', 3),
    'h3': ('B', 'fault', 2),
    'h2post': ('B', 'fault', 4),     # first check of step K+1
    'none': None,
}
POLICY = [('preempt', 'retain'), ('fault', 'revoke')]  # [policy]
LR = 0.125


def flag(name, default):
    if name in sys.argv:
        return sys.argv[sys.argv.index(name) + 1]
    return default


def shard(n, k):
    return [float(((i * (5 + 2 * k) + k) % 9) - 4) * 0.5 for i in range(n)]


def grad(x, s):
    return (x * s).relu() + x


def counters():
    loops = bridges = 0
    if pypyjit is not None:
        try:
            c = pypyjit.get_stats_snapshot().counters
            loops = c['TOTAL_COMPILED_LOOPS']
            bridges = c['TOTAL_COMPILED_BRIDGES']
        except Exception:
            pass
    return loops, bridges, mt.launch_count()


# --sync ends every timed step with a host read (w.sum().item()). Without it a step is timed to the end of its asynchronous launches.
SYNC = '--sync' in sys.argv
STEADY = [0, 0]


def mark(step, k):
    """Launch count at the start and end of the hot window [k/2, k)."""
    if step == k // 2:
        STEADY[0] = mt.launch_count()
    elif step == k:
        STEADY[1] = mt.launch_count()


def event_poll(history, k):
    ev = HISTORIES[history]
    if ev is None:
        return None
    who, reason, when = ev
    return who, reason, 3 * k + when


def run_motion(history, steps, k, n, times):
    g = mt.group(['A', 'B'])
    policy = POLICY
    if flag('--policy', '') == 'retain-only':
        policy = [(reason, 'retain') for reason, _ in POLICY]
    for reason, action in policy:  # [motion-policy]
        g.declare(reason, action)  # [motion-policy]
    ev = event_poll(history, k)
    if ev is not None:
        g.arm(*ev)
    xa = mt.tensor(shard(n, 0))
    xb = mt.tensor(shard(n, 1))
    w = mt.tensor([0.0] * n)
    w_k = None
    for step in range(steps):
        mark(step, k)
        t0 = time.time()
        w = motion_step(g, xa, xb, w, mt.scalar(float(1 + step % 4),
                                                'float64'))
        if times is not None:
            if SYNC:
                w.sum().item()
            else:
                w.force()
            times.append((time.time() - t0) * 1e6)
        if step == k:
            w_k = w.tolist()
    return w_k, w.tolist(), g.commits(), g.log(), g.retains()


def motion_step(g, xa, xb, w, s):
    """One training step. A function so its locals die on return, as this interpreter keeps dead locals alive at the loop back-edge."""
    agg = g.aggregate()
    agg.sync()
    agg.contribute('A', grad(xa, s))
    agg.sync()
    agg.contribute('B', grad(xb, s))
    return g.commit(agg, w, LR)


def run_adapter(history, steps, k, n, times):
    ch = adapter_leave.Channel()
    ev = event_poll(history, k)
    if ev is not None:
        ch.arm(*ev)
    agg = adapter_leave.ManualAggregator(['A', 'B'], POLICY, ch)
    xa = mt.tensor(shard(n, 0))
    xb = mt.tensor(shard(n, 1))
    w = mt.tensor([0.0] * n)
    lr = mt.scalar(LR, 'float64')
    w_k = None
    for step in range(steps):
        mark(step, k)
        t0 = time.time()
        w = adapter_step(agg, step, xa, xb, w, lr,
                         mt.scalar(float(1 + step % 4), 'float64'))
        if times is not None:
            if SYNC:
                w.sum().item()
            else:
                w.force()
            times.append((time.time() - t0) * 1e6)
        if step == k:
            w_k = w.tolist()
    return w_k, w.tolist(), agg.commits, agg.events, True


def adapter_step(agg, step, xa, xb, w, lr, s):
    agg.begin(step)
    agg.check()
    agg.contribute('A', grad(xa, s))
    agg.check()
    agg.contribute('B', grad(xb, s))
    return agg.commit(w, lr)


def median(xs):
    s = sorted(xs)
    m = len(s)
    return 0.0 if not m else (s[m // 2] if m % 2
                              else 0.5 * (s[m // 2 - 1] + s[m // 2]))


def hexlist(xs):
    return ','.join([float.hex(x) for x in xs])


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith('--')]
    skip = set()
    for i, a in enumerate(sys.argv):
        if a in ('--impl', '--steps', '--at', '--n', '--save', '--policy'):
            skip.add(sys.argv[i + 1])
    argv = [a for a in argv if a not in skip]
    history = argv[0] if argv else 'h2'
    if history not in HISTORIES:
        sys.stderr.write('leave_probe: unknown history %r\n' % history)
        return 2
    impl = flag('--impl', 'motion')
    steps = int(flag('--steps', '200'))
    k = int(flag('--at', str(steps // 2)))
    n = int(flag('--n', '64'))
    times = [] if '--times' in sys.argv else None

    c0 = counters()
    run = run_motion if impl == 'motion' else run_adapter
    w_k, w_end, commits, log, retains = run(history, steps, k, n, times)
    c1 = counters()

    fields = [('history', history), ('impl', impl), ('steps', steps),
              ('at', k), ('n', n), ('commits', commits),
              ('retains', int(retains)),
              ('loops', c1[0] - c0[0]), ('bridges', c1[1] - c0[1]),
              ('launches_per_step', '%.3f' % ((c1[2] - c0[2]) /
                                              float(steps))),
              ('steady_launches', '%.3f' % ((STEADY[1] - STEADY[0]) /
                                            float(k - k // 2))),
              ('log', '|'.join([e.replace(' ', ':') for e in log]) or '-'),
              ('w_at', hexlist(w_k)), ('w_end', hexlist(w_end))]
    if times is not None:
        fields.append(('step_us', '%.1f' % median(times[k + 10:] or times)))
        fields.append(('before_us', '%.1f' % median(times[k // 2:k])))
        fields.append(('event_us', '%.1f' % times[k]))
    line = ' '.join(['%s=%s' % kv for kv in fields])
    print('leave ' + line)
    save = flag('--save', None)
    if save:
        f = open(save, 'w')
        f.write(line + '\n')
        f.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
