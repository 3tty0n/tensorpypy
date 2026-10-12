"""INPUT_SEED: the same model on a different input, derived by integer arithmetic so pypy2, torch and jax agree bitwise.
Seed 0 is the stored input. For k>0 text ids are rotated left by k and shifted by k, vision pixels get +-0.05 uniform noise.
Self-check: python3 inputs.py
"""

import os

M = 2147483647


def seed():
    try:
        return int(os.environ.get('INPUT_SEED', '0') or 0)
    except ValueError:
        return 0


def tokens(toks, vocab, k):
    n = len(toks)
    if not k or not n:
        return list(toks)
    return [(toks[(i + k) % n] + k) % vocab for i in range(n)]


def noise(i, k):
    h = (i * 2654435761 + k * 40503 + 12345) % M
    h = (h * 1103515245 + 12345) % M
    return 0.05 * (2.0 * h / float(M) - 1.0)


def perturb_cfg(cfg, k):
    """Perturb cfg['tokens'] in place for the text models (an int for vit/mixer)."""
    if k and isinstance(cfg.get('tokens'), list):
        cfg['tokens'] = tokens(cfg['tokens'], cfg['vocab'], k)


def image_span(cfg):
    """(offset, count) of the image inside weights.bin, or None."""
    entry = cfg.get('index', {}).get('image')
    if not entry:
        return None
    off, shape = entry
    n = 1
    for d in shape:
        n *= d
    return off, n


def reference_name(dtype, k):
    """Per-seed reference name. Seed 0 keeps the stored name."""
    base = 'logits_pypy' if dtype == 'float32' else 'logits_pypy_%s' % dtype
    return '%s%s.bin' % (base, '_seed%d' % k if k else '')


# --- early-exit control-flow experiment (bench.sh control) -------------------
# One schedule for all three systems, so the exit-layer sequence belongs to the experiment, not the runtime.
# Confidence is mean(x*x) over the hidden state, measured on distilgpt2 over input rotations k=0..4, float32:
#
#   layer     1       2       3       4       5       6
#   conf    3.9-4.5 59-62   65-70   75-80   93-99   173-190
#
# Each threshold sits mid-gap, a >=3% margin on every input. A driver within 2% of its threshold must fail.
#
# (k, threshold) per iteration, rotating:
#   stable   one input, one threshold, every iteration exits at layer 4
#   varying  a rotating request mix, exits at layers 2, 3, 5, 6
CONTROL = {
    'stable':  [(0, 72.5)],
    'varying': [(0, 30.0), (1, 63.5), (2, 86.0), (3, 1e9)],
}
# Expected exit layers, in order. 1e9 is never reached, so that slot runs all six layers.
CONTROL_EXITS = {'stable': [4], 'varying': [2, 3, 5, 6]}
CONTROL_MARGIN = 0.02


def control_slot(regime, i):
    """(input rotation k, confidence threshold) for iteration i."""
    s = CONTROL[regime]
    return s[i % len(s)]


def control_check(conf, tau):
    """Fail when the confidence is too close to its threshold for the exit layer to reproduce."""
    if abs(conf - tau) < CONTROL_MARGIN * abs(tau):
        raise AssertionError(
            'confidence %.6g is within %g%% of threshold %.6g: the exit layer '
            'is not reproducible across systems' %
            (conf, CONTROL_MARGIN * 100, tau))


def pctl(xs, q):
    """Nearest-rank percentile, defined once so the three drivers agree."""
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))]


def exit_hash(seq):
    """FNV-1a over the exit-layer sequence, as a decimal the tsv can hold."""
    h = 2166136261
    for v in seq:
        h = ((h ^ (v & 0xff)) * 16777619) & 0xffffffff
    return h


def _demo():
    assert seed() >= 0
    t = [1, 2, 3, 4]
    assert tokens(t, 10, 0) == t
    assert tokens(t, 10, 1) == [3, 4, 5, 2]          # rotate left 1, +1, mod 10
    assert tokens([9], 10, 1) == [0]                 # wraps in the vocabulary
    assert len(set(tuple(tokens(t, 10, k)) for k in range(1, 6))) == 5
    for k in range(1, 6):
        vals = [noise(i, k) for i in range(1000)]
        assert all(-0.05 <= v <= 0.05 for v in vals)
        assert len(set(vals)) > 900                  # not a constant
        assert noise(7, k) != noise(7, k + 1)        # the seed moves it
    assert noise(3, 0) == noise(3, 0)                # deterministic
    assert reference_name('float32', 0) == 'logits_pypy.bin'
    assert reference_name('float32', 3) == 'logits_pypy_seed3.bin'
    assert reference_name('float16', 3) == 'logits_pypy_float16_seed3.bin'
    for regime, exits in CONTROL_EXITS.items():
        assert len(exits) == len(CONTROL[regime])
        assert [control_slot(regime, i)[0] for i in range(len(exits))] == \
            list(range(len(exits))) or regime == 'stable'
    assert control_slot('varying', 5) == control_slot('varying', 1)
    assert pctl([1, 2, 3, 4], 0.0) == 1 and pctl([1, 2, 3, 4], 1.0) == 4
    assert pctl([1, 2, 3, 4, 5], 0.5) == 3
    assert exit_hash([2, 3]) != exit_hash([3, 2])
    assert exit_hash([4]) == exit_hash([4])
    control_check(100.0, 72.5)
    try:
        control_check(72.6, 72.5)
    except AssertionError:
        pass
    else:
        raise AssertionError('control_check missed a threshold straddle')
    print('inputs.py ok')


if __name__ == '__main__':
    _demo()
