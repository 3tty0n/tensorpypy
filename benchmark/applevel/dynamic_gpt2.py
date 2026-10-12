"""Dynamic sequence-length sweep, MetaTensor side.
Same timed step as dynamic_gpt2_torch.py: build_us (token ids, position slice, causal mask), step_us (model plus logits.sum().item()), total_us.
Phases: `first` walks ORDER cold, `revisit` walks it again warm, `new` walks NEW_ORDER. Warm-up runs at WARMUP_LEN, outside both.
"""

import os, sys, time

import pypyjit

import _metatensor
import common
import gpt2

ORDER = [32, 48, 64, 96, 128]
NEW_ORDER = [160, 192, 224, 256, 288]
WARMUP_LEN = 16
PHASES = [('first', ORDER), ('revisit', ORDER), ('new', NEW_ORDER)]


def jit_counters():
    s = pypyjit.get_stats_snapshot()
    c = s.counters
    t = s.counter_times
    return (c['TOTAL_COMPILED_LOOPS'], c['TOTAL_COMPILED_BRIDGES'],
            t.get('TRACING', 0.0) + t.get('BACKEND', 0.0))


def build(cfg, buf, model, masks, t, dtype):
    tokens = cfg['tokens']
    ids = [tokens[i % len(tokens)] for i in range(t)]
    idx = common.tensor([float(tok) for tok in ids], [t], False, dtype)
    d = cfg['n_embd']
    wpe_off = cfg['index']['wpe'][0]
    pos = common.tensor(list(buf[wpe_off:wpe_off + t * d]), [t, d], False,
                        dtype)
    mask = masks.get(t)
    if mask is None:
        mask = masks[t] = common.causal_mask(cfg['n_head'], t, dtype)
    for blk in model.blocks:
        blk.attn.mask = mask
    return idx, pos


def main():
    outdir = sys.argv[1]
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    dtype = os.environ.get('RTENSOR_DTYPE', 'float32')
    cfg, buf = common.load(outdir)
    model = gpt2.build_model(cfg, buf, dtype)
    masks = {}

    for _ in range(5):
        idx, pos = build(cfg, buf, model, masks, WARMUP_LEN, dtype)
        model(idx, pos).sum().item()

    per_phase = max(1, iters // len(PHASES))
    print('pass\tlength\tstep_us\tbuild_us\tloops\tbridges\tkernels'
          '\tlaunches\tgraphs\tnew_graphs\tcompile_ms')
    for phase, lengths in PHASES:
        for i in range(per_phase):
            t = lengths[i % len(lengths)]
            loops0, bridges0, ct0 = jit_counters()
            k0 = _metatensor.kernel_count()
            l0 = _metatensor.launch_count()
            b0 = time.time()
            idx, pos = build(cfg, buf, model, masks, t, dtype)
            b1 = time.time()
            out = model(idx, pos)
            out.sum().item()
            s1 = time.time()
            loops1, bridges1, ct1 = jit_counters()
            print('%s\t%d\t%.1f\t%.1f\t%d\t%d\t%d\t%d\t0\t0\t%.3f' % (
                phase, t, (s1 - b1) * 1e6, (b1 - b0) * 1e6,
                loops1 - loops0, bridges1 - bridges0,
                _metatensor.kernel_count() - k0,
                _metatensor.launch_count() - l0,
                (ct1 - ct0) * 1e3))


if __name__ == '__main__':
    main()
