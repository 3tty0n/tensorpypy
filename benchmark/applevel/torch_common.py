import array, json, os, sys, time

import torch

import inputs
import warmup_common

# Policy: matmul TF32 off, cudnn TF32 on, unless TORCH_CUDNN_TF32 overrides both (resnet_torch.py re-sets the same flags).
if os.environ.get('TORCH_CUDNN_TF32') == '0':
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
else:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
_tf32_matmul = int(torch.backends.cuda.matmul.allow_tf32)
_tf32_cudnn = int(torch.backends.cudnn.allow_tf32)
if os.environ.get('TORCH_CUDNN_TF32') != '0' and (_tf32_matmul, _tf32_cudnn) != (0, 1):
    raise AssertionError('tf32 policy violated: tf32_matmul=%d tf32_cudnn=%d' %
                         (_tf32_matmul, _tf32_cudnn))
sys.stderr.write('torch_common: tf32_matmul=%d tf32_cudnn=%d\n' %
                 (_tf32_matmul, _tf32_cudnn))


class Args(object):
    pass


def argv():
    a = Args()
    a.mode = sys.argv[1]
    a.outdir = sys.argv[2]
    a.iters = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    a.warmup = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    a.dtname = os.environ.get('RTENSOR_DTYPE', 'float32')
    a.dtype = getattr(torch, a.dtname)
    a.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    a.cfg = json.load(open(os.path.join(a.outdir, 'index.json')))
    # INPUT_SEED=k>0 selects a derived input (see inputs.py), the same integer recipe as the other sides.
    a.seed = inputs.seed()
    inputs.perturb_cfg(a.cfg, a.seed)
    return a


def batch_argv():
    return int(sys.argv[5]) if len(sys.argv) > 5 else 1


def flat_weights(outdir):
    buf = array.array('f')
    path = os.path.join(outdir, 'weights.bin')
    buf.fromfile(open(path, 'rb'), os.path.getsize(path) // 4)
    if sys.byteorder != 'little':
        buf.byteswap()
    return torch.tensor(buf, dtype=torch.float32)


def image(a):
    off, shape = a.cfg['index']['image']
    n = shape[0] * shape[1] * shape[2]
    buf = array.array('f')
    path = os.path.join(a.outdir, 'weights.bin')
    buf.fromfile(open(path, 'rb'), os.path.getsize(path) // 4)
    px = buf[off:off + n]
    k = inputs.seed()
    if k:
        # Added in double, stored as float32, as the pypy side does.
        px = array.array('f', [px[i] + inputs.noise(i, k) for i in range(n)])
    return torch.tensor(px).view(1, *shape)


def hf_kwargs(a):
    """Torch-TensorRT 2.9 lowers masked scaled_dot_product_attention wrongly (BERT logits off by ~25), so use eager attention."""
    return {'attn_implementation': 'eager'} if a.mode == 'tensorrt' else {}



# TensorRT has no f64 kernels (i8/f16/f32/bf16/f8/f4). A rejected dtype falls back to eager without raising, so refuse the point.
TRT_PRECISIONS = ('float16', 'float32', 'bfloat16', 'int8')


def require_trt_dtype(dtype):
    name = str(dtype).replace('torch.', '')
    if name not in TRT_PRECISIONS:
        raise SystemExit('TensorRT has no %s kernels; refusing to report an '
                         'eager fallback as a tensorrt row' % name)

def compiled(fwd, a):
    """eager: as is. compile: Inductor. compile-ro/compile-mat: Inductor under reduce-overhead (CUDA graphs) / max-autotune. tensorrt: Torch-TensorRT via torch.compile."""
    if a.mode == 'compile':
        return torch.compile(fwd)
    if a.mode == 'compile-ro':
        return torch.compile(fwd, mode='reduce-overhead')
    if a.mode == 'compile-mat':
        return torch.compile(fwd, mode='max-autotune')
    if a.mode == 'tensorrt':
        import torch_tensorrt  # noqa: F401  registers the backend
        require_trt_dtype(a.dtype)
        return torch.compile(fwd, backend='tensorrt',
                             options={'enabled_precisions': {a.dtype},
                                      'disable_tf32': True,
                                      'min_block_size': 1})
    return fwd


def profile_launches(fwd, args, a, n=5):
    """Dead-code guard: CUDA kernels launched per forward.

    Runs in a separate window after the timed loop, as the profiler adds milliseconds per iteration."""
    if a.dev != 'cuda':
        return None
    try:
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU,
                                 ProfilerActivity.CUDA]) as prof:
            for _ in range(n):
                fwd(*args)
            torch.cuda.synchronize()
        total = 0
        for ev in prof.events():
            total += len(getattr(ev, 'kernels', None) or [])
        return float(total) / n
    except Exception as exc:
        sys.stderr.write('torch_common: launch profiling failed: %s\n' % exc)
        return None


def explain(fwd, args):
    """mode=explain: one torch._dynamo.explain call, no timing. Prints
    'graphs=<n> breaks=<n> ops=<n>' and exits."""
    import torch._dynamo as dynamo
    with torch.no_grad():
        out = dynamo.explain(fwd)(*args)
    print('graphs=%d breaks=%d ops=%d' %
         (out.graph_count, out.graph_break_count, out.op_count))
    sys.exit(0)


def timed(fwd, args, a):
    """Warm-up, sync, timed loop, sync. The first forward goes to a.first_run_ms (compile plus one run in compile mode)."""
    if a.mode == 'explain':
        explain(fwd, args)
    with torch.no_grad():
        n = warmup_common.trace_n()
        if n:
            us = []
            for i in range(n):
                t0 = time.time()
                logits = fwd(*args)
                if a.dev == 'cuda':
                    torch.cuda.synchronize()
                dt = (time.time() - t0) * 1e6
                us.append(dt)
                print('iter=%d us=%.1f' % (i, dt))
            warmup_common.report_trace(us)
            sys.exit(0)
        t0 = time.time()
        logits = fwd(*args)
        if a.dev == 'cuda':
            torch.cuda.synchronize()
        a.first_run_ms = (time.time() - t0) * 1e3
        for i in range(a.warmup - 1):
            logits = fwd(*args)
        if a.dev == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        for i in range(a.iters):
            logits = fwd(*args)
        if a.dev == 'cuda':
            torch.cuda.synchronize()
        acc = logits.double().sum().item()
        steady_us = (time.time() - t0) / a.iters * 1e6
        if a.mode in ('compile-ro', 'compile-mat'):
            # Both modes use cudagraphs and replay into a static output buffer. The caller reads logits after this returns, so clone (outside the timed loop).
            logits = logits.clone()
        # After the clone, the profiling window would overwrite the cudagraph output buffer.
        a.launches_per_iter = profile_launches(fwd, args, a)
    return logits, acc, steady_us


def batch_identical(logits):
    """[B, seq, vocab] from B copies of the same sequence must be bit-identical row block to row block."""
    if logits.dim() < 3 or logits.shape[0] == 1:
        return 1
    return int((logits - logits[0]).abs().max().item() == 0.0)


def reference_name():
    dtype = os.environ.get('RTENSOR_DTYPE', 'float32')
    return inputs.reference_name(dtype, inputs.seed())


def tolerance(refmax=1.0):
    """Absolute tolerance: config.sh's tolerance_for fraction (MODEL_TOL) times the reference's largest absolute value, the same for every system."""
    try:
        frac = float(os.environ.get('MODEL_TOL', '2e-5'))
    except ValueError:
        frac = 2e-5
    return frac * refmax


def compare(outdir, logits, argmax=None):
    ref = os.path.join(outdir, reference_name())
    if not os.path.exists(ref):
        return ''
    other = torch.frombuffer(open(ref, 'rb').read(),
                             dtype=torch.float32).to(logits.device)
    other = other.view_as(logits.float())
    d = (logits.float() - other).abs().max().item()
    tol = tolerance(other.abs().max().item())
    diff = ' maxabsdiff=%.6g tol=%.6g pass=%d' % (d, tol, int(d <= tol))
    if argmax is not None:
        mine = other.argmax(-1).tolist()
        diff += ' argmax_match=%d/%d' % (
            sum(int(x == y) for x, y in zip(argmax, mine)), len(argmax))
    return diff


def report(line, order, diff, a=None):
    if a is not None and getattr(a, 'launches_per_iter', None) is not None:
        line += ' launches_per_iter=%.1f' % a.launches_per_iter
    if a is not None and getattr(a, 'first_run_ms', None) is not None:
        line += ' compile_ms=-1 first_run_ms=%.1f' % a.first_run_ms
    print(line)
    print('argmax %s%s' % (' '.join(str(a) for a in order), diff))
