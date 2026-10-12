from rpython.rlib import jit
from rpython.rlib.rfloat import formatd
from rpython.rlib.rmd5 import md5
from rpython.rlib.rstring import assert_str0
from rpython.rtyper.lltypesystem import lltype
from rpython.rtyper.lltypesystem import rffi
import os
from rpython.metatensor.core import (BINARY, UNARY, CD_STAGES, CD_WARPS, CONV, GA_ROTHALF, GA_SCATTER, NO_NARROW, gather_dh, gather_heads, gather_kind, init_conv, init_drain, init_gather, init_lazy, init_split, init_static_sizes, ARITY, AXIS_ALL, F64, KERNEL, NO_CONSTS, NDTYPES, NODEARRAY, NOPCODES, NPARAMS, SHAPEARRAY, SUM, config, conv_descs, conv_per_image, has_conv, is_reduction, main_conv, param_slot, slot_param, slot_used, BC_L_COL, BC_L_ROW, BC_L_SCALAR, GA_STRIDED, GA_VIEW, GATHER, bc_mode, nvals, view_of, view_params_list)
from rpython.metatensor.device import (_env, _here, gpu_enabled, profile, rt_cuda_debug_text, rt_cuda_load, rt_cuda_set_budget)
from rpython.metatensor.ttir import (NARROW_COLS, NARROW_WARPS, all_modes, has_gather, input_modes, kernel_row_mode, out_modes, row_root, row_tile, row_warps, to_tile_ir, to_ttir, to_ttir_attn, to_ttir_gather, to_ttir_row_multi, to_ttir_row_tiled, scatter_block)

class SingleKernels(object):
    def __init__(self):
        self.kernels = [_empty_kernel()
                        for i in range(NOPCODES * NPARAMS * NDTYPES)]
        self.done = [False] * NDTYPES
        self.lazy = {}
        self.none = _empty_kernel()

def _empty_kernel():
    k = lltype.malloc(KERNEL)
    k.ninputs = 0
    k.nodes = lltype.malloc(NODEARRAY, 0)
    k.fn = k.sumroot = k.threads = k.shared = k.nextra = 0
    k.wfn = k.wthreads = k.wshared = k.wnextra = 0
    k.narrow = NO_NARROW
    k.vsrc = 0
    k.vroot = -1
    k.rowmode = 0
    k.nouts = 0
    k.n = 0
    k.cols = 0
    k.dtype = F64
    k.modes = 0
    k.outmodes = 0
    k.consts = NO_CONSTS
    k.outputs = lltype.malloc(SHAPEARRAY, 0)
    return k
single_kernels = SingleKernels()

def single_kernel(opcode, p, dtype):
    if opcode == UNARY or opcode == BINARY:
        return single_kernels.lazy.get(_lazy_key(opcode, p, dtype),
                                       single_kernels.none)
    return single_kernels.kernels[(dtype * NOPCODES + opcode) * NPARAMS +
                                  param_slot(opcode, p)]

def _lazy_key(opcode, p, dtype):
    return (p * NOPCODES + opcode) * NDTYPES + dtype

@jit.dont_look_inside
def ensure_single(opcode, p, dtype):
    """Build a UNARY or BINARY kernel on first use, never under an oopspec (single_kernel only looks up)."""
    key = _lazy_key(opcode, p, dtype)
    if key in single_kernels.lazy:
        return
    right = 1 if ARITY[opcode] == 2 else -1
    single_kernels.lazy[key] = build_kernel(ARITY[opcode], [opcode], [0],
                                            [right], [p], dtype)

def init_device():
    init_lazy()
    init_drain()
    init_split()
    init_static_sizes()
    init_gather()
    init_conv()
    try:
        config.block = int(_env('RTENSOR_BLOCK', '4096'))
        config.flat = int(_env('RTENSOR_FLAT_BLOCK', '4096'))
        config.spread = _env('RTENSOR_SPREAD', '1') != '0'
        config.num_warps = int(_env('RTENSOR_WARPS', '8'))
        profile.enabled = os.environ.get('RTENSOR_PROFILE') is not None
        config.debug_unfused = os.environ.get('RTENSOR_DEBUG_UNFUSED') is not None
        config.narrow = _env('RTENSOR_NARROW', '1') != '0'
        config.conv_prec = _env('RTENSOR_CONV_PREC', 'auto')
        if _env('RTENSOR_TF32', '1') == '0':
            config.conv_prec = 'ieee'
        rt_cuda_set_budget(int(_env('RTENSOR_BUDGET_MB', '8')) << 20)
    except ValueError:
        pass
    init_dtype(F64)

@jit.elidable
def dtype_ready(dtype):
    return single_kernels.done[dtype]

def init_dtype(dtype):
    if single_kernels.done[dtype]:
        return
    single_kernels.done[dtype] = True
    for opcode in range(NOPCODES):
        for slot in range(NPARAMS):
            if not slot_used(opcode, slot):
                continue
            opcodes = []
            opcodes.append(opcode)
            lefts = []
            lefts.append(0)
            rights = []
            rights.append(1 if ARITY[opcode] == 2 else -1)
            params = []
            params.append(slot_param(opcode, slot))
            single_kernels.kernels[(dtype * NOPCODES + opcode) * NPARAMS +
                                   slot] = build_kernel(
                ARITY[opcode], opcodes, lefts, rights, params, dtype)
class KernelCache(object):
    def __init__(self):
        self.kernels = {}
kernel_cache = KernelCache()

def cached_kernel(key):
    return kernel_cache.kernels.get(key, lltype.nullptr(KERNEL))

def cache_kernel(key, kernel):
    kernel_cache.kernels[key] = kernel

def drop_kernel(key):
    if key in kernel_cache.kernels:
        del kernel_cache.kernels[key]

def new_kernel(ninputs, nnodes, dtype=F64):
    kernel = lltype.malloc(KERNEL)
    kernel.ninputs = ninputs
    kernel.nodes = lltype.malloc(NODEARRAY, nnodes)
    kernel.fn = kernel.sumroot = kernel.threads = kernel.shared = kernel.nextra = 0
    kernel.wfn = kernel.wthreads = kernel.wshared = kernel.wnextra = 0
    kernel.narrow = NO_NARROW
    kernel.vsrc = 0
    kernel.vroot = -1
    kernel.rowmode = 0
    kernel.nouts = 0
    kernel.n = 0
    kernel.cols = 0
    kernel.dtype = dtype
    kernel.modes = 0
    kernel.outmodes = 0
    kernel.consts = NO_CONSTS
    kernel.outputs = lltype.malloc(SHAPEARRAY, 0)
    return kernel

def add_output(kernel, node):
    old = kernel.outputs
    # The cache holds this object, so drop its entry before the key changes.
    if kernel.fn:
        drop_kernel(kernel_key(kernel))
    new = lltype.malloc(SHAPEARRAY, len(old) + 1)
    for i in range(len(old)):
        new[i] = old[i]
    new[len(old)] = node
    kernel.outputs = new
    if kernel.fn:
        # The compiled code has the old signature. Launching it would pass one
        # output pointer too many and shift n and cols. Recompile.
        kernel.fn = 0
        kernel.wfn = 0
        kernel.narrow = NO_NARROW
        compile_or_reuse(kernel)
    return len(old)

def kernel_key(kernel):
    rowmode = kernel_row_mode(kernel)
    conv = has_conv(kernel)
    parts = [str(kernel.ninputs),
             '0' if rowmode or conv else str(kernel.n)]
    if rowmode:
        parts.append('r%d' % row_tile(kernel))
    for i in range(len(kernel.nodes)):
        node = kernel.nodes[i]
        p = '%d' % node.p
        if node.opcode == CONV and node.p not in conv_descs.static:
            p = 'g%d' % conv_per_image(node.p)
        parts.append('%d:%d:%d:%s' % (node.opcode, node.a, node.b, p))
    for i in range(len(kernel.consts)):
        # The scalar is a literal in the code, so each distinct value is one
        # kernel. The values come from a small fixed set (gelu/layer-norm
        # coefficients), so the cache hits after warm-up.
        parts.append('k%s' % formatd(kernel.consts[i], 'r', 0))
    for i in range(len(kernel.outputs)):
        parts.append('o%d' % kernel.outputs[i])
    parts.append('d%d' % kernel.dtype)
    return ','.join(parts)

def compile_or_reuse(kernel):
    key = kernel_key(kernel)
    cached = cached_kernel(key)
    if cached:
        kernel.fn = cached.fn
        kernel.threads = cached.threads
        kernel.shared = cached.shared
        kernel.nextra = cached.nextra
        kernel.sumroot = cached.sumroot
        kernel.rowmode = cached.rowmode
        kernel.modes = cached.modes
        kernel.outmodes = cached.outmodes
        kernel.nouts = cached.nouts
        kernel.wfn = cached.wfn
        kernel.wthreads = cached.wthreads
        kernel.wshared = cached.wshared
        kernel.wnextra = cached.wnextra
        kernel.narrow = cached.narrow
        kernel.vsrc = cached.vsrc
        kernel.vroot = cached.vroot
        if kernel.wfn == 0 and kernel.fn != 0 and wants_wide(kernel):
            compile_wide(kernel, 'rtensor_k%d_w' % counter.n)
            counter.n += 1
            cached.wfn = kernel.wfn
            cached.wthreads = kernel.wthreads
            cached.wshared = kernel.wshared
            cached.wnextra = kernel.wnextra
        return kernel
    finish_kernel(kernel)
    cache_kernel(key, kernel)
    return kernel

def set_node(kernel, i, opcode, a, b, p):
    node = kernel.nodes[i]
    node.opcode = opcode
    node.a = a
    node.b = b
    node.p = p
def packed_out_modes(kernel):
    modes = out_modes(kernel)
    packed = 0
    for i in range(len(modes)):
        packed |= modes[i] << (2 * i)
    return packed

def packed_modes(kernel):
    modes = input_modes(kernel)
    packed = 0
    for i in range(len(modes)):
        packed |= modes[i] << (2 * i)
    return packed

def view_layout(kernel):
    nin = nvals(kernel)
    any_view = False
    for k in range(len(kernel.nodes)):
        if (kernel.nodes[k].opcode == GATHER and
                gather_kind(kernel.nodes[k].p) == GA_VIEW):
            any_view = True
    kernel.vsrc = 0
    if any_view and len(kernel.nodes) > 0:
        outer = [False] * (nin + len(kernel.nodes))
        stack = [nin + len(kernel.nodes) - 1]
        for k in range(len(kernel.outputs)):
            stack.append(kernel.outputs[k])
        while stack:
            v = stack.pop()
            if v < 0 or outer[v]:
                continue
            outer[v] = True
            if v < nin:
                continue
            node = kernel.nodes[v - nin]
            if not (node.opcode == GATHER and
                    gather_kind(node.p) == GA_VIEW):
                stack.append(node.a)
            stack.append(node.b)
        for k in range(kernel.ninputs):
            if not outer[k]:
                kernel.vsrc |= 1 << k
    kernel.vroot = -1
    if not any_view or len(kernel.nodes) == 0:
        return
    v = nin + len(kernel.nodes) - 1
    while v >= nin:
        node = kernel.nodes[v - nin]
        if node.opcode == GATHER and gather_kind(node.p) == GA_VIEW:
            kernel.vroot = -2 - view_of(node.p)
            return
        bc = bc_mode(node.opcode, node.p)
        if (not is_reduction(node.opcode) and
                (bc == BC_L_ROW or bc == BC_L_SCALAR or bc == BC_L_COL)):
            v = node.b
        else:
            v = node.a
    if v < kernel.ninputs:
        kernel.vroot = v
    else:
        kernel.vroot = 0

def finish_kernel(kernel):
    view_layout(kernel)
    n = len(kernel.nodes)
    kernel.sumroot = int(n > 0 and is_reduction(kernel.nodes[n - 1].opcode))
    if not kernel.sumroot and row_root(kernel, all_modes(kernel)):
        kernel.sumroot = 2
    kernel.rowmode = int(kernel_row_mode(kernel))
    if has_conv(kernel):
        kernel.sumroot = 0
        kernel.rowmode = 0
        kernel.n = 0
    kernel.modes = packed_modes(kernel)
    kernel.outmodes = packed_out_modes(kernel)
    kernel.nouts = 1 + len(kernel.outputs)
    kernel.fn = compile_gpu(kernel)
    if kernel.fn == 0 and config.debug_unfused and gpu_enabled():
        p = rffi.str2charp('no kernel for ' + kernel_key(kernel))
        rt_cuda_debug_text(p)
        rffi.free_charp(p)
    return kernel

def build_kernel(ninputs, opcodes, lefts, rights, params, dtype=F64):
    kernel = new_kernel(ninputs, len(opcodes), dtype)
    for i in range(len(opcodes)):
        set_node(kernel, i, opcodes[i], lefts[i], rights[i], params[i])
    return finish_kernel(kernel)

class Counter(object):
    n = 0
counter = Counter()

class MissCounter(object):
    # Incremented only when compile_ttir spawns the Triton subprocess, not on a
    # disk-cache hit. counter.n counts every kernel name request, so the
    # difference of their deltas is the cache hits.
    n = 0
miss_counter = MissCounter()

def get_cc():
    # Resolve the real cc before the first cache key is built. "auto" in a key
    # would reuse cubins across GPU architectures.
    if config.cc is not None:
        return config.cc
    if os.environ.get('RTENSOR_CPU') is not None:
        config.cc = 'cpu'
        return config.cc
    cc = _env('RTENSOR_CC', 'auto')
    if cc != 'auto':
        config.cc = cc
        return config.cc
    config.cc = _detect_cc()
    return config.cc

def _detect_cc():
    path = _env('TMPDIR', '/tmp') + '/rtensor-cc-%d' % os.getpid()
    cmd = '%s -P %s --cc %s' % (
        _env('RTENSOR_PYTHON', 'python3'), _here + '/triton_compile.py', path)
    if os.system(assert_str0(cmd)) != 0:
        return 'unknown'
    result = _read_or_empty(path).strip()
    try:
        os.unlink(assert_str0(path))
    except OSError:
        pass
    return result if result else 'unknown'

def _write(path, data):
    fd = os.open(assert_str0(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0644)
    os.write(fd, data)
    os.close(fd)

def _read(path):
    fd = os.open(assert_str0(path), os.O_RDONLY, 0)
    chunks = []
    while True:
        chunk = os.read(fd, 65536)
        if not chunk:
            break
        chunks.append(chunk)
    os.close(fd)
    return ''.join(chunks)
def _read_or_empty(path):
    try:
        return _read(path)
    except OSError:
        return ''

def _cache_base(src, warps, stages=3):
    if stages != 3:
        src = 's%d|%s' % (stages, src)
    key = md5('%s|%d|%s' % (get_cc(), warps, src)).hexdigest()
    return _env('TMPDIR', '/tmp') + '/rtensor-k-' + key

def _meta_cc_matches(meta):
    words = meta.strip().split(' ')
    return len(words) > 3 and words[3] == get_cc()

def _tune_path(key):
    return (_env('TMPDIR', '/tmp') + '/rtensor-tune-' +
            md5('%s|%s' % (get_cc(), key)).hexdigest())

def tuned_config(key):
    if _env('RTENSOR_KERNEL_CACHE', '1') == '0':
        return -1
    text = _read_or_empty(_tune_path(key)).strip()
    if not text.isdigit():
        return -1
    return int(text)

def remember_config(key, i):
    if _env('RTENSOR_KERNEL_CACHE', '1') == '0':
        return
    try:
        _write(_tune_path(key), '%d\n' % i)
    except OSError:
        pass

def _unnamed(src, name):
    i = src.find('@' + name + '(')
    if i < 0:
        return src
    return src[:i + 1] + src[i + 1 + len(name):]

def compile_ttir(src, name, warps, stages=3):
    # Each compile spawns a Python process and imports Triton (~0.6 s, ~50
    # kernels at startup). The output depends only on the TTIR, compute
    # capability and warp count, so cache files are keyed on those.
    # RTENSOR_KERNEL_CACHE=0 always recompiles.
    cached = _env('RTENSOR_KERNEL_CACHE', '1') != '0'
    meta = ''
    image = ''
    if cached:
        base = _cache_base(_unnamed(src, name), warps, stages)
        meta = _read_or_empty(base + '.meta')
        if meta and not _meta_cc_matches(meta):
            # Refuse a cubin built for a different architecture.
            meta = ''
        image = _read_or_empty(base + '.cubin')
        if not image:
            image = _read_or_empty(base + '.ptx')
    else:
        base = _env('TMPDIR', '/tmp') + '/' + name
    if not meta or not image:
        miss_counter.n += 1
        _write(base + '.ttir', src)
        cmd = '%s -P %s %s.ttir %s.ptx %s.meta %s %d %d' % (
            _env('RTENSOR_PYTHON', 'python3'), _here + '/triton_compile.py',
            base, base, base, get_cc(), warps, stages)
        if os.system(assert_str0(cmd)) != 0:
            return 0, 0, 0, 0
        meta = _read(base + '.meta').strip() + ' ' + name + '\n'
        _write(base + '.meta', meta)
        image = _read_or_empty(base + '.cubin')
        if not image:
            image = _read(base + '.ptx')
    words = meta.strip().split(' ')
    if len(words) > 4:
        name = words[4]
    p_ptx = rffi.str2charp(image)
    p_name = rffi.str2charp(name)
    fn = rt_cuda_load(p_ptx, p_name)
    rffi.free_charp(p_ptx)
    rffi.free_charp(p_name)
    return fn, int(words[0]), int(words[1]), int(words[2])

def compile_gpu(kernel):
    if not gpu_enabled():
        return 0
    try:
        return _compile_gpu(kernel)
    except (OSError, ValueError, IndexError):
        return 0

def _compile_gpu(kernel):
    name = 'rtensor_k%d' % counter.n
    counter.n += 1
    src = to_ttir(kernel, name)
    if not src:
        return 0
    warps = config.num_warps
    stages = 3
    if kernel.rowmode and kernel.cols > 0:
        warps = row_warps(row_tile(kernel))
    k = main_conv(kernel)
    if k >= 0:
        d = conv_descs.params[kernel.nodes[k].p]
        warps = d[CD_WARPS]
        stages = d[CD_STAGES]
    fn, threads, shared, nextra = compile_ttir(src, name, warps, stages)
    kernel.threads = threads
    kernel.shared = shared
    kernel.nextra = nextra
    compile_wide(kernel, name + '_w')
    compile_narrow(kernel, name)
    return fn

def wants_wide(kernel):
    return kernel.rowmode and (kernel.cols <= 0 or kernel.cols > config.block)

def compile_wide(kernel, name):
    """Compile the tiled variant launch_gpu uses for rows wider than kernel.fn's tile."""
    if kernel.wfn != 0 or not wants_wide(kernel):
        return
    src = to_ttir_row_tiled(kernel, name, all_modes(kernel))
    if not src:
        return
    try:
        fn, threads, shared, nextra = compile_ttir(src, name,
                                                   config.wide_warps)
    except (OSError, ValueError, IndexError):
        return
    kernel.wfn = fn
    kernel.wthreads = threads
    kernel.wshared = shared
    kernel.wnextra = nextra

def compile_narrow(kernel, name):
    if not kernel.rowmode or kernel.cols > NARROW_COLS[len(NARROW_COLS) - 1]:
        return
    modes = all_modes(kernel)
    narrow = lltype.malloc(SHAPEARRAY, 4 * len(NARROW_COLS))
    for i in range(len(narrow)):
        narrow[i] = 0
    for slot in range(len(NARROW_COLS)):
        bc = NARROW_COLS[slot]
        if kernel.cols > 0 and (kernel.cols > bc or
                                (slot > 0 and kernel.cols <= NARROW_COLS[slot - 1])):
            continue
        src = to_ttir_row_multi(kernel, name + '_n%d' % bc, modes, bc)
        if not src:
            continue
        try:
            fn, threads, shared, nextra = compile_ttir(src, name + '_n%d' % bc,
                                                       NARROW_WARPS)
        except (OSError, ValueError, IndexError):
            continue
        narrow[4 * slot] = fn
        narrow[4 * slot + 1] = threads
        narrow[4 * slot + 2] = shared
        narrow[4 * slot + 3] = nextra
    kernel.narrow = narrow

def needs_zero(kernel):
    if not kernel.sumroot:
        return 0
    root = kernel.nodes[len(kernel.nodes) - 1]
    if root.opcode != SUM:
        return 0
    if kernel.rowmode and root.p != AXIS_ALL:
        return 0
    return 1


class GatherKernel(object):
    def __init__(self, fn, threads, shared, nextra):
        self.fn = fn
        self.threads = threads
        self.shared = shared
        self.nextra = nextra
        self.block = config.flat


class GatherCache(object):
    def __init__(self):
        self.kernels = {}
gather_cache = GatherCache()

def gather_key(op, params, dtype, fill=0.0):
    parts = ['g%d' % op]
    for i in range(len(params)):
        parts.append(str(params[i]))
    parts.append('d%d' % dtype)
    if fill != 0.0:
        parts.append('f' + formatd(fill, 'r', 0))
    return ','.join(parts)


def gather_kernel(op, params, dtype, fill=0.0):
    key = gather_key(op, params, dtype, fill)
    k = gather_cache.kernels.get(key, None)
    if k is None:
        k = _gather_compile(op, params, dtype, fill)
        gather_cache.kernels[key] = k
    return k


def lookup_gather(op, params, dtype):
    """The compiled gather, or None. Never compiles, since it runs under oopspecs."""
    return gather_cache.kernels.get(gather_key(op, params, dtype), None)


def gather_params(p, n):
    """Kernel parameters of the gather a GATHER param describes, for n operand elements."""
    kind = gather_kind(p)
    dh = gather_dh(p)
    if kind == GA_ROTHALF:
        return [dh]
    heads = gather_heads(p)
    rows = n // (dh * heads) if dh * heads > 0 else 0
    return [rows, dh, heads]


@jit.dont_look_inside
def ensure_view(vid, dtype):
    gather_kernel(GA_STRIDED, view_params_list(vid), dtype)

@jit.dont_look_inside
def ensure_scatter(vid, dtype):
    gather_kernel(GA_SCATTER, view_params_list(vid), dtype)

@jit.dont_look_inside
def ensure_gather(p, n, dtype):
    """Compile the standalone gather a GATHER node falls back to, outside any oopspec."""
    return gather_kernel(gather_kind(p), gather_params(p, n), dtype).fn


def _gather_compile(op, params, dtype, fill=0.0):
    if not gpu_enabled():
        return GatherKernel(0, 0, 0, 0)
    try:
        return _gather_compile_gpu(op, params, dtype, fill)
    except (OSError, ValueError, IndexError):
        return GatherKernel(0, 0, 0, 0)

def _gather_compile_gpu(op, params, dtype, fill=0.0):
    name = 'rtensor_g%d' % counter.n
    counter.n += 1
    src = to_ttir_gather(op, params, name, dtype, fill)
    fn, threads, shared, nextra = compile_ttir(src, name, config.num_warps)
    k = GatherKernel(fn, threads, shared, nextra)
    if op == GA_SCATTER:
        k.block = scatter_block(params)
    return k


ATTN_PRECS = ['ieee', 'tf32', 'tf32x3', 'split']
ATTN_ARGS = 26


def attn_tiles(d, masked, prec):
    """Query rows per program (32 key rows at a time) for head width d.

    The most whose two-stage pipeline fits an sm_86 SM's shared memory, or 0
    for a width that is not a power of two from 16 to 128."""
    if d not in (16, 32, 64, 128):
        return 0
    if d == 128:
        return 64 if prec == 1 else 32
    return 64 if masked else 128


def attn_kernel(d, causal, masked, scale, prec, align, bm):
    key = 'a%d,%d,%d,%d,%d,%d,%s' % (d, causal, masked, prec, align, bm,
                                     formatd(scale, 'r', 0))
    k = gather_cache.kernels.get(key, None)
    if k is None:
        k = GatherKernel(0, 0, 0, 0)
        if gpu_enabled() and bm > 0:
            name = 'rtensor_a%d' % counter.n
            counter.n += 1
            src = to_ttir_attn(name, d, d, bm, 32, scale, causal, masked,
                               ATTN_PRECS[prec], align)
            try:
                fn, threads, shared, nextra = compile_ttir(src, name, 4, 2)
                k = GatherKernel(fn, threads, shared, nextra)
            except (OSError, ValueError, IndexError):
                pass
        gather_cache.kernels[key] = k
    return k
