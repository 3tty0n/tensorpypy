from rpython.rlib import jit
from rpython.rlib.rarithmetic import intmask
from rpython.rlib.rfloat import INFINITY
from rpython.rlib.rfloat import NAN
from rpython.rlib.rfloat import erf
from rpython.rtyper.lltypesystem import lltype
from rpython.rtyper.lltypesystem import rffi
import math
from rpython.metatensor.core import (ADD, ARITY, B_GE, B_GT, B_KEEP_NZ, B_KEEP_Z, B_LE, B_LT, B_MAX, B_MIN, B_NE, B_POW, BINARY, U_ABS, U_COS, U_ERF, U_FLOOR, U_GELU, U_LOG, U_SIGMOID, U_SILU, U_SIN, U_TANH, UNARY, bc_mode, binary_fn, BC_L_COL, BC_L_ROW, BC_L_SCALAR, BC_R_COL, BC_R_ROW, BC_R_SCALAR, CONV, DIV, EQMASK, EXP, GATHER, MAXR, MUL, NDTYPES, NEG_INF, NULLTENSOR, RELU, SHAPEARRAY, SUB, SUM, TENSORARRAY, _shape1, cols, config, gather_changes_shape, gather_shape, has_conv, nbytes, new_tensor, nvals, view_numel, view_shape)
from rpython.metatensor.device import (SIGNEDARRAY, collect_if_needed, dev, device_tensor, host, note_cpu_fallback, note_unfused, prof_begin, prof_end, profile_report, device_alloc, rt_cuda_free, rt_cuda_debug_text, rt_cuda_launch, rt_cuda_reset, rt_cuda_warn_arity, rt_cuda_warn_cpu, rt_cuda_warn_unfused)
from rpython.metatensor.kernels import (has_gather, kernel_key, needs_zero, row_tile, single_kernel)
from rpython.metatensor.ttir import NARROW_COLS, narrow_rows
from rpython.metatensor.devops import (_make_ones, ones, adaptive_avg_pool2d, attention, avg_pool2d, cat, conv_launch_gpu, conv_op, depthwise_conv2d, device_zeros, place, im2col2, im2col_t, pad, pool_out, strided, take, col2chw, gather_op, head_merge, head_split, im2col, im2col_nhwc, maxpool2, maxpool2_nhwc, rot_half, rowgather, scalar, scalar_of, cached_scalar, scalars, tensor_assign, tensor_bmm, tensor_matmul, tensor_write_rows, iota_of)
from rpython.metatensor.ttir import NARROW_COLS, flat_block, narrow_rows

def eval_op(opcode, a, b, p):
    if opcode == GATHER:
        return gather_op(a, p, b)
    if opcode == CONV:
        return conv_op(a, b, p)
    kernel = single_kernel(opcode, p, a.dtype)
    if kernel.fn != 0:
        inputs = [a]
        if ARITY[opcode] == 2:
            inputs.append(b)
        r = launch_gpu(kernel, inputs)
        if r:
            return r
        rt_cuda_warn_cpu(kernel.fn)
    return eval_op_cpu(opcode, a, b, p)

def _exp(v):
    if v > 709.0:
        return INFINITY
    return math.exp(v)

def _sqrt(v):
    if v < 0.0:
        return NAN
    return math.sqrt(v)

def _div(x, y):
    if y == 0.0:
        if x == 0.0:
            return NAN
        return INFINITY if x > 0.0 else NEG_INF
    return x / y

def _log(v):
    if v < 0.0:
        return NAN
    if v == 0.0:
        return NEG_INF
    return math.log(v)

_RSQRT2 = 1.0 / math.sqrt(2.0)

def _unary(fn, v):
    if fn == U_TANH:
        return math.tanh(v)
    if fn == U_SIGMOID:
        return 1.0 / (1.0 + _exp(-v))
    if fn == U_LOG:
        return _log(v)
    if fn == U_ABS:
        return abs(v)
    if fn == U_SIN or fn == U_COS:
        if v - v != 0.0:
            return NAN
        return math.sin(v) if fn == U_SIN else math.cos(v)
    if fn == U_ERF:
        return erf(v)
    if fn == U_FLOOR:
        return math.floor(v)
    if fn == U_GELU:
        return 0.5 * v * (1.0 + erf(v * _RSQRT2))
    if fn == U_SILU:
        return v / (1.0 + _exp(-v))
    return -v

def _pow(x, y):
    try:
        return math.pow(x, y)
    except ValueError:
        return INFINITY if x == 0.0 else NAN
    except OverflowError:
        if x < 0.0 and math.fmod(y, 2.0) != 0.0:
            return NEG_INF
        return INFINITY

def _binary(fn, x, y):
    if fn == B_MAX or fn == B_MIN:
        if x != x or y != y:
            return NAN
        if fn == B_MAX:
            return x if x > y else y
        return x if x < y else y
    if fn == B_POW:
        return _pow(x, y)
    if fn == B_LT:
        t = x < y
    elif fn == B_LE:
        t = x <= y
    elif fn == B_GT:
        t = x > y
    elif fn == B_GE:
        t = x >= y
    elif fn == B_NE:
        t = x != y
    elif fn == B_KEEP_NZ:
        return y if x != 0.0 else 0.0
    elif fn == B_KEEP_Z:
        return y if x == 0.0 else 0.0
    else:
        t = x == y
    return 1.0 if t else 0.0

def eval_op_cpu(opcode, a, b, p):
    note_cpu_fallback()
    if opcode == SUM or opcode == MAXR:
        return reduce_cpu(opcode, a, p)
    ha = host(a)
    if ARITY[opcode] == 1:
        n = a.size
        r = new_tensor(n, a.shape, a.dtype)
        hr = r.host
        for i in range(n):
            v = ha[i]
            if opcode == RELU:
                hr[i] = v if v > 0.0 else 0.0
            elif opcode == EXP:
                hr[i] = _exp(v)
            elif opcode == UNARY:
                hr[i] = _unary(p, v)
            else:
                hr[i] = _sqrt(v)
        return r
    fn = binary_fn(p)
    p = bc_mode(opcode, p)
    big = a
    if p == BC_L_ROW or p == BC_L_SCALAR or p == BC_L_COL:
        big = b
    n = big.size
    c = cols(big)
    hb = host(b)
    r = new_tensor(n, big.shape, big.dtype)
    hr = r.host
    for i in range(n):
        ia = i
        ib = i
        if p == BC_R_ROW:
            ib = i % c
        elif p == BC_R_SCALAR:
            ib = 0
        elif p == BC_R_COL:
            ib = i // c
        elif p == BC_L_ROW:
            ia = i % c
        elif p == BC_L_SCALAR:
            ia = 0
        elif p == BC_L_COL:
            ia = i // c
        assert ia >= 0
        assert ib >= 0
        if opcode == ADD:
            hr[i] = ha[ia] + hb[ib]
        elif opcode == MUL:
            hr[i] = ha[ia] * hb[ib]
        elif opcode == SUB:
            hr[i] = ha[ia] - hb[ib]
        elif opcode == DIV:
            hr[i] = _div(ha[ia], hb[ib])
        elif opcode == EQMASK:
            hr[i] = 1.0 if ha[ia] == hb[ib] else 0.0
        elif opcode == BINARY:
            hr[i] = _binary(fn, ha[ia], hb[ib])
        else:
            hr[i] = hb[ib] if ha[ia] > 0.0 else 0.0
    return r

def reduce_cpu(opcode, a, axis):
    ha = host(a)
    n = a.size
    c = cols(a)
    if axis == 0:
        m = c
    elif axis == 1:
        m = n // c
    else:
        m = 1
    if m <= 0:
        m = 1
    r = new_tensor(m, lltype.nullptr(SHAPEARRAY), a.dtype)
    hr = r.host
    for i in range(m):
        hr[i] = 0.0 if opcode == SUM else NEG_INF
    for i in range(n):
        if axis == 0:
            k = i % c
        elif axis == 1:
            k = i // c
        else:
            k = 0
        if k >= m:
            k = m - 1
        assert k >= 0
        if opcode == SUM:
            hr[k] += ha[i]
        elif ha[i] > hr[k]:
            hr[k] = ha[i]
    return r

def extra_size(kernel, k, n, c):
    mode = (kernel.outmodes >> (2 * k)) & 3
    if mode == 1:
        m = c
    elif mode == 2:
        m = 1
    elif mode == 3:
        m = n // c if c > 0 else n
    else:
        m = n
    if m <= 0:
        m = 1
    return m

def arity_mismatch(kernel):
    """True when kernel.fn was compiled for a different number of outputs.

    Output pointers are packed ahead of n and cols, so a mismatch shifts cols
    into a device pointer. Refuse the launch and fall back to the CPU."""
    if kernel.nouts == 1 + len(kernel.outputs):
        return False
    rt_cuda_warn_arity(kernel.nouts, 1 + len(kernel.outputs))
    return True

@jit.unroll_safe
def modes_fit(kernel, inputs, n, c):
    packed = kernel.modes
    for k in range(len(inputs)):
        if (kernel.vsrc >> k) & 1:
            continue
        mode = (packed >> (2 * k)) & 3
        if mode == 1:
            if inputs[k].size != c:
                return False
            continue
        elif mode == 2:
            need = 1
        elif mode == 3:
            need = n // c if c > 0 else n
        else:
            need = n
        if inputs[k].size < need:
            return False
    return True

def result_shape(kernel, inputs, n, like, v=-1):
    """Shape of value v (default the root), as eager evaluation finds it."""
    nin = nvals(kernel)
    if v < 0:
        v = nin + len(kernel.nodes) - 1
    while v >= nin:
        node = kernel.nodes[v - nin]
        if node.opcode == GATHER and gather_changes_shape(node.p):
            return gather_shape(node.p, n, like)
        bc = bc_mode(node.opcode, node.p)
        if bc == BC_L_ROW or bc == BC_L_SCALAR or bc == BC_L_COL:
            v = node.b
        else:
            v = node.a
    if v < kernel.ninputs:
        return inputs[v].shape
    return like

def row_cols(kernel, inputs, n, c):
    """Row width a flat kernel broadcasts row operands over: c, unless a row operand has another length dividing n.

    Same-size inputs may have different row layouts ([B, N*C] vs [B*N, C]), and i % len is the same broadcast."""
    if kernel.rowmode:
        return c
    for k in range(len(inputs)):
        if (kernel.vsrc >> k) & 1 == 0 and (kernel.modes >> (2 * k)) & 3 == 1:
            m = inputs[k].size
            if m != c and m > 0 and n % m == 0:
                return m
    return c

def refusal(kernel, inputs):
    """Index into the reasons rt_cuda_note_unfused prints for why kernel did not launch."""
    if kernel.fn == 0:
        return 1
    big = 0
    for k in range(len(inputs)):
        if inputs[k].size > inputs[big].size:
            big = k
    n = inputs[big].size
    c = row_cols(kernel, inputs, n, cols(inputs[big]))
    if kernel.n != 0 and kernel.n != n:
        return 2
    if kernel.nouts != 1 + len(kernel.outputs):
        return 3
    for k in range(len(inputs)):
        if inputs[k].dtype != kernel.dtype:
            return 4
    if kernel.rowmode:
        if c <= 0 or n % c != 0:
            return 5
        if kernel.cols > 0 and c != kernel.cols:
            return 6
        if c > row_tile(kernel) and kernel.wfn == 0:
            return 7
    if not modes_fit(kernel, inputs, n, c):
        return 8
    return 9

@jit.dont_look_inside
def debug_unfused(kernel, inputs):
    if not config.debug_unfused:
        return
    parts = ['unfused', kernel_key(kernel), 'vroot=%d' % kernel.vroot,
             'vsrc=%d' % kernel.vsrc, 'modes=%d' % kernel.modes]
    for k in range(len(inputs)):
        parts.append('in%d:%d' % (k, inputs[k].size))
    p = rffi.str2charp(' '.join(parts))
    rt_cuda_debug_text(p)
    rffi.free_charp(p)

def narrow_slot(kernel, c):
    if not config.narrow:
        return -1
    narrow = kernel.narrow
    for slot in range(len(narrow) // 4):
        if c <= NARROW_COLS[slot] and narrow[4 * slot] != 0:
            return slot
    return -1

def launch_gpu(kernel, inputs):
    if has_conv(kernel):
        return conv_launch_gpu(kernel, inputs)
    nin = len(inputs)
    dt = kernel.dtype
    if arity_mismatch(kernel):
        return NULLTENSOR
    for k in range(nin):
        if inputs[k].dtype != dt:
            return NULLTENSOR
    big = 0
    for k in range(1, nin):
        if inputs[k].size > inputs[big].size:
            big = k
    n = inputs[big].size
    c = row_cols(kernel, inputs, n, cols(inputs[big]))
    shape = inputs[big].shape
    if kernel.vroot >= 0:
        big = kernel.vroot
        n = inputs[big].size
        c = cols(inputs[big])
        shape = inputs[big].shape
    elif kernel.vroot != -1:
        shape = view_shape(-2 - kernel.vroot)
        n = view_numel(-2 - kernel.vroot)
        c = shape[len(shape) - 1] if len(shape) > 1 else 1
    outlen = n
    elems = flat_block(kernel)
    fn = kernel.fn
    threads = kernel.threads
    shared = kernel.shared
    nextra = kernel.nextra
    if kernel.rowmode:
        if c <= 0 or n % c != 0:
            return NULLTENSOR
        if kernel.cols > 0 and c != kernel.cols:
            return NULLTENSOR
        if c > row_tile(kernel):
            if kernel.wfn == 0:
                return NULLTENSOR
            fn = kernel.wfn
            threads = kernel.wthreads
            shared = kernel.wshared
            nextra = kernel.wnextra
        elems = c
        slot = narrow_slot(kernel, c)
        if slot >= 0:
            fn = kernel.narrow[4 * slot]
            threads = kernel.narrow[4 * slot + 1]
            shared = kernel.narrow[4 * slot + 2]
            nextra = kernel.narrow[4 * slot + 3]
            elems = c * narrow_rows(slot)
    if not modes_fit(kernel, inputs, n, c):
        return NULLTENSOR
    like = shape
    if kernel.sumroot:
        axis = kernel.nodes[len(kernel.nodes) - 1].p
        if kernel.sumroot == 2:
            axis = 1
        if axis == 0:
            outlen = c
        elif axis == 1:
            outlen = n // c
        else:
            outlen = 1
        if outlen <= 0:
            outlen = 1
        shape = _shape1(outlen)
    if not kernel.sumroot and has_gather(kernel):
        shape = result_shape(kernel, inputs, n, shape)
    nout = 1 + len(kernel.outputs)
    collect_if_needed(nbytes(n, dt))
    dptrs = lltype.malloc(SIGNEDARRAY, nin, flavor='raw')
    outs = lltype.malloc(SIGNEDARRAY, nout, flavor='raw')
    ok = True
    for k in range(nin):
        dptrs[k] = dev(inputs[k])
        if dptrs[k] == 0:
            ok = False
    esizes = lltype.malloc(SIGNEDARRAY, nout, flavor='raw')
    esizes[0] = outlen
    outs[0] = device_alloc(nbytes(outlen, dt),
                            needs_zero(kernel)) if ok else 0
    if outs[0] == 0:
        ok = False
    for k in range(1, nout):
        esizes[k] = extra_size(kernel, k - 1, n, c)
        outs[k] = device_alloc(nbytes(esizes[k], dt), 0) if ok else 0
        if outs[k] == 0:
            ok = False
    if ok:
        ok = rffi.cast(lltype.Signed, rt_cuda_launch(
            fn, dptrs, rffi.cast(rffi.INT, nin), n,
            outs, rffi.cast(rffi.INT, nout),
            rffi.cast(rffi.INT, threads), elems,
            rffi.cast(rffi.INT, shared),
            rffi.cast(rffi.INT, nextra), c)) != 0
    result = NULLTENSOR
    if not ok:
        # Allocation or launch failed part way: free the buffers obtained, nothing owns them.
        for k in range(nout):
            if outs[k] != 0:
                rt_cuda_free(outs[k], nbytes(esizes[k], dt))
    if ok:
        result = device_tensor(outlen, outs[0], shape, dt)
        if nout > 1:
            result.extra = lltype.malloc(TENSORARRAY, nout - 1)
            gathered = has_gather(kernel)
            for k in range(1, nout):
                eshape = like
                if esizes[k] != n:
                    eshape = _shape1(esizes[k])
                elif gathered:
                    eshape = result_shape(kernel, inputs, n, eshape,
                                          kernel.outputs[k - 1])
                result.extra[k - 1] = device_tensor(esizes[k], outs[k],
                                                    eshape, dt)
    lltype.free(esizes, flavor='raw')
    lltype.free(dptrs, flavor='raw')
    lltype.free(outs, flavor='raw')
    return result

def launch(kernel, t0, t1, t2, t3=NULLTENSOR, t4=NULLTENSOR, t5=NULLTENSOR,
           t6=NULLTENSOR, t7=NULLTENSOR, t8=NULLTENSOR, t9=NULLTENSOR,
           t10=NULLTENSOR, t11=NULLTENSOR, t12=NULLTENSOR, t13=NULLTENSOR,
           t14=NULLTENSOR, t15=NULLTENSOR):
    values = [t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13,
              t14, t15]
    del values[max(kernel.ninputs, 1):]
    nmax = 0
    for k in range(len(values)):
        if values[k].size > nmax:
            nmax = values[k].size
    if kernel.fn != 0 and (kernel.n == 0 or kernel.n == nmax):
        t0 = prof_begin()
        r = launch_gpu(kernel, values)
        if r:
            prof_end(intmask(0), intmask(kernel.fn % 1000000000), t0)
            return r
    note_unfused(kernel.fn, refusal(kernel, values))
    debug_unfused(kernel, values)
    if kernel.fn != 0:
        rt_cuda_warn_unfused(kernel.fn)
    t0 = prof_begin()
    for j in range(len(kernel.consts)):
        values.append(cached_scalar(kernel.consts[j], kernel.dtype))
    nodes = kernel.nodes
    for i in range(len(nodes)):
        node = nodes[i]
        opcode = node.opcode
        assert opcode >= 0
        right = values[node.b] if node.b >= 0 else NULLTENSOR
        values.append(eval_op(opcode, values[node.a], right, node.p))
    result = values[len(values) - 1]
    prof_end(intmask(1), intmask(len(kernel.nodes)), t0)
    nout = len(kernel.outputs)
    if nout > 0:
        result.extra = lltype.malloc(TENSORARRAY, nout)
        for k in range(nout):
            result.extra[k] = values[kernel.outputs[k]]
    return result

def reset_device():
    profile_report()
    ones.cache.clear()
    for i in range(NDTYPES):
        scalars.tensors[i].clear()
        ones.n[i] = -1
        ones.t[i] = NULLTENSOR
        ones.one[i] = NULLTENSOR
    rt_cuda_reset()
