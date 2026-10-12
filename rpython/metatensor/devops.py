from rpython.rlib import jit
from rpython.rlib.objectmodel import specialize
from rpython.rlib.rarithmetic import intmask
from rpython.rtyper.lltypesystem import lltype
from rpython.rtyper.lltypesystem import rffi
from rpython.metatensor.core import (NDTYPES, F16, F32, CD_G, CD_O, CD_ROWS, CONV, KERNEL, TENSORARRAY, conv_desc, conv_descs, main_conv, conv_geometry, conv_programs, GA_IM2COL2, GA_IM2COL_T, GA_POOL, GA_SCATTER, GA_STRIDED, GA_TAKE, POOL_ADAPTIVE, POOL_AVG, POOL_DEPTHWISE, conv_out, GA_ROWS, GA_COL2CHW, GA_HEADMERGE, GA_HEADSPLIT, GA_IM2COL, GA_IM2COL_NHWC, GA_ROTHALF, GA_MAXPOOL, GA_MAXPOOL_NHWC, HOSTARRAY, NEG_INF, NULLTENSOR, SHAPEARRAY, _shape2, cols, config, gather_kind, gather_shape, nbytes, new_tensor, note_dtype, policy, view_clamp, TENSOR, view_is_identity, view_numel, GA_VIEW, view_of, view_offset, view_params_list, view_shape)
from rpython.metatensor.device import (SIGNEDARRAY, attach_buffer, upload, collect_if_needed, dev, device_tensor, download, gpu_enabled, host, note_cpu_fallback, prof_begin, prof_end, device_alloc, rt_cuda_bmm2, rt_cuda_copy, rt_cuda_free, rt_cuda_launch, rt_cuda_launch_grid, rt_cuda_matmul, rt_cuda_matmul_bias, rt_cuda_now, rt_cuda_sync)
from rpython.metatensor.kernels import (ATTN_ARGS, attn_kernel, attn_tiles, compile_or_reuse, gather_kernel, gather_params, lookup_gather, new_kernel, remember_config, set_node, tuned_config)
from rpython.rtyper.rclass import OBJECTPTR

def matmul_cpu(a, b, rows, cols, inner, ta, tb):
    note_cpu_fallback()
    ha = host(a)
    hb = host(b)
    shape = lltype.malloc(SHAPEARRAY, 2)
    shape[0] = rows
    shape[1] = cols
    r = new_tensor(rows * cols, shape, a.dtype)
    hr = r.host
    for i in range(rows):
        for j in range(cols):
            acc = 0.0
            for k in range(inner):
                if ta:
                    va = ha[k * rows + i]
                else:
                    va = ha[i * inner + k]
                if tb:
                    vb = hb[j * inner + k]
                else:
                    vb = hb[k * cols + j]
                acc += va * vb
            hr[i * cols + j] = acc
    return r

@jit.dont_look_inside
def _tensor_matmul_impl(a, b, rows, cols, inner, ta, tb, tf32, bias):
    if gpu_enabled() and a.dtype == b.dtype:
        dt = a.dtype
        dptr_a = dev(a)
        dptr_b = dev(b)
        dptr_c = 0
        if bias:
            dptr_c = dev(bias)
        if dptr_a != 0 and dptr_b != 0 and (dptr_c != 0 or not bias):
            n = rows * cols
            nb = nbytes(n, dt)
            collect_if_needed(nb)
            outptr = device_alloc(nb, 0)
            if outptr != 0:
                if bias:
                    ok = rffi.cast(lltype.Signed, rt_cuda_matmul_bias(
                        dptr_a, dptr_b, outptr, dptr_c, rows, inner, cols,
                        ta, tb, tf32)) != 0
                else:
                    ok = rffi.cast(lltype.Signed, rt_cuda_matmul(
                        dptr_a, dptr_b, outptr, rows, inner, cols, ta, tb,
                        dt, tf32)) != 0
                if ok:
                    shape = lltype.malloc(SHAPEARRAY, 2)
                    shape[0] = rows
                    shape[1] = cols
                    return device_tensor(n, outptr, shape, dt)
                rt_cuda_free(outptr, nb)
    r = matmul_cpu(a, b, rows, cols, inner, ta, tb)
    if bias:
        hb = host(bias)
        for i in range(rows):
            for j in range(cols):
                r.host[i * cols + j] += hb[j]
    return r

@jit.dont_look_inside
def tensor_matmul(a, b, rows, cols, inner, ta, tb, tf32=0, bias=NULLTENSOR):
    t0 = prof_begin()
    r = _tensor_matmul_impl(a, b, rows, cols, inner, ta, tb, tf32, bias)
    prof_end(intmask(2), intmask(rows * 1000000 + inner * 1000 + cols), t0)
    return r


def _lds(rows, cols, inner, ta, tb, lda, ldb, ldc, sa, sb, sc):
    if lda <= 0:
        lda = rows if ta else inner
    if ldb <= 0:
        ldb = inner if tb else cols
    if ldc <= 0:
        ldc = cols
    if sa <= 0:
        sa = rows * inner
    if sb <= 0:
        sb = inner * cols
    if sc <= 0:
        sc = rows * cols
    return lda, ldb, ldc, sa, sb, sc

def bmm_cpu(a, b, batch, rows, cols, inner, ta, tb, lda, ldb, ldc,
            sa, sb, sc, outn, outrows, oa, ob, oc, nb2=1, sa2=0, sb2=0,
            sc2=0):
    note_cpu_fallback()
    lda, ldb, ldc, sa, sb, sc = _lds(rows, cols, inner, ta, tb, lda, ldb,
                                     ldc, sa, sb, sc)
    ha = host(a)
    hb = host(b)
    r = new_tensor(outn, _shape2(outrows, outn // outrows), a.dtype)
    hr = r.host
    for i in range(outn):
        hr[i] = 0.0
    for g in range(nb2):
        ga = oa + g * sa2
        gb = ob + g * sb2
        gc = oc + g * sc2
        for t in range(batch):
            for i in range(rows):
                for j in range(cols):
                    acc = 0.0
                    for k in range(inner):
                        if ta:
                            ia = ga + t * sa + k * lda + i
                        else:
                            ia = ga + t * sa + i * lda + k
                        if tb:
                            ib = gb + t * sb + j * ldb + k
                        else:
                            ib = gb + t * sb + k * ldb + j
                        assert ia >= 0
                        assert ib >= 0
                        acc += ha[ia] * hb[ib]
                    ic = gc + t * sc + i * ldc + j
                    assert ic >= 0
                    hr[ic] = acc
    return r

@jit.dont_look_inside
def _tensor_bmm_impl(a, b, batch, rows, cols, inner, ta, tb, lda, ldb, ldc,
                     sa, sb, sc, outn, outrows, oa, ob, oc, zc, nb2=1,
                     sa2=0, sb2=0, sc2=0, tf32=0):
    if gpu_enabled() and a.dtype == b.dtype:
        dt = a.dtype
        dptr_a = dev(a)
        dptr_b = dev(b)
        if dptr_a != 0 and dptr_b != 0:
            nb = nbytes(outn, dt)
            collect_if_needed(nb)
            outptr = device_alloc(nb, zc)
            if outptr != 0:
                isz = nbytes(1, dt)
                # The outer batch level is a second stride per operand, which a
                # strided-batched GEMM cannot express. rt_cuda_bmm2 issues the
                # two-level batch as one gemmBatched over a device pointer array.
                ok = rffi.cast(lltype.Signed, rt_cuda_bmm2(
                    dptr_a + oa * isz, dptr_b + ob * isz, outptr + oc * isz,
                    nb2, sa2, sb2, sc2, batch, rows, inner, cols,
                    ta, tb, dt, lda, ldb, ldc, sa, sb, sc, tf32)) != 0
                if ok:
                    return device_tensor(outn, outptr,
                                         _shape2(outrows, outn // outrows),
                                         dt)
                rt_cuda_free(outptr, nb)
    return bmm_cpu(a, b, batch, rows, cols, inner, ta, tb, lda, ldb, ldc,
                   sa, sb, sc, outn, outrows, oa, ob, oc, nb2, sa2, sb2, sc2)

@jit.dont_look_inside
def tensor_bmm(a, b, batch, rows, cols, inner, ta, tb, lda=0, ldb=0, ldc=0,
               sa=0, sb=0, sc=0, outn=0, outrows=0, oa=0, ob=0, oc=0, zc=0,
               nb2=1, sa2=0, sb2=0, sc2=0, tf32=0):
    """nb2 is an outer batch level: nb2 groups of `batch` matrices, the groups
    sa2/sb2/sc2 elements apart.  Attention over a folded batch of sequences
    needs it: the offset b*seq*lda + head*dh over (sequence, head) is not an
    arithmetic progression in one index."""
    if outn <= 0:
        outn = nb2 * batch * rows * cols
    if outrows <= 0:
        outrows = nb2 * batch * rows
    t0 = prof_begin()
    r = _tensor_bmm_impl(a, b, batch, rows, cols, inner, ta, tb, lda, ldb,
                         ldc, sa, sb, sc, outn, outrows, oa, ob, oc, zc,
                         nb2, sa2, sb2, sc2, tf32)
    prof_end(intmask(3), intmask(rows * 1000000 + inner * 1000 + cols), t0)
    return r


@jit.dont_look_inside
def attention(q, k, v, m, p, scale):
    """softmax(q k^T * scale + m) v in one kernel, or NULLTENSOR.  p holds
    b, h, hk, sq, sk, d, then offset and (batch, head, row) strides of q, k,
    v and m, then causal, bshd, prec, and the number of batches the mask
    repeats over (0: its batch is the queries').  The result is [b*sq, h*d]
    rows when bshd is set, [b*h*sq, d] otherwise."""
    if not gpu_enabled() or q.dtype != F32:
        return NULLTENSOR
    t0 = prof_begin()
    b, h, hk, sq, sk, d = p[0], p[1], p[2], p[3], p[4], p[5]
    masked = 1 if m else 0
    srcs = [q, k, v, m if m else q]
    offs = [p[6], p[10], p[14], p[18]]
    ptrs = _dptrs(srcs, offs)
    if not ptrs:
        return NULLTENSOR
    qkv = True
    mk = masked == 1
    for i in range(4):
        ok = ptrs[i] % 16 == 0
        for j in range(3):
            ok = ok and p[7 + 4 * i + j] % 16 == 0
        if i < 3:
            qkv = qkv and ok
        else:
            mk = mk and ok
    align = (1 if qkv else 0) | (4 if mk else 0)
    if masked and p[22] == 1:
        align |= 2
    bm = attn_tiles(d, masked, p[25])
    kern = attn_kernel(d, p[23], masked, scale, p[25], align, bm)
    if kern.fn == 0:
        return NULLTENSOR
    outn = b * h * sq * d
    nb = nbytes(outn, F32)
    collect_if_needed(nb)
    out = device_alloc(nb, 0)
    if out == 0:
        return NULLTENSOR
    args = lltype.malloc(SIGNEDARRAY, ATTN_ARGS, flavor='raw')
    for i in range(4):
        args[i] = ptrs[i]
    args[4] = out
    args[5] = sq
    args[6] = sk
    args[7] = h
    args[8] = h // hk
    for i in range(3):
        args[9 + i] = p[7 + i]
        args[12 + i] = p[11 + i]
        args[15 + i] = p[15 + i]
    if p[24]:
        args[18], args[19], args[20] = sq * h * d, d, h * d
    else:
        args[18], args[19], args[20] = h * sq * d, sq * d, d
    for i in range(4):
        args[21 + i] = p[19 + i]
    args[25] = p[26] if p[26] > 0 else b
    ok = rffi.cast(lltype.Signed, rt_cuda_launch_grid(
        kern.fn, args, rffi.cast(rffi.INT, ATTN_ARGS), (sq + bm - 1) // bm,
        b * h, rffi.cast(rffi.INT, kern.threads),
        rffi.cast(rffi.INT, kern.shared),
        rffi.cast(rffi.INT, kern.nextra))) != 0
    lltype.free(args, flavor='raw')
    if not ok:
        rt_cuda_free(out, nb)
        return NULLTENSOR
    shape = _shape2(b * sq, h * d) if p[24] else _shape2(b * h * sq, d)
    r = device_tensor(outn, out, shape, F32)
    prof_end(intmask(15), intmask(d), t0)
    return r


def gather_gpu(op, params, x, outn, shape):
    return gather_launch(gather_kernel(op, params, x.dtype), x, outn, shape)


@jit.dont_look_inside
def gather_launch(kernel, x, outn, shape):
    dt = x.dtype
    if kernel is None or kernel.fn == 0:
        return NULLTENSOR
    dptr = dev(x)
    if dptr == 0:
        return NULLTENSOR
    collect_if_needed(nbytes(outn, dt))
    ins = lltype.malloc(SIGNEDARRAY, 1, flavor='raw')
    outs = lltype.malloc(SIGNEDARRAY, 1, flavor='raw')
    ins[0] = dptr
    outs[0] = device_alloc(nbytes(outn, dt), 0)
    ok = outs[0] != 0
    if ok:
        ok = rffi.cast(lltype.Signed, rt_cuda_launch(
            kernel.fn, ins, rffi.cast(rffi.INT, 1), outn, outs,
            rffi.cast(rffi.INT, 1), rffi.cast(rffi.INT, kernel.threads),
            config.flat, rffi.cast(rffi.INT, kernel.shared),
            rffi.cast(rffi.INT, kernel.nextra), 0)) != 0
    result = NULLTENSOR
    if ok:
        result = device_tensor(outn, outs[0], shape, dt)
    lltype.free(ins, flavor='raw')
    lltype.free(outs, flavor='raw')
    return result


@jit.dont_look_inside
def gather_op(x, p, offset=NULLTENSOR):
    """One GATHER node on its own: the precompiled gather if ensure_gather
    has built it, the host loop otherwise.  Reached from under the oopspecs,
    so it only looks kernels up."""
    kind = gather_kind(p)
    if kind == GA_VIEW:
        dyn = 0
        if offset and offset != x:
            dyn = int(host(offset)[0])
        return view_eval(x, view_of(p), dyn)
    n = x.size
    shape = gather_shape(p, n, _shape2(n // cols(x), cols(x)))
    params = gather_params(p, n)
    if gpu_enabled():
        r = gather_launch(lookup_gather(kind, params, x.dtype), x, n, shape)
        if r:
            return r
    if kind == GA_ROTHALF:
        return rot_half_cpu(x, params[0])
    if kind == GA_HEADSPLIT:
        return head_split_cpu(x, params[0], params[1], params[2])
    return head_merge_cpu(x, params[0], params[1], params[2])


def view_eval(x, vid, dyn=0):
    params = view_params_list(vid)
    off = view_offset(vid) + dyn
    src = view_shape(vid)
    shape = lltype.malloc(SHAPEARRAY, len(src))
    for i in range(len(src)):
        shape[i] = src[i]
    if dyn == 0 and view_is_identity(vid) and view_numel(vid) == x.size:
        return _alias(x, shape)
    if view_clamp(vid) > 0:
        return _clamped_cpu(x, params, off, view_clamp(vid), shape)
    if gpu_enabled():
        k = lookup_gather(GA_STRIDED, params, x.dtype)
        if k is not None:
            r = _launch(k, _dptrs([x], [off]), _numel(params), shape, x.dtype)
            if r:
                return r
    return strided_cpu([x], [off], params, 0.0, shape)


def _clamped_cpu(x, params, off, extent, shape):
    rank = params[1]
    n = _numel(params)
    hx = host(x)
    r = new_tensor(n, shape, x.dtype)
    for i in range(n):
        rem = i
        addr = off
        ok = True
        for j in range(rank - 1, -1, -1):
            d = params[2 + j]
            c = rem % d
            rem = rem // d
            if c < params[2 + 2 * rank + j] or c >= params[2 + 3 * rank + j]:
                ok = False
            addr += c * params[2 + rank + j]
        if addr < 0:
            addr = 0
        elif addr >= extent:
            addr = extent - 1
        r.host[i] = hx[addr] if ok else 0.0
    return r


def _alias(x, shape):
    if x.dptr == 0 and x.host:
        dev(x)
    r = lltype.malloc(TENSOR)
    r.size = x.size
    r.shape = shape
    r.dptr = x.dptr
    r.host = x.host
    r.extra = lltype.nullptr(TENSORARRAY)
    r.dtype = x.dtype
    r.buf = x.buf
    r.lazy = lltype.nullptr(OBJECTPTR.TO)
    return r


def rowgather_gpu(table, idx, rows, cols):
    dt = table.dtype
    kernel = gather_kernel(GA_ROWS, [cols], dt)
    if kernel.fn == 0:
        return NULLTENSOR
    tptr = dev(table)
    iptr = dev(idx)
    if tptr == 0 or iptr == 0:
        return NULLTENSOR
    outn = rows * cols
    collect_if_needed(nbytes(outn, dt))
    ins = lltype.malloc(SIGNEDARRAY, 2, flavor='raw')
    outs = lltype.malloc(SIGNEDARRAY, 1, flavor='raw')
    ins[0] = tptr
    ins[1] = iptr
    outs[0] = device_alloc(nbytes(outn, dt), 0)
    ok = outs[0] != 0
    if ok:
        ok = rffi.cast(lltype.Signed, rt_cuda_launch(
            kernel.fn, ins, rffi.cast(rffi.INT, 2), outn, outs,
            rffi.cast(rffi.INT, 1), rffi.cast(rffi.INT, kernel.threads),
            config.flat, rffi.cast(rffi.INT, kernel.shared),
            rffi.cast(rffi.INT, kernel.nextra), 0)) != 0
    result = NULLTENSOR
    if ok:
        result = device_tensor(outn, outs[0], _shape2(rows, cols), dt)
    lltype.free(ins, flavor='raw')
    lltype.free(outs, flavor='raw')
    return result


def rowgather_cpu(table, idx, rows, cols):
    note_cpu_fallback()
    ht = host(table)
    hi = host(idx)
    r = new_tensor(rows * cols, _shape2(rows, cols), table.dtype)
    hr = r.host
    for i in range(rows * cols):
        row = int(hi[i // cols])
        assert row >= 0
        hr[i] = ht[row * cols + i % cols]
    return r


@jit.dont_look_inside
def rowgather(table, idx, rows, cols):
    if gpu_enabled() and table.dtype != F16:
        r = rowgather_gpu(table, idx, rows, cols)
        if r:
            return r
    return rowgather_cpu(table, idx, rows, cols)


def im2col_cpu(x, n, c, h, w, k, pad, stride):
    note_cpu_fallback()
    hx = host(x)
    oh = (h + 2 * pad - k) // stride + 1
    ow = (w + 2 * pad - k) // stride + 1
    ohw = oh * ow
    chw = c * h * w
    kk = k * k
    rows = n * ohw
    cols = c * kk
    r = new_tensor(rows * cols, _shape2(rows, cols), x.dtype)
    hr = r.host
    for i in range(rows * cols):
        row = i // cols
        col = i % cols
        img = row // ohw
        pos = row % ohw
        ih = pos // ow * stride + col % kk // k - pad
        iw = pos % ow * stride + col % k - pad
        v = 0.0
        if ih >= 0 and ih < h and iw >= 0 and iw < w:
            idx = img * chw + col // kk * h * w + ih * w + iw
            assert idx >= 0
            v = hx[idx]
        hr[i] = v
    return r


@jit.dont_look_inside
def _im2col_impl(x, c, h, w, k, pad, stride):
    n = x.size // (c * h * w)
    oh = (h + 2 * pad - k) // stride + 1
    ow = (w + 2 * pad - k) // stride + 1
    rows = n * oh * ow
    cols = c * k * k
    if gpu_enabled():
        params = [n]
        params.append(c)
        params.append(h)
        params.append(w)
        params.append(k)
        params.append(pad)
        params.append(stride)
        r = gather_gpu(GA_IM2COL, params, x, rows * cols,
                       _shape2(rows, cols))
        if r:
            return r
    return im2col_cpu(x, n, c, h, w, k, pad, stride)

@jit.dont_look_inside
def im2col(x, c, h, w, k, pad, stride=1):
    t0 = prof_begin()
    r = _im2col_impl(x, c, h, w, k, pad, stride)
    prof_end(intmask(4), intmask(0), t0)
    return r


def im2col_nhwc_cpu(x, n, c, h, w, k, pad, stride):
    note_cpu_fallback()
    hx = host(x)
    oh = (h + 2 * pad - k) // stride + 1
    ow = (w + 2 * pad - k) // stride + 1
    ohw = oh * ow
    kk = k * k
    rows = n * ohw
    cols = kk * c
    r = new_tensor(rows * cols, _shape2(rows, cols), x.dtype)
    hr = r.host
    for i in range(rows * cols):
        row = i // cols
        col = i % cols
        ci = col % c
        rk = col // c
        img = row // ohw
        pos = row % ohw
        ih = pos // ow * stride + rk // k - pad
        iw = pos % ow * stride + rk % k - pad
        v = 0.0
        if ih >= 0 and ih < h and iw >= 0 and iw < w:
            idx = ((img * h + ih) * w + iw) * c + ci
            assert idx >= 0
            v = hx[idx]
        hr[i] = v
    return r


@jit.dont_look_inside
def _im2col_nhwc_impl(x, c, h, w, k, pad, stride):
    n = x.size // (c * h * w)
    oh = (h + 2 * pad - k) // stride + 1
    ow = (w + 2 * pad - k) // stride + 1
    rows = n * oh * ow
    cols = k * k * c
    if gpu_enabled():
        params = [n]
        params.append(c)
        params.append(h)
        params.append(w)
        params.append(k)
        params.append(pad)
        params.append(stride)
        r = gather_gpu(GA_IM2COL_NHWC, params, x, rows * cols,
                       _shape2(rows, cols))
        if r:
            return r
    return im2col_nhwc_cpu(x, n, c, h, w, k, pad, stride)

@jit.dont_look_inside
def im2col_nhwc(x, c, h, w, k, pad, stride=1):
    t0 = prof_begin()
    r = _im2col_nhwc_impl(x, c, h, w, k, pad, stride)
    prof_end(intmask(4), intmask(0), t0)
    return r


def maxpool2_nhwc_cpu(x, n, c, h, w, k, stride, pad):
    note_cpu_fallback()
    hx = host(x)
    oh = (h + 2 * pad - k) // stride + 1
    ow = (w + 2 * pad - k) // stride + 1
    ohw = oh * ow
    rows = n * ohw
    r = new_tensor(rows * c, _shape2(rows, c), x.dtype)
    hr = r.host
    for i in range(rows * c):
        ci = i % c
        row = i // c
        img = row // ohw
        pos = row % ohw
        ph = pos // ow * stride
        pw = pos % ow * stride
        v = NEG_INF
        for a in range(k):
            ih = ph + a - pad
            if ih < 0:
                ih = 0
            elif ih >= h:
                ih = h - 1
            for b in range(k):
                iw = pw + b - pad
                if iw < 0:
                    iw = 0
                elif iw >= w:
                    iw = w - 1
                idx = ((img * h + ih) * w + iw) * c + ci
                assert idx >= 0
                if hx[idx] > v:
                    v = hx[idx]
        hr[i] = v
    return r


@jit.dont_look_inside
def _maxpool2_nhwc_impl(x, c, h, w, k, stride, pad):
    n = x.size // (c * h * w)
    oh = (h + 2 * pad - k) // stride + 1
    ow = (w + 2 * pad - k) // stride + 1
    rows = n * oh * ow
    if gpu_enabled():
        params = [n]
        params.append(c)
        params.append(h)
        params.append(w)
        params.append(k)
        params.append(stride)
        params.append(pad)
        r = gather_gpu(GA_MAXPOOL_NHWC, params, x, rows * c,
                       _shape2(rows, c))
        if r:
            return r
    return maxpool2_nhwc_cpu(x, n, c, h, w, k, stride, pad)

@jit.dont_look_inside
def maxpool2_nhwc(x, c, h, w, k=2, stride=2, pad=0):
    t0 = prof_begin()
    r = _maxpool2_nhwc_impl(x, c, h, w, k, stride, pad)
    prof_end(intmask(6), intmask(0), t0)
    return r


def col2chw_cpu(y, n, hw, o):
    note_cpu_fallback()
    hy = host(y)
    r = new_tensor(n * o * hw, _shape2(n, o * hw), y.dtype)
    hr = r.host
    for i in range(n * o * hw):
        img = i // (o * hw)
        rem = i % (o * hw)
        idx = (img * hw + rem % hw) * o + rem // hw
        assert idx >= 0
        hr[i] = hy[idx]
    return r


@jit.dont_look_inside
def _col2chw_impl(y, n, hw, o):
    outn = n * o * hw
    if gpu_enabled():
        params = [n]
        params.append(hw)
        params.append(o)
        r = gather_gpu(GA_COL2CHW, params, y, outn, _shape2(n, o * hw))
        if r:
            return r
    return col2chw_cpu(y, n, hw, o)

@jit.dont_look_inside
def col2chw(y, n, hw, o):
    t0 = prof_begin()
    r = _col2chw_impl(y, n, hw, o)
    prof_end(intmask(5), intmask(0), t0)
    return r


def maxpool2_cpu(x, n, c, h, w, k, stride, pad):
    note_cpu_fallback()
    hx = host(x)
    oh = (h + 2 * pad - k) // stride + 1
    ow = (w + 2 * pad - k) // stride + 1
    outn = n * c * oh * ow
    r = new_tensor(outn, _shape2(n, c * oh * ow), x.dtype)
    hr = r.host
    for i in range(outn):
        pw = i % ow
        ph = i // ow % oh
        plane = i // (oh * ow)
        v = NEG_INF
        for a in range(k):
            ih = ph * stride + a - pad
            if ih < 0:
                ih = 0
            elif ih >= h:
                ih = h - 1
            for b in range(k):
                iw = pw * stride + b - pad
                if iw < 0:
                    iw = 0
                elif iw >= w:
                    iw = w - 1
                idx = (plane * h + ih) * w + iw
                assert idx >= 0
                if hx[idx] > v:
                    v = hx[idx]
        hr[i] = v
    return r


@jit.dont_look_inside
def _maxpool2_impl(x, c, h, w, k, stride, pad):
    n = x.size // (c * h * w)
    oh = (h + 2 * pad - k) // stride + 1
    ow = (w + 2 * pad - k) // stride + 1
    outn = n * c * oh * ow
    if gpu_enabled():
        params = [n]
        params.append(c)
        params.append(h)
        params.append(w)
        params.append(k)
        params.append(stride)
        params.append(pad)
        r = gather_gpu(GA_MAXPOOL, params, x, outn,
                       _shape2(n, c * oh * ow))
        if r:
            return r
    return maxpool2_cpu(x, n, c, h, w, k, stride, pad)

@jit.dont_look_inside
def maxpool2(x, c, h, w, k=2, stride=2, pad=0):
    t0 = prof_begin()
    r = _maxpool2_impl(x, c, h, w, k, stride, pad)
    prof_end(intmask(6), intmask(0), t0)
    return r


def _dptrs(srcs, offs):
    """The device pointers of srcs, each moved on by offs[i] elements; []
    if one of them cannot be put on the device."""
    ptrs = []
    for i in range(len(srcs)):
        p = dev(srcs[i])
        if p == 0:
            return []
        ptrs.append(p + offs[i] * nbytes(1, srcs[i].dtype))
    return ptrs


@jit.dont_look_inside
def _launch(kernel, ptrs, outn, shape, dt):
    if kernel.fn == 0 or not ptrs or outn <= 0:
        return NULLTENSOR
    nin = len(ptrs)
    collect_if_needed(nbytes(outn, dt))
    ins = lltype.malloc(SIGNEDARRAY, nin, flavor='raw')
    outs = lltype.malloc(SIGNEDARRAY, 1, flavor='raw')
    for i in range(nin):
        ins[i] = ptrs[i]
    outs[0] = device_alloc(nbytes(outn, dt), 0)
    ok = outs[0] != 0
    if ok:
        ok = rffi.cast(lltype.Signed, rt_cuda_launch(
            kernel.fn, ins, rffi.cast(rffi.INT, nin), outn, outs,
            rffi.cast(rffi.INT, 1), rffi.cast(rffi.INT, kernel.threads),
            config.flat, rffi.cast(rffi.INT, kernel.shared),
            rffi.cast(rffi.INT, kernel.nextra), 0)) != 0
    result = NULLTENSOR
    if ok:
        result = device_tensor(outn, outs[0], shape, dt)
    elif outs[0] != 0:
        rt_cuda_free(outs[0], nbytes(outn, dt))
    lltype.free(ins, flavor='raw')
    lltype.free(outs, flavor='raw')
    return result


def _shape_of(dims):
    shape = lltype.malloc(SHAPEARRAY, len(dims))
    for i in range(len(dims)):
        shape[i] = dims[i]
    return shape


def _numel(params):
    n = 1
    for j in range(params[1]):
        n *= params[2 + j]
    return n


def strided_cpu(srcs, offs, params, fill, shape):
    nsrc = params[0]
    rank = params[1]
    n = _numel(params)
    if n > 0:
        note_cpu_fallback()
    hs = [host(srcs[s]) for s in range(nsrc)]
    r = new_tensor(n, shape, srcs[0].dtype)
    hr = r.host
    for i in range(n):
        v = fill
        for s in range(nsrc):
            base = 2 + rank + 3 * rank * s
            rem = i
            addr = offs[s]
            ok = True
            for j in range(rank - 1, -1, -1):
                d = params[2 + j]
                x = rem % d
                rem = rem // d
                if (x < params[base + rank + j] or
                        x >= params[base + 2 * rank + j]):
                    ok = False
                    break
                addr += x * params[base + j]
            if ok:
                assert addr >= 0
                v = hs[s][addr]
                break
        hr[i] = v
    return r


@jit.dont_look_inside
def strided_launch(srcs, offs, params, fill, shape):
    """The GA_STRIDED kernel (see ttir.to_ttir_strided), source s read from
    offs[s] elements into srcs[s]."""
    n = _numel(params)
    dt = srcs[0].dtype
    if gpu_enabled() and n > 0:
        r = _launch(gather_kernel(GA_STRIDED, params, dt, fill),
                    _dptrs(srcs, offs), n, shape, dt)
        if r:
            return r
    return strided_cpu(srcs, offs, params, fill, shape)


def _coalesce(shape, strides):
    """Drop unit dimensions and merge neighbours that are contiguous in the
    source, so a permute or slice needs as few divisions as it can."""
    ns = []
    nt = []
    for j in range(len(shape)):
        if shape[j] == 1:
            continue
        k = len(ns) - 1
        if k >= 0 and nt[k] == strides[j] * shape[j]:
            ns[k] = ns[k] * shape[j]
            nt[k] = strides[j]
        else:
            ns.append(shape[j])
            nt.append(strides[j])
    if not ns:
        ns.append(1)
        nt.append(0)
    return ns, nt


@jit.dont_look_inside
def strided(x, shape, strides, offset):
    """A new contiguous tensor of the given shape, element (i0, ..) of which
    is x[offset + sum(i_j * strides[j])]; strides may be 0 or negative.
    Callers check the bounds."""
    t0 = prof_begin()
    cs, ct = _coalesce(shape, strides)
    rank = len(cs)
    params = [1, rank]
    params.extend(cs)
    params.extend(ct)
    params.extend([0] * rank)
    params.extend(cs)
    r = strided_launch([x], [offset], params, 0.0, _shape_of(shape))
    prof_end(intmask(12), intmask(0), t0)
    return r


MAX_CAT = 8

@jit.dont_look_inside
def _cat_impl(ts, outer):
    if len(ts) > MAX_CAT:
        # ponytail: past the launcher's 8 inputs, groups are joined and then
        # joined again, one extra pass; a strided store into the slots of one
        # buffer would avoid it.
        parts = []
        i = 0
        while i < len(ts):
            parts.append(_cat_impl(ts[i:min(i + MAX_CAT, len(ts))], outer))
            i += MAX_CAT
        return _cat_impl(parts, outer)
    total = 0
    for i in range(len(ts)):
        total += ts[i].size // outer
    rank = 1 if outer == 1 else 2
    params = [len(ts), rank]
    if rank == 2:
        params.append(outer)
    params.append(total)
    offs = []
    c = 0
    for i in range(len(ts)):
        wk = ts[i].size // outer
        if rank == 2:
            params.extend([wk, 1, 0, c, outer, c + wk])
        else:
            params.extend([1, c, c + wk])
        offs.append(-c)
        c += wk
    return strided_launch(ts, offs, params, 0.0, _shape2(outer, total))


@jit.dont_look_inside
def cat(ts, outer):
    """Each of ts viewed as [outer, size // outer], joined along the second
    axis into [outer, sum of the widths].  Callers check the sizes."""
    t0 = prof_begin()
    r = _cat_impl(ts, outer)
    prof_end(intmask(12), intmask(len(ts)), t0)
    return r


@jit.dont_look_inside
def pad(x, shape, pads, value):
    """x as a contiguous tensor of the given shape, pads[2*j] and
    pads[2*j+1] elements of value added before and after dimension j
    (negative pads crop)."""
    t0 = prof_begin()
    rank = len(shape)
    strides = [0] * rank
    stride = 1
    for j in range(rank - 1, -1, -1):
        strides[j] = stride
        stride *= shape[j]
    oshape = []
    off = 0
    for j in range(rank):
        oshape.append(shape[j] + pads[2 * j] + pads[2 * j + 1])
        off -= pads[2 * j] * strides[j]
    params = [1, rank]
    params.extend(oshape)
    params.extend(strides)
    for j in range(rank):
        params.append(pads[2 * j])
    for j in range(rank):
        params.append(pads[2 * j] + shape[j])
    r = strided_launch([x], [off], params, value, _shape_of(oshape))
    prof_end(intmask(12), intmask(1), t0)
    return r


@jit.dont_look_inside
def place(x, src, shape, starts, sizes):
    """x as a contiguous tensor of the given shape with the box starts ..
    starts + sizes replaced by src, contiguous in the box's shape: one
    launch that reads each element from src inside the box and from x
    elsewhere.  Callers check the bounds."""
    t0 = prof_begin()
    rank = len(shape)
    xst = [0] * rank
    sst = [0] * rank
    a = 1
    b = 1
    for j in range(rank - 1, -1, -1):
        xst[j] = a
        sst[j] = b
        a *= shape[j]
        b *= sizes[j]
    off = 0
    for j in range(rank):
        off -= starts[j] * sst[j]
    params = [2, rank]
    params.extend(shape)
    params.extend(sst)
    params.extend(starts)
    for j in range(rank):
        params.append(starts[j] + sizes[j])
    params.extend(xst)
    params.extend([0] * rank)
    params.extend(shape)
    r = strided_launch([src, x], [off, 0], params, 0.0, _shape_of(shape))
    prof_end(intmask(12), intmask(2), t0)
    return r


@jit.dont_look_inside
def device_zeros(shape_list, dtype):
    """Zeros of the given shape allocated and cleared on the device, or on
    the host when there is no device."""
    n = 1
    for d in shape_list:
        n *= d
    shape = _shape_of(shape_list)
    note_dtype(dtype)
    if gpu_enabled() and n > 0:
        collect_if_needed(nbytes(n, dtype))
        p = device_alloc(nbytes(n, dtype), 1)
        if p != 0:
            return device_tensor(n, p, shape, dtype)
    r = new_tensor(n, shape, dtype)
    for i in range(n):
        r.host[i] = 0.0
    return r


def take_cpu(x, idx, params, outn, shape):
    note_cpu_fallback()
    d, k, inner, so, sj, si = (params[0], params[1], params[2], params[3],
                               params[4], params[5])
    hx = host(x)
    hi = host(idx)
    r = new_tensor(outn, shape, x.dtype)
    hr = r.host
    for i in range(outn):
        xi = i % inner
        q = i // inner
        xj = q % k
        xo = q // k
        pos = xo * so + xj * sj + xi * si
        assert pos >= 0
        fv = hi[pos]
        v = 0.0
        if fv > -1.0 and fv < float(d):
            src = (xo * d + int(fv)) * inner + xi
            assert src >= 0
            v = hx[src]
        hr[i] = v
    return r


@jit.dont_look_inside
def take(x, idx, outer, k, inner, so, sj, si):
    """x as [outer, d, inner]; out[o, j, i] = x[o, idx[o*so + j*sj + i*si], i]
    as [outer*k, inner], zero for an index outside [0, d).  so, sj, si =
    k*inner, inner, 1 is torch.gather along the middle axis, 0, 1, 0 is
    index_select."""
    t0 = prof_begin()
    d = x.size // (outer * inner)
    params = [d, k, inner, so, sj, si]
    outn = outer * k * inner
    shape = _shape2(outer * k, inner)
    r = NULLTENSOR
    if gpu_enabled():
        r = _launch(gather_kernel(GA_TAKE, params, x.dtype),
                    _dptrs([x, idx], [0, 0]), outn, shape, x.dtype)
    if not r:
        r = take_cpu(x, idx, params, outn, shape)
    prof_end(intmask(13), intmask(0), t0)
    return r


def pool_cpu(x, wt, bias, params, outn, shape):
    note_cpu_fallback()
    mode, h, w, oh, ow = params[0], params[1], params[2], params[3], params[4]
    kh, kw, sh, sw = params[5], params[6], params[7], params[8]
    ph, pw, dh, dw = params[9], params[10], params[11], params[12]
    cout, mult, flags = params[13], params[14], params[15]
    conv = mode == POOL_DEPTHWISE
    hx = host(x)
    hw = hx
    hb = hx
    if conv:
        hw = host(wt)
        if flags & 1:
            hb = host(bias)
    r = new_tensor(outn, shape, x.dtype)
    hr = r.host
    cin = cout // mult
    for i in range(outn):
        if flags & 4:
            oc = i % cout
            t = i // cout
            owi = t % ow
            t = t // ow
            ohi = t % oh
            img = t // oh
        else:
            owi = i % ow
            t = i // ow
            ohi = t % oh
            oc = t // oh % cout
            img = t // oh // cout
        ci = oc // mult
        wbase = 0
        if conv:
            wbase = ci * kh * kw * mult + oc % mult
        cnt = 0
        if mode == POOL_ADAPTIVE:
            hs = ohi * h // oh
            he = (ohi * h + h + oh - 1) // oh
            ws = owi * w // ow
            we = (owi * w + w + ow - 1) // ow
            cnt = (he - hs) * (we - ws)
        else:
            hs = ohi * sh - ph
            ws = owi * sw - pw
            he = min(hs + kh, h + ph)
            we = min(ws + kw, w + pw)
            if flags & 1:
                cnt = (he - hs) * (we - ws)
            else:
                cnt = (min(he, h) - max(hs, 0)) * (min(we, w) - max(ws, 0))
        acc = 0.0
        for a in range(kh):
            ih = hs + a * dh
            if ih < 0 or ih >= h or (mode == POOL_ADAPTIVE and ih >= he):
                continue
            for b in range(kw):
                iw = ws + b * dw
                if iw < 0 or iw >= w or (mode == POOL_ADAPTIVE and iw >= we):
                    continue
                if flags & 2:
                    idx = ((img * h + ih) * w + iw) * cin + ci
                else:
                    idx = ((img * cin + ci) * h + ih) * w + iw
                assert idx >= 0
                v = hx[idx]
                if conv:
                    v *= hw[wbase + (a * kw + b) * mult]
                acc += v
        if conv:
            if flags & 1:
                acc += hb[oc]
        elif cnt > 0:
            acc /= cnt
        hr[i] = acc
    return r


@jit.dont_look_inside
def _pool(x, wt, bias, params, outn, shape):
    t0 = prof_begin()
    r = NULLTENSOR
    if gpu_enabled():
        srcs = [x]
        if params[0] == POOL_DEPTHWISE:
            srcs.append(wt)
            if params[15] & 1:
                srcs.append(bias)
        r = _launch(gather_kernel(GA_POOL, params, x.dtype),
                    _dptrs(srcs, [0] * len(srcs)), outn, shape, x.dtype)
    if not r:
        r = pool_cpu(x, wt, bias, params, outn, shape)
    prof_end(intmask(14), intmask(params[0]), t0)
    return r


def pool_out(h, k, stride, pad, ceil_mode):
    if not ceil_mode:
        return (h + 2 * pad - k) // stride + 1
    o = (h + 2 * pad - k + stride - 1) // stride + 1
    if (o - 1) * stride >= h + pad:
        o -= 1
    return o


def adaptive_window(h, oh):
    m = 0
    for i in range(oh):
        m = max(m, ((i + 1) * h + oh - 1) // oh - i * h // oh)
    return m


def _pool_params(mode, h, w, oh, ow, kh, kw, sh, sw, ph, pw, dh, dw, cout,
                 mult, flags):
    return [mode, h, w, oh, ow, kh, kw, sh, sw, ph, pw, dh, dw, cout, mult,
            flags]


def _pool_shape(n, c, oh, ow, layout):
    if layout & 2:
        return _shape2(n * oh * ow, c)
    return _shape2(n, c * oh * ow)


@jit.dont_look_inside
def avg_pool2d(x, c, h, w, kh, kw, sh, sw, ph, pw, include_pad, ceil_mode,
               layout=0):
    """Average pooling as [n, c*oh*ow], PyTorch's divisor rules.  layout
    bit 0 says the input is NHWC, bit 1 asks for an NHWC [n*oh*ow, c]."""
    n = x.size // (c * h * w)
    oh = pool_out(h, kh, sh, ph, ceil_mode)
    ow = pool_out(w, kw, sw, pw, ceil_mode)
    params = _pool_params(POOL_AVG, h, w, oh, ow, kh, kw, sh, sw, ph, pw, 1,
                          1, c if layout else 1, 1,
                          (1 if include_pad else 0) | layout << 1)
    return _pool(x, NULLTENSOR, NULLTENSOR, params, n * c * oh * ow,
                 _pool_shape(n, c, oh, ow, layout))


@jit.dont_look_inside
def adaptive_avg_pool2d(x, c, h, w, oh, ow, layout=0):
    """Adaptive average pooling to oh x ow as [n, c*oh*ow], the windows
    [floor(i*h/oh), ceil((i+1)*h/oh)); layout as for avg_pool2d."""
    n = x.size // (c * h * w)
    params = _pool_params(POOL_ADAPTIVE, h, w, oh, ow, adaptive_window(h, oh),
                          adaptive_window(w, ow), 1, 1, 0, 0, 1, 1,
                          c if layout else 1, 1, layout << 1)
    return _pool(x, NULLTENSOR, NULLTENSOR, params, n * c * oh * ow,
                 _pool_shape(n, c, oh, ow, layout))


@jit.dont_look_inside
def depthwise_conv2d(x, wt, bias, c, h, w, kh, kw, sh, sw, ph, pw, dh, dw,
                     mult, layout=0):
    """groups == c convolution in one pass, weight [c*kh*kw, mult], output
    channel ci*mult + m, as [n, c*mult*oh*ow]; bias may be null; layout as
    for avg_pool2d."""
    n = x.size // (c * h * w)
    oh = conv_out(h, kh, sh, ph, dh)
    ow = conv_out(w, kw, sw, pw, dw)
    params = _pool_params(POOL_DEPTHWISE, h, w, oh, ow, kh, kw, sh, sw, ph,
                          pw, dh, dw, c * mult, mult,
                          (1 if bias else 0) | layout << 1)
    return _pool(x, wt, bias, params, n * c * mult * oh * ow,
                 _pool_shape(n, c * mult, oh, ow, layout))


def conv_cpu(x, wt, p):
    note_cpu_fallback()
    d = conv_descs.params[p]
    layout, c, h, w, o = d[0], d[1], d[2], d[3], d[4]
    kh, kw, sh, sw = d[5], d[6], d[7], d[8]
    ph, pw, dh, dw = d[9], d[10], d[11], d[12]
    oh, ow, rows, g = d[13], d[14], d[15], d[CD_G]
    cg = c // g
    og = o // g
    hx = host(x)
    hw = host(wt)
    r = new_tensor(rows * o, _shape2(rows, o), x.dtype)
    hr = r.host
    for i in range(rows * o):
        row = i // o
        oc = i % o
        img = row // (oh * ow)
        pos = row % (oh * ow)
        acc = 0.0
        for k in range(kh * kw * cg):
            ci = oc // og * cg + k % cg
            ih = pos // ow * sh - ph + k // cg // kw * dh
            iw = pos % ow * sw - pw + k // cg % kw * dw
            if ih < 0 or ih >= h or iw < 0 or iw >= w:
                continue
            if layout & 1:
                idx = ((img * h + ih) * w + iw) * c + ci
            else:
                idx = ((img * c + ci) * h + ih) * w + iw
            assert idx >= 0
            acc += hx[idx] * hw[k * o + oc]
        hr[i] = acc
    return r


def conv_configs(c, o, kdim, nhwc, g):
    cg = c // g
    og = o // g
    if g > 1 and (cg % 16 != 0 or og < 16):
        bn = 16
        while bn < o and bn < 64:
            bn *= 2
        return [[32, bn, 1, 8, 4, 1], [64, bn, 1, 8, 4, 1],
                [16, min(bn * 2, 128), 1, 8, 4, 1],
                [64, max(bn // 2, 16), 1, 8, 4, 1]]
    if nhwc and cg % 32 == 0:
        bk = 32
    elif (nhwc and cg % 16 == 0) or kdim <= 16:
        bk = 16
    else:
        bk = 32
    bn = 64
    if og <= 16:
        bn = 16
    elif og <= 32:
        bn = 32
    cfgs = [[128, bn, bk, 8, 4, 3], [64, bn, bk, 8, 4, 4],
            [32, bn, bk, 8, 4, 4]]
    if og > 64:
        cfgs.append([64, 128, bk, 8, 4, 4])
        cfgs.append([128, 128, bk, 8, 8, 3])
    return cfgs


class ConvTune(object):
    def __init__(self):
        self.best = {}
        self.kernels = {}
        self.cfg = {}
conv_tune = ConvTune()


def _conv_key(layout, c, h, w, o, kh, kw, sh, sw, ph, pw, dh, dw, g, n):
    return ','.join([str(v) for v in [layout, c, h, w, o, kh, kw, sh, sw, ph,
                                      pw, dh, dw, g, n]])


@jit.elidable
def conv_choice(layout, c, h, w, o, kh, kw, sh, sw, ph, pw, dh, dw, g, n):
    return conv_tune.best.get(_conv_key(layout, c, h, w, o, kh, kw, sh, sw,
                                        ph, pw, dh, dw, g, n), -1)


def conv_launch_gpu(kernel, inputs):
    dt = kernel.dtype
    for k in range(len(inputs)):
        if inputs[k].dtype != dt:
            return NULLTENSOR
    if kernel.nouts != 1 + len(kernel.outputs):
        return NULLTENSOR
    d = conv_descs.params[_first_conv(kernel)]
    rows, o = d[CD_ROWS], d[CD_O]
    images = rows // (d[13] * d[14])
    n = rows * o
    nb = nbytes(n, dt)
    nin = len(inputs)
    nout = kernel.nouts
    collect_if_needed(nb * nout)
    ins = lltype.malloc(SIGNEDARRAY, nin, flavor='raw')
    outs = lltype.malloc(SIGNEDARRAY, nout, flavor='raw')
    ok = True
    for k in range(nin):
        ins[k] = dev(inputs[k])
        if ins[k] == 0:
            ok = False
    for k in range(nout):
        outs[k] = device_alloc(nb, 0) if ok else 0
        if outs[k] == 0:
            ok = False
    if ok:
        ok = rffi.cast(lltype.Signed, rt_cuda_launch(
            kernel.fn, ins, rffi.cast(rffi.INT, nin), conv_programs(d), outs,
            rffi.cast(rffi.INT, nout), rffi.cast(rffi.INT, kernel.threads), 1,
            rffi.cast(rffi.INT, kernel.shared),
            rffi.cast(rffi.INT, kernel.nextra), images)) != 0
    result = NULLTENSOR
    if ok:
        result = device_tensor(n, outs[0], _shape2(rows, o), dt)
        if nout > 1:
            result.extra = lltype.malloc(TENSORARRAY, nout - 1)
            for k in range(1, nout):
                result.extra[k - 1] = device_tensor(n, outs[k],
                                                    _shape2(rows, o), dt)
    else:
        for k in range(nout):
            if outs[k] != 0:
                rt_cuda_free(outs[k], nb)
    lltype.free(ins, flavor='raw')
    lltype.free(outs, flavor='raw')
    return result


def _first_conv(kernel):
    k = main_conv(kernel)
    if k < 0:
        return -1
    return kernel.nodes[k].p


@jit.dont_look_inside
def conv_op(x, wt, p):
    k = conv_tune.kernels.get(p, lltype.nullptr(KERNEL))
    if k and k.fn != 0 and gpu_enabled():
        r = conv_launch_gpu(k, [x, wt])
        if r:
            return r
    return conv_cpu(x, wt, p)


def _conv_kernel(p, dt):
    k = new_kernel(2, 1, dt)
    set_node(k, 0, CONV, 0, 1, p)
    compile_or_reuse(k)
    conv_tune.kernels[p] = k
    return k


@jit.dont_look_inside
def conv_ensure(x, wt, layout, c, h, w, o, kh, kw, sh, sw, ph, pw, dh, dw, g,
                n):
    key = _conv_key(layout, c, h, w, o, kh, kw, sh, sw, ph, pw, dh, dw, g, n)
    if key in conv_tune.best:
        return
    cfgs = conv_configs(c, o, kh * kw * (c // g), layout & 1, g)
    gkey = _conv_key(layout, c, h, w, o, kh, kw, sh, sw, ph, pw, dh, dw, g, 0)
    best = conv_desc(conv_geometry(layout, c, h, w, o, kh, kw, sh, sw, ph,
                                   pw, dh, dw, g, n, cfgs[0]))
    dt = x.dtype
    if gpu_enabled() and dt == F32 and wt.dtype == dt and dev(x) != 0 and \
            dev(wt) != 0:
        known = conv_tune.cfg.get(gkey, -1)
        if known >= 0:
            p = conv_desc(conv_geometry(layout, c, h, w, o, kh, kw, sh, sw,
                                        ph, pw, dh, dw, g, n, cfgs[known]))
            if _conv_kernel(p, dt).fn != 0:
                conv_tune.best[key] = p
                return
        known = tuned_config(key)
        if known >= 0 and known < len(cfgs):
            p = conv_desc(conv_geometry(layout, c, h, w, o, kh, kw, sh, sw,
                                        ph, pw, dh, dw, g, n, cfgs[known]))
            conv_descs.static[p] = True
            if _conv_kernel(p, dt).fn != 0:
                conv_tune.cfg[gkey] = known
                conv_tune.best[key] = p
                return
        bestt = 0.0
        besti = -1
        for i in range(len(cfgs)):
            p = conv_desc(conv_geometry(layout, c, h, w, o, kh, kw, sh, sw,
                                        ph, pw, dh, dw, g, n, cfgs[i]))
            conv_descs.static[p] = True
            k = _conv_kernel(p, dt)
            if k.fn == 0 or not conv_launch_gpu(k, [x, wt]):
                continue
            rt_cuda_sync()
            t0 = rt_cuda_now()
            for j in range(10):
                conv_launch_gpu(k, [x, wt])
            rt_cuda_sync()
            t = rt_cuda_now() - t0
            if besti < 0 or t < bestt:
                best = p
                bestt = t
                besti = i
        if besti >= 0:
            remember_config(key, besti)
            conv_tune.cfg[gkey] = besti
    conv_tune.best[key] = best


def im2col2_cpu(x, n, c, h, w, kh, kw, sh, sw, ph, pw, dh, dw, nhwc=0):
    note_cpu_fallback()
    hx = host(x)
    oh = conv_out(h, kh, sh, ph, dh)
    ow = conv_out(w, kw, sw, pw, dw)
    ohw = oh * ow
    kk = kh * kw
    rows = n * ohw
    cols = c * kk
    r = new_tensor(rows * cols, _shape2(rows, cols), x.dtype)
    hr = r.host
    for i in range(rows * cols):
        row = i // cols
        col = i % cols
        img = row // ohw
        pos = row % ohw
        rk = col % kk
        ih = pos // ow * sh + rk // kw * dh - ph
        iw = pos % ow * sw + rk % kw * dw - pw
        v = 0.0
        if ih >= 0 and ih < h and iw >= 0 and iw < w:
            if nhwc:
                idx = ((img * h + ih) * w + iw) * c + col // kk
            else:
                idx = ((img * c + col // kk) * h + ih) * w + iw
            assert idx >= 0
            v = hx[idx]
        hr[i] = v
    return r


@jit.dont_look_inside
def im2col2(x, c, h, w, kh, kw, sh, sw, ph, pw, dh, dw, nhwc=0):
    """im2col with a kh x kw kernel, per-axis stride, padding and dilation;
    the square undilated case of an NCHW input is im2col itself.  nhwc says
    the input is NHWC; the columns are (channel, kernel row, kernel column)
    either way."""
    if (not nhwc and kh == kw and sh == sw and ph == pw and dh == 1 and
            dw == 1):
        return im2col(x, c, h, w, kh, ph, sh)
    t0 = prof_begin()
    n = x.size // (c * h * w)
    rows = n * conv_out(h, kh, sh, ph, dh) * conv_out(w, kw, sw, pw, dw)
    cols = c * kh * kw
    r = NULLTENSOR
    if gpu_enabled():
        params = [n, c, h, w, kh, kw, sh, sw, ph, pw, dh, dw]
        if nhwc:
            params.append(1)
        r = gather_gpu(GA_IM2COL2, params, x, rows * cols,
                       _shape2(rows, cols))
    if not r:
        r = im2col2_cpu(x, n, c, h, w, kh, kw, sh, sw, ph, pw, dh, dw, nhwc)
    prof_end(intmask(4), intmask(0), t0)
    return r


def im2col_t_cpu(x, n, c, h, w, kh, kw, sh, sw, ph, pw, dh, dw, oh, ow):
    note_cpu_fallback()
    hx = host(x)
    ohw = oh * ow
    kk = kh * kw
    rows = n * ohw
    cols = c * kk
    r = new_tensor(rows * cols, _shape2(rows, cols), x.dtype)
    hr = r.host
    for i in range(rows * cols):
        row = i // cols
        col = i % cols
        img = row // ohw
        pos = row % ohw
        rk = col % kk
        ty = pos // ow + ph - rk // kw * dh
        tx = pos % ow + pw - rk % kw * dw
        v = 0.0
        if (ty >= 0 and tx >= 0 and ty % sh == 0 and tx % sw == 0 and
                ty // sh < h and tx // sw < w):
            idx = ((img * c + col // kk) * h + ty // sh) * w + tx // sw
            assert idx >= 0
            v = hx[idx]
        hr[i] = v
    return r


@jit.dont_look_inside
def im2col_t(x, c, h, w, kh, kw, sh, sw, ph, pw, dh, dw, oh, ow):
    """The im2col of a transposed convolution with an oh x ow output: row
    (img, y, x), column (channel, a, b) is input pixel ((y + ph - a*dh) / sh,
    (x + pw - b*dw) / sw) where that divides, 0 elsewhere."""
    t0 = prof_begin()
    n = x.size // (c * h * w)
    rows = n * oh * ow
    cols = c * kh * kw
    r = NULLTENSOR
    if gpu_enabled():
        params = [n, c, h, w, kh, kw, sh, sw, ph, pw, dh, dw, oh, ow]
        r = gather_gpu(GA_IM2COL_T, params, x, rows * cols,
                       _shape2(rows, cols))
    if not r:
        r = im2col_t_cpu(x, n, c, h, w, kh, kw, sh, sw, ph, pw, dh, dw, oh,
                         ow)
    prof_end(intmask(4), intmask(0), t0)
    return r


def rot_half_cpu(x, dh):
    note_cpu_fallback()
    hx = host(x)
    n = x.size
    half = dh // 2
    r = new_tensor(n, _shape2(n // cols(x), cols(x)), x.dtype)
    hr = r.host
    for i in range(n):
        idx = i // dh * dh + (i % dh + half) % dh
        assert idx >= 0
        hr[i] = hx[idx]
    return r


@jit.dont_look_inside
def _rot_half_impl(x, dh):
    if gpu_enabled():
        params = [dh]
        r = gather_gpu(GA_ROTHALF, params, x, x.size,
                       _shape2(x.size // cols(x), cols(x)))
        if r:
            return r
    return rot_half_cpu(x, dh)

@jit.dont_look_inside
def rot_half(x, dh):
    t0 = prof_begin()
    r = _rot_half_impl(x, dh)
    prof_end(intmask(10), intmask(0), t0)
    return r


def head_split_cpu(x, rows, dh, heads):
    note_cpu_fallback()
    hx = host(x)
    r = new_tensor(rows * dh * heads, _shape2(heads * rows, dh), x.dtype)
    hr = r.host
    for i in range(rows * dh * heads):
        hi = i // (rows * dh)
        rem = i % (rows * dh)
        idx = rem // dh * (heads * dh) + hi * dh + rem % dh
        assert idx >= 0
        hr[i] = hx[idx]
    return r


def head_merge_cpu(x, rows, dh, heads):
    note_cpu_fallback()
    hx = host(x)
    r = new_tensor(rows * dh * heads, _shape2(rows, heads * dh), x.dtype)
    hr = r.host
    for i in range(rows * dh * heads):
        ri = i // (heads * dh)
        rem = i % (heads * dh)
        idx = rem // dh * (rows * dh) + ri * dh + rem % dh
        assert idx >= 0
        hr[i] = hx[idx]
    return r

@jit.dont_look_inside
def _head_split_impl(x, rows, dh, heads):
    outn = rows * dh * heads
    if gpu_enabled():
        params = [rows]
        params.append(dh)
        params.append(heads)
        r = gather_gpu(GA_HEADSPLIT, params, x, outn,
                       _shape2(heads * rows, dh))
        if r:
            return r
    return head_split_cpu(x, rows, dh, heads)


@jit.dont_look_inside
def _head_merge_impl(x, rows, dh, heads):
    outn = rows * dh * heads
    if gpu_enabled():
        params = [rows]
        params.append(dh)
        params.append(heads)
        r = gather_gpu(GA_HEADMERGE, params, x, outn,
                       _shape2(rows, heads * dh))
        if r:
            return r
    return head_merge_cpu(x, rows, dh, heads)

@jit.dont_look_inside
def head_split(x, rows, dh, heads):
    t0 = prof_begin()
    r = _head_split_impl(x, rows, dh, heads)
    prof_end(intmask(7), intmask(0), t0)
    return r

@jit.dont_look_inside
def head_merge(x, rows, dh, heads):
    t0 = prof_begin()
    r = _head_merge_impl(x, rows, dh, heads)
    prof_end(intmask(8), intmask(0), t0)
    return r


@jit.dont_look_inside
def _tensor_assign_impl(dst, src):
    if dst.dptr != 0 and gpu_enabled():
        dptr_src = dev(src)
        if dptr_src != 0:
            ok = rffi.cast(lltype.Signed, rt_cuda_copy(
                dst.dptr, dptr_src, nbytes(dst.size, dst.dtype))) != 0
            if ok:
                if dst.host:
                    download(dst.dptr, dst.host, dst.dtype)
                return dst
    note_cpu_fallback()
    hdst = host(dst)
    hsrc = host(src)
    for i in range(dst.size):
        hdst[i] = hsrc[i]
    return dst

@jit.dont_look_inside
def tensor_write_rows(dst, src, row):
    """dst[row:row+rows(src)] = src, in place, on the device when dst lives
    there.  Callers check the bounds."""
    off = row * cols(dst)
    if dst.dptr != 0 and gpu_enabled():
        dptr_src = dev(src)
        if dptr_src != 0:
            isz = nbytes(1, dst.dtype)
            ok = rffi.cast(lltype.Signed, rt_cuda_copy(
                dst.dptr + off * isz, dptr_src,
                nbytes(src.size, dst.dtype))) != 0
            if ok:
                if dst.host:
                    download(dst.dptr, dst.host, dst.dtype)
                return dst
    # The host path is for a host-resident dst (RTENSOR_CPU).  A failed copy
    # into a device dst would leave the device copy stale.
    note_cpu_fallback()
    hdst = host(dst)
    hsrc = host(src)
    for i in range(src.size):
        hdst[off + i] = hsrc[i]
    return dst


def _scatter_launch(k, src, dptr):
    ins = lltype.malloc(SIGNEDARRAY, 1, flavor='raw')
    outs = lltype.malloc(SIGNEDARRAY, 1, flavor='raw')
    ins[0] = dev(src)
    outs[0] = dptr
    ok = ins[0] != 0
    if ok:
        ok = rffi.cast(lltype.Signed, rt_cuda_launch(
            k.fn, ins, rffi.cast(rffi.INT, 1), src.size, outs,
            rffi.cast(rffi.INT, 1), rffi.cast(rffi.INT, k.threads),
            k.block, rffi.cast(rffi.INT, k.shared),
            rffi.cast(rffi.INT, k.nextra), 0)) != 0
    lltype.free(ins, flavor='raw')
    lltype.free(outs, flavor='raw')
    return ok


def scatter_cpu(x, src, params, off):
    note_cpu_fallback()
    hx = host(x)
    hs = host(src)
    rank = params[1]
    for i in range(src.size):
        rem = i
        addr = off
        for j in range(rank - 1, -1, -1):
            d = params[2 + j]
            addr += (rem % d) * params[2 + rank + j]
            rem = rem // d
        assert addr >= 0
        hx[addr] = hs[i]
    if x.dptr != 0:
        p = upload(hx, x.dtype)
        if p != 0:
            attach_buffer(x, p, nbytes(x.size, x.dtype))
        else:
            x.dptr = 0
            x.buf = lltype.nullptr(OBJECTPTR.TO)


@jit.dont_look_inside
def _place_into_impl(x, src, vid, off):
    params = view_params_list(vid)
    if gpu_enabled() and dev(x) != 0:
        k = lookup_gather(GA_SCATTER, params, x.dtype)
        if (k is not None and k.fn != 0 and
                _scatter_launch(k, src, x.dptr + off * nbytes(1, x.dtype))):
            if x.host:
                download(x.dptr, x.host, x.dtype)
            return x
    scatter_cpu(x, src, params, off)
    return x


@jit.dont_look_inside
def place_into(x, src, vid, off):
    """Write src, contiguous in the shape of view vid, into x's own buffer at
    the elements vid (moved on by off) selects; x.  Callers check the bounds
    and that nothing else holds x's buffer."""
    t0 = prof_begin()
    r = _place_into_impl(x, src, vid, off)
    prof_end(intmask(9), intmask(1), t0)
    return r


@jit.dont_look_inside
def tensor_assign(dst, src):
    t0 = prof_begin()
    r = _tensor_assign_impl(dst, src)
    prof_end(intmask(9), intmask(0), t0)
    return r


class Ones(object):
    def __init__(self):
        self.n = [-1] * NDTYPES
        self.t = [NULLTENSOR] * NDTYPES
        self.one = [NULLTENSOR] * NDTYPES
        self.cache = {}
ones = Ones()

def _make_ones(n, dtype):
    t = new_tensor(n, lltype.nullptr(SHAPEARRAY), dtype)
    for i in range(n):
        t.host[i] = 1.0
    dev(t)
    return t


class IotaCache(object):
    def __init__(self):
        self.tensors = [{}, {}, {}]
iotas = IotaCache()


@jit.elidable
def iota_of(n, dtype):
    t = iotas.tensors[dtype].get(n, NULLTENSOR)
    if not t:
        t = new_tensor(n, lltype.nullptr(SHAPEARRAY), dtype)
        for i in range(n):
            t.host[i] = float(i)
        dev(t)
        iotas.tensors[dtype][n] = t
    return t


class ScalarCache(object):
    def __init__(self):
        self.tensors = [{}, {}, {}]
scalars = ScalarCache()


def scalar(value):
    return scalar_of(value, policy.dtype)


@jit.elidable
def scalar_of(value, dtype):
    return cached_scalar(value, dtype)


@specialize.call_location()
def cached_scalar(value, dtype):
    """One copy per caller: the launchers' per-node fallback is annotated
    late, and must not widen the annotation scalar_of already has."""
    t = scalars.tensors[dtype].get(value, NULLTENSOR)
    if not t:
        t = new_tensor(1, lltype.nullptr(SHAPEARRAY), dtype)
        t.host[0] = value
        dev(t)
        scalars.tensors[dtype][value] = t
    return t
