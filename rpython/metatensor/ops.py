from rpython.rlib import jit
from rpython.rtyper.lltypesystem import lltype
from rpython.rtyper.rclass import OBJECTPTR
from rpython.metatensor.core import (ADD, B_KEEP_NZ, B_KEEP_Z, BINARY, CONV, conv_knob, NPARAMS, UNARY, AXIS_ALL, GA_HEADSPLIT, GA_ROTHALF, GATHER, gather_knob, gather_param, BC_L_COL, BC_L_ROW, BC_L_SCALAR, BC_NONE, BC_R_COL, BC_R_ROW, BC_R_SCALAR, DIV, EQMASK, EXP, MAXR, MUL, NDTYPES, NULLTENSOR, RELU, RELUGRAD, SHAPEARRAY, SQRT, SUB, SUM, TENSOR, TENSORARRAY, _shape2, cols, new_tensor, note_cols, note_dtype, note_size, policy, B_GE, B_LT, F16, view_root_clamped, view_dim, view_param, view_root)
from rpython.metatensor.device import (dev, host, note_cpu_fallback)
from rpython.metatensor.kernels import (ensure_gather, ensure_scatter, ensure_single, ensure_view)
from rpython.metatensor import lazy
from rpython.metatensor.devops import conv_choice, conv_ensure, place_into
from rpython.metatensor.runtime import (_make_ones, eval_op, head_merge, head_split, rot_half, ones, tensor_assign, tensor_matmul, iota_of, place, scalar_of, strided)

def astype(t, dtype):
    if lazy.enabled():
        lazy.force(t)
    if t.dtype == dtype:
        return t
    note_dtype(dtype)
    note_cpu_fallback()
    h = host(t)
    r = new_tensor(t.size, t.shape, dtype)
    for i in range(t.size):
        r.host[i] = h[i]
    return r

@jit.unroll_safe
def reshape(a, shape_list):
    n = 1
    for d in shape_list:
        n *= d
    if (jit.we_are_jitted() and not lazy.enabled() and
            not gather_knob.off and len(shape_list) <= MAX_VIEW_RANK):
        if n != tensor_size(a):
            raise ValueError("shape mismatch")
        rank = len(shape_list)
        strides = [0] * rank
        acc = 1
        for j in range(rank - 1, -1, -1):
            strides[j] = acc
            acc *= shape_list[j]
        return view_node(a, _strided_vid(shape_list, strides, 0))
    a = tensor_force(a)
    if n != a.size:
        raise ValueError("shape mismatch")
    shape = lltype.malloc(SHAPEARRAY, len(shape_list))
    for i in range(len(shape_list)):
        shape[i] = shape_list[i]
    return view(a, shape)

def view(a, shape):
    if lazy.enabled():
        lazy.force(a)
    if a.dptr == 0 and a.host:
        dev(a)
    r = lltype.malloc(TENSOR)
    r.size = a.size
    r.shape = shape
    r.dptr = a.dptr
    r.host = a.host
    r.extra = lltype.nullptr(TENSORARRAY)
    r.dtype = a.dtype
    r.buf = a.buf
    r.lazy = lltype.nullptr(OBJECTPTR.TO)
    return r


@jit.unroll_safe
def ones_like(a):
    n = tensor_size(a)
    nd = tensor_ndim(a)
    dtype = tensor_dtype(a)
    shape = lltype.malloc(SHAPEARRAY, nd)
    for i in range(nd):
        shape[i] = tensor_shape(a, i)
    if n == 1:
        if not ones.one[dtype]:
            ones.one[dtype] = _make_ones(1, dtype)
        return view(ones.one[dtype], shape)
    key = n * NDTYPES + dtype
    t = ones.cache.get(key, NULLTENSOR)
    if not t:
        t = _make_ones(n, dtype)
        ones.cache[key] = t
    return view(t, shape)

def view2(a, rows, cols):
    return view(a, _shape2(rows, cols))



@jit.oopspec("tensor.add(a, b, bcast)")
def tensor_add(a, b, bcast):
    return eval_op(ADD, a, b, bcast)

@jit.oopspec("tensor.mul(a, b, bcast)")
def tensor_mul(a, b, bcast):
    return eval_op(MUL, a, b, bcast)

@jit.oopspec("tensor.relu(a)")
def tensor_relu(a):
    return eval_op(RELU, a, NULLTENSOR, 0)

@jit.oopspec("tensor.sum(a, axis)")
def tensor_sum(a, axis):
    return eval_op(SUM, a, NULLTENSOR, axis)

@jit.oopspec("tensor.relugrad(y, g, bcast)")
def tensor_relugrad(y, g, bcast):
    return eval_op(RELUGRAD, y, g, bcast)

@jit.oopspec("tensor.sub(a, b, bcast)")
def tensor_sub(a, b, bcast):
    return eval_op(SUB, a, b, bcast)

@jit.oopspec("tensor.div(a, b, bcast)")
def tensor_div(a, b, bcast):
    return eval_op(DIV, a, b, bcast)

@jit.oopspec("tensor.exp(a)")
def tensor_exp(a):
    return eval_op(EXP, a, NULLTENSOR, 0)

@jit.oopspec("tensor.sqrt(a)")
def tensor_sqrt(a):
    return eval_op(SQRT, a, NULLTENSOR, 0)

@jit.oopspec("tensor.maxr(a, axis)")
def tensor_maxr(a, axis):
    return eval_op(MAXR, a, NULLTENSOR, axis)

@jit.oopspec("tensor.eqmask(a, b, bcast)")
def tensor_eqmask(a, b, bcast):
    return eval_op(EQMASK, a, b, bcast)

@jit.oopspec("tensor.unary(a, fn)")
def tensor_unary(a, fn):
    return eval_op(UNARY, a, NULLTENSOR, fn)

@jit.oopspec("tensor.binary(a, b, p)")
def tensor_binary(a, b, p):
    return eval_op(BINARY, a, b, p)

@jit.oopspec("tensor.gather2(a, b, p)")
def tensor_gather2(a, b, p):
    return eval_op(GATHER, a, b, p)

@jit.oopspec("tensor.gather(a, p)")
def tensor_gather(a, p):
    # `a` in the unused slot: a constant NULLTENSOR would pin eval_op's annotation.
    return eval_op(GATHER, a, a, p)

def gather(a, kind, dh, heads):
    """rot_half / head_split / head_merge as a fusible node.

    Deferred execution and METATENSOR_NO_GATHER_FUSION keep standalone gathers."""
    if lazy.enabled() or gather_knob.off:
        if lazy.enabled():
            lazy.force(a)
        if kind == GA_ROTHALF:
            return rot_half(a, dh)
        rows = tensor_size(a) // (dh * heads)
        if kind == GA_HEADSPLIT:
            return head_split(a, rows, dh, heads)
        return head_merge(a, rows, dh, heads)
    # The pass fuses a node only when its param is a trace constant, and dh
    # usually comes from a shape read on a value the trace does not know.
    dh = jit.promote(dh)
    heads = jit.promote(heads)
    p = gather_param(kind, dh, heads)
    _ensure_gather(p, a)
    return tensor_gather(a, p)

MAX_VIEW_RANK = 8

MAX_DYNAMIC_OFFSET = 1 << 24

def strided_view(a, shape, strides, offset):
    if (not jit.we_are_jitted() or lazy.enabled() or gather_knob.off or
            len(shape) > MAX_VIEW_RANK):
        return strided(a, shape, strides, offset)
    if jit.isconstant(offset):
        return view_node(a, _strided_vid(shape, strides, offset))
    if not _dynamic_fits(a, offset):
        return strided(a, shape, strides, offset)
    return view_node_at(a, _strided_vid(shape, strides, 0), offset)

def _dynamic_fits(a, offset):
    return (tensor_dtype(a) != F16 and -MAX_DYNAMIC_OFFSET < offset <
            MAX_DYNAMIC_OFFSET)

@jit.unroll_safe
def _known(xs):
    for j in range(len(xs)):
        if not jit.isconstant(xs[j]):
            return False
    return True

@jit.unroll_safe
def _strided_vid(shape, strides, offset):
    vid = view_root(offset)
    for j in range(jit.promote(len(shape))):
        e = jit.promote(shape[j])
        vid = view_dim(vid, e, jit.promote(strides[j]), 0, e)
    return vid

def place_box_into(x, src, shape, starts, sizes):
    """place_box into x's own buffer. The caller guarantees nothing else holds it."""
    if lazy.enabled():
        lazy.force(x)
        lazy.force(src)
        lazy.barrier()
    strides, off = _box_strides(shape, starts)
    vid = _strided_vid(sizes, strides, 0)
    _ensure_scatter(vid, x)
    return tensor_place_into(x, src, vid, off)

@jit.unroll_safe
def _box_strides(shape, starts):
    rank = jit.promote(len(shape))
    strides = [0] * rank
    b = 1
    off = 0
    for j in range(rank - 1, -1, -1):
        strides[j] = b
        off += starts[j] * b
        b *= jit.promote(shape[j])
    return strides, off

@jit.not_in_trace
def _ensure_scatter(vid, a):
    ensure_scatter(vid, a.dtype)

@jit.oopspec("tensor.place_into(x, src, vid, off)")
def tensor_place_into(x, src, vid, off):
    return place_into(x, src, vid, off)

def place_box(x, src, shape, starts, sizes):
    if (not jit.we_are_jitted() or lazy.enabled() or gather_knob.off or
            len(shape) > MAX_VIEW_RANK):
        return place(x, src, shape, starts, sizes)
    known = _known(starts)
    dyn = 0
    if not known:
        dyn = _dynamic_offset(starts, sizes)
        if not _dynamic_fits(x, dyn):
            return place(x, src, shape, starts, sizes)
    pad, box = _place_vids(shape, starts, sizes, known)
    dtype = jit.promote(tensor_dtype(x))
    one = scalar_of(float(dtype >= 0), dtype)
    m = view_node(one, box)
    if known:
        return add(binary(m, x, B_KEEP_Z), view_node(src, pad))
    m = _dynamic_mask(m, shape, starts, sizes, dtype)
    pv = binary(m, view_node_at(src, pad, dyn), B_KEEP_NZ)
    return add(binary(m, x, B_KEEP_Z), pv)

@jit.unroll_safe
def _dynamic_offset(starts, sizes):
    rank = len(starts)
    b = 1
    off = 0
    for j in range(rank - 1, -1, -1):
        if not jit.isconstant(starts[j]):
            off -= starts[j] * b
        b *= jit.promote(sizes[j])
    return off

@jit.unroll_safe
def _dynamic_mask(m, shape, starts, sizes, dtype):
    rank = len(shape)
    for j in range(rank):
        if jit.isconstant(starts[j]):
            continue
        e = jit.promote(shape[j])
        vid = view_root(0)
        for d in range(rank):
            vid = view_dim(vid, jit.promote(shape[d]), int(d == j), 0,
                           jit.promote(shape[d]))
        coord = view_node(iota_of(e, dtype), vid)
        lo = starts[j]
        hi = lo + jit.promote(sizes[j])
        inside = mul(binary(coord, scalar_of(float(lo), dtype), B_GE),
                     binary(coord, scalar_of(float(hi), dtype), B_LT))
        m = mul(m, inside)
    return m

@jit.unroll_safe
def _place_vids(shape, starts, sizes, known):
    rank = jit.promote(len(shape))
    sst = [0] * rank
    b = 1
    for j in range(rank - 1, -1, -1):
        sst[j] = b
        b *= jit.promote(sizes[j])
    off = 0
    for j in range(rank):
        if known or jit.isconstant(starts[j]):
            off -= starts[j] * sst[j]
    pad = view_root(off) if known else view_root_clamped(off, b)
    box = view_root(0)
    for j in range(rank):
        e = jit.promote(shape[j])
        if known or jit.isconstant(starts[j]):
            lo = starts[j]
            hi = lo + jit.promote(sizes[j])
            pad = view_dim(pad, e, sst[j], lo, hi)
            box = view_dim(box, e, 0, lo, hi)
        else:
            pad = view_dim(pad, e, sst[j], 0, e)
            box = view_dim(box, e, 0, 0, e)
    return pad, box

def view_node(a, vid):
    _ensure_view(vid, a)
    return tensor_gather(a, view_param(vid))

def view_node_at(a, vid, offset):
    dtype = jit.promote(tensor_dtype(a))
    _ensure_view(vid, a)
    return tensor_gather2(a, scalar_of(float(offset), dtype),
                          view_param(vid))

@jit.not_in_trace
def _ensure_view(vid, a):
    ensure_view(vid, a.dtype)

@jit.not_in_trace
def _ensure_gather(p, a):
    ensure_gather(p, a.size, a.dtype)

@jit.not_in_trace
def _ensure_single(opcode, p, a):
    ensure_single(opcode, p, a.dtype)

@jit.oopspec("tensor.conv(x, w, p)")
def tensor_conv(x, w, p):
    return eval_op(CONV, x, w, p)

@jit.dont_look_inside
def conv_call(x, w, p):
    return eval_op(CONV, x, w, p)

@jit.not_in_trace
def _ensure_conv(x, w, layout, c, h, wd, o, kh, kw, sh, sw, ph, pw, dh, dw,
                 g, n):
    conv_ensure(x, w, layout, c, h, wd, o, kh, kw, sh, sw, ph, pw, dh, dw, g,
                n)

def conv(x, w, layout, c, h, wd, o, kh, kw, sh, sw, ph, pw, dh, dw, g=1):
    if lazy.enabled():
        lazy.force(x)
        lazy.force(w)
    n = jit.promote(tensor_size(x) // (c * h * wd))
    layout = jit.promote(layout)
    c = jit.promote(c)
    h = jit.promote(h)
    wd = jit.promote(wd)
    o = jit.promote(o)
    kh = jit.promote(kh)
    kw = jit.promote(kw)
    sh = jit.promote(sh)
    sw = jit.promote(sw)
    ph = jit.promote(ph)
    pw = jit.promote(pw)
    dh = jit.promote(dh)
    dw = jit.promote(dw)
    g = jit.promote(g)
    _ensure_conv(x, w, layout, c, h, wd, o, kh, kw, sh, sw, ph, pw, dh, dw, g,
                 n)
    p = conv_choice(layout, c, h, wd, o, kh, kw, sh, sw, ph, pw, dh, dw, g, n)
    if p < 0:
        raise ValueError("convolution not prepared")
    if lazy.enabled() or conv_knob.off:
        return conv_call(x, w, p)
    return tensor_conv(x, w, p)

@jit.oopspec("tensor.assign(dst, src)")
def tensor_assign_op(dst, src):
    return tensor_assign(dst, src)

@jit.oopspec("tensor.ndim(a)")
def tensor_ndim(a):
    return len(a.shape)

@jit.oopspec("tensor.size(a)")
def tensor_size(a):
    return a.size

@jit.oopspec("tensor.shape(a, axis)")
def tensor_shape(a, axis):
    return a.shape[axis]

@jit.oopspec("tensor.dtype(a)")
def tensor_dtype(a):
    return a.dtype


def cols_of(t):
    c = tensor_shape(t, 1)
    if policy.static_cols:
        c = jit.promote(c)
        note_cols(c)
    return c

def size(t):
    n = tensor_size(t)
    if policy.static:
        n = jit.promote(n)
        note_size(n)
    return n

def bcast_of(big, small, m):
    if m == 1:
        return BC_R_SCALAR
    if tensor_ndim(big) == 2:
        rows = tensor_shape(big, 0)
        c = tensor_shape(big, 1)
        if m == rows and tensor_ndim(small) == 2 and tensor_shape(small, 1) == 1:
            cols_of(big)
            return BC_R_COL
        if m == c:
            return BC_R_ROW
        if m == rows:
            cols_of(big)
            return BC_R_COL
    elif tensor_ndim(big) > 2 and m == tensor_shape(big, tensor_ndim(big) - 1):
        return BC_R_ROW
    raise ValueError("shape mismatch")

def flip_bcast(p):
    if p == BC_R_ROW:
        return BC_L_ROW
    if p == BC_R_SCALAR:
        return BC_L_SCALAR
    return BC_L_COL

def bcast(a, b):
    if tensor_dtype(a) != tensor_dtype(b):
        raise ValueError("dtype mismatch")
    na = size(a)
    nb = size(b)
    if na == nb:
        return BC_NONE
    if na > nb:
        return bcast_of(a, b, nb)
    return flip_bcast(bcast_of(b, a, na))

# The deferred arm branches in plain wrappers, never inside the oopspec
# primitives, whose declared effect is asserted while building a chain reaches
# the compiler, allocator and collector. With the pass on, lazy.enabled() is a
# quasi-immutable constant and the branch folds away.

def add_p(a, b, p):
    if lazy.enabled():
        return lazy.lazy_op(ADD, a, b, p)
    return tensor_add(a, b, p)

def mul_p(a, b, p):
    if lazy.enabled():
        return lazy.lazy_op(MUL, a, b, p)
    return tensor_mul(a, b, p)

def sub_p(a, b, p):
    if lazy.enabled():
        return lazy.lazy_op(SUB, a, b, p)
    return tensor_sub(a, b, p)

def div_p(a, b, p):
    if lazy.enabled():
        return lazy.lazy_op(DIV, a, b, p)
    return tensor_div(a, b, p)

def relugrad_p(y, g, p):
    if lazy.enabled():
        return lazy.lazy_op(RELUGRAD, y, g, p)
    return tensor_relugrad(y, g, p)

def eqmask(a, b, p):
    if lazy.enabled():
        return lazy.lazy_op(EQMASK, a, b, p)
    return tensor_eqmask(a, b, p)

def relu_p(a):
    if lazy.enabled():
        return lazy.lazy_op(RELU, a, NULLTENSOR, 0)
    return tensor_relu(a)

def exp_p(a):
    if lazy.enabled():
        return lazy.lazy_op(EXP, a, NULLTENSOR, 0)
    return tensor_exp(a)

def sqrt_p(a):
    if lazy.enabled():
        return lazy.lazy_op(SQRT, a, NULLTENSOR, 0)
    return tensor_sqrt(a)

def unary_p(a, fn):
    if lazy.enabled():
        return lazy.lazy_op(UNARY, a, NULLTENSOR, fn)
    _ensure_single(UNARY, fn, a)
    return tensor_unary(a, fn)

def binary_p(a, b, p):
    if lazy.enabled():
        return lazy.lazy_op(BINARY, a, b, p)
    _ensure_single(BINARY, p, a)
    return tensor_binary(a, b, p)

def sum_p(a, axis):
    if lazy.enabled():
        return lazy.lazy_op(SUM, a, NULLTENSOR, axis)
    return tensor_sum(a, axis)

def maxr_p(a, axis):
    if lazy.enabled():
        return lazy.lazy_op(MAXR, a, NULLTENSOR, axis)
    return tensor_maxr(a, axis)

def add(a, b):
    return add_p(a, b, bcast(a, b))

def mul(a, b):
    return mul_p(a, b, bcast(a, b))

def add_(a, b):
    return assign(a, add(a, b))

def mul_(a, b):
    return assign(a, mul(a, b))

def relu(a):
    return relu_p(a)

def relugrad(y, g):
    return relugrad_p(y, g, bcast(y, g))

def sum(a, axis=AXIS_ALL):
    if axis == 1 and tensor_ndim(a) > 1:
        cols_of(a)
    return sum_p(a, axis)

def sub(a, b):
    return sub_p(a, b, bcast(a, b))

def div(a, b):
    return div_p(a, b, bcast(a, b))

def exp(a):
    return exp_p(a)

def sqrt(a):
    return sqrt_p(a)

def unary(a, fn):
    return unary_p(a, fn)

def binary(a, b, fn, p=-1):
    if p < 0:
        p = bcast(a, b)
    return binary_p(a, b, p + NPARAMS * fn)

def where(c, a, b):
    # a kept -0.0 comes out +0.0, from the add
    return add(binary(c, a, B_KEEP_NZ), binary(c, b, B_KEEP_Z))

def max(a, axis=AXIS_ALL):
    if axis == 1 and tensor_ndim(a) > 1:
        cols_of(a)
    return maxr_p(a, axis)

def matmul_shape(a, b):
    if tensor_ndim(a) != 2 or tensor_ndim(b) != 2:
        raise ValueError("shape mismatch")
    inner = tensor_shape(a, 1)
    if inner != tensor_shape(b, 0):
        raise ValueError("shape mismatch")
    return tensor_shape(a, 0), tensor_shape(b, 1), inner

def matmul(a, b, transpose_b=False):
    if lazy.enabled():
        lazy.force(a)
        lazy.force(b)
    if transpose_b:
        rows, cols, inner = matmul_shape_t(a, b)
        return tensor_matmul(a, b, rows, cols, inner, 0, 1)
    rows, cols, inner = matmul_shape(a, b)
    return tensor_matmul(a, b, rows, cols, inner, 0, 0)

def matmul_shape_t(a, b):
    if tensor_ndim(a) != 2 or tensor_ndim(b) != 2:
        raise ValueError("shape mismatch")
    inner = tensor_shape(a, 1)
    if inner != tensor_shape(b, 1):
        raise ValueError("shape mismatch")
    return tensor_shape(a, 0), tensor_shape(b, 0), inner

def assign(dst, src):
    if lazy.enabled():
        # Deferred nodes must read their operands before the write. Kept out of
        # the oopspec below because forcing reaches the collector and allocator.
        lazy.force(src)
        lazy.barrier()
    if (tensor_size(dst) != tensor_size(src) or
            tensor_dtype(dst) != tensor_dtype(src)):
        raise ValueError("shape mismatch")
    return tensor_assign_op(dst, src)

def item(a):
    if lazy.enabled():
        lazy.force(a)
    return tensor_item(a)

def tensor_output(t, k):
    return t.extra[k]

@jit.dont_look_inside
def tensor_force(a):
    if lazy.enabled():
        lazy.force(a)
    return a

@jit.elidable
def tensor_item(a):
    return host(a)[0]
