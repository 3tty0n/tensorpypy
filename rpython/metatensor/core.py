import os

from rpython.rlib import jit
from rpython.rlib.rfloat import INFINITY
from rpython.rtyper.lltypesystem import lltype
from rpython.rtyper.rclass import OBJECTPTR

HOSTARRAY = lltype.GcArray(lltype.Float)
SHAPEARRAY = lltype.GcArray(lltype.Signed)
TENSOR = lltype.GcForwardReference()
TENSORPTR = lltype.Ptr(TENSOR)
TENSORARRAY = lltype.GcArray(TENSORPTR)
TENSOR.become(lltype.GcStruct('TENSOR', ('size', lltype.Signed),
                              ('shape', lltype.Ptr(SHAPEARRAY)),
                              ('dptr', lltype.Signed),
                              ('host', lltype.Ptr(HOSTARRAY)),
                              ('extra', lltype.Ptr(TENSORARRAY)),
                              ('dtype', lltype.Signed),
                              ('buf', OBJECTPTR),
                              ('lazy', OBJECTPTR)))
NULLTENSOR = lltype.nullptr(TENSOR)

F64, F32, F16 = 0, 1, 2
NDTYPES = 3
DTYPE_BYTES = [8, 4, 2]
DTYPE_NAMES = ['float64', 'float32', 'float16']
STORE_TYPE = ['f64', 'f32', 'f16']
COMP_TYPE = ['f64', 'f32', 'f32']
COMP_NEG_INF = ['0xFFF0000000000000', '0xFF800000', '0xFF800000']

@jit.elidable
def dtype_of_name(name):
    for i in range(NDTYPES):
        if DTYPE_NAMES[i] == name:
            return i
    raise ValueError("unknown dtype")

def nbytes(n, dtype):
    return n * DTYPE_BYTES[dtype]

ADD, MUL, RELU, SUM, RELUGRAD = 0, 1, 2, 3, 4
SUB, DIV, EXP, SQRT, MAXR = 5, 6, 7, 8, 9
EQMASK = 10
# A permutation of its operand's elements. The param packs the kind and sizes
# (gather_param below).
GATHER = 11
# UNARY's param is a U_*. BINARY's param is bcast + NPARAMS * fn with fn a B_*,
# so p % NPARAMS is the BC_* of any binary node.
UNARY, BINARY = 12, 13
CONV = 14
NOPCODES = 15
ARITY = [2, 2, 1, 1, 2, 2, 2, 1, 1, 1, 2, 1, 1, 2, 2]
NAMES = ['add', 'mul', 'relu', 'sum', 'relugrad',
         'sub', 'div', 'exp', 'sqrt', 'maxr', 'eqmask', 'gather',
         'unary', 'binary', 'conv']
HAS_PARAM = [True, True, False, True, True,
             True, True, False, False, True, True, True, True, True, True]
U_TANH, U_SIGMOID, U_LOG, U_ABS, U_SIN, U_COS, U_ERF, U_FLOOR, U_NEG, \
    U_GELU, U_SILU = range(11)
UNARY_NAMES = ['tanh', 'sigmoid', 'log', 'abs', 'sin', 'cos', 'erf', 'floor',
               'neg', 'gelu_erf', 'silu']
B_MAX, B_MIN, B_POW, B_LT, B_LE, B_GT, B_GE, B_EQ, B_NE = range(9)
# keep(c, x): x where c != 0 (KEEP_NZ) or c == 0 (KEEP_Z), else 0. A select, so
# inf or NaN in dropped lanes stays dropped. where(c, a, b) = keep_nz(c, a) + keep_z(c, b).
B_KEEP_NZ, B_KEEP_Z = 9, 10
BINARY_NAMES = ['maximum', 'minimum', 'pow', 'lt', 'le', 'gt', 'ge', 'eq',
                'ne', 'keep_nz', 'keep_z']
# Compile-time ceiling on a fused kernel's input slots (the tensor.launch
# oopspec has that many tensor arguments). The runtime limit is max_inputs().
MAX_INPUTS_LIMIT = 16
DEFAULT_MAX_INPUTS = 16
MAX_EXTRA_OUTPUTS = 30

class _Knobs(object):
    max_inputs = 0
_knobs = _Knobs()

class _Tracing(object):
    pressure = False
tracing = _Tracing()

class _LazyKnob(object):
    # Quasi-immutable, so with the pass on the check costs one
    # guard_not_invalidated per trace.
    _immutable_fields_ = ['on?']
    on = False
lazy_knob = _LazyKnob()

class _DrainKnob(object):
    # How a value leaves an already launched fused region.
    #   off  keep the descriptor: the kernel gains one output and only the
    #        signature is recompiled.
    #   on   drain to canonical form: walk the graph back to the leaves and
    #        record a fresh kernel.
    # Read once at device init, so the optimizer's branch is a constant.
    _immutable_fields_ = ['on?']
    on = False
drain_knob = _DrainKnob()

class _GatherKnob(object):
    # METATENSOR_NO_GATHER_FUSION=1: rot_half, head_split and head_merge run
    # as standalone gather kernels (ablation of the GATHER fusion node).
    _immutable_fields_ = ['off?']
    off = False
gather_knob = _GatherKnob()

class _ConvKnob(object):
    _immutable_fields_ = ['off?']
    off = False
conv_knob = _ConvKnob()

def init_conv():
    value = os.environ.get('METATENSOR_NO_CONV_FUSION')
    conv_knob.off = value is not None and value != '' and value != '0'

def init_gather():
    value = os.environ.get('METATENSOR_NO_GATHER_FUSION')
    gather_knob.off = value is not None and value != '' and value != '0'

def init_lazy():
    value = os.environ.get('METATENSOR_LAZY')
    lazy_knob.on = value is not None and value != '' and value != '0'

class _SplitKnob(object):
    mask = 0
split_knob = _SplitKnob()
SPLIT_BRANCH = 1
SPLIT_EXC = 2

def init_split():
    value = os.environ.get('METATENSOR_SPLIT')
    mask = 0
    if value is not None and len(value) > 0:
        try:
            mask = int(value)
        except ValueError:
            mask = 0
    split_knob.mask = mask

def init_drain():
    value = os.environ.get('METATENSOR_DRAIN')
    drain_knob.on = value is not None and value != '' and value != '0'

def max_inputs():
    """Leaves a fused kernel may take, from RTENSOR_MAX_INPUTS (4..16, default 16)."""
    n = _knobs.max_inputs
    if n != 0:
        return n
    n = DEFAULT_MAX_INPUTS
    value = os.environ.get('RTENSOR_MAX_INPUTS')
    if value is not None and len(value) > 0:
        try:
            n = int(value)
        except ValueError:
            n = DEFAULT_MAX_INPUTS
        if n < 4:
            n = 4
        elif n > MAX_INPUTS_LIMIT:
            n = MAX_INPUTS_LIMIT
    _knobs.max_inputs = n
    return n

BC_NONE, BC_R_ROW, BC_R_SCALAR, BC_L_ROW, BC_L_SCALAR = 0, 1, 2, 3, 4
BC_R_COL, BC_L_COL = 5, 6
NPARAMS = 7

def bc_mode(opcode, p):
    if ARITY[opcode] != 2 or opcode == CONV:
        return BC_NONE
    return p % NPARAMS

def binary_fn(p):
    return p // NPARAMS
AXIS_ALL = -1
NEG_INF = -INFINITY

NODE = lltype.Struct('TENSOR_NODE', ('opcode', lltype.Signed),
                     ('a', lltype.Signed), ('b', lltype.Signed),
                     ('p', lltype.Signed))
NODEARRAY = lltype.GcArray(NODE)
KERNEL = lltype.GcStruct('TENSOR_KERNEL', ('ninputs', lltype.Signed),
                         ('nodes', lltype.Ptr(NODEARRAY)),
                         ('fn', lltype.Signed),
                         ('sumroot', lltype.Signed),
                         ('rowmode', lltype.Signed),
                         ('threads', lltype.Signed),
                         ('shared', lltype.Signed),
                         ('nextra', lltype.Signed),
                         ('n', lltype.Signed),
                         ('cols', lltype.Signed),
                         ('dtype', lltype.Signed),
                         ('modes', lltype.Signed),
                         ('outmodes', lltype.Signed),
                         ('nouts', lltype.Signed),
                         ('consts', lltype.Ptr(HOSTARRAY)),
                         ('outputs', lltype.Ptr(SHAPEARRAY)),
                         ('wfn', lltype.Signed),
                         ('wthreads', lltype.Signed),
                         ('wshared', lltype.Signed),
                         ('wnextra', lltype.Signed),
                         ('narrow', lltype.Ptr(SHAPEARRAY)),
                         ('vsrc', lltype.Signed),
                         ('vroot', lltype.Signed))
KERNELPTR = lltype.Ptr(KERNEL)
NO_CONSTS = lltype.malloc(HOSTARRAY, 0, immortal=True)
NO_NARROW = lltype.malloc(SHAPEARRAY, 0, immortal=True)
NO_SHAPE = lltype.nullptr(SHAPEARRAY)

def nvals(kernel):
    """Scalar leaves share the inputs' value-index space, after them, but are literals in generated code."""
    return kernel.ninputs + len(kernel.consts)

def _shape1(n):
    shape = lltype.malloc(SHAPEARRAY, 1)
    shape[0] = n
    return shape

def new_tensor(n, shape=lltype.nullptr(SHAPEARRAY), dtype=F64):
    t = lltype.malloc(TENSOR)
    t.size = n
    t.shape = shape if shape else _shape1(n)
    t.dptr = 0
    t.host = lltype.malloc(HOSTARRAY, n)
    t.extra = lltype.nullptr(TENSORARRAY)
    t.dtype = dtype
    t.buf = lltype.nullptr(OBJECTPTR.TO)
    t.lazy = lltype.nullptr(OBJECTPTR.TO)
    return t

def zeros(shape_list, dtype=F64):
    n = 1
    for d in shape_list:
        n *= d
    shape = lltype.malloc(SHAPEARRAY, len(shape_list))
    for i in range(len(shape_list)):
        shape[i] = shape_list[i]
    note_dtype(dtype)
    return new_tensor(n, shape, dtype)

def from_list(values, dtype=F64):
    note_dtype(dtype)
    t = new_tensor(len(values), lltype.nullptr(SHAPEARRAY), dtype)
    for i in range(len(values)):
        t.host[i] = values[i]
    return t

def cols(t):
    nd = len(t.shape)
    if nd > 1:
        c = t.shape[nd - 1]
        if c > 0:
            return c
    return 1

def is_reduction(opcode):
    return opcode == SUM or opcode == MAXR

def param_slot(opcode, p):
    if is_reduction(opcode):
        if p == 0 or p == 1:
            return p + 1
        return 0
    if not HAS_PARAM[opcode]:
        return 0
    if p > 0 and p < NPARAMS:
        return p
    return 0

def slot_param(opcode, slot):
    if is_reduction(opcode):
        return slot - 1
    return slot

def slot_used(opcode, slot):
    if (opcode == GATHER or opcode == UNARY or opcode == BINARY or
            opcode == CONV):
        return False
    if is_reduction(opcode):
        return slot < 3
    if not HAS_PARAM[opcode]:
        return slot == 0
    return True

class SizePolicy(object):
    _immutable_fields_ = ['static?', 'static_cols?', 'dtype?']
    def __init__(self):
        self.static = True
        self.seen = []
        self.static_cols = True
        self.seen_cols = []
        self.dtype = F64
policy = SizePolicy()

def note_dtype(dtype):
    if policy.dtype != dtype:
        policy.dtype = dtype
MAX_STATIC_SIZES = 3

class _StaticSizes(object):
    limit = MAX_STATIC_SIZES
static_sizes = _StaticSizes()

def init_static_sizes():
    value = os.environ.get('RTENSOR_STATIC_SIZES')
    n = MAX_STATIC_SIZES
    if value is not None and len(value) > 0:
        try:
            n = int(value)
        except ValueError:
            n = MAX_STATIC_SIZES
    static_sizes.limit = n

@jit.elidable
def note_size(n):
    if n not in policy.seen:
        policy.seen.append(n)
        if len(policy.seen) > static_sizes.limit:
            policy.static = False
    return n

@jit.elidable
def note_cols(c):
    if c not in policy.seen_cols:
        policy.seen_cols.append(c)
        if len(policy.seen_cols) > static_sizes.limit:
            policy.static_cols = False
    return c

class Config(object):
    block = 4096
    wide = 8192
    wide_warps = 16
    flat = 4096
    num_warps = 8
    cc = None
    debug_unfused = False
    narrow = True
    conv_prec = 'auto'
    spread = True
config = Config()
GA_IM2COL, GA_COL2CHW, GA_MAXPOOL = 0, 1, 2
GA_HEADSPLIT, GA_HEADMERGE = 3, 4
GA_ROWS = 5
GA_ROTHALF = 6
GA_IM2COL_NHWC = 7
GA_MAXPOOL_NHWC = 8
# Standalone only, not GATHER node kinds.
GA_STRIDED = 9
GA_TAKE = 10
GA_POOL = 11
GA_IM2COL2 = 12
GA_IM2COL_T = 13
GA_VIEW = 14
GA_SCATTER = 15
POOL_AVG, POOL_ADAPTIVE, POOL_DEPTHWISE = 0, 1, 2

def conv_out(h, k, stride, pad, dil):
    return (h + 2 * pad - dil * (k - 1) - 1) // stride + 1

CD_O, CD_ROWS, CD_G = 4, 15, 16
CD_BM, CD_BN, CD_BK, CD_GM, CD_WARPS, CD_STAGES = 17, 18, 19, 20, 21, 22

class ConvDescs(object):
    def __init__(self):
        self.params = [[0] * (CD_STAGES + 1)]
        self.index = {}
        self.static = {}
conv_descs = ConvDescs()

def conv_desc(params):
    key = ','.join([str(v) for v in params])
    p = conv_descs.index.get(key, -1)
    if p < 0:
        p = len(conv_descs.params)
        conv_descs.params.append(params)
        conv_descs.index[key] = p
    return p

def conv_geometry(layout, c, h, w, o, kh, kw, sh, sw, ph, pw, dh, dw, g, n,
                  cfg):
    oh = conv_out(h, kh, sh, ph, dh)
    ow = conv_out(w, kw, sw, pw, dw)
    params = [layout, c, h, w, o, kh, kw, sh, sw, ph, pw, dh, dw, oh, ow,
              n * oh * ow, g]
    for v in cfg:
        params.append(v)
    return params

def has_conv(kernel):
    for k in range(len(kernel.nodes)):
        if kernel.nodes[k].opcode == CONV:
            return True
    return False

def conv_work(d):
    return d[5] * d[6] * (d[1] // d[CD_G])

def main_conv(kernel):
    best = -1
    for k in range(len(kernel.nodes)):
        if kernel.nodes[k].opcode == CONV and (best < 0 or conv_work(
                conv_descs.params[kernel.nodes[k].p]) > conv_work(
                conv_descs.params[kernel.nodes[best].p])):
            best = k
    return best

def conv_grouped_dot(d):
    g = d[CD_G]
    return g > 1 and (d[1] // g) % 16 == 0 and d[CD_O] // g >= 16

def conv_col_tiles(d):
    o, bn, g = d[CD_O], d[CD_BN], d[CD_G]
    if conv_grouped_dot(d):
        return g * ((o // g + bn - 1) // bn)
    return (o + bn - 1) // bn

def conv_per_image(p):
    d = conv_descs.params[p]
    ohw = d[13] * d[14]
    if d[CD_ROWS] == ohw:
        return p
    q = d[:]
    q[CD_ROWS] = ohw
    return conv_desc(q)

def conv_programs(d):
    m, bm = d[CD_ROWS], d[CD_BM]
    return ((m + bm - 1) // bm) * conv_col_tiles(d)

# kind in the low 4 bits, dh in the next 20, heads above. rows (size // (dh * heads))
# is not stored, so there is one kernel per layout rather than per sequence length.
def gather_param(kind, dh, heads):
    return kind + ((dh + (heads << 20)) << 4)

def gather_kind(p):
    return p & 15

def gather_dh(p):
    return (p >> 4) & 0xFFFFF

def gather_heads(p):
    return p >> 24

def gather_changes_shape(p):
    k = gather_kind(p)
    return k == GA_HEADSPLIT or k == GA_HEADMERGE or k == GA_VIEW

def gather_shape(p, n, like):
    k = gather_kind(p)
    dh = gather_dh(p)
    heads = gather_heads(p)
    if k == GA_HEADSPLIT:
        return _shape2(n // dh, dh)
    if k == GA_HEADMERGE:
        return _shape2(n // (heads * dh), heads * dh)
    if k == GA_VIEW:
        return view_shape(view_of(p))
    return like


VIEWTABLE = lltype.GcStruct('VIEWTABLE', ('n', lltype.Signed),
                            ('data', lltype.Ptr(SHAPEARRAY)))
VREC = 6
view_table = lltype.malloc(VIEWTABLE, immortal=True)
view_table.n = 0
view_table.data = lltype.malloc(SHAPEARRAY, 0, immortal=True)


class _Views(object):
    def __init__(self):
        self.ids = {}
        self.params = {}
        self.shapes = {}
views = _Views()


def _vrec(i, k):
    return view_table.data[VREC * i + k]


def _intern(parent, a, b, lo, hi):
    key = (parent, a, b, lo, hi)
    i = views.ids.get(key, -1)
    if i >= 0:
        return i
    t = view_table
    d = t.data
    for i in range(t.n):
        base = VREC * i
        if (d[base] == parent and d[base + 1] == a and d[base + 2] == b and
                d[base + 3] == lo and d[base + 4] == hi):
            views.ids[key] = i
            return i
    i = t.n
    if VREC * (i + 1) > len(d):
        grown = lltype.malloc(SHAPEARRAY, 2 * len(d) + VREC * 64)
        for k in range(VREC * i):
            grown[k] = d[k]
        t.data = grown
        d = grown
    base = VREC * i
    d[base] = parent
    d[base + 1] = a
    d[base + 2] = b
    d[base + 3] = lo
    d[base + 4] = hi
    d[base + 5] = 0 if parent < 0 else _vrec(parent, 5) + 1
    t.n = i + 1
    views.ids[key] = i
    return i


@jit.elidable
def view_root(offset):
    return _intern(-1, offset, 0, 0, 0)


@jit.elidable
def view_root_clamped(offset, extent):
    return _intern(-1, offset, extent, 0, 0)


def view_clamp(vid):
    root, chain = _view_chain(vid)
    return _vrec(root, 2)


@jit.elidable
def view_dim(vid, extent, stride, lo, hi):
    return _intern(vid, extent, stride, lo, hi)


def view_param(vid):
    return GA_VIEW + (vid << 4)


def view_of(p):
    return p >> 4


def view_rank(vid):
    return _vrec(vid, 5)


def _view_chain(vid):
    chain = []
    while _vrec(vid, 0) >= 0:
        chain.append(vid)
        vid = _vrec(vid, 0)
    chain.reverse()
    return vid, chain


def view_offset(vid):
    root, chain = _view_chain(vid)
    return _vrec(root, 1)


def view_bounded(vid):
    root, chain = _view_chain(vid)
    for v in chain:
        if _vrec(v, 3) > 0 or _vrec(v, 4) < _vrec(v, 1):
            return True
    return False


def view_spreads(vid):
    root, chain = _view_chain(vid)
    if _vrec(root, 2) > 0:
        inside = _vrec(root, 2)
    else:
        inside = 1
        for v in chain:
            if _vrec(v, 2) != 0:
                inside *= _vrec(v, 4) - _vrec(v, 3)
    return view_numel(vid) > 2 * inside


def view_numel(vid):
    root, chain = _view_chain(vid)
    n = 1
    for v in chain:
        n *= _vrec(v, 1)
    return n


def view_shape(vid):
    shape = views.shapes.get(vid, NO_SHAPE)
    if not shape:
        root, chain = _view_chain(vid)
        shape = lltype.malloc(SHAPEARRAY, len(chain))
        for j in range(len(chain)):
            shape[j] = _vrec(chain[j], 1)
        views.shapes[vid] = shape
    return shape


def view_params(vid):
    params = views.params.get(vid, NO_SHAPE)
    if params:
        return params
    root, chain = _view_chain(vid)
    ns, nt, nl, nh = [], [], [], []
    bounded = view_bounded(vid)
    for v in chain:
        e = _vrec(v, 1)
        st = _vrec(v, 2)
        if not bounded:
            if e == 1:
                continue
            k = len(ns) - 1
            if k >= 0 and nt[k] == st * e:
                ns[k] = ns[k] * e
                nt[k] = st
                nh[k] = ns[k]
                continue
        ns.append(e)
        nt.append(st)
        nl.append(_vrec(v, 3))
        nh.append(_vrec(v, 4))
    if not ns:
        ns.append(1)
        nt.append(0)
        nl.append(0)
        nh.append(1)
    rank = len(ns)
    params = lltype.malloc(SHAPEARRAY, 2 + 4 * rank)
    params[0] = 1
    params[1] = rank
    for j in range(rank):
        params[2 + j] = ns[j]
        params[2 + rank + j] = nt[j]
        params[2 + 2 * rank + j] = nl[j]
        params[2 + 3 * rank + j] = nh[j]
    views.params[vid] = params
    return params


def view_is_identity(vid):
    if view_offset(vid) != 0 or view_clamp(vid) != 0:
        return False
    params = view_params(vid)
    return params[1] == 1 and params[3] == 1 and not view_bounded(vid)


def view_params_list(vid):
    params = view_params(vid)
    return [params[i] for i in range(len(params))]


def _shape2(rows, cols):
    shape = lltype.malloc(SHAPEARRAY, 2)
    shape[0] = rows
    shape[1] = cols
    return shape


def column(rows, dtype=F64):
    return new_tensor(rows, _shape2(rows, 1), dtype)

