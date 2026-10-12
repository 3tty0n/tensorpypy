"""The part of torch's API the ported suite models use, over _metatensor.

A ported model's source is the upstream file with Python 2 syntax; it keeps
`import torch`, and the harness puts lib_pypy/tensorpypy/compat on sys.path
so that import lands here.  Every choice about how an op runs lives in this
package and applies to every model alike.

A Tensor is a _metatensor tensor holding its elements contiguously in
logical (row-major) order, plus the logical shape.  The engine's kernels see
2-D views of it; which 2-D view an op needs is decided here.  An op whose
operands do not fit a view the engine has raises NotImplementedError naming
the case, rather than computing something else.

Two refinements of that picture:

  perm   transpose/permute do not move data: the tensor keeps its physical
         (contiguous) storage and records, for each logical dim, which
         physical dim it is.  Ops that understand a permutation (attention
         reading q/k/v as [B, S, H, D] storage viewed [B, H, S, D]) take it
         as is; every other op asks for dense() and gets NotImplementedError
         if the permutation would need a copy the engine does not have.
  col    split() along the last dim does not copy either: each piece is a
         window of columns (offset, row stride, and how many trailing
         logical dims the window spans) into the same [rows, ld] storage.
         view() may regroup the window's dims, transpose may reorder them,
         and attention reads q/k/v straight out of a packed projection.
  host   integer tensors (token ids, positions, masks) are host lists: they
         are index data, a few thousand elements computed from shapes and
         inputs, and uploading them only where they index device memory
         keeps their arithmetic off the device.
"""
import os as _os

import _metatensor
try:
    from pypyjit import residual_call as _residual
except ImportError:
    def _residual(f, *args):
        return f(*args)

_float, _int = float, int
long_, bool_ = long, bool
_all, _any, _abs, _max, _min, _sum = all, any, abs, max, min, sum

float32 = "float32"
float64 = "float64"
float16 = "float16"
bfloat16 = "bfloat16"
half = float16
double = float64
float = float32
int64 = "int64"
int32 = "int32"
int8 = "int8"
uint8 = "uint8"
long = int64
int = int32
bool = "bool"
__version__ = "2.14.0"

_NHWC = (0, 3, 1, 2)
_HEADS = (0, 2, 1, 3)
_channels_last = not _os.environ.get("METATENSOR_NCHW")
_igemm = _channels_last and not _os.environ.get("METATENSOR_NO_IGEMM")


class _Flag(object):
    def __init__(self, value):
        self.allow_tf32 = value


class backends(object):
    class cuda(object):
        matmul = _Flag(False)

    class cudnn(object):
        allow_tf32 = True
        benchmark = False


def _scalar(v, dtype):
    # _metatensor.scalar promotes the value, so the trace sees a constant
    # and the fusion pass writes it into the kernel as a literal; a cache
    # here would hand the trace a dict lookup instead, and every constant
    # would take one of the fused kernel's few input slots
    return _metatensor.scalar(v, dtype)


_SHAPES = hasattr(_metatensor, "same_shape")


def _numel(shape):
    if _SHAPES:
        return _metatensor.numel(shape)
    n = 1
    for d in shape:
        n *= d
    return n


def _norm_dim(d, nd):
    return d + nd if d < 0 else d


class device(object):
    def __init__(self, type, index=None):
        self.type = type
        self.index = index

    def __str__(self):
        return self.type

    def __eq__(self, other):
        return str(self) == str(other)

    __hash__ = object.__hash__


_CUDA = device("cuda")


class Size(tuple):
    def numel(self):
        return _numel(self)


_new = object.__new__
_ROWS = hasattr(_metatensor.Tensor, "dim")
_INDEX = hasattr(_metatensor, "basic_index")
_INDEX2 = hasattr(_metatensor.Tensor, "index")
_ASSIGN = hasattr(_metatensor.Tensor, "assign")
_INPLACE = hasattr(_metatensor.Tensor, "place_into") and \
    not _os.environ.get("METATENSOR_NO_INPLACE")
_INPLACE_MIN = _int(_os.environ.get("METATENSOR_INPLACE_MIN", 1 << 17))


class Tensor(object):
    perm = host = col = None

    def __init__(self, raw, shape, perm=None, host=None, col=None):
        self.raw = raw
        # a plain tuple: comparing two Size (tuple subclass) instances is a
        # user-level __eq__ the JIT leaves as a residual call, and a residual
        # call that may force ends every fused region still open
        self.shape = shape if shape.__class__ is tuple else tuple(shape)
        if perm is not None or host is not None or col is not None:
            self.__class__ = _Laid
            self.perm = perm
            self.host = host
            self.col = col

    def _plain(self):
        if self.__class__ is not Tensor:
            self.perm = self.host = self.col = None
            self.__class__ = Tensor

    requires_grad = property(lambda self: False, lambda self, v: None)

    # -- metadata ---------------------------------------------------------
    @property
    def dtype(self):
        if self.host is not None:
            return int64
        return self.raw.dtype

    @property
    def device(self):
        return _CUDA

    @property
    def is_host(self):
        return self.host is not None

    def dense(self):
        """This tensor with its elements contiguous in logical order: itself,
        or one strided copy when it is a permutation or a column window."""
        if self.perm is None and self.col is None:
            return self
        st, off = _layout(self)
        return Tensor(self.raw.strided(self.shape, st, off),
                      self.shape)

    def physical_shape(self):
        if self.perm is None:
            return tuple(self.shape)
        out = [0] * len(self.shape)
        for i, p in enumerate(self.perm):
            out[p] = self.shape[i]
        return tuple(out)

    @property
    def ndim(self):
        return len(self.shape)

    def dim(self):
        return len(self.shape)

    def size(self, d=None):
        if d is None:
            return Size(self.shape)
        return self.shape[_norm_dim(d, len(self.shape))]

    def numel(self):
        return _numel(self.shape)

    def __len__(self):
        return self.shape[0]

    def __repr__(self):
        return "tensor(shape=%s, dtype=%s)" % (list(self.shape), self.dtype)

    # -- views ------------------------------------------------------------
    def view(self, *shape):
        if len(shape) == 1 and not isinstance(shape[0], (_int, long_)):
            shape = shape[0]
        n = _numel(self.shape)
        if _SHAPES:
            fit = _metatensor.infer_shape(shape, n)
            if fit is None:
                raise ValueError("shape %s is invalid for %d elements"
                                 % (list(shape), n))
            shape = fit
        else:
            k = -1
            rest = 1
            for i in range(len(shape)):
                if shape[i] == -1:
                    k = i
                else:
                    rest *= shape[i]
            if k >= 0:
                shape = list(shape)
                shape[k] = n // rest
                rest *= shape[k]
            if rest != n:
                raise ValueError("shape %s is invalid for %d elements"
                                 % (list(shape), n))
        shape = shape if shape.__class__ is tuple else tuple(shape)
        if self.host is not None:
            return Tensor(None, shape, host=self.host)
        if self.col is not None and self.perm is None:
            return self._window_view(shape)
        r = _nhwc_view(self, shape)
        if r is not None:
            return r
        return Tensor(self.dense().raw, shape)

    reshape = view

    def reshape_as(self, other):
        return self.view(other.shape)

    view_as = reshape_as

    def _window_view(self, shape):
        # the rows (every dim before the window) must stay the rows; the
        # window's own dims may be regrouped
        off, ld, ndw = self.col
        rows = _numel(self.shape[:len(self.shape) - ndw])
        acc, k = 1, len(shape)
        while k > 0 and acc < self.numel() // rows:
            k -= 1
            acc *= shape[k]
        if acc != self.numel() // rows or _numel(shape[:k]) != rows:
            raise NotImplementedError("view %s of a column window over %s"
                                      % (list(shape), list(self.shape)))
        return Tensor(self.raw, shape, col=(off, ld, len(shape) - k))

    def split(self, split_size, dim=0):
        nd = len(self.shape)
        dim = _norm_dim(dim, nd)
        n = self.shape[dim]
        sizes = [split_size] * ((n + split_size - 1) // split_size) \
            if isinstance(split_size, _int) else list(split_size)
        if isinstance(split_size, _int) and _sum(sizes) != n:
            sizes[-1] = n - split_size * (len(sizes) - 1)
        if self.host is not None or dim != nd - 1 or self.perm is not None \
                or self.col is not None:
            out, start = [], 0
            for w in sizes:
                out.append(self.narrow(dim, start, w))
                start += w
            return tuple(out)
        ld = self.shape[-1]
        if _sum(sizes) != ld:
            raise ValueError("split sizes %s for %d" % (sizes, ld))
        out, off = [], 0
        for w in sizes:
            out.append(Tensor(self.raw, self.shape[:-1] + (w,),
                              col=(off, ld, 1)))
            off += w
        return tuple(out)

    def transpose(self, a, b):
        nd = len(self.shape)
        a, b = _norm_dim(a, nd), _norm_dim(b, nd)
        order = list(range(nd))
        order[a], order[b] = order[b], order[a]
        return self.permute(*order)

    def permute(self, *order):
        if len(order) == 1 and isinstance(order[0], (tuple, list)):
            order = tuple(order[0])
        nd = len(self.shape)
        if _SHAPES and self.host is None:
            shape, perm = _metatensor.permute_layout(self.shape, self.perm,
                                                     order)
            if perm is not None and _same(perm, _NHWC):
                perm = _NHWC
            return Tensor(self.raw, shape, perm, col=self.col)
        order = [_norm_dim(d, nd) for d in order]
        if self.host is not None:
            return _host_permute(self, order)
        base = self.perm or tuple(range(nd))
        perm = tuple(base[d] for d in order)
        shape = tuple(self.shape[d] for d in order)
        if _same(perm, tuple(range(nd))):
            perm = None
        elif _same(perm, _NHWC):
            perm = _NHWC
        return Tensor(self.raw, shape, perm, col=self.col)

    def expand(self, *shape):
        if len(shape) == 1 and not isinstance(shape[0], (_int, long_)):
            shape = shape[0]
        nd = len(shape)
        k = nd - len(self.shape)
        if _SHAPES and k >= 0 and self.host is None:
            st, off = _layout(self)
            out, est = _metatensor.expand_layout(self.shape, st, shape)
            if k == 0 and _same(out, self.shape):
                return self
            return Tensor(self.raw.strided(out, est, off), out)
        out = [0] * nd
        same = k == 0
        for i in range(nd):
            s = shape[i]
            if i >= k:
                c = self.shape[i - k]
                if s == -1:
                    s = c
                elif s != c:
                    same = False
            out[i] = s
        out = tuple(out)
        if same:
            return self
        if self.host is not None:
            cur = (1,) * k + tuple(self.shape)
            if _same(out, cur):
                return self.view(out)
            if _numel(out) > _HOST_MAX:
                return _on_device(self).expand(out)
            return _host_expand(self, cur, out)
        st, off = _layout(self)
        est = [0] * nd
        for i in range(k, nd):
            if self.shape[i - k] != 1 or out[i] == 1:
                est[i] = st[i - k]
        return Tensor(self.raw.strided(out, est, off), out)

    def repeat(self, *reps):
        if len(reps) == 1 and isinstance(reps[0], (tuple, list)):
            reps = tuple(reps[0])
        cur = (1,) * (len(reps) - len(self.shape)) + tuple(self.shape)
        # repeat = expand over a new dim before each, then merge the pairs
        inter = []
        for r, c in zip(reps, cur):
            inter.extend([r, c])
        x = self.view(tuple(v for c in cur for v in (1, c)))
        return x.expand(tuple(inter)).view(
            tuple(r * c for r, c in zip(reps, cur)))

    def chunk(self, chunks, dim=0):
        n = self.shape[_norm_dim(dim, len(self.shape))]
        size = (n + chunks - 1) // chunks
        return self.split(size, dim)

    def unbind(self, dim=0):
        dim = _norm_dim(dim, len(self.shape))
        return tuple(self.select(dim, i) for i in range(self.shape[dim]))

    def select(self, dim, i):
        nd = len(self.shape)
        p = self.perm[_norm_dim(dim, nd)] if self.perm is not None else \
            _norm_dim(dim, nd)
        if self.host is None and self.col is None and \
                self.perm is not None and 0 < p < nd - 1:
            return _select_window(self, _norm_dim(dim, nd), p, i)
        idx = [slice(None)] * _norm_dim(dim, len(self.shape)) + [i]
        return self[tuple(idx)]

    def narrow(self, dim, start, length):
        idx = [slice(None)] * _norm_dim(dim, len(self.shape)) + \
            [slice(start, start + length)]
        return self[tuple(idx)]

    def expand_as(self, other):
        return self.expand(other.shape)

    def __getitem__(self, idx):
        if self.host is not None:
            if len(self.host) > _HOST_MAX:
                return _device_getitem(_on_device(self), idx)
            return _host_getitem(self, idx)
        return _device_getitem(self, idx)

    def flatten(self, start_dim=0, end_dim=-1):
        nd = len(self.shape)
        s = _norm_dim(start_dim, nd)
        e = _norm_dim(end_dim, nd)
        if nd == 0:
            return self.view(1)
        mid = _numel(self.shape[s:e + 1])
        return self.view(self.shape[:s] + (mid,) + self.shape[e + 1:])

    def contiguous(self):
        r = _nhwc_view(self, self.shape)
        if r is not None:
            return r
        return self.dense()

    def detach(self):
        return self

    def clone(self, memory_format=None):
        if self.host is not None:
            return _host(self.host, self.shape)
        return Tensor(self.raw, self.shape, self.perm, col=self.col)

    def index_select(self, dim, index):
        return index_select(self, dim, index)

    def tril(self, diagonal=0):
        return tril(self, diagonal)

    def triu(self, diagonal=0):
        return triu(self, diagonal)

    def to(self, *args, **kwargs):
        for a in args + tuple(kwargs.values()):
            if a in (float32, float64, float16) and a != self.dtype:
                if self.host is not None:
                    return from_flat([_float(v) for v in self.host],
                                     self.shape, a)
                return Tensor(self.dense().raw.astype(a), self.shape)
            if a in (int64, int32) and self.host is None:
                raise NotImplementedError("device tensor to integers")
        return self

    def float(self):
        return self.to(float32)

    def unsqueeze(self, d):
        d = _norm_dim(d, len(self.shape) + 1)
        return self.view(self.shape[:d] + (1,) + self.shape[d:])

    def squeeze(self, d=None):
        if d is None:
            return self.view(tuple(x for x in self.shape if x != 1))
        d = _norm_dim(d, len(self.shape))
        if self.shape[d] != 1:
            return self
        return self.view(self.shape[:d] + self.shape[d + 1:])

    # -- host -------------------------------------------------------------
    def tolist(self):
        if self.host is not None:
            return list(self.host)
        return self.dense().raw.tolist()

    def index_tensor(self, dtype):
        """The device copy of an index tensor, made once and kept in raw."""
        if self.raw is None or self.raw.dtype != dtype:
            self.raw = _metatensor._tensor_flat(
                [_float(v) for v in self.host], [len(self.host)], False, dtype)
        return self.raw

    def item(self):
        return self.raw.item()

    def force(self):
        return Tensor(self.raw.force(), self.shape)

    # -- elementwise ------------------------------------------------------
    # The common cases - two tensors laid out alike, or a tensor and a
    # number - go straight to the engine from the operator itself.  Every
    # Python call the tracer inlines costs frame bookkeeping in the trace,
    # and a long elementwise expression (pyhpc's equation of state is ~300
    # of them) has to fit in one trace for the fusion pass to see it whole.
    def __add__(self, other):
        if other.__class__ is Tensor:
            if other.shape is self.shape or _same(self.shape, other.shape):
                r = _new(Tensor)
                r.raw = self.raw.add(other.raw)
                r.shape = self.shape
                return r
        elif other.__class__ is _float or other.__class__ is _int:
            r = _new(Tensor)
            r.raw = self.raw.add(other)
            r.shape = self.shape
            return r
        return _binary(self, other, "add")

    __radd__ = __add__
    __iadd__ = __add__

    def __mul__(self, other):
        if other.__class__ is Tensor:
            if other.shape is self.shape or _same(self.shape, other.shape):
                r = _new(Tensor)
                r.raw = self.raw.mul(other.raw)
                r.shape = self.shape
                return r
        elif other.__class__ is _float or other.__class__ is _int:
            r = _new(Tensor)
            r.raw = self.raw.mul(other)
            r.shape = self.shape
            return r
        return _binary(self, other, "mul")

    __rmul__ = __mul__
    __imul__ = __mul__

    def __sub__(self, other):
        if other.__class__ is Tensor:
            if other.shape is self.shape or _same(self.shape, other.shape):
                r = _new(Tensor)
                r.raw = self.raw.sub(other.raw)
                r.shape = self.shape
                return r
        elif other.__class__ is _float or other.__class__ is _int:
            r = _new(Tensor)
            r.raw = self.raw.sub(other)
            r.shape = self.shape
            return r
        return _binary(self, other, "sub")

    __isub__ = __sub__

    def __rsub__(self, other):
        if other.__class__ is _float or other.__class__ is _int:
            r = _new(Tensor)
            r.raw = self.raw.mul(-1.0).add(other)
            r.shape = self.shape
            return r
        return _binary(_as_tensor(other, self.dtype), self, "sub")

    def __div__(self, other):
        if other.__class__ is Tensor:
            if other.shape is self.shape or _same(self.shape, other.shape):
                r = _new(Tensor)
                r.raw = self.raw.div(other.raw)
                r.shape = self.shape
                return r
        elif other.__class__ is _float or other.__class__ is _int:
            r = _new(Tensor)
            r.raw = self.raw.div(other)
            r.shape = self.shape
            return r
        return _binary(self, other, "div")

    __truediv__ = __div__
    __itruediv__ = __div__
    __idiv__ = __div__

    def __rdiv__(self, other):
        if other.__class__ is _float or other.__class__ is _int:
            r = _new(Tensor)
            r.raw = _metatensor.scalar(other, self.raw.dtype).div(self.raw)
            r.shape = self.shape
            return r
        return _binary(_as_tensor(other, self.dtype), self, "div")

    __rtruediv__ = __rdiv__

    def __neg__(self):
        return self * -1.0

    def _logical(self, other, host_op, dev_op):
        if self.host is not None and (not isinstance(other, Tensor) or
                                      other.host is not None):
            return _host_binary(self, other, host_op)
        return _binary(self, other, dev_op)

    def __and__(self, other):
        return self._logical(other, "bitwise_and", "mul")

    def __or__(self, other):
        return self._logical(other, "bitwise_or", "maximum")

    def __xor__(self, other):
        return self._logical(other, "bitwise_xor", "ne")

    __rand__, __ror__, __rxor__ = __and__, __or__, __xor__

    def logical_not(self):
        if self.host is not None:
            if len(self.host) > _HOST_MAX:
                return 1.0 - _on_device(self)
            return _host([_int(not v) for v in self.host], self.shape)
        return 1.0 - self

    __invert__ = logical_not

    def add(self, other, alpha=1):
        if alpha != 1:
            other = other * alpha
        return self + other

    def _become(self, r):
        self.raw, self.shape = r.raw, r.shape
        if r.__class__ is not Tensor:
            self.__class__ = r.__class__
            self.perm, self.host, self.col = r.perm, r.host, r.col
        else:
            self._plain()
        return self

    def add_(self, other, alpha=1):
        return self._become(self.add(other, alpha))

    def mul(self, other):
        return self * other

    def mul_(self, other):
        return self._become(self * other)

    def relu(self):
        return _like(self, self.raw.relu())

    def relu_(self):
        return self._become(self.relu())

    def exp(self):
        return _like(self, self.raw.exp())

    def _unary(self, name):
        return _like(self, self.raw.unary(name))

    def tanh(self):
        return self._unary("tanh")

    def sigmoid(self):
        return self._unary("sigmoid")

    def log(self):
        return self._unary("log")

    def abs(self):
        return self._unary("abs")

    __abs__ = abs

    def sin(self):
        return self._unary("sin")

    def cos(self):
        return self._unary("cos")

    def erf(self):
        return self._unary("erf")

    def floor(self):
        return self._unary("floor")

    def round(self):
        f = self.floor()
        d = self - f
        odd = f - (f * 0.5).floor() * 2.0
        return f + where(d > 0.5, 1.0, where(d == 0.5, odd, 0.0))

    def rsqrt(self):
        return 1.0 / self.sqrt()

    def reciprocal(self):
        return 1.0 / self

    def __pow__(self, other):
        return _binary(self, other, "pow")

    def __rpow__(self, other):
        return _binary(_as_tensor(other, self.dtype), self, "pow")

    pow = __pow__

    def maximum(self, other):
        return _binary(self, other, "maximum")

    def minimum(self, other):
        return _binary(self, other, "minimum")

    def __lt__(self, other):
        return _binary(self, other, "lt")

    def __le__(self, other):
        return _binary(self, other, "le")

    def __gt__(self, other):
        return _binary(self, other, "gt")

    def __ge__(self, other):
        return _binary(self, other, "ge")

    def eq(self, other):
        return _binary(self, other, "eq")

    def ne(self, other):
        return _binary(self, other, "ne")

    __eq__ = eq
    __ne__ = ne
    __hash__ = object.__hash__

    # -- index-tensor reductions and casts ----------------------------------
    def _host_only(self, what):
        if self.host is None:
            raise NotImplementedError("%s of a device tensor" % what)

    def all(self):
        self._host_only("all")
        return _all(self.host)

    def any(self):
        self._host_only("any")
        return _any(self.host)

    def _as_index(self, what):
        """A device tensor used as index data (a mask cumsum'd into
        positions, a count turned into ints): read to the host, where index
        data lives, at the cost of one sync."""
        if self.host is not None:
            return self
        return _host([_int(v) for v in self.dense().tolist()], self.shape)

    def cumsum(self, dim, dtype=None):
        if self.host is None:
            return self._as_index("cumsum").cumsum(dim, dtype)
        nd = len(self.shape)
        dim = _norm_dim(dim, nd)
        st = _strides(self.shape)
        out = list(self.host)
        n = self.shape[dim]
        for flat in range(len(out)):
            if (flat // st[dim]) % n:
                out[flat] += out[flat - st[dim]]
        return _host(out, self.shape)

    def masked_fill(self, mask, value):
        """value where mask is true: a select, so what it replaces never
        leaks, as masked_fill's does not."""
        if mask.host is not None:
            m = Tensor(mask.index_tensor(self.dtype), mask.shape)
        else:
            m = mask
        if not _same(m.shape, self.shape):
            m = m.expand(self.shape)
        x = self.dense()
        return Tensor(m.dense().raw.where(_float(value), x.raw), x.shape)

    def masked_fill_(self, mask, value):
        return self._become(self.masked_fill(mask, value))

    def long(self):
        return self._as_index("long")

    int = long

    def type_as(self, other):
        if (self.host is None) == (other.host is None):
            return self if self.host is not None or \
                self.dtype == other.dtype else self.to(other.dtype)
        if other.host is not None:
            return self._as_index("type_as")
        return self.to(other.dtype)

    def clamp(self, min=None, max=None):
        x = self
        if min is not None:
            x = x.maximum(min)
        if max is not None:
            x = x.minimum(max)
        return x

    clamp_min = lambda self, v: self.clamp(min=v)
    clamp_max = lambda self, v: self.clamp(max=v)

    def sqrt(self):
        return _like(self, self.raw.sqrt())

    def mean(self, dim=None, keepdim=False):
        if dim is None:
            return Tensor(self.raw.sum(), ()) * (1.0 / self.numel())
        nd = len(self.shape)
        if _is_nhwc(self) and _spatial(dim):
            b, c = self.shape[0], self.shape[1]
            return Tensor(_spatial_mean(self), (b, c, 1, 1) if keepdim
                          else (b, c))
        n = _numel([self.shape[_norm_dim(d, nd)] for d in _dims(dim)])
        return _scale(_reduce_last(self, dim, keepdim, "sum"), 1.0 / n)

    def sum(self, dim=None, keepdim=False):
        if self.host is not None:
            return _host_sum(self, dim, keepdim)
        if dim is None:
            return Tensor(self.raw.sum(), ())
        return _reduce_last(self, dim, keepdim, "sum")

    def var(self, dim=None, unbiased=True, keepdim=False, correction=None):
        if correction is None:
            correction = 1 if unbiased else 0
        nd = len(self.shape)
        n = self.numel() if dim is None else _numel(
            [self.shape[_norm_dim(d, nd)] for d in _dims(dim)])
        d = self - self.mean(dim, keepdim=True) if dim is not None \
            else self - self.mean()
        return _scale((d * d).sum(dim, keepdim), 1.0 / (n - correction))

    def std(self, dim=None, unbiased=True, keepdim=False, correction=None):
        return self.var(dim, unbiased, keepdim, correction).sqrt()

    def norm(self, p="fro", dim=None, keepdim=False, dtype=None):
        if p not in (2, "fro"):
            raise NotImplementedError("%s-norm" % (p,))
        return (self * self).sum(dim, keepdim).sqrt()

    def __setitem__(self, idx, value):
        """Assignment into a box of the tensor (ints and step-1 slices) is
        done on the device: the value is padded out to the full shape and
        selected in where a padded mask of ones says so.  Anything else is
        done on the host and counted in host_writes, so a timed forward that
        needs it is rejected."""
        if self.host is not None:
            pos = _host_getitem(_host(list(range(len(self.host))),
                                      self.shape), idx)
            vals = value.expand(pos.shape).host if isinstance(value, Tensor) \
                else [value] * pos.numel()
            for i, v in zip(pos.host, vals):
                self.host[i] = v
            self.raw = None
            return
        if _ASSIGN and self.__class__ is Tensor:
            c = value.__class__
            if c is Tensor:
                r = self.raw.assign(self.shape, idx, value.raw, value.shape,
                                    False)
            elif c is _float or c is _int:
                r = self.raw.assign(self.shape, idx, value, None, False)
            else:
                r = None
            if r is not None:
                self.raw = r
                if _INPLACE and _numel(self.shape) >= _INPLACE_MIN:
                    self.__class__ = _Own
                return
        box = _metatensor.box_of(self.shape, idx) if _INDEX else \
            _box_of(self.shape, idx)
        if box is not None:
            starts, sizes, kept, full = box
            x = self if self.__class__ is Tensor else self.dense()
            if not isinstance(value, Tensor):
                if full:
                    r = _metatensor.zeros(x.shape, False, x.dtype)
                    self.raw = r if value == 0 else r.add(_float(value))
                    self._plain()
                    return
                value = Tensor(_scalar(_float(value), x.dtype), ())
            elif value.host is not None:
                value = Tensor(value.index_tensor(x.dtype), value.shape)
            if value.__class__ is Tensor and _same(value.shape, kept):
                v = value
            elif value.__class__ is Tensor and len(value.shape) == 0:
                v = Tensor(value.raw.strided(kept, [0] * len(kept), 0),
                           kept)
            else:
                v = value.expand(kept).dense()
            if full:
                self.raw = v.raw
                self._plain()
                return
            if _ROWS:
                self.raw = x.raw.place(v.raw, x.shape, starts, sizes)
                self._plain()
                if _INPLACE and _numel(x.shape) >= _INPLACE_MIN:
                    self.__class__ = _Own
                return
            nd = len(x.shape)
            v = v.view(tuple(sizes))
            pads = []
            for d in range(nd):
                pads += [starts[d], x.shape[d] - starts[d] - sizes[d]]
            placed = v.raw.pad(list(sizes), pads, 0.0)
            ones = (v * 0.0 + 1.0).dense().raw.pad(list(sizes), pads, 0.0)
            self.raw = ones.where(placed, x.raw)
            self._plain()
            return
        _counters["host_writes"] += 1
        x = self.dense()
        flat = x.tolist()
        pos = _host_getitem(_host(list(range(len(flat))), x.shape), idx)
        if isinstance(value, Tensor):
            vals = value.expand(pos.shape).dense().tolist()
        else:
            vals = [_float(value)] * pos.numel()
        for i, v in zip(pos.host, vals):
            flat[i] = v
        self.raw = from_flat(flat, x.shape, x.dtype).raw
        self._plain()

    def argmax(self, dim=None, keepdim=False):
        x = self if self.host is not None else \
            _host(self.dense().tolist(), self.shape)
        vals = x.host
        if dim is None:
            return _host([vals.index(_max(vals))], ())
        nd = len(x.shape)
        if _norm_dim(dim, nd) != nd - 1:
            return x.transpose(dim, -1).contiguous().argmax(-1, keepdim) \
                .transpose(dim, -1) if keepdim else \
                x.transpose(dim, -1).contiguous().argmax(-1)
        n = x.shape[-1]
        out = []
        for i in range(0, len(vals), n):
            row = vals[i:i + n]
            out.append(row.index(_max(row)))
        return _host(out, x.shape[:-1] + ((1,) if keepdim else ()))

    def amax(self, dim=-1, keepdim=False):
        return _reduce_last(self, dim, keepdim, "max")

    def max(self, dim=None, keepdim=False):
        if dim is not None:
            raise NotImplementedError("max over a dim (values and indices)")
        if self.host is not None:
            return _host([_max(self.host)], ())
        return Tensor(self.dense().raw.max(), ())

    def softmax(self, dim=-1):
        return _rowwise(self, dim, "softmax")

    def matmul(self, other):
        return matmul(self, other)

    __matmul__ = matmul


class _Laid(Tensor):

    def __add__(self, other):
        return _binary(self, other, "add")

    __radd__ = __iadd__ = __add__

    def __mul__(self, other):
        if (other.__class__ is _float or other.__class__ is _int) and \
                self.__class__ is _Laid and self.host is None and \
                _same(self.perm, _HEADS):
            return _Scaled(self, other)
        return _binary(self, other, "mul")

    __rmul__ = __imul__ = __mul__

    def __sub__(self, other):
        return _binary(self, other, "sub")

    __isub__ = __sub__

    def __rsub__(self, other):
        return _binary(_as_tensor(other, self.dtype), self, "sub")

    def __div__(self, other):
        return _binary(self, other, "div")

    __truediv__ = __itruediv__ = __idiv__ = __div__

    def __rdiv__(self, other):
        return _binary(_as_tensor(other, self.dtype), self, "div")

    __rtruediv__ = __rdiv__


def _own_get(self):
    self.__class__ = Tensor
    return self.raw


def _own_set(self, raw):
    self.__class__ = Tensor
    self.raw = raw


class _Own(Tensor):
    """A plain tensor whose buffer came out of its last box assignment and
    that no other tensor has been given: a later box assignment writes into
    that buffer.  Reading raw hands the buffer out, so it makes the tensor
    a plain Tensor again; only indexing that copies keeps it owned."""

    raw = property(_own_get, _own_set)

    def __getitem__(self, idx):
        self.__class__ = Tensor
        r = _device_getitem(self, idx)
        if r.__class__ is Tensor and r.raw is not self.raw and \
                _numel(r.shape) < _numel(self.shape):
            self.__class__ = _Own
        return r

    def __setitem__(self, idx, value):
        self.__class__ = Tensor
        c = value.__class__
        if c is Tensor:
            r = self.raw.assign(self.shape, idx, value.raw, value.shape, True)
        elif c is _float or c is _int:
            r = self.raw.assign(self.shape, idx, value, None, True)
        else:
            r = None
        if r is not None:
            self.raw = r
            self.__class__ = _Own
            return
        Tensor.__setitem__(self, idx, value)

    def __reduce_ex__(self, proto):
        self.__class__ = Tensor
        return Tensor.__reduce_ex__(self, proto)


class _Scaled(_Laid):
    """A head view times a number, held back: attention scores computed
    from it take the number as their scale instead of a pass over the
    projection it is a view of."""
    _val = None

    def __init__(self, base, s):
        self._base, self._s = base, s
        self.shape, self.perm, self.col = base.shape, base.perm, base.col

    raw = property(lambda self: self._value().raw)
    dtype = property(lambda self: self._base.dtype)

    def _value(self):
        v = self._val
        if v is None:
            v = self._val = _binary(self._base, self._s, "mul")
        return v

    def _become(self, r):
        self.__class__ = _Laid
        return Tensor._become(self, r)


class _Cstate(object):
    epoch = 0
    last = None


_cstate = _Cstate()


class _CNode(object):
    def __init__(self, fn, parents, args, kw, val):
        self.fn = fn
        self.parents = parents
        self.args = args
        self.kw = kw
        self.val = val
        self.epoch = _cstate.epoch
        self.kids = {}
        self.meta = None

    def value(self, keep):
        v = self.val
        if v is not None and (self.fn is None or
                              self.epoch == _cstate.epoch):
            return v
        last = _cstate.last
        if last is not None and last[0] is self and \
                last[2] == _cstate.epoch:
            v = last[1]
        else:
            v = self.fn(*([p.value(False) for p in self.parents] +
                          list(self.args)), **self.kw)
        if keep:
            self.val = v
            self.epoch = _cstate.epoch
        return v


def _frz(v):
    c = v.__class__
    if c is _Const:
        return v._node
    if c is tuple or c is list:
        out = []
        for e in v:
            k = _frz(e)
            if k is None and e is not None:
                return None
            out.append(k)
        return tuple(out)
    if c is slice:
        return (slice, v.start, v.stop, v.step)
    if isinstance(v, Tensor):
        return None
    return v


def _wrap_const(node):
    r = _new(_Const)
    r._node = node
    r.shape, r.perm, r.col, r._dtype = node.meta
    return r


def _cop(fn, parents, args, kw):
    key = _frz((args, sorted(kw.items())) if kw else args)
    if key is None:
        return None
    p0 = parents[0]
    key = (fn, parents[1:], key)
    node = p0.kids.get(key)
    if node is None:
        node = _residual(_cnew, fn, parents, args, kw, key)
        if node is None:
            return None
    return _wrap_const(node)


def _cnew(fn, parents, args, kw, key):
    node = _CNode(fn, parents, args, kw, None)
    v = node.value(False)
    if not isinstance(v, Tensor) or v.host is not None:
        return None
    node.meta = (v.shape, v.perm, v.col, v.dtype)
    parents[0].kids[key] = node
    _cstate.last = (node, v, _cstate.epoch)
    return node


def _param(t):
    if not isinstance(t, Tensor) or t.host is not None or \
            t.__class__ is _Const:
        return t
    node = _CNode(None, (), (), {}, Tensor(t.raw, t.shape, t.perm, col=t.col))
    node.meta = (t.shape, t.perm, t.col, t.dtype)
    return _wrap_const(node)


def _plain_value(t):
    return t._node.value(True) if t.__class__ is _Const else t


def _deferred(f):
    def m(self, *args, **kw):
        r = _cop(f, (self._node,), args, kw)
        if r is None:
            return f(_plain_value(self), *args, **kw)
        return r
    return m


def _const_binary(a, b, op):
    if a.__class__ is _Const and b.__class__ is _Const:
        return _cop(_binary, (a._node, b._node), (op,), {})
    if a.__class__ is _Const and not isinstance(b, Tensor):
        return _cop(_binary_num, (a._node,), (b, op), {})
    return None


def _binary_num(a, b, op):
    return _binary(a, b, op)


def _rbinary_num(b, a, op):
    return _binary(_as_tensor(a, b.dtype), b, op)


class _Const(_Laid):
    raw = property(lambda self: self._node.value(True).raw)
    dtype = property(lambda self: self._dtype)

    def __rsub__(self, other):
        if isinstance(other, Tensor):
            return _binary(other, self, "sub")
        return _cop(_rbinary_num, (self._node,), (other, "sub"), {})

    def __rdiv__(self, other):
        if isinstance(other, Tensor):
            return _binary(other, self, "div")
        return _cop(_rbinary_num, (self._node,), (other, "div"), {})

    __rtruediv__ = __rdiv__

    def dense(self):
        if self.perm is None and self.col is None:
            return self
        return self._dense()

    _dense = _deferred(Tensor.__dict__["dense"])

    def _become(self, r):
        v = _plain_value(r)
        node = self._node
        if node.fn is None:
            node.val = Tensor(v.raw, v.shape, v.perm, col=v.col)
            node.meta = (v.shape, v.perm, v.col, v.dtype)
            _cstate.epoch += 1
            self.shape, self.perm, self.col, self._dtype = node.meta
            return self
        self.__class__ = _Laid
        return Tensor._become(self, v)

    def __setitem__(self, idx, value):
        v = self._node.value(True)
        t = Tensor(v.raw, v.shape, v.perm, col=v.col)
        t.__setitem__(idx, _plain_value(value))
        self._become(t)


for _name in ("view", "permute", "expand", "__getitem__", "mean", "sum",
              "sqrt", "exp", "relu", "_unary", "to", "softmax", "amax"):
    setattr(_Const, _name, _deferred(Tensor.__dict__[_name]))
_Const.reshape = _Const.__dict__["view"]
del _name


def _like(t, raw):
    if t.__class__ is Tensor:
        r = _new(Tensor)
        r.raw = raw
        r.shape = t.shape
        return r
    return Tensor(raw, t.shape, t.perm, col=t.col)


def _nhwc_view(t, shape):
    nd = len(t.shape)
    k = len(shape)
    p = t.perm
    if p is None or t.col is not None or nd < 4 or k < 4 or \
            p[0] != 0 or p[nd - 2] != 1 or p[nd - 1] != 2 or \
            shape[0] != t.shape[0] or shape[k - 2] != t.shape[nd - 2] or \
            shape[k - 1] != t.shape[nd - 1]:
        return None
    st, off = _layout(t)
    order = [0, nd - 2, nd - 1] + list(range(1, nd - 2))
    dims = [t.shape[d] for d in order]
    sts = [st[d] for d in order]
    raw = t.raw
    if off != 0 or sts != _strides(dims):
        raw = raw.strided(dims, sts, off)
    perm = _NHWC if k == 4 else (0,) + tuple(range(3, k)) + (1, 2)
    return Tensor(raw, tuple(shape), perm)


def _layout(t):
    """Element strides of t's logical dims in its storage, and the offset of
    its first element."""
    if t.perm is None and t.col is None:
        return _strides(t.shape), 0
    base = t.physical_shape()
    nd = len(base)
    if t.col is None:
        st, off = _strides(base), 0
    else:
        off, ld, ndw = t.col
        st = [0] * nd
        acc = 1
        for d in range(nd - 1, nd - ndw - 1, -1):
            st[d] = acc
            acc *= base[d]
        acc = ld
        for d in range(nd - ndw - 1, -1, -1):
            st[d] = acc
            acc *= base[d]
    if t.perm is not None:
        st = [st[p] for p in t.perm]
    return list(st), off


def _select_window(x, d, p, i):
    """x.select(d, i) where logical dim d is physical dim p, 0 < p < last:
    the window of columns i*inner .. (i+1)*inner of the [rows, ld] view,
    rows being the physical dims before p."""
    phys = x.physical_shape()
    i = i + phys[p] if i < 0 else i
    if not 0 <= i < phys[p]:
        raise IndexError("index %d out of range for %d" % (i, phys[p]))
    inner = _numel(phys[p + 1:])
    perm = tuple(q - (q > p) for q in x.perm if q != p)
    if _same(perm, tuple(range(len(perm)))):
        perm = None
    return Tensor(x.raw, x.shape[:d] + x.shape[d + 1:], perm,
                  col=(i * inner, phys[p] * inner, len(phys) - 1 - p))


def _device_getitem(x, idx):
    """Basic indexing (ints, slices with steps, None, Ellipsis) of a device
    tensor: one strided copy, or x itself when nothing is selected away."""
    if _INDEX2 and x.__class__ is Tensor:
        r = x.raw.index(x.shape, idx)
        if r is not None:
            v, t = r
            if v is None:
                return x.view(t) if len(t) != len(x.shape) else x
            r = _new(Tensor)
            r.raw = v
            r.shape = t
            return r
    elif _INDEX and x.__class__ is Tensor:
        r = _metatensor.basic_index(x.shape, idx)
        if r is not None:
            shape, t, strides, off, identity = r
            if identity:
                return x.view(t) if len(t) != len(x.shape) else x
            return Tensor(x.raw.strided(shape, strides, off), t)
    items = idx if idx.__class__ is tuple else (idx,)
    nd = len(x.shape)
    n = len(items)
    ell = -1
    nnone = 0
    for i in range(n):
        it = items[i]
        c = it.__class__
        if c is Tensor or c is _Laid or c is _Own or c is list:
            return _advanced_getitem(x, items)
        if it is Ellipsis:
            ell = i
        elif it is None:
            nnone += 1
    if x.__class__ is Tensor:
        st = [1] * nd
        acc = 1
        for k in range(nd - 1, -1, -1):
            st[k] = acc
            acc *= x.shape[k]
        off = 0
    else:
        st, off = _layout(x)
    shape, strides, d, identity = [], [], 0, True
    for k in range(n):
        it = items[k]
        if it is None:
            shape.append(1)
            strides.append(0)
            continue
        if k == ell:
            e = d + nd - (n - 1 - nnone)
            while d < e:
                shape.append(x.shape[d])
                strides.append(st[d])
                d += 1
            continue
        if it.__class__ is slice:
            start, stop, step = it.indices(x.shape[d])
            if step == 1:
                m = stop - start if stop > start else 0
            else:
                m = (stop - start + (step - (1 if step > 0 else -1))) // step
                if m < 0:
                    m = 0
            if start != 0 or m != x.shape[d] or step != 1:
                identity = False
            off += start * st[d]
            shape.append(m)
            strides.append(st[d] * step)
        else:
            i = _int(it)
            i = i + x.shape[d] if i < 0 else i
            if not 0 <= i < x.shape[d]:
                raise IndexError("index %d out of range for %d" % (i, x.shape[d]))
            off += i * st[d]
            identity = False
        d += 1
    for k in range(d, nd):
        shape.append(x.shape[k])
        strides.append(st[k])
    if identity:
        return x.view(tuple(shape)) if len(shape) != nd else x
    if len(shape) == 4 and nd == 4 and _is_nhwc(x) and \
            _all(isinstance(it, slice) for it in items):
        return Tensor(x.raw.strided([shape[0], shape[2], shape[3], shape[1]],
                                    [strides[0], strides[2], strides[3],
                                     strides[1]], off), tuple(shape), _NHWC)
    return Tensor(x.raw.strided(shape, strides, off), canonical_shape(shape))


def _advanced_getitem(x, items):
    """x[:, ..., idx] with one index tensor on a leading or trailing axis:
    index_select."""
    pos = [i for i, it in enumerate(items) if not isinstance(it, slice)]
    if len(pos) > 1 and pos == list(range(len(pos))) and \
            _all(isinstance(items[i], Tensor) for i in pos) and \
            _all(it == slice(None) for it in items[len(pos):]):
        return _gather_leading(x, [items[i] for i in pos])
    if len(pos) != 1 or _any(it != slice(None) for i, it in enumerate(items)
                              if i != pos[0]):
        raise NotImplementedError("advanced indexing %r" % (items,))
    d = pos[0]
    ix = items[d]
    if isinstance(ix, list):
        ix = tensor(ix)
    return index_select(x, d, ix)


def _gather_leading(x, index):
    """x[i0, i1, ...] with one index tensor per leading dim: the rows at the
    flat positions of the leading dims, by one index_select."""
    shape = _broadcast_shape(index[0].shape, index[-1].shape)[0]
    for ix in index[1:-1]:
        shape = _broadcast_shape(shape, ix.shape)[0]
    k = len(index)
    flat = _host([0] * _numel(shape), shape)
    for d, ix in enumerate(index):
        flat = flat * x.shape[d] + ix._as_index("index").expand(shape)
    rest = x.shape[k:]
    return index_select(x.dense().view((_numel(x.shape[:k]),) + rest), 0,
                        flat.view(-1)).view(tuple(shape) + rest)


def index_select(x, dim, index):
    x = x.dense()
    dim = _norm_dim(dim, len(x.shape))
    outer = _numel(x.shape[:dim])
    inner = _numel(x.shape[dim + 1:])
    index = index._as_index("index_select")
    n = x.shape[dim]
    if _any(v < 0 for v in index.host):
        index = _host([v + n if v < 0 else v for v in index.host],
                      index.shape)
    r = x.raw.index_select(index.index_tensor(x.dtype), outer, inner)
    return Tensor(r, x.shape[:dim] + tuple(index.shape) + x.shape[dim + 1:])


def cat(tensors, dim=0):
    tensors = [t for t in tensors if t.numel() > 0] or list(tensors[:1])
    if _all(t.host is not None for t in tensors):
        dim = _norm_dim(dim, len(tensors[0].shape))
        outer = _numel(tensors[0].shape[:dim])
        vals = []
        for o in range(outer):
            for t in tensors:
                w = t.numel() // outer
                vals.extend(t.host[o * w:(o + 1) * w])
        shape = list(tensors[0].shape)
        shape[dim] = _sum(t.shape[dim] for t in tensors)
        return _host(vals, shape)
    dim = _norm_dim(dim, len(tensors[0].shape))
    if dim == 1 and len(tensors[0].shape) == 4 and \
            _any(_is_nhwc(t) for t in tensors):
        return _cat_nhwc(tensors)
    tensors = [t.dense() for t in tensors]
    outer = _numel(tensors[0].shape[:dim])
    shape = list(tensors[0].shape)
    shape[dim] = _sum(t.shape[dim] for t in tensors)
    return Tensor(_metatensor.cat([t.raw for t in tensors], outer),
                  tuple(shape))


concat = concatenate = cat


def _is_nhwc(t):
    return t.perm is _NHWC and t.col is None


def _to_nhwc(t):
    """t's elements in [N*H*W, C] storage: t's own, or one copy."""
    if _is_nhwc(t):
        return t.raw
    return t.permute(0, 2, 3, 1).dense().raw


def _spatial(dim):
    """dim names the two spatial dims of an NCHW tensor, (2, 3) or (-2, -1)
    in either order."""
    ds = _dims(dim)
    if len(ds) != 2:
        return False
    a, b = ds[0] % 4, ds[1] % 4
    return (a == 2 and b == 3) or (a == 3 and b == 2)


def _binary_nhwc(a, b, op):
    """a op b where one is channels-last and the other a per-channel or
    per-(image, channel) value ([C, 1, 1], [1, C, 1, 1], [N, C, 1, 1], a
    scalar): the result stays channels-last.  None when it is not that."""
    if _is_nhwc(a) and (not _is_nhwc(b) or
                        _numel(a.shape) >= _numel(b.shape)):
        big, small, flip = a, b, False
    else:
        big, small, flip = b, a, True
    if not _is_nhwc(big) or small.col is not None or len(small.shape) > 4:
        return None
    if small.perm is not None and not (
            _is_nhwc(small) and small.shape[2] == 1 and small.shape[3] == 1):
        return None
    n, c, h, w = big.shape
    ps = (1,) * (4 - len(small.shape)) + tuple(small.shape)
    if ps[2] != 1 or ps[3] != 1 or ps[0] not in (1, n) or \
            ps[1] not in (1, c):
        return None
    bt = big.raw
    if ps[0] == 1 or n == 1:
        st = small.raw.reshape([ps[1]]) if ps[1] == c else \
            small.raw.reshape([1])
    else:
        st = small.raw.strided([n, h * w, c],
                               [ps[1], 0, 1 if ps[1] == c else 0], 0)
    r = _apply(st, op, bt) if flip else _apply(bt, op, st)
    return Tensor(r, big.shape, _NHWC)


_mean_rows = {}


def _spatial_mean(x):
    """The per-(image, channel) mean of channels-last x as [N, C]: a batched
    GEMM of a 1/(H*W) row against each image's [H*W, C] storage, where the
    pooling kernel would sum H*W elements serially per output."""
    n, c, h, w = x.shape
    return _slab_sum(x.raw, n, h * w, c, x.dtype, 1.0 / (h * w))


def _slab_sum(raw, a, m, b, dtype, scale):
    key = (a, m, dtype, scale)
    ones = _mean_rows.get(key)
    if ones is None:
        ones = from_flat([scale] * (a * m), [a, m], dtype)
        _mean_rows[key] = ones
    return ones.raw.bmm(raw.reshape([a * m, b]), a)


def _cat_nhwc(tensors):
    n, _, h, w = tensors[0].shape
    c = _sum(t.shape[1] for t in tensors)
    return Tensor(_metatensor.cat([_to_nhwc(t) for t in tensors], n * h * w),
                  (n, c, h, w), _NHWC)


def stack(tensors, dim=0):
    dim = _norm_dim(dim, len(tensors[0].shape) + 1)
    return cat([t.unsqueeze(dim) for t in tensors], dim)


_counters = {"host_writes": 0}


def host_writes():
    return _counters["host_writes"]


def _box_of(shape, idx):
    """(starts, sizes, kept, full) of the box idx selects, when it is one:
    ints and step-1 slices, an Ellipsis standing for whole dims; else None.
    kept is the shape x[idx] has: sizes without the int-indexed dims, and
    full says the box is all of x."""
    items = idx if idx.__class__ is tuple else (idx,)
    nd = len(shape)
    n = len(items)
    if n > nd + 1:
        return None
    starts = [0] * nd
    sizes = list(shape)
    kept = []
    full = True
    d = 0
    for k in range(n):
        it = items[k]
        if it is Ellipsis:
            e = d + nd - n + 1
            while d < e:
                kept.append(shape[d])
                d += 1
            continue
        if d >= nd:
            return None
        if it.__class__ is slice:
            a, b, step = it.indices(shape[d])
            if step != 1:
                return None
            starts[d] = a
            sizes[d] = b - a if b > a else 0
            kept.append(sizes[d])
            if a != 0 or sizes[d] != shape[d]:
                full = False
        elif it.__class__ is _int or it.__class__ is long_:
            starts[d] = it + shape[d] if it < 0 else it
            sizes[d] = 1
            if shape[d] != 1:
                full = False
        else:
            return None
        d += 1
    while d < nd:
        kept.append(shape[d])
        d += 1
    return starts, sizes, tuple(kept), full


def _num(v, like):
    return _metatensor.scalar(_float(v), like.raw.dtype)


def _same(p, q):
    """p == q for two shapes or permutations, as an element loop: the tracer
    unrolls it into int compares, where tuple == would be a residual call
    that may force, and a forcing call closes every open fused region."""
    if p is q:
        return True
    if p is None or q is None:
        return False
    if _SHAPES:
        return _metatensor.same_shape(p, q)
    n = len(p)
    if n != len(q):
        return False
    if n == 3:
        return p[0] == q[0] and p[1] == q[1] and p[2] == q[2]
    i = 0
    while i < n:
        if p[i] != q[i]:
            return False
        i += 1
    return True


if _SHAPES:
    try:
        if not _metatensor.same_shape(None, ()):
            _same = _metatensor.same_shape
    except TypeError:
        pass


def _dims(dim):
    return list(dim) if isinstance(dim, (tuple, list)) else [dim]


def _reduce_last(x, dim, keepdim, how):
    """Reduce over trailing dimensions: a row reduction of the [rows, cols]
    view with cols = the product of the reduced dims."""
    if how == "sum" and _is_nhwc(x) and _spatial(dim):
        n, c, h, w = x.shape
        return Tensor(_slab_sum(x.raw, n, h * w, c, x.dtype, 1.0),
                      (n, c, 1, 1) if keepdim else (n, c))
    x = x.dense()
    nd = len(x.shape)
    ds = sorted(_norm_dim(d, nd) for d in _dims(dim))
    if ds != list(range(nd - len(ds), nd)) and \
            ds == list(range(ds[0], ds[-1] + 1)) and \
            _all(x.shape[d] == 1 for d in range(ds[-1] + 1, nd)):
        r = _reduce_last(x.view(x.shape[:ds[-1] + 1]), ds, keepdim, how)
        return r.view(r.shape + (1,) * (nd - 1 - ds[-1]))
    if ds != list(range(nd - len(ds), nd)):
        if how == "sum" and ds == list(range(ds[0], ds[-1] + 1)):
            lo, hi = ds[0], ds[-1] + 1
            r = _slab_sum(x.raw, _numel(x.shape[:lo]), _numel(x.shape[lo:hi]),
                          _numel(x.shape[hi:]), x.dtype, 1.0)
            return Tensor(r, x.shape[:lo] + (1,) * len(ds) + x.shape[hi:]
                          if keepdim else x.shape[:lo] + x.shape[hi:])
        rest = [d for d in range(nd) if d not in ds]
        r = _reduce_last(x.permute(*(rest + ds)).dense(),
                         list(range(len(rest), nd)), False, how)
        if keepdim:
            r = r.view(tuple(1 if d in ds else x.shape[d]
                             for d in range(nd)))
        return r
    cols = _numel(x.shape[nd - len(ds):])
    rows = x.numel() // cols
    t2 = functional._rows2(x.raw, rows, cols)
    r = t2.sum(1) if how == "sum" else t2.max(1)
    keep = x.shape[:nd - len(ds)]
    shape = keep + (1,) * len(ds) if keepdim else keep
    return Tensor(r, shape)


def _rowwise(x, dim, how):
    nd = len(x.shape)
    d = _norm_dim(dim, nd)
    if d != nd - 1:
        order = [i for i in range(nd) if i != d] + [d]
        back = [order.index(i) for i in range(nd)]
        return _rowwise(x.permute(*order), -1, how).permute(*back)
    x = x.dense()
    cols = x.shape[-1]
    t2 = functional._rows2(x.raw, x.numel() // cols, cols)
    return Tensor(getattr(t2, how)(), x.shape)


def _as_tensor(v, dtype):
    if isinstance(v, Tensor):
        return v
    return Tensor(_scalar(_float(v), dtype), ())


def _bshape(a, b):
    if _SHAPES:
        return _metatensor.broadcast_shape(a, b)
    return _broadcast_shape(a, b)[0]


def _broadcast_shape(a, b):
    n = len(a)
    if len(b) != n:
        n = _max(n, len(b))
        a = (1,) * (n - len(a)) + tuple(a)
        b = (1,) * (n - len(b)) + tuple(b)
    out = [0] * n
    for i in range(n):
        x = a[i]
        y = b[i]
        if x == y or y == 1:
            out[i] = x
        elif x == 1:
            out[i] = y
        else:
            raise ValueError("shapes %s and %s do not broadcast" % (a, b))
    return tuple(out), a, b


class _engine(object):
    """The engine call behind each binary op name.  _apply looks the name up
    as a class attribute, which the JIT folds for a constant name; a string
    comparison or a dict lookup would stay in every trace."""
    add = staticmethod(lambda x, y: x.add(y))
    mul = staticmethod(lambda x, y: x.mul(y))
    sub = staticmethod(lambda x, y: x.sub(y))
    div = staticmethod(lambda x, y: x.div(y))
    pow = staticmethod(lambda x, y: x.binary("pow", y))
    maximum = staticmethod(lambda x, y: x.binary("maximum", y))
    minimum = staticmethod(lambda x, y: x.binary("minimum", y))
    lt = staticmethod(lambda x, y: x.binary("lt", y))
    le = staticmethod(lambda x, y: x.binary("le", y))
    gt = staticmethod(lambda x, y: x.binary("gt", y))
    ge = staticmethod(lambda x, y: x.binary("ge", y))
    eq = staticmethod(lambda x, y: x.binary("eq", y))
    ne = staticmethod(lambda x, y: x.binary("ne", y))


class _bengine(object):
    add = staticmethod(lambda x, y, p: x.add_bc(y, p))
    mul = staticmethod(lambda x, y, p: x.mul_bc(y, p))
    sub = staticmethod(lambda x, y, p: x.sub_bc(y, p))
    div = staticmethod(lambda x, y, p: x.div_bc(y, p))
    pow = staticmethod(lambda x, y, p: x.binary_bc("pow", y, p))
    maximum = staticmethod(lambda x, y, p: x.binary_bc("maximum", y, p))
    minimum = staticmethod(lambda x, y, p: x.binary_bc("minimum", y, p))
    lt = staticmethod(lambda x, y, p: x.binary_bc("lt", y, p))
    le = staticmethod(lambda x, y, p: x.binary_bc("le", y, p))
    gt = staticmethod(lambda x, y, p: x.binary_bc("gt", y, p))
    ge = staticmethod(lambda x, y, p: x.binary_bc("ge", y, p))
    eq = staticmethod(lambda x, y, p: x.binary_bc("eq", y, p))
    ne = staticmethod(lambda x, y, p: x.binary_bc("ne", y, p))


_BC_R_ROW, _BC_R_SCALAR, _BC_L_ROW, _BC_R_COL, _BC_L_COL = 1, 2, 3, 5, 6
_BCAST = not _os.environ.get("METATENSOR_NO_EXPLICIT_BCAST")


def _scale(t, k):
    if _BCAST and t.__class__ is Tensor and t.host is None:
        return Tensor(t.raw.mul_bc(k, _BC_R_SCALAR), t.shape)
    return t * k


def _apply(x, op, y):
    return getattr(_engine, op)(x, y)


def _binary(a, b, op):
    if a.__class__ is _Const or b.__class__ is _Const:
        r = _const_binary(a, b, op)
        if r is not None:
            return r
    if not isinstance(b, Tensor):
        if a.host is not None:
            return _host_binary(a, b, op)
        return _like(a, getattr(_engine, op)(a.raw, _float(b)))
    if not isinstance(a, Tensor):
        a = _as_tensor(a, b.dtype)
    if a.host is not None or b.host is not None:
        return _host_binary(a, b, op)
    same = a.shape is b.shape or _same(a.shape, b.shape)
    if same and (a.perm is b.perm or _same(a.perm, b.perm)) and \
            (a.col is b.col or _same(a.col, b.col)):
        return _like(a, getattr(_engine, op)(a.raw, b.raw))
    if a.perm is _NHWC or b.perm is _NHWC:
        r = _binary_nhwc(a, b, op)
        if r is not None:
            return r
    if a.__class__ is not Tensor:
        a = a.dense()
    if b.__class__ is not Tensor:
        b = b.dense()
    if same:
        return Tensor(getattr(_engine, op)(a.raw, b.raw), a.shape)
    sa = a.shape
    sb = b.shape
    la = len(sa)
    lb = len(sb)
    nd = la if la > lb else lb
    out = _bshape(sa, sb)
    n = _numel(out)
    na = _numel(sa)
    nb = _numel(sb)
    if na == n and la == nd:
        shape = sa
    elif nb == n and lb == nd:
        shape = sb
    else:
        shape = out
    if na != n and nb != n:
        return Tensor(getattr(_engine, op)(_spread(a, shape),
                                           _spread(b, shape)), shape)
    if na == n and nb == n:
        return Tensor(getattr(_engine, op)(a.raw, b.raw), shape)
    if na == n:
        big, m, last = a, nb, sb[lb - 1] if lb else 1
    else:
        big, m, last = b, na, sa[la - 1] if la else 1
    if m == 1 or (_ROWS and last == m and big.raw.dim(-1) == m and
                  big.raw.dim(0) != m):
        return Tensor(getattr(_engine, op)(a.raw, b.raw), shape)
    small = b if na == n else a
    r = _block_broadcast(big, small, shape, n, m, op, small is a)
    if r is not None:
        return r
    if na == n:
        return Tensor(getattr(_engine, op)(a.raw, _spread(b, shape)), shape)
    return Tensor(getattr(_engine, op)(_spread(a, shape), b.raw), shape)


def _block_broadcast(big, small, shape, n, m, op, flip):
    """big op small when small's extents are a leading or trailing block of
    shape: a column or row broadcast over big as [m, n / m] or [n / m, m],
    instead of spreading small to the full size."""
    nd = len(shape)
    if _SHAPES:
        span = _metatensor.block_span(small.shape, shape)
        if span is None:
            return None
        lo, hi = span
    else:
        ps = (1,) * (nd - len(small.shape)) + tuple(small.shape)
        lo = 0
        while ps[lo] == 1:
            lo += 1
        hi = nd
        while ps[hi - 1] == 1:
            hi -= 1
        for i in range(lo, hi):
            if ps[i] != shape[i]:
                return None
    bt = big.raw
    if hi == nd:
        if bt.dim(-1) != m or bt.dim(0) * m != n:
            bt = bt.reshape([n // m, m])
        if _BCAST:
            r = getattr(_bengine, op)(small.raw, bt, _BC_L_ROW) if flip \
                else getattr(_bengine, op)(bt, small.raw, _BC_R_ROW)
            return Tensor(r, shape)
        st = small.raw.reshape([m])
    elif lo == 0:
        if bt.dim(0) != m or bt.dim(-1) * m != n:
            bt = bt.reshape([m, n // m])
        if _BCAST:
            r = getattr(_bengine, op)(small.raw, bt, _BC_L_COL) if flip \
                else getattr(_bengine, op)(bt, small.raw, _BC_R_COL)
            return Tensor(r, shape)
        st = small.raw.reshape([m, 1])
    else:
        return None
    r = getattr(_engine, op)(st, bt) if flip else getattr(_engine, op)(bt, st)
    return Tensor(r, shape)


def _spread(t, shape):
    nd = len(shape)
    if _SHAPES:
        return t.raw.strided(shape, _metatensor.spread_strides(t.shape, nd),
                             0)
    k = nd - len(t.shape)
    st = [0] * nd
    acc = 1
    for i in range(nd - 1, k - 1, -1):
        c = t.shape[i - k]
        if c != 1:
            st[i] = acc
        acc *= c
    return t.raw.strided(shape, st, 0)


def matmul(a, b):
    """a @ b: [..., n] @ [n, m] as one GEMM, and [..., s, n] @ [..., n, t]
    with equal batch dims as one batched GEMM."""
    if len(b.shape) > 2:
        if a.__class__ is _Attn:
            r = a._run(b)
            if r is not None:
                return r
        elif a.__class__ is _Scaled and len(b.shape) == 4 and \
                b.perm is not None and functional._FLASH and \
                b.host is None and _same(a.shape[:2], b.shape[:2]):
            return _Attn(a._base, b, a._s, None, False, 0, False, True)
        elif len(a.shape) == 4 and len(b.shape) == 4 and \
                b.perm is not None and functional._FLASH and \
                a.host is None and b.host is None and \
                _same(a.shape[:2], b.shape[:2]):
            return _Attn(a, b, 1.0, None, False)
        return _batched_matmul(a, b)
    a, b = a.dense(), b.dense()
    n = a.shape[-1]
    rows = a.numel() // n
    r = _as2d(a, rows, n).matmul(_as2d(b, n, b.shape[1]), False,
                                 backends.cuda.matmul.allow_tf32)
    return Tensor(r, a.shape[:-1] + (b.shape[1],))


def _batched_matmul(a, b):
    ba, bb = a.shape[:-2], b.shape[:-2]
    if not _same(ba, bb):
        shape = _broadcast_shape(ba, bb)[0]
        a = a.expand(shape + a.shape[-2:])
        b = b.expand(shape + b.shape[-2:])
        ba = shape
    batch = _numel(ba)
    s, n = a.shape[-2], a.shape[-1]
    t = b.shape[-1]
    at = a.dense().raw.reshape([batch * s, n])
    if b.perm is not None and b.col is None and len(b.perm) >= 2 and \
            b.perm[-1] == len(b.perm) - 2 and b.perm[-2] == len(b.perm) - 1 \
            and _all(b.perm[i] == i for i in range(len(b.perm) - 2)):
        r = at.bmm(b.raw.reshape([batch * t, n]), batch, True)
    else:
        r = at.bmm(b.dense().raw.reshape([batch * n, t]), batch)
    return Tensor(r, tuple(ba) + (s, t))


class _Attn(_Laid):
    """q @ kt, then scaled by a number, a mask added and softmax over the
    keys, held back as a description: the probabilities' matmul with v runs
    all of it as one fused attention kernel, as inductor's attention
    patterns do.  Any other use computes what the ops would have.  A view
    that splits the batch into [-1, per, ...] and back keeps it, and a mask
    added in the split view repeats every per batches (Swin's shifted
    windows)."""
    _val = None
    _per = 0
    _split = False
    _qs = False

    def __init__(self, q, kt, scale, mask, probs, per=0, split=False,
                 qs=False):
        self._q, self._kt, self._scale = q, kt, scale
        self._mask, self._probs = mask, probs
        if qs:
            self._qs = True
        self.shape = q.shape[:-1] + (kt.shape[-1],)
        if per:
            self._per = per
            if split:
                self._split = True
                s = self.shape
                self.shape = (s[0] // per, per) + s[1:]

    raw = property(lambda self: self._value().raw)
    dtype = property(lambda self: self._q.dtype)

    def _value(self):
        v = self._val
        if v is None:
            if self._qs:
                v = _batched_matmul(self._q * self._scale, self._kt)
            else:
                v = _batched_matmul(self._q, self._kt)
                if self._scale != 1.0:
                    v = v * self._scale
            if self._mask is not None:
                if self._per:
                    s = v.shape
                    v = (v.view((s[0] // self._per, self._per) + s[1:]) +
                         self._mask.unsqueeze(0)).view(s)
                else:
                    v = v + self._mask
            if self._probs:
                v = v.softmax(-1)
            if self._split:
                v = v.view(self.shape)
            v = v.dense()
            self._val = v
        return v

    def _open(self):
        return self._val is None and not self._probs

    def _with(self, scale, mask, probs, per, split):
        return _Attn(self._q, self._kt, scale, mask, probs, per, split,
                     self._qs)

    def __mul__(self, other):
        if (other.__class__ is _float or other.__class__ is _int) and \
                self._open() and self._mask is None:
            return self._with(self._scale * other, None, False, self._per,
                              self._split)
        return _Laid.__mul__(self, other)

    __rmul__ = __mul__

    def __div__(self, other):
        if (other.__class__ is _float or other.__class__ is _int) and \
                self._open() and self._mask is None:
            return self._with(self._scale / other, None, False, self._per,
                              self._split)
        return _Laid.__div__(self, other)

    __truediv__ = __div__

    def __add__(self, other):
        if isinstance(other, Tensor) and self._open():
            if not self._split and self._mask is None:
                return _Attn(self._q, self._kt, self._scale, other, False, 0,
                             False, self._qs)
            m = self._window_mask(other)
            if m is not None:
                return self._with(self._scale, m, False, self._per, True)
        return _Laid.__add__(self, other)

    __radd__ = __add__

    def _window_mask(self, other):
        s = self.shape
        if not self._split or len(other.shape) != 5 or \
                other.shape[0] != 1 or other.shape[1] != s[1] or \
                other.shape[3] != s[3] or other.shape[4] != s[4]:
            return None
        per, h = s[1], s[2]
        m = other.view(other.shape[1:])
        old = self._mask
        if old is not None:
            if old.shape[0] != per:
                return None
            m = old + m
        return m.expand(per, h, s[3], s[4])

    def view(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list, Size)):
            shape = tuple(shape[0])
        s = self.shape
        m = self._mask
        if self._open() and not self._split and not self._per and \
                len(shape) == 5 and len(s) == 4 and \
                tuple(shape[2:]) == s[1:] and shape[1] > 0 and \
                s[0] % shape[1] == 0 and \
                shape[0] in (-1, s[0] // shape[1]) and \
                (m is None or len(m.shape) == 4 and m.shape[0] == 1):
            per = shape[1]
            if m is not None:
                m = m.view(m.shape[1:]).unsqueeze(0).expand(
                    per, s[1], s[2], s[3])
            return self._with(self._scale, m, False, per, True)
        if self._open() and self._split and len(shape) == 4 and \
                tuple(shape[1:]) == s[2:] and \
                shape[0] in (-1, s[0] * s[1]):
            return self._with(self._scale, self._mask, False, self._per,
                              False)
        return Tensor.view(self, *shape)

    reshape = view

    def softmax(self, dim=-1):
        if (dim == -1 or dim == len(self.shape) - 1) and self._open():
            return self._with(self._scale, self._mask, True, self._per,
                              self._split)
        return Tensor.softmax(self, dim)

    def _run(self, v):
        if not self._probs or self._val is not None or self._split:
            return None
        return functional._flash(self._q, self._kt.transpose(-2, -1), v,
                                 self._mask, False, self._scale, self._per)

    def _become(self, r):
        self.__class__ = _Laid
        return Tensor._become(self, r)


def bmm(a, b):
    return _batched_matmul(a, b)


def _as2d(x, rows, cols):
    """x's storage as [rows, cols], without a new view when it already is
    one: a view of a weight not yet on the device uploads a copy of its own,
    and the weight, never uploaded itself, is copied again next time."""
    if len(x.shape) == 2 and x.shape[0] == rows:
        return x.raw
    return x.raw.reshape([rows, cols])


def relu(x, inplace=False):
    return x.relu()


def addmm(bias, a, b, beta=1, alpha=1):
    """bias + a @ b, for b in [in, out] as GPT-2's Conv1D keeps it."""
    if beta != 1 or alpha != 1:
        raise NotImplementedError("addmm with beta/alpha")
    if functional._GEMM_BIAS and len(a.shape) == 2 and len(b.shape) == 2 \
            and len(bias.shape) == 1 and bias.shape[0] == b.shape[1] and \
            a.dtype == "float32" and b.dtype == "float32" and \
            bias.dtype == "float32" and bias.host is None and \
            a.host is None and b.host is None and \
            bias.perm is None and bias.col is None:
        a, b = a.dense(), b.dense()
        r = functional._rows2(a.raw, a.shape[0], a.shape[1]).matmul_bias(
            functional._rows2(b.raw, b.shape[0], b.shape[1]), False,
            backends.cuda.matmul.allow_tf32, bias.raw)
        return Tensor(r, (a.shape[0], b.shape[1]))
    return matmul(a, b) + bias


def empty(*shape, **kw):
    return zeros(*shape, **kw)


def tanh(x):
    return x.tanh()


def sigmoid(x):
    return x.sigmoid()


def log(x):
    return x.log()


def exp(x):
    return x.exp()


def sqrt(x):
    return x.sqrt()


def abs(x):
    return x.abs()


def sin(x):
    return x.sin()


def cos(x):
    return x.cos()


def rsqrt(x):
    return x.rsqrt()


def reciprocal(x):
    return x.reciprocal()


def pow(x, y):
    return x ** y if isinstance(x, Tensor) else _as_tensor(x, y.dtype) ** y


def maximum(a, b):
    return a.maximum(b)


def minimum(a, b):
    return a.minimum(b)


def min(x, *args, **kw):
    if args and isinstance(args[0], Tensor):
        return x.minimum(args[0])
    return (-x).max(*args, **kw).__neg__()


def max(x, *args, **kw):
    if args and isinstance(args[0], Tensor):
        return x.maximum(args[0])
    return x.max(*args, **kw)


def bitwise_xor(a, b):
    return _host_binary(a, b, "bitwise_xor")


def bitwise_and(a, b):
    return _host_binary(a, b, "bitwise_and")


def bitwise_or(a, b):
    return _host_binary(a, b, "bitwise_or")


def logical_not(x):
    return x.logical_not()


def logical_and(a, b):
    return a & b


def logical_or(a, b):
    return a | b


def clone(x, memory_format=None):
    return x.clone()


def clamp(x, min=None, max=None):
    return x.clamp(min, max)


def _fits(t, shape):
    if t.__class__ is Tensor:
        return t.shape is shape or _same(t.shape, shape)
    return not isinstance(t, Tensor)


def _fits_nhwc(t, shape):
    if isinstance(t, Tensor):
        return _is_nhwc(t) and _same(t.shape, shape)
    return True


def where(cond, a, b):
    if not isinstance(a, Tensor) and not isinstance(b, Tensor):
        raise NotImplementedError("where with two scalars")
    s = cond.shape
    if cond.__class__ is Tensor and _fits(a, s) and _fits(b, s):
        r = _new(Tensor)
        r.raw = cond.raw.where(a.raw if a.__class__ is Tensor else _float(a),
                               b.raw if b.__class__ is Tensor else _float(b))
        r.shape = s
        return r
    if _is_nhwc(cond) and _fits_nhwc(a, s) and _fits_nhwc(b, s):
        return Tensor(cond.raw.where(
            a.raw if isinstance(a, Tensor) else _float(a),
            b.raw if isinstance(b, Tensor) else _float(b)), s, _NHWC)
    dt = a.dtype if isinstance(a, Tensor) else b.dtype
    a = _as_tensor(a, dt)
    b = _as_tensor(b, dt)
    shape = _bshape(cond.shape, _bshape(a.shape, b.shape))
    def fit(t):
        if t.host is not None:
            t = Tensor(t.index_tensor(dt), t.shape)
        if t.shape == ():
            return t
        if not _same(t.shape, shape):
            t = t.expand(shape)
        return t.dense()
    return Tensor(fit(cond).raw.where(fit(a).raw, fit(b).raw), shape)


def cumsum(x, dim, dtype=None):
    return x.cumsum(dim, dtype)


def round(x):
    return x.round()


_sorted = sorted


def unique(x, sorted=True, return_inverse=False, return_counts=False,
           dim=None):
    """The distinct values, ascending.  How many there are decides the
    result's shape, so a device tensor's values are read back here, as
    torch synchronises on them too."""
    if return_inverse or dim is not None:
        raise NotImplementedError("unique options")
    flat = x.tolist()
    vals = _sorted(set(flat))
    if x.host is not None:
        out = _host(vals, (len(vals),))
    else:
        out = from_flat([_float(v) for v in vals], (len(vals),), x.dtype)
    if return_counts:
        return out, _host([flat.count(v) for v in vals], (len(vals),))
    return out


def all(x):
    return x.all()


def any(x):
    return x.any()


def eq(a, b):
    return a.eq(b)


def ne(a, b):
    return a.ne(b)


def lt(a, b):
    return a < b


def gt(a, b):
    return a > b


def le(a, b):
    return a <= b


def ge(a, b):
    return a >= b


def mul(a, b):
    return a * b


def sub(a, b):
    return a - b


def div(a, b):
    return a / b


def get_default_dtype():
    return float32


def ones(*shape, **kw):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    dtype = kw.get("dtype") or float32
    if _is_int_dtype(dtype):
        return _host([1] * _numel(shape), shape)
    return zeros(*shape, dtype=dtype) + 1.0


def full(shape, fill_value, dtype=None, device=None):
    dtype = dtype or (int64 if isinstance(fill_value, _int) else float32)
    if _is_int_dtype(dtype):
        return _host([fill_value] * _numel(shape), shape)
    return zeros(*shape, dtype=dtype) + _float(fill_value)


def ones_like(x, dtype=None):
    return zeros_like(x, dtype) + 1.0


def _tri(x, keep):
    r, c = x.shape[-2:]
    m = [_int(keep(i, j)) for i in range(r) for j in range(c)]
    if x.host is not None:
        return _host([v if m[k % (r * c)] else 0
                      for k, v in enumerate(x.host)], x.shape)
    return x * _host(m, (r, c))


def tril(x, diagonal=0):
    return _tri(x, lambda i, j: j - i <= diagonal)


def triu(x, diagonal=0):
    return _tri(x, lambda i, j: j - i >= diagonal)


def zeros_like(x, dtype=None):
    return zeros(*x.shape, dtype=dtype or x.dtype)


def empty_like(x, dtype=None):
    return zeros_like(x, dtype)


def transpose(x, a, b):
    return x.transpose(a, b)


def permute(x, dims):
    return x.permute(*dims)


def flatten(x, start_dim=0, end_dim=-1):
    return x.flatten(start_dim, end_dim)


def add(a, b):
    return a + b


def mean(x, dim=None, keepdim=False):
    return x.mean(dim, keepdim)


def softmax(x, dim=-1):
    return x.softmax(dim)


_shapes = {}


def canonical_shape(shape):
    """One tuple object per distinct shape, so the elementwise fast path can
    compare shapes by identity: ops hand their operand's shape object on."""
    shape = tuple(shape)
    return _shapes.setdefault(shape, shape)


def from_flat(values, shape, dtype):
    return Tensor(_metatensor._tensor_flat(values, list(shape) or [1], False,
                                           dtype), canonical_shape(shape))


def zeros(*shape, **kw):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    if "size" in kw:
        shape = tuple(kw["size"])
    dtype = kw.get("dtype") or float32
    if _is_int_dtype(dtype):
        return _host([0] * _numel(shape), shape)
    return Tensor(_metatensor.zeros(shape, False, dtype), shape)


def gather(x, dim, index):
    """torch.gather.  On index tensors it runs on the host (the embedding-
    side lookups upstream does on its integer buffers); a device source
    with index data gathers along dim in one kernel."""
    if x.host is None and index.host is not None:
        x = x.dense()
        nd = len(x.shape)
        dim = _norm_dim(dim, nd)
        if _any(index.shape[i] != x.shape[i] for i in range(nd) if i != dim):
            raise NotImplementedError("gather with a smaller index")
        outer = _numel(x.shape[:dim])
        inner = _numel(x.shape[dim + 1:])
        r = x.raw.gather(index.index_tensor(x.dtype), outer, inner)
        return Tensor(r, tuple(index.shape))
    if x.host is None or index.host is None:
        raise NotImplementedError("gather on device tensors")
    nd = len(x.shape)
    dim = _norm_dim(dim, nd)
    xs, ist = _strides(x.shape), _strides(index.shape)
    out = []
    for flat in range(index.numel()):
        pos, rem = [], flat
        for s in ist:
            pos.append(rem // s)
            rem %= s
        pos[dim] = index.host[flat]
        out.append(x.host[sum(p * s for p, s in zip(pos, xs))])
    return _host(out, index.shape)


# -- index tensors on the host ---------------------------------------------

def _host(values, shape):
    return Tensor(None, tuple(shape), host=list(values))


def _strides(shape):
    if _SHAPES:
        return _metatensor.contiguous_strides(shape)
    n = len(shape)
    st = [1] * n
    acc = 1
    for i in range(n - 1, -1, -1):
        st[i] = acc
        acc *= shape[i]
    return st


def _host_getitem(x, idx):
    if not isinstance(idx, tuple):
        idx = (idx,)
    if Ellipsis in idx:
        i = idx.index(Ellipsis)
        fill = len(x.shape) - (len(idx) - 1 - idx.count(None))
        idx = idx[:i] + (slice(None),) * fill + idx[i + 1:]
    shape, src = list(x.shape), _strides(x.shape)
    axes = []                  # (start, step, count, stride) per output dim
    base, d = 0, 0
    out_shape = []
    for it in idx:
        if it is None:
            axes.append((0, 0, 1, 0))
            out_shape.append(1)
            continue
        if isinstance(it, slice):
            start, stop, step = it.indices(shape[d])
            n = _max(0, (stop - start + (step - (1 if step > 0 else -1)))
                    // step)
            axes.append((start, step, n, src[d]))
            out_shape.append(n)
        else:
            i = _int(it)
            base += (i + shape[d] if i < 0 else i) * src[d]
        d += 1
    for k in range(d, len(shape)):
        axes.append((0, 1, shape[k], src[k]))
        out_shape.append(shape[k])
    vals = []

    def walk(level, off):
        if level == len(axes):
            vals.append(x.host[off])
            return
        start, step, n, stride = axes[level]
        for j in range(n):
            walk(level + 1, off + (start + j * step) * stride)
    walk(0, base)
    return _host(vals, out_shape)


def _host_permute(x, order):
    st = _strides(x.shape)
    shape = [x.shape[d] for d in order]
    sst = [st[d] for d in order]
    vals = []

    def walk(level, off):
        if level == len(shape):
            vals.append(x.host[off])
            return
        for j in range(shape[level]):
            walk(level + 1, off + j * sst[level])
    walk(0, 0)
    return _host(vals, shape)


def _host_sum(x, dim, keepdim):
    if dim is None:
        return _host([_sum(x.host)], ())
    nd = len(x.shape)
    ds = sorted(set(_norm_dim(d, nd) for d in _dims(dim)))
    rest = [d for d in range(nd) if d not in ds]
    y = _host_permute(x, rest + ds)
    n = _numel([x.shape[d] for d in ds])
    vals = [_sum(y.host[i:i + n]) for i in range(0, len(y.host), n)]
    shape = [1 if d in ds else x.shape[d] for d in range(nd)] if keepdim \
        else [x.shape[d] for d in rest]
    return _host(vals, shape)


def meshgrid(*tensors, **kw):
    """torch.meshgrid of index tensors, 'ij' (and torch's default) or 'xy'
    indexing."""
    if len(tensors) == 1 and isinstance(tensors[0], (tuple, list)):
        tensors = tuple(tensors[0])
    xy = kw.get("indexing", "ij") == "xy" and len(tensors) >= 2
    if xy:
        tensors = (tensors[1], tensors[0]) + tuple(tensors[2:])
    shape = tuple(t.numel() for t in tensors)
    out = []
    for i, t in enumerate(tensors):
        if t.host is None:
            raise NotImplementedError("meshgrid of device tensors")
        cur = tuple(shape[i] if j == i else 1 for j in range(len(shape)))
        out.append(t.view(cur).expand(shape))
    if xy:
        out[0], out[1] = out[1], out[0]
    return tuple(out)


def _host_expand(x, cur, out):
    src = _strides(cur)
    vals = []

    def walk(level, off):
        if level == len(out):
            vals.append(x.host[off])
            return
        for j in range(out[level]):
            walk(level + 1, off + (0 if cur[level] == 1 else j * src[level]))
    walk(0, 0)
    return _host(vals, out)


_HOST_OPS = {
    "add": lambda p, q: p + q, "sub": lambda p, q: p - q,
    "mul": lambda p, q: p * q, "div": lambda p, q: p / _float(q),
    "lt": lambda p, q: _int(p < q), "le": lambda p, q: _int(p <= q),
    "gt": lambda p, q: _int(p > q), "ge": lambda p, q: _int(p >= q),
    "eq": lambda p, q: _int(p == q), "ne": lambda p, q: _int(p != q),
    "bitwise_xor": lambda p, q: p ^ q, "bitwise_and": lambda p, q: p & q,
    "bitwise_or": lambda p, q: p | q,
    "maximum": _max, "minimum": _min, "pow": lambda p, q: p ** q,
}


_HOST_MAX = 1 << 15
_DEV_OPS = {"bitwise_and": "mul", "bitwise_or": "maximum",
            "bitwise_xor": "ne"}


def _on_device(t):
    return Tensor(t.index_tensor(float32), t.shape)


def _host_binary(a, b, op):
    if isinstance(b, Tensor) and b.host is None:
        return _binary(Tensor(a.index_tensor(b.dtype), a.shape), b,
                       _DEV_OPS.get(op, op))
    if isinstance(a, Tensor) and a.host is None:
        return _binary(a, Tensor(b.index_tensor(a.dtype), b.shape),
                       _DEV_OPS.get(op, op))
    f = _HOST_OPS[op]
    if not isinstance(b, Tensor):
        if len(a.host) > _HOST_MAX:
            return _binary(_on_device(a), b, _DEV_OPS.get(op, op))
        return _host([f(v, b) for v in a.host], a.shape)
    shape = _broadcast_shape(a.shape, b.shape)[0]
    if _numel(shape) > _HOST_MAX:
        return _binary(_on_device(a), _on_device(b), _DEV_OPS.get(op, op))
    if a.shape != b.shape:
        a, b = a.expand(shape), b.expand(shape)
    return _host([f(p, q) for p, q in zip(a.host, b.host)], a.shape)


def _is_int_dtype(dtype):
    return dtype in (int64, "int32", bool, bool_, _int, long_)


def arange(start, end=None, step=1, dtype=None, device=None):
    if end is None:
        start, end = 0, start
    vals = list(range(start, end, step))
    if dtype is None or _is_int_dtype(dtype):
        return _host(vals, (len(vals),))
    return from_flat([_float(v) for v in vals], (len(vals),), dtype)


def tensor(data, dtype=None, device=None):
    if data.__class__ is list and len(data) == 1 and \
            data[0].__class__ is _float and (dtype is None or
                                             dtype is float32):
        return Tensor(_scalar(data[0], float32), (1,))

    def shape_of(v):
        return (len(v),) + shape_of(v[0]) if isinstance(v, (list, tuple)) \
            else ()

    def flat_of(v, out):
        if isinstance(v, (list, tuple)):
            for e in v:
                flat_of(e, out)
        else:
            out.append(v)
        return out
    shape, vals = shape_of(data), flat_of(data, [])
    if dtype is None:
        dtype = float32 if _any(isinstance(v, _float) for v in vals) \
            else int64
    if _is_int_dtype(dtype):
        return _host([_int(v) for v in vals], shape)
    if len(vals) == 1:
        return Tensor(_scalar(_float(vals[0]), dtype), shape)
    return from_flat([_float(v) for v in vals], shape, dtype)


class finfo(object):
    _MIN = {float32: -3.4028234663852886e+38,
            float64: -1.7976931348623157e+308, float16: -65504.0}

    def __init__(self, dtype):
        self.min = self._MIN[dtype]
        self.max = -self.min


class jit(object):
    @staticmethod
    def is_tracing():
        return False

    @staticmethod
    def is_scripting():
        return False

    @staticmethod
    def unused(fn):
        return fn

    @staticmethod
    def _overload_method(fn):
        # upstream declares typed overloads for TorchScript and then the
        # real method under the same name, which is the one a class keeps
        return fn

    @staticmethod
    def export(fn):
        return fn

    @staticmethod
    def ignore(fn=None, **kw):
        return fn if fn is not None else (lambda f: f)


class no_grad(object):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __call__(self, fn):
        return fn


inference_mode = no_grad


def is_tensor(x):
    return isinstance(x, Tensor)


class fx(object):
    @staticmethod
    def wrap(fn_or_name):
        return fn_or_name


def get_default_dtype():
    return float32


def meshgrid(*tensors, **kw):
    """Each 1-D input viewed along its own axis and expanded to the grid;
    'xy' swaps the first two inputs and outputs, as torch does."""
    indexing = kw.get("indexing", "ij")
    ts = list(tensors[0] if len(tensors) == 1 and
              isinstance(tensors[0], (tuple, list)) else tensors)
    xy = indexing == "xy" and len(ts) > 1
    if xy:
        ts[0], ts[1] = ts[1], ts[0]
    shape = tuple(t.numel() for t in ts)
    out = [t.reshape(tuple(-1 if j == i else 1 for j in range(len(ts))))
           .expand(shape) for i, t in enumerate(ts)]
    if xy:
        out[0], out[1] = out[1], out[0]
    return tuple(out)


def roll(x, shifts, dims=None):
    """torch.roll along the given dims: the two pieces of each dim swapped,
    by narrow and cat."""
    if dims is None:
        raise NotImplementedError("roll of the flattened tensor")
    if not isinstance(shifts, (tuple, list)):
        shifts, dims = (shifts,), (dims,)
    for s, d in zip(shifts, dims):
        d = _norm_dim(d, len(x.shape))
        n = x.shape[d]
        s %= n
        if s:
            x = cat([x.narrow(d, n - s, s), x.narrow(d, 0, n - s)], d)
    return x


def addcmul(input, tensor1, tensor2, value=1):
    t = tensor1 * tensor2
    return input + (t if value == 1 else t * value)


def linspace(start, end, steps, dtype=None, device=None):
    step = (end - start) / _float(steps - 1) if steps > 1 else 0.0
    vals = [start + i * step if i < steps // 2 else
            end - (steps - 1 - i) * step for i in range(steps)]
    return from_flat([_float(v) for v in vals], (steps,), dtype or float32)


_rng = None


def randn(*shape, **kw):
    """Standard normal samples drawn on the host (a module's initial values,
    which the checkpoint replaces)."""
    global _rng
    import random
    if _rng is None:
        _rng = random.Random(0)
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return from_flat([_rng.gauss(0.0, 1.0) for _ in range(_numel(shape))],
                     shape, kw.get("dtype") or float32)


from torch import nn  # noqa: E402
from torch.nn import functional  # noqa: E402,F401
