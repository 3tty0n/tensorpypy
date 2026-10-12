from rpython.jit.codewriter.effectinfo import EffectInfo
from rpython.jit.metainterp.history import ConstInt, ConstPtr, CONST_NULL
from rpython.jit.metainterp.optimizeopt.optimizer import REMOVED, Optimization
from rpython.jit.metainterp.optimizeopt.util import (
    make_dispatcher_method, get_box_replacement)
from rpython.jit.metainterp.resoperation import rop, ResOperation
from rpython.jit.metainterp.optimizeopt.info import (
    AbstractVirtualPtrInfo, getptrinfo)
from rpython.metatensor import core, kernels
from rpython.rlib.debug import (debug_start, debug_stop, debug_print,
                                have_debug_prints)
from rpython.rlib.objectmodel import specialize
from rpython.rtyper.lltypesystem import lltype, llmemory

def vtensor_info(box):
    info = getptrinfo(box)
    if isinstance(info, VTensorInfo) and info.is_virtual():
        return info
    return None

class VTensorInfo(AbstractVirtualPtrInfo):

    launched_kernel = lltype.nullptr(core.KERNEL)
    launched_box = None
    node_index = -1

    def __init__(self, opcode, args, param, opt=None):
        self.opcode = opcode
        self.args = args
        self.param = param
        self.opt = opt
        self._is_virtual = True
        self.flags = node_flags(opcode, param)
        self.count = 1
        for box in args:
            sub = vtensor_info(box)
            if sub is not None:
                self.flags |= sub.flags
                self.count = min(self.count + sub.count, BIG_CHAIN)

    def is_identity_view(self):
        return (self.opcode == core.GATHER and len(self.args) == 1 and
                core.gather_kind(self.param) == core.GA_VIEW and
                core.view_is_identity(core.view_of(self.param)))

    def is_reindexing(self):
        return (self.opcode == core.GATHER and
                core.gather_kind(self.param) == core.GA_VIEW and
                not self.is_identity_view())

    def force_as_alias(self, op, optforce):
        src = optforce.optimizer.force_box(self.args[0])
        cic = optforce.optimizer.metainterp_sd.callinfocollection
        calldescr, func = cic.callinfo_for_oopspec(EffectInfo.OS_TENSOR_GATHER)
        newop = ResOperation(rop.CALL_R, [ConstInt(func),
                                          get_box_replacement(src),
                                          ConstInt(self.param)],
                             descr=calldescr)
        optforce.emit_extra(newop)
        newop = optforce.optimizer.getlastop()
        op = get_box_replacement(op)
        op.set_forwarded(newop)
        return newop

    def root_view(self):
        if self.opcode == core.GATHER and core.gather_kind(self.param) == \
                core.GA_VIEW:
            return core.view_of(self.param)
        if core.is_reduction(self.opcode) or self.opcode == core.CONV:
            return -1
        sub = vtensor_info(self.args[self.big_arg()])
        if sub is not None:
            return sub.root_view()
        return -1

    def is_virtual(self):
        return self._is_virtual

    def force_box(self, op, optforce):
        if not self._is_virtual:
            return op
        self._is_virtual = False
        if (self.launched_kernel and not core.drain_knob.on and
                len(self.launched_kernel.outputs) < core.MAX_EXTRA_OUTPUTS):
            # The region already ran, so the value is a device buffer with a
            # known layout. Ask the kernel for it instead of rebuilding the
            # graph. METATENSOR_DRAIN discards this on purpose.
            return self.force_as_extra_output(op, optforce)
        if self.is_identity_view() and vtensor_info(self.args[0]) is not None:
            return self.force_as_alias(op, optforce)
        leaves, consts = [], []
        opcodes, lefts, rights, params, infos = [], [], [], [], []
        _collect_indexed(self, leaves, consts, opcodes, lefts, rights, params,
                         infos)
        if len(leaves) > core.max_inputs():
            for box in self.args:
                if vtensor_info(box) is not None:
                    optforce.optimizer.force_box(box)
            leaves, consts = [], []
            opcodes, lefts, rights, params, infos = [], [], [], [], []
            _collect_indexed(self, leaves, consts, opcodes, lefts, rights,
                             params, infos)
        if not leaves:
            # All-scalar chain: keep the scalars as real inputs to give the launcher a size.
            leaves, consts = [], []
            opcodes, lefts, rights, params, infos = [], [], [], [], []
            _collect_indexed(self, leaves, consts, opcodes, lefts, rights,
                             params, infos, False)
        n = 0
        cols = 0
        if self.opt is not None and not self.flags & F_VIEW:
            n = self.opt.static_size(leaves)
            cols = self.opt.static_cols(self.big_leaf())
        kernel = lltype.malloc(core.KERNEL)
        kernel.ninputs = len(leaves)
        kernel.consts = lltype.malloc(core.HOSTARRAY, len(consts))
        for i in range(len(consts)):
            kernel.consts[i] = _const_value(consts[i])
        kernel.nodes = lltype.malloc(core.NODEARRAY, len(opcodes))
        kernel.fn = kernel.sumroot = kernel.threads = kernel.shared = kernel.nextra = 0
        kernel.wfn = kernel.wthreads = kernel.wshared = kernel.wnextra = 0
        kernel.narrow = core.NO_NARROW
        kernel.vsrc = 0
        kernel.vroot = -1
        kernel.rowmode = 0
        kernel.nouts = 0
        kernel.n = n
        kernel.cols = cols
        kernel.dtype = core.policy.dtype
        kernel.modes = 0
        kernel.outmodes = 0
        kernel.outputs = lltype.malloc(core.SHAPEARRAY, 0)
        for i in range(len(opcodes)):
            node = kernel.nodes[i]
            node.opcode = opcodes[i]
            node.a = lefts[i]
            node.b = rights[i]
            node.p = params[i]
        if self.opt is not None:
            self.opt.pending.append(kernel)
        else:
            kernels.compile_or_reuse(kernel)
        gcref = lltype.cast_opaque_ptr(llmemory.GCREF, kernel)
        cic = optforce.optimizer.metainterp_sd.callinfocollection
        calldescr, func = cic.callinfo_for_oopspec(EffectInfo.OS_TENSOR_LAUNCH)
        args = [ConstInt(func), ConstPtr(gcref)] + leaves
        while len(args) < 2 + core.MAX_INPUTS_LIMIT:
            args.append(CONST_NULL)
        newop = ResOperation(rop.CALL_R, args, descr=calldescr)
        optforce.emit_extra(newop)
        newop = optforce.optimizer.getlastop()
        op = get_box_replacement(op)
        op.set_forwarded(newop)
        outer = {}
        if self.flags & F_VIEW:
            _outer_space(self, outer)
        for i in range(len(infos)):
            info = infos[i]
            if self.flags & F_VIEW and info not in outer:
                continue
            if info is not self and not core.is_reduction(info.opcode):
                info.launched_kernel = kernel
                info.launched_box = newop
                info.node_index = len(leaves) + len(consts) + i
        return newop

    def force_as_extra_output(self, op, optforce):
        k = kernels.add_output(self.launched_kernel, self.node_index)
        cic = optforce.optimizer.metainterp_sd.callinfocollection
        calldescr, func = cic.callinfo_for_oopspec(EffectInfo.OS_TENSOR_OUTPUT)
        newop = ResOperation(rop.CALL_R, [ConstInt(func), self.launched_box,
                                          ConstInt(k)], descr=calldescr)
        optforce.emit_extra(newop)
        newop = optforce.optimizer.getlastop()
        op = get_box_replacement(op)
        op.set_forwarded(newop)
        return newop

    def big_arg(self):
        bc = core.bc_mode(self.opcode, self.param)
        if (bc == core.BC_L_ROW or bc == core.BC_L_SCALAR or
                bc == core.BC_L_COL):
            return 1
        return 0

    def size_leaf(self):
        if core.is_reduction(self.opcode) or self.opcode == core.CONV:
            return None
        box = self.args[self.big_arg()]
        sub = vtensor_info(box)
        if sub is not None:
            return sub.size_leaf()
        return box

    def shape_leaf(self):
        """A leaf with this value's shape, or None if a gather between them changes it."""
        if self.opcode == core.GATHER and core.gather_changes_shape(self.param):
            return None
        if core.is_reduction(self.opcode) or self.opcode == core.CONV:
            return None
        box = self.args[self.big_arg()]
        sub = vtensor_info(box)
        if sub is not None:
            return sub.shape_leaf()
        return box

    def conv_dims(self):
        if core.is_reduction(self.opcode):
            return None
        if self.opcode == core.CONV:
            d = core.conv_descs.params[self.param]
            return [d[core.CD_ROWS], d[core.CD_O]]
        sub = vtensor_info(self.args[self.big_arg()])
        if sub is not None:
            return sub.conv_dims()
        return None

    def has_conv(self):
        return bool(self.flags & F_CONV)

    def big_leaf(self):
        if core.is_reduction(self.opcode):
            box = self.args[0]
        else:
            box = self.args[self.big_arg()]
        sub = vtensor_info(box)
        if sub is not None:
            return sub.big_leaf()
        return box

    def is_scalar(self):
        if core.is_reduction(self.opcode):
            return self.param == core.AXIS_ALL
        sub = vtensor_info(self.args[self.big_arg()])
        if sub is not None:
            return sub.is_scalar()
        return False

    def _visitor_walk_recursive(self, instbox, visitor):
        visitor.register_virtual_fields(instbox, self.args)
        for box in self.args:
            sub = vtensor_info(box)
            if sub is not None:
                sub.visitor_walk_recursive(box, visitor)

    @specialize.argtype(1)
    def visitor_dispatch_virtual_type(self, visitor):
        return visitor.visit_vtensor(self.opcode, self.param)

def _conv_barrier(opcode, param, i, sub):
    if opcode == core.CONV:
        return True
    if not sub.has_conv():
        return False
    if core.is_reduction(opcode) or opcode == core.GATHER:
        return True
    bc = core.bc_mode(opcode, param)
    if bc == core.BC_NONE:
        return False
    small = 0
    if bc == core.BC_R_ROW or bc == core.BC_R_SCALAR or bc == core.BC_R_COL:
        small = 1
    return i == small

F_VIEW, F_BOX, F_BCAST, F_REDUCE, F_GATHER, F_CONV = 1, 2, 4, 8, 16, 32
BIG_CHAIN = 1000
SMALL_VALUE = 8

def _outer_space(info, seen):
    seen[info] = 0
    if info.is_reindexing():
        return
    for box in info.args:
        sub = vtensor_info(box)
        if sub is not None and sub not in seen:
            _outer_space(sub, seen)

def node_flags(opcode, param):
    if opcode == core.CONV:
        return F_CONV
    if opcode == core.GATHER:
        if core.gather_kind(param) != core.GA_VIEW:
            return F_GATHER
        if core.view_bounded(core.view_of(param)):
            return F_VIEW | F_BOX
        return F_VIEW
    if core.is_reduction(opcode):
        return F_REDUCE
    bc = core.bc_mode(opcode, param)
    if bc != core.BC_NONE and bc != core.BC_R_SCALAR and \
            bc != core.BC_L_SCALAR:
        return F_BCAST
    return 0

def _collect_indexed(info, leaves, consts, opcodes, lefts, rights, params,
                     infos, split=True):
    _collect_leaves(info, leaves, consts, split, {})
    _emit_nodes(info, leaves, consts, opcodes, lefts, rights, params, infos,
                {})

def _is_const_scalar(box):
    """A 0-d tensor whose pointer the trace knows (cached _scalar()/runtime.scalar()).

    Its value becomes a literal in the kernel body, so it costs no input slot."""
    if not box.is_constant():
        return False
    ref = box.getref_base()
    if not ref:
        return False
    t = lltype.cast_opaque_ptr(core.TENSORPTR, ref)
    if not t or t.size != 1 or not t.host or len(t.host) < 1:
        return False
    v = t.host[0]
    return v == v and v - v == 0.0

def _const_value(box):
    t = lltype.cast_opaque_ptr(core.TENSORPTR, box.getref_base())
    return t.host[0]

def _collect_leaves(info, leaves, consts, split, seen):
    for box in info.args:
        sub = vtensor_info(box)
        if sub is not None:
            if sub not in seen:
                seen[sub] = 0
                _collect_leaves(sub, leaves, consts, split, seen)
        elif split and _is_const_scalar(box):
            if _leaf_index(consts, box) < 0:
                consts.append(box)
        elif _leaf_index(leaves, box) < 0:
            leaves.append(box)

def _leaf_index(leaves, box):
    for j in range(len(leaves)):
        if leaves[j].same_box(box):
            return j
    return -1

def _emit_nodes(info, leaves, consts, opcodes, lefts, rights, params, infos,
                emitted):
    idx = [-1, -1]
    for i in range(len(info.args)):
        box = info.args[i]
        sub = vtensor_info(box)
        if sub is not None:
            j = emitted.get(sub, -1)
            if j < 0:
                j = _emit_nodes(sub, leaves, consts, opcodes, lefts, rights,
                                params, infos, emitted)
            idx[i] = j
        else:
            j = _leaf_index(leaves, box)
            if j < 0:
                j = len(leaves) + _leaf_index(consts, box)
            idx[i] = j
    opcodes.append(info.opcode)
    lefts.append(idx[0])
    rights.append(idx[1])
    params.append(info.param)
    infos.append(info)
    j = len(leaves) + len(consts) + len(opcodes) - 1
    emitted[info] = j
    return j

class OptTensor(Optimization):

    def setup(self):
        self.sizes = {}
        self.cols = {}
        self.pending = []
        self.live = []

    def propagate_forward(self, op):
        if self.live and op.is_guard():
            self.at_guard(op)
        return dispatch_opt(self, op)

    def at_guard(self, op):
        opnum = op.getopnum()
        if (opnum == rop.GUARD_NOT_INVALIDATED or
                opnum == rop.GUARD_FUTURE_CONDITION):
            return
        if opnum == rop.GUARD_NO_EXCEPTION:
            if self.last_emitted_operation is REMOVED:
                return
            kind = core.SPLIT_EXC
        elif (opnum == rop.GUARD_EXCEPTION or
                opnum == rop.GUARD_NOT_FORCED or
                opnum == rop.GUARD_NOT_FORCED_2):
            kind = core.SPLIT_EXC
        else:
            kind = core.SPLIT_BRANCH
        debug_start("jit-tensor-span")
        if have_debug_prints():
            n = 0
            for box in self.live:
                if vtensor_info(box) is not None:
                    n += 1
            if n:
                debug_print("span", kind, op.getopname(), n)
        debug_stop("jit-tensor-span")
        if core.split_knob.mask & kind:
            self.force_roots()

    def force_roots(self):
        interior = {}
        for box in self.live:
            info = vtensor_info(box)
            if info is not None:
                for arg in info.args:
                    sub = vtensor_info(arg)
                    if sub is not None:
                        interior[sub] = None
        live = self.live
        self.live = []
        for box in live:
            info = vtensor_info(box)
            if info is not None and info not in interior:
                self.optimizer.force_box(box)

    def flush(self):
        self.compile_pending()
        self.sizes = {}
        self.cols = {}
        self.live = []

    def compile_pending(self):
        pending = self.pending
        self.pending = []
        for kernel in pending:
            kernels.compile_or_reuse(kernel)

    def static_size(self, leaves):
        n = -1
        for leaf in leaves:
            sizebox = self.sizes.get(leaf, None)
            if sizebox is None:
                return 0
            sizebox = get_box_replacement(sizebox)
            if not sizebox.is_constant():
                return 0
            value = sizebox.getint()
            if n != -1 and value != n:
                return 0
            n = value
        if n <= 0:
            return 0
        return n

    def static_cols(self, leaf):
        box = self.cols.get(leaf, None)
        if box is None:
            return 0
        box = get_box_replacement(box)
        if not box.is_constant():
            return 0
        value = box.getint()
        if value <= 0:
            return 0
        return value

    def optimize_CALL_R(self, op):
        effectinfo = op.getdescr().get_extra_info()
        idx = effectinfo.oopspecindex
        if idx == EffectInfo.OS_TENSOR_ASSIGN:
            # Pending chains may still read the tensor being written. Materialize
            # them first so a deferred read sees the operand as built.
            self.force_live()
            return self.emit(op)
        if idx == EffectInfo.OS_TENSOR_PLACE_INTO:
            dst = get_box_replacement(op.getarg(1))
            if vtensor_info(dst) is not None:
                self.optimizer.force_box(dst)
                dst = get_box_replacement(dst)
            src = get_box_replacement(op.getarg(2))
            if vtensor_info(src) is not None:
                self.optimizer.force_box(src)
            self.force_readers(dst)
            return self.emit(op)
        if (EffectInfo.OS_TENSOR_ADD <= idx <= EffectInfo.OS_TENSOR_CONV or
                idx == EffectInfo.OS_TENSOR_GATHER2):
            if idx == EffectInfo.OS_TENSOR_GATHER2:
                opcode = core.GATHER
                nargs = 2
            else:
                opcode = idx - EffectInfo.OS_TENSOR_ADD
                nargs = core.ARITY[opcode]
            param = 0
            if core.HAS_PARAM[opcode]:
                pbox = get_box_replacement(op.getarg(1 + nargs))
                if not pbox.is_constant():
                    return self.emit(op)
                param = pbox.getint()
            args = [get_box_replacement(op.getarg(1 + i))
                    for i in range(nargs)]
            for i in range(nargs):
                sub = vtensor_info(args[i])
                if sub is not None and (sub.launched_kernel or
                                        _conv_barrier(opcode, param, i, sub)):
                    self.optimizer.force_box(args[i])
            args = [get_box_replacement(box) for box in args]
            if self._view_boundary(opcode, param, args):
                args = [get_box_replacement(box) for box in args]
            if (opcode == core.GATHER and nargs == 1 and
                    vtensor_info(args[0]) is None and
                    core.gather_kind(param) == core.GA_VIEW and
                    core.view_is_identity(core.view_of(param))):
                return self.emit(op)
            if self._too_wide(args):
                for box in args:
                    self.optimizer.force_box(box)
                if self._too_wide(args):
                    # Constants do not force. Emit a residual call so the
                    # result is a single leaf.
                    return self.emit(op)
            info = VTensorInfo(opcode, args, param, self)
            op = self.replace_op_with(op, op.getopnum())
            op.set_forwarded(info)
            self.live.append(op)
            self.last_emitted_operation = REMOVED
            return
        return self.emit(op)
    optimize_CALL_PURE_R = optimize_CALL_R

    def optimize_CALL_I(self, op):
        effectinfo = op.getdescr().get_extra_info()
        idx = effectinfo.oopspecindex
        if idx == EffectInfo.OS_TENSOR_DTYPE:
            info = vtensor_info(op.getarg(1))
            if info is not None:
                args = [op.getarg(0), info.big_leaf()]
                op = self.replace_op_with(op, op.getopnum(), args=args)
            return self.emit(op)
        if (idx == EffectInfo.OS_TENSOR_SIZE or
                idx == EffectInfo.OS_TENSOR_SHAPE or
                idx == EffectInfo.OS_TENSOR_NDIM):
            info = vtensor_info(op.getarg(1))
            if info is not None and info.root_view() >= 0:
                value = self._view_extent(idx, info.root_view(), op)
                if value >= 0:
                    self.make_constant(op, ConstInt(value))
                    self.last_emitted_operation = REMOVED
                    return
            if info is not None:
                if idx == EffectInfo.OS_TENSOR_SHAPE:
                    leaf = info.shape_leaf()
                else:
                    leaf = info.size_leaf()
                if leaf is None:
                    dims = info.conv_dims()
                    if dims is not None:
                        value = 2
                        if idx == EffectInfo.OS_TENSOR_SIZE:
                            value = dims[0] * dims[1]
                        elif idx == EffectInfo.OS_TENSOR_SHAPE:
                            axis = get_box_replacement(op.getarg(2))
                            if not axis.is_constant():
                                return self.emit(op)
                            value = dims[axis.getint() & 1]
                        self.make_constant(op, ConstInt(value))
                        self.last_emitted_operation = REMOVED
                        return
                    if not info.is_scalar():
                        return self.emit(op)
                    self.make_constant(op, ConstInt(1))
                    self.last_emitted_operation = REMOVED
                    return
                args = [op.getarg(0), leaf]
                if idx == EffectInfo.OS_TENSOR_SHAPE:
                    args.append(op.getarg(2))
                op = self.replace_op_with(op, op.getopnum(), args=args)
            if idx == EffectInfo.OS_TENSOR_SIZE:
                self.sizes[get_box_replacement(op.getarg(1))] = op
            elif idx == EffectInfo.OS_TENSOR_SHAPE:
                axis = get_box_replacement(op.getarg(2))
                if axis.is_constant() and axis.getint() == 1:
                    self.cols[get_box_replacement(op.getarg(1))] = op
        return self.emit(op)
    optimize_CALL_PURE_I = optimize_CALL_I

    def _view_boundary(self, opcode, param, args):
        conv = opcode == core.CONV
        for box in args:
            sub = vtensor_info(box)
            if sub is not None and sub.flags & F_CONV:
                conv = True
        if conv:
            forced = False
            for box in args:
                sub = vtensor_info(box)
                if sub is not None and sub.flags & F_VIEW:
                    self.optimizer.force_box(box)
                    forced = True
            if forced:
                return True
        if opcode == core.GATHER and core.gather_kind(param) == core.GA_VIEW:
            stop = F_BOX | F_BCAST | F_REDUCE | F_GATHER | F_CONV
        elif opcode == core.GATHER or (core.is_reduction(opcode) and not (
                opcode == core.SUM and param == core.AXIS_ALL and
                core.policy.dtype != core.F16)):
            stop = F_VIEW
        else:
            return False
        forced = False
        spread = (opcode == core.GATHER and
                  core.gather_kind(param) == core.GA_VIEW and
                  core.view_spreads(core.view_of(param)))
        for box in args:
            sub = vtensor_info(box)
            if sub is not None and (sub.flags & stop or
                                    (spread and sub.count > SMALL_VALUE)):
                self.optimizer.force_box(box)
                forced = True
        return forced

    def _view_extent(self, idx, vid, op):
        shape = core.view_shape(vid)
        if idx == EffectInfo.OS_TENSOR_SIZE:
            return core.view_numel(vid)
        if idx == EffectInfo.OS_TENSOR_NDIM:
            return len(shape)
        axis = get_box_replacement(op.getarg(2))
        if not axis.is_constant():
            return -1
        k = axis.getint()
        if k < 0 or k >= len(shape):
            return -1
        return shape[k]

    def force_live(self):
        live = self.live
        self.live = []
        for box in live:
            if vtensor_info(box) is not None:
                self.optimizer.force_box(box)

    def force_readers(self, dst):
        live = self.live
        self.live = []
        for box in live:
            info = vtensor_info(box)
            if info is None:
                continue
            leaves, consts = [], []
            _collect_leaves(info, leaves, consts, False, {})
            reads = False
            for leaf in leaves:
                if get_box_replacement(leaf).same_box(dst):
                    reads = True
            if reads:
                self.optimizer.force_box(box)
            else:
                self.live.append(box)

    def _too_wide(self, args):
        """Would fusing this node need more input slots than the launcher has?

        With no tensor leaf, force_box recollects with scalar folding off, so
        constants count as inputs."""
        leaves, consts = [], []
        _collect_leaves(VTensorInfo(0, args, 0), leaves, consts, True, {})
        n = len(leaves)
        if n == 0:
            n = len(consts)
        return n > core.max_inputs()

    def optimize_GUARD_NO_EXCEPTION(self, op):
        if self.last_emitted_operation is REMOVED:
            return
        return self.emit(op)

dispatch_opt = make_dispatcher_method(OptTensor, 'optimize_',
                                      default=OptTensor.emit)
