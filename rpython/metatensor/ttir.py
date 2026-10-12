import sys
from rpython.rlib.rfloat import INFINITY, formatd
from rpython.metatensor.core import (ADD, ARITY, CD_BK, CD_BM, CD_BN, CD_G, CD_GM, CD_O, CD_ROWS, CONV, conv_col_tiles, conv_descs, conv_grouped_dot, main_conv, B_KEEP_NZ, B_KEEP_Z, B_LT, B_MAX, B_MIN, B_NE, B_POW, BINARY, U_GELU, U_NEG, U_SIGMOID, U_SILU, U_TANH, UNARY, bc_mode, binary_fn, GA_ROWS, AXIS_ALL, BC_L_COL, BC_L_ROW, BC_L_SCALAR, BC_R_COL, BC_R_ROW, BC_R_SCALAR, COMP_NEG_INF, COMP_TYPE, DIV, EQMASK, EXP, GATHER, GA_COL2CHW, GA_HEADMERGE, GA_HEADSPLIT, GA_IM2COL, GA_IM2COL_NHWC, GA_IM2COL2, GA_IM2COL_T, GA_MAXPOOL_NHWC, GA_POOL, GA_ROTHALF, GA_SCATTER, GA_STRIDED, GA_TAKE, POOL_ADAPTIVE, POOL_AVG, POOL_DEPTHWISE, conv_out, MAXR, MUL, RELU, RELUGRAD, SQRT, STORE_TYPE, SUB, SUM, config, gather_changes_shape, gather_dh, gather_heads, gather_kind, has_conv, is_reduction, nvals, view_clamp, GA_VIEW, view_numel, view_of, view_offset, view_params, view_params_list)


def next_pow2(c):
    t = 32
    while t < c:
        t *= 2
    return t

def row_tile(kernel):
    if kernel.cols > 0:
        t = next_pow2(kernel.cols)
        if t <= config.block:
            return t
    return config.block

def row_warps(tile):
    w = tile // 128
    if w < 1:
        return 1
    if w > 8:
        return 8
    return w

def set_mode(modes, v, m):
    if modes[v] < 0:
        modes[v] = m
        return True
    return modes[v] == m

def all_modes(kernel):
    modes = _modes_from(kernel, 0)
    if _needs_row_root(kernel, modes) or (not modes and
                                          _row_reduced_root(kernel)):
        modes = _modes_from(kernel, 3)
        nin = nvals(kernel)
        for k in range(len(kernel.nodes)):
            node = kernel.nodes[k]
            if len(modes) == 0 or modes[nin + k] != 3:
                continue
            bc = bc_mode(node.opcode, node.p)
            if bc == BC_R_ROW or bc == BC_L_ROW:
                return []
    return modes

def _row_reduced_root(kernel):
    nodes = kernel.nodes
    last = len(nodes) - 1
    if last < 0 or is_reduction(nodes[last].opcode):
        return False
    for k in range(last):
        if is_reduction(nodes[k].opcode) and nodes[k].p == 1:
            return True
    return False

def _needs_row_root(kernel, modes):
    nin = nvals(kernel)
    nodes = kernel.nodes
    last = len(nodes) - 1
    if last < 0 or len(modes) != nin + len(nodes):
        return False
    if is_reduction(nodes[last].opcode):
        return False
    for k in range(last):
        node = nodes[k]
        if (is_reduction(node.opcode) and node.p == 1 and
                modes[nin + k] != 3):
            return True
    return False

def row_root(kernel, modes):
    nin = nvals(kernel)
    last = len(kernel.nodes) - 1
    return (last >= 0 and len(modes) == nin + last + 1 and
            not is_reduction(kernel.nodes[last].opcode) and
            modes[nin + last] == 3)

def _modes_from(kernel, root):
    nin = nvals(kernel)
    nodes = kernel.nodes
    modes = [-1] * (nin + len(nodes))
    if len(nodes) > 0:
        modes[nin + len(nodes) - 1] = root
    for k in range(len(nodes) - 1, -1, -1):
        node = nodes[k]
        m = modes[nin + k]
        if m < 0:
            m = 0
        ma = m
        mb = m
        if is_view(node):
            ma = 0
            mb = 2
        elif is_reduction(node.opcode):
            ma = 0
        elif ARITY[node.opcode] == 2:
            bc = bc_mode(node.opcode, node.p)
            if bc == BC_R_ROW:
                mb = 1
            elif bc == BC_R_SCALAR:
                mb = 2
            elif bc == BC_R_COL:
                mb = 3
            elif bc == BC_L_ROW:
                ma = 1
            elif bc == BC_L_SCALAR:
                ma = 2
            elif bc == BC_L_COL:
                ma = 3
        if not _literal(kernel, node.a) and not set_mode(modes, node.a, ma):
            return []
        if (node.b >= 0 and not _literal(kernel, node.b) and
                not set_mode(modes, node.b, mb)):
            return []
    for i in range(len(modes)):
        if modes[i] < 0:
            modes[i] = 0
    return modes

def _literal(kernel, v):
    return kernel.ninputs <= v < nvals(kernel)

def input_modes(kernel):
    modes = all_modes(kernel)
    nin = nvals(kernel)
    if len(modes) != nin + len(kernel.nodes):
        return []
    result = []
    for i in range(kernel.ninputs):
        result.append(modes[i])
    return result

def row_mode(kernel, modes):
    nin = nvals(kernel)
    nodes = kernel.nodes
    if len(modes) != nin + len(nodes):
        return False
    if has_gather(kernel):
        return False
    for k in range(len(nodes)):
        node = nodes[k]
        if is_reduction(node.opcode) and node.p == 1:
            if k == len(nodes) - 1 or modes[nin + k] == 3:
                return True
    if kernel.cols <= 0 or kernel.cols > config.block:
        return False
    for i in range(nin):
        if modes[i] == 3:
            return True
    return False

def kernel_row_mode(kernel):
    return row_mode(kernel, all_modes(kernel))

def out_modes(kernel):
    modes = all_modes(kernel)
    if len(modes) != nvals(kernel) + len(kernel.nodes):
        return []
    result = []
    for k in range(len(kernel.outputs)):
        result.append(modes[kernel.outputs[k]])
    return result

def _fliteral(v):
    # MLIR wants a decimal point before any exponent, so 1e-05 is rejected;
    # %e always has one, and 16 digits round-trips an f64.
    t = formatd(v, 'e', 16)
    if 'inf' in t or 'nan' in t:
        return ''
    return t


def _tile_types(block, S, C):
    return ('tensor<%dx%s>' % (block, C), 'tensor<%dx%s>' % (block, S),
            'tensor<%dx!tt.ptr<%s>>' % (block, S), 'tensor<%dxi32>' % block,
            'tensor<%dxi64>' % block, 'tensor<%dxi1>' % block)

BIG_KERNEL_NODES = 32
BIG_KERNEL_BLOCK = 512

MIN_SPREAD_BLOCK = 256
SPREAD_PROGRAMS = 128

def spread_block(block, n):
    """block, halved while a launch over n elements would have fewer than
    SPREAD_PROGRAMS programs: a small launch then still covers the SMs."""
    if config.spread:
        while block > MIN_SPREAD_BLOCK and n < block * SPREAD_PROGRAMS:
            block //= 2
    return block

def _static_numel(kernel):
    if kernel.n > 0:
        return kernel.n
    if kernel.vroot < -1:
        return view_numel(-2 - kernel.vroot)
    return 0

def flat_block(kernel):
    if (len(kernel.nodes) > BIG_KERNEL_NODES and
            config.flat > BIG_KERNEL_BLOCK):
        return BIG_KERNEL_BLOCK
    n = _static_numel(kernel)
    if n <= 0 or kernel.sumroot:
        return config.flat
    for k in range(len(kernel.nodes)):
        if is_reduction(kernel.nodes[k].opcode):
            return config.flat
    return spread_block(config.flat, n)

def _flat_prologue(lines, block, I32, I64):
    lines.append('    %%bs = arith.constant %d : i32' % block)
    lines.append('    %pid = tt.get_program_id x : i32')
    lines.append('    %start = arith.muli %pid, %bs : i32')
    lines.append('    %%range = tt.make_range {end = %d : i32, start = 0 : i32} '
                 ': %s' % (block, I32))
    lines.append('    %%starts = tt.splat %%start : i32 -> %s' % I32)
    lines.append('    %%offs = arith.addi %%starts, %%range : %s' % I32)
    lines.append('    %%offs64 = arith.extsi %%offs : %s to %s' % (I32, I64))
    lines.append('    %%ns = tt.splat %%n : i64 -> %s' % I64)
    lines.append('    %%mask = arith.cmpi slt, %%offs64, %%ns : %s' % I64)


def to_tile_ir(kernel, name, n):
    ty = STORE_TYPE[kernel.dtype]
    tile = 'tile<%dx%s>' % (n, ty)
    params = ', '.join(['%%in%d: !cuda_tile.ptr<%s>' % (i, ty)
                        for i in range(kernel.ninputs)] +
                       ['%%out: !cuda_tile.ptr<%s>' % ty])
    lines = ['cuda_tile.module {',
             '  cuda_tile.entry @%s(%s) {' % (name, params)]
    for i in range(kernel.ninputs):
        lines.append('    %%v%d = cuda_tile.load_ptr_tko %%in%d : %s'
                     % (i, i, tile))
    result = tile
    nodes = kernel.nodes
    for i in range(len(nodes)):
        node = nodes[i]
        v = kernel.ninputs + i
        if node.opcode == ADD:
            lines.append('    %%v%d = arith.addf %%v%d, %%v%d : %s'
                         % (v, node.a, node.b, tile))
        elif node.opcode == MUL:
            lines.append('    %%v%d = arith.mulf %%v%d, %%v%d : %s'
                         % (v, node.a, node.b, tile))
        elif node.opcode == RELU:
            lines.append('    %%v%d = arith.maximumf %%v%d, %%zero : %s'
                         % (v, node.a, tile))
        elif node.opcode == RELUGRAD:
            lines.append('    %%v%d = cuda_tile.select %%v%d, %%v%d, %%zero : %s'
                         % (v, node.a, node.b, tile))
        else:
            result = 'tile<1x%s>' % ty
            lines.append('    %%v%d = cuda_tile.reduce add %%v%d : %s -> %s'
                         % (v, node.a, tile, result))
    lines.append('    cuda_tile.store_ptr_tko %%out, %%v%d : %s'
                 % (kernel.ninputs + len(nodes) - 1, result))
    lines.append('    cuda_tile.return')
    lines.append('  }')
    lines.append('}')
    return '\n'.join(lines)

def is_view(node):
    return node.opcode == GATHER and gather_kind(node.p) == GA_VIEW

def has_gather(kernel):
    for k in range(len(kernel.nodes)):
        if kernel.nodes[k].opcode == GATHER:
            return True
    return False

def to_ttir(kernel, name):
    modes = all_modes(kernel)
    if len(modes) != nvals(kernel) + len(kernel.nodes):
        return ''
    if has_conv(kernel):
        return to_ttir_conv(kernel, name, modes)
    if has_gather(kernel):
        return to_ttir_gathered(kernel, name, modes)
    if row_mode(kernel, modes):
        return to_ttir_row(kernel, name, modes)
    return to_ttir_flat(kernel, name)

def _elementwise(lines, node, v, T, I1):
    return _ew(lines, node.opcode, node.p, '%%v%d' % v, '%%c%d' % v,
               '%%v%d' % node.a, '%%v%d' % node.b, T, I1)

def _ew(lines, opcode, p, dst, cmp, a, b, T, I1):
    if opcode == ADD:
        lines.append('    %s = arith.addf %s, %s : %s' % (dst, a, b, T))
    elif opcode == MUL:
        lines.append('    %s = arith.mulf %s, %s : %s' % (dst, a, b, T))
    elif opcode == RELU:
        lines.append('    %s = arith.cmpf ogt, %s, %%zero : %s' % (cmp, a, T))
        lines.append('    %s = arith.select %s, %s, %%zero : %s, %s'
                     % (dst, cmp, a, I1, T))
    elif opcode == RELUGRAD:
        lines.append('    %s = arith.cmpf ogt, %s, %%zero : %s' % (cmp, a, T))
        lines.append('    %s = arith.select %s, %s, %%zero : %s, %s'
                     % (dst, cmp, b, I1, T))
    elif opcode == SUB:
        lines.append('    %s = arith.subf %s, %s : %s' % (dst, a, b, T))
    elif opcode == DIV:
        lines.append('    %s = arith.divf %s, %s : %s' % (dst, a, b, T))
    elif opcode == EXP:
        lines.append('    %s = math.exp %s : %s' % (dst, a, T))
    elif opcode == SQRT:
        lines.append('    %s = math.sqrt %s : %s' % (dst, a, T))
    elif opcode == EQMASK:
        lines.append('    %s = arith.cmpf oeq, %s, %s : %s' % (cmp, a, b, T))
        lines.append('    %s = arith.select %s, %%one, %%zero : %s, %s'
                     % (dst, cmp, I1, T))
    elif opcode == UNARY:
        return _unary(lines, p, dst, a, T)
    elif opcode == BINARY:
        return _binary(lines, binary_fn(p), dst, cmp, a, b, T, I1)
    else:
        return False
    return True

def _libdevice(T, name):
    # f16 computes in f32, so the compute type picks the variant
    if 'xf64>' in T:
        return '__nv_' + name
    return '__nv_' + name + 'f'

_UNARY_OPS = ['', '', 'math.log', 'math.absf', 'math.sin', 'math.cos',
              'math.erf', 'math.floor', 'arith.negf']

def _unary(lines, fn, dst, a, T):
    if fn == U_TANH:
        # Triton lowers no math.tanh
        lines.append('    %s = tt.extern_elementwise %s {libname = "", '
                     'libpath = "", pure = true, symbol = "%s"} : (%s) -> %s'
                     % (dst, a, _libdevice(T, 'tanh'), T, T))
    elif fn == U_SIGMOID:
        lines.append('    %s_n = arith.negf %s : %s' % (dst, a, T))
        lines.append('    %s_e = math.exp %s_n : %s' % (dst, dst, T))
        lines.append('    %s_d = arith.addf %s_e, %%one : %s' % (dst, dst, T))
        lines.append('    %s = arith.divf %%one, %s_d : %s' % (dst, dst, T))
    elif fn > U_SIGMOID and fn <= U_NEG:
        lines.append('    %s = %s %s : %s' % (dst, _UNARY_OPS[fn], a, T))
    elif fn == U_GELU:
        lines.append('    %s_r = arith.constant dense<7.0710678118654757e-01> '
                     ': %s' % (dst, T))
        lines.append('    %s_h = arith.constant dense<5.0e-01> : %s' % (dst, T))
        lines.append('    %s_s = arith.mulf %s, %s_r : %s' % (dst, a, dst, T))
        lines.append('    %s_e = math.erf %s_s : %s' % (dst, dst, T))
        lines.append('    %s_p = arith.addf %s_e, %%one : %s' % (dst, dst, T))
        lines.append('    %s_x = arith.mulf %s, %s_h : %s' % (dst, a, dst, T))
        lines.append('    %s = arith.mulf %s_x, %s_p : %s' % (dst, dst, dst, T))
    elif fn == U_SILU:
        lines.append('    %s_n = arith.negf %s : %s' % (dst, a, T))
        lines.append('    %s_e = math.exp %s_n : %s' % (dst, dst, T))
        lines.append('    %s_d = arith.addf %s_e, %%one : %s' % (dst, dst, T))
        lines.append('    %s = arith.divf %s, %s_d : %s' % (dst, a, dst, T))
    else:
        return False
    return True

_CMP = ['', '', '', 'olt', 'ole', 'ogt', 'oge', 'oeq', 'une']

def _binary(lines, fn, dst, cmp, a, b, T, I1):
    if fn == B_MAX:
        lines.append('    %s = arith.maximumf %s, %s : %s' % (dst, a, b, T))
    elif fn == B_MIN:
        lines.append('    %s = arith.minimumf %s, %s : %s' % (dst, a, b, T))
    elif fn == B_POW:
        lines.append('    %s = tt.extern_elementwise %s, %s {libname = "", '
                     'libpath = "", pure = true, symbol = "%s"} : '
                     '(%s, %s) -> %s' % (dst, a, b, _libdevice(T, 'pow'),
                                         T, T, T))
    elif fn >= B_LT and fn <= B_NE:
        lines.append('    %s = arith.cmpf %s, %s, %s : %s'
                     % (cmp, _CMP[fn], a, b, T))
        lines.append('    %s = arith.select %s, %%one, %%zero : %s, %s'
                     % (dst, cmp, I1, T))
    elif fn == B_KEEP_NZ or fn == B_KEEP_Z:
        pred = 'une' if fn == B_KEEP_NZ else 'oeq'
        lines.append('    %s = arith.cmpf %s, %s, %%zero : %s'
                     % (cmp, pred, a, T))
        lines.append('    %s = arith.select %s, %s, %%zero : %s, %s'
                     % (dst, cmp, b, I1, T))
    else:
        return False
    return True

def _trunc(lines, tag, v, half, T, TS):
    if not half:
        return '%%v%d' % v
    lines.append('    %%%s = arith.truncf %%v%d : %s to %s' % (tag, v, T, TS))
    return '%%%s' % tag

def _emit_consts(lines, kernel, nreal, T):
    for j in range(len(kernel.consts)):
        text = _fliteral(kernel.consts[j])
        if not text:
            return False
        lines.append('    %%v%d = arith.constant dense<%s> : %s'
                     % (nreal + j, text, T))
    return True


def _row_reductions(kernel, modes, half):
    """Which values are row reductions, or [] if the chain has a reduction
    row mode cannot run."""
    nodes = kernel.nodes
    nin = nvals(kernel)
    last = len(nodes) - 1
    isred = [False] * (nin + len(nodes))
    for k in range(len(nodes)):
        node = nodes[k]
        if not is_reduction(node.opcode):
            continue
        if node.p == 1:
            if k != last and modes[nin + k] != 3:
                return []
        elif node.p == AXIS_ALL:
            if k != last or node.opcode != SUM or half:
                return []
        else:
            return []
        isred[nin + k] = True
    for k in range(len(kernel.outputs)):
        if isred[kernel.outputs[k]]:
            return []
    return isred

def to_ttir_row(kernel, name, modes):
    nodes = kernel.nodes
    nreal = kernel.ninputs
    nin = nvals(kernel)
    last = len(nodes) - 1
    if last < 0:
        return ''
    BLOCK = row_tile(kernel)
    dt = kernel.dtype
    S = STORE_TYPE[dt]
    C = COMP_TYPE[dt]
    half = S != C
    T, TS, P, I32, I64, I1 = _tile_types(BLOCK, S, C)
    isred = _row_reductions(kernel, modes, half)
    if not isred:
        return ''
    omodes = out_modes(kernel)
    if len(omodes) != len(kernel.outputs):
        return ''
    params = ['%%in%d: !tt.ptr<%s>' % (i, S) for i in range(nreal)]
    params.append('%%out: !tt.ptr<%s>' % S)
    for k in range(len(kernel.outputs)):
        params.append('%%out%d: !tt.ptr<%s>' % (k, S))
    lines = ['module {',
             '  tt.func public @%s(%s, %%n: i64, %%c: i64) '
             'attributes {noinline = false} {' % (name, ', '.join(params)),
             '    %%zero = arith.constant dense<0.0> : %s' % T,
             '    %%one = arith.constant dense<1.0> : %s' % T,
             '    %pid = tt.get_program_id x : i32',
             '    %rowi = arith.extsi %pid : i32 to i64',
             '    %%range = tt.make_range {end = %d : i32, start = 0 : i32} '
             ': %s' % (BLOCK, I32),
             '    %%ar = arith.extsi %%range : %s to %s' % (I32, I64),
             '    %%cs = tt.splat %%c : i64 -> %s' % I64,
             '    %%mask = arith.cmpi slt, %%ar, %%cs : %s' % I64,
             '    %base = arith.muli %rowi, %c : i64',
             '    %%bases = tt.splat %%base : i64 -> %s' % I64,
             '    %%offs = arith.addi %%bases, %%ar : %s' % I64]
    if half:
        lines.append('    %%zeros = arith.constant dense<0.0> : %s' % TS)
    if not _emit_consts(lines, kernel, nreal, T):
        return ''
    for i in range(nreal):
        if modes[i] == 2 or modes[i] == 3:
            src = '%%in%d' % i
            if modes[i] == 3:
                lines.append('    %%sp%d = tt.addptr %%in%d, %%pid : '
                             '!tt.ptr<%s>, i32' % (i, i, S))
                src = '%%sp%d' % i
            lines.append('    %%sv%d = tt.load %s : !tt.ptr<%s>' % (i, src, S))
            sv = '%%sv%d' % i
            if half:
                lines.append('    %%se%d = arith.extf %s : %s to %s'
                             % (i, sv, S, C))
                sv = '%%se%d' % i
            lines.append('    %%v%d = tt.splat %s : %s -> %s' % (i, sv, C, T))
            continue
        offs = '%offs'
        if modes[i] == 1:
            offs = '%ar'
        lines.append('    %%p%d = tt.splat %%in%d : !tt.ptr<%s> -> %s'
                     % (i, i, S, P))
        lines.append('    %%q%d = tt.addptr %%p%d, %s : %s, %s'
                     % (i, i, offs, P, I64))
        if half:
            lines.append('    %%rv%d = tt.load %%q%d, %%mask, %%zeros : %s'
                         % (i, i, P))
            lines.append('    %%v%d = arith.extf %%rv%d : %s to %s'
                         % (i, i, TS, T))
        else:
            lines.append('    %%v%d = tt.load %%q%d, %%mask, %%zero : %s'
                         % (i, i, P))
    for k in range(len(nodes)):
        node = nodes[k]
        v = nin + k
        if _elementwise(lines, node, v, T, I1):
            continue
        if node.opcode == MAXR:
            lines.append('    %%ninf%d = arith.constant dense<%s> : %s'
                         % (v, COMP_NEG_INF[dt], T))
            init = '%%ninf%d' % v
            combine = 'maximumf'
        else:
            init = '%zero'
            combine = 'addf'
        lines.append('    %%rm%d = arith.select %%mask, %%v%d, %s : %s, %s'
                     % (v, node.a, init, I1, T))
        lines.append('    %%rs%d = "tt.reduce"(%%rm%d) <{axis = 0 : i32}> ({'
                     % (v, v))
        lines.append('    ^bb0(%%x%d: %s, %%y%d: %s):' % (v, C, v, C))
        lines.append('      %%rr%d = arith.%s %%x%d, %%y%d : %s'
                     % (v, combine, v, v, C))
        lines.append('      tt.reduce.return %%rr%d : %s' % (v, C))
        lines.append('    }) : (%s) -> %s' % (T, C))
        lines.append('    %%v%d = tt.splat %%rs%d : %s -> %s' % (v, v, C, T))
        if k != last:
            continue
        if node.p == 1:
            lines.append('    %%po = tt.addptr %%out, %%pid : !tt.ptr<%s>, i32'
                         % S)
            sv = '%%rs%d' % v
            if half:
                lines.append('    %%rt%d = arith.truncf %s : %s to %s'
                             % (v, sv, C, S))
                sv = '%%rt%d' % v
            lines.append('    tt.store %%po, %s : !tt.ptr<%s>' % (sv, S))
        else:
            lines.append('    %true = arith.constant true')
            # The accumulator has the kernel's storage type, not always f64.
            # f16 reduces in f32 and is narrowed first, as in the tt.store branch.
            sv = '%%rs%d' % v
            if half:
                lines.append('    %%rt%d = arith.truncf %s : %s to %s'
                             % (v, sv, C, S))
                sv = '%%rt%d' % v
            lines.append('    %%o = tt.atomic_rmw fadd, acq_rel, gpu, %%out, '
                         '%s, %%true : (!tt.ptr<%s>, %s, i1) -> %s'
                         % (sv, S, S, S))
    if row_root(kernel, modes):
        lines.append('    %%pr = tt.addptr %%out, %%pid : !tt.ptr<%s>, i32' % S)
        lines.append('    %%po = tt.splat %%pr : !tt.ptr<%s> -> %s' % (S, P))
        lines.append('    %%z0 = arith.constant dense<0> : %s' % I64)
        lines.append('    %%m0 = arith.cmpi eq, %%ar, %%z0 : %s' % I64)
        lines.append('    tt.store %%po, %s, %%m0 : %s'
                     % (_trunc(lines, 'so', nin + last, half, T, TS), P))
    elif not isred[nin + last]:
        lines.append('    %%po = tt.splat %%out : !tt.ptr<%s> -> %s' % (S, P))
        lines.append('    %%qo = tt.addptr %%po, %%offs : %s, %s' % (P, I64))
        lines.append('    tt.store %%qo, %s, %%mask : %s'
                     % (_trunc(lines, 'so', nin + last, half, T, TS), P))
    for k in range(len(kernel.outputs)):
        ooffs = '%offs'
        if omodes[k] == 1:
            ooffs = '%ar'
        elif omodes[k] == 2:
            lines.append('    %%zo%d = arith.constant dense<0> : %s' % (k, I64))
            ooffs = '%%zo%d' % k
        elif omodes[k] == 3:
            lines.append('    %%ro%d = tt.splat %%rowi : i64 -> %s' % (k, I64))
            ooffs = '%%ro%d' % k
        lines.append('    %%po%d = tt.splat %%out%d : !tt.ptr<%s> -> %s'
                     % (k, k, S, P))
        lines.append('    %%qo%d = tt.addptr %%po%d, %s : %s, %s'
                     % (k, k, ooffs, P, I64))
        lines.append('    tt.store %%qo%d, %s, %%mask : %s'
                     % (k, _trunc(lines, 'sx%d' % k, kernel.outputs[k], half,
                                  T, TS), P))
    lines.append('    tt.return')
    lines.append('  }')
    lines.append('}')
    return '\n'.join(lines) + '\n'

NARROW_COLS = [4, 16, 64, 256]
NARROW_TILE = 2048
NARROW_WARPS = 4

def narrow_rows(slot):
    return NARROW_TILE // NARROW_COLS[slot]

def to_ttir_row_multi(kernel, name, modes, BC):
    nodes = kernel.nodes
    nreal = kernel.ninputs
    nin = nvals(kernel)
    last = len(nodes) - 1
    if last < 0:
        return ''
    R = NARROW_TILE // BC
    dt = kernel.dtype
    S = STORE_TYPE[dt]
    C = COMP_TYPE[dt]
    half = S != C
    isred = _row_reductions(kernel, modes, half)
    if not isred:
        return ''
    for k in range(len(nodes)):
        if is_reduction(nodes[k].opcode) and nodes[k].p != 1:
            return ''
    omodes = out_modes(kernel)
    if len(omodes) != len(kernel.outputs):
        return ''
    T = 'tensor<%dx%dx%s>' % (R, BC, C)
    TS = 'tensor<%dx%dx%s>' % (R, BC, S)
    P = 'tensor<%dx%dx!tt.ptr<%s>>' % (R, BC, S)
    I64 = 'tensor<%dx%dxi64>' % (R, BC)
    I1 = 'tensor<%dx%dxi1>' % (R, BC)
    TR = 'tensor<%dx%s>' % (R, C)
    TRS = 'tensor<%dx%s>' % (R, S)
    TR1 = 'tensor<%dx1x%s>' % (R, C)
    PR = 'tensor<%dx!tt.ptr<%s>>' % (R, S)
    R32 = 'tensor<%dxi32>' % R
    R64 = 'tensor<%dxi64>' % R
    C32 = 'tensor<%dxi32>' % BC
    C64 = 'tensor<%dxi64>' % BC
    params = ['%%in%d: !tt.ptr<%s>' % (i, S) for i in range(nreal)]
    params.append('%%out: !tt.ptr<%s>' % S)
    for k in range(len(kernel.outputs)):
        params.append('%%out%d: !tt.ptr<%s>' % (k, S))
    lines = ['module {',
             '  tt.func public @%s(%s, %%n: i64, %%c: i64) '
             'attributes {noinline = false} {' % (name, ', '.join(params)),
             '    %%zero = arith.constant dense<0.0> : %s' % T,
             '    %%one = arith.constant dense<1.0> : %s' % T,
             '    %pid = tt.get_program_id x : i32',
             '    %pid64 = arith.extsi %pid : i32 to i64',
             '    %%rpp = arith.constant %d : i64' % R,
             '    %row0 = arith.muli %pid64, %rpp : i64',
             '    %nrows = arith.divsi %n, %c : i64',
             '    %%rrng = tt.make_range {end = %d : i32, start = 0 : i32} : %s'
             % (R, R32),
             '    %%rrng64 = arith.extsi %%rrng : %s to %s' % (R32, R64),
             '    %%row0s = tt.splat %%row0 : i64 -> %s' % R64,
             '    %%rows = arith.addi %%row0s, %%rrng64 : %s' % R64,
             '    %%nrs = tt.splat %%nrows : i64 -> %s' % R64,
             '    %%rmask = arith.cmpi slt, %%rows, %%nrs : %s' % R64,
             '    %%cr = tt.make_range {end = %d : i32, start = 0 : i32} : %s'
             % (BC, C32),
             '    %%cr64 = arith.extsi %%cr : %s to %s' % (C32, C64),
             '    %%rowsx = tt.expand_dims %%rows {axis = 1 : i32} : %s -> '
             'tensor<%dx1xi64>' % (R64, R),
             '    %%rowsb = tt.broadcast %%rowsx : tensor<%dx1xi64> -> %s'
             % (R, I64),
             '    %%colsx = tt.expand_dims %%cr64 {axis = 0 : i32} : %s -> '
             'tensor<1x%dxi64>' % (C64, BC),
             '    %%ar = tt.broadcast %%colsx : tensor<1x%dxi64> -> %s'
             % (BC, I64),
             '    %%cs = tt.splat %%c : i64 -> %s' % I64,
             '    %%nrb = tt.splat %%nrows : i64 -> %s' % I64,
             '    %%cm = arith.cmpi slt, %%ar, %%cs : %s' % I64,
             '    %%rowok = arith.cmpi slt, %%rowsb, %%nrb : %s' % I64,
             '    %%mask = arith.andi %%cm, %%rowok : %s' % I1,
             '    %%base = arith.muli %%rowsb, %%cs : %s' % I64,
             '    %%offs = arith.addi %%base, %%ar : %s' % I64]
    if half:
        lines.append('    %%zeros = arith.constant dense<0.0> : %s' % TS)
    if not _emit_consts(lines, kernel, nreal, T):
        return ''
    for i in range(nreal):
        if modes[i] == 2:
            lines.append('    %%sv%d = tt.load %%in%d : !tt.ptr<%s>' % (i, i, S))
            sv = '%%sv%d' % i
            if half:
                lines.append('    %%se%d = arith.extf %s : %s to %s'
                             % (i, sv, S, C))
                sv = '%%se%d' % i
            lines.append('    %%v%d = tt.splat %s : %s -> %s' % (i, sv, C, T))
            continue
        offs = '%offs'
        mask = '%mask'
        if modes[i] == 1:
            offs = '%ar'
        elif modes[i] == 3:
            offs = '%rowsb'
            mask = '%rowok'
        lines.append('    %%p%d = tt.splat %%in%d : !tt.ptr<%s> -> %s'
                     % (i, i, S, P))
        lines.append('    %%q%d = tt.addptr %%p%d, %s : %s, %s'
                     % (i, i, offs, P, I64))
        if half:
            lines.append('    %%rv%d = tt.load %%q%d, %s, %%zeros : %s'
                         % (i, i, mask, P))
            lines.append('    %%v%d = arith.extf %%rv%d : %s to %s'
                         % (i, i, TS, T))
        else:
            lines.append('    %%v%d = tt.load %%q%d, %s, %%zero : %s'
                         % (i, i, mask, P))
    for k in range(len(nodes)):
        node = nodes[k]
        v = nin + k
        if _elementwise(lines, node, v, T, I1):
            continue
        if not is_reduction(node.opcode):
            return ''
        if node.opcode == MAXR:
            lines.append('    %%ninf%d = arith.constant dense<%s> : %s'
                         % (v, COMP_NEG_INF[dt], T))
            init = '%%ninf%d' % v
            combine = 'maximumf'
        else:
            init = '%zero'
            combine = 'addf'
        lines.append('    %%rm%d = arith.select %%mask, %%v%d, %s : %s, %s'
                     % (v, node.a, init, I1, T))
        lines.append('    %%rs%d = "tt.reduce"(%%rm%d) <{axis = 1 : i32}> ({'
                     % (v, v))
        lines.append('    ^bb0(%%x%d: %s, %%y%d: %s):' % (v, C, v, C))
        lines.append('      %%rr%d = arith.%s %%x%d, %%y%d : %s'
                     % (v, combine, v, v, C))
        lines.append('      tt.reduce.return %%rr%d : %s' % (v, C))
        lines.append('    }) : (%s) -> %s' % (T, TR))
        lines.append('    %%re%d = tt.expand_dims %%rs%d {axis = 1 : i32} : '
                     '%s -> %s' % (v, v, TR, TR1))
        lines.append('    %%v%d = tt.broadcast %%re%d : %s -> %s'
                     % (v, v, TR1, T))
        if k != last:
            continue
        sv = '%%rs%d' % v
        if half:
            lines.append('    %%rt%d = arith.truncf %s : %s to %s'
                         % (v, sv, TR, TRS))
            sv = '%%rt%d' % v
        lines.append('    %%por = tt.splat %%out : !tt.ptr<%s> -> %s' % (S, PR))
        lines.append('    %%qor = tt.addptr %%por, %%rows : %s, %s' % (PR, R64))
        lines.append('    tt.store %%qor, %s, %%rmask : %s' % (sv, PR))
    if row_root(kernel, modes):
        lines.append('    %%z0 = arith.constant dense<0> : %s' % I64)
        lines.append('    %%c0 = arith.cmpi eq, %%ar, %%z0 : %s' % I64)
        lines.append('    %%m0 = arith.andi %%c0, %%rowok : %s' % I1)
        lines.append('    %%po = tt.splat %%out : !tt.ptr<%s> -> %s' % (S, P))
        lines.append('    %%qo = tt.addptr %%po, %%rowsb : %s, %s' % (P, I64))
        lines.append('    tt.store %%qo, %s, %%m0 : %s'
                     % (_trunc(lines, 'so', nin + last, half, T, TS), P))
    elif not isred[nin + last]:
        lines.append('    %%po = tt.splat %%out : !tt.ptr<%s> -> %s' % (S, P))
        lines.append('    %%qo = tt.addptr %%po, %%offs : %s, %s' % (P, I64))
        lines.append('    tt.store %%qo, %s, %%mask : %s'
                     % (_trunc(lines, 'so', nin + last, half, T, TS), P))
    for k in range(len(kernel.outputs)):
        ooffs = '%offs'
        if omodes[k] == 1:
            ooffs = '%ar'
        elif omodes[k] == 2:
            lines.append('    %%zo%d = arith.constant dense<0> : %s' % (k, I64))
            ooffs = '%%zo%d' % k
        elif omodes[k] == 3:
            ooffs = '%rowsb'
        lines.append('    %%po%d = tt.splat %%out%d : !tt.ptr<%s> -> %s'
                     % (k, k, S, P))
        lines.append('    %%qo%d = tt.addptr %%po%d, %s : %s, %s'
                     % (k, k, ooffs, P, I64))
        lines.append('    tt.store %%qo%d, %s, %%mask : %s'
                     % (k, _trunc(lines, 'sx%d' % k, kernel.outputs[k], half,
                                  T, TS), P))
    lines.append('    tt.return')
    lines.append('  }')
    lines.append('}')
    return '\n'.join(lines) + '\n'

def _cone(kernel, glob, need, v):
    if v < 0 or glob[v] or need[v]:
        return
    need[v] = True
    nin = nvals(kernel)
    if v >= nin:
        node = kernel.nodes[v - nin]
        _cone(kernel, glob, need, node.a)
        _cone(kernel, glob, need, node.b)


def _tname(glob, v, ph):
    if v < 0:
        return ''
    if glob[v]:
        return '%%v%d' % v
    return '%%w%d_%d' % (v, ph)


def _tile_reduce(lines, v, mask, val, init, combine, T, C, I1):
    lines.append('    %%rm%d = arith.select %s, %s, %s : %s, %s'
                 % (v, mask, val, init, I1, T))
    lines.append('    %%rs%d = "tt.reduce"(%%rm%d) <{axis = 0 : i32}> ({'
                 % (v, v))
    lines.append('    ^bb0(%%x%d: %s, %%y%d: %s):' % (v, C, v, C))
    lines.append('      %%rr%d = arith.%s %%x%d, %%y%d : %s'
                 % (v, combine, v, v, C))
    lines.append('      tt.reduce.return %%rr%d : %s' % (v, C))
    lines.append('    }) : (%s) -> %s' % (T, C))


def _online_pairs(kernel, walks):
    """pair[ph] is the value of a sum of exp(y - m) that can share walk
    ph, the row max m of y (softmax, log-sum-exp), or -1."""
    nodes = kernel.nodes
    nin = nvals(kernel)
    last = len(nodes) - 1
    pair = [-1] * len(walks)
    for ph in range(len(walks) - 1):
        k = walks[ph]
        k2 = walks[ph + 1]
        m = nodes[k]
        sm = nodes[k2]
        if (m.opcode != MAXR or sm.opcode != SUM or sm.p != 1 or
                k2 == last or sm.a < nin):
            continue
        e = nodes[sm.a - nin]
        if e.opcode != EXP or e.a < nin:
            continue
        d = nodes[e.a - nin]
        if (d.opcode == SUB and d.p == BC_R_COL and d.a == m.a and
                d.b == nin + k and (ph == 0 or pair[ph - 1] < 0)):
            pair[ph] = nin + k2
    return pair


def to_ttir_row_tiled(kernel, name, modes):
    """Row mode for rows wider than one tile.  One program still takes one
    row, but walks it a tile at a time: once per row reduction, carrying the
    partial result, and once more for the stores (the last reduction's walk
    when the chain ends in one).  Each walk recomputes the elementwise values
    it needs from the inputs and the reductions already done."""
    nodes = kernel.nodes
    nreal = kernel.ninputs
    nin = nvals(kernel)
    last = len(nodes) - 1
    if last < 0:
        return ''
    BLOCK = config.wide
    dt = kernel.dtype
    S = STORE_TYPE[dt]
    C = COMP_TYPE[dt]
    half = S != C
    T, TS, P, I32, I64, I1 = _tile_types(BLOCK, S, C)
    isred = _row_reductions(kernel, modes, half)
    if not isred:
        return ''
    omodes = out_modes(kernel)
    if len(omodes) != len(kernel.outputs):
        return ''
    rowroot = row_root(kernel, modes)
    if rowroot and len(kernel.outputs) > 0:
        return ''
    params = ['%%in%d: !tt.ptr<%s>' % (i, S) for i in range(nreal)]
    params.append('%%out: !tt.ptr<%s>' % S)
    for k in range(len(kernel.outputs)):
        params.append('%%out%d: !tt.ptr<%s>' % (k, S))
    lines = ['module {',
             '  tt.func public @%s(%s, %%n: i64, %%c: i64) '
             'attributes {noinline = false} {' % (name, ', '.join(params)),
             '    %%zero = arith.constant dense<0.0> : %s' % T,
             '    %%one = arith.constant dense<1.0> : %s' % T,
             '    %%ninf = arith.constant dense<%s> : %s' % (COMP_NEG_INF[dt],
                                                          T),
             '    %pid = tt.get_program_id x : i32',
             '    %rowi = arith.extsi %pid : i32 to i64',
             '    %%range = tt.make_range {end = %d : i32, start = 0 : i32} '
             ': %s' % (BLOCK, I32),
             '    %%ar = arith.extsi %%range : %s to %s' % (I32, I64),
             '    %%cs = tt.splat %%c : i64 -> %s' % I64,
             '    %base = arith.muli %rowi, %c : i64',
             '    %%bases = tt.splat %%base : i64 -> %s' % I64,
             '    %c32 = arith.trunci %c : i64 to i32',
             '    %tlo = arith.constant 0 : i32',
             '    %%tb = arith.constant %d : i32' % BLOCK,
             '    %%sone = arith.constant 1.0 : %s' % C]
    if half:
        lines.append('    %%zeros = arith.constant dense<0.0> : %s' % TS)
    if not _emit_consts(lines, kernel, nreal, T):
        return ''
    glob = [False] * (nin + len(nodes))
    for i in range(nreal, nin):
        glob[i] = True
    for i in range(nreal):
        if modes[i] == 2 or modes[i] == 3:
            glob[i] = True
            src = '%%in%d' % i
            if modes[i] == 3:
                lines.append('    %%sp%d = tt.addptr %%in%d, %%pid : '
                             '!tt.ptr<%s>, i32' % (i, i, S))
                src = '%%sp%d' % i
            lines.append('    %%sv%d = tt.load %s : !tt.ptr<%s>' % (i, src, S))
            sv = '%%sv%d' % i
            if half:
                lines.append('    %%se%d = arith.extf %s : %s to %s'
                             % (i, sv, S, C))
                sv = '%%se%d' % i
            lines.append('    %%v%d = tt.splat %s : %s -> %s' % (i, sv, C, T))
        else:
            lines.append('    %%p%d = tt.splat %%in%d : !tt.ptr<%s> -> %s'
                         % (i, i, S, P))
    if not isred[nin + last] and not rowroot:
        lines.append('    %%po = tt.splat %%out : !tt.ptr<%s> -> %s' % (S, P))
    for k in range(len(kernel.outputs)):
        lines.append('    %%po%d = tt.splat %%out%d : !tt.ptr<%s> -> %s'
                     % (k, k, S, P))
        if omodes[k] == 2:
            lines.append('    %%zo%d = arith.constant dense<0> : %s' % (k, I64))
        elif omodes[k] == 3:
            lines.append('    %%ro%d = tt.splat %%rowi : i64 -> %s' % (k, I64))
    walks = []
    for k in range(len(nodes)):
        if isred[nin + k]:
            walks.append(k)
    if not isred[nin + last] and not rowroot:
        walks.append(last)
    pair = _online_pairs(kernel, walks)
    for ph in range(len(walks)):
        if ph > 0 and pair[ph - 1] >= 0:
            continue
        k = walks[ph]
        v = nin + k
        node = nodes[k]
        red = isred[v]
        final = k == last
        need = [False] * (nin + len(nodes))
        _cone(kernel, glob, need, node.a if red else v)
        if final:
            for j in range(len(kernel.outputs)):
                _cone(kernel, glob, need, kernel.outputs[j])
        maxr = red and node.opcode == MAXR
        v2 = pair[ph]
        if v2 >= 0:
            lines.append('    %%ri%d = arith.constant %s : %s'
                         % (v, COMP_NEG_INF[dt], C))
            lines.append('    %%ri%d = arith.constant 0.0 : %s' % (v2, C))
            lines.append('    %%r%d:2 = scf.for %%t%d = %%tlo to %%c32 step '
                         '%%tb iter_args(%%a%d = %%ri%d, %%a%d = %%ri%d) -> '
                         '(%s, %s) : i32 {' % (v, ph, v, v, v2, v2, C, C))
        elif red:
            init = COMP_NEG_INF[dt] if maxr else '0.0'
            lines.append('    %%ri%d = arith.constant %s : %s' % (v, init, C))
            lines.append('    %%r%d = scf.for %%t%d = %%tlo to %%c32 step %%tb '
                         'iter_args(%%a%d = %%ri%d) -> (%s) : i32 {'
                         % (v, ph, v, v, C))
        else:
            lines.append('    scf.for %%t%d = %%tlo to %%c32 step %%tb : i32 {'
                         % ph)
        lines.append('    %%t64_%d = arith.extsi %%t%d : i32 to i64' % (ph, ph))
        lines.append('    %%ts_%d = tt.splat %%t64_%d : i64 -> %s'
                     % (ph, ph, I64))
        col = '%%col_%d' % ph
        lines.append('    %s = arith.addi %%ar, %%ts_%d : %s' % (col, ph, I64))
        mask = '%%m_%d' % ph
        lines.append('    %s = arith.cmpi slt, %s, %%cs : %s'
                     % (mask, col, I64))
        offs = '%%o_%d' % ph
        lines.append('    %s = arith.addi %%bases, %s : %s' % (offs, col, I64))
        for i in range(nreal):
            if not need[i]:
                continue
            q = '%%q%d_%d' % (i, ph)
            lines.append('    %s = tt.addptr %%p%d, %s : %s, %s'
                         % (q, i, col if modes[i] == 1 else offs, P, I64))
            dst = _tname(glob, i, ph)
            if half:
                lines.append('    %%rv%d_%d = tt.load %s, %s, %%zeros : %s'
                             % (i, ph, q, mask, P))
                lines.append('    %s = arith.extf %%rv%d_%d : %s to %s'
                             % (dst, i, ph, TS, T))
            else:
                lines.append('    %s = tt.load %s, %s, %%zero : %s'
                             % (dst, q, mask, P))
        for j in range(len(nodes)):
            w = nin + j
            if not need[w]:
                continue
            nj = nodes[j]
            if not _ew(lines, nj.opcode, nj.p, _tname(glob, w, ph),
                       '%%c%d_%d' % (w, ph), _tname(glob, nj.a, ph),
                       _tname(glob, nj.b, ph), T, I1):
                return ''
        if final:
            for j in range(len(kernel.outputs)):
                ooffs = offs
                if omodes[j] == 1:
                    ooffs = col
                elif omodes[j] == 2:
                    ooffs = '%%zo%d' % j
                elif omodes[j] == 3:
                    ooffs = '%%ro%d' % j
                val = _tname(glob, kernel.outputs[j], ph)
                if half:
                    lines.append('    %%sx%d_%d = arith.truncf %s : %s to %s'
                                 % (j, ph, val, T, TS))
                    val = '%%sx%d_%d' % (j, ph)
                lines.append('    %%qo%d_%d = tt.addptr %%po%d, %s : %s, %s'
                             % (j, ph, j, ooffs, P, I64))
                lines.append('    tt.store %%qo%d_%d, %s, %s : %s'
                             % (j, ph, val, mask, P))
        if not red:
            val = _tname(glob, v, ph)
            if half:
                lines.append('    %%so_%d = arith.truncf %s : %s to %s'
                             % (ph, val, T, TS))
                val = '%%so_%d' % ph
            lines.append('    %%qo_%d = tt.addptr %%po, %s : %s, %s'
                         % (ph, offs, P, I64))
            lines.append('    tt.store %%qo_%d, %s, %s : %s'
                         % (ph, val, mask, P))
            lines.append('    }')
            continue
        y = _tname(glob, node.a, ph)
        combine = 'maximumf' if maxr else 'addf'
        _tile_reduce(lines, v, mask, y, '%ninf' if maxr else '%zero',
                     combine, T, C, I1)
        if v2 >= 0:
            lines.append('    %%mn%d = arith.maximumf %%a%d, %%rs%d : %s'
                         % (v, v, v, C))
            lines.append('    %%eq%d = arith.cmpf oeq, %%a%d, %%mn%d : %s'
                         % (v, v, v, C))
            lines.append('    %%dd%d = arith.subf %%a%d, %%mn%d : %s'
                         % (v, v, v, C))
            lines.append('    %%xe%d = math.exp %%dd%d : %s' % (v, v, C))
            lines.append('    %%al%d = arith.select %%eq%d, %%sone, %%xe%d : '
                         '%s' % (v, v, v, C))
            lines.append('    %%ms%d = tt.splat %%mn%d : %s -> %s'
                         % (v, v, C, T))
            lines.append('    %%sb%d = arith.subf %s, %%ms%d : %s'
                         % (v, y, v, T))
            lines.append('    %%ex%d = math.exp %%sb%d : %s' % (v, v, T))
            _tile_reduce(lines, v2, mask, '%%ex%d' % v, '%zero', 'addf', T,
                         C, I1)
            lines.append('    %%sc%d = arith.mulf %%a%d, %%al%d : %s'
                         % (v2, v2, v, C))
            lines.append('    %%ac%d = arith.addf %%sc%d, %%rs%d : %s'
                         % (v2, v2, v2, C))
            lines.append('    scf.yield %%mn%d, %%ac%d : %s, %s'
                         % (v, v2, C, C))
            lines.append('    }')
            glob[v] = True
            glob[v2] = True
            lines.append('    %%v%d = tt.splat %%r%d#0 : %s -> %s'
                         % (v, v, C, T))
            lines.append('    %%v%d = tt.splat %%r%d#1 : %s -> %s'
                         % (v2, v, C, T))
            continue
        lines.append('    %%ac%d = arith.%s %%a%d, %%rs%d : %s'
                     % (v, combine, v, v, C))
        lines.append('    scf.yield %%ac%d : %s' % (v, C))
        lines.append('    }')
        glob[v] = True
        lines.append('    %%v%d = tt.splat %%r%d : %s -> %s' % (v, v, C, T))
        if not final:
            continue
        sv = '%%r%d' % v
        if half:
            lines.append('    %%rt%d = arith.truncf %s : %s to %s'
                         % (v, sv, C, S))
            sv = '%%rt%d' % v
        if node.p == 1:
            lines.append('    %%pr = tt.addptr %%out, %%pid : !tt.ptr<%s>, i32'
                         % S)
            lines.append('    tt.store %%pr, %s : !tt.ptr<%s>' % (sv, S))
        else:
            lines.append('    %true = arith.constant true')
            lines.append('    %%o = tt.atomic_rmw fadd, acq_rel, gpu, %%out, '
                         '%s, %%true : (!tt.ptr<%s>, %s, i1) -> %s'
                         % (sv, S, S, S))
    if rowroot:
        need = [False] * (nin + len(nodes))
        _cone(kernel, glob, need, nin + last)
        for j in range(len(nodes)):
            w = nin + j
            if not need[w]:
                continue
            nj = nodes[j]
            if not _ew(lines, nj.opcode, nj.p, '%%v%d' % w, '%%c%d' % w,
                       '%%v%d' % nj.a, '%%v%d' % nj.b, T, I1):
                return ''
            glob[w] = True
        lines.append('    %%pr = tt.addptr %%out, %%pid : !tt.ptr<%s>, i32' % S)
        lines.append('    %%po = tt.splat %%pr : !tt.ptr<%s> -> %s' % (S, P))
        lines.append('    %%z0 = arith.constant dense<0> : %s' % I64)
        lines.append('    %%m0 = arith.cmpi eq, %%ar, %%z0 : %s' % I64)
        lines.append('    tt.store %%po, %s, %%m0 : %s'
                     % (_trunc(lines, 'so', nin + last, half, T, TS), P))
    lines.append('    tt.return')
    lines.append('  }')
    lines.append('}')
    return '\n'.join(lines) + '\n'

def to_ttir_flat(kernel, name):
    nodes = kernel.nodes
    nreal = kernel.ninputs
    nin = nvals(kernel)
    BLOCK = flat_block(kernel)
    masked = kernel.n == 0 or kernel.n % BLOCK != 0
    dt = kernel.dtype
    S = STORE_TYPE[dt]
    C = COMP_TYPE[dt]
    half = S != C
    T, TS, P, I32, I64, I1 = _tile_types(BLOCK, S, C)
    modes = input_modes(kernel)
    if len(modes) != nreal:
        return ''
    omodes = out_modes(kernel)
    if len(omodes) != len(kernel.outputs):
        return ''
    axis = AXIS_ALL
    if kernel.sumroot:
        axis = nodes[len(nodes) - 1].p
    need_mod = axis == 0
    need_div = axis == 1
    need_zero_off = False
    for i in range(nreal):
        if modes[i] == 1:
            need_mod = True
        elif modes[i] == 2:
            need_zero_off = True
        elif modes[i] == 3:
            need_div = True
    for k in range(len(omodes)):
        if omodes[k] == 1:
            need_mod = True
        elif omodes[k] == 2:
            need_zero_off = True
        elif omodes[k] == 3:
            need_div = True
    if half and kernel.sumroot and nodes[len(nodes) - 1].opcode == SUM:
        return ''
    params = ['%%in%d: !tt.ptr<%s>' % (i, S) for i in range(nreal)]
    params.append('%%out: !tt.ptr<%s>' % S)
    for k in range(len(kernel.outputs)):
        params.append('%%out%d: !tt.ptr<%s>' % (k, S))
    lines = ['module {',
             '  tt.func public @%s(%s, %%n: i64, %%c: i64) '
             'attributes {noinline = false} {' % (name, ', '.join(params)),
             '    %%zero = arith.constant dense<0.0> : %s' % T,
             '    %%one = arith.constant dense<1.0> : %s' % T]
    if not _emit_consts(lines, kernel, nreal, T):
        return ''
    _flat_prologue(lines, BLOCK, I32, I64)
    if need_mod or need_div:
        lines.append('    %%cs = tt.splat %%c : i64 -> %s' % I64)
    if need_mod:
        lines.append('    %%offsm = arith.remsi %%offs64, %%cs : %s' % I64)
    if need_div:
        lines.append('    %%offsd = arith.divsi %%offs64, %%cs : %s' % I64)
    if need_zero_off:
        lines.append('    %%zoffs = arith.constant dense<0> : %s' % I32)
    if masked:
        tmask = '%mask'
    else:
        tmask = '%tmask'
        lines.append('    %%tmask = arith.constant dense<true> : %s' % I1)
    if half:
        lines.append('    %%zeros = arith.constant dense<0.0> : %s' % TS)
    for i in range(nreal):
        lines.append('    %%p%d = tt.splat %%in%d : !tt.ptr<%s> -> %s'
                     % (i, i, S, P))
        if modes[i] == 1:
            lines.append('    %%q%d = tt.addptr %%p%d, %%offsm : %s, %s'
                         % (i, i, P, I64))
        elif modes[i] == 2:
            lines.append('    %%q%d = tt.addptr %%p%d, %%zoffs : %s, %s'
                         % (i, i, P, I32))
        elif modes[i] == 3:
            lines.append('    %%q%d = tt.addptr %%p%d, %%offsd : %s, %s'
                         % (i, i, P, I64))
        else:
            lines.append('    %%q%d = tt.addptr %%p%d, %%offs : %s, %s'
                         % (i, i, P, I32))
        dst = '%%v%d' % i
        if half:
            dst = '%%rv%d' % i
        if masked:
            lines.append('    %s = tt.load %%q%d, %%mask, %s : %s'
                         % (dst, i, '%zeros' if half else '%zero', P))
        else:
            lines.append('    %s = tt.load %%q%d : %s' % (dst, i, P))
        if half:
            lines.append('    %%v%d = arith.extf %%rv%d : %s to %s'
                         % (i, i, TS, T))
    last = nin + len(nodes) - 1
    for k in range(len(nodes)):
        node = nodes[k]
        v = nin + k
        if _elementwise(lines, node, v, T, I1):
            continue
        if k != len(nodes) - 1 or not is_reduction(node.opcode):
            return ''
        if node.opcode == MAXR:
            if axis != AXIS_ALL or kernel.n == 0 or kernel.n > BLOCK:
                return ''
            src = '%%v%d' % node.a
            if masked:
                lines.append('    %%ninf = arith.constant dense<%s> : %s'
                             % (COMP_NEG_INF[dt], T))
                lines.append('    %%m%d = arith.select %%mask, %%v%d, %%ninf '
                             ': %s, %s' % (v, node.a, I1, T))
                src = '%%m%d' % v
            lines.append('    %%v%d = "tt.reduce"(%s) <{axis = 0 : i32}> ({'
                         % (v, src))
            lines.append('    ^bb0(%%x: %s, %%y: %s):' % (C, C))
            lines.append('      %%r = arith.maximumf %%x, %%y : %s' % C)
            lines.append('      tt.reduce.return %%r : %s' % C)
            lines.append('    }) : (%s) -> %s' % (T, C))
            out = '%%v%d' % v
            if half:
                lines.append('    %%mt%d = arith.truncf %s : %s to %s'
                             % (v, out, C, S))
                out = '%%mt%d' % v
            lines.append('    tt.store %%out, %s : !tt.ptr<%s>' % (out, S))
        else:
            if axis == 0 or axis == 1:
                offs = '%offsm'
                if axis == 1:
                    offs = '%offsd'
                lines.append('    %%po = tt.splat %%out : !tt.ptr<%s> -> %s'
                             % (S, P))
                lines.append('    %%qo = tt.addptr %%po, %s : %s, %s'
                             % (offs, P, I64))
                lines.append('    %%v%d = tt.atomic_rmw fadd, acq_rel, gpu, '
                             '%%qo, %%v%d, %s : (%s, %s, %s) -> %s'
                             % (v, node.a, tmask, P, T, I1, T))
            else:
                # Masked tail lanes are zero only if every op maps 0 to 0,
                # which a folded constant breaks, so zero them before reducing.
                src = '%%v%d' % node.a
                if masked:
                    lines.append('    %%rm%d = arith.select %%mask, %%v%d, '
                                 '%%zero : %s, %s' % (v, node.a, I1, T))
                    src = '%%rm%d' % v
                lines.append('    %%v%d = "tt.reduce"(%s) <{axis = 0 : i32}> ({'
                             % (v, src))
                lines.append('    ^bb0(%%x: %s, %%y: %s):' % (C, C))
                lines.append('      %%r = arith.addf %%x, %%y : %s' % C)
                lines.append('      tt.reduce.return %%r : %s' % C)
                lines.append('    }) : (%s) -> %s' % (T, C))
                lines.append('    %true = arith.constant true')
                lines.append('    %%o = tt.atomic_rmw fadd, acq_rel, gpu, '
                             '%%out, %%v%d, %%true : (!tt.ptr<%s>, %s, i1) '
                             '-> %s' % (v, S, C, C))
    if not kernel.sumroot:
        lines.append('    %%po = tt.splat %%out : !tt.ptr<%s> -> %s' % (S, P))
        lines.append('    %%qo = tt.addptr %%po, %%offs : %s, %s' % (P, I32))
        val = _trunc(lines, 'so', last, half, T, TS)
        if masked:
            lines.append('    tt.store %%qo, %s, %%mask : %s' % (val, P))
        else:
            lines.append('    tt.store %%qo, %s : %s' % (val, P))
    for k in range(len(kernel.outputs)):
        ooffs = '%offs'
        OI = I32
        if omodes[k] == 1:
            ooffs = '%offsm'
            OI = I64
        elif omodes[k] == 2:
            ooffs = '%zoffs'
        elif omodes[k] == 3:
            ooffs = '%offsd'
            OI = I64
        lines.append('    %%po%d = tt.splat %%out%d : !tt.ptr<%s> -> %s'
                     % (k, k, S, P))
        lines.append('    %%qo%d = tt.addptr %%po%d, %s : %s, %s'
                     % (k, k, ooffs, P, OI))
        val = _trunc(lines, 'sx%d' % k, kernel.outputs[k], half, T, TS)
        if masked:
            lines.append('    tt.store %%qo%d, %s, %%mask : %s' % (k, val, P))
        else:
            lines.append('    tt.store %%qo%d, %s : %s' % (k, val, P))
    lines.append('    tt.return')
    lines.append('  }')
    lines.append('}')
    return '\n'.join(lines) + '\n'

def _gdivv(e, a, v, ty):
    if v > 0:
        return _gdiv(e, a, v, ty)
    return _gbin(e, 'divsi', a, '%rd', ty)


def _gmodv(e, a, v, ty):
    if v > 0:
        return _gmod(e, a, v, ty)
    return _gbin(e, 'remsi', a, '%rd', ty)


def _gmulv(e, a, v, ty):
    if v > 0:
        return _gmul(e, a, v, ty)
    return _gbin(e, 'muli', a, '%rd', ty)


def _gather_src(e, p, idx, rd, I64):
    """Where output lane `idx` of a GATHER node reads its operand.  `rd` is
    rows * dh: a constant when the kernel's size is, else 0 for the %rd
    splat of n / heads.  Dividing by a constant is a multiply and a shift;
    dividing by %rd is a real division in every lane."""
    kind = gather_kind(p)
    dh = gather_dh(p)
    if kind == GA_ROTHALF:
        blk = _gdiv(e, idx, dh, I64)
        c = _gmod(e, idx, dh, I64)
        rc = _gmod(e, _gaddc(e, c, dh // 2, I64), dh, I64)
        return _gbin(e, 'addi', _gmul(e, blk, dh, I64), rc, I64)
    hd = gather_heads(p) * dh
    if kind == GA_HEADSPLIT:
        oi = _gdivv(e, idx, rd, I64)
        rem = _gmodv(e, idx, rd, I64)
        return _gbin(e, 'addi', _gbin(e, 'addi',
                                      _gmul(e, _gdiv(e, rem, dh, I64), hd, I64),
                                      _gmul(e, oi, dh, I64), I64),
                     _gmod(e, rem, dh, I64), I64)
    oi = _gdiv(e, idx, hd, I64)
    rem = _gmod(e, idx, hd, I64)
    return _gbin(e, 'addi', _gbin(e, 'addi',
                                  _gmulv(e, _gdiv(e, rem, dh, I64), rd, I64),
                                  _gmul(e, oi, dh, I64), I64),
                 _gmod(e, rem, dh, I64), I64)


class Gathered(object):
    """Emits a flat kernel whose nodes may include GATHERs.  A value is
    emitted once per index it is read at: a gather reads its operand at a
    permuted index, and everything under it, leaves included, is evaluated
    there, so the permutation costs addressing and nothing else."""

    def __init__(self, kernel, e, modes, T, P, IX, I1, S, half, rd):
        self.rd = rd
        self.kernel = kernel
        self.e = e
        self.modes = modes
        self.T = T
        self.P = P
        self.IX = IX
        self.I1 = I1
        self.S = S
        self.half = half
        self.memo = {}

    def val(self, v, idx, pred='%mask'):
        key = '%d@%s@%s' % (v, idx, pred)
        r = self.memo.get(key, '')
        if r:
            return r
        r = self._val(v, idx, pred)
        self.memo[key] = r
        return r

    def _val(self, v, idx, pred):
        kernel = self.kernel
        e = self.e
        nreal = kernel.ninputs
        nin = nvals(kernel)
        if v >= nreal and v < nin:
            return '%%v%d' % v
        if v < nreal:
            mode = self.modes[v]
            if mode == 1:
                off = _gbin(e, 'remsi', idx, '%cs', self.IX)
            elif mode == 3:
                off = _gbin(e, 'divsi', idx, '%cs', self.IX)
            elif mode == 2:
                off = _gconst(e, 0, self.IX)
            else:
                off = idx
            q = e.tmp()
            e.add('    %s = tt.addptr %%p%d, %s : %s, %s'
                  % (q, v, off, self.P, self.IX))
            r = e.tmp()
            if self.half:
                e.add('    %s = tt.load %s, %s, %%zeros : %s'
                      % (r, q, pred, self.P))
                x = e.tmp()
                e.add('    %s = arith.extf %s : %s to %s'
                      % (x, r, 'tensor<%dx%s>' % (flat_block(self.kernel),
                                                  self.S), self.T))
                return x
            e.add('    %s = tt.load %s, %s, %%zero : %s'
                  % (r, q, pred, self.P))
            return r
        node = kernel.nodes[v - nin]
        if is_view(node):
            return self._view(v, node, idx, pred)
        if node.opcode == GATHER:
            src = _gather_src(e, node.p, idx, self.rd, self.IX)
            return self.val(node.a, src, pred)
        a = self.val(node.a, idx, pred)
        if node.opcode == BINARY and (binary_fn(node.p) == B_KEEP_NZ or
                                      binary_fn(node.p) == B_KEEP_Z):
            if not a:
                return ''
            cond = e.tmp()
            e.add('    %s = arith.cmpf %s, %s, %%zero : %s'
                  % (cond, 'une' if binary_fn(node.p) == B_KEEP_NZ else 'oeq',
                     a, self.T))
            b = self.val(node.b, idx, _gbin(e, 'andi', pred, cond, self.I1))
            if not b:
                return ''
            r = e.tmp()
            e.add('    %s = arith.select %s, %s, %%zero : %s, %s'
                  % (r, cond, b, self.I1, self.T))
            return r
        b = ''
        if node.b >= 0:
            b = self.val(node.b, idx, pred)
        r = e.tmp()
        if not _ew(e.lines, node.opcode, node.p, r, e.tmp(), a, b, self.T,
                   self.I1):
            return ''
        return r


    def _view(self, v, node, idx, pred):
        e = self.e
        IX = self.IX
        mode = self.modes[v]
        if mode == 1:
            idx = _gbin(e, 'remsi', idx, '%cs', IX)
        elif mode == 3:
            idx = _gbin(e, 'divsi', idx, '%cs', IX)
        elif mode == 2:
            idx = _gconst(e, 0, IX)
        vid = view_of(node.p)
        params = view_params(vid)
        rank = params[1]
        shape = [params[2 + j] for j in range(rank)]
        index = [''] * rank
        addr = ''
        ok = ''
        for j in range(rank):
            st = params[2 + rank + j]
            lo = params[2 + 2 * rank + j]
            hi = params[2 + 3 * rank + j]
            if st != 0:
                x = _coord(e, index, shape, j, idx, IX)
                if st != 1:
                    x = _gmul(e, x, st, IX)
                addr = _gbin(e, 'addi', addr, x, IX) if addr else x
            if lo > 0:
                ok = _and(e, ok, _gcmp(e, 'sge', _coord(e, index, shape, j,
                                                        idx, IX),
                                       _gconst(e, lo, IX), IX), self.I1)
            if hi < shape[j]:
                ok = _and(e, ok, _gcmp(e, 'slt', _coord(e, index, shape, j,
                                                        idx, IX),
                                       _gconst(e, hi, IX), IX), self.I1)
        off = view_offset(vid)
        if not addr:
            addr = _gconst(e, off, IX)
        elif off != 0:
            addr = _gaddc(e, addr, off, IX)
        if node.b >= 0:
            d = self.val(node.b, idx, pred)
            if not d:
                return ''
            di = e.tmp()
            e.add('    %s = arith.fptosi %s : %s to %s' % (di, d, self.T, IX))
            addr = _gbin(e, 'addi', addr, di, IX)
        clamp = view_clamp(vid)
        if clamp > 0:
            addr = _gclamp(e, addr, clamp - 1, IX)
        if ok:
            r = e.tmp()
            e.add('    %s = arith.select %s, %s, %s : %s, %s'
                  % (r, ok, addr, _gconst(e, 0, IX), self.I1, IX))
            addr = r
            pred = _gbin(e, 'andi', pred, ok, self.I1)
        x = self.val(node.a, addr, pred)
        if not x or not ok:
            return x
        r = e.tmp()
        e.add('    %s = arith.select %s, %s, %%zero : %s, %s'
              % (r, ok, x, self.I1, self.T))
        return r


def views_fit_i32(kernel):
    for k in range(len(kernel.nodes)):
        node = kernel.nodes[k]
        if is_view(node):
            vid = view_of(node.p)
            if not strided_fits_i32(view_params_list(vid)):
                return False
            if abs(view_offset(vid)) >= 1 << 30:
                return False
    return True


def to_ttir_gathered(kernel, name, modes):
    nodes = kernel.nodes
    nreal = kernel.ninputs
    nin = nvals(kernel)
    last = len(nodes) - 1
    if last < 0:
        return ''
    omodes = out_modes(kernel)
    if len(omodes) != len(kernel.outputs):
        return ''
    for k in range(len(omodes)):
        if omodes[k] != 0:
            return ''
    BLOCK = flat_block(kernel)
    dt = kernel.dtype
    S = STORE_TYPE[dt]
    C = COMP_TYPE[dt]
    half = S != C
    T, TS, P, I32, I64, I1 = _tile_types(BLOCK, S, C)
    reshaped = False
    heads = 1
    if not views_fit_i32(kernel):
        return ''
    for k in range(len(nodes)):
        node = nodes[k]
        if is_view(node):
            continue
        if node.opcode == GATHER:
            if gather_changes_shape(node.p):
                if reshaped and heads != gather_heads(node.p):
                    return ''
                reshaped = True
                heads = gather_heads(node.p)
        elif is_reduction(node.opcode):
            if k != last or node.p != AXIS_ALL:
                return ''
            if node.opcode == MAXR and (kernel.n == 0 or kernel.n > BLOCK):
                return ''
            if half and node.opcode == SUM:
                return ''
    for i in range(nreal):
        # A row- or column-broadcast operand is addressed by the column count
        # of the big input, which a reshaping gather makes ambiguous.
        if reshaped and (modes[i] == 1 or modes[i] == 3):
            return ''
    e = Emitter()
    e.add('module {')
    params = ['%%in%d: !tt.ptr<%s>' % (i, S) for i in range(nreal)]
    params.append('%%out: !tt.ptr<%s>' % S)
    for k in range(len(kernel.outputs)):
        params.append('%%out%d: !tt.ptr<%s>' % (k, S))
    e.add('  tt.func public @%s(%s, %%n: i64, %%c: i64) '
          'attributes {noinline = false} {' % (name, ', '.join(params)))
    e.add('    %%zero = arith.constant dense<0.0> : %s' % T)
    e.add('    %%one = arith.constant dense<1.0> : %s' % T)
    if not _emit_consts(e.lines, kernel, nreal, T):
        return ''
    _flat_prologue(e.lines, BLOCK, I32, I64)
    # Lane indices are i32: a flat kernel's size is below 2**31, and 64-bit
    # division and remainder cost several times the 32-bit ones.
    e.add('    %c32 = arith.trunci %c : i64 to i32')
    e.add('    %%cs = tt.splat %%c32 : i32 -> %s' % I32)
    rd = 0
    if kernel.n > 0 and kernel.n % heads == 0:
        rd = kernel.n // heads
    else:
        e.add('    %%hc = arith.constant %d : i64' % heads)
        e.add('    %nh = arith.divsi %n, %hc : i64')
        e.add('    %nh32 = arith.trunci %nh : i64 to i32')
        e.add('    %%rd = tt.splat %%nh32 : i32 -> %s' % I32)
    if half:
        e.add('    %%zeros = arith.constant dense<0.0> : %s' % TS)
    for i in range(nreal):
        e.add('    %%p%d = tt.splat %%in%d : !tt.ptr<%s> -> %s' % (i, i, S, P))
    g = Gathered(kernel, e, modes, T, P, I32, I1, S, half, rd)
    root = nodes[last]
    if is_reduction(root.opcode):
        x = g.val(root.a, '%offs')
        if not x:
            return ''
        init = '%zero'
        combine = 'addf'
        if root.opcode == MAXR:
            e.add('    %%ninf = arith.constant dense<%s> : %s'
                  % (COMP_NEG_INF[dt], T))
            init = '%ninf'
            combine = 'maximumf'
        m = e.tmp()
        e.add('    %s = arith.select %%mask, %s, %s : %s, %s'
              % (m, x, init, I1, T))
        r = e.tmp()
        e.add('    %s = "tt.reduce"(%s) <{axis = 0 : i32}> ({' % (r, m))
        e.add('    ^bb0(%%x: %s, %%y: %s):' % (C, C))
        e.add('      %%r = arith.%s %%x, %%y : %s' % (combine, C))
        e.add('      tt.reduce.return %%r : %s' % C)
        e.add('    }) : (%s) -> %s' % (T, C))
        if root.opcode == MAXR:
            if half:
                t = e.tmp()
                e.add('    %s = arith.truncf %s : %s to %s' % (t, r, C, S))
                r = t
            e.add('    tt.store %%out, %s : !tt.ptr<%s>' % (r, S))
        else:
            e.add('    %true = arith.constant true')
            e.add('    %%o = tt.atomic_rmw fadd, acq_rel, gpu, %%out, %s, '
                  '%%true : (!tt.ptr<%s>, %s, i1) -> %s' % (r, S, C, C))
    else:
        x = g.val(nin + last, '%offs')
        if not x:
            return ''
        if half:
            t = e.tmp()
            e.add('    %s = arith.truncf %s : %s to %s' % (t, x, T, TS))
            x = t
        e.add('    %%po = tt.splat %%out : !tt.ptr<%s> -> %s' % (S, P))
        e.add('    %%qo = tt.addptr %%po, %%offs : %s, %s' % (P, I32))
        e.add('    tt.store %%qo, %s, %%mask : %s' % (x, P))
    # An extra output is a node's own value, so it is read at the lane's own
    # index even when the root reads it through a gather.
    for k in range(len(kernel.outputs)):
        x = g.val(kernel.outputs[k], '%offs')
        if not x:
            return ''
        if half:
            t = e.tmp()
            e.add('    %s = arith.truncf %s : %s to %s' % (t, x, T, TS))
            x = t
        po = e.tmp()
        e.add('    %s = tt.splat %%out%d : !tt.ptr<%s> -> %s' % (po, k, S, P))
        qo = e.tmp()
        e.add('    %s = tt.addptr %s, %%offs : %s, %s' % (qo, po, P, I32))
        e.add('    tt.store %s, %s, %%mask : %s' % (qo, x, P))
    e.add('    tt.return')
    e.add('  }')
    e.add('}')
    return '\n'.join(e.lines) + '\n'


class Emitter(object):
    def __init__(self):
        self.lines = []
        self.k = 0

    def tmp(self):
        self.k += 1
        return '%%t%d' % self.k

    def add(self, line):
        self.lines.append(line)

def _gconst(e, v, ty):
    r = e.tmp()
    e.add('    %s = arith.constant dense<%d> : %s' % (r, v, ty))
    return r


def _gbin(e, op, a, b, ty):
    r = e.tmp()
    e.add('    %s = arith.%s %s, %s : %s' % (r, op, a, b, ty))
    return r


def _gcmp(e, pred, a, b, ty):
    r = e.tmp()
    e.add('    %s = arith.cmpi %s, %s, %s : %s' % (r, pred, a, b, ty))
    return r


def _gdiv(e, a, v, ty):
    return _gbin(e, 'divsi', a, _gconst(e, v, ty), ty)


def _gmod(e, a, v, ty):
    return _gbin(e, 'remsi', a, _gconst(e, v, ty), ty)


def _gmul(e, a, v, ty):
    return _gbin(e, 'muli', a, _gconst(e, v, ty), ty)


def _gaddc(e, a, v, ty):
    return _gbin(e, 'addi', a, _gconst(e, v, ty), ty)


def _gclamp(e, a, hi, ty):
    a = _gbin(e, 'maxsi', a, _gconst(e, 0, ty), ty)
    return _gbin(e, 'minsi', a, _gconst(e, hi, ty), ty)

def _gather_index(e, op, params, I64, I1):
    off = '%offs64'
    srcs = []
    if op == GA_IM2COL:
        n, c, h, w, k, pad, stride = (params[0], params[1], params[2],
                                      params[3], params[4], params[5],
                                      params[6])
        hw = h * w
        kk = k * k
        ckk = c * kk
        oh = (h + 2 * pad - k) // stride + 1
        ow = (w + 2 * pad - k) // stride + 1
        ohw = oh * ow
        row = _gdiv(e, off, ckk, I64)
        col = _gmod(e, off, ckk, I64)
        img = _gdiv(e, row, ohw, I64)
        pos = _gmod(e, row, ohw, I64)
        ph = _gmul(e, _gdiv(e, pos, ow, I64), stride, I64)
        pw = _gmul(e, _gmod(e, pos, ow, I64), stride, I64)
        ch = _gdiv(e, col, kk, I64)
        rk = _gmod(e, col, kk, I64)
        ih = _gaddc(e, _gbin(e, 'addi', ph, _gdiv(e, rk, k, I64), I64),
                    -pad, I64)
        iw = _gaddc(e, _gbin(e, 'addi', pw, _gmod(e, rk, k, I64), I64),
                    -pad, I64)
        zero = _gconst(e, 0, I64)
        ok = _gbin(e, 'andi',
                   _gbin(e, 'andi', _gcmp(e, 'sge', ih, zero, I64),
                         _gcmp(e, 'slt', ih, _gconst(e, h, I64), I64), I1),
                   _gbin(e, 'andi', _gcmp(e, 'sge', iw, zero, I64),
                         _gcmp(e, 'slt', iw, _gconst(e, w, I64), I64), I1), I1)
        base = _gbin(e, 'addi', _gmul(e, img, c * hw, I64),
                     _gmul(e, ch, hw, I64), I64)
        src = _gbin(e, 'addi', base,
                    _gbin(e, 'addi', _gmul(e, ih, w, I64), iw, I64), I64)
        srcs.append(src)
        return srcs, ok
    if op == GA_IM2COL_NHWC:
        n, c, h, w, k, pad, stride = (params[0], params[1], params[2],
                                      params[3], params[4], params[5],
                                      params[6])
        kk = k * k
        kkc = kk * c
        oh = (h + 2 * pad - k) // stride + 1
        ow = (w + 2 * pad - k) // stride + 1
        ohw = oh * ow
        row = _gdiv(e, off, kkc, I64)
        col = _gmod(e, off, kkc, I64)
        ci = _gmod(e, col, c, I64)
        rk = _gdiv(e, col, c, I64)
        img = _gdiv(e, row, ohw, I64)
        pos = _gmod(e, row, ohw, I64)
        ph = _gmul(e, _gdiv(e, pos, ow, I64), stride, I64)
        pw = _gmul(e, _gmod(e, pos, ow, I64), stride, I64)
        ih = _gaddc(e, _gbin(e, 'addi', ph, _gdiv(e, rk, k, I64), I64),
                    -pad, I64)
        iw = _gaddc(e, _gbin(e, 'addi', pw, _gmod(e, rk, k, I64), I64),
                    -pad, I64)
        zero = _gconst(e, 0, I64)
        ok = _gbin(e, 'andi',
                   _gbin(e, 'andi', _gcmp(e, 'sge', ih, zero, I64),
                         _gcmp(e, 'slt', ih, _gconst(e, h, I64), I64), I1),
                   _gbin(e, 'andi', _gcmp(e, 'sge', iw, zero, I64),
                         _gcmp(e, 'slt', iw, _gconst(e, w, I64), I64), I1), I1)
        pix = _gbin(e, 'addi',
                    _gbin(e, 'addi', _gmul(e, img, h * w, I64),
                          _gmul(e, ih, w, I64), I64), iw, I64)
        srcs.append(_gbin(e, 'addi', _gmul(e, pix, c, I64), ci, I64))
        return srcs, ok
    if op == GA_IM2COL2:
        c, h, w, kh, kw = params[1], params[2], params[3], params[4], params[5]
        sh, sw, ph, pw = params[6], params[7], params[8], params[9]
        dh, dw = params[10], params[11]
        hw = h * w
        kk = kh * kw
        ckk = c * kk
        ow = conv_out(w, kw, sw, pw, dw)
        ohw = conv_out(h, kh, sh, ph, dh) * ow
        row = _gdiv(e, off, ckk, I64)
        col = _gmod(e, off, ckk, I64)
        img = _gdiv(e, row, ohw, I64)
        pos = _gmod(e, row, ohw, I64)
        ch = _gdiv(e, col, kk, I64)
        rk = _gmod(e, col, kk, I64)
        ih = _gaddc(e, _gbin(e, 'addi', _gmul(e, _gdiv(e, pos, ow, I64), sh,
                                              I64),
                             _gmul(e, _gdiv(e, rk, kw, I64), dh, I64), I64),
                    -ph, I64)
        iw = _gaddc(e, _gbin(e, 'addi', _gmul(e, _gmod(e, pos, ow, I64), sw,
                                              I64),
                             _gmul(e, _gmod(e, rk, kw, I64), dw, I64), I64),
                    -pw, I64)
        zero = _gconst(e, 0, I64)
        ok = _gbin(e, 'andi',
                   _gbin(e, 'andi', _gcmp(e, 'sge', ih, zero, I64),
                         _gcmp(e, 'slt', ih, _gconst(e, h, I64), I64), I1),
                   _gbin(e, 'andi', _gcmp(e, 'sge', iw, zero, I64),
                         _gcmp(e, 'slt', iw, _gconst(e, w, I64), I64), I1), I1)
        pix = _gbin(e, 'addi', _gmul(e, ih, w, I64), iw, I64)
        if len(params) > 12 and params[12]:
            base = _gbin(e, 'addi', _gmul(e, img, c * hw, I64), ch, I64)
            srcs.append(_gbin(e, 'addi', base, _gmul(e, pix, c, I64), I64))
            return srcs, ok
        base = _gbin(e, 'addi', _gmul(e, img, c * hw, I64),
                     _gmul(e, ch, hw, I64), I64)
        srcs.append(_gbin(e, 'addi', base, pix, I64))
        return srcs, ok
    if op == GA_IM2COL_T:
        c, h, w, kh, kw = params[1], params[2], params[3], params[4], params[5]
        sh, sw, ph, pw = params[6], params[7], params[8], params[9]
        dh, dw, oh, ow = params[10], params[11], params[12], params[13]
        hw = h * w
        kk = kh * kw
        ckk = c * kk
        row = _gdiv(e, off, ckk, I64)
        col = _gmod(e, off, ckk, I64)
        img = _gdiv(e, row, oh * ow, I64)
        pos = _gmod(e, row, oh * ow, I64)
        ch = _gdiv(e, col, kk, I64)
        rk = _gmod(e, col, kk, I64)
        ty = _gaddc(e, _gbin(e, 'subi', _gdiv(e, pos, ow, I64),
                             _gmul(e, _gdiv(e, rk, kw, I64), dh, I64), I64),
                    ph, I64)
        tx = _gaddc(e, _gbin(e, 'subi', _gmod(e, pos, ow, I64),
                             _gmul(e, _gmod(e, rk, kw, I64), dw, I64), I64),
                    pw, I64)
        iy = _gdiv(e, ty, sh, I64)
        ix = _gdiv(e, tx, sw, I64)
        zero = _gconst(e, 0, I64)
        ok = _gbin(e, 'andi',
                   _gbin(e, 'andi', _gcmp(e, 'sge', ty, zero, I64),
                         _gcmp(e, 'slt', iy, _gconst(e, h, I64), I64), I1),
                   _gbin(e, 'andi', _gcmp(e, 'sge', tx, zero, I64),
                         _gcmp(e, 'slt', ix, _gconst(e, w, I64), I64), I1), I1)
        ok = _gbin(e, 'andi', ok,
                   _gbin(e, 'andi',
                         _gcmp(e, 'eq', _gmod(e, ty, sh, I64), zero, I64),
                         _gcmp(e, 'eq', _gmod(e, tx, sw, I64), zero, I64),
                         I1), I1)
        base = _gbin(e, 'addi', _gmul(e, img, c * hw, I64),
                     _gmul(e, ch, hw, I64), I64)
        srcs.append(_gbin(e, 'addi', base,
                          _gbin(e, 'addi', _gmul(e, iy, w, I64), ix, I64),
                          I64))
        return srcs, ok
    if op == GA_MAXPOOL_NHWC:
        n, c, h, w, k, stride, pad = (params[0], params[1], params[2],
                                      params[3], params[4], params[5],
                                      params[6])
        oh = (h + 2 * pad - k) // stride + 1
        ow = (w + 2 * pad - k) // stride + 1
        ohw = oh * ow
        ci = _gmod(e, off, c, I64)
        row = _gdiv(e, off, c, I64)
        img = _gdiv(e, row, ohw, I64)
        pos = _gmod(e, row, ohw, I64)
        ph = _gmul(e, _gdiv(e, pos, ow, I64), stride, I64)
        pw = _gmul(e, _gmod(e, pos, ow, I64), stride, I64)
        base = _gmul(e, img, h * w, I64)
        for a in range(k):
            ih = _gclamp(e, _gaddc(e, ph, a - pad, I64), h - 1, I64)
            rh = _gbin(e, 'addi', base, _gmul(e, ih, w, I64), I64)
            for b in range(k):
                iw = _gclamp(e, _gaddc(e, pw, b - pad, I64), w - 1, I64)
                pix = _gbin(e, 'addi', rh, iw, I64)
                srcs.append(_gbin(e, 'addi', _gmul(e, pix, c, I64), ci, I64))
        return srcs, ''
    if op == GA_COL2CHW:
        n, hw, o = params[0], params[1], params[2]
        img = _gdiv(e, off, o * hw, I64)
        rem = _gmod(e, off, o * hw, I64)
        ch = _gdiv(e, rem, hw, I64)
        pos = _gmod(e, rem, hw, I64)
        row = _gbin(e, 'addi', _gmul(e, img, hw, I64), pos, I64)
        srcs.append(_gbin(e, 'addi', _gmul(e, row, o, I64), ch, I64))
        return srcs, ''
    if op == GA_ROTHALF:
        dh = params[0]
        half = dh // 2
        blk = _gdiv(e, off, dh, I64)
        c = _gmod(e, off, dh, I64)
        rc = _gmod(e, _gaddc(e, c, half, I64), dh, I64)
        srcs.append(_gbin(e, 'addi', _gmul(e, blk, dh, I64), rc, I64))
        return srcs, ''
    if op == GA_HEADSPLIT or op == GA_HEADMERGE:
        rows, dh, heads = params[0], params[1], params[2]
        if op == GA_HEADSPLIT:
            ostride, sstride = rows * dh, heads * dh
        else:
            ostride, sstride = heads * dh, rows * dh
        oi = _gdiv(e, off, ostride, I64)
        rem = _gmod(e, off, ostride, I64)
        ii = _gdiv(e, rem, dh, I64)
        ci = _gmod(e, rem, dh, I64)
        srcs.append(_gbin(e, 'addi',
                          _gbin(e, 'addi', _gmul(e, ii, sstride, I64),
                                _gmul(e, oi, dh, I64), I64), ci, I64))
        return srcs, ''
    n, c, h, w, k, stride, pad = (params[0], params[1], params[2], params[3],
                                  params[4], params[5], params[6])
    oh = (h + 2 * pad - k) // stride + 1
    ow = (w + 2 * pad - k) // stride + 1
    pw = _gmul(e, _gmod(e, off, ow, I64), stride, I64)
    q = _gdiv(e, off, ow, I64)
    ph = _gmul(e, _gmod(e, q, oh, I64), stride, I64)
    plane = _gdiv(e, q, oh, I64)
    base = _gmul(e, plane, h * w, I64)
    for a in range(k):
        ih = _gclamp(e, _gaddc(e, ph, a - pad, I64), h - 1, I64)
        for b in range(k):
            iw = _gclamp(e, _gaddc(e, pw, b - pad, I64), w - 1, I64)
            srcs.append(_gbin(e, 'addi', base,
                              _gbin(e, 'addi', _gmul(e, ih, w, I64), iw,
                                    I64), I64))
    return srcs, ''

def to_ttir_rowgather(params, name, dtype):
    BLOCK = config.flat
    S = STORE_TYPE[dtype]
    cols = params[0]
    T, TS, P, I32, I64, I1 = _tile_types(BLOCK, S, S)
    e = Emitter()
    e.add('module {')
    e.add('  tt.func public @%s(%%in: !tt.ptr<%s>, %%idx: !tt.ptr<%s>, '
          '%%out: !tt.ptr<%s>, %%n: i64, %%c: i64) '
          'attributes {noinline = false} {' % (name, S, S, S))
    e.add('    %%zero = arith.constant dense<0.0> : %s' % T)
    _flat_prologue(e.lines, BLOCK, I32, I64)
    row = _gdiv(e, '%offs64', cols, I64)
    col = _gmod(e, '%offs64', cols, I64)
    e.add('    %%pidx = tt.splat %%idx : !tt.ptr<%s> -> %s' % (S, P))
    q = e.tmp()
    e.add('    %s = tt.addptr %%pidx, %s : %s, %s' % (q, row, P, I64))
    iv = e.tmp()
    e.add('    %s = tt.load %s, %%mask, %%zero : %s' % (iv, q, P))
    ii = e.tmp()
    e.add('    %s = arith.fptosi %s : %s to %s' % (ii, iv, T, I64))
    src = _gbin(e, 'addi', _gbin(e, 'muli', ii, _gconst(e, cols, I64), I64),
                col, I64)
    e.add('    %%pin = tt.splat %%in : !tt.ptr<%s> -> %s' % (S, P))
    qs = e.tmp()
    e.add('    %s = tt.addptr %%pin, %s : %s, %s' % (qs, src, P, I64))
    v = e.tmp()
    e.add('    %s = tt.load %s, %%mask, %%zero : %s' % (v, qs, P))
    e.add('    %%pout = tt.splat %%out : !tt.ptr<%s> -> %s' % (S, P))
    e.add('    %%qout = tt.addptr %%pout, %%offs : %s, %s' % (P, I32))
    e.add('    tt.store %%qout, %s, %%mask : %s' % (v, P))
    e.add('    tt.return')
    e.add('  }')
    e.add('}')
    return '\n'.join(e.lines) + '\n'

def to_ttir_gather(op, params, name, dtype, fill=0.0):
    if op == GA_ROWS:
        return to_ttir_rowgather(params, name, dtype)
    if op == GA_STRIDED:
        return to_ttir_strided(params, fill, name, dtype)
    if op == GA_SCATTER:
        return to_ttir_scatter(params, name, dtype)
    if op == GA_TAKE:
        return to_ttir_take(params, name, dtype)
    if op == GA_POOL:
        return to_ttir_pool(params, name, dtype)
    BLOCK = config.flat
    S = STORE_TYPE[dtype]
    C = COMP_TYPE[dtype]
    half = S != C
    T, TS, P, I32, I64, I1 = _tile_types(BLOCK, S, S)
    e = Emitter()
    e.add('module {')
    e.add('  tt.func public @%s(%%in: !tt.ptr<%s>, %%out: !tt.ptr<%s>, '
          '%%n: i64, %%c: i64) attributes {noinline = false} {' % (name, S, S))
    e.add('    %%zero = arith.constant dense<0.0> : %s' % T)
    _flat_prologue(e.lines, BLOCK, I32, I64)
    srcs, ok = _gather_index(e, op, params, I64, I1)
    mask = '%mask'
    if ok:
        mask = _gbin(e, 'andi', mask, ok, I1)
    e.add('    %%pin = tt.splat %%in : !tt.ptr<%s> -> %s' % (S, P))
    vals = []
    for i in range(len(srcs)):
        q = e.tmp()
        e.add('    %s = tt.addptr %%pin, %s : %s, %s' % (q, srcs[i], P, I64))
        v = e.tmp()
        e.add('    %s = tt.load %s, %s, %%zero : %s' % (v, q, mask, P))
        vals.append(v)
    acc = vals[0]
    if len(vals) > 1 and half:
        TC = 'tensor<%dx%s>' % (BLOCK, C)
        for i in range(len(vals)):
            r = e.tmp()
            e.add('    %s = arith.extf %s : %s to %s' % (r, vals[i], T, TC))
            vals[i] = r
        acc = vals[0]
        for i in range(1, len(vals)):
            acc = _gbin(e, 'maximumf', acc, vals[i], TC)
        r = e.tmp()
        e.add('    %s = arith.truncf %s : %s to %s' % (r, acc, TC, T))
        acc = r
    else:
        for i in range(1, len(vals)):
            acc = _gbin(e, 'maximumf', acc, vals[i], T)
    e.add('    %%pout = tt.splat %%out : !tt.ptr<%s> -> %s' % (S, P))
    e.add('    %%qout = tt.addptr %%pout, %%offs : %s, %s' % (P, I32))
    e.add('    tt.store %%qout, %s, %%mask : %s' % (acc, P))
    e.add('    tt.return')
    e.add('  }')
    e.add('}')
    return '\n'.join(e.lines) + '\n'


def _float_text(v, C):
    wide = C == 'f64'
    if v != v:
        return '0x7FF8000000000000' if wide else '0x7FC00000'
    if v == INFINITY:
        return '0x7FF0000000000000' if wide else '0x7F800000'
    if v == -INFINITY:
        return '0xFFF0000000000000' if wide else '0xFF800000'
    return _fliteral(v)


def _sconst(e, v, ty):
    r = e.tmp()
    e.add('    %s = arith.constant %d : %s' % (r, v, ty))
    return r


def _splat(e, v, ty, tty):
    r = e.tmp()
    e.add('    %s = tt.splat %s : %s -> %s' % (r, v, ty, tty))
    return r


def _load(e, ptr, idx, mask, S, P, IX):
    p = _splat(e, ptr, '!tt.ptr<%s>' % S, P)
    q = e.tmp()
    e.add('    %s = tt.addptr %s, %s : %s, %s' % (q, p, idx, P, IX))
    v = e.tmp()
    e.add('    %s = tt.load %s, %s, %%zero : %s' % (v, q, mask, P))
    return v


def _ext(e, v, T, TC, half):
    if not half:
        return v
    r = e.tmp()
    e.add('    %s = arith.extf %s : %s to %s' % (r, v, T, TC))
    return r


def _and(e, a, b, I1):
    if not a:
        return b
    return _gbin(e, 'andi', a, b, I1)


def _store(e, v, S, P, I32):
    e.add('    %%pout = tt.splat %%out : !tt.ptr<%s> -> %s' % (S, P))
    e.add('    %%qout = tt.addptr %%pout, %%offs : %s, %s' % (P, I32))
    e.add('    tt.store %%qout, %s, %%mask : %s' % (v, P))
    e.add('    tt.return')
    e.add('  }')
    e.add('}')
    return '\n'.join(e.lines) + '\n'


def _header(e, name, ins, S):
    args = []
    for i in range(len(ins)):
        args.append('%%%s: !tt.ptr<%s>' % (ins[i], S))
    e.add('module {')
    e.add('  tt.func public @%s(%s, %%out: !tt.ptr<%s>, %%n: i64, %%c: i64) '
          'attributes {noinline = false} {' % (name, ', '.join(args), S))


# 1 << 31 is a long on a 32-bit host, which RPython rejects; a word there
# cannot reach 2**31 anyway, so sys.maxint is the same test.
I32_LIMIT = min(1 << 31, sys.maxint)


def strided_fits_i32(params):
    nsrc = params[0]
    rank = params[1]
    n = 1
    for j in range(rank):
        n *= params[2 + j]
    if n >= I32_LIMIT:
        return False
    for s in range(nsrc):
        base = 2 + rank + 3 * rank * s
        ext = 0
        for j in range(rank):
            ext += (params[2 + j] - 1) * abs(params[base + j])
        if ext >= I32_LIMIT:
            return False
    return True


def _coord(e, index, shape, j, off, IX):
    if not index[j]:
        inner = 1
        for q in range(j + 1, len(shape)):
            inner *= shape[q]
        v = off
        if inner != 1:
            v = _gdiv(e, v, inner, IX)
        if j > 0:
            v = _gmod(e, v, shape[j], IX)
        index[j] = v
    return index[j]


def to_ttir_strided(params, fill, name, dtype):
    """params = [nsrc, rank, shape..., then per source strides..., lo...,
    hi...].  Element i of the output, at coordinates x, is source s at
    sum(x[j] * stride[j]) for the first s with lo <= x < hi in every
    dimension, fill if there is none.  Each source's base offset is in its
    pointer."""
    BLOCK = config.flat
    S = STORE_TYPE[dtype]
    C = COMP_TYPE[dtype]
    T, TS, P, I32, I64, I1 = _tile_types(BLOCK, S, S)
    nsrc = params[0]
    rank = params[1]
    shape = []
    for j in range(rank):
        shape.append(params[2 + j])
    IX, off = I64, '%offs64'
    if strided_fits_i32(params):
        IX, off = I32, '%offs'
    e = Emitter()
    ins = []
    for s in range(nsrc):
        ins.append('in%d' % s)
    _header(e, name, ins, S)
    e.add('    %%zero = arith.constant dense<0.0> : %s' % T)
    _flat_prologue(e.lines, BLOCK, I32, I64)
    index = [''] * rank
    vals = []
    oks = []
    for s in range(nsrc):
        base = 2 + rank + 3 * rank * s
        addr = ''
        ok = ''
        for j in range(rank):
            st = params[base + j]
            lo = params[base + rank + j]
            hi = params[base + 2 * rank + j]
            if st != 0:
                x = _coord(e, index, shape, j, off, IX)
                if st != 1:
                    x = _gmul(e, x, st, IX)
                addr = _gbin(e, 'addi', addr, x, IX) if addr else x
            if lo > 0:
                ok = _and(e, ok, _gcmp(e, 'sge', _coord(e, index, shape, j,
                                                        off, IX),
                                       _gconst(e, lo, IX), IX), I1)
            if hi < shape[j]:
                ok = _and(e, ok, _gcmp(e, 'slt', _coord(e, index, shape, j,
                                                        off, IX),
                                       _gconst(e, hi, IX), IX), I1)
        if not addr:
            addr = _gconst(e, 0, IX)
        mask = _and(e, ok, '%mask', I1)
        vals.append(_load(e, '%%in%d' % s, addr, mask, S, P, IX))
        oks.append(ok)
    acc = '%zero'
    if fill != 0.0:
        TC = 'tensor<%dx%s>' % (BLOCK, C)
        e.add('    %%fillc = arith.constant dense<%s> : %s'
              % (_float_text(fill, C), TC))
        acc = '%fillc'
        if S != C:
            e.add('    %%fill = arith.truncf %%fillc : %s to %s' % (TC, T))
            acc = '%fill'
    for s in range(nsrc - 1, -1, -1):
        if oks[s]:
            r = e.tmp()
            e.add('    %s = arith.select %s, %s, %s : %s, %s'
                  % (r, oks[s], vals[s], acc, I1, T))
            acc = r
        else:
            acc = vals[s]
    return _store(e, acc, S, P, I32)


def scatter_block(params):
    n = 1
    for j in range(params[1]):
        n *= params[2 + j]
    return spread_block(config.flat, n)


def to_ttir_scatter(params, name, dtype):
    """params as to_ttir_strided's, for one source: element i of the
    contiguous input goes to the output at sum(x[j] * stride[j]), x being
    i's coordinates in shape.  The output's base offset is in its pointer."""
    BLOCK = scatter_block(params)
    S = STORE_TYPE[dtype]
    T, TS, P, I32, I64, I1 = _tile_types(BLOCK, S, S)
    rank = params[1]
    shape = []
    for j in range(rank):
        shape.append(params[2 + j])
    IX, off = I64, '%offs64'
    if strided_fits_i32(params):
        IX, off = I32, '%offs'
    e = Emitter()
    _header(e, name, ['in0'], S)
    e.add('    %%zero = arith.constant dense<0.0> : %s' % T)
    _flat_prologue(e.lines, BLOCK, I32, I64)
    v = _load(e, '%in0', '%offs', '%mask', S, P, I32)
    index = [''] * rank
    addr = ''
    for j in range(rank):
        st = params[2 + rank + j]
        if st != 0:
            x = _coord(e, index, shape, j, off, IX)
            if st != 1:
                x = _gmul(e, x, st, IX)
            addr = _gbin(e, 'addi', addr, x, IX) if addr else x
    if not addr:
        addr = _gconst(e, 0, IX)
    e.add('    %%pout = tt.splat %%out : !tt.ptr<%s> -> %s' % (S, P))
    e.add('    %%qout = tt.addptr %%pout, %s : %s, %s' % (addr, P, IX))
    e.add('    tt.store %%qout, %s, %%mask : %s' % (v, P))
    e.add('    tt.return')
    e.add('  }')
    e.add('}')
    return '\n'.join(e.lines) + '\n'


def to_ttir_take(params, name, dtype):
    """out[o, j, i] = in[o, idx[o*so + j*sj + i*si], i] over in as
    [outer, d, inner], zero where the index is out of range."""
    d, k, inner, so, sj, si = (params[0], params[1], params[2], params[3],
                               params[4], params[5])
    BLOCK = config.flat
    S = STORE_TYPE[dtype]
    C = COMP_TYPE[dtype]
    T, TS, P, I32, I64, I1 = _tile_types(BLOCK, S, S)
    TC = 'tensor<%dx%s>' % (BLOCK, C)
    e = Emitter()
    _header(e, name, ['in', 'idx'], S)
    e.add('    %%zero = arith.constant dense<0.0> : %s' % T)
    _flat_prologue(e.lines, BLOCK, I32, I64)
    off = '%offs64'
    xi = _gmod(e, off, inner, I64)
    q = _gdiv(e, off, inner, I64)
    xj = _gmod(e, q, k, I64)
    xo = _gdiv(e, q, k, I64)
    pos = _gconst(e, 0, I64)
    if so:
        pos = _gbin(e, 'addi', pos, _gmul(e, xo, so, I64), I64)
    if sj:
        pos = _gbin(e, 'addi', pos, _gmul(e, xj, sj, I64), I64)
    if si:
        pos = _gbin(e, 'addi', pos, _gmul(e, xi, si, I64), I64)
    iv = _ext(e, _load(e, '%idx', pos, '%mask', S, P, I64), T, TC, S != C)
    ii = e.tmp()
    e.add('    %s = arith.fptosi %s : %s to %s' % (ii, iv, TC, I64))
    ok = _gbin(e, 'andi', _gcmp(e, 'sge', ii, _gconst(e, 0, I64), I64),
               _gcmp(e, 'slt', ii, _gconst(e, d, I64), I64), I1)
    src = _gbin(e, 'addi', _gmul(e, _gbin(e, 'addi', _gmul(e, xo, d, I64), ii,
                                          I64), inner, I64), xi, I64)
    v = _load(e, '%in', src, _gbin(e, 'andi', '%mask', ok, I1), S, P, I64)
    return _store(e, v, S, P, I32)


def to_ttir_pool(params, name, dtype):
    """Average, adaptive average and depthwise convolution over NCHW planes:
    each output element sums a window of its plane in an scf.for.

    params = [mode, h, w, oh, ow, kh, kw, sh, sw, ph, pw, dh, dw, cout,
    mult, flags]: cout and mult (output channels, outputs per input channel)
    are for depthwise, whose weight is [cin, kh*kw, mult] and whose flags
    bit 0 says there is a bias; for average pooling that bit is
    count_include_pad and cout the channels (1 when both sides are NCHW).
    Flags bit 1 says the input is NHWC, bit 2 the output.  Adaptive windows
    come from oh/ow alone, kh and kw are the largest of them."""
    mode, h, w, oh, ow = params[0], params[1], params[2], params[3], params[4]
    kh, kw, sh, sw = params[5], params[6], params[7], params[8]
    ph, pw, dh, dw = params[9], params[10], params[11], params[12]
    cout, mult, flags = params[13], params[14], params[15]
    BLOCK = config.flat
    S = STORE_TYPE[dtype]
    C = COMP_TYPE[dtype]
    half = S != C
    T, TS, P, I32, I64, I1 = _tile_types(BLOCK, S, S)
    TC = 'tensor<%dx%s>' % (BLOCK, C)
    X = I64
    conv = mode == POOL_DEPTHWISE
    bias = conv and (flags & 1) != 0
    ins = ['in']
    if conv:
        ins.append('wt')
        if bias:
            ins.append('bias')
    e = Emitter()
    _header(e, name, ins, S)
    e.add('    %%zero = arith.constant dense<0.0> : %s' % T)
    e.add('    %%zc = arith.constant dense<0.0> : %s' % TC)
    _flat_prologue(e.lines, BLOCK, I32, I64)
    off = '%offs64'
    cin = cout // mult
    if flags & 4:
        oc = _gmod(e, off, cout, X)
        t = _gdiv(e, off, cout, X)
        owi = _gmod(e, t, ow, X)
        t = _gdiv(e, t, ow, X)
        ohi = _gmod(e, t, oh, X)
        img = _gdiv(e, t, oh, X)
    else:
        owi = _gmod(e, off, ow, X)
        t = _gdiv(e, off, ow, X)
        ohi = _gmod(e, t, oh, X)
        plane = _gdiv(e, t, oh, X)
        oc = _gmod(e, plane, cout, X)
        img = _gdiv(e, plane, cout, X)
    ci = oc
    if mult > 1:
        ci = _gdiv(e, oc, mult, X)
    wbase = ''
    if conv:
        wbase = _gmul(e, ci, kh * kw * mult, X)
        if mult > 1:
            wbase = _gbin(e, 'addi', wbase, _gmod(e, oc, mult, X), X)
    pstride = 1
    if flags & 2:
        base = _gbin(e, 'addi', _gmul(e, img, h * w * cin, X), ci, X)
        pstride = cin
    else:
        base = _gmul(e, _gbin(e, 'addi', _gmul(e, img, cin, X), ci, X),
                     h * w, X)
    cnt = ''
    if mode == POOL_ADAPTIVE:
        hs = _gdiv(e, _gmul(e, ohi, h, X), oh, X)
        he = _gdiv(e, _gaddc(e, _gmul(e, ohi, h, X), h + oh - 1, X), oh, X)
        ws = _gdiv(e, _gmul(e, owi, w, X), ow, X)
        we = _gdiv(e, _gaddc(e, _gmul(e, owi, w, X), w + ow - 1, X), ow, X)
        cnt = _gbin(e, 'muli', _gbin(e, 'subi', he, hs, X),
                    _gbin(e, 'subi', we, ws, X), X)
    else:
        hs = _gaddc(e, _gmul(e, ohi, sh, X), -ph, X)
        ws = _gaddc(e, _gmul(e, owi, sw, X), -pw, X)
        he = we = ''
        if mode == POOL_AVG:
            he = _gbin(e, 'minsi', _gaddc(e, hs, kh, X),
                       _gconst(e, h + ph, X), X)
            we = _gbin(e, 'minsi', _gaddc(e, ws, kw, X),
                       _gconst(e, w + pw, X), X)
            if flags & 1:
                lh, lw, uh, uw = hs, ws, he, we
            else:
                zero = _gconst(e, 0, X)
                lh = _gbin(e, 'maxsi', hs, zero, X)
                lw = _gbin(e, 'maxsi', ws, zero, X)
                uh = _gbin(e, 'minsi', he, _gconst(e, h, X), X)
                uw = _gbin(e, 'minsi', we, _gconst(e, w, X), X)
            cnt = _gbin(e, 'muli', _gbin(e, 'subi', uh, lh, X),
                        _gbin(e, 'subi', uw, lw, X), X)
    lb = _sconst(e, 0, 'i32')
    ub = _sconst(e, kh * kw, 'i32')
    step = _sconst(e, 1, 'i32')
    res = e.tmp()
    iv = e.tmp()
    acc = e.tmp()
    e.add('    %s = scf.for %s = %s to %s step %s iter_args(%s = %%zc) -> '
          '(%s) : i32 {' % (res, iv, lb, ub, step, acc, TC))
    a = _gbin(e, 'divsi', iv, _sconst(e, kw, 'i32'), 'i32')
    b = _gbin(e, 'remsi', iv, _sconst(e, kw, 'i32'), 'i32')
    a = _gbin(e, 'muli', a, _sconst(e, dh, 'i32'), 'i32')
    b = _gbin(e, 'muli', b, _sconst(e, dw, 'i32'), 'i32')
    a64 = e.tmp()
    e.add('    %s = arith.extsi %s : i32 to i64' % (a64, a))
    b64 = e.tmp()
    e.add('    %s = arith.extsi %s : i32 to i64' % (b64, b))
    ih = _gbin(e, 'addi', hs, _splat(e, a64, 'i64', X), X)
    iw = _gbin(e, 'addi', ws, _splat(e, b64, 'i64', X), X)
    if mode == POOL_ADAPTIVE:
        ok = _gbin(e, 'andi', _gcmp(e, 'slt', ih, he, X),
                   _gcmp(e, 'slt', iw, we, X), I1)
    else:
        zero = _gconst(e, 0, X)
        ok = _gbin(e, 'andi',
                   _gbin(e, 'andi', _gcmp(e, 'sge', ih, zero, X),
                         _gcmp(e, 'slt', ih, _gconst(e, h, X), X), I1),
                   _gbin(e, 'andi', _gcmp(e, 'sge', iw, zero, X),
                         _gcmp(e, 'slt', iw, _gconst(e, w, X), X), I1), I1)
    pix = _gbin(e, 'addi', _gmul(e, ih, w, X), iw, X)
    if pstride > 1:
        pix = _gmul(e, pix, pstride, X)
    src = _gbin(e, 'addi', base, pix, X)
    v = _ext(e, _load(e, '%in', src, _gbin(e, 'andi', '%mask', ok, I1), S, P,
                      X), T, TC, half)
    if conv:
        r = _gbin(e, 'muli', iv, _sconst(e, mult, 'i32'), 'i32')
        r64 = e.tmp()
        e.add('    %s = arith.extsi %s : i32 to i64' % (r64, r))
        wi = _gbin(e, 'addi', wbase, _splat(e, r64, 'i64', X), X)
        wv = _ext(e, _load(e, '%wt', wi, '%mask', S, P, X), T, TC, half)
        v = _gbin(e, 'mulf', v, wv, TC)
    nxt = _gbin(e, 'addf', acc, v, TC)
    e.add('      scf.yield %s : %s' % (nxt, TC))
    e.add('    }')
    out = res
    if cnt:
        cf = e.tmp()
        e.add('    %s = arith.sitofp %s : %s to %s' % (cf, cnt, X, TC))
        out = _gbin(e, 'divf', out, cf, TC)
    if bias:
        bv = _ext(e, _load(e, '%bias', oc, '%mask', S, P, X), T, TC, half)
        out = _gbin(e, 'addf', out, bv, TC)
    if half:
        r = e.tmp()
        e.add('    %s = arith.truncf %s : %s to %s' % (r, out, TC, T))
        out = r
    return _store(e, out, S, P, I32)


def _cv(e, v, ty):
    r = e.tmp()
    e.add('    %s = arith.constant %d : %s' % (r, v, ty))
    return r


def _expand(e, v, axis, src, dst):
    r = e.tmp()
    e.add('    %s = tt.expand_dims %s {axis = %d : i32} : %s -> %s'
          % (r, v, axis, src, dst))
    return r


def _bcast(e, v, src, dst, r=''):
    if not r:
        r = e.tmp()
    e.add('    %s = tt.broadcast %s : %s -> %s' % (r, v, src, dst))
    return r


def _col(e, v, rows, cols, ty, r=''):
    one = 'tensor<%dx1x%s>' % (rows, ty)
    return _bcast(e, _expand(e, v, 1, 'tensor<%dx%s>' % (rows, ty), one),
                  one, 'tensor<%dx%dx%s>' % (rows, cols, ty), r)


def _row(e, v, rows, cols, ty, r=''):
    one = 'tensor<1x%dx%s>' % (cols, ty)
    return _bcast(e, _expand(e, v, 0, 'tensor<%dx%s>' % (cols, ty), one),
                  one, 'tensor<%dx%dx%s>' % (rows, cols, ty), r)


def _outer(e, col, row, op, rows, cols, ty):
    return _gbin(e, op, _col(e, col, rows, cols, ty),
                 _row(e, row, rows, cols, ty),
                 'tensor<%dx%dx%s>' % (rows, cols, ty))


def _range(e, start, block):
    I = 'tensor<%dxi32>' % block
    r = e.tmp()
    e.add('    %s = tt.make_range {end = %d : i32, start = 0 : i32} : %s'
          % (r, block, I))
    if not start:
        return r
    return _gbin(e, 'addi', _splat(e, start, 'i32', I), r, I)


def _inside(e, v, ok, hi, ty, I1):
    a = _gcmp(e, 'sge', v, _gconst(e, 0, ty), ty)
    b = _gcmp(e, 'slt', v, _gconst(e, hi, ty), ty)
    return _gbin(e, 'andi', _and(e, ok, a, I1), b, I1)


def _addptr(e, base, off, S, shape):
    P = 'tensor<%s!tt.ptr<%s>>' % (shape, S)
    q = e.tmp()
    e.add('    %s = tt.addptr %s, %s : %s, tensor<%si32>'
          % (q, _splat(e, base, '!tt.ptr<%s>' % S, P), off, P, shape))
    return q, P


def _tf32(e, v, T):
    r = e.tmp()
    e.add('    %s = tt.elementwise_inline_asm "cvt.rna.tf32.f32 $0, $1;" '
          '{constraints = "=r,r", packed_element = 1 : i32, pure = true} %s '
          ': %s -> %s' % (r, v, T, T))
    return r




def _window(e, d, rm, BM):
    layout, c, h, w = d[0], d[1], d[2], d[3]
    sh, sw, ph, pw = d[7], d[8], d[9], d[10]
    oh, ow = d[13], d[14]
    IM = 'tensor<%dxi32>' % BM
    img = _gdiv(e, rm, oh * ow, IM)
    pos = _gmod(e, rm, oh * ow, IM)
    ih0 = _gaddc(e, _gmul(e, _gdiv(e, pos, ow, IM), sh, IM), -ph, IM)
    iw0 = _gaddc(e, _gmul(e, _gmod(e, pos, ow, IM), sw, IM), -pw, IM)
    if layout & 1:
        base = _gbin(e, 'addi',
                     _gbin(e, 'addi', _gmul(e, img, h * w * c, IM),
                           _gmul(e, ih0, w * c, IM), IM),
                     _gmul(e, iw0, c, IM), IM)
    else:
        base = _gbin(e, 'addi',
                     _gbin(e, 'addi', _gmul(e, img, c * h * w, IM),
                           _gmul(e, ih0, w, IM), IM), iw0, IM)
    return base, ih0, iw0


def _conv_loop(e, d, xin, win, dst, rm, rn, grp, mm, nm, BM, BN, S):
    if d[CD_G] > 1 and not conv_grouped_dot(d):
        return _conv_loop_fma(e, d, xin, win, dst, rm, rn, mm, nm, BM, BN, S)
    layout, c, h, w, o = d[0], d[1], d[2], d[3], d[4]
    kh, kw, dh, dw = d[5], d[6], d[11], d[12]
    g, BK = d[CD_G], d[CD_BK]
    cg = c // g
    nhwc = layout & 1
    kdim = kh * kw * cg
    chunked = nhwc != 0 and cg % BK == 0
    IM = 'tensor<%dxi32>' % BM
    IK = 'tensor<%dxi32>' % BK
    AK = 'tensor<%dx%dx%s>' % (BM, BK, S)
    KN = 'tensor<%dx%dx%s>' % (BK, BN, S)
    MN = 'tensor<%dx%dx%s>' % (BM, BN, S)
    rk = _range(e, '', BK)
    rowbase, ih0, iw0 = _window(e, d, rm, BM)
    if g > 1:
        cbase = _gbin(e, 'muli', grp, _cv(e, cg, 'i32'), 'i32')
        cvec = _splat(e, cbase, 'i32', IM)
        if not nhwc:
            cvec = _gbin(e, 'muli', cvec, _gconst(e, h * w, IM), IM)
        rowbase = _gbin(e, 'addi', rowbase, cvec, IM)
    acc0 = e.tmp()
    e.add('    %s = arith.constant dense<0.0> : %s' % (acc0, MN))
    za = e.tmp()
    e.add('    %s = arith.constant dense<0.0> : %s' % (za, AK))
    zb = e.tmp()
    e.add('    %s = arith.constant dense<0.0> : %s' % (zb, KN))
    lb = _cv(e, 0, 'i32')
    ub = _cv(e, kdim, 'i32')
    step = _cv(e, BK, 'i32')
    kk = e.tmp()
    a = e.tmp()
    e.add('    %s = scf.for %s = %s to %s step %s iter_args(%s = %s) -> (%s) '
          ': i32 {' % (dst, kk, lb, ub, step, a, acc0, MN))
    kvec = _gbin(e, 'addi', _splat(e, kk, 'i32', IK), rk, IK)
    kin = _gcmp(e, 'slt', kvec, _gconst(e, kdim, IK), IK)
    if chunked:
        rs = _gbin(e, 'divsi', kk, _cv(e, cg, 'i32'), 'i32')
        c0 = _gbin(e, 'remsi', kk, _cv(e, cg, 'i32'), 'i32')
        krd = _gbin(e, 'muli', _gbin(e, 'divsi', rs, _cv(e, kw, 'i32'),
                                     'i32'), _cv(e, dh, 'i32'), 'i32')
        kcd = _gbin(e, 'muli', _gbin(e, 'remsi', rs, _cv(e, kw, 'i32'),
                                     'i32'), _cv(e, dw, 'i32'), 'i32')
        ih = _gbin(e, 'addi', ih0, _splat(e, krd, 'i32', IM), IM)
        iw = _gbin(e, 'addi', iw0, _splat(e, kcd, 'i32', IM), IM)
        B1 = 'tensor<%dxi1>' % BM
        ok = _inside(e, iw, _inside(e, ih, mm, h, IM, B1), w, IM, B1)
        koff = _gbin(e, 'addi',
                     _gbin(e, 'addi',
                           _gbin(e, 'muli', krd, _cv(e, w * c, 'i32'), 'i32'),
                           _gbin(e, 'muli', kcd, _cv(e, c, 'i32'), 'i32'),
                           'i32'), c0, 'i32')
        arow = _gbin(e, 'addi', rowbase, _splat(e, koff, 'i32', IM), IM)
        aoff = _outer(e, arow, rk, 'addi', BM, BK, 'i32')
        amask = _col(e, ok, BM, BK, 'i1')
    else:
        ci = _gmod(e, kvec, cg, IK)
        rs = _gdiv(e, kvec, cg, IK)
        krd = _gmul(e, _gdiv(e, rs, kw, IK), dh, IK)
        kcd = _gmul(e, _gmod(e, rs, kw, IK), dw, IK)
        if nhwc:
            koff = _gbin(e, 'addi',
                         _gbin(e, 'addi', _gmul(e, krd, w * c, IK),
                               _gmul(e, kcd, c, IK), IK), ci, IK)
        else:
            koff = _gbin(e, 'addi',
                         _gbin(e, 'addi', _gmul(e, ci, h * w, IK),
                               _gmul(e, krd, w, IK), IK), kcd, IK)
        I2 = 'tensor<%dx%dxi32>' % (BM, BK)
        B2 = 'tensor<%dx%dxi1>' % (BM, BK)
        ih = _outer(e, ih0, krd, 'addi', BM, BK, 'i32')
        iw = _outer(e, iw0, kcd, 'addi', BM, BK, 'i32')
        mk = _outer(e, mm, kin, 'andi', BM, BK, 'i1')
        amask = _inside(e, iw, _inside(e, ih, mk, h, I2, B2), w, I2, B2)
        aoff = _outer(e, rowbase, koff, 'addi', BM, BK, 'i32')
    pa, PA = _addptr(e, xin, aoff, S, '%dx%dx' % (BM, BK))
    av = e.tmp()
    e.add('    %s = tt.load %s, %s, %s : %s' % (av, pa, amask, za, PA))
    boff = _outer(e, _gmul(e, kvec, o, IK), rn, 'addi', BK, BN, 'i32')
    bmask = _outer(e, kin, nm, 'andi', BK, BN, 'i1')
    pb, PB = _addptr(e, win, boff, S, '%dx%dx' % (BK, BN))
    bv = e.tmp()
    e.add('    %s = tt.load %s, %s, %s : %s' % (bv, pb, bmask, zb, PB))
    prec = config.conv_prec
    if prec == 'auto':
        prec = 'tf32' if o // g >= 64 or kdim >= 1024 else 'ieee'
    if prec == 'tf32':
        av = _tf32(e, av, AK)
        bv = _tf32(e, bv, KN)
    r = e.tmp()
    e.add('    %s = tt.dot %s, %s, %s, inputPrecision = %s : %s * %s -> %s'
          % (r, av, bv, a, prec, AK, KN, MN))
    e.add('      scf.yield %s : %s' % (r, MN))
    e.add('    }')


def _conv_loop_fma(e, d, xin, win, dst, rm, rn, mm, nm, BM, BN, S):
    layout, c, h, w, o = d[0], d[1], d[2], d[3], d[4]
    kh, kw, dh, dw = d[5], d[6], d[11], d[12]
    g = d[CD_G]
    cg = c // g
    og = o // g
    IM = 'tensor<%dxi32>' % BM
    IN = 'tensor<%dxi32>' % BN
    B1 = 'tensor<%dxi1>' % BM
    MN = 'tensor<%dx%dx%s>' % (BM, BN, S)
    TN = 'tensor<%dx%s>' % (BN, S)
    rowbase, ih0, iw0 = _window(e, d, rm, BM)
    chan = _gmul(e, _gdiv(e, rn, og, IN), cg, IN)
    if not layout & 1:
        chan = _gmul(e, chan, h * w, IN)
    acc0 = e.tmp()
    e.add('    %s = arith.constant dense<0.0> : %s' % (acc0, MN))
    za = e.tmp()
    e.add('    %s = arith.constant dense<0.0> : %s' % (za, MN))
    zb = e.tmp()
    e.add('    %s = arith.constant dense<0.0> : %s' % (zb, TN))
    lb = _cv(e, 0, 'i32')
    ub = _cv(e, kh * kw * cg, 'i32')
    step = _cv(e, 1, 'i32')
    t = e.tmp()
    a = e.tmp()
    e.add('    %s = scf.for %s = %s to %s step %s iter_args(%s = %s) -> (%s) '
          ': i32 {' % (dst, t, lb, ub, step, a, acc0, MN))
    rs = _gbin(e, 'divsi', t, _cv(e, cg, 'i32'), 'i32')
    j = _gbin(e, 'remsi', t, _cv(e, cg, 'i32'), 'i32')
    krd = _gbin(e, 'muli', _gbin(e, 'divsi', rs, _cv(e, kw, 'i32'), 'i32'),
                _cv(e, dh, 'i32'), 'i32')
    kcd = _gbin(e, 'muli', _gbin(e, 'remsi', rs, _cv(e, kw, 'i32'), 'i32'),
                _cv(e, dw, 'i32'), 'i32')
    ih = _gbin(e, 'addi', ih0, _splat(e, krd, 'i32', IM), IM)
    iw = _gbin(e, 'addi', iw0, _splat(e, kcd, 'i32', IM), IM)
    ok = _inside(e, iw, _inside(e, ih, mm, h, IM, B1), w, IM, B1)
    if layout & 1:
        koff = _gbin(e, 'addi',
                     _gbin(e, 'addi',
                           _gbin(e, 'muli', krd, _cv(e, w * c, 'i32'), 'i32'),
                           _gbin(e, 'muli', kcd, _cv(e, c, 'i32'), 'i32'),
                           'i32'), j, 'i32')
    else:
        koff = _gbin(e, 'addi',
                     _gbin(e, 'addi',
                           _gbin(e, 'muli', krd, _cv(e, w, 'i32'), 'i32'),
                           kcd, 'i32'),
                     _gbin(e, 'muli', j, _cv(e, h * w, 'i32'), 'i32'), 'i32')
    arow = _gbin(e, 'addi', rowbase, _splat(e, koff, 'i32', IM), IM)
    aoff = _outer(e, arow, chan, 'addi', BM, BN, 'i32')
    amask = _outer(e, ok, nm, 'andi', BM, BN, 'i1')
    pa, PA = _addptr(e, xin, aoff, S, '%dx%dx' % (BM, BN))
    av = e.tmp()
    e.add('    %s = tt.load %s, %s, %s : %s' % (av, pa, amask, za, PA))
    woff = _gbin(e, 'addi', _splat(e, _gbin(e, 'muli', t, _cv(e, o, 'i32'),
                                             'i32'), 'i32', IN), rn, IN)
    pb, PB = _addptr(e, win, woff, S, '%dx' % BN)
    bv = e.tmp()
    e.add('    %s = tt.load %s, %s, %s : %s' % (bv, pb, nm, zb, PB))
    prod = _gbin(e, 'mulf', av, _row(e, bv, BM, BN, S), MN)
    r = _gbin(e, 'addf', a, prod, MN)
    e.add('      scf.yield %s : %s' % (r, MN))
    e.add('    }')


def to_ttir_conv(kernel, name, modes):
    nodes = kernel.nodes
    nreal = kernel.ninputs
    nin = nvals(kernel)
    S = STORE_TYPE[kernel.dtype]
    if S != 'f32' or len(modes) != nin + len(nodes):
        return ''
    first = -1
    elementwise = [False] * nreal
    for k in range(len(nodes)):
        node = nodes[k]
        if node.opcode == CONV:
            if first < 0:
                first = k
            continue
        if node.opcode == GATHER or is_reduction(node.opcode):
            return ''
        if node.a < nreal:
            elementwise[node.a] = True
        if node.b >= 0 and node.b < nreal:
            elementwise[node.b] = True
    if first < 0:
        return ''
    first = main_conv(kernel)
    d0 = conv_descs.params[nodes[first].p]
    m, o = d0[CD_ROWS], d0[CD_O]
    BM, BN, GM = d0[CD_BM], d0[CD_BN], d0[CD_GM]
    for k in range(len(nodes)):
        if nodes[k].opcode == CONV:
            d = conv_descs.params[nodes[k].p]
            if d[CD_ROWS] != m or d[CD_O] != o:
                return ''
            if d[CD_G] != d0[CD_G] or \
                    conv_grouped_dot(d) != conv_grouped_dot(d0):
                return ''
    omodes = out_modes(kernel)
    for k in range(len(omodes)):
        if omodes[k] != 0:
            return ''
    T = 'tensor<%dx%dx%s>' % (BM, BN, S)
    I1 = 'tensor<%dx%dxi1>' % (BM, BN)
    IM = 'tensor<%dxi32>' % BM
    IN = 'tensor<%dxi32>' % BN
    P = '!tt.ptr<%s> {tt.divisibility = 16 : i32}' % S
    args = ['%%in%d: %s' % (i, P) for i in range(nreal)]
    args.append('%%out: %s' % P)
    for k in range(len(kernel.outputs)):
        args.append('%%out%d: %s' % (k, P))
    e = Emitter()
    e.add('module {')
    e.add('  tt.func public @%s(%s, %%n: i64, %%c: i64) '
          'attributes {noinline = false} {' % (name, ', '.join(args)))
    e.add('    %%zero = arith.constant dense<0.0> : %s' % T)
    e.add('    %%one = arith.constant dense<1.0> : %s' % T)
    if not _emit_consts(e.lines, kernel, nreal, T):
        return ''
    static = nodes[first].p in conv_descs.static
    mr = npm = ''
    if not static:
        nimg = e.tmp()
        e.add('    %s = arith.trunci %%c : i64 to i32' % nimg)
        mr = _gbin(e, 'muli', nimg, _cv(e, d0[13] * d0[14], 'i32'), 'i32')
        npm = _gbin(e, 'divsi', _gbin(e, 'addi', mr, _cv(e, BM - 1, 'i32'),
                                      'i32'), _cv(e, BM, 'i32'), 'i32')
    npn = conv_col_tiles(d0)
    e.add('    %pid = tt.get_program_id x : i32')
    width = _cv(e, GM * npn, 'i32')
    gm = _cv(e, GM, 'i32')
    group = _gbin(e, 'muli', _gbin(e, 'divsi', '%pid', width, 'i32'), gm,
                  'i32')
    if static:
        npm = _cv(e, (m + BM - 1) // BM, 'i32')
    gsz = _gbin(e, 'minsi', _gbin(e, 'subi', npm, group,
                                  'i32'), gm, 'i32')
    within = _gbin(e, 'remsi', '%pid', width, 'i32')
    pm = _gbin(e, 'addi', group, _gbin(e, 'remsi', within, gsz, 'i32'),
               'i32')
    pn = _gbin(e, 'divsi', within, gsz, 'i32')
    rm = _range(e, _gbin(e, 'muli', pm, _cv(e, BM, 'i32'), 'i32'), BM)
    bound = _gconst(e, m, IM) if static else _splat(e, mr, 'i32', IM)
    mm = _gcmp(e, 'slt', rm, bound, IM)
    grp = _cv(e, 0, 'i32')
    if conv_grouped_dot(d0):
        og = o // d0[CD_G]
        tpg = _cv(e, (og + BN - 1) // BN, 'i32')
        grp = _gbin(e, 'divsi', pn, tpg, 'i32')
        inner = _range(e, _gbin(e, 'muli', _gbin(e, 'remsi', pn, tpg, 'i32'),
                                _cv(e, BN, 'i32'), 'i32'), BN)
        nm = _gcmp(e, 'slt', inner, _gconst(e, og, IN), IN)
        rn = _gbin(e, 'addi', inner,
                   _splat(e, _gbin(e, 'muli', grp, _cv(e, og, 'i32'), 'i32'),
                          'i32', IN), IN)
    else:
        rn = _range(e, _gbin(e, 'muli', pn, _cv(e, BN, 'i32'), 'i32'), BN)
        nm = _gcmp(e, 'slt', rn, _gconst(e, o, IN), IN)
    ooff = _outer(e, _gmul(e, rm, o, IM), rn, 'addi', BM, BN, 'i32')
    omask = _outer(e, mm, nm, 'andi', BM, BN, 'i1')
    shape = '%dx%dx' % (BM, BN)
    for i in range(nreal):
        if not elementwise[i]:
            continue
        mode = modes[i]
        dst = '%%v%d' % i
        if mode == 0:
            q, P = _addptr(e, '%%in%d' % i, ooff, S, shape)
            e.add('    %s = tt.load %s, %s, %%zero : %s' % (dst, q, omask, P))
        elif mode == 1 or mode == 3:
            idx, mask, B = rn, nm, BN
            if mode == 3:
                idx, mask, B = rm, mm, BM
            TV = 'tensor<%dx%s>' % (B, S)
            zv = e.tmp()
            e.add('    %s = arith.constant dense<0.0> : %s' % (zv, TV))
            q, P = _addptr(e, '%%in%d' % i, idx, S, '%dx' % B)
            v = e.tmp()
            e.add('    %s = tt.load %s, %s, %s : %s' % (v, q, mask, zv, P))
            if mode == 1:
                _row(e, v, BM, BN, S, dst)
            else:
                _col(e, v, BM, BN, S, dst)
        else:
            v = e.tmp()
            e.add('    %s = tt.load %%in%d : !tt.ptr<%s>' % (v, i, S))
            e.add('    %s = tt.splat %s : %s -> %s' % (dst, v, S, T))
    order = [first]
    for k in range(len(nodes)):
        if k != first:
            order.append(k)
    for k in order:
        node = nodes[k]
        v = nin + k
        if node.opcode == CONV:
            if node.a >= nreal or node.b < 0 or node.b >= nreal:
                return ''
            _conv_loop(e, conv_descs.params[node.p], '%%in%d' % node.a,
                       '%%in%d' % node.b, '%%v%d' % v, rm, rn, grp, mm, nm, BM,
                       BN, S)
        elif not _elementwise(e.lines, node, v, T, I1):
            return ''
    q, P = _addptr(e, '%out', ooff, S, shape)
    e.add('    tt.store %s, %%v%d, %s : %s' % (q, nin + len(nodes) - 1, omask,
                                              P))
    for k in range(len(kernel.outputs)):
        q, P = _addptr(e, '%%out%d' % k, ooff, S, shape)
        e.add('    tt.store %s, %%v%d, %s : %s' % (q, kernel.outputs[k],
                                                 omask, P))
    e.add('    tt.return')
    e.add('  }')
    e.add('}')
    return '\n'.join(e.lines) + '\n'


_LOG2E = '1.4426950408889634e+00'


def _t(a, b, e):
    if b < 0:
        return 'tensor<%dx%s>' % (a, e)
    return 'tensor<%dx%dx%s>' % (a, b, e)


def to_ttir_attn(name, d, bd, bm, bn, scale, causal, masked, prec, align):
    """Attention over one block of bm query rows per program (grid x) and one
    (batch, head) per program row (grid y), streaming key/value blocks of bn
    rows through an online softmax, so a score block only ever lives in
    registers.  Every operand is addressed through its own strides: heads
    are read out of, and written back into, projection rows directly.
    align & 1: q, k, v and the output are 16-byte aligned with strides that
    are multiples of 16; align & 4: so are the mask and its batch, head and
    row strides; align & 2: the mask's key stride is 1."""
    F = 'f32'
    P = '!tt.ptr<f32>'
    MD, MN = _t(bm, bd, F), _t(bm, bn, F)
    DN, ND = _t(bd, bn, F), _t(bn, bd, F)
    M1 = _t(bm, -1, F)
    e = Emitter()
    ints = []
    div = ' {tt.divisibility = 16 : i32}'
    divs = [div if align & 1 else '', div if align & 4 else '']
    for nm in ['sq', 'sk', 'H', 'G']:
        ints.append('%%%s: i64' % nm)
    for nm in ['qsb', 'qsh', 'qss', 'ksb', 'ksh', 'kss', 'vsb', 'vsh', 'vss',
               'osb', 'osh', 'oss', 'msb', 'msh', 'msq']:
        ints.append('%%%s: i64%s' % (nm, divs[nm[0] == 'm']))
    ints.append('%msk: i64')
    ints.append('%mper: i64')
    ptrs = []
    for nm in ['Q', 'K', 'V', 'M', 'O']:
        ptrs.append('%%%s: %s%s' % (nm, P, divs[nm == 'M']))
    e.add('module {')
    e.add('  tt.func public @%s(%s, %s) attributes {noinline = false} {'
          % (name, ', '.join(ptrs), ', '.join(ints)))
    e.add('    %c0 = arith.constant 0 : i64')
    e.add('    %%cbn = arith.constant %d : i64' % bn)
    e.add('    %%cbm = arith.constant %d : i32' % bm)
    e.add('    %%zmd = arith.constant dense<0.000000e+00> : %s' % MD)
    e.add('    %%zmn = arith.constant dense<0.000000e+00> : %s' % MN)
    e.add('    %%zdn = arith.constant dense<0.000000e+00> : %s' % DN)
    e.add('    %%znd = arith.constant dense<0.000000e+00> : %s' % ND)
    e.add('    %%zm = arith.constant dense<0.000000e+00> : %s' % M1)
    e.add('    %%ninfm = arith.constant dense<0xFF800000> : %s' % M1)
    e.add('    %%ninfmn = arith.constant dense<0xFF800000> : %s' % MN)
    e.add('    %%scale = arith.constant dense<%s> : %s'
          % (_fliteral(scale), MN))
    e.add('    %%l2emn = arith.constant dense<%s> : %s' % (_LOG2E, MN))
    e.add('    %%l2em = arith.constant dense<%s> : %s' % (_LOG2E, M1))
    e.add('    %%dlr = arith.constant dense<%d> : %s' % (d, _t(1, bd, 'i32')))
    e.add('    %%dlc = arith.constant dense<%d> : %s' % (d, _t(bd, 1, 'i32')))
    if prec == 'split':
        seen = []
        for sh in [(bm, bd), (bd, bn), (bm, bn), (bn, bd)]:
            if sh in seen:
                continue
            seen.append(sh)
            e.add('    %%rnd%dx%d = arith.constant dense<4096> : %s'
                  % (sh[0], sh[1], _t(sh[0], sh[1], 'i32')))
            e.add('    %%top%dx%d = arith.constant dense<-8192> : %s'
                  % (sh[0], sh[1], _t(sh[0], sh[1], 'i32')))
    e.add('    %pm = tt.get_program_id x : i32')
    e.add('    %bh32 = tt.get_program_id y : i32')
    e.add('    %bh = arith.extsi %bh32 : i32 to i64')
    e.add('    %b = arith.divsi %bh, %H : i64')
    e.add('    %h = arith.remsi %bh, %H : i64')
    e.add('    %hk = arith.divsi %h, %G : i64')
    e.add('    %m0 = arith.muli %pm, %cbm : i32')
    _attn_range(e, '%rm', bm, 'i32')
    e.add('    %%m0s = tt.splat %%m0 : i32 -> %s' % _t(bm, -1, 'i32'))
    e.add('    %%om = arith.addi %%m0s, %%rm : %s' % _t(bm, -1, 'i32'))
    _attn_range(e, '%on32', bn, 'i32')
    e.add('    %%on = arith.extsi %%on32 : %s to %s' % (_t(bn, -1, 'i32'),
                                                       _t(bn, -1, 'i64')))
    _attn_range(e, '%od', bd, 'i32')
    _attn_expand(e, '    ', '%omc32', '%om', 1, bm, 'i32')
    e.add('    %%omc = arith.extsi %%omc32 : %s to %s' % (_t(bm, 1, 'i32'),
                                                         _t(bm, 1, 'i64')))
    e.add('    %%sqs = tt.splat %%sq : i64 -> %s' % _t(bm, 1, 'i64'))
    e.add('    %%mok = arith.cmpi slt, %%omc, %%sqs : %s' % _t(bm, 1, 'i64'))
    _attn_expand(e, '    ', '%odr', '%od', 0, bd, 'i32')
    e.add('    %%dokr = arith.cmpi slt, %%odr, %%dlr : %s' % _t(1, bd, 'i32'))
    _attn_bcast(e, '    ', '%mokb', '%mok', bm, 1, bm, bd, 'i1')
    _attn_bcast(e, '    ', '%dokb', '%dokr', 1, bd, bm, bd, 'i1')
    qmask = '%mokb'
    if d < bd:
        e.add('    %%qmask = arith.andi %%mokb, %%dokb : %s'
              % _t(bm, bd, 'i1'))
        qmask = '%qmask'
    _attn_bcast(e, '    ', '%odrb', '%odr', 1, bd, bm, bd, 'i32')
    _attn_expand(e, '    ', '%odc', '%od', 1, bd, 'i32')
    e.add('    %%dokc = arith.cmpi slt, %%odc, %%dlc : %s' % _t(bd, 1, 'i32'))
    _attn_bcast(e, '    ', '%odcb', '%odc', bd, 1, bd, bn, 'i32')
    _attn_bcast(e, '    ', '%dokcb', '%dokc', bd, 1, bd, bn, 'i1')
    _attn_bcast(e, '    ', '%odrv', '%odr', 1, bd, bn, bd, 'i32')
    _attn_bcast(e, '    ', '%dokv', '%dokr', 1, bd, bn, bd, 'i1')
    e.add('    %%skr = tt.splat %%sk : i64 -> %s' % _t(1, bn, 'i64'))
    e.add('    %%skc = tt.splat %%sk : i64 -> %s' % _t(bn, 1, 'i64'))
    _attn_bcast(e, '    ', '%mokmn', '%mok', bm, 1, bm, bn, 'i1')
    _attn_rows(e, 'q', '%Q', '%h', 'qsb', 'qsh', 'qss', bm, bd)
    e.add('    %%qt = tt.load %%qrp, %s, %%zmd : %s' % (qmask, _t(bm, bd, P)))
    if prec == 'split':
        _attn_split(e, '    ', '%qt', bm, bd)
    _attn_base(e, 'k', '%K', '%hk', 'ksb', 'ksh')
    e.add('    %%kbr = tt.splat %%kb : %s -> %s' % (P, _t(1, bn, P)))
    e.add('    %%kssr = tt.splat %%kss : i64 -> %s' % _t(1, bn, 'i64'))
    _attn_base(e, 'v', '%V', '%hk', 'vsb', 'vsh')
    e.add('    %%vbc = tt.splat %%vb : %s -> %s' % (P, _t(bn, 1, P)))
    e.add('    %%vssc = tt.splat %%vss : i64 -> %s' % _t(bn, 1, 'i64'))
    if masked:
        e.add('    %mbat = arith.remsi %b, %mper : i64')
        _attn_base(e, 'm', '%M', '%h', 'msb', 'msh', '%mbat')
        e.add('    %%mqs = tt.splat %%msq : i64 -> %s' % _t(bm, 1, 'i64'))
        e.add('    %%mro = arith.muli %%omc, %%mqs : %s' % _t(bm, 1, 'i64'))
        e.add('    %%mbs = tt.splat %%mb : %s -> %s' % (P, _t(bm, 1, P)))
        e.add('    %%mrow = tt.addptr %%mbs, %%mro : %s, %s'
              % (_t(bm, 1, P), _t(bm, 1, 'i64')))
        _attn_bcast(e, '    ', '%mrowb', '%mrow', bm, 1, bm, bn, P)
        e.add('    %%mskr = tt.splat %%msk : i64 -> %s' % _t(1, bn, 'i64'))
    hi = '%sk'
    if causal:
        e.add('    %hi32 = arith.addi %m0, %cbm : i32')
        e.add('    %hi64 = arith.extsi %hi32 : i32 to i64')
        e.add('    %hi = arith.minsi %sk, %hi64 : i64')
        hi = '%hi'
    e.add('    %%acc:3 = scf.for %%st = %%c0 to %s step %%cbn iter_args('
          '%%mi = %%ninfm, %%li = %%zm, %%ac = %%zmd) -> (%s, %s, %s)  : i64 {'
          % (hi, M1, M1, MD))
    e.add('      %%sts = tt.splat %%st : i64 -> %s' % _t(bn, -1, 'i64'))
    e.add('      %%n = arith.addi %%sts, %%on : %s' % _t(bn, -1, 'i64'))
    _attn_expand(e, '      ', '%nr', '%n', 0, bn, 'i64')
    e.add('      %%nokr = arith.cmpi slt, %%nr, %%skr : %s' % _t(1, bn, 'i64'))
    _attn_bcast(e, '      ', '%nokk', '%nokr', 1, bn, bd, bn, 'i1')
    kmask = '%nokk'
    if d < bd:
        e.add('      %%kmask = arith.andi %%nokk, %%dokcb : %s'
              % _t(bd, bn, 'i1'))
        kmask = '%kmask'
    e.add('      %%kro = arith.muli %%nr, %%kssr : %s' % _t(1, bn, 'i64'))
    e.add('      %%kp1 = tt.addptr %%kbr, %%kro : %s, %s' % (_t(1, bn, P),
                                                          _t(1, bn, 'i64')))
    _attn_bcast(e, '      ', '%kp2', '%kp1', 1, bn, bd, bn, P)
    e.add('      %%kp = tt.addptr %%kp2, %%odcb : %s, %s' % (_t(bd, bn, P),
                                                          _t(bd, bn, 'i32')))
    e.add('      %%kt = tt.load %%kp, %s, %%zdn : %s' % (kmask, _t(bd, bn, P)))
    _attn_dot(e, '%s0', '%qt', '%kt', '%zmn', prec, bm, bd, bn)
    e.add('      %%s1 = arith.mulf %%s0, %%scale : %s' % MN)
    _attn_bcast(e, '      ', '%nokmn', '%nokr', 1, bn, bm, bn, 'i1')
    s = '%s1'
    if masked:
        e.add('      %%mmask = arith.andi %%mokmn, %%nokmn : %s'
              % _t(bm, bn, 'i1'))
        mco = '%nr'
        if not align & 2:
            e.add('      %%mco = arith.muli %%nr, %%mskr : %s'
                  % _t(1, bn, 'i64'))
            mco = '%mco'
        _attn_bcast(e, '      ', '%mcob', mco, 1, bn, bm, bn, 'i64')
        e.add('      %%mp = tt.addptr %%mrowb, %%mcob : %s, %s'
              % (_t(bm, bn, P), _t(bm, bn, 'i64')))
        e.add('      %%mv = tt.load %%mp, %%mmask, %%zmn : %s' % _t(bm, bn, P))
        e.add('      %%s2 = arith.addf %%s1, %%mv : %s' % MN)
        s = '%s2'
    ok = '%nokmn'
    if causal:
        _attn_bcast(e, '      ', '%nrb', '%nr', 1, bn, bm, bn, 'i64')
        _attn_bcast(e, '      ', '%omb', '%omc', bm, 1, bm, bn, 'i64')
        e.add('      %%cz = arith.cmpi sle, %%nrb, %%omb : %s'
              % _t(bm, bn, 'i64'))
        e.add('      %%ok = arith.andi %%nokmn, %%cz : %s' % _t(bm, bn, 'i1'))
        ok = '%ok'
    e.add('      %%s = arith.select %s, %s, %%ninfmn : %s, %s'
          % (ok, s, _t(bm, bn, 'i1'), MN))
    _attn_reduce(e, '%rmax', '%s', 'maxnumf', MN, M1)
    e.add('      %%mn = arith.maxnumf %%mi, %%rmax : %s' % M1)
    _attn_expand(e, '      ', '%mnc', '%mn', 1, bm, F)
    _attn_bcast(e, '      ', '%mnb', '%mnc', bm, 1, bm, bn, F)
    e.add('      %%dd = arith.subf %%s, %%mnb : %s' % MN)
    e.add('      %%d2 = arith.mulf %%dd, %%l2emn : %s' % MN)
    e.add('      %%p = math.exp2 %%d2 : %s' % MN)
    e.add('      %%al0 = arith.subf %%mi, %%mn : %s' % M1)
    e.add('      %%al1 = arith.mulf %%al0, %%l2em : %s' % M1)
    e.add('      %%al = math.exp2 %%al1 : %s' % M1)
    e.add('      %%li1 = arith.mulf %%li, %%al : %s' % M1)
    _attn_reduce(e, '%rsum', '%p', 'addf', MN, M1)
    e.add('      %%li2 = arith.addf %%li1, %%rsum : %s' % M1)
    _attn_expand(e, '      ', '%nc', '%n', 1, bn, 'i64')
    e.add('      %%nokc = arith.cmpi slt, %%nc, %%skc : %s' % _t(bn, 1, 'i64'))
    _attn_bcast(e, '      ', '%nokv', '%nokc', bn, 1, bn, bd, 'i1')
    vmask = '%nokv'
    if d < bd:
        e.add('      %%vmask = arith.andi %%nokv, %%dokv : %s'
              % _t(bn, bd, 'i1'))
        vmask = '%vmask'
    e.add('      %%vro = arith.muli %%nc, %%vssc : %s' % _t(bn, 1, 'i64'))
    e.add('      %%vp1 = tt.addptr %%vbc, %%vro : %s, %s' % (_t(bn, 1, P),
                                                          _t(bn, 1, 'i64')))
    _attn_bcast(e, '      ', '%vp2', '%vp1', bn, 1, bn, bd, P)
    e.add('      %%vp = tt.addptr %%vp2, %%odrv : %s, %s' % (_t(bn, bd, P),
                                                          _t(bn, bd, 'i32')))
    e.add('      %%vt = tt.load %%vp, %s, %%znd : %s' % (vmask, _t(bn, bd, P)))
    _attn_expand(e, '      ', '%alc', '%al', 1, bm, F)
    _attn_bcast(e, '      ', '%alb', '%alc', bm, 1, bm, bd, F)
    e.add('      %%ac1 = arith.mulf %%ac, %%alb : %s' % MD)
    if prec == 'split':
        _attn_split(e, '      ', '%p', bm, bn)
    _attn_dot(e, '%ac2', '%p', '%vt', '%ac1', prec, bm, bn, bd)
    e.add('      scf.yield %%mn, %%li2, %%ac2 : %s, %s, %s' % (M1, M1, MD))
    e.add('    }')
    _attn_expand(e, '    ', '%lic', '%acc#1', 1, bm, F)
    _attn_bcast(e, '    ', '%lib', '%lic', bm, 1, bm, bd, F)
    e.add('    %%res = arith.divf %%acc#2, %%lib : %s' % MD)
    _attn_rows(e, 'o', '%O', '%h', 'osb', 'osh', 'oss', bm, bd)
    e.add('    tt.store %%orp, %%res, %s : %s' % (qmask, _t(bm, bd, P)))
    e.add('    tt.return')
    e.add('  }')
    e.add('}')
    return '\n'.join(e.lines) + '\n'


def _attn_split(e, ind, x, a, b):
    I = _t(a, b, 'i32')
    F = _t(a, b, 'f32')
    e.add('%s%s_i = tt.bitcast %s : %s -> %s' % (ind, x, x, F, I))
    e.add('%s%s_r = arith.addi %s_i, %%rnd%dx%d : %s' % (ind, x, x, a, b, I))
    e.add('%s%s_m = arith.andi %s_r, %%top%dx%d : %s' % (ind, x, x, a, b, I))
    e.add('%s%s_h = tt.bitcast %s_m : %s -> %s' % (ind, x, x, I, F))
    e.add('%s%s_l = arith.subf %s, %s_h : %s' % (ind, x, x, x, F))


def _attn_dot(e, dst, a, b, acc, prec, m, k, n):
    """acc + a b.  'split' is 3xTF32 with a already split into its
    tf32-rounded head a_h and the rest a_l, and b split here: a_l b_h +
    a_h b_l + a_h b_h, the small terms first."""
    A, B, C = _t(m, k, 'f32'), _t(k, n, 'f32'), _t(m, n, 'f32')
    if prec != 'split':
        e.add('      %s = tt.dot %s, %s, %s, inputPrecision = %s : '
              '%s * %s -> %s' % (dst, a, b, acc, prec, A, B, C))
        return
    _attn_split(e, '      ', b, k, n)
    terms = [(a + '_l', b + '_h'), (a + '_h', b + '_l'), (a + '_h', b + '_h')]
    for i in range(3):
        out = dst if i == 2 else '%s_%d' % (dst, i)
        e.add('      %s = tt.dot %s, %s, %s, inputPrecision = tf32 : '
              '%s * %s -> %s' % (out, terms[i][0], terms[i][1], acc, A, B, C))
        acc = out


def _attn_range(e, dst, n, ty):
    e.add('    %s = tt.make_range {end = %d : i32, start = 0 : i32} : %s'
          % (dst, n, _t(n, -1, ty)))


def _attn_expand(e, ind, dst, src, axis, n, ty):
    to = _t(n, 1, ty) if axis == 1 else _t(1, n, ty)
    e.add('%s%s = tt.expand_dims %s {axis = %d : i32} : %s -> %s'
          % (ind, dst, src, axis, _t(n, -1, ty), to))


def _attn_bcast(e, ind, dst, src, a, b, c, d, ty):
    e.add('%s%s = tt.broadcast %s : %s -> %s'
          % (ind, dst, src, _t(a, b, ty), _t(c, d, ty)))


def _attn_base(e, x, ptr, h, sb, sh, b='%b'):
    e.add('    %%%so1 = arith.muli %s, %%%s : i64' % (x, b, sb))
    e.add('    %%%so2 = arith.muli %s, %%%s : i64' % (x, h, sh))
    e.add('    %%%so = arith.addi %%%so1, %%%so2 : i64' % (x, x, x))
    e.add('    %%%sb = tt.addptr %s, %%%so : !tt.ptr<f32>, i64'
          % (x, ptr, x))


def _attn_rows(e, x, ptr, h, sb, sh, ss, bm, bd):
    P = '!tt.ptr<f32>'
    _attn_base(e, x, ptr, h, sb, sh)
    e.add('    %%%sstr = tt.splat %%%s : i64 -> %s'
          % (x, ss, _t(bm, 1, 'i64')))
    e.add('    %%%srow = arith.muli %%omc, %%%sstr : %s'
          % (x, x, _t(bm, 1, 'i64')))
    e.add('    %%%sbs = tt.splat %%%sb : %s -> %s'
          % (x, x, P, _t(bm, 1, P)))
    e.add('    %%%sp1 = tt.addptr %%%sbs, %%%srow : %s, %s'
          % (x, x, x, _t(bm, 1, P), _t(bm, 1, 'i64')))
    _attn_bcast(e, '    ', '%%%sp2' % x, '%%%sp1' % x, bm, 1, bm, bd, P)
    e.add('    %%%srp = tt.addptr %%%sp2, %%odrb : %s, %s'
          % (x, x, _t(bm, bd, P), _t(bm, bd, 'i32')))


def _attn_reduce(e, dst, src, op, MN, M1):
    e.add('      %s = "tt.reduce"(%s) <{axis = 1 : i32}> ({' % (dst, src))
    e.add('      ^bb0(%ra: f32, %rb: f32):')
    e.add('        %%rc = arith.%s %%ra, %%rb : f32' % op)
    e.add('        tt.reduce.return %rc : f32')
    e.add('      }) : (%s) -> %s' % (MN, M1))
