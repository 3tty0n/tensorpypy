"""A change rule: the same vector, addressed along the other axis.

For square m (side n) and vector v (n values), along columns out[i, j] = m[i, j] + v[j]
and along rows out[i, j] = m[i, j] + v[i]. The equation names no storage, device or kernel.
Checked by benchmark/eval/audit_rules.py.
"""


def index_map(axis, i, j):
    """Which element of v the output at (i, j) reads."""
    return j if axis == 'columns' else i


def vector_shape(axis, n):
    return [1, n] if axis == 'columns' else [n, 1]


def apply(m, v):
    """The rule in the interpreter's vocabulary. `v` carries the axis in its shape ([1, n] or [n, 1])."""
    return m.add(v)


def reference(mvals, vvals, n, axis):
    """The rule on plain numbers, to check an implementation against the equation."""
    out = 0.0
    for i in range(n):
        for j in range(n):
            out += mvals[i * n + j] + vvals[index_map(axis, i, j)]
    return out
