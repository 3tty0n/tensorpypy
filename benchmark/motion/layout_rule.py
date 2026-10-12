"""A change rule: the same values, addressed as blocks instead of as a matrix.

blocked[b * rows + i, j] = matrix[i, b * (cols // heads) + j], with `heads` dividing `cols`.
The equation names no storage, device, kernel or layout. Checked by benchmark/eval/audit_rules.py.
"""


def index_map(rows, cols, heads, b, i, j):
    """Where blocked[b, i, j] comes from in the matrix: (row, column)."""
    return i, b * (cols // heads) + j


def blocked_shape(rows, cols, heads):
    return heads * rows, cols // heads


def apply(t, heads):
    """The rule, in the interpreter's vocabulary."""
    return t.head_split(heads)


def unapply(t, heads):
    """The same equation, read right to left."""
    return t.head_merge(heads)


def reference(values, rows, cols, heads):
    """The rule on plain numbers, to check an implementation against the equation."""
    br, bc = blocked_shape(rows, cols, heads)
    out = [0.0] * (br * bc)
    for b in range(heads):
        for i in range(rows):
            for j in range(bc):
                r, c = index_map(rows, cols, heads, b, i, j)
                out[(b * rows + i) * bc + j] = values[r * cols + c]
    return out
