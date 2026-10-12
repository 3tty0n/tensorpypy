"""Negative control: layout_rule.py plus a source test, a target name and a tile-sized constant.

benchmark/eval/audit_rules.py must fail this file.
"""
from rpython.metatensor import kernels


def blocked_shape(rows, cols, heads):
    return heads * rows, cols // heads


def apply(t, heads):
    if t.is_virtual():                      # a test of the source
        kernels.add_output(t.kernel, 0)     # a name from the target
    if t.cols % 4096 == 0:                  # a tile-sized constant
        return t.head_split(heads)
    return t.head_split(heads)
