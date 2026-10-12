#!/usr/bin/env python3
"""Mechanical audit of an interpreter change rule.

A change rule must be derived by specialising the interpreter, not written
against the transition code.  A rule fails this requirement when it encodes
the adapter, for example by testing the source representation, naming parts
of the target representation, or using kernel-specific constants.  The
audit checks this property mechanically, so it can be re-run rather than
taken on trust.

A rule module passes when:

  * every import it makes is in ALLOWED_IMPORTS, the abstract tensor
    vocabulary;
  * no identifier it mentions, at any depth, is in FORBIDDEN, the names of
    the target representation and of the machinery that produces it;
  * it branches only on declared abstract properties (shape, dtype, axis),
    never on the source representation;
  * it names no tile, block, warp or kernel-signature constant.

The audit prints the full vocabulary of each rule.

    audit_rules.py MODULE.py [MODULE.py ...]

Exit status is non-zero if any rule fails.
"""
import ast
import os
import sys

# Modules a change rule may import from; widening this list is visible in a diff.
ALLOWED_IMPORTS = {
    "rpython.metatensor.ops",        # the interpreter's primitive operations
    "rpython.rlib.jit",              # annotations only; checked below
    "math",
    # membership, declared policy and uncommitted contributions, nothing about their representation
    "pypy.module._metatensor.motion_iface",
}

# Names from the target representation and its machinery; a rule mentioning
# them depends on the target representation.
FORBIDDEN = {
    # the fused representation and its parts
    "KERNEL", "NODE", "NODEARRAY", "SHAPEARRAY", "kernel", "kernels",
    "node_index", "add_output", "launched_kernel", "launched_box",
    "force_box", "force_as_extra_output", "vtensor_info", "VTensorInfo",
    "compile_or_reuse", "kernel_key", "drop_kernel", "pending",
    # the emitter and the device toolchain
    "ttir", "to_ttir", "triton", "triton_compile", "cubin", "ptx",
    "row_tile", "row_warps", "num_warps", "rowmode", "flat_block",
    "launch", "launch_gpu", "eval_op", "runtime",
    # the optimizer itself
    "optimizeopt", "metainterp", "resoperation", "ResOperation", "optforce",
    "Optimization", "getptrinfo", "emit_extra",
    # representation tests
    "is_virtual", "_is_virtual", "virtual", "forced", "unforce",
    # the recovery machinery a leave rule must not know about: guards, resume
    # data, and the representation the meta-tracer gives contributions
    "resume", "resumedata", "numb", "snapshot", "fail_args", "guard",
    "invalidate", "quasi_immut", "blackhole", "bridge", "fused", "fusion",
    "parts", "Part", "retains", "total", "materialise", "materialize",
    "interp_group", "W_Group", "W_Aggregate", "version",
}

# Forbidden as attributes but ordinary as local names: `.pending` is the fusion
# pass's kernel list, a parameter called `pending` is not.
FORBIDDEN_ATTR_ONLY = {"pending"}

# Constants that only mean something given the kernel signature. Small ints
# (axes, arity) are fine; tile or block sizes are target knowledge.
SUSPICIOUS_INTS = {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192}


class Audit(object):
    def __init__(self, path):
        self.path = path
        self.tree = ast.parse(open(path).read(), path)
        self.failures = []
        self.imports = []
        self.names = set()
        self.attrs = set()
        self.ints = set()

    def fail(self, node, why):
        self.failures.append("%s:%d: %s" % (os.path.basename(self.path),
                                            getattr(node, "lineno", 0), why))

    def run(self):
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    self.imports.append(a.name)
                    if a.name not in ALLOWED_IMPORTS:
                        self.fail(node, "imports %s, which is not in the "
                                        "declared vocabulary" % a.name)
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                self.imports.append(mod)
                if mod not in ALLOWED_IMPORTS:
                    self.fail(node, "imports from %s, which is not in the "
                                    "declared vocabulary" % mod)
            elif isinstance(node, ast.Name):
                self.names.add(node.id)
                if node.id in FORBIDDEN and node.id not in FORBIDDEN_ATTR_ONLY:
                    self.fail(node, "mentions %r, a name from the target "
                                    "representation" % node.id)
            elif isinstance(node, ast.Attribute):
                self.attrs.add(node.attr)
                if node.attr in FORBIDDEN:
                    self.fail(node, "mentions .%s, a name from the target "
                                    "representation" % node.attr)
            elif isinstance(node, ast.Constant) and isinstance(node.value, int):
                self.ints.add(node.value)
                if node.value in SUSPICIOUS_INTS:
                    self.fail(node, "uses the constant %d, which is a tile or "
                                    "block size, not an abstract quantity"
                              % node.value)
        return not self.failures

    def report(self):
        print("== %s" % self.path)
        print("   imports:    %s" % (", ".join(sorted(set(self.imports)))
                                     or "none"))
        vocab = sorted(self.names | self.attrs)
        print("   vocabulary: %s" % (", ".join(vocab) or "none"))
        print("   constants:  %s" % (", ".join(str(i) for i in
                                               sorted(self.ints)) or "none"))
        if self.failures:
            for f in self.failures:
                print("   FAIL %s" % f)
        else:
            print("   pass: the rule speaks only the abstract vocabulary")


def main(argv):
    paths = argv[1:]
    if not paths:
        print(__doc__.strip())
        return 2
    ok = True
    for p in paths:
        a = Audit(p)
        if not a.run():
            ok = False
        a.report()
    print("\n%s" % ("all rules pass the audit" if ok
                    else "AUDIT FAILED: a rule depends on the target representation"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
