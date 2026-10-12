"""Shared per-iteration warm-up trace for common.py, torch_common.py and jax_models.py.
WARMUP_TRACE=N in the environment prints N `iter=<i> us=<latency>` lines, then one summary line.
"""
import os


def trace_n():
    return int(os.environ.get('WARMUP_TRACE', '0') or '0')


def _median(xs):
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def steady_at(us, steady_us, window=20, tol=0.05):
    """First iteration i (0-indexed, as in `iter=`) whose next `window` forwards are all within `tol` of steady_us, else 'none'."""
    lo, hi = steady_us * (1 - tol), steady_us * (1 + tol)
    for i in range(len(us) - window + 1):
        if all(lo <= x <= hi for x in us[i:i + window]):
            return i
    return 'none'


def report_trace(us, compiled=None):
    """us: per-forward latencies in microseconds. Prints the `total_us=... steady_us=... steady_at=...` line.

    compiled (ours only): per-forward kernels compiled through Triton (disk-cache misses). Adds cold_compiles and first_forward_compiles."""
    last = us[-50:] if len(us) >= 50 else us
    steady_us = _median(last)
    line = 'total_us=%.1f steady_us=%.1f steady_at=%s' % (
        sum(us), steady_us, steady_at(us, steady_us))
    if compiled is not None:
        line += ' cold_compiles=%d first_forward_compiles=%d' % (
            sum(compiled), compiled[0] if compiled else 0)
    print(line)
