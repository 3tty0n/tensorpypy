"""Checks of the compat torch layer, run under a translated pypy-c:

    pypy-c lib_pypy/tensorpypy/compat/test_compat.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _metatensor  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402


def close(a, b, tol=1e-5):
    a, b = a.tolist(), b.tolist()
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert abs(x - y) <= tol * (1.0 + abs(y)), (x, y)


def vals(n, k=0):
    return [((i * 7 + k * 3) % 11) * 0.25 - 1.0 for i in range(n)]


class WeightStd(nn.Module):
    def __init__(self):
        self.weight = nn.Parameter(4, 3)
        self.gain = nn.Parameter(4, 1)

    def forward(self, x):
        w = self.weight - self.weight.mean(-1, keepdim=True)
        w = w * (w * w).mean(-1, keepdim=True).add(1e-6).rsqrt()
        w = (w * self.gain).permute(1, 0).contiguous()
        return torch.matmul(x, w)


def reference(m, x):
    w = torch.from_flat(m.weight.tolist(), (4, 3), torch.float32)
    g = torch.from_flat(m.gain.tolist(), (4, 1), torch.float32)
    w = w - w.mean(-1, keepdim=True)
    w = w * (w * w).mean(-1, keepdim=True).add(1e-6).rsqrt()
    w = (w * g).permute(1, 0).contiguous()
    return torch.matmul(x, w)


def test_parameter_values_are_computed_once():
    m = WeightStd()
    m.load_state_dict({
        "weight": torch.from_flat(vals(12), (4, 3), torch.float32),
        "gain": torch.from_flat(vals(4, 1), (4, 1), torch.float32)})
    x = torch.from_flat(vals(6, 2), (2, 3), torch.float32)
    y = m(x)
    close(y, reference(m, x))
    n0 = _metatensor.launch_count()
    y2 = m(x)
    close(y2, y)
    assert _metatensor.launch_count() - n0 <= 2


def test_writing_a_parameter_invalidates():
    m = WeightStd()
    m.load_state_dict({
        "weight": torch.from_flat(vals(12), (4, 3), torch.float32),
        "gain": torch.from_flat(vals(4, 1), (4, 1), torch.float32)})
    x = torch.from_flat(vals(6, 2), (2, 3), torch.float32)
    y = m(x)
    m.gain.mul_(2.0)
    close(m(x), reference(m, x))
    close(m(x), y * 2.0)
    m.weight[0, 0] = 3.0
    close(m(x), reference(m, x))
    m.load_state_dict({
        "weight": torch.from_flat(vals(12, 5), (4, 3), torch.float32)})
    close(m(x), reference(m, x))


def test_derived_values_are_not_shared_writes():
    m = WeightStd()
    m.load_state_dict({
        "weight": torch.from_flat(vals(12), (4, 3), torch.float32),
        "gain": torch.from_flat(vals(4, 1), (4, 1), torch.float32)})
    a = m.weight * 2.0
    b = m.weight * 2.0
    a.add_(1.0)
    close(b, m.weight * 2.0)
    close(a, m.weight * 2.0 + 1.0)


def test_permute_back_to_channels_last():
    h = vals(2 * 4 * 5 * 3)
    x = torch.from_flat(h, (2, 4, 5, 3), torch.float32)
    y = x.permute(0, 3, 1, 2)
    assert torch._is_nhwc(y)
    want = [h[((n * 4 + i) * 5 + j) * 3 + c] for n in range(2)
            for c in range(3) for i in range(4) for j in range(5)]
    close(y.contiguous(), torch.from_flat(want, (2, 3, 4, 5), torch.float32))
    close(y * 2.0 + y, torch.from_flat(want, (2, 3, 4, 5),
                                      torch.float32) * 3.0)


def test_channel_shuffle_stays_channels_last():
    def shuffle(x, g):
        n, c, h, w = x.shape
        x = x.view(n, g, c // g, h, w).transpose(1, 2).contiguous()
        return x.view(n, -1, h, w)
    x = torch.from_flat(vals(2 * 6 * 3 * 4), (2, 6, 3, 4), torch.float32)
    y = x.permute(0, 2, 3, 1).contiguous().permute(0, 3, 1, 2)
    assert torch._is_nhwc(y)
    s = shuffle(y, 3)
    assert torch._is_nhwc(s)
    close(s, shuffle(x, 3))
    close(s * 2.0 + 1.0, shuffle(x, 3) * 2.0 + 1.0)


def test_sum_over_middle_dims():
    x = torch.from_flat(vals(2 * 3 * 4 * 5), (2, 3, 4, 5), torch.float32)
    h = x.tolist()
    s = x.sum((1, 2), keepdim=True)
    assert tuple(s.shape) == (2, 1, 1, 5)
    want = [sum(h[((a * 3 + i) * 4 + j) * 5 + b]
                for i in range(3) for j in range(4))
            for a in range(2) for b in range(5)]
    close(s, torch.from_flat(want, (2, 1, 1, 5), torch.float32))
    close(x.mean(1), (x.sum(1) * (1.0 / 3)))
    y = x.permute(0, 3, 1, 2)
    assert torch._is_nhwc(y)
    close(y.sum((2, 3), keepdim=True), y.contiguous().sum((2, 3), True))
    close(y.norm(2, (2, 3)), y.contiguous().norm(2, (2, 3)))


def _sdpa_both(*args, **kw):
    got = F.scaled_dot_product_attention(*args, **kw)
    F._FLASH = False
    try:
        want = F.scaled_dot_product_attention(*args, **kw)
    finally:
        F._FLASH = True
    assert tuple(got.shape) == tuple(want.shape)
    close(got.contiguous(), want.contiguous(), 1e-4)
    return got


def test_flash_attention_matches_unfused():
    b, s, h, d = 2, 37, 4, 16
    qkv = torch.from_flat(vals(b * s * 3 * h * d), (b, s, 3 * h * d),
                          torch.float32)
    q, k, v = qkv.reshape(b, s, 3, h, d).permute(2, 0, 3, 1, 4).unbind(0)
    assert F._flash(q, k, v, None, False, None) is not None
    out = _sdpa_both(q, k, v)
    assert out.perm is not None
    _sdpa_both(q, k, v, is_causal=True)
    kv = torch.from_flat(vals(b * 50 * 2 * d, 3), (b, 50, 2 * d),
                         torch.float32)
    kk, vv = kv.view(b, 50, 2, 1, d).permute(2, 0, 3, 1, 4).unbind(0)
    kk, vv = kk.expand(b, h, 50, d), vv.expand(b, h, 50, d)
    mask = torch.from_flat(vals(b * s * 50, 5), (b, 1, s, 50), torch.float32)
    _sdpa_both(q, kk.contiguous(), vv.contiguous(), attn_mask=mask)
    g = qkv.reshape(b, s, 3 * h, d)[:, :, :2].transpose(1, 2).contiguous()
    _sdpa_both(q.contiguous(), g, g, enable_gqa=True)


def _eager_attention(q, k, v, mask):
    w = torch.matmul(q, k.transpose(2, 3)) * 0.25
    w = F.softmax(w + mask, dim=-1).to(q.dtype)
    return torch.matmul(F.dropout(w), v), w


def test_eager_attention_runs_fused():
    b, s, h, d = 2, 21, 2, 16
    x = torch.from_flat(vals(b * s * 3 * h * d, 4), (b, s, 3 * h * d),
                        torch.float32)
    q, k, v = x.reshape(b, s, 3, h, d).permute(2, 0, 3, 1, 4).unbind(0)
    mask = torch.from_flat(vals(b * s * s, 6), (b, 1, s, s), torch.float32)
    out, w = _eager_attention(q, k, v, mask)
    assert w.__class__ is torch._Attn
    F._FLASH = False
    try:
        want, ww = _eager_attention(q, k, v, mask)
    finally:
        F._FLASH = True
    close(out.contiguous(), want.contiguous(), 1e-4)
    close(w, ww, 1e-5)


def _tridiag(a, b, c, d):
    n = a.shape[-1]
    for i in range(1, n):
        w = a[..., i] / b[..., i - 1]
        b[..., i] += -w * c[..., i - 1]
        d[..., i] += -w * d[..., i - 1]
    out = torch.empty_like(a)
    out[..., -1] = d[..., -1] / b[..., -1]
    for i in range(n - 2, -1, -1):
        out[..., i] = (d[..., i] - c[..., i] * out[..., i + 1]) / b[..., i]
    return out


def _ref_tridiag(a, b, c, d, rows, n):
    out = []
    for r in range(rows):
        aa, bb, cc, dd = [x[r * n:(r + 1) * n] for x in (a, b, c, d)]
        for i in range(1, n):
            w = aa[i] / bb[i - 1]
            bb[i] -= w * cc[i - 1]
            dd[i] -= w * dd[i - 1]
        o = [0.0] * n
        o[-1] = dd[-1] / bb[-1]
        for i in range(n - 2, -1, -1):
            o[i] = (dd[i] - cc[i] * o[i + 1]) / bb[i]
        out.extend(o)
    return out


def test_repeated_box_writes_match_a_host_solve():
    old = torch._INPLACE_MIN
    torch._INPLACE_MIN = 1
    try:
        rows, n = 6, 9
        a, c, d = [vals(rows * n, k) for k in (1, 2, 3)]
        b = [4.0 + v for v in vals(rows * n, 4)]
        want = _ref_tridiag(a, list(b), c, list(d), rows, n)
        for _ in range(8):
            t = [torch.from_flat(x, (rows, n), torch.float32)
                 for x in (a, b, c, d)]
            bt = t[1] * 1.0
            dt = t[3] * 1.0
            got = _tridiag(t[0], bt, t[2], dt)
            close(got, torch.from_flat(want, (rows, n), torch.float32), 1e-4)
            close(t[1], torch.from_flat(b, (rows, n), torch.float32))
            close(t[3], torch.from_flat(d, (rows, n), torch.float32))
    finally:
        torch._INPLACE_MIN = old


def test_box_writes_keep_other_tensors_apart():
    old = torch._INPLACE_MIN
    torch._INPLACE_MIN = 1
    try:
        for _ in range(8):
            x = torch.zeros(4, 5)
            y = torch.zeros(4, 5)
            x[0] = 1.0
            y[0] = 1.0
            x[1] = 2.0
            assert x.tolist()[5:10] == [2.0] * 5
            assert y.tolist()[5:10] == [0.0] * 5
            c = x.clone()
            v = x.view(20)
            r = x[2]
            s = x[0:2, 1:3]
            x[2] = 3.0
            x[0, 1] = 9.0
            assert c.tolist()[10:15] == [0.0] * 5
            assert c.tolist()[1] == 1.0
            assert v.tolist()[10:15] == [0.0] * 5
            assert r.tolist() == [0.0] * 5
            assert s.tolist() == [1.0, 1.0, 2.0, 2.0]
            assert x.tolist()[10:15] == [3.0] * 5
            assert x.tolist()[:2] == [1.0, 9.0]
            z = x
            x[3] = 4.0
            assert z.tolist()[15:20] == [4.0] * 5
            x[3, 1:4] = torch.from_flat([5.0, 6.0, 7.0], (3,),
                                        torch.float32)
            assert x.tolist()[15:20] == [4.0, 5.0, 6.0, 7.0, 4.0]
    finally:
        torch._INPLACE_MIN = old
def _both(flags, fn):
    got = fn()
    old = [getattr(m, k) for m, k in flags]
    for m, k in flags:
        setattr(m, k, False)
    try:
        want = fn()
    finally:
        for (m, k), v in zip(flags, old):
            setattr(m, k, v)
    return got, want


def test_linear_bias_in_the_gemm():
    x = torch.from_flat(vals(6 * 5), (2, 3, 5), torch.float32)
    w = torch.from_flat(vals(4 * 5, 1), (4, 5), torch.float32)
    b = torch.from_flat(vals(4, 2), (4,), torch.float32)
    for _ in range(5):
        got, want = _both([(F, "_GEMM_BIAS")], lambda: F.linear(x, w, b))
        close(got, want, 1e-4)
        wt = torch.from_flat(vals(5 * 4, 3), (5, 4), torch.float32)
        x2 = torch.from_flat(vals(6 * 5), (6, 5), torch.float32)
        got, want = _both([(F, "_GEMM_BIAS")],
                          lambda: torch.addmm(b, x2, wt))
        close(got, want, 1e-4)


def test_cross_entropy_and_log_softmax_rows():
    for n, v in [(6, 7), (5, 9000)]:
        x = torch.from_flat(vals(n * v, 4), (n, v), torch.float32) * 3.0
        t = torch.tensor([(i * 5) % v for i in range(n - 1)] + [-100])
        for _ in range(5):
            got, want = _both([(F, "_LSE")],
                              lambda: F.cross_entropy(x, t))
            close(got.view(1), want.view(1), 1e-4)
            got, want = _both([(F, "_LSE")],
                              lambda: F.log_softmax(x, -1))
            close(got, want, 1e-4)


def _composed_norm(x, g, b):
    mean = x.mean(-1, keepdim=True)
    std = x.std(-1, keepdim=True)
    return g * (x - mean) / (std + 1e-6) + b


def test_composed_norm_fuses_alike():
    x = torch.from_flat(vals(4 * 3 * 8, 5), (4, 3, 8), torch.float32)
    y = torch.from_flat(vals(4 * 3 * 8, 6), (4, 3, 8), torch.float32)
    g = torch.from_flat(vals(8, 1), (8,), torch.float32)
    b = torch.from_flat(vals(8, 2), (8,), torch.float32)
    lw = torch.from_flat(vals(8, 3), (8,), torch.float32)
    for _ in range(5):
        got, want = _both([(torch, "_BCAST"), (F, "_ROWS2")],
                          lambda: _composed_norm(x + y, g, b))
        close(got, want, 1e-4)
        got, want = _both([(torch, "_BCAST"), (F, "_ROWS2")],
                          lambda: F.layer_norm(x * 2.0 + y, (8,), lw, b))
        close(got, want, 1e-4)
        got, want = _both([(torch, "_BCAST"), (F, "_ROWS2")],
                          lambda: F.softmax(x - x.amax(-1, keepdim=True),
                                            -1))
        close(got, want, 1e-5)


def _window_attention(q, k, v, bias, mask):
    w = (q * 0.25).__matmul__(k.transpose(-2, -1)) + bias
    nw, n = mask.shape[0], q.shape[2]
    w = w.view(-1, nw, q.shape[1], n, n) + mask.unsqueeze(1).unsqueeze(0)
    w = w.view(-1, q.shape[1], n, n).softmax(-1)
    return w.__matmul__(v)


def test_window_attention_runs_fused():
    b, s, h, d, nw = 4, 9, 2, 16, 2
    x = torch.from_flat(vals(b * s * 3 * h * d, 7), (b, s, 3 * h * d),
                        torch.float32)
    q, k, v = x.reshape(b, s, 3, h, d).permute(2, 0, 3, 1, 4).unbind(0)
    bias = torch.from_flat(vals(h * s * s, 8), (1, h, s, s), torch.float32)
    mask = torch.from_flat(vals(nw * s * s, 9), (nw, s, s), torch.float32)
    w = (q * 0.25).__matmul__(k.transpose(-2, -1)) + bias
    w = w.view(-1, nw, h, s, s) + mask.unsqueeze(1).unsqueeze(0)
    assert w.__class__ is torch._Attn
    assert w.view(-1, h, s, s).softmax(-1)._per == nw
    out = _window_attention(q, k, v, bias, mask)
    F._FLASH = False
    try:
        want = _window_attention(q, k, v, bias, mask)
    finally:
        F._FLASH = True
    close(out.contiguous(), want.contiguous(), 1e-4)


if __name__ == "__main__":
    for name, f in sorted(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok %s" % name)
