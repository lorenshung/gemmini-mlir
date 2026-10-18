"""Quantize one corpus case PT2E-style (per-tensor symmetric int8 around the op) and capture it with m2m.

usage: cap_case.py CORPUS OVERLOAD BUNDLE_DIR   (writes BUNDLE_DIR/model.mlir and eager_q.npy)
The bundle's own semantic_io contract (outputs then post-call inputs) is kept: the module returns the
dequantized result followed by its tensor inputs.
"""
import json, sys
import numpy as np, torch
import torch.ao.quantization.fx._decomposed  # noqa
import m2m
QD = torch.ops.quantized_decomposed
corpus, overload, bundle = sys.argv[1:4]
case = next(c for c in json.load(open(corpus))["cases"] if c["overload"] == overload)
_, name, ov = overload.split(".")
op = getattr(getattr(torch.ops.aten, name), ov)
tensors = []
def dec(x):
    if isinstance(x, dict) and x.get("kind") == "tensor":
        tensors.append(torch.tensor(x["values"], dtype=getattr(torch, x["dtype"])).reshape(x["shape"]))
        return ("T", len(tensors) - 1)
    if isinstance(x, list):
        return [dec(v) for v in x]
    return x
def kw(v):
    if isinstance(v, dict) and v.get("kind") == "dtype":
        return getattr(torch, v["value"])
    if isinstance(v, dict) and v.get("kind") == "memory_format":
        return getattr(torch, v["value"])
    return v
args = dec(case["arguments"]["args"]["items"])
kwargs = {k: kw(v) for k, v in case["arguments"]["kwargs"].items()}
# a quantized graph carries int8 codes; a floating dtype request only changes the host container
if isinstance(kwargs.get("dtype"), torch.dtype) and kwargs["dtype"].is_floating_point and overload != "aten._to_copy.default":
    kwargs.pop("dtype")
# PT2E: a contraction's bias is int32 at the accumulator scale s_a * s_w
BIAS = {"aten.addmm.default": (0, 1, 2), "aten.convolution.default": (2, 0, 1)}
sc = lambda t: max(float(t.abs().max()), 1e-8) / 127.5
s_in = [sc(t) for t in tensors]
dts = [torch.int8] * len(tensors)
if overload in BIAS and len(tensors) > BIAS[overload][0]:
    b, a_, w_ = BIAS[overload]
    s_in[b] = s_in[a_] * s_in[w_]
    dts[b] = torch.int32
LIM = {torch.int8: (-128, 127), torch.int32: (-(2 ** 31), 2 ** 31 - 1)}
def qdq(t, s, dt):
    lo, hi = LIM[dt]
    return QD.dequantize_per_tensor(QD.quantize_per_tensor(t, s, 0, lo, hi, dt), s, 0, lo, hi, dt)
def build(vals):
    def rebuild(x):
        if isinstance(x, tuple) and len(x) == 2 and x[0] == "T":
            return vals[x[1]]
        if isinstance(x, list):
            return [rebuild(v) for v in x]
        return x
    return [rebuild(a) for a in args]
with torch.no_grad():
    deq0 = [qdq(t, s, dt) for t, s, dt in zip(tensors, s_in, dts)]
    y0 = op(*build(deq0), **kwargs)
multi = isinstance(y0, (tuple, list))
s_out = [sc(y) for y in y0] if multi else sc(y0)
class ToCopy(torch.nn.Module):
    """_to_copy in a quantized graph: the int8 codes widen to int32 (then the host container dtype)."""
    def forward(self, a):
        q = QD.quantize_per_tensor(a, s_in[0], 0, -128, 127, torch.int8)
        w = torch.ops.aten._to_copy.default(q, dtype=torch.int32)
        return w.to(kwargs.get("dtype", torch.float32)), a


class M(torch.nn.Module):
    def forward(self, *xs):
        ds = [qdq(t, s, dt) for t, s, dt in zip(xs, s_in, dts)]
        y = op(*build(ds), **kwargs)
        if multi:
            return (*[qdq(v, so, torch.int8) for v, so in zip(y, s_out)], *xs)
        return (qdq(y, s_out, torch.int8), *xs)
if overload == "aten._to_copy.default":
    M = ToCopy
    multi, s_out = False, 1.0
r = m2m.convert(M().eval(), tuple(tensors), backend="fx_importer")
assert r.ok, r.diagnostics
open(f"{bundle}/model.mlir", "w").write(r.mlir_text)
outs = M()(*tensors)
json.dump(dict(s_out=s_out if multi else [s_out],
               sizes=[int(o.numel()) for o in (outs[:len(s_out)] if multi else outs[:1])]),
          open(f"{bundle}/eager_meta.json", "w"))
np.save(f"{bundle}/eager_q.npy", np.concatenate([o.detach().numpy().reshape(-1) for o in (outs[:len(s_out)] if multi else outs[:1])]))
print(overload, [tuple(t.shape) for t in tensors], s_out)
