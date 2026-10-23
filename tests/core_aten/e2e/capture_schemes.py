"""Real model2MLIR captures for signed/asymmetric and unsigned/narrow schemes."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import m2m
from capture import QD
from capture_rtl import rtl_set


class Scheme(torch.nn.Module):
    def __init__(self, prototype, low, high, zp):
        super().__init__()
        self.fn = prototype.fn
        self.scales = prototype.scales
        self.kinds = prototype.dtypes
        self.out_scales = getattr(prototype, "s_outs", [prototype.s_out])
        self.low, self.high, self.zp = low, high, zp
        self.dtype = torch.uint8 if low >= 0 else torch.int8

    def forward(self, *codes):
        values = []
        for code, scale, kind in zip(codes, self.scales, self.kinds):
            if kind == "i32":
                values.append(
                    QD.dequantize_per_tensor(
                        code, scale, 0, -(1 << 31), (1 << 31) - 1, torch.int32
                    )
                )
            else:
                values.append(
                    QD.dequantize_per_tensor(
                        code, scale, self.zp, self.low, self.high, self.dtype
                    )
                )
        result = self.fn(*values)
        if isinstance(result, tuple):
            return tuple(
                QD.quantize_per_tensor(v, s, self.zp, self.low, self.high, self.dtype)
                for v, s in zip(result, self.out_scales)
            )
        return QD.quantize_per_tensor(
            result, self.out_scales[0], self.zp, self.low, self.high, self.dtype
        )


def capture(root):
    root.mkdir(parents=True, exist_ok=True)
    selected = {
        "neg",
        "mean_dim",
        "add_Tensor",
        "cat",
        "mm",
        "convolution",
        "convolution_backward",
    }
    rng = np.random.default_rng(12751)
    summary = {}
    for name, prototype, original in rtl_set():
        if name not in selected:
            continue
        for low, high, zp in [(-127, 127, -7), (1, 255, 128)]:
            label = name + ("_uint8_narrow" if low >= 0 else "_int8_asymmetric")
            wrapper = Scheme(prototype, low, high, zp).eval()
            if name == "neg":
                wrapper.out_scales = [float(np.float32(prototype.scales[0] * 2))]
            tensors = []
            for code, kind in zip(original, prototype.dtypes):
                if kind == "i32":
                    data = rng.integers(-128, 128, tuple(code.shape), dtype=np.int32)
                else:
                    shape = (1, 256) if name == "neg" else tuple(code.shape)
                    codes = rng.permutation(
                        np.arange(0, 256) if low >= 0 else np.arange(-128, 128)
                    )
                    data = np.resize(codes, shape).astype(
                        np.uint8 if low >= 0 else np.int8
                    )
                tensors.append(torch.from_numpy(data))
            converted = m2m.convert(wrapper, tuple(tensors), backend="fx_importer")
            assert converted.ok, (label, converted.diagnostics)
            directory = root / label
            directory.mkdir(exist_ok=True)
            (directory / "model.mlir").write_text(converted.mlir_text)
            np.savez(
                directory / "inputs.npz",
                **{
                    f"arg{i}": x.numpy().view(np.int8)
                    if x.dtype == torch.uint8
                    else x.numpy()
                    for i, x in enumerate(tensors)
                },
            )
            eager = wrapper(*tensors)
            for i, value in enumerate(eager if isinstance(eager, tuple) else (eager,)):
                array = value.numpy()
                array = array.view(np.int8) if array.dtype == np.uint8 else array
                np.save(
                    directory
                    / (
                        f"expected{i}.npy"
                        if isinstance(eager, tuple)
                        else "expected.npy"
                    ),
                    array,
                )
            summary[label] = dict(
                quant_min=low,
                quant_max=high,
                zero_point=zp,
                storage_dtype=str(wrapper.dtype),
                scale_canonicalization="fp32_once_at_capture",
            )
    (root / "capture_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    capture(parser.parse_args().output)
