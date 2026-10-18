"""Command buffers using the final v0.1 extension opcodes, in Merlin's own form.

Each case is a buffer ``{tensors, commands}`` exactly as Merlin's reference engine reads it; the
interface text is rendered from it by Merlin's ``emit_interface_mlir``, so the backend sees the same
surface ``capsule_grade`` hands it.  ``representative`` selects larger shapes.
"""

from __future__ import annotations

import numpy as np


def _t(shape, role="input", dtype="i8"):
    return {"shape": list(shape), "dtype": dtype, "role": role}


def _cb(tensors, commands):
    return {"abi_version": "0.1", "target": "gemmini", "tensors": tensors, "commands": commands}


def _one(opcode, srcs: dict, dst_shape, attrs, dst_dtype="i8"):
    tensors = {name: _t(sh) for name, sh in srcs.items()}
    keys = {"AFFINE_MAP": ["x", "y"], "CONCAT": [f"src{i}" for i in range(8)],
            "INSERT_SLICE": ["dest", "src"], "MOVEMENT": ["src"], "WINDOW_SUM": ["src"], "FILL": []}[opcode]
    ops = dict(zip(keys, srcs))
    tensors["Y"] = _t(dst_shape, "output", dst_dtype)
    ops["dst"] = "Y"
    return _cb(tensors, [{"opcode": opcode, "operands": ops, "attributes": attrs}])


def cases(representative: bool = False) -> dict[str, dict]:
    B, H, W, C = (1, 28, 28, 64) if representative else (1, 8, 8, 32)
    N = (B, H, W, C)
    cs = {}
    # AFFINE_MAP (add/sub.Tensor, add/sub.Scalar, mul/div.Scalar, neg, relu, _to_copy)
    cs["affine_add_tensor"] = _one("AFFINE_MAP", {"A": N, "Bt": N}, N, dict(
        weights=[304, -245], constant=0, activation="none", readout_scale=0.0021, output_dtype="i8"))
    cs["affine_add_bcast"] = _one("AFFINE_MAP", {"A": N, "Bt": (W, C)}, N, dict(
        weights=[7, 5], constant=0, activation="none", readout_scale=0.071, output_dtype="i8"))
    cs["affine_add_scalar"] = _one("AFFINE_MAP", {"A": N}, N, dict(
        weights=[93], constant=-1234, activation="none", readout_scale=0.0107, output_dtype="i8"))
    cs["affine_mul_scalar"] = _one("AFFINE_MAP", {"A": N}, N, dict(
        weights=[1], constant=0, activation="none", readout_scale=1.37, output_dtype="i8"))
    cs["affine_neg"] = _one("AFFINE_MAP", {"A": N}, N, dict(
        weights=[1], constant=0, activation="none", readout_scale=-1.0, output_dtype="i8", quant_min=-128))
    cs["affine_relu"] = _one("AFFINE_MAP", {"A": N}, N, dict(
        weights=[1], constant=0, activation="relu", readout_scale=0.83, output_dtype="i8"))
    cs["affine_to_copy"] = _one("AFFINE_MAP", {"A": N}, N, dict(
        weights=[1], constant=0, activation="none", readout_scale=1.0, output_dtype="i32"), "i32")
    cs["affine_i32_lincomb"] = _one("AFFINE_MAP", {"A": N, "Bt": N}, N, dict(
        weights=[3, -2], constant=17, activation="none", readout_scale=1.0, output_dtype="i32"), "i32")
    # MOVEMENT forms
    cs["move_identity_widen"] = _one("MOVEMENT", {"A": N}, N, dict(output_dtype="i32", semantic="mvin_mvout"), "i32")
    cs["move_identity_scale"] = _one("MOVEMENT", {"A": N}, N, dict(form="identity", scale=0.77, output_dtype="i8"))
    cs["move_permute_nchw"] = _one("MOVEMENT", {"A": N}, (B, C, H, W), dict(form="permute", perm=[0, 3, 1, 2],
                                                                             output_dtype="i8"))
    cs["move_permute_inner_kept"] = _one("MOVEMENT", {"A": N}, (W, B, H, C), dict(form="permute", perm=[2, 0, 1, 3],
                                                                                  output_dtype="i8"))
    cs["move_slice"] = _one("MOVEMENT", {"A": N}, (B, H // 2, W // 2 - 1, C // 2), dict(
        form="slice", offsets=[0, 1, 2, 8], sizes=[B, H // 2, W // 2 - 1, C // 2], strides=[1, 2, 2, 1],
        output_dtype="i8"))
    cs["move_pad_constant"] = _one("MOVEMENT", {"A": N}, (B, H + 3, W + 2, C), dict(
        form="pad", low=[0, 1, 1, 0], high=[0, 2, 1, 0], mode="constant", value_code=-7, scale=0.9,
        output_dtype="i8"))
    cs["move_pad_reflect"] = _one("MOVEMENT", {"A": N}, (B, H + 2, W + 4, C), dict(
        form="pad", low=[0, 1, 2, 0], high=[0, 1, 2, 0], mode="reflect", output_dtype="i8"))
    cs["move_pad_replicate"] = _one("MOVEMENT", {"A": N}, (B, H + 1, W + 3, C), dict(
        form="pad", low=[0, 0, 1, 0], high=[0, 1, 2, 0], mode="replicate", output_dtype="i8"))
    cs["move_broadcast"] = _one("MOVEMENT", {"A": (1, W, C)}, (4, W, C), dict(form="broadcast", output_dtype="i8"))
    cs["move_repeat"] = _one("MOVEMENT", {"A": (2, W, C)}, (4, 2 * W, C), dict(form="repeat", repeats=[2, 2, 1],
                                                                            output_dtype="i8"))
    # CONCAT / INSERT_SLICE / FILL
    cs["concat_scales"] = _one("CONCAT", {"A": N, "Bt": (B, H, W, C // 2), "Ct": (B, H, W, 16)},
                               (B, H, W, C + C // 2 + 16), dict(axis=3, output_dtype="i8", scales=[1.0, 0.6, 1.9]))
    cs["concat_axis1"] = _one("CONCAT", {"A": N, "Bt": N}, (B, 2 * H, W, C), dict(axis=1, output_dtype="i8"))
    cs["insert_slice"] = _one("INSERT_SLICE", {"D": N, "S": (B, H // 2, W, C)}, N, dict(
        offsets=[0, 1, 0, 0], strides=[1, 2, 1, 1], output_dtype="i8"))
    cs["fill"] = _one("FILL", {}, N, dict(value_code=-77, output_dtype="i8"))
    cs["fill_zero"] = _one("FILL", {}, N, dict(value_code=0, output_dtype="i8"))
    # WINDOW_SUM, every mode
    cs["ws_avg_pool2d"] = _one("WINDOW_SUM", {"A": N}, (B, H // 2, W // 2, C), dict(
        mode="window", kernel=[3, 3], stride=[2, 2], padding=[1, 1], count_include_pad=False,
        readout_scale=0.93, output_dtype="i8"))
    cs["ws_avg_pool2d_ceil_div"] = _one("WINDOW_SUM", {"A": N}, (B, (H - 2 + 1) // 2 + 1, (W - 2 + 1) // 2 + 1, C), dict(
        mode="window", kernel=[2, 2], stride=[2, 2], padding=[0, 0], count_include_pad=True, ceil_mode=True,
        divisor_override=3, readout_scale=0.5, output_dtype="i8")) if (H % 2) else None
    cs["ws_avg_pool1d_nchw"] = _one("WINDOW_SUM", {"A": (B, C, 4 * W)}, (B, C, 2 * W), dict(
        mode="window", kernel=[2], stride=[2], padding=[0], count_include_pad=True, layout="nchw",
        readout_scale=1.1, output_dtype="i8"))
    cs["ws_adaptive"] = _one("WINDOW_SUM", {"A": N}, (B, 3, 5, C), dict(mode="adaptive", readout_scale=0.8,
                                                                          output_dtype="i8"))
    cs["ws_mean_hw"] = _one("WINDOW_SUM", {"A": N}, (B, 1, 1, C), dict(mode="reduce", dims=[1, 2], average=True,
                                                                         readout_scale=1.3, output_dtype="i8"))
    cs["ws_mean_last"] = _one("WINDOW_SUM", {"A": N}, (B, H, W), dict(mode="reduce", dims=[-1], average=True,
                                                                        readout_scale=0.9, output_dtype="i8"))
    cs["ws_sum_noncontig"] = _one("WINDOW_SUM", {"A": (2, 4, 6, 16)}, (4, 16), dict(
        mode="reduce", dims=[0, 2], average=False, readout_scale=0.05, output_dtype="i8"))
    cs["ws_avg_pool2d_backward"] = _one("WINDOW_SUM", {"A": (B, H // 2, W // 2, C)}, N, dict(
        mode="window_transpose", kernel=[3, 3], stride=[2, 2], padding=[1, 1], count_include_pad=False,
        readout_scale=0.7, output_dtype="i8"))
    cs["ws_adaptive_backward"] = _one("WINDOW_SUM", {"A": (B, 3, 3, C)}, N, dict(
        mode="adaptive_transpose", readout_scale=0.6, output_dtype="i8"))
    k = 3
    Lb = (H - k + 2) // 1 + 1
    cs["ws_fold_col2im"] = _one("WINDOW_SUM", {"A": (B, 4 * k * k, Lb * Lb)}, (B, 4, H, H), dict(
        mode="fold", kernel=[k, k], stride=[1, 1], padding=[1, 1], dilation=[1, 1], readout_scale=0.4,
        output_dtype="i8"))
    # a chained program: AFFINE_MAP -> FENCE -> WINDOW_SUM -> FENCE -> MOVEMENT
    tensors = {"A": _t(N), "Bt": _t(N), "Z": _t(N, "output"), "P": _t((B, H // 2, W // 2, C), "output"),
               "Q": _t((B, C, H // 2, W // 2), "output")}
    cs["chain_fenced"] = _cb(tensors, [
        {"opcode": "AFFINE_MAP", "operands": {"x": "A", "y": "Bt", "dst": "Z"},
         "attributes": dict(weights=[11, 9], constant=0, activation="relu", readout_scale=0.05, output_dtype="i8")},
        {"opcode": "FENCE", "operands": {}},
        {"opcode": "WINDOW_SUM", "operands": {"src": "Z", "dst": "P"},
         "attributes": dict(mode="window", kernel=[2, 2], stride=[2, 2], padding=[0, 0], count_include_pad=True,
                            readout_scale=1.0, output_dtype="i8")},
        {"opcode": "FENCE", "operands": {}},
        {"opcode": "MOVEMENT", "operands": {"src": "P", "dst": "Q"},
         "attributes": dict(form="permute", perm=[0, 3, 1, 2], output_dtype="i8")}])
    return {k_: v for k_, v in cs.items() if v is not None}


def stimulus(cb: dict, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    produced = {c["operands"].get("dst") for c in cb["commands"]}
    out = {}
    for name, t in cb["tensors"].items():
        if name in produced:
            continue
        out[name] = rng.integers(-128, 128, size=t["shape"]).astype(np.int8)
    return out
