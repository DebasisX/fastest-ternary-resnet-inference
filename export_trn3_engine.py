#!/usr/bin/env python3
"""
Robust lean-TRN3 exporter for the standalone ternary inference engine.

This script avoids common export pitfalls by:
    1) Loading checkpoints with alpha_scale buffer registration.
    2) Verifying ternary conv weights are truly in {-1,0,+1}.
    3) Optionally baking weights if needed.
    4) Writing only the payload required by the fast C++ path.
    5) Validating output header/magic and file size after export.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from train_ternary_engine import ResNet50, TernaryConv2d, bake_ternary_weights  # noqa: E402


TERNARY_ENCODE = {-1: 0b01, 0: 0b00, 1: 0b11}


def weights_to_int2(w: np.ndarray) -> bytes:
    n = len(w)
    pad = (4 - n % 4) % 4
    if pad:
        w = np.concatenate([w, np.zeros(pad, dtype=np.int8)])
    out = bytearray(len(w) // 4)
    for i in range(len(out)):
        b = 0
        for j in range(4):
            b |= TERNARY_ENCODE.get(int(w[i * 4 + j]), 0) << (j * 2)
        out[i] = b & 0xFF
    return bytes(out)
def write_i32(f, v):
    f.write(struct.pack("<i", int(v)))


def write_f32(f, v):
    f.write(struct.pack("<f", float(v)))


def write_f32_arr(f, a):
    f.write(np.asarray(a, dtype=np.float32).flatten().tobytes())


def write_str(f, s: str):
    b = s.encode("utf-8")
    write_i32(f, len(b))
    f.write(b)


def load_model_with_alpha(ckpt_path: Path) -> nn.Module:
    model = ResNet50(num_classes=10, conv_class=TernaryConv2d)
    sd = torch.load(str(ckpt_path), map_location="cpu")

    # Critical: register alpha_scale buffers before loading state_dict.
    for name, m in model.named_modules():
        if isinstance(m, TernaryConv2d):
            key = f"{name}.alpha_scale"
            if key in sd:
                m.register_buffer("alpha_scale", sd[key])

    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def verify_ternary_weights(model: nn.Module):
    bad = []
    missing_alpha = []
    for name, m in model.named_modules():
        if not isinstance(m, TernaryConv2d):
            continue
        w = m.weight.detach().cpu().numpy()
        uq = np.unique(np.round(w).astype(np.int8))
        if not np.all(np.isin(uq, np.array([-1, 0, 1], dtype=np.int8))):
            bad.append((name, uq.tolist()))
        if not hasattr(m, "alpha_scale"):
            missing_alpha.append(name)
    return bad, missing_alpha


def export_trn3(model: nn.Module, out_path: Path, act_scales: dict[str, float]):
    convs, bns, fc = [], [], None
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, TernaryConv2d)):
            convs.append((name, m))
        elif isinstance(m, nn.BatchNorm2d):
            bns.append((name, m))
        elif isinstance(m, nn.Linear):
            fc = m

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(b"TRN3")
        write_i32(f, 4)
        write_i32(f, len(convs))
        write_i32(f, len(bns))
        write_i32(f, 1 if fc else 0)

        for name, m in convs:
            w = m.weight.data.cpu().float().numpy()
            is_ternary = isinstance(m, TernaryConv2d)

            write_str(f, name)
            write_i32(f, w.shape[0])
            write_i32(f, w.shape[1])
            write_i32(f, w.shape[2])
            write_i32(f, w.shape[3])

            s = m.stride if isinstance(m.stride, int) else m.stride[0]
            p = m.padding if isinstance(m.padding, int) else m.padding[0]
            write_i32(f, s)
            write_i32(f, p)
            write_i32(f, m.groups)
            write_i32(f, 1 if is_ternary else 0)

            if is_ternary:
                w_flat = np.round(np.clip(w.flatten(), -1, 1)).astype(np.int8)
                alpha = 1.0
                if hasattr(m, "alpha_scale"):
                    alpha = float(m.alpha_scale.detach().cpu().item())

                write_f32(f, alpha)
                write_f32(f, float(act_scales.get(name, 0.0)))
                write_i32(f, len(w_flat))

                d = weights_to_int2(w_flat)
                write_i32(f, len(d))
                f.write(d)

            else:
                write_f32(f, 0.0)
                write_f32(f, 0.0)
                d = w.flatten().astype(np.float32).tobytes()
                write_i32(f, w.size)
                write_i32(f, len(d))
                f.write(d)

        for name, m in bns:
            write_str(f, name)
            write_i32(f, m.num_features)
            write_f32_arr(f, m.weight.data.cpu())
            write_f32_arr(f, m.bias.data.cpu())
            write_f32_arr(f, m.running_mean.cpu())
            write_f32_arr(f, m.running_var.cpu())
            write_f32(f, m.eps)

        if fc:
            write_i32(f, fc.in_features)
            write_i32(f, fc.out_features)
            write_f32_arr(f, fc.weight.data.cpu())
            write_i32(f, 1 if fc.bias is not None else 0)
            if fc.bias is not None:
                write_f32_arr(f, fc.bias.data.cpu())


def validate_export(path: Path):
    size = path.stat().st_size
    if size < 1_000_000:
        raise RuntimeError(f"Exported file is too small ({size} bytes). Likely invalid export.")
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"TRN3":
            raise RuntimeError("Invalid magic in exported file (expected TRN3).")
        version = struct.unpack("<i", f.read(4))[0]
        n_conv = struct.unpack("<i", f.read(4))[0]
        n_bn = struct.unpack("<i", f.read(4))[0]
        has_fc = struct.unpack("<i", f.read(4))[0]
    if version != 4:
        raise RuntimeError(f"Unexpected TRN3 version {version}; expected lean format v4.")
    return version, n_conv, n_bn, has_fc, size


def main():
    parser = argparse.ArgumentParser(description="Export robust TRN3 binary for C++ ternary engine")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to .pth checkpoint")
    parser.add_argument("--act-scales", type=str, default=None, help="Path to activation_scales.json")
    parser.add_argument("--out", type=str, default=None, help="Output TRN3 .bin path")
    parser.add_argument("--bake-if-needed", action="store_true", help="Bake weights if checkpoint is not ternary")
    args = parser.parse_args()

    weights_dir = THIS_DIR / "weights"
    default_ckpt = weights_dir / "resnet50_ternary_engine_ready.pth"
    fallback_ckpt = weights_dir / "resnet50_ternary.pth"

    ckpt_path = Path(args.ckpt) if args.ckpt else (default_ckpt if default_ckpt.exists() else fallback_ckpt)
    act_scale_path = Path(args.act_scales) if args.act_scales else (weights_dir / "activation_scales.json")
    out_path = Path(args.out) if args.out else (weights_dir / "resnet50_ternary_packed.bin")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")
    model = load_model_with_alpha(ckpt_path)

    bad, missing_alpha = verify_ternary_weights(model)
    if bad and not args.bake_if_needed:
        names = ", ".join(n for n, _ in bad[:8])
        raise RuntimeError(
            "Checkpoint appears non-ternary for some conv layers. "
            "Use an engine-ready checkpoint or rerun with --bake-if-needed. "
            f"Examples: {names}"
        )
    if bad and args.bake_if_needed:
        print("Baking checkpoint weights to ternary form...")
        bake_ternary_weights(model)
        bad, missing_alpha = verify_ternary_weights(model)
        if bad:
            raise RuntimeError("Weights are still non-ternary after baking.")

    if missing_alpha:
        print(f"Warning: {len(missing_alpha)} ternary layers missing alpha_scale; baked alpha values will be used.")

    act_scales = {}
    if act_scale_path.exists():
        with open(act_scale_path, "r", encoding="utf-8") as f:
            act_scales = json.load(f)
        print(f"Loaded activation scales: {act_scale_path} ({len(act_scales)} layers)")
    else:
        print("Warning: activation scales not found; writing zeros for act_scale fields.")

    export_trn3(model, out_path, act_scales)
    version, n_conv, n_bn, has_fc, size = validate_export(out_path)
    print(f"Exported: {out_path}")
    print(f"Header: magic=TRN3 version={version} conv={n_conv} bn={n_bn} fc={has_fc}")
    print(f"Size: {size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
