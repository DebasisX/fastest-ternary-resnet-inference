#!/usr/bin/env python3
"""
Train ternary ResNet-50 for the standalone C++ ternary inference engine.

Outputs (by default under ./weights):
  - resnet50_ternary_float_best.pth     (best validation float checkpoint)
  - resnet50_ternary_engine_ready.pth   (baked ternary checkpoint, export-safe)
  - activation_scales.json              (per-layer input scales for export)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
import torchvision.transforms as transforms


def compute_threshold(weight: torch.Tensor, factor: float = 0.05) -> torch.Tensor:
    return factor * weight.abs().mean()


def ternarize(weight: torch.Tensor, factor: float = 0.05) -> torch.Tensor:
    thr = compute_threshold(weight, factor)
    return weight.sign() * (weight.abs() >= thr).float()


class TernaryConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        bias: bool = False,
        threshold_factor: float = 0.05,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.threshold_factor = threshold_factor
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size))
        self.bias = None if not bias else nn.Parameter(torch.zeros(out_channels))
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            w_t = ternarize(self.weight, self.threshold_factor)
            mask = (w_t != 0).float()
            alpha = self.weight.abs().sum() / (mask.sum() + 1e-8)
        else:
            w_t = self.weight
            alpha = getattr(self, "alpha_scale", torch.tensor(1.0, device=x.device)).to(x.device)
        return F.conv2d(x, w_t, self.bias, self.stride, self.padding, groups=self.groups) * alpha


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inp: int, planes: int, stride: int = 1, downsample=None, conv_cls=TernaryConv2d):
        super().__init__()
        self.conv1 = conv_cls(inp, planes, 1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv_cls(planes, planes, 3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv_cls(planes, planes * 4, 1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet50(nn.Module):
    def __init__(self, num_classes: int = 10, conv_class=TernaryConv2d):
        super().__init__()
        self.in_planes = 64
        self.conv_class = conv_class
        self.conv1 = conv_class(3, 64, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, TernaryConv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes: int, blocks: int, stride: int):
        downsample = None
        if stride != 1 or self.in_planes != planes * 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * 4, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * 4),
            )
        layers = [Bottleneck(self.in_planes, planes, stride, downsample, conv_cls=self.conv_class)]
        self.in_planes = planes * 4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.in_planes, planes, conv_cls=self.conv_class))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return self.fc(torch.flatten(x, 1))


def bake_ternary_weights(model: nn.Module):
    for _, m in model.named_modules():
        if isinstance(m, TernaryConv2d):
            with torch.no_grad():
                w = m.weight.data
                thr = compute_threshold(w, m.threshold_factor)
                w_t = w.sign() * (w.abs() >= thr).float()
                mask = (w_t != 0).float()
                alpha = w.abs().sum() / (mask.sum() + 1e-8)
                m.weight.data.copy_(w_t)
                if hasattr(m, "alpha_scale"):
                    m.alpha_scale.copy_(alpha.unsqueeze(0))
                else:
                    m.register_buffer("alpha_scale", alpha.unsqueeze(0))


@torch.no_grad()
def recalibrate_bn(model: nn.Module, loader, device: torch.device, n_batches: int = 50):
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None
    for i, (imgs, _) in enumerate(loader):
        if i >= n_batches:
            break
        model(imgs.to(device))
    model.eval()


@torch.no_grad()
def calibrate_activation_scales(model: nn.Module, loader, device: torch.device, n_batches: int = 20):
    model.eval()
    scales = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, inp, out):
            absmax = inp[0].abs().amax().item()
            scales[name] = max(scales.get(name, 0.0), absmax)

        return hook_fn

    for name, m in model.named_modules():
        if isinstance(m, TernaryConv2d):
            hooks.append(m.register_forward_hook(make_hook(name)))

    for i, (imgs, _) in enumerate(loader):
        if i >= n_batches:
            break
        model(imgs.to(device))

    for h in hooks:
        h.remove()

    return {n: v / 127.0 for n, v in scales.items()}


@torch.no_grad()
def eval_top1(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        pred = model(imgs).argmax(1)
        correct += pred.eq(labels).sum().item()
        total += imgs.size(0)
    return 100.0 * correct / max(1, total)


def main():
    parser = argparse.ArgumentParser(description="Train and bake ternary ResNet-50 for C++ engine")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--weights-dir", type=str, default=None)
    parser.add_argument("--bn-recalib-batches", type=int, default=50)
    parser.add_argument("--act-calib-batches", type=int, default=20)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    here = Path(__file__).resolve().parent
    data_root = Path(args.data_root) if args.data_root else (here / "data")
    weights_dir = Path(args.weights_dir) if args.weights_dir else (here / "weights")
    weights_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    norm = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        norm,
    ])
    val_tf = transforms.Compose([transforms.ToTensor(), norm])

    train_ds = torchvision.datasets.CIFAR10(root=str(data_root), train=True, download=True, transform=train_tf)
    val_ds = torchvision.datasets.CIFAR10(root=str(data_root), train=False, download=True, transform=val_tf)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = ResNet50(num_classes=10, conv_class=TernaryConv2d).to(device)
    print(f"Device: {device}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_acc = 0.0
    float_best_path = weights_dir / "resnet50_ternary_float_best.pth"
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        total_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            correct += logits.argmax(1).eq(labels).sum().item()
            total += imgs.size(0)

        scheduler.step()

        val_acc = eval_top1(model, val_loader, device)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), float_best_path)

        print(
            f"Epoch {epoch}/{args.epochs} "
            f"loss={total_loss / max(1, total):.4f} "
            f"train_acc={100.0 * correct / max(1, total):.2f}% "
            f"val_acc={val_acc:.2f}% best={best_acc:.2f}% "
            f"[{time.time() - t0:.1f}s]"
        )

    print(f"Loading best float checkpoint: {float_best_path}")
    model.load_state_dict(torch.load(float_best_path, map_location=device))

    print("Baking ternary weights and alpha scales...")
    bake_ternary_weights(model)

    print(f"Recalibrating BN with {args.bn_recalib_batches} batches...")
    recalibrate_bn(model, val_loader, device, n_batches=args.bn_recalib_batches)

    engine_ready_path = weights_dir / "resnet50_ternary_engine_ready.pth"
    torch.save(model.state_dict(), engine_ready_path)
    print(f"Saved engine-ready checkpoint: {engine_ready_path}")

    final_acc = eval_top1(model, val_loader, device)
    print(f"Post-bake validation accuracy: {final_acc:.2f}%")

    print(f"Calibrating activation scales ({args.act_calib_batches} batches)...")
    act_scales = calibrate_activation_scales(model, val_loader, device, n_batches=args.act_calib_batches)
    act_scale_path = weights_dir / "activation_scales.json"
    with open(act_scale_path, "w", encoding="utf-8") as f:
        json.dump(act_scales, f, indent=2)
    print(f"Saved activation scales: {act_scale_path} ({len(act_scales)} layers)")


if __name__ == "__main__":
    main()
