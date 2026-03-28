# Ternary ResNet-50 Inference Engine — Documentation

## Table of Contents
1. [Overview](#overview)
2. [File Structure](#file-structure)
3. [Quick Start](#quick-start)
4. [API Reference](#api-reference)
5. [Python Weight Export](#python-weight-export)
   - [TRN3 Binary Format](#trn3-binary-format)
   - [How to Export](#how-to-export)
6. [Pipeline Internals](#pipeline-internals)
   - [Calibration Phase](#calibration-phase)
   - [Inference Phase](#inference-phase)
   - [NHWC Weight Layout](#nhwc-weight-layout)
   - [NR=4 Panel Packing](#nr4-panel-packing)
   - [FusedQuantParams Math](#fusedquantparams-math)
   - [GEBP Microkernel MR=3 NR=4](#gebp-microkernel-mr3-nr4)
7. [Build Instructions](#build-instructions)
8. [Performance Notes](#performance-notes)
9. [Hardware Requirements](#hardware-requirements)

---

## Overview

This engine provides production-ready, fully-quantized inference for a ternary
ResNet-50 trained on CIFAR-10.  All weights are quantized to ternary
{−1, 0, +1} during training, then packed to int8 for SIMD-accelerated GEMM.

**Key properties:**
- Single inference: about 3 ms on the target i7-13700HX with P-core threading
- Lean standalone `TRN3` v4 format with int2 ternary payload only
- Accuracy depends on the checkpoint used; correctness is established by parity
  between PyTorch and this C++ engine for the same engine-ready ternary model
- No dynamic allocation on the hot path — everything is pre-allocated at calibration time
- Only AVX2 + AVX-VNNI are required; no AVX-512, no special libraries

---

## File Structure

```
resnet_inference_engine/
├── engine.h            — All public types and TernaryCNN class declaration
├── engine.cpp          — Full implementation (calibration + Q2Q inference)
├── inference_main.cpp  — CIFAR-10 batch inference demo / benchmark
├── CMakeLists.txt      — Build system
└── DOCUMENTATION.md    — This file
```

The core engine has no dependencies beyond the C++17 standard library, OpenMP,
and POSIX threads (for affinity). The standalone workflow in this folder also
includes local `data/`, `weights/`, training, export, and flow documentation.

---

## Quick Start

```bash
# Build
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Export weights from the standalone folder
cd ..
python3 export_trn3_engine.py --bake-if-needed

# Run inference on the standalone raw CIFAR binary (7 threads)
cd build
./inference_demo ../weights/resnet50_ternary_packed.bin \
                 ../data/test_batch.bin \
                 7
```

Expected output:
```
Loading model from ../weights/resnet50_ternary_packed.bin
Loading TRN3 v4: 53 conv, 53 bn, fc=1
Loaded 53 conv layers (49 ternary)
Calibrating fused Q2Q pipeline...
Fused Q2Q pipeline calibrated: 53 layers, 16 blocks
Benchmarking (100 runs, 7 threads)...
  Latency (ms):  min=...  p50=...  p95=...  mean=...
Running inference on 10000 images...
  Accuracy:       ...
  Avg latency/img: ...
  Throughput:      ...
```

---

## API Reference

### `bool TernaryCNN::load(const std::string& bin_path)`

Parses the TRN3 binary file.  Performs:
- Magic check ("TRN3")
- Conv layer parsing (ternary or FP32 downsample)
- int2→i8 unpacking for all ternary layers
- BN parameter folding: A[oc] = gamma/sqrt(var+eps), B[oc] = beta − A*mean
- FC layer parsing

Returns `true` on success.  Sets `loaded = true`.

**After `load`, the engine's memory footprint is ~35 MB:**
- ~4 MB: 49 ternary layers × ~80K int8 weights each
- ~1 MB: 4 FP32 downsample layers
- The int2_packed buffers are freed immediately after unpacking (saves ~4 MB)

---

### `void TernaryCNN::calibrate_fused(const f32* sample_chw, int C, int H, int W, int n_calib = 1)`

One-time setup.  Must be called before `infer_fused`.

**Input:**
- `sample_chw`: CIFAR-10-normalised float32 image in CHW format (shape [3, 32, 32])
- `C, H, W`: channel, height, width (3, 32, 32 for CIFAR-10)

**What it does:**

1. **FP32 calibration pass** — runs float inference on `n_calib` images using
  `conv_calib()` and records robust per-layer output min/max.

2. **FusedQuantParams computation** — for each conv layer, precomputes
   `oc_mul[oc]` and `oc_add[oc]` (see [FusedQuantParams Math](#fusedquantparams-math)).

3. **FP32 downsample weight packing** — quantizes the 4 identity-shortcut conv
   layers to symmetric int8 using max-abs scaling.

4. **NHWC weight repack** — permutes 3×3 weights from `[IC][kH][kW]` to
   `[kH][kW][IC]` so the inner GEBP loop reads contiguous IC-vectors.

5. **NR=4 panel packing** — packs into `[OC/4][K_pad/32][4][32]` layout
   (see [NR=4 Panel Packing](#nr4-panel-packing)).

6. **Scratch buffer allocation** — allocates 6 QTensor buffers (q_a, q_b, q_t1,
   q_t2, q_t3, q_ds) sized to the largest intermediate tensor.

**Calibration sample:** The standalone demo defaults to multiple calibration
images. Using more than one image is recommended for stable quantization ranges.

---

### `std::vector<f32> TernaryCNN::infer_fused(const f32* img_chw, int C, int H, int W)`

Runs fully-quantized inference.

**Input:** Same format as `calibrate_fused` — CIFAR-10-normalised CHW float32.

**Output:** 10-element vector of raw logits (no softmax).  Argmax gives the
predicted class.

**Data flow:**
```
CHW float32 ──quantize──> NHWC u8
    → conv_q2q (stem, stride=1, 3×3, 3→64)
    → 16× bottleneck_q2q:
        [3 conv_q2q + q_add_relu]
    → GAP (AVX2 dequantize + accumulate)
    → FC (AVX2 FP32 GEMV)
    → logits [10]
```

No float values are produced between the initial quantize step and the final
GAP step.  All intermediate activations are u8 NHWC tensors.

---

### `void TernaryCNN::set_num_threads(int n)`

Sets the OpenMP thread count for conv_q2q.  Calling this before `infer_fused`
takes effect on the next inference.  Default: 8.

Optimal on i7-13700HX: `n=7` (uses 7 P-cores, avoids E-core scheduling jitter).

---

The standalone demo performs CIFAR-10 normalisation in `inference_main.cpp`
using the dataset's native CHW binary layout. There is no separate HWC
normalisation helper in the engine API.

---

### Profiling

```cpp
engine.set_profiling(true);
auto logits = engine.infer_fused(img, 3, 32, 32);
auto stats = engine.get_last_profile();
printf("Total: %.2f ms  Conv: %.2f ms  GAP: %.2f us  FC: %.2f us\n",
       stats.total_us/1000, stats.conv_total_us/1000,
       stats.gap_us, stats.fc_us);
```

Profiling is optional and disabled by default. The fast inference path is
unchanged unless the C++ API caller enables it with `set_profiling(true)`.

---

## Python Weight Export

### TRN3 Binary Format

All multi-byte integers are **little-endian**.  All floats are **IEEE 754 f32**.

```
┌─────────────────────────────────────────────────────────┐
│  HEADER                                                 │
│  [4 bytes]  magic = "TRN3"                             │
│  [4 bytes]  version (i32) = 4                          │
│  [4 bytes]  n_conv  (i32)                              │
│  [4 bytes]  n_bn    (i32)  — always equals n_conv      │
│  [4 bytes]  has_fc  (i32)  — 1 if FC layer present     │
├─────────────────────────────────────────────────────────┤
│  CONV LAYERS  (n_conv records)                         │
│                                                         │
│  Each record begins with:                               │
│    [4+len] name string: [4b len_bytes] + [len_bytes UTF8]│
│    [4 bytes]  out_channels  (i32)                      │
│    [4 bytes]  in_channels   (i32)                      │
│    [4 bytes]  kernel_H      (i32)                      │
│    [4 bytes]  kernel_W      (i32)                      │
│    [4 bytes]  stride        (i32)                      │
│    [4 bytes]  padding       (i32)                      │
│    [4 bytes]  groups        (i32) — always 1           │
│    [4 bytes]  is_ternary    (i32) — 1 or 0             │
│                                                         │
│  If is_ternary == 1 (ternary conv):                     │
│    [4 bytes]  alpha         (f32) — weight scale       │
│    [4 bytes]  act_scale     (f32) — ignored by engine  │
│    [4 bytes]  n_weights     (i32)                      │
│    [4 bytes]  int2_bytes    (i32) — = ceil(n_weights/4)│
│    [int2_bytes] int2_packed  — 4 weights per byte:     │
│                  bits[1:0] = weight 0                   │
│                  bits[3:2] = weight 1                   │
│                  bits[5:4] = weight 2                   │
│                  bits[7:6] = weight 3                   │
│                  encoding: -1→0b01  0→0b00  +1→0b11    │
│                                                         │
│  If is_ternary == 0 (FP32 downsample conv):             │
│    [4 bytes]  alpha         (f32) — ignored            │
│    [4 bytes]  act_scale     (f32) — ignored            │
│    [4 bytes]  n_weights     (i32)                      │
│    [4 bytes]  data_bytes    (i32) — = n_weights * 4    │
│    [data_bytes] raw float32 weights (NCHW order)        │
├─────────────────────────────────────────────────────────┤
│  BN LAYERS  (n_bn records, parallel to conv layers)    │
│                                                         │
│  Each record:                                           │
│    [4+len] name string                                  │
│    [4 bytes]  num_features (i32)                        │
│    [num_features * 4] gamma   (f32 array)              │
│    [num_features * 4] beta    (f32 array)              │
│    [num_features * 4] running_mean (f32 array)         │
│    [num_features * 4] running_var  (f32 array)         │
│    [4 bytes]  eps (f32)                                 │
├─────────────────────────────────────────────────────────┤
│  FC LAYER  (if has_fc == 1)                            │
│    [4 bytes]  in_features  (i32)                        │
│    [4 bytes]  out_features (i32)                        │
│    [in_features * out_features * 4] weight (f32, row-major)│
│    [4 bytes]  has_bias (i32)                            │
│    [out_features * 4] bias (f32, if has_bias == 1)     │
└─────────────────────────────────────────────────────────┘
```

#### int2 Encoding Detail

The `int2_packed` buffer stores 4 ternary weights per byte.  Weight `i` is at:
```
byte  = i // 4
shift = (i %  4) * 2        # 0, 2, 4, or 6
code  = (byte_val >> shift) & 0b11
value = {0b00: 0, 0b01: -1, 0b10: 0, 0b11: +1}[code]
```

This packing is produced by `export_trn3_engine.py::weights_to_int2()`:
```python
def weights_to_int2(weights):
    # Map: -1 → 1 (0b01),  0 → 0 (0b00),  +1 → 3 (0b11)
    codes = np.where(weights == -1, 1, np.where(weights == 1, 3, 0)).astype(np.uint8)
    padded = np.zeros(((len(codes) + 3) // 4) * 4, dtype=np.uint8)
    padded[:len(codes)] = codes
    # Pack 4 codes per byte, LSB-first
    packed = (padded[0::4]       |
              padded[1::4] << 2  |
              padded[2::4] << 4  |
              padded[3::4] << 6)
    return packed.tobytes()
```

#### Weight Ordering

Conv weights are stored in **NCHW** PyTorch order:
```
weight[out_channel][in_channel][kH][kW]
```
Linear (flat) index = `oc * (IC * kH * kW) + ic * (kH * kW) + dh * kW + dw`

The engine unpacks them in this order, yielding `unpacked[oc * K + k]`.

---

### How to Export

From this standalone folder:

```python
python3 export_trn3_engine.py --bake-if-needed
```

The exported file should be about 16-17 MB for the current lean standalone format.

---

## Pipeline Internals

### Calibration Phase

Calibration runs one complete FP32 inference (using `conv_calib()`, not the
GEBP kernel) to measure the actual output range at every layer boundary.
These ranges are used to choose per-layer quantization scales.

```
Input (float CHW)
  ↓ conv_calib (stem) → record min/max → calibRange[0]
  ↓ conv_calib (block 0, c1) → calibRange[1]
  ↓ conv_calib (block 0, c2) → calibRange[2]
  ↓ conv_calib (block 0, c3) → calibRange[3]
    + conv_calib (block 0, ds) → calibRange[4]
    → residual add + relu → record block_out_scale[0]
  ...
```

**Why not use the training statistics?**
Training-time BN statistics are per-channel mean/variance of activations.
For quantization we need the actual output range under our specific test
distribution.  One image is usually sufficient because ResNet-50's activations
are tightly bounded by BN + ReLU.

---

### Inference Phase

All 53 conv layers run as `conv_q2q()`, which is a fused kernel that:
1. Computes INT8 GEMM (u8 activations × i8 weights)
2. Applies `oc_mul` / `oc_add` per output channel (requantize + BN + optional ReLU)
3. Writes u8 output in NHWC layout

The only float operations in the inference hot path are:
- Initial CHW f32 → NHWC u8 (input quantize, ~1 µs)
- Per-output-channel requantize scalar inside `quantize_store_nr4` (fused into conv)
- `q_add_relu` (residual addition, AVX2 f32, ~5 µs per block)
- GAP dequantize (~50 µs)
- FC GEMV (~100 µs)

---

### NHWC Weight Layout

All intermediate activation tensors use **NHWC** layout:
```
data[h * W * C + w * C + c]
```

This is chosen because `conv_q2q`'s implicit im2col reads all `IC` channels
for a given `(h, w)` position as a contiguous byte vector.  In NHWC, this
is `data + (h * W + w) * IC` — a direct pointer, no stride calculation.

For 3×3 convolutions, the weights must be permuted to match:
- Original NCHW im2col order: `w[oc][ic][dh][dw]` → K-axis is `[IC][kH][kW]`
- Implicit NHWC im2col reads `IC` channels per `(dh, dw)` step → K-axis must be `[kH][kW][IC]`

`calibrate_fused()` performs this permutation once via `nhwc_w_[li]`.

```
NCHW→NHWC permutation:
  for oc in [0, OC):
    for ic in [0, IC):
      for dh in [0, kH):
        for dw in [0, kW):
          nhwc[oc][dh * kW * IC + dw * IC + ic] =
              nchw[oc][ic * kH * kW + dh * kW + dw]
```

---

### NR=4 Panel Packing

The GEBP microkernel processes `NR=4` output channels simultaneously.  To keep
all 4 weight vectors for a given k-step contiguous in memory, weights are
repacked into:

```
panel_w[OC_tile][k_chunk][NR][32]
  where OC_tile = OC / NR
        k_chunk = K_pad / 32
```

Memory layout (for one k-chunk of one OC tile):
```
Byte offset:   0         32        64        96       128
Channel:    [ OC+0 ][  OC+1 ][  OC+2 ][  OC+3 ]
            <- 32 weights -> <- 32 -> <- 32 -> <- 32 ->
            \_________________________128 bytes_______/
```

This 128-byte block is 2 cache lines.  In the inner loop, loading it brings
all 4 weight vectors into L1 with one prefetch instruction.  Each vector is
then used against MR=3 activation vectors, amortising the load over 3 MACs.

---

### FusedQuantParams Math

For each output channel `oc` of layer `li`, the INT8 GEMM produces a raw
int32 dot product `D`:

```
D = sum_k  activation_u8[k] * weight_i8[k]
```

The relationship to the true floating-point convolution output `y` is:

```
y_oc ≈ (D - in_zp * w_col_sum[oc]) * alpha * in_scale
```

After BN (y_bn = A[oc] * y + B[oc]) and quantisation to output u8:

```
out_u8 = round((y_bn - out_range_min) / out_scale) + out_zp
       = round(D * M_oc + C_oc)
```

where the two constants precomputed in `calibrate_fused()` are:

```
M_oc   = alpha * in_scale * A[oc] / out_scale       ← oc_mul[oc]
C_oc   = (B[oc] - in_zp * w_col_sum[oc] * alpha * in_scale * A[oc])
         / out_scale + out_zp                        ← oc_add[oc]
```

The runtime kernel (`quantize_store_nr4`) then computes for each of the
NR=4 output channels:
```cpp
int q = round(D * oc_mul[oc] + oc_add[oc]);
out = clamp(q, relu_min, 255);
```

This collapses dequantise → multiply alpha → BN affine → requantise into
exactly **one multiply + one add + one clamp** per dot product.

---

### GEBP Microkernel MR=3 NR=4

The innermost loop of `conv_q2q` (for 1×1 layers) looks like this in
pseudo-code:

```
for k in [0, K_pad, step=32]:
    w0 = load256(panel_w[oc_tile][k/32][0])   // OC+0 weights
    w1 = load256(panel_w[oc_tile][k/32][1])   // OC+1 weights
    w2 = load256(panel_w[oc_tile][k/32][2])   // OC+2 weights
    w3 = load256(panel_w[oc_tile][k/32][3])   // OC+3 weights
    a0 = load256(act[m0 + 0][k])              // M-row 0 activations
    a1 = load256(act[m0 + 1][k])              // M-row 1 activations
    a2 = load256(act[m0 + 2][k])              // M-row 2 activations
    // 12 VNNI fused multiply-accumulate:
    acc[0][0] = dpbusd(acc[0][0], a0, w0)
    acc[0][1] = dpbusd(acc[0][1], a0, w1)
    acc[0][2] = dpbusd(acc[0][2], a0, w2)
    acc[0][3] = dpbusd(acc[0][3], a0, w3)
    acc[1][0] = dpbusd(acc[1][0], a1, w0)
    // ... (12 total)
```

`_mm256_dpbusd_epi32(acc, u8, i8)` is the Intel AVX-VNNI instruction that
computes the 32-way u8×i8 dot product and accumulates into i32 in one clock.

**Register allocation:**  12 accumulators + 3 activation registers + 1 active
weight register = 16 total YMM registers.  This is exactly the AVX2 register
file size — zero register spills.

**Weight reuse ratio:** Each weight register (w0, w1, w2, w3) is loaded once
but used for all MR=3 rows.  This gives 3× weight reuse per load.

---

## Build Instructions

### Release build (recommended)
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Debug build with sanitisers
```bash
mkdir build_debug && cd build_debug
cmake .. -DCMAKE_BUILD_TYPE=Debug -DSANITIZE=ON
make -j$(nproc)
```

### Without AVX-VNNI (e.g. older Skylake)
The scalar fallback is compiled unconditionally in the `#else` branches.
The standalone lean TRN3 format removes nibble-LUT and bitmask payloads,
because the engine only executes the optimized int2-based path.
Removing `-march=native` or adding `-mno-avxvnni` will use the scalar path.
Performance will be roughly 3–4× lower.

### Verifying VNNI availability
```bash
grep -m1 avxvnni /proc/cpuinfo   # should print "avxvnni" on 12th-gen+
# Or:
python3 -c "import cpuinfo; print(cpuinfo.get_cpu_info()['flags'])"
```

---

## Performance Notes

**Thread scaling:**
- Parallelism is over the M-dimension (spatial output positions) or
  OC-tiles for layers where M < num_threads.
- Layer 1 (56×56 = 3136 positions): good M parallelism, 7× speedup achievable
- Layer 4 (2×2 = 4 positions): M is tiny; parallelism flips to OC-tiles

**Memory bandwidth is not the bottleneck** for most layers:
- 1×1 layers with 256→256 channels: compute-bound (VNNI throughput ~32 TOPS/cycle)
- 3×3 layers with IC=64: bandwidth-bound, but NHWC layout + implicit im2col
  avoid any im2col copy

**ONNX Runtime comparison:**
C++ wins from 2 threads upward because ONNX RT's threading overhead is higher
and it doesn't use VNNI for INT8 GEMM on this model shape.

**Accuracy:**
The single-sample calibration approach occasionally misses rare activation
extremes, potentially causing 0.1–0.2% accuracy loss vs per-channel float.
Calibrate with 500+ diverse samples for production use.

---

## Hardware Requirements

| Feature | Required | Notes |
|---------|----------|-------|
| x86-64  | Yes      | Linux or macOS |
| AVX2    | Yes      | Intel Haswell(2013)+, AMD Zen1(2017)+ |
| FMA3    | Yes      | Same generation as AVX2 |
| AVX-VNNI | Recommended | Intel Alder Lake(2021)+; falls back to scalar |
| OpenMP  | Yes      | GCC/Clang libomp |

**Tested on:** Intel Core i7-13700HX (Raptor Lake), Ubuntu 22.04, GCC 11.4.
