# End-to-End Flow: Ternary Model to Fast C++ Inference

This folder is a standalone workflow for:

1. training a ternary ResNet-50,
2. exporting it to the lean `TRN3` binary format expected by the C++ engine,
3. building the fast C++ inference binary,
4. running inference on CIFAR-10 binary data.

All commands below assume you start from this folder:

```bash
cd /home/debas/code/stellon-labs-assignment/final/resnet_inference_engine
```

## 1. Folder Contents

- `train_ternary_engine.py`
  Trains the model and writes an engine-ready baked ternary checkpoint.

- `export_trn3_engine.py`
  Exports a checkpoint to `TRN3` and validates the output.

- `engine.h`, `engine.cpp`
  Standalone ternary inference engine implementation.

- `inference_main.cpp`
  Demo executable for calibration, benchmarking, and accuracy evaluation.

- `CMakeLists.txt`
  Build file for the standalone C++ inference binary.

## 2. What Files Matter

The normal artifact flow is:

```text
train_ternary_engine.py
  -> ./weights/resnet50_ternary_float_best.pth
  -> ./weights/resnet50_ternary_engine_ready.pth
  -> ./weights/activation_scales.json

export_trn3_engine.py
  -> ./weights/resnet50_ternary_packed.bin

inference_demo
  reads ../weights/resnet50_ternary_packed.bin
  reads CIFAR-10 binary test file
```

## 3. Train an Engine-Ready Ternary Model

Run:

```bash
python3 train_ternary_engine.py
```

What this does:

1. trains a ternary ResNet-50 on CIFAR-10,
2. saves the best float-training checkpoint as:
  `./weights/resnet50_ternary_float_best.pth`
3. bakes ternary weights into `{-1, 0, +1}` form,
4. stores `alpha_scale` buffers needed by the exporter,
5. recalibrates BatchNorm running stats after baking,
6. saves the export-safe checkpoint as:
  `./weights/resnet50_ternary_engine_ready.pth`
7. saves activation calibration data as:
  `./weights/activation_scales.json`

Useful options:

```bash
python3 train_ternary_engine.py \
  --epochs 20 \
  --batch-size 128 \
  --workers 4 \
  --bn-recalib-batches 50 \
  --act-calib-batches 20
```

Important:

- Use `resnet50_ternary_engine_ready.pth` for export when possible.
- This avoids the earlier issue where a checkpoint contained float weights or missing `alpha_scale` state for ternary export.

## 4. Export to TRN3 for the C++ Engine

Run:

```bash
python3 export_trn3_engine.py
```

Default behavior:

1. prefers `./weights/resnet50_ternary_engine_ready.pth`,
2. falls back to `./weights/resnet50_ternary.pth` if needed,
3. loads `alpha_scale` buffers correctly before `load_state_dict`,
4. reads `./weights/activation_scales.json` if present,
5. writes a lean ternary-only payload:
  `./weights/resnet50_ternary_packed.bin`
6. validates:
   - magic header is `TRN3`
   - version is `4`
   - file size is plausible

Recommended safe export command:

```bash
python3 export_trn3_engine.py --bake-if-needed
```

Useful custom export:

```bash
python3 export_trn3_engine.py \
  --ckpt ./weights/resnet50_ternary_engine_ready.pth \
  --act-scales ./weights/activation_scales.json \
  --out ./weights/resnet50_ternary_packed.bin
```

Expected successful output looks like:

```text
Loading checkpoint: ...
Loaded activation scales: ...
Exported: ./weights/resnet50_ternary_packed.bin
Header: magic=TRN3 version=4 conv=53 bn=53 fc=1
Size: about 16-17 MB
```

Version 4 is the lean standalone format. It removes legacy nibble-LUT and
bitmask payloads because the C++ engine never uses them.

## 5. Build the Fast C++ Inference Binary

Configure and build:

```bash
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

This produces:

```text
build/inference_demo
```

Build notes:

- Release uses `-O3 -march=native -funroll-loops -ffast-math`.
- OpenMP is enabled.
- The engine is optimized for AVX2 and can benefit from AVX-VNNI when available.

## 6. Prepare the Input Dataset

`inference_demo` expects a CIFAR-10 binary file in standard CIFAR format:

```text
1 byte label + 3072 bytes image
```

with image data in CHW order:

```text
[R(1024), G(1024), B(1024)]
```

Typical paths from this standalone folder:

```text
./data/test_batch.bin
```

or

```text
/tmp/cifar10_test.bin
```

Important:

- `./data/test_batch.bin` must be a raw CIFAR record file with fixed-size
  `3073`-byte records.
- Do not replace it with the original Python CIFAR batch artifact from
  `cifar-10-batches-py/test_batch`; that file is a pickle file and will be
  rejected by the stricter input validation.

## 7. Run Inference

From `resnet_inference_engine/build`:

```bash
./inference_demo ../weights/resnet50_ternary_packed.bin ../data/test_batch.bin
```

Full CLI:

```bash
./inference_demo <weights.bin> <cifar_test_batch.bin> [num_threads [max_images [calib_images]]]
```

Arguments:

1. `weights.bin`
   `TRN3` binary produced by `export_trn3_engine.py`
2. `cifar_test_batch.bin`
   CIFAR-10 binary test file
3. `num_threads`
   optional, default `8`
4. `max_images`
   optional, evaluate only the first N images
5. `calib_images`
   optional, number of images used for quantization calibration, default `64`

Example quick run:

```bash
./inference_demo ../weights/resnet50_ternary_packed.bin ../data/test_batch.bin 8 100 64
```

Example full test-set run:

```bash
./inference_demo ../weights/resnet50_ternary_packed.bin ../data/test_batch.bin 8 10000 64
```

What happens internally:

1. load `TRN3` weights,
2. load CIFAR-10 binary records,
3. normalize input using CIFAR-10 mean/std,
4. calibrate the fused Q2Q pipeline from `calib_images` samples,
5. run a short warmup,
6. benchmark 100 inferences on the first image,
7. evaluate accuracy on `max_images` images.

## 8. Expected Output

A normal run prints:

```text
Loading model from ...
Loading TRN3 v4: 53 conv, 53 bn, fc=1
Loaded 53 conv layers (49 ternary)
Loaded 10000 images from ...
Calibrating fused Q2Q pipeline with 64 image(s)...
Calibration done.

Benchmarking (100 runs, 8 threads)...
  Latency (ms): min=... p50=... p95=... mean=...

Running inference on 10000 images...

Results:
  Accuracy:         ...
  Avg latency/img:  ...
  Total time:       ...
  Throughput:       ...
```

## 9. Known Accuracy Reality

For this current ternary pipeline, the important success criterion is parity between:

1. PyTorch using the same engine-ready ternary checkpoint/loading path,
2. the C++ standalone engine using the exported `TRN3` file.

In this workspace, that parity has been verified.

That means:

- if C++ and PyTorch agree, the engine/export path is correct,
- if accuracy is low, that is a model-quality issue, not necessarily an engine bug.

## 10. Common Failure Modes and Fixes

### A. Exported `.bin` is tiny, for example 4 KB

Cause:

- corrupted placeholder file,
- wrong export path,
- export did not actually run.

Fix:

```bash
python3 export_trn3_engine.py --bake-if-needed
```

Then verify the exporter reports:

```text
magic=TRN3
Size: about 16-17 MB
```

### B. Checkpoint exports badly because `alpha_scale` is missing

Cause:

- loading checkpoint without registering `alpha_scale` buffers first.

Fix:

- use `export_trn3_engine.py` from this folder,
- do not use ad hoc checkpoint loading logic.

### C. Exported model behaves like float model during eval, not ternary

Cause:

- using a float-training checkpoint directly instead of the baked engine-ready checkpoint.

Fix:

- train with `train_ternary_engine.py`,
- export `resnet50_ternary_engine_ready.pth`, or use `--bake-if-needed`.

### D. Inference input format is wrong

Cause:

- using HWC image bytes when the loader expects CIFAR binary CHW records.

Fix:

- use standard CIFAR-10 binary files,
- keep the input file format as `1 label byte + 3072 CHW bytes` per image,
- do not point the demo at the Python pickle batch from `cifar-10-batches-py`.

## 11. Minimal Happy Path

If you just want the shortest reliable sequence:

```bash
cd /home/debas/code/stellon-labs-assignment/final/resnet_inference_engine

python3 train_ternary_engine.py
python3 export_trn3_engine.py --bake-if-needed

mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

./inference_demo ../weights/resnet50_ternary_packed.bin ../data/test_batch.bin 8 1000 64
```

## 12. Summary

The intended standalone flow is:

```text
train_ternary_engine.py
  -> resnet50_ternary_engine_ready.pth
  -> activation_scales.json

export_trn3_engine.py
  -> resnet50_ternary_packed.bin

cmake + make
  -> inference_demo

inference_demo
  -> fast C++ ternary inference
```

If you stick to the two scripts in this folder for training and export, you avoid the checkpoint/export mismatch that caused the earlier model export issues.