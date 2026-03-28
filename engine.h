#pragma once
// engine.h — Ternary ResNet-50 Inference Engine
// ================================================
// Stripped-down, production-ready header exposing only what is needed
// for the fused quantized (u8 → u8) inference pipeline.
//
// Everything here is used by at least one code path.  This file has no dead
// code from experimentation — only the final, best-performing design.
//
// Requires: x86-64, AVX2 + FMA (runtime check via HAVE_AVX2).
//           AVX-VNNI or AVX-512VNNI improves throughput but is optional.

#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <memory>
#include <chrono>

#if defined(__AVX2__)
#  include <immintrin.h>
#  define HAVE_AVX2 1
#else
#  define HAVE_AVX2 0
#endif

#if defined(__AVXVNNI__) || defined(__AVX512VNNI__)
// AVX-VNNI: _mm256_dpbusd_epi32(acc, u8_activation, i8_weight)
// Computes 32-way u8×i8 dot product in a single fused instruction,
// adding the result directly into the 32-bit accumulator.
// This is what enables efficient INT8 GEMM on Intel Alder/Raptor Lake.
#  define HAVE_VNNI 1
#else
#  define HAVE_VNNI 0
#endif

#ifndef TC_ALIGN
#  define TC_ALIGN 64   // 64-byte alignment (one AVX-512 register / cache line)
#endif

namespace te {

using f32 = float;
using i32 = int32_t;
using u8  = uint8_t;
using i8  = int8_t;

// ── Aligned allocator ────────────────────────────────────────────────────
// All weight buffers and activation scratch space are 64-byte aligned so
// that _mm256_load_si256 (aligned load, faster than loadu) can be used.
inline void* tc_alloc(size_t n) {
    void* p = nullptr;
    if (posix_memalign(&p, TC_ALIGN, n + TC_ALIGN) != 0) p = nullptr;
    return p;
}
inline void tc_free(void* p) { free(p); }

// ── TernaryWeights ────────────────────────────────────────────────────────
// Holds one conv layer's ternary weights in two forms:
//   int2_packed : 4 weights per byte, as written by export_trn3_engine.py
//   unpacked    : one i8 {-1,0,+1} per weight, 64-byte padded
// At load time int2_packed is unpacked → unpacked.
// The unpacked buffer is used by calibrate_fused() to build the int8 GEMM
// weight matrix.  After calibrate_fused() the engine works entirely from
// the panel-packed (NR=4) GEMM layout, so unpacked is no longer hot.
struct TernaryWeights {
    u8*    int2_packed = nullptr;  // raw from file
    size_t int2_bytes  = 0;
    i8*    unpacked    = nullptr;  // {-1,0,+1}; K_pad bytes per OC row
    size_t n_weights   = 0;
    f32    alpha       = 1.0f;    // per-layer scale: real_weight = alpha * sign(w)

    ~TernaryWeights() { tc_free(int2_packed); tc_free(unpacked); }
    TernaryWeights() = default;
    TernaryWeights(const TernaryWeights&) = delete;
    TernaryWeights& operator=(const TernaryWeights&) = delete;
};

// ── FusedBN ──────────────────────────────────────────────────────────────
// BatchNorm folded into a per-channel affine: y = A[c]*x + B[c].
// A = gamma / sqrt(var+eps),  B = beta - A*mean.
// This fusion is done at load time and eliminates the division at runtime.
struct FusedBN {
    std::vector<f32> A, B;  // per output-channel scale and bias
    bool valid = false;
    void init(const f32* g, const f32* b, const f32* m, const f32* v,
              int C, f32 eps = 1e-5f) {
        A.resize(C); B.resize(C);
        for (int c = 0; c < C; ++c) {
            A[c] = g[c] / std::sqrt(v[c] + eps);
            B[c] = b[c] - A[c] * m[c];
        }
        valid = true;
    }
};

// ── ConvParams ────────────────────────────────────────────────────────────
// All state for a single conv layer.
// packed_w / nhwc_w / panel_w are materialised by prepare_weights() and
// calibrate_fused() respectively.  They are nullptr for FP32 downsample
// layers until calibrate_fused() packs them.
struct ConvParams {
    std::string name;
    int out_c=0, in_c=0, kH=0, kW=0, stride=1, padding=0;
    bool is_ternary = false;

    // Ternary layers: tw holds alpha + unpacked {-1,0,+1} weights
    std::unique_ptr<TernaryWeights> tw;
    // FP32 downsample layers: weight_fp32 holds the original float weights
    std::vector<f32> weight_fp32;
    int n_weights = 0;

    FusedBN fbn;  // BN fused into this conv (set at load)

    // INT8 GEMM layout: [OC × K_pad], 64-byte aligned, K_pad = round_up(K,64)
    i8*  packed_w = nullptr;
    int  packed_K = 0;

    // Per-OC sum of weights: needed for zero-point correction.
    //   real_dot = int_dot * w_scale * a_scale
    //              - in_zp * w_col_sum * w_scale * a_scale
    // w_col_sum[oc] = sum_{k} packed_w[oc*K_pad + k]
    std::vector<i32> w_col_sum;

    bool is_1x1() const { return kH == 1 && kW == 1 && padding == 0; }

    ~ConvParams() { tc_free(packed_w); }
    ConvParams() = default;
    ConvParams(const ConvParams&) = delete;
    ConvParams& operator=(const ConvParams&) = delete;
    ConvParams(ConvParams&&) noexcept;
    ConvParams& operator=(ConvParams&&) noexcept;
};

struct FCParams {
    int in_f=0, out_f=0;
    std::vector<f32> weight;
    bool has_bias = false;
    std::vector<f32> bias;
};

// ── Float Tensor (NCHW) ──────────────────────────────────────────────────
// Used only during calibration.  Not allocated during inference.
struct Tensor {
    std::vector<f32> data;
    int N=0, C=0, H=0, W=0;
    Tensor() = default;
    Tensor(int n, int c, int h, int w)
        : N(n), C(c), H(h), W(w), data(size_t(n)*c*h*w, 0.f) {}
    f32* ptr() { return data.data(); }
    const f32* ptr() const { return data.data(); }
    size_t numel() const { return data.size(); }
};

// ── QTensor (NHWC u8) ────────────────────────────────────────────────────
// The quantized activation tensor used throughout the inference pipeline.
// NHWC layout: data[h*W*C + w*C + c] — chosen because im2col reads contiguous
// IC-vectors from a single (h,w) position, which maps directly to this layout.
//
// Quantization: real_value = (u8_value - zero_point) * scale
struct QTensor {
    u8* data = nullptr;
    int C=0, H=0, W=0;
    f32 scale = 1.0f;
    int zero_point = 0;

    QTensor() = default;
    ~QTensor() { if (data) tc_free(data); }
    QTensor(QTensor&&) noexcept;
    QTensor& operator=(QTensor&&) noexcept;
    QTensor(const QTensor&) = delete;
    QTensor& operator=(const QTensor&) = delete;

    // Resize (or reuse existing allocation) without zeroing.
    // Called per conv output; allocation only happens when capacity grows.
    void ensure(int c, int h, int w) {
        size_t need = (size_t)c * h * w;
        if (need > cap_) {
            if (data) tc_free(data);
            cap_ = need + 64;
            data = reinterpret_cast<u8*>(tc_alloc(cap_));
        }
        C = c; H = h; W = w;
    }
    size_t numel() const { return (size_t)C * H * W; }
private:
    size_t cap_ = 0;
};

// ── FusedQuantParams ─────────────────────────────────────────────────────
// Pre-computed per-layer quantization constants.  Computed once in
// calibrate_fused() and used every inference in conv_q2q().
//
// For each output channel oc:
//   oc_mul[oc]  = (alpha * in_scale * bn_A[oc]) / out_scale
//   oc_add[oc]  = (bn_B[oc] - in_zp * w_col_sum[oc] * alpha * in_scale * bn_A[oc])
//                 / out_scale + out_zp
//
// The conv output formula then becomes:
//   output_u8[oc] = clamp(round(int_dot * oc_mul[oc] + oc_add[oc]), relu_min, 255)
//
// This single multiply-add fuses: dequantize → multiply alpha → BN affine → requantize.
// No division, no float-to-int intermediate writes, no separate BN pass.
struct FusedQuantParams {
    f32 in_scale = 0.f;
    int in_zp    = 0;
    f32 out_scale = 1.f;
    int out_zp    = 0;
    std::vector<f32> oc_mul;  // per output channel
    std::vector<f32> oc_add;  // per output channel
    bool valid = false;
};

// ── TernaryCNN ────────────────────────────────────────────────────────────
// Main inference engine.  Call sequence:
//   1. load(path)             — parse TRN3 binary, unpack weights
//   2. calibrate_fused(imgs)  — run FP32 calibration on 1..N images,
//                               then build all quantized pipeline state
//   3. set_num_threads(n)     — optional, default=8
//   4. infer_fused(img)       — fully-quantized inference, returns logits
class TernaryCNN {
public:
    ~TernaryCNN();

    // Optional per-call profiling counters.
    struct ProfileStats {
        double total_us      = 0.0;
        double conv_total_us = 0.0;
        double gap_us        = 0.0;
        double fc_us         = 0.0;
        uint64_t conv_ops    = 0;
        uint64_t fc_ops      = 0;
        uint64_t total_ops   = 0;
        std::vector<double> conv_layer_us;
    };

    // Parse lean TRN3 binary written by export_trn3_engine.py.
    // Returns false on error (bad magic, file not found, etc.).
    bool load(const std::string& bin_path);

    // Calibration.  Must be called before infer_fused().
    // sample_chw: pointer to n_calib contiguous CIFAR-10-normalised images
    //             in CHW float32 layout.
    // n_calib: number of images in the calibration buffer (default 1).
    // Internally:
    //   a) Run FP32 forward passes to capture robust per-layer min/max.
    //   b) Compute FusedQuantParams (oc_mul, oc_add) per conv layer.
    //   c) Pack weights to NHWC + NR=4 panel layout for GEBP microkernel.
    void calibrate_fused(const f32* sample_chw, int C, int H, int W,
                         int n_calib = 1);

    bool fused_ready() const { return fused_ready_; }

    // Fully-quantized inference.  Input must be CIFAR-10-normalised CHW float.
    // Everything inside (conv, BN, ReLU, residual add) runs in u8 arithmetic.
    // Returns 10-element logit vector.
    std::vector<f32> infer_fused(const f32* img_chw, int C, int H, int W);

    void set_num_threads(int n) { num_threads_ = std::max(1, n); }
    int  get_num_threads() const { return num_threads_; }
    void set_profiling(bool enabled) { profiling_enabled_ = enabled; }
    void reset_profile();
    ProfileStats get_last_profile() const { return profile_; }

    bool loaded = false;

private:
    // All conv layers. Index matches TRN3 file order:
    //   [0]        stem (3→64, 3×3, stride 1)
    //   [1..3]     layer1 block 0 (1×1, 3×3, 1×1 bottleneck)
    //   [4]        layer1 block 0 downsample (1×1 FP32)
    //   [5..7]     layer1 block 1
    //   [8..10]    layer1 block 2
    //   ... (16 bottleneck blocks total, 4 downsample, 53 conv total)
    std::vector<ConvParams> convs_;
    std::vector<FusedBN>    bns_;   // parallel to convs_
    FCParams                fc_;

    int  num_threads_       = 8;
    bool profiling_enabled_ = false;
    ProfileStats profile_;

    // ── Fused Q2Q pipeline state ──────────────────────────────────────
    bool fused_ready_ = false;

    // Per-layer calibration range (float output min/max from one sample)
    struct CalibRange { f32 mn = 0.f, mx = 0.f; };
    std::vector<CalibRange>     calib_ranges_;
    std::vector<f32>            block_out_scales_; // per block: scale after add+relu
    std::vector<FusedQuantParams> fqp_;            // per conv layer

    f32 stem_in_scale_ = 0.f;
    int stem_in_zp_    = 0;

    // NHWC-repacked weights for 3×3 conv layers.
    // Original NCHW im2col order [IC][kH][kW] → NHWC order [kH][kW][IC].
    // nullptr for 1×1 layers (already compatible).
    std::vector<i8*> nhwc_w_;

    // NR=4 panel-packed weights: [OC/NR][K_pad/32][NR][32].
    // This layout puts the 4 weight vectors for NR=4 OC tiles contiguous,
    // so each k-step loads 128 bytes (2 cache lines) for all 4 OC simultaneously.
    // The GEBP microkernel loads wk[0..127] and reuses them across MR=3 activations.
    std::vector<i8*> panel_w_;

    // Pre-allocated QTensor buffers.  Reused every inference; no allocation on
    // the hot path.  Sized to the maximum intermediate tensor (512×8×8 for CIFAR).
    QTensor q_a_, q_b_, q_t1_, q_t2_, q_t3_, q_ds_;

    // ── Private helpers ──────────────────────────────────────────────
    void prepare_weights();  // called by calibrate_fused if needed

    // Core Q2Q kernel: INT8 GEMM + fused requantize + optional ReLU.
    // Uses GEBP MR=3 NR=4 for IC≥64, MR=1 for stem.
    void conv_q2q(int li, const QTensor& in, QTensor& out, bool do_relu);

    // Fused residual add + ReLU + requantize (u8,u8 → u8).
    // Eliminates separate dequant/requant steps for the residual connection.
    static void q_add_relu(const QTensor& a, const QTensor& b,
                           QTensor& out, f32 out_scale);

    // Execute one bottleneck block: 3 convs + downsample (if any) + add + relu.
    void bottleneck_q2q(const QTensor& in, QTensor& out,
                        int conv_start, int ds_idx, f32 block_out_scale);

    void add_conv_profile(int li, int H_out, int W_out, double us);
};

// ── P-core pinning ────────────────────────────────────────────────────────
// On Intel hybrid CPUs (i7-13xxxHX, etc.), bind one OpenMP thread per
// physical P-core.  P-cores are at even logical CPU IDs (0,2,4,...).
// Call once before inference on the first thread count, or call
// set_num_threads + pin again to re-pin.
void pin_threads_to_pcores(int n_pcores = 8);

} // namespace te
