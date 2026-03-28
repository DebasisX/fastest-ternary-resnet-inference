// engine.cpp — Ternary ResNet-50 Inference Engine Implementation
// ==============================================================
// Only the optimal inference path is included here:
//
//   Load → prepare_weights → calibrate_fused → infer_fused (loop)
//
// The GEBP MR=3 NR=4 microkernel (conv_q2q) is the only convolution
// implementation.  All experimental/comparison paths have been removed.
//
// AVX-VNNI path (_mm256_dpbusd_epi32) is selected at compile time if
// -mavxvnni is passed.  A scalar fallback is provided for portability.

#include "engine.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <omp.h>
#include <sched.h>
#include <pthread.h>

namespace te {

// ══════════════════════════════════════════════════════════════════════════
//  BASIC HELPERS
// ══════════════════════════════════════════════════════════════════════════

static double now_us() {
    using namespace std::chrono;
    return duration<double, std::micro>(
        high_resolution_clock::now().time_since_epoch()).count();
}

#if HAVE_AVX2
// Reduce 8×i32 YMM register to a single i32 scalar.
// Used at the end of the k-loop in the GEBP microkernel to collapse the
// 8-lane accumulator into the final dot product.
static inline i32 hsum_epi32(__m256i v) {
    __m128i lo  = _mm256_castsi256_si128(v);
    __m128i hi  = _mm256_extracti128_si256(v, 1);
    __m128i sum = _mm_add_epi32(lo, hi);
    __m128i shuf = _mm_shuffle_epi32(sum, _MM_SHUFFLE(1,0,3,2));
    sum = _mm_add_epi32(sum, shuf);
    shuf = _mm_shuffle_epi32(sum, _MM_SHUFFLE(2,3,0,1));
    sum = _mm_add_epi32(sum, shuf);
    return _mm_cvtsi128_si32(sum);
}
#endif

// AVX2 FP32 dot product, 2× unrolled for ILP.
// Only used during the one-shot calibration forward pass.
static f32 fp32_dot(const f32* w, const f32* a, int n) {
#if HAVE_AVX2
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(w+i),   _mm256_loadu_ps(a+i),   acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(w+i+8), _mm256_loadu_ps(a+i+8), acc1);
    }
    for (; i + 8 <= n; i += 8)
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(w+i), _mm256_loadu_ps(a+i), acc0);
    acc0 = _mm256_add_ps(acc0, acc1);
    __m128 lo = _mm256_castps256_ps128(acc0);
    __m128 hi = _mm256_extractf128_ps(acc0, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s);
    f32 r = _mm_cvtss_f32(s);
    for (; i < n; ++i) r += w[i] * a[i];
    return r;
#else
    f32 r = 0.f;
    for (int i = 0; i < n; ++i) r += w[i] * a[i];
    return r;
#endif
}

// ══════════════════════════════════════════════════════════════════════════
//  ConvParams / QTensor move operations
// ══════════════════════════════════════════════════════════════════════════

ConvParams::ConvParams(ConvParams&& o) noexcept
    : name(std::move(o.name)), out_c(o.out_c), in_c(o.in_c),
      kH(o.kH), kW(o.kW), stride(o.stride), padding(o.padding),
      is_ternary(o.is_ternary), tw(std::move(o.tw)),
      weight_fp32(std::move(o.weight_fp32)), n_weights(o.n_weights),
      fbn(std::move(o.fbn)), packed_w(o.packed_w), packed_K(o.packed_K),
      w_col_sum(std::move(o.w_col_sum))
{ o.packed_w = nullptr; }

ConvParams& ConvParams::operator=(ConvParams&& o) noexcept {
    if (this != &o) {
        tc_free(packed_w);
        name = std::move(o.name); out_c = o.out_c; in_c = o.in_c;
        kH = o.kH; kW = o.kW; stride = o.stride; padding = o.padding;
        is_ternary = o.is_ternary; tw = std::move(o.tw);
        weight_fp32 = std::move(o.weight_fp32); n_weights = o.n_weights;
        fbn = std::move(o.fbn); packed_w = o.packed_w; packed_K = o.packed_K;
        w_col_sum = std::move(o.w_col_sum);
        o.packed_w = nullptr;
    }
    return *this;
}

QTensor::QTensor(QTensor&& o) noexcept
    : data(o.data), C(o.C), H(o.H), W(o.W),
      scale(o.scale), zero_point(o.zero_point), cap_(o.cap_)
{ o.data = nullptr; o.cap_ = 0; }

QTensor& QTensor::operator=(QTensor&& o) noexcept {
    if (this != &o) {
        if (data) tc_free(data);
        data = o.data; C = o.C; H = o.H; W = o.W;
        scale = o.scale; zero_point = o.zero_point; cap_ = o.cap_;
        o.data = nullptr; o.cap_ = 0;
    }
    return *this;
}

// ══════════════════════════════════════════════════════════════════════════
//  TernaryCNN destructor
// ══════════════════════════════════════════════════════════════════════════

TernaryCNN::~TernaryCNN() {
    for (auto* p : nhwc_w_)  if (p) tc_free(p);
    for (auto* p : panel_w_) if (p) tc_free(p);
}

// ══════════════════════════════════════════════════════════════════════════
//  WEIGHT LOADING (lean TRN3 binary format)
// ══════════════════════════════════════════════════════════════════════════

static bool read_exact(FILE* f, void* dst, size_t bytes) {
    return bytes == 0 || fread(dst, 1, bytes, f) == bytes;
}

static bool read_i32(FILE* f, i32& v) {
    return read_exact(f, &v, sizeof(v));
}

static bool read_f32(FILE* f, f32& v) {
    return read_exact(f, &v, sizeof(v));
}

static bool read_f32_arr(FILE* f, f32* dst, int n) {
    return n >= 0 && read_exact(f, dst, sizeof(f32) * (size_t)n);
}

static bool read_str(FILE* f, std::string& s) {
    i32 len = 0;
    if (!read_i32(f, len) || len < 0) return false;
    s.assign((size_t)len, '\0');
    return len == 0 || read_exact(f, &s[0], (size_t)len);
}

// Unpack 4-weights-per-byte int2 encoding to one signed byte per weight.
// Encoding used by export_trn3_engine.py:
//   -1 → 0b01,  0 → 0b00,  +1 → 0b11
// We use the lookup table DEC4[code] to handle this in one array access.
static void unpack_int2(TernaryWeights* tw) {
    static constexpr i8 DEC4[4] = { 0, -1, 0, 1 };  // 00→0, 01→-1, 10→0, 11→+1
    size_t n = tw->n_weights;
    size_t padded = (n + 63) & ~63u;  // pad to 64-byte for aligned SIMD loads later
    tw->unpacked = reinterpret_cast<i8*>(tc_alloc(padded));
    std::memset(tw->unpacked, 0, padded);
    for (size_t i = 0; i < n; ++i) {
        size_t bi = i >> 2;
        int    sh = (int)(i & 3) << 1;
        tw->unpacked[i] = DEC4[(tw->int2_packed[bi] >> sh) & 3u];
    }
}

bool TernaryCNN::load(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path.c_str()); return false; }

    auto fail = [&](const char* msg) {
        fprintf(stderr, "Failed to load %s: %s\n", path.c_str(), msg);
        fclose(f);
        return false;
    };

    char magic[4];
    if (!read_exact(f, magic, sizeof(magic)))
        return fail("truncated header");
    if (memcmp(magic, "TRN3", 4) != 0) {
        fprintf(stderr, "Bad magic: expected TRN3\n");
        fclose(f); return false;
    }
    i32 version = 0, n_conv = 0, n_bn = 0, has_fc = 0;
    if (!read_i32(f, version) || !read_i32(f, n_conv) || !read_i32(f, n_bn) || !read_i32(f, has_fc))
        return fail("truncated TRN3 header fields");
    if (version != 4) {
        fprintf(stderr, "Unsupported TRN3 version %d (expected 4). Re-export with export_trn3_engine.py\n", version);
        fclose(f); return false;
    }
    printf("Loading TRN3 v%d: %d conv, %d bn, fc=%d\n", version, n_conv, n_bn, has_fc);

    convs_.resize(n_conv);
    bns_.resize(n_conv);

    for (int i = 0; i < n_conv; ++i) {
        auto& cp    = convs_[i];
        i32 groups = 0, is_ternary = 0;
        if (!read_str(f, cp.name) ||
            !read_i32(f, cp.out_c) ||
            !read_i32(f, cp.in_c) ||
            !read_i32(f, cp.kH) ||
            !read_i32(f, cp.kW) ||
            !read_i32(f, cp.stride) ||
            !read_i32(f, cp.padding) ||
            !read_i32(f, groups) ||
            !read_i32(f, is_ternary))
            return fail("truncated conv record");
        cp.is_ternary = (is_ternary == 1);

        if (cp.is_ternary) {
            auto tw = std::make_unique<TernaryWeights>();
            f32 act_scale_unused = 0.f;
            i32 n_weights_i32 = 0;
            if (!read_f32(f, tw->alpha) ||
                !read_f32(f, act_scale_unused) ||
                !read_i32(f, n_weights_i32) ||
                n_weights_i32 < 0)
                return fail("truncated ternary conv header");
            tw->n_weights = (size_t)n_weights_i32;

            // Lean ternary payload: int2 packed only.
            i32 int2_bytes_i32 = 0;
            if (!read_i32(f, int2_bytes_i32) || int2_bytes_i32 < 0)
                return fail("invalid ternary payload size");
            tw->int2_bytes = (size_t)int2_bytes_i32;
            tw->int2_packed  = reinterpret_cast<u8*>(tc_alloc(tw->int2_bytes));
            if (!tw->int2_packed)
                return fail("out of memory allocating ternary payload");
            if (!read_exact(f, tw->int2_packed, (size_t)tw->int2_bytes)) {
                tc_free(tw->int2_packed);
                tw->int2_packed = nullptr;
                return fail("truncated ternary payload");
            }
            unpack_int2(tw.get());
            // int2_packed is no longer needed after unpacking; free to save ~4MB
            tc_free(tw->int2_packed);
            tw->int2_packed = nullptr;

            cp.n_weights = tw->n_weights;
            cp.tw        = std::move(tw);
        } else {
            // FP32 downsample layer (4 of them in ResNet-50)
            f32 alpha_unused = 0.f, act_scale_unused = 0.f;
            i32 data_bytes = 0;
            if (!read_f32(f, alpha_unused) ||
                !read_f32(f, act_scale_unused) ||
                !read_i32(f, cp.n_weights) ||
                !read_i32(f, data_bytes) ||
                cp.n_weights < 0 || data_bytes < 0)
                return fail("invalid FP32 downsample record");
            cp.weight_fp32.resize(cp.n_weights);
            if (!read_exact(f, cp.weight_fp32.data(), (size_t)data_bytes))
                return fail("truncated FP32 downsample payload");
        }
    }

    // BN layers: read all, then assign to convs 1:1 by index
    std::vector<FusedBN> all_bns;
    all_bns.reserve(n_bn);
    for (int i = 0; i < n_bn; ++i) {
        std::string name;
        int nf = 0;
        if (!read_str(f, name) || !read_i32(f, nf) || nf < 0)
            return fail("truncated BN record header");
        std::vector<f32> g(nf), b(nf), m(nf), v(nf);
        f32 eps = 0.f;
        if (!read_f32_arr(f, g.data(), nf) ||
            !read_f32_arr(f, b.data(), nf) ||
            !read_f32_arr(f, m.data(), nf) ||
            !read_f32_arr(f, v.data(), nf) ||
            !read_f32(f, eps))
            return fail("truncated BN payload");
        FusedBN bn;
        bn.init(g.data(), b.data(), m.data(), v.data(), nf, eps);
        all_bns.push_back(std::move(bn));
    }
    for (size_t i = 0; i < all_bns.size() && i < convs_.size(); ++i)
        bns_[i] = std::move(all_bns[i]);

    if (has_fc) {
        if (!read_i32(f, fc_.in_f) || !read_i32(f, fc_.out_f) || fc_.in_f < 0 || fc_.out_f < 0)
            return fail("truncated FC header");
        fc_.weight.resize(fc_.out_f * fc_.in_f);
        if (!read_exact(f, fc_.weight.data(), sizeof(f32) * (size_t)fc_.out_f * (size_t)fc_.in_f))
            return fail("truncated FC weight payload");
        i32 has_bias_i32 = 0;
        if (!read_i32(f, has_bias_i32))
            return fail("truncated FC bias flag");
        fc_.has_bias = (has_bias_i32 == 1);
        if (fc_.has_bias) {
            fc_.bias.resize(fc_.out_f);
            if (!read_exact(f, fc_.bias.data(), sizeof(f32) * (size_t)fc_.out_f))
                return fail("truncated FC bias payload");
        }
    }

    fclose(f);
    loaded = true;
    printf("Loaded %d conv layers (%d ternary)\n", n_conv,
           (int)std::count_if(convs_.begin(), convs_.end(),
                              [](const ConvParams& c){ return c.is_ternary; }));
    return true;
}

// ══════════════════════════════════════════════════════════════════════════
//  THREAD AFFINITY
// ══════════════════════════════════════════════════════════════════════════

// Bind one OpenMP thread per physical P-core.
// On i7-13700HX: logical CPUs 0,2,4,6,8,10,12,14 are P-core HT0s.
// Binding here prevents the OS from migrating threads to E-cores mid-run.
void pin_threads_to_pcores(int n_pcores) {
    omp_set_num_threads(n_pcores);
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(tid * 2, &cpuset);  // tid 0→CPU0, tid 1→CPU2, ...
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    }
}

void TernaryCNN::reset_profile() {
    profile_ = ProfileStats{};
    profile_.conv_layer_us.assign(convs_.size(), 0.0);
}

// ══════════════════════════════════════════════════════════════════════════
//  INT8 WEIGHT PREPARATION
// ══════════════════════════════════════════════════════════════════════════
// Pack each ternary conv's unpacked i8 weights into a 64-byte-aligned layout
// [OC × K_pad] where K_pad = round_up(K, 64).  This is the base layout that
// calibrate_fused() will then repack into NHWC and NR=4 panel formats.
// Also precompute w_col_sum[oc] for zero-point correction in the Q2Q kernel.
void TernaryCNN::prepare_weights() {
    for (auto& cp : convs_) {
        if (!cp.is_ternary || !cp.tw) continue;
        int K     = cp.in_c * cp.kH * cp.kW;
        int K_pad = (K + 63) & ~63;
        int N     = cp.out_c;

        tc_free(cp.packed_w);
        cp.packed_w = reinterpret_cast<i8*>(tc_alloc((size_t)N * K_pad));
        std::memset(cp.packed_w, 0, (size_t)N * K_pad);
        cp.packed_K = K_pad;

        for (int oc = 0; oc < N; ++oc)
            std::memcpy(cp.packed_w + (size_t)oc * K_pad,
                        cp.tw->unpacked + (size_t)oc * K, K);

        cp.w_col_sum.resize(N);
        for (int oc = 0; oc < N; ++oc) {
            const i8* row = cp.packed_w + (size_t)oc * K_pad;
            i32 s = 0;
            for (int k = 0; k < K; ++k) s += row[k];
            cp.w_col_sum[oc] = s;
        }
    }
    printf("Weights pre-packed: %zu ternary layers\n",
           std::count_if(convs_.begin(), convs_.end(),
                         [](const ConvParams& c){ return c.packed_w != nullptr; }));
}

// ══════════════════════════════════════════════════════════════════════════
//  CALIBRATION FORWARD PASS HELPERS
// ══════════════════════════════════════════════════════════════════════════
// These functions run exactly once per model.  They are not on the inference
// critical path, so correctness of output values matters more than speed.
// We still use AVX2 fp32_dot to keep calibration under 1 second on CIFAR.

// FP32 im2col: rearranges input patches into a matrix where each row is
// a flattened IC×kH×kW patch for output position (oh, ow).
// Input NCHW → output [H_out*W_out, K] row-major.
static void im2col_fp32(const f32* data_im, int C, int H, int W,
                         int kH, int kW, int stride, int pad,
                         f32* data_col) {
    int H_out = (H + 2*pad - kH) / stride + 1;
    int W_out = (W + 2*pad - kW) / stride + 1;
    int K     = C * kH * kW;
    for (int c = 0; c < C; ++c)
        for (int dh = 0; dh < kH; ++dh)
            for (int dw = 0; dw < kW; ++dw) {
                int col_idx = (c * kH + dh) * kW + dw;
                for (int oh = 0; oh < H_out; ++oh)
                    for (int ow = 0; ow < W_out; ++ow) {
                        int ih = oh * stride - pad + dh;
                        int iw = ow * stride - pad + dw;
                        int row_idx = oh * W_out + ow;
                        f32 v = (ih >= 0 && ih < H && iw >= 0 && iw < W)
                                ? data_im[(c*H + ih)*W + iw] : 0.f;
                        data_col[row_idx * K + col_idx] = v;
                    }
            }
}

// Simple float convolution + BN + optional ReLU.
// Produces a NCHW float Tensor used only inside calibrate_fused().
// For ternary layers: weights are alpha * i8{-1,0,+1}  (already in tw->unpacked).
// For FP32 downsample layers: weights are weight_fp32.
static Tensor conv_calib(const ConvParams& cp, const Tensor& in,
                          const FusedBN& bn, bool do_relu) {
    const int IC    = cp.in_c, OC = cp.out_c;
    const int H = in.H, W = in.W;
    const int H_out = (H + 2*cp.padding - cp.kH) / cp.stride + 1;
    const int W_out = (W + 2*cp.padding - cp.kW) / cp.stride + 1;
    const int K     = IC * cp.kH * cp.kW;
    const int M     = H_out * W_out;

    // Build float weight matrix [OC × K] (ternary weights scaled by alpha)
    std::vector<f32> w_f32((size_t)OC * K);
    if (cp.is_ternary) {
        f32 alpha = cp.tw->alpha;
        for (size_t i = 0; i < (size_t)OC * K; ++i)
            w_f32[i] = (f32)cp.tw->unpacked[i] * alpha;
    } else {
        w_f32 = cp.weight_fp32;
    }

    // im2col: [M, K] patch matrix
    std::vector<f32> col((size_t)M * K);
    im2col_fp32(in.data.data(), IC, H, W, cp.kH, cp.kW, cp.stride, cp.padding, col.data());

    // GEMM + BN + optional ReLU
    Tensor out(1, OC, H_out, W_out);
    for (int oc = 0; oc < OC; ++oc) {
        const f32* w_row = w_f32.data() + (size_t)oc * K;
        f32 bn_a = bn.valid ? bn.A[oc] : 1.f;
        f32 bn_b = bn.valid ? bn.B[oc] : 0.f;
        f32* o   = out.data.data() + (size_t)oc * M;
        for (int m = 0; m < M; ++m) {
            f32 val = fp32_dot(w_row, col.data() + (size_t)m * K, K);
            val = bn_a * val + bn_b;
            if (do_relu && val < 0.f) val = 0.f;
            o[m] = val;
        }
    }
    return out;
}

// ══════════════════════════════════════════════════════════════════════════
//  CALIBRATE FUSED
// ══════════════════════════════════════════════════════════════════════════
// This is the one-time setup step.  Its output drives all subsequent
// infer_fused() calls.  Three phases:
//
//   Phase 1 — Forward pass (float precision)
//     Run a single FP32 inference, collecting per-layer output min/max.
//
//   Phase 2 — Compute FusedQuantParams
//     For each conv layer, precompute oc_mul[oc] and oc_add[oc]:
//       oc_mul  = (alpha * in_scale * bn_A) / out_scale
//       oc_add  = (bn_B - zp * w_col_sum * alpha * in_scale * bn_A)
//                 / out_scale  +  out_zp
//     This collapses dequantize → alpha → BN → requantize into a single
//     multiply-add in the Q2Q kernel.  Zero information loss (linear chain).
//
//   Phase 3 — Weight repacking
//     a) NHWC order (3×3 layers): permute weight K-axis from [IC][kH][kW]
//        to [kH][kW][IC] so the inner loop reads contiguous IC channels.
//     b) NR=4 panel layout: [OC/4][K_pad/32][4][32], so 4 weight vectors
//        per k-chunk are 128 contiguous bytes (2 cache lines, no waste).

// ResNet-50 block table: start conv index and downsample index (-1 = identity)
struct BlockDef { int start; int ds; };
static const BlockDef RESNET_BLOCKS[] = {
    { 1,  4}, { 5, -1}, { 8, -1},                               // layer1
    {11, 14}, {15, -1}, {18, -1}, {21, -1},                     // layer2
    {24, 27}, {28, -1}, {31, -1}, {34, -1}, {37, -1}, {40, -1}, // layer3
    {43, 46}, {47, -1}, {50, -1},                                // layer4
};
static constexpr int N_BLOCKS = 16;

void TernaryCNN::calibrate_fused(const f32* sample_chw, int C, int H, int W,
                                 int n_calib) {
    if (!loaded) return;
    if (!sample_chw) return;
    pin_threads_to_pcores(num_threads_);
    if (!convs_.empty() && convs_[0].is_ternary && !convs_[0].packed_w)
        prepare_weights();

    // ── Phase 1: FP32 forward pass ───────────────────────────────────────
    // Scan helper: find finite float min/max of a tensor
    auto scan = [](const Tensor& t) -> CalibRange {
        f32 mn = 0.f, mx = 0.f;
        for (f32 v : t.data) { if (v < mn) mn = v; if (v > mx) mx = v; }
        return {mn, mx};
    };

    calib_ranges_.assign(convs_.size(), CalibRange{0.f, 0.f});
    block_out_scales_.assign(N_BLOCKS, 0.f);

    const int sample_size = C * H * W;
    n_calib = std::max(1, n_calib);
    bool first_sample = true;
    f32 img_min = 0.f, img_max = 0.f;

    auto merge_range = [&](int li, const CalibRange& r) {
        if (first_sample) {
            calib_ranges_[li] = r;
        } else {
            calib_ranges_[li].mn = std::min(calib_ranges_[li].mn, r.mn);
            calib_ranges_[li].mx = std::max(calib_ranges_[li].mx, r.mx);
        }
    };

    for (int si = 0; si < n_calib; ++si) {
        const f32* cur_sample = sample_chw + (size_t)si * sample_size;

        if (first_sample) {
            img_min = cur_sample[0];
            img_max = cur_sample[0];
        }
        for (int i = 0; i < sample_size; ++i) {
            img_min = std::min(img_min, cur_sample[i]);
            img_max = std::max(img_max, cur_sample[i]);
        }

        Tensor cur(1, C, H, W);
        std::memcpy(cur.data.data(), cur_sample, sizeof(f32) * (size_t)sample_size);

        // Stem
        {
            Tensor out = conv_calib(convs_[0], cur, bns_[0], /*relu=*/true);
            merge_range(0, scan(out));
            cur = std::move(out);
        }

        // Bottleneck blocks
        for (int bi = 0; bi < N_BLOCKS; ++bi) {
            const auto& blk = RESNET_BLOCKS[bi];
            const int c1 = blk.start, c2 = c1+1, c3 = c1+2;

            Tensor x1 = conv_calib(convs_[c1], cur,  bns_[c1], true);  merge_range(c1, scan(x1));
            Tensor x2 = conv_calib(convs_[c2], x1,   bns_[c2], true);  merge_range(c2, scan(x2));
            Tensor x3 = conv_calib(convs_[c3], x2,   bns_[c3], false); merge_range(c3, scan(x3));

            if (blk.ds >= 0) {
                Tensor ds = conv_calib(convs_[blk.ds], cur, bns_[blk.ds], false);
                merge_range(blk.ds, scan(ds));
                // x3 += ds (residual add)
                for (size_t i = 0; i < x3.numel(); ++i) x3.data[i] += ds.data[i];
            } else {
                for (size_t i = 0; i < x3.numel(); ++i) x3.data[i] += cur.data[i];
            }
            // ReLU
            for (f32& v : x3.data) if (v < 0.f) v = 0.f;

            f32 bmax = *std::max_element(x3.data.begin(), x3.data.end());
            f32 block_scale = (bmax > 1e-7f) ? bmax / 255.f : 1.f;
            block_out_scales_[bi] = std::max(block_out_scales_[bi], block_scale);

            cur = std::move(x3);
        }

        first_sample = false;
    }

    for (f32& s : block_out_scales_)
        if (s <= 1e-7f) s = 1.f;

    // ── Phase 2a: Pack FP32 downsample weights to symmetric int8 ─────────
    // We need a weight scale to fold into oc_mul.  Use symmetric quantization:
    // w_int8 = round(w_fp32 / ds_w_scale), ds_w_scale = max(|w|) / 127.
    std::vector<f32> ds_w_scale(convs_.size(), 0.f);
    for (int bi = 0; bi < N_BLOCKS; ++bi) {
        int ds = RESNET_BLOCKS[bi].ds;
        if (ds < 0 || convs_[ds].is_ternary) continue;
        auto& cp = convs_[ds];
        int K = cp.in_c * cp.kH * cp.kW, K_pad = (K + 63) & ~63;

        f32 maxabs = 0.f;
        for (f32 v : cp.weight_fp32) maxabs = std::max(maxabs, std::abs(v));
        ds_w_scale[ds] = (maxabs > 1e-7f) ? (maxabs / 127.f) : 1.f;

        tc_free(cp.packed_w);
        cp.packed_K = K_pad;
        cp.packed_w = reinterpret_cast<i8*>(tc_alloc((size_t)cp.out_c * K_pad));
        std::memset(cp.packed_w, 0, (size_t)cp.out_c * K_pad);
        cp.w_col_sum.assign(cp.out_c, 0);

        for (int oc = 0; oc < cp.out_c; ++oc) {
            i8* dst = cp.packed_w + (size_t)oc * K_pad;
            i32 sum = 0;
            for (int k = 0; k < K; ++k) {
                int v = (int)std::lroundf(cp.weight_fp32[oc*K + k] / ds_w_scale[ds]);
                v = std::max(-127, std::min(127, v));
                dst[k] = (i8)v;
                sum += v;
            }
            cp.w_col_sum[oc] = sum;
        }
    }

    // ── Phase 2b: Determine input quantization for stem ───────────────────
    stem_in_scale_ = (img_max - img_min > 1e-7f) ? (img_max - img_min) / 255.f : 1.f;
    stem_in_zp_    = std::max(0, std::min(255, (int)std::round(-img_min / stem_in_scale_)));

    // ── Phase 2c: Compute FusedQuantParams per layer ─────────────────────
    // For each output channel oc:
    //   M_oc     = alpha * in_scale * bn_A[oc]
    //   oc_mul   = M_oc / out_scale
    //   oc_add   = (bn_B[oc] - in_zp * w_col_sum[oc] * M_oc) / out_scale + out_zp
    //
    // The zero-point correction term (in_zp * w_col_sum * M_oc) accounts for
    // the fact that unsigned u8 activations include the input zero offset.
    // Without it, the GEMM dot product would be biased by (in_zp * sum_of_weights).
    fqp_.resize(convs_.size());

    auto compute_fqp = [&](int li, f32 in_scale, int in_zp, f32 override_alpha = 0.f) {
        auto& fqp = fqp_[li];
        auto& cp  = convs_[li];

        fqp.in_scale = in_scale;
        fqp.in_zp    = in_zp;

        f32 mn = calib_ranges_[li].mn, mx = calib_ranges_[li].mx;
        if (mn >= -1e-7f) {
            fqp.out_scale = (mx > 1e-7f) ? mx / 255.f : 1.f;
            fqp.out_zp    = 0;
        } else {
            f32 range     = mx - mn;
            fqp.out_scale = (range > 1e-7f) ? range / 255.f : 1.f;
            fqp.out_zp    = std::max(0, std::min(255, (int)std::round(-mn / fqp.out_scale)));
        }

        if (!cp.packed_w || cp.w_col_sum.empty()) { fqp.valid = false; return; }

        int N = cp.out_c;
        fqp.oc_mul.resize(N);
        fqp.oc_add.resize(N);

        f32 alpha = (override_alpha > 0.f) ? override_alpha
                                           : (cp.tw ? cp.tw->alpha : 1.f);
        const auto& bn = bns_[li];
        for (int oc = 0; oc < N; ++oc) {
            f32 bn_a = bn.valid ? bn.A[oc] : 1.f;
            f32 bn_b = bn.valid ? bn.B[oc] : 0.f;
            f32 M_oc = alpha * in_scale * bn_a;
            fqp.oc_mul[oc] = M_oc / fqp.out_scale;
            fqp.oc_add[oc] = (bn_b - (f32)in_zp * (f32)cp.w_col_sum[oc] * M_oc)
                              / fqp.out_scale + (f32)fqp.out_zp;
        }
        fqp.valid = true;
    };

    compute_fqp(0, stem_in_scale_, stem_in_zp_);
    f32 prev_scale = fqp_[0].out_scale;
    int prev_zp    = fqp_[0].out_zp;

    for (int bi = 0; bi < N_BLOCKS; ++bi) {
        const auto& blk = RESNET_BLOCKS[bi];
        int c1 = blk.start, c2 = c1+1, c3 = c1+2;
        compute_fqp(c1, prev_scale, prev_zp);
        compute_fqp(c2, fqp_[c1].out_scale, fqp_[c1].out_zp);
        compute_fqp(c3, fqp_[c2].out_scale, fqp_[c2].out_zp);
        if (blk.ds >= 0) compute_fqp(blk.ds, prev_scale, prev_zp, ds_w_scale[blk.ds]);
        prev_scale = block_out_scales_[bi];
        prev_zp    = 0;  // post-ReLU → unsigned → zp=0
    }

    // ── Phase 3a: Repack weights to NHWC order for implicit im2col ────────
    // 1×1 layers: NCHW == NHWC (K == IC), no repack needed.
    // 3×3 layers: K-axis was [IC][kH][kW] (NCHW GEMM order).
    //             Implicit im2col reads NHWC input rows, so weights must be in
    //             [kH][kW][IC] order.  We permute once here instead of on every
    //             inference.
    nhwc_w_.assign(convs_.size(), nullptr);
    for (size_t li = 0; li < convs_.size(); ++li) {
        auto& cp = convs_[li];
        if (!cp.packed_w || cp.kH <= 1) continue;
        int IC = cp.in_c, kH = cp.kH, kW = cp.kW, K_pad = cp.packed_K, N = cp.out_c;
        i8* nhwc = reinterpret_cast<i8*>(tc_alloc((size_t)N * K_pad));
        std::memset(nhwc, 0, (size_t)N * K_pad);
        for (int oc = 0; oc < N; ++oc) {
            const i8* src = cp.packed_w + (size_t)oc * K_pad;
            i8* dst = nhwc + (size_t)oc * K_pad;
            for (int ic = 0; ic < IC; ++ic)
                for (int dh = 0; dh < kH; ++dh)
                    for (int dw = 0; dw < kW; ++dw)
                        dst[dh * kW * IC + dw * IC + ic] =
                            src[ic * kH * kW + dh * kW + dw];
        }
        nhwc_w_[li] = nhwc;
    }

    // ── Phase 3b: NR=4 panel packing ─────────────────────────────────────
    // Layout: [OC/4][K_pad/32][4][32]
    // Motivation: for each 32-element k-chunk, all 4 OC weight vectors are
    // laid out contiguously (128 bytes = 2 cache lines).  The inner GEBP loop
    // loads wk[NR*0..31], wk[NR*32..63], wk[NR*64..95], wk[NR*96..127] —
    // sequential reads, no strided access, hardware prefetcher works optimally.
    static constexpr int NR = 4;
    for (auto* p : panel_w_) if (p) tc_free(p);
    panel_w_.assign(convs_.size(), nullptr);
    for (size_t li = 0; li < convs_.size(); ++li) {
        const auto& cp = convs_[li];
        const i8* src_w = nhwc_w_[li] ? nhwc_w_[li] : cp.packed_w;
        if (!src_w) continue;
        int N_oc = cp.out_c, K_pad = cp.packed_K;
        int n_oc_tiles = N_oc / NR;
        size_t total = (size_t)n_oc_tiles * K_pad * NR;
        i8* panel = reinterpret_cast<i8*>(tc_alloc(total));
        for (int g = 0; g < n_oc_tiles; ++g)
            for (int k = 0; k + 32 <= K_pad; k += 32) {
                i8* dst = panel + (size_t)g * K_pad * NR + (size_t)(k/32) * NR * 32;
                for (int r = 0; r < NR; ++r)
                    std::memcpy(dst + r*32, src_w + (size_t)(g*NR+r)*K_pad + k, 32);
            }
        panel_w_[li] = panel;
    }

    // ── Pre-allocate QTensor scratch buffers ──────────────────────────────
    // Size to the largest intermediate tensor: 512 channels × 8×8 spatial.
    // All six buffers are reused every inference call — no hot-path allocation.
    size_t max_buf = 0;
    for (const auto& cp : convs_)
        max_buf = std::max(max_buf, (size_t)cp.out_c * 32 * 32);
    q_a_.ensure(1, 1, (int)max_buf);
    q_b_.ensure(1, 1, (int)max_buf);
    q_t1_.ensure(1, 1, (int)max_buf);
    q_t2_.ensure(1, 1, (int)max_buf);
    q_t3_.ensure(1, 1, (int)max_buf);
    q_ds_.ensure(1, 1, (int)max_buf);

    fused_ready_ = true;
    printf("Fused Q2Q pipeline calibrated: %zu layers, %d blocks\n",
           fqp_.size(), N_BLOCKS);
}

// ══════════════════════════════════════════════════════════════════════════
//  CONV Q2Q — GEBP MR=3 NR=4 MICROKERNEL
// ══════════════════════════════════════════════════════════════════════════
//
// This is the performance-critical function.  It implements a General Block
// Panel (GEBP) convolution directly operating on NHWC quantized tensors
// without materialising an im2col buffer.
//
// Register blocking (MR=3, NR=4):
//   - Process 3 output spatial positions (M dimension) simultaneously.
//   - Process 4 output channels (N dimension, == NR) simultaneously.
//   - This gives 12 accumulator registers (3×4 = MR×NR).
//   - Plus 3 activation registers + 1 weight register = 16 YMM total.
//     AVX2 has exactly 16 YMM registers — a perfect fit.
//
// Weight reuse: each weight vector (wk for OC+i) is loaded once per k-step
// and used by all 3 activation vectors.  Loading 1 weight for every 3
// activations is 3× better than loading 1 weight per 1 activation (MR=1).
//
// Three spatial paths are selected based on layer characteristics:
//   1. 1×1 conv: direct NHWC pointer — zero im2col overhead
//   2. 3×3 conv (IC ≥ 64): gather 9 pointers per k-step from NHWC input
//   3. Stem (IC < 64): micro-buffer [27 → 64 bytes] packs 3×3×3 patch first
//
// Parallelism:
//   M-outer (par over spatial groups of MR) when M ≥ threads.
//   OC-tile-outer (par over n_oc_tiles) for small-M late layers.

// Per-channel requantize: apply oc_mul/oc_add and clamp to [relu_min, 255].
// Result written to out[m*N + oc..oc+3] — 4 bytes per spatial position.
static inline void quantize_store_nr4(
    i32 d0, i32 d1, i32 d2, i32 d3,
    f32 mul0, f32 add0, f32 mul1, f32 add1,
    f32 mul2, f32 add2, f32 mul3, f32 add3,
    int relu_min, u8* out, int N, int m, int oc)
{
    // _mm_cvtss_si32 uses SSE round-to-nearest-even, matching numpy behaviour
    int q0 = (int)_mm_cvtss_si32(_mm_set_ss((f32)d0 * mul0 + add0));
    int q1 = (int)_mm_cvtss_si32(_mm_set_ss((f32)d1 * mul1 + add1));
    int q2 = (int)_mm_cvtss_si32(_mm_set_ss((f32)d2 * mul2 + add2));
    int q3 = (int)_mm_cvtss_si32(_mm_set_ss((f32)d3 * mul3 + add3));
    q0 = q0 < relu_min ? relu_min : (q0 > 255 ? 255 : q0);
    q1 = q1 < relu_min ? relu_min : (q1 > 255 ? 255 : q1);
    q2 = q2 < relu_min ? relu_min : (q2 > 255 ? 255 : q2);
    q3 = q3 < relu_min ? relu_min : (q3 > 255 ? 255 : q3);
    out[m * N + oc]     = (u8)q0;
    out[m * N + oc + 1] = (u8)q1;
    out[m * N + oc + 2] = (u8)q2;
    out[m * N + oc + 3] = (u8)q3;
}

// MR=3 × NR=4 VNNI microkernel for 1×1 conv.
// 12 dpbusd accumulators; weight panel is loaded sequentially.
// Software prefetch brings the next 128-byte chunk into L1 before it's needed,
// hiding ~4-cycle L2 latency.
#if HAVE_VNNI
static inline void microkernel_1x1_mr3nr4(
    const u8* a0, const u8* a1, const u8* a2,
    const i8* wp, int K_pad,
    i32 d[3][4])
{
    __m256i acc00 = _mm256_setzero_si256(), acc01 = _mm256_setzero_si256();
    __m256i acc02 = _mm256_setzero_si256(), acc03 = _mm256_setzero_si256();
    __m256i acc10 = _mm256_setzero_si256(), acc11 = _mm256_setzero_si256();
    __m256i acc12 = _mm256_setzero_si256(), acc13 = _mm256_setzero_si256();
    __m256i acc20 = _mm256_setzero_si256(), acc21 = _mm256_setzero_si256();
    __m256i acc22 = _mm256_setzero_si256(), acc23 = _mm256_setzero_si256();

    const i8* wk = wp;
    for (int k = 0; k + 32 <= K_pad; k += 32, wk += 4 * 32) {
        __m256i va0 = _mm256_load_si256((const __m256i*)(a0 + k));
        __m256i va1 = _mm256_load_si256((const __m256i*)(a1 + k));
        __m256i va2 = _mm256_load_si256((const __m256i*)(a2 + k));
        // Prefetch next weight chunk (128 bytes = 4 × 32-byte weight vectors)
        _mm_prefetch((const char*)(wk + 4*32),    _MM_HINT_T0);
        _mm_prefetch((const char*)(wk + 4*32+64), _MM_HINT_T0);
        __m256i w;
        w = _mm256_load_si256((const __m256i*)(wk));       // OC+0
        acc00 = _mm256_dpbusd_epi32(acc00, va0, w);
        acc10 = _mm256_dpbusd_epi32(acc10, va1, w);
        acc20 = _mm256_dpbusd_epi32(acc20, va2, w);
        w = _mm256_load_si256((const __m256i*)(wk + 32));  // OC+1
        acc01 = _mm256_dpbusd_epi32(acc01, va0, w);
        acc11 = _mm256_dpbusd_epi32(acc11, va1, w);
        acc21 = _mm256_dpbusd_epi32(acc21, va2, w);
        w = _mm256_load_si256((const __m256i*)(wk + 64));  // OC+2
        acc02 = _mm256_dpbusd_epi32(acc02, va0, w);
        acc12 = _mm256_dpbusd_epi32(acc12, va1, w);
        acc22 = _mm256_dpbusd_epi32(acc22, va2, w);
        w = _mm256_load_si256((const __m256i*)(wk + 96));  // OC+3
        acc03 = _mm256_dpbusd_epi32(acc03, va0, w);
        acc13 = _mm256_dpbusd_epi32(acc13, va1, w);
        acc23 = _mm256_dpbusd_epi32(acc23, va2, w);
    }
    d[0][0]=hsum_epi32(acc00); d[0][1]=hsum_epi32(acc01);
    d[0][2]=hsum_epi32(acc02); d[0][3]=hsum_epi32(acc03);
    d[1][0]=hsum_epi32(acc10); d[1][1]=hsum_epi32(acc11);
    d[1][2]=hsum_epi32(acc12); d[1][3]=hsum_epi32(acc13);
    d[2][0]=hsum_epi32(acc20); d[2][1]=hsum_epi32(acc21);
    d[2][2]=hsum_epi32(acc22); d[2][3]=hsum_epi32(acc23);
}
#endif // HAVE_VNNI

void TernaryCNN::add_conv_profile(int li, int H_out, int W_out, double us) {
    if (!profiling_enabled_) return;
    profile_.conv_total_us += us;
    const auto& cp = convs_[li];
    uint64_t K = (uint64_t)cp.in_c * cp.kH * cp.kW;
    uint64_t M = (uint64_t)H_out * W_out;
    profile_.conv_ops += (uint64_t)cp.out_c * M * K;
    if (li < (int)profile_.conv_layer_us.size())
        profile_.conv_layer_us[li] += us;
}

void TernaryCNN::conv_q2q(int li, const QTensor& in, QTensor& out, bool do_relu) {
    double t0 = profiling_enabled_ ? now_us() : 0.0;

    const auto& cp  = convs_[li];
    const auto& fqp = fqp_[li];
    if (!fqp.valid) { fprintf(stderr, "conv_q2q[%d]: fqp invalid\n", li); abort(); }

    const int IC    = cp.in_c;
    const int K_pad = cp.packed_K;
    const int H_out = (in.H + 2*cp.padding - cp.kH) / cp.stride + 1;
    const int W_out = (in.W + 2*cp.padding - cp.kW) / cp.stride + 1;
    const int M     = H_out * W_out;
    const int N     = cp.out_c;

    out.ensure(N, H_out, W_out);
    out.scale       = fqp.out_scale;
    out.zero_point  = fqp.out_zp;

    const i8*  W_panel  = panel_w_[li];
    const i8*  W_ptr    = nhwc_w_[li] ? nhwc_w_[li] : cp.packed_w; // fallback
    u8*        O_ptr    = out.data;
    const f32* oc_mul   = fqp.oc_mul.data();
    const f32* oc_add   = fqp.oc_add.data();
    const u8*  I_ptr    = in.data;
    const int  H_in     = in.H, W_in = in.W;
    const u8   pad_val  = (u8)std::max(0, std::min(255, fqp.in_zp));
    const int  relu_min = do_relu ? std::max(0, fqp.out_zp) : 0;

    const int gemm_threads = ((int64_t)N * M >= 4096) ? num_threads_ : 1;
    static constexpr int NR = 4;
    static constexpr int MR = 3;
    const int n_oc_tiles  = N / NR;
    const bool par_over_m = (M >= gemm_threads);

    // ── Path 1: 1×1 conv — direct NHWC pointer ───────────────────────────
    if (cp.kH == 1 && cp.kW == 1) {
        const int S      = cp.stride;
        const int M_body = (M / MR) * MR;

        if (par_over_m) {
            #pragma omp parallel for schedule(static) num_threads(gemm_threads) if(gemm_threads > 1)
            for (int m0 = 0; m0 < M_body; m0 += MR) {
                const u8* a_ptrs[MR];
                for (int r = 0; r < MR; ++r) {
                    int m = m0 + r;
                    int oh = m / W_out, ow = m % W_out;
                    a_ptrs[r] = I_ptr + ((size_t)(oh*S) * W_in + ow*S) * IC;
                }
                for (int oc_t = 0; oc_t < n_oc_tiles; ++oc_t) {
                    int oc = oc_t * NR;
                    const i8* wp = W_panel + (size_t)oc_t * K_pad * NR;
#if HAVE_VNNI
                    i32 d[3][4];
                    microkernel_1x1_mr3nr4(a_ptrs[0], a_ptrs[1], a_ptrs[2], wp, K_pad, d);
                    for (int r = 0; r < MR; ++r)
                        quantize_store_nr4(d[r][0], d[r][1], d[r][2], d[r][3],
                            oc_mul[oc], oc_add[oc], oc_mul[oc+1], oc_add[oc+1],
                            oc_mul[oc+2], oc_add[oc+2], oc_mul[oc+3], oc_add[oc+3],
                            relu_min, O_ptr, N, m0+r, oc);
#else
                    for (int r = 0; r < MR; ++r) {
                        i32 d0=0, d1=0, d2=0, d3=0;
                        for (int k = 0; k < IC; ++k) {
                            i32 a = (i32)a_ptrs[r][k];
                            d0 += a * (i32)W_ptr[(size_t)oc*K_pad+k];
                            d1 += a * (i32)W_ptr[(size_t)(oc+1)*K_pad+k];
                            d2 += a * (i32)W_ptr[(size_t)(oc+2)*K_pad+k];
                            d3 += a * (i32)W_ptr[(size_t)(oc+3)*K_pad+k];
                        }
                        quantize_store_nr4(d0, d1, d2, d3,
                            oc_mul[oc], oc_add[oc], oc_mul[oc+1], oc_add[oc+1],
                            oc_mul[oc+2], oc_add[oc+2], oc_mul[oc+3], oc_add[oc+3],
                            relu_min, O_ptr, N, m0+r, oc);
                    }
#endif
                }
            }
            // M-tail: remainder positions processed one at a time
            for (int m = M_body; m < M; ++m) {
                int oh = m / W_out, ow = m % W_out;
                const u8* ap = I_ptr + ((size_t)(oh*S)*W_in + ow*S) * IC;
                for (int oc_t = 0; oc_t < n_oc_tiles; ++oc_t) {
                    int oc = oc_t * NR;
                    const i8* wp = W_panel + (size_t)oc_t * K_pad * NR;
#if HAVE_VNNI
                    __m256i acc0=_mm256_setzero_si256(), acc1=_mm256_setzero_si256();
                    __m256i acc2=_mm256_setzero_si256(), acc3=_mm256_setzero_si256();
                    const i8* wk = wp;
                    for (int k = 0; k+32 <= K_pad; k+=32, wk+=NR*32) {
                        __m256i a = _mm256_load_si256((const __m256i*)(ap+k));
                        acc0 = _mm256_dpbusd_epi32(acc0, a, _mm256_load_si256((const __m256i*)(wk)));
                        acc1 = _mm256_dpbusd_epi32(acc1, a, _mm256_load_si256((const __m256i*)(wk+32)));
                        acc2 = _mm256_dpbusd_epi32(acc2, a, _mm256_load_si256((const __m256i*)(wk+64)));
                        acc3 = _mm256_dpbusd_epi32(acc3, a, _mm256_load_si256((const __m256i*)(wk+96)));
                    }
                    quantize_store_nr4(hsum_epi32(acc0),hsum_epi32(acc1),
                        hsum_epi32(acc2),hsum_epi32(acc3),
                        oc_mul[oc],oc_add[oc],oc_mul[oc+1],oc_add[oc+1],
                        oc_mul[oc+2],oc_add[oc+2],oc_mul[oc+3],oc_add[oc+3],
                        relu_min, O_ptr, N, m, oc);
#else
                    i32 d0=0,d1=0,d2=0,d3=0;
                    for (int k=0; k<IC; ++k) {
                        i32 a=(i32)ap[k];
                        d0+=a*(i32)W_ptr[(size_t)oc*K_pad+k];
                        d1+=a*(i32)W_ptr[(size_t)(oc+1)*K_pad+k];
                        d2+=a*(i32)W_ptr[(size_t)(oc+2)*K_pad+k];
                        d3+=a*(i32)W_ptr[(size_t)(oc+3)*K_pad+k];
                    }
                    quantize_store_nr4(d0,d1,d2,d3,
                        oc_mul[oc],oc_add[oc],oc_mul[oc+1],oc_add[oc+1],
                        oc_mul[oc+2],oc_add[oc+2],oc_mul[oc+3],oc_add[oc+3],
                        relu_min, O_ptr, N, m, oc);
#endif
                }
            }
        } else {
            // OC-tile-outer parallel: for very small M (layer4: 4 positions)
            #pragma omp parallel for schedule(static) num_threads(gemm_threads) if(gemm_threads > 1)
            for (int oc_t = 0; oc_t < n_oc_tiles; ++oc_t) {
                int oc = oc_t * NR;
                const i8* wp = W_panel + (size_t)oc_t * K_pad * NR;
                f32 qm0=oc_mul[oc],qa0=oc_add[oc],qm1=oc_mul[oc+1],qa1=oc_add[oc+1];
                f32 qm2=oc_mul[oc+2],qa2=oc_add[oc+2],qm3=oc_mul[oc+3],qa3=oc_add[oc+3];
                for (int m = 0; m < M; ++m) {
                    int oh = m/W_out, ow = m%W_out;
                    const u8* ap = I_ptr + ((size_t)(oh*S)*W_in + ow*S)*IC;
#if HAVE_VNNI
                    __m256i acc0=_mm256_setzero_si256(), acc1=_mm256_setzero_si256();
                    __m256i acc2=_mm256_setzero_si256(), acc3=_mm256_setzero_si256();
                    const i8* wk = wp;
                    for (int k=0; k+32<=K_pad; k+=32, wk+=NR*32) {
                        __m256i a = _mm256_load_si256((const __m256i*)(ap+k));
                        acc0=_mm256_dpbusd_epi32(acc0,a,_mm256_load_si256((const __m256i*)(wk)));
                        acc1=_mm256_dpbusd_epi32(acc1,a,_mm256_load_si256((const __m256i*)(wk+32)));
                        acc2=_mm256_dpbusd_epi32(acc2,a,_mm256_load_si256((const __m256i*)(wk+64)));
                        acc3=_mm256_dpbusd_epi32(acc3,a,_mm256_load_si256((const __m256i*)(wk+96)));
                    }
                    quantize_store_nr4(hsum_epi32(acc0),hsum_epi32(acc1),
                        hsum_epi32(acc2),hsum_epi32(acc3),
                        qm0,qa0,qm1,qa1,qm2,qa2,qm3,qa3,relu_min,O_ptr,N,m,oc);
#else
                    i32 d0=0,d1=0,d2=0,d3=0;
                    for (int k=0; k<IC; ++k) {
                        i32 a=(i32)ap[k];
                        d0+=a*(i32)W_ptr[(size_t)oc*K_pad+k];
                        d1+=a*(i32)W_ptr[(size_t)(oc+1)*K_pad+k];
                        d2+=a*(i32)W_ptr[(size_t)(oc+2)*K_pad+k];
                        d3+=a*(i32)W_ptr[(size_t)(oc+3)*K_pad+k];
                    }
                    quantize_store_nr4(d0,d1,d2,d3,
                        qm0,qa0,qm1,qa1,qm2,qa2,qm3,qa3,relu_min,O_ptr,N,m,oc);
#endif
                }
            }
        }

    } else if (IC >= 64) {
        // ── Path 2: 3×3 conv (IC ≥ 64) — gather 9 NHWC row pointers ─────
        // Each kernel position (dh, dw) contributes IC input channels.
        // We read directly from NHWC input: I_ptr[(ih*W_in + iw)*IC + k].
        // ZERO_PAD is used as the activation for out-of-bounds positions.
        alignas(64) static const u8 ZERO_PAD[2048] = {};
        const int kH = cp.kH, kW = cp.kW, S = cp.stride, P = cp.padding;
        const int M_body = (M / MR) * MR;

        if (par_over_m) {
            #pragma omp parallel for schedule(static) num_threads(gemm_threads) if(gemm_threads > 1)
            for (int m0 = 0; m0 < M_body; m0 += MR) {
                for (int oc_t = 0; oc_t < n_oc_tiles; ++oc_t) {
                    int oc = oc_t * NR;
                    const i8* wp = W_panel + (size_t)oc_t * K_pad * NR;
#if HAVE_VNNI
                    __m256i acc00=_mm256_setzero_si256(),acc01=_mm256_setzero_si256();
                    __m256i acc02=_mm256_setzero_si256(),acc03=_mm256_setzero_si256();
                    __m256i acc10=_mm256_setzero_si256(),acc11=_mm256_setzero_si256();
                    __m256i acc12=_mm256_setzero_si256(),acc13=_mm256_setzero_si256();
                    __m256i acc20=_mm256_setzero_si256(),acc21=_mm256_setzero_si256();
                    __m256i acc22=_mm256_setzero_si256(),acc23=_mm256_setzero_si256();
                    const i8* wk = wp;
                    for (int dh = 0; dh < kH; ++dh)
                        for (int dw = 0; dw < kW; ++dw) {
                            const u8* a_ptrs[MR];
                            for (int r = 0; r < MR; ++r) {
                                int m=m0+r, oh=m/W_out, ow=m%W_out;
                                int ih=oh*S-P+dh, iw=ow*S-P+dw;
                                a_ptrs[r]=(ih>=0&&ih<H_in&&iw>=0&&iw<W_in)
                                    ?I_ptr+((size_t)ih*W_in+iw)*IC:ZERO_PAD;
                            }
                            for (int k=0; k+32<=IC; k+=32, wk+=NR*32) {
                                __m256i va0=_mm256_load_si256((const __m256i*)(a_ptrs[0]+k));
                                __m256i va1=_mm256_load_si256((const __m256i*)(a_ptrs[1]+k));
                                __m256i va2=_mm256_load_si256((const __m256i*)(a_ptrs[2]+k));
                                _mm_prefetch((const char*)(wk+NR*32),   _MM_HINT_T0);
                                _mm_prefetch((const char*)(wk+NR*32+64),_MM_HINT_T0);
                                __m256i w;
                                w=_mm256_load_si256((const __m256i*)(wk));
                                acc00=_mm256_dpbusd_epi32(acc00,va0,w);
                                acc10=_mm256_dpbusd_epi32(acc10,va1,w);
                                acc20=_mm256_dpbusd_epi32(acc20,va2,w);
                                w=_mm256_load_si256((const __m256i*)(wk+32));
                                acc01=_mm256_dpbusd_epi32(acc01,va0,w);
                                acc11=_mm256_dpbusd_epi32(acc11,va1,w);
                                acc21=_mm256_dpbusd_epi32(acc21,va2,w);
                                w=_mm256_load_si256((const __m256i*)(wk+64));
                                acc02=_mm256_dpbusd_epi32(acc02,va0,w);
                                acc12=_mm256_dpbusd_epi32(acc12,va1,w);
                                acc22=_mm256_dpbusd_epi32(acc22,va2,w);
                                w=_mm256_load_si256((const __m256i*)(wk+96));
                                acc03=_mm256_dpbusd_epi32(acc03,va0,w);
                                acc13=_mm256_dpbusd_epi32(acc13,va1,w);
                                acc23=_mm256_dpbusd_epi32(acc23,va2,w);
                            }
                        }
                    i32 d[3][4] = {
                        {hsum_epi32(acc00),hsum_epi32(acc01),hsum_epi32(acc02),hsum_epi32(acc03)},
                        {hsum_epi32(acc10),hsum_epi32(acc11),hsum_epi32(acc12),hsum_epi32(acc13)},
                        {hsum_epi32(acc20),hsum_epi32(acc21),hsum_epi32(acc22),hsum_epi32(acc23)},
                    };
                    for (int r=0; r<MR; ++r)
                        quantize_store_nr4(d[r][0],d[r][1],d[r][2],d[r][3],
                            oc_mul[oc],oc_add[oc],oc_mul[oc+1],oc_add[oc+1],
                            oc_mul[oc+2],oc_add[oc+2],oc_mul[oc+3],oc_add[oc+3],
                            relu_min,O_ptr,N,m0+r,oc);
#else
                    for (int r=0; r<MR; ++r) {
                        int m=m0+r, oh=m/W_out, ow=m%W_out;
                        i32 d0=0,d1=0,d2=0,d3=0;
                        const i8* w0=W_ptr+(size_t)oc*K_pad;
                        int w_off=0;
                        for (int dh=0;dh<kH;++dh) { int ih=oh*S-P+dh;
                            for (int dw=0;dw<kW;++dw) { int iw=ow*S-P+dw;
                                const u8* ap=(ih>=0&&ih<H_in&&iw>=0&&iw<W_in)
                                    ?I_ptr+((size_t)ih*W_in+iw)*IC:ZERO_PAD;
                                for (int k=0;k<IC;++k) {
                                    i32 a=(i32)ap[k];
                                    d0+=a*(i32)w0[w_off+k];d1+=a*(i32)w0[K_pad+w_off+k];
                                    d2+=a*(i32)w0[2*K_pad+w_off+k];d3+=a*(i32)w0[3*K_pad+w_off+k];
                                } w_off+=IC; } }
                        quantize_store_nr4(d0,d1,d2,d3,
                            oc_mul[oc],oc_add[oc],oc_mul[oc+1],oc_add[oc+1],
                            oc_mul[oc+2],oc_add[oc+2],oc_mul[oc+3],oc_add[oc+3],
                            relu_min,O_ptr,N,m,oc);
                    }
#endif
                }
            }
            // M tail
            for (int m=M_body; m<M; ++m) {
                int oh=m/W_out, ow=m%W_out;
                for (int oc_t=0; oc_t<n_oc_tiles; ++oc_t) {
                    int oc=oc_t*NR;
                    const i8* wp=W_panel+(size_t)oc_t*K_pad*NR;
#if HAVE_VNNI
                    __m256i acc0=_mm256_setzero_si256(),acc1=_mm256_setzero_si256();
                    __m256i acc2=_mm256_setzero_si256(),acc3=_mm256_setzero_si256();
                    const i8* wk=wp;
                    for (int dh=0;dh<kH;++dh) { int ih=oh*S-P+dh;
                        for (int dw=0;dw<kW;++dw) { int iw=ow*S-P+dw;
                            const u8* ap=(ih>=0&&ih<H_in&&iw>=0&&iw<W_in)
                                ?I_ptr+((size_t)ih*W_in+iw)*IC:ZERO_PAD;
                            for (int k=0;k+32<=IC;k+=32,wk+=NR*32) {
                                __m256i a=_mm256_load_si256((const __m256i*)(ap+k));
                                acc0=_mm256_dpbusd_epi32(acc0,a,_mm256_load_si256((const __m256i*)(wk)));
                                acc1=_mm256_dpbusd_epi32(acc1,a,_mm256_load_si256((const __m256i*)(wk+32)));
                                acc2=_mm256_dpbusd_epi32(acc2,a,_mm256_load_si256((const __m256i*)(wk+64)));
                                acc3=_mm256_dpbusd_epi32(acc3,a,_mm256_load_si256((const __m256i*)(wk+96)));
                            }
                        }
                    }
                    quantize_store_nr4(hsum_epi32(acc0),hsum_epi32(acc1),
                        hsum_epi32(acc2),hsum_epi32(acc3),
                        oc_mul[oc],oc_add[oc],oc_mul[oc+1],oc_add[oc+1],
                        oc_mul[oc+2],oc_add[oc+2],oc_mul[oc+3],oc_add[oc+3],
                        relu_min,O_ptr,N,m,oc);
#else
                    i32 d0=0,d1=0,d2=0,d3=0;
                    const i8* w0=W_ptr+(size_t)oc*K_pad; int w_off=0;
                    for (int dh=0;dh<kH;++dh){int ih=oh*S-P+dh;
                        for (int dw=0;dw<kW;++dw){int iw=ow*S-P+dw;
                            const u8* ap=(ih>=0&&ih<H_in&&iw>=0&&iw<W_in)
                                ?I_ptr+((size_t)ih*W_in+iw)*IC:ZERO_PAD;
                            for (int k=0;k<IC;++k){
                                i32 a=(i32)ap[k];
                                d0+=a*(i32)w0[w_off+k];d1+=a*(i32)w0[K_pad+w_off+k];
                                d2+=a*(i32)w0[2*K_pad+w_off+k];d3+=a*(i32)w0[3*K_pad+w_off+k];
                            } w_off+=IC; } }
                    quantize_store_nr4(d0,d1,d2,d3,
                        oc_mul[oc],oc_add[oc],oc_mul[oc+1],oc_add[oc+1],
                        oc_mul[oc+2],oc_add[oc+2],oc_mul[oc+3],oc_add[oc+3],
                        relu_min,O_ptr,N,m,oc);
#endif
                }
            }
        } else {
            // OC-tile-outer for small-M 3×3 layers (layer4 spatial: 2×2 = 4 positions)
            #pragma omp parallel for schedule(static) num_threads(gemm_threads) if(gemm_threads > 1)
            for (int oc_t=0; oc_t<n_oc_tiles; ++oc_t) {
                int oc=oc_t*NR;
                const i8* wp=W_panel+(size_t)oc_t*K_pad*NR;
                f32 qm0=oc_mul[oc],qa0=oc_add[oc];
                f32 qm1=oc_mul[oc+1],qa1=oc_add[oc+1];
                f32 qm2=oc_mul[oc+2],qa2=oc_add[oc+2];
                f32 qm3=oc_mul[oc+3],qa3=oc_add[oc+3];
                for (int m=0; m<M; ++m) {
                    int oh=m/W_out, ow=m%W_out;
#if HAVE_VNNI
                    __m256i acc0=_mm256_setzero_si256(),acc1=_mm256_setzero_si256();
                    __m256i acc2=_mm256_setzero_si256(),acc3=_mm256_setzero_si256();
                    const i8* wk=wp;
                    for (int dh=0;dh<kH;++dh){int ih=oh*S-P+dh;
                        for (int dw=0;dw<kW;++dw){int iw=ow*S-P+dw;
                            const u8* ap=(ih>=0&&ih<H_in&&iw>=0&&iw<W_in)
                                ?I_ptr+((size_t)ih*W_in+iw)*IC:ZERO_PAD;
                            for (int k=0;k+32<=IC;k+=32,wk+=NR*32){
                                __m256i a=_mm256_load_si256((const __m256i*)(ap+k));
                                acc0=_mm256_dpbusd_epi32(acc0,a,_mm256_load_si256((const __m256i*)(wk)));
                                acc1=_mm256_dpbusd_epi32(acc1,a,_mm256_load_si256((const __m256i*)(wk+32)));
                                acc2=_mm256_dpbusd_epi32(acc2,a,_mm256_load_si256((const __m256i*)(wk+64)));
                                acc3=_mm256_dpbusd_epi32(acc3,a,_mm256_load_si256((const __m256i*)(wk+96)));
                            }
                        }
                    }
                    quantize_store_nr4(hsum_epi32(acc0),hsum_epi32(acc1),
                        hsum_epi32(acc2),hsum_epi32(acc3),
                        qm0,qa0,qm1,qa1,qm2,qa2,qm3,qa3,relu_min,O_ptr,N,m,oc);
#else
                    i32 d0=0,d1=0,d2=0,d3=0;
                    const i8* w0=W_ptr+(size_t)oc*K_pad; int w_off=0;
                    for (int dh=0;dh<kH;++dh){int ih=oh*S-P+dh;
                        for (int dw=0;dw<kW;++dw){int iw=ow*S-P+dw;
                            const u8* ap=(ih>=0&&ih<H_in&&iw>=0&&iw<W_in)
                                ?I_ptr+((size_t)ih*W_in+iw)*IC:ZERO_PAD;
                            for (int k=0;k<IC;++k){
                                i32 a=(i32)ap[k];
                                d0+=a*(i32)w0[w_off+k];d1+=a*(i32)w0[K_pad+w_off+k];
                                d2+=a*(i32)w0[2*K_pad+w_off+k];d3+=a*(i32)w0[3*K_pad+w_off+k];
                            } w_off+=IC; } }
                    quantize_store_nr4(d0,d1,d2,d3,
                        qm0,qa0,qm1,qa1,qm2,qa2,qm3,qa3,relu_min,O_ptr,N,m,oc);
#endif
                }
            }
        }

    } else {
        // ── Path 3: Stem (IC=3, 3×3) — micro-buffer ──────────────────────
        // Stem has only IC=3 channels, so the 3×3 patch is 27 bytes.
        // We pack it into a 64-byte aligned micro[64] buffer that fits in
        // one cache line.  This allows the VNNI loop to use aligned loads.
        // MR=1 here (each output position gets its own micro-buffer; the
        // gain from MR=3 is small since IC=3 is dominated by loop overhead).
        #pragma omp parallel for schedule(static) num_threads(gemm_threads) if(gemm_threads > 1)
        for (int m = 0; m < M; ++m) {
            int oh=m/W_out, ow=m%W_out;
            alignas(64) u8 micro[64] = {};
            int idx = 0;
            for (int dh=0; dh<cp.kH; ++dh) {
                int ih=oh*cp.stride-cp.padding+dh;
                for (int dw=0; dw<cp.kW; ++dw) {
                    int iw=ow*cp.stride-cp.padding+dw;
                    if (ih>=0&&ih<H_in&&iw>=0&&iw<W_in) {
                        const u8* src=I_ptr+((size_t)ih*W_in+iw)*IC;
                        for (int c=0; c<IC; ++c) micro[idx++]=src[c];
                    } else {
                        for (int c=0; c<IC; ++c) micro[idx++]=pad_val;
                    }
                }
            }
            for (int oc_t=0; oc_t<n_oc_tiles; ++oc_t) {
                int oc=oc_t*NR;
                const i8* wp=W_panel+(size_t)oc_t*K_pad*NR;
#if HAVE_VNNI
                __m256i acc0=_mm256_setzero_si256(),acc1=_mm256_setzero_si256();
                __m256i acc2=_mm256_setzero_si256(),acc3=_mm256_setzero_si256();
                const i8* wk=wp;
                for (int k=0; k+32<=K_pad; k+=32, wk+=NR*32) {
                    __m256i a=_mm256_load_si256((const __m256i*)(micro+k));
                    acc0=_mm256_dpbusd_epi32(acc0,a,_mm256_load_si256((const __m256i*)(wk)));
                    acc1=_mm256_dpbusd_epi32(acc1,a,_mm256_load_si256((const __m256i*)(wk+32)));
                    acc2=_mm256_dpbusd_epi32(acc2,a,_mm256_load_si256((const __m256i*)(wk+64)));
                    acc3=_mm256_dpbusd_epi32(acc3,a,_mm256_load_si256((const __m256i*)(wk+96)));
                }
                quantize_store_nr4(hsum_epi32(acc0),hsum_epi32(acc1),
                    hsum_epi32(acc2),hsum_epi32(acc3),
                    oc_mul[oc],oc_add[oc],oc_mul[oc+1],oc_add[oc+1],
                    oc_mul[oc+2],oc_add[oc+2],oc_mul[oc+3],oc_add[oc+3],
                    relu_min,O_ptr,N,m,oc);
#else
                i32 d0=0,d1=0,d2=0,d3=0;
                const i8* w0=W_ptr+(size_t)oc*K_pad;
                const i8* w1=w0+K_pad, *w2=w1+K_pad, *w3=w2+K_pad;
                for (int k=0; k<idx; ++k) {
                    i32 a=(i32)micro[k];
                    d0+=a*(i32)w0[k]; d1+=a*(i32)w1[k];
                    d2+=a*(i32)w2[k]; d3+=a*(i32)w3[k];
                }
                quantize_store_nr4(d0,d1,d2,d3,
                    oc_mul[oc],oc_add[oc],oc_mul[oc+1],oc_add[oc+1],
                    oc_mul[oc+2],oc_add[oc+2],oc_mul[oc+3],oc_add[oc+3],
                    relu_min,O_ptr,N,m,oc);
#endif
            }
        }
    }

    if (profiling_enabled_)
        add_conv_profile(li, H_out, W_out, now_us() - t0);
}

// ══════════════════════════════════════════════════════════════════════════
//  Q_ADD_RELU — fused residual add + ReLU + requantize
// ══════════════════════════════════════════════════════════════════════════
// Computes:  out[i] = clamp(round(a[i]*mul_a + b[i]*mul_b + bias), 0, 255)
// Where mul_a = a.scale/out_scale, mul_b = b.scale/out_scale, post-ReLU so
// out.zero_point = 0.
//
// AVX2 path: process 32 elements/iteration using cvtepu8_epi32 + fmadd_ps +
// pack round-trip (epi32→epi16→epu8 within 256-bit).
void TernaryCNN::q_add_relu(const QTensor& a, const QTensor& b,
                             QTensor& out, f32 out_scale) {
    size_t n = a.numel();
    out.ensure(a.C, a.H, a.W);
    out.scale       = out_scale;
    out.zero_point  = 0;  // post-ReLU → values are non-negative

    f32 mul_a = a.scale / out_scale;
    f32 mul_b = b.scale / out_scale;
    // Zero-point correction: the u8 values include offsets zp_a and zp_b
    f32 bias  = (-(f32)a.zero_point * a.scale - (f32)b.zero_point * b.scale) / out_scale;

#if HAVE_AVX2
    __m256 vmul_a = _mm256_set1_ps(mul_a);
    __m256 vmul_b = _mm256_set1_ps(mul_b);
    __m256 vbias  = _mm256_set1_ps(bias);
    __m256 vzero  = _mm256_setzero_ps();
    __m256 v255   = _mm256_set1_ps(255.f);
    size_t i = 0;
    // Process 32 u8 elements per iteration (4 groups of 8)
    for (; i + 32 <= n; i += 32) {
        __m256i ri32_0, ri32_1, ri32_2, ri32_3;
        for (int g = 0; g < 4; ++g) {
            __m256i a32 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)&a.data[i + g*8]));
            __m256i b32 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)&b.data[i + g*8]));
            __m256 res  = _mm256_fmadd_ps(_mm256_cvtepi32_ps(a32), vmul_a,
                              _mm256_fmadd_ps(_mm256_cvtepi32_ps(b32), vmul_b, vbias));
            res = _mm256_min_ps(_mm256_max_ps(res, vzero), v255);
            __m256i r = _mm256_cvtps_epi32(res);
            if      (g==0) ri32_0=r;
            else if (g==1) ri32_1=r;
            else if (g==2) ri32_2=r;
            else           ri32_3=r;
        }
        // Pack 4×8 i32 → 32 u8 via i32→i16→u8
        __m256i lo16 = _mm256_packs_epi32(ri32_0, ri32_1);
        __m256i hi16 = _mm256_packs_epi32(ri32_2, ri32_3);
        lo16 = _mm256_permute4x64_epi64(lo16, 0xD8);
        hi16 = _mm256_permute4x64_epi64(hi16, 0xD8);
        __m256i packed = _mm256_packus_epi16(lo16, hi16);
        packed = _mm256_permute4x64_epi64(packed, 0xD8);
        _mm256_storeu_si256((__m256i*)&out.data[i], packed);
    }
    for (; i + 8 <= n; i += 8) {
        __m256i a32 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)&a.data[i]));
        __m256i b32 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)&b.data[i]));
        __m256 res  = _mm256_fmadd_ps(_mm256_cvtepi32_ps(a32), vmul_a,
                          _mm256_fmadd_ps(_mm256_cvtepi32_ps(b32), vmul_b, vbias));
        res = _mm256_min_ps(_mm256_max_ps(res, vzero), v255);
        __m256i ri32 = _mm256_cvtps_epi32(res);
        __m256i ri16 = _mm256_permute4x64_epi64(_mm256_packs_epi32(ri32, ri32), 0xD8);
        __m256i ri8  = _mm256_permute4x64_epi64(_mm256_packus_epi16(ri16, ri16), 0xD8);
        _mm_storel_epi64((__m128i*)&out.data[i], _mm256_castsi256_si128(ri8));
    }
    for (; i < n; ++i) {
        f32 val = (f32)a.data[i]*mul_a + (f32)b.data[i]*mul_b + bias;
        out.data[i] = (u8)std::max(0.f, std::min(255.f, std::round(val)));
    }
#else
    for (size_t i = 0; i < n; ++i) {
        f32 val = (f32)a.data[i]*mul_a + (f32)b.data[i]*mul_b + bias;
        out.data[i] = (u8)std::max(0.f, std::min(255.f, std::round(val)));
    }
#endif
}

// ══════════════════════════════════════════════════════════════════════════
//  BOTTLENECK Q2Q
// ══════════════════════════════════════════════════════════════════════════
// One full ResNet-50 bottleneck block:
//   x1 = relu(bn(conv1×1(in)))            — reduce channels
//   x2 = relu(bn(conv3×3(x1)))            — spatial transform
//   x3 = bn(conv1×1(x2))                  — expand channels (no relu yet)
//   if downsample: ds = bn(conv_ds(in))
//   out = relu(x3 + (ds or in))           — residual + relu, all in u8
void TernaryCNN::bottleneck_q2q(const QTensor& in, QTensor& out,
                                 int conv_start, int ds_idx,
                                 f32 block_out_scale) {
    const int c1=conv_start, c2=c1+1, c3=c1+2;
    conv_q2q(c1, in,   q_t1_, true);   // conv1 + BN + ReLU → q_t1_
    conv_q2q(c2, q_t1_,q_t2_, true);   // conv2 + BN + ReLU → q_t2_
    conv_q2q(c3, q_t2_,q_t3_, false);  // conv3 + BN (no ReLU) → q_t3_
    if (ds_idx >= 0) {
        conv_q2q(ds_idx, in, q_ds_, false);
        q_add_relu(q_t3_, q_ds_, out, block_out_scale);
    } else {
        q_add_relu(q_t3_, in, out, block_out_scale);
    }
}

// ══════════════════════════════════════════════════════════════════════════
//  INFER FUSED — main entry point
// ══════════════════════════════════════════════════════════════════════════
// Data flow:
//   CHW float → NHWC u8 (one-time conv at entry)
//   → conv_q2q (stem) → [16× bottleneck_q2q]
//   → GAP (dequantize + accumulate + divide) → FC → logits
//
// GAP and FC use AVX2 FP32: they are bandwidth-bound, not compute-bound,
// so VNNI doesn't help here.  AVX2 FMA processes 8 f32/cycle.
std::vector<f32> TernaryCNN::infer_fused(const f32* img_chw, int C, int H, int W) {
    if (!fused_ready_) {
        fprintf(stderr, "infer_fused: call calibrate_fused() first\n");
        return {};
    }

    pin_threads_to_pcores(num_threads_);
    if (profiling_enabled_) reset_profile();
    double t_total0 = profiling_enabled_ ? now_us() : 0.0;

    // ── Quantize CHW float → NHWC u8 ─────────────────────────────────────
    // This is the ONLY float→u8 conversion in the entire pipeline.
    // All subsequent layers work entirely in u8 domain.
    {
        q_a_.ensure(C, H, W);
        q_a_.scale       = stem_in_scale_;
        q_a_.zero_point  = stem_in_zp_;
        f32 inv_s = 1.f / q_a_.scale;
        f32 zp_f  = (f32)q_a_.zero_point;
        for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w) {
                u8* dst = q_a_.data + (h * W + w) * C;
                for (int c = 0; c < C; ++c) {
                    int v = (int)std::lroundf(img_chw[c*H*W + h*W + w] * inv_s + zp_f);
                    dst[c] = (u8)std::max(0, std::min(255, v));
                }
            }
    }

    QTensor* cur = &q_a_;
    QTensor* nxt = &q_b_;

    // ── Stem ─────────────────────────────────────────────────────────────
    conv_q2q(0, *cur, *nxt, true);
    std::swap(cur, nxt);

    // ── 16 bottleneck blocks ──────────────────────────────────────────────
    for (int bi = 0; bi < N_BLOCKS; ++bi) {
        bottleneck_q2q(*cur, *nxt,
                       RESNET_BLOCKS[bi].start, RESNET_BLOCKS[bi].ds,
                       block_out_scales_[bi]);
        std::swap(cur, nxt);
    }

    // ── Global Average Pooling ─────────────────────────────────────────────
    // Dequantize NHWC u8 → f32 and accumulate, then divide by H*W.
    // AVX2: cvtepu8_epi32 × 8 elements per cycle.
    double t_gap0 = profiling_enabled_ ? now_us() : 0.0;
    const int HW = cur->H * cur->W, Cf = cur->C;
    std::vector<f32> pooled(Cf, 0.f);
    f32 dq_scale = cur->scale, dq_zp_f = (f32)cur->zero_point;
#if HAVE_AVX2
    {
        __m256 v_scale = _mm256_set1_ps(dq_scale), v_zp = _mm256_set1_ps(dq_zp_f);
        for (int hw = 0; hw < HW; ++hw) {
            const u8* row = cur->data + hw * Cf;
            int c = 0;
            for (; c + 8 <= Cf; c += 8) {
                __m256 fv = _mm256_sub_ps(
                    _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(
                        _mm_loadl_epi64((const __m128i*)(row+c)))), v_zp);
                _mm256_storeu_ps(&pooled[c],
                    _mm256_fmadd_ps(fv, v_scale, _mm256_loadu_ps(&pooled[c])));
            }
            for (; c < Cf; ++c) pooled[c] += ((f32)row[c] - dq_zp_f) * dq_scale;
        }
        f32 inv_hw = 1.f / (f32)HW;
        __m256 v_inv = _mm256_set1_ps(inv_hw);
        int c = 0;
        for (; c + 8 <= Cf; c += 8)
            _mm256_storeu_ps(&pooled[c], _mm256_mul_ps(_mm256_loadu_ps(&pooled[c]), v_inv));
        for (; c < Cf; ++c) pooled[c] *= inv_hw;
    }
#else
    for (int hw=0;hw<HW;++hw) {
        const u8* row=cur->data+hw*Cf;
        for (int c=0;c<Cf;++c) pooled[c]+=((f32)row[c]-dq_zp_f)*dq_scale;
    }
    for (int c=0;c<Cf;++c) pooled[c]/=(f32)HW;
#endif
    if (profiling_enabled_) profile_.gap_us += now_us() - t_gap0;

    // ── Fully-Connected layer ─────────────────────────────────────────────
    // FP32 GEMV: [out_f × in_f] × [in_f] → [out_f].
    // AVX2 FMA kernel, 2× unrolled.
    double t_fc0 = profiling_enabled_ ? now_us() : 0.0;
    std::vector<f32> logits(fc_.out_f, 0.f);
    const f32* fc_w  = fc_.weight.data();
    const f32* fc_b  = fc_.has_bias ? fc_.bias.data() : nullptr;
    const f32* pool_p = pooled.data();
    const int   in_f  = fc_.in_f;
    for (int o = 0; o < fc_.out_f; ++o) {
        const f32* wrow = fc_w + o * in_f;
        f32 s = fc_b ? fc_b[o] : 0.f;
#if HAVE_AVX2
        __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
        int j = 0;
        for (; j + 16 <= in_f; j += 16) {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(wrow+j),   _mm256_loadu_ps(pool_p+j),   acc0);
            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(wrow+j+8), _mm256_loadu_ps(pool_p+j+8), acc1);
        }
        for (; j + 8 <= in_f; j += 8)
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(wrow+j), _mm256_loadu_ps(pool_p+j), acc0);
        acc0 = _mm256_add_ps(acc0, acc1);
        __m128 lo = _mm256_castps256_ps128(acc0), hi = _mm256_extractf128_ps(acc0, 1);
        __m128 sum4 = _mm_add_ps(lo, hi);
        sum4 = _mm_hadd_ps(sum4, sum4); sum4 = _mm_hadd_ps(sum4, sum4);
        s += _mm_cvtss_f32(sum4);
        for (; j < in_f; ++j) s += wrow[j] * pool_p[j];
#else
        for (int j=0; j<in_f; ++j) s += wrow[j] * pool_p[j];
#endif
        logits[o] = s;
    }
    if (profiling_enabled_) {
        profile_.fc_us   += now_us() - t_fc0;
        profile_.fc_ops  += (uint64_t)fc_.out_f * fc_.in_f;
        profile_.total_ops = profile_.conv_ops + profile_.fc_ops;
        profile_.total_us  = now_us() - t_total0;
    }
    return logits;
}

} // namespace te
