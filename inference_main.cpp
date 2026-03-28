// inference_main.cpp — CIFAR-10 batch inference demo
// =====================================================
// Loads a ternary ResNet-50 model, calibrates the Q2Q pipeline, then runs
// inference over the full 10 000-image CIFAR-10 test set.
//
// Usage:
//   ./inference_demo <weights.bin> <cifar_test_batch.bin>
//                    [num_threads [max_images [calib_images]]]
//
// weights.bin       — produced by export_trn3_engine.py (TRN3 format)
// cifar_test_batch.bin — CIFAR-10 test_batch binary (standard distribution)
// num_threads       — optional, default 8 (7 P-cores recommended on 13700HX)

#include "engine.h"

#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <chrono>
#include <numeric>
#include <algorithm>

using namespace te;

// ── CIFAR-10 normalisation (CHW layout, standard binary format) ──────────
// Standard CIFAR-10 binary: 3072 bytes per image in CHW order
//   [R0..R1023, G0..G1023, B0..B1023]
// Mirrors torchvision.transforms.Normalize([0.4914,0.4822,0.4465],[0.2023,0.1994,0.2010])
static std::vector<f32> normalise_chw(const u8* chw_u8) {
    static const f32 mean[3] = {0.4914f, 0.4822f, 0.4465f};
    static const f32 std_[3] = {0.2023f, 0.1994f, 0.2010f};
    std::vector<f32> out(3 * 32 * 32);
    for (int c = 0; c < 3; c++)
        for (int i = 0; i < 1024; i++)
            out[c * 1024 + i] = (chw_u8[c * 1024 + i] / 255.f - mean[c]) / std_[c];
    return out;
}

// ── CIFAR-10 binary batch reader ─────────────────────────────────────────
// Format: each record = 1 byte label + 3072 bytes image (RGB, 32×32, CHW)
struct CifarBatch {
    std::vector<u8>  images;  // [N × 3072] row-major
    std::vector<int> labels;
    int N = 0;
};

static CifarBatch load_cifar_batch(const char* path) {
    CifarBatch b;
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return b; }

    if (fseek(f, 0, SEEK_END) != 0) {
        fprintf(stderr, "Cannot seek %s\n", path);
        fclose(f);
        return b;
    }
    long fsize = ftell(f);
    if (fsize < 0) {
        fprintf(stderr, "Cannot stat %s\n", path);
        fclose(f);
        return b;
    }
    rewind(f);

    constexpr int RECORD = 1 + 3072;
    if (fsize % RECORD != 0) {
        fprintf(stderr, "Invalid CIFAR binary size for %s\n", path);
        fclose(f);
        return b;
    }
    int n = (int)(fsize / RECORD);
    b.N = n;
    b.labels.resize(n);
    b.images.resize((size_t)n * 3072);

    for (int i = 0; i < n; ++i) {
        u8 label_byte;
        if (fread(&label_byte, 1, 1, f) != 1) {
            fprintf(stderr, "Failed reading label %d from %s\n", i, path);
            fclose(f);
            b = CifarBatch{};
            return b;
        }
        b.labels[i] = (int)label_byte;
        if (fread(b.images.data() + (size_t)i * 3072, 1, 3072, f) != 3072) {
            fprintf(stderr, "Failed reading image %d from %s\n", i, path);
            fclose(f);
            b = CifarBatch{};
            return b;
        }
    }
    fclose(f);
    printf("Loaded %d images from %s\n", n, path);
    return b;
}

// ── Main ──────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr,
            "Usage: %s <weights.bin> <cifar_test_batch.bin> [num_threads [max_images [calib_images]]]\n",
            argv[0]);
        return 1;
    }

    const char* weight_path = argv[1];
    const char* cifar_path  = argv[2];
    int         n_threads   = (argc >= 4) ? std::stoi(argv[3]) : 8;
    int         max_images  = (argc >= 5) ? std::stoi(argv[4]) : -1;  // -1 = all
    int         calib_images = (argc >= 6) ? std::stoi(argv[5]) : 64;

    // ── Load model ────────────────────────────────────────────────────────
    TernaryCNN engine;
    engine.set_num_threads(n_threads);
    printf("Loading model from %s\n", weight_path);
    if (!engine.load(weight_path)) {
        fprintf(stderr, "Failed to load weights\n");
        return 1;
    }

    // ── Load CIFAR-10 test batch ──────────────────────────────────────────
    CifarBatch batch = load_cifar_batch(cifar_path);
    if (batch.N == 0) { fprintf(stderr, "Empty CIFAR batch\n"); return 1; }

    // ── Calibrate with a representative subset ────────────────────────────
    int n_calib = std::max(1, std::min(calib_images, batch.N));
    printf("Calibrating fused Q2Q pipeline with %d image(s)...\n", n_calib);
    {
        const size_t CHW = 3u * 32u * 32u;
        std::vector<f32> calib_buf((size_t)n_calib * CHW);
        for (int i = 0; i < n_calib; ++i) {
            const u8* img_chw = batch.images.data() + (size_t)i * 3072;
            std::vector<f32> chw = normalise_chw(img_chw);
            std::memcpy(calib_buf.data() + (size_t)i * CHW, chw.data(), sizeof(f32) * CHW);
        }
        engine.calibrate_fused(calib_buf.data(), 3, 32, 32, n_calib);
    }
    printf("Calibration done.\n\n");

    // ── Warm-up run (not timed) ───────────────────────────────────────────
    {
        std::vector<f32> chw = normalise_chw(batch.images.data());
        auto _ = engine.infer_fused(chw.data(), 3, 32, 32);
        (void)_;
    }

    // ── Benchmark: 100 consecutive inferences on the first image ─────────
    printf("Benchmarking (100 runs, %d thread%s)...\n",
           n_threads, n_threads > 1 ? "s" : "");
    {
        std::vector<f32> chw = normalise_chw(batch.images.data());
        const int BENCH_N = 100;
        std::vector<double> lat_ms(BENCH_N);
        for (int r = 0; r < BENCH_N; ++r) {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto logits = engine.infer_fused(chw.data(), 3, 32, 32);
            auto t1 = std::chrono::high_resolution_clock::now();
            lat_ms[r] = std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        std::sort(lat_ms.begin(), lat_ms.end());
        double sum = std::accumulate(lat_ms.begin(), lat_ms.end(), 0.0);
        printf("  Latency (ms):  min=%.3f  p50=%.3f  p95=%.3f  mean=%.3f\n",
               lat_ms[0], lat_ms[BENCH_N/2], lat_ms[(int)(BENCH_N*0.95)],
               sum / BENCH_N);
    }

    // ── Accuracy over test set ────────────────────────────────────────────
    int n_eval = (max_images > 0) ? std::min(max_images, batch.N) : batch.N;
    printf("\nRunning inference on %d images...\n", n_eval);
    int correct = 0;
    double total_ms = 0.0;

    for (int i = 0; i < n_eval; ++i) {
        const u8* img_chw = batch.images.data() + (size_t)i * 3072;
        std::vector<f32> chw = normalise_chw(img_chw);

        auto t0 = std::chrono::high_resolution_clock::now();
        auto logits = engine.infer_fused(chw.data(), 3, 32, 32);
        auto t1 = std::chrono::high_resolution_clock::now();
        total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

        // argmax
        int pred = (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
        if (pred == batch.labels[i]) ++correct;
    }

    float acc = 100.f * (float)correct / (float)n_eval;
    printf("\nResults:\n");
    printf("  Accuracy:         %d / %d  (%.2f%%)\n", correct, n_eval, acc);
    printf("  Avg latency/img:  %.3f ms\n", total_ms / n_eval);
    printf("  Total time:       %.1f ms\n", total_ms);
    printf("  Throughput:       %.1f img/s\n", 1000.0 * n_eval / total_ms);

    return 0;
}
