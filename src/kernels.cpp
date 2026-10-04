// numdiff kernels: small numerical kernels, several source-level variants each.
// Usage: runner <kernel/variant[+bf16]> <in.bin> <out.bin> [reps]
// in.bin / out.bin are raw little-endian float32 arrays. Prints "time_ns=<min over reps>".
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

using Vec = std::vector<float>;
using Fn = Vec (*)(const Vec &);

// ---------- helpers ----------
static float pairwise(const float *p, size_t n) {
    if (n <= 32) { float s = 0.f; for (size_t i = 0; i < n; i++) s += p[i]; return s; }
    size_t h = n / 2;
    return pairwise(p, h) + pairwise(p + h, n - h);
}
static float round_bf16(float x) {  // round-to-nearest-even to bfloat16, back to fp32
    uint32_t u; std::memcpy(&u, &x, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return x;  // NaN
    u += 0x7fffu + ((u >> 16) & 1u);
    u &= 0xffff0000u;
    float y; std::memcpy(&y, &u, 4); return y;
}

// ---------- sum ----------
static Vec sum_naive(const Vec &x) { float s = 0.f; for (float v : x) s += v; return {s}; }
static Vec sum_unroll4(const Vec &x) {
    float a = 0, b = 0, c = 0, d = 0; size_t i = 0, n = x.size();
    for (; i + 4 <= n; i += 4) { a += x[i]; b += x[i+1]; c += x[i+2]; d += x[i+3]; }
    for (; i < n; i++) a += x[i];
    return {(a + b) + (c + d)};
}
static Vec sum_pairwise(const Vec &x) { return {pairwise(x.data(), x.size())}; }

// ---------- dot (input = a ++ b) ----------
static Vec dot_naive(const Vec &x) {
    size_t n = x.size() / 2; float s = 0.f;
    for (size_t i = 0; i < n; i++) s += x[i] * x[n + i];
    return {s};
}
static Vec dot_unroll4(const Vec &x) {
    size_t n = x.size() / 2, i = 0; float a = 0, b = 0, c = 0, d = 0;
    for (; i + 4 <= n; i += 4) {
        a += x[i] * x[n+i]; b += x[i+1] * x[n+i+1]; c += x[i+2] * x[n+i+2]; d += x[i+3] * x[n+i+3];
    }
    for (; i < n; i++) a += x[i] * x[n + i];
    return {(a + b) + (c + d)};
}
static Vec dot_pairwise(const Vec &x) {
    size_t n = x.size() / 2; Vec p(n);
    for (size_t i = 0; i < n; i++) p[i] = x[i] * x[n + i];
    return {pairwise(p.data(), n)};
}

// ---------- softmax ----------
static Vec softmax_naive(const Vec &x) {
    float m = -INFINITY; for (float v : x) m = v > m ? v : m;
    Vec y(x.size()); float s = 0.f;
    for (size_t i = 0; i < x.size(); i++) { y[i] = std::exp(x[i] - m); s += y[i]; }
    for (float &v : y) v /= s;
    return y;
}
static Vec softmax_online(const Vec &x) {  // single-pass running max + rescaled sum
    float m = -INFINITY, s = 0.f;
    for (float v : x) {
        float nm = v > m ? v : m;
        s = s * std::exp(m - nm) + std::exp(v - nm);
        m = nm;
    }
    Vec y(x.size());
    for (size_t i = 0; i < x.size(); i++) y[i] = std::exp(x[i] - m) / s;
    return y;
}

// ---------- layernorm (no affine, eps = 1e-5) ----------
static Vec layernorm_naive(const Vec &x) {
    size_t n = x.size(); float mean = 0.f;
    for (float v : x) mean += v;
    mean /= (float)n;
    float var = 0.f; for (float v : x) var += (v - mean) * (v - mean);
    var /= (float)n;
    float inv = 1.0f / std::sqrt(var + 1e-5f);
    Vec y(n); for (size_t i = 0; i < n; i++) y[i] = (x[i] - mean) * inv;
    return y;
}
static Vec layernorm_welford(const Vec &x) {
    size_t n = x.size(); float mean = 0.f, m2 = 0.f; size_t k = 0;
    for (float v : x) { k++; float d = v - mean; mean += d / (float)k; m2 += d * (v - mean); }
    float inv = 1.0f / std::sqrt(m2 / (float)n + 1e-5f);
    Vec y(n); for (size_t i = 0; i < n; i++) y[i] = (x[i] - mean) * inv;
    return y;
}

// ---------- inclusive prefix sum ----------
static Vec prefix_naive(const Vec &x) {
    Vec y(x.size()); float s = 0.f;
    for (size_t i = 0; i < x.size(); i++) { s += x[i]; y[i] = s; }
    return y;
}
static Vec prefix_blocked(const Vec &x) {  // local scans per 64-block, then add carried offsets
    const size_t B = 64; size_t n = x.size(); Vec y(n); float carry = 0.f;
    for (size_t b = 0; b < n; b += B) {
        size_t e = b + B < n ? b + B : n; float s = 0.f;
        for (size_t i = b; i < e; i++) { s += x[i]; y[i] = s; }
        for (size_t i = b; i < e; i++) y[i] += carry;
        carry += s;
    }
    return y;
}

static const std::map<std::string, Fn> &registry() {
    static const std::map<std::string, Fn> r = {
        {"sum/naive", sum_naive}, {"sum/unroll4", sum_unroll4}, {"sum/pairwise", sum_pairwise},
        {"dot/naive", dot_naive}, {"dot/unroll4", dot_unroll4}, {"dot/pairwise", dot_pairwise},
        {"softmax/naive", softmax_naive}, {"softmax/online", softmax_online},
        {"layernorm/naive", layernorm_naive}, {"layernorm/welford", layernorm_welford},
        {"prefix/naive", prefix_naive}, {"prefix/blocked", prefix_blocked},
    };
    return r;
}

int main(int argc, char **argv) {
    if (argc == 2 && std::string(argv[1]) == "--list") {
        for (auto &kv : registry()) std::printf("%s\n", kv.first.c_str());
        return 0;
    }
    if (argc < 4) { std::fprintf(stderr, "usage: %s kernel/variant[+bf16] in.bin out.bin [reps]\n", argv[0]); return 2; }
    std::string name = argv[1];
    bool bf16 = false;
    if (name.size() > 5 && name.compare(name.size() - 5, 5, "+bf16") == 0) { bf16 = true; name.resize(name.size() - 5); }
    auto it = registry().find(name);
    if (it == registry().end()) { std::fprintf(stderr, "unknown kernel %s\n", name.c_str()); return 2; }
    int reps = argc > 4 ? std::atoi(argv[4]) : 1;

    std::ifstream in(argv[2], std::ios::binary | std::ios::ate);
    if (!in) { std::fprintf(stderr, "cannot open input\n"); return 2; }
    size_t bytes = (size_t)in.tellg(); in.seekg(0);
    Vec x(bytes / 4); in.read(reinterpret_cast<char *>(x.data()), (std::streamsize)(x.size() * 4));
    if (bf16) for (float &v : x) v = round_bf16(v);

    Fn fn = it->second; Vec y; double best = 1e300;
    for (int r = 0; r < (reps > 0 ? reps : 1); r++) {
        auto t0 = std::chrono::steady_clock::now();
        y = fn(x);
        auto t1 = std::chrono::steady_clock::now();
        double ns = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        if (ns < best) best = ns;
    }
    std::ofstream out(argv[3], std::ios::binary);
    out.write(reinterpret_cast<const char *>(y.data()), (std::streamsize)(y.size() * 4));
    std::printf("time_ns=%.0f\n", best);
    return 0;
}
