/* tools/test_logits_e4m3.cpp -- card unit test for the logits_e4m3 kernel body.
 *
 * The joined 27B head (7.1 GiB) cannot be gated against libref while the
 * co-tenant holds the host RAM, so this proves the kernel's math on tiny
 * geometries instead: a CPU oracle decodes the same bytes through a
 * spec-independent table walk, and the kernel must match to f32 rounding.
 * Covers the block scale grid (br 128 and 1), the ragged-K tail, live<M
 * groups across grid.y, and the bf16/f16/f32 scale types.
 *
 * Build (in radiance-dev, on the card):
 *   hipcc -O2 --offload-arch=gfx1100 -I/opt/radiance/include/radiance \
 *         -I/work/src -I/work/../upstream/abi tools/test_logits_e4m3.cpp -o /tmp/t
 * The include pulls the real kernel; no fixture, no engine.
 */
#include "../src/r3d_gemm_logits.hip"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

/* Independent e4m3 decoder: straight from the 1-4-3 spec, table walk, no
 * shared code with the kernel's ported rad_plugin.h helper. */
static float e4m3_cpu(uint8_t b) {
    const int s = (b >> 7) & 1, e = (b >> 3) & 0xF, m = b & 7;
    const float sign = s ? -1.0f : 1.0f;
    if (e == 0) return sign * (float)m / 8.0f * std::ldexp(1.0f, -6);  /* m * 2^-9 */
    if (e == 0xF && m == 7) return INFINITY;
    return sign * (1.0f + (float)m / 8.0f) * std::ldexp(1.0f, e - 7);
}

static float bf16_cpu(uint16_t b) {
    const uint32_t u = (uint32_t)b << 16;
    float f; std::memcpy(&f, &u, 4); return f;
}
static float f16_cpu(uint16_t b) {                    /* decode f16 by spec walk */
    const int s = (b >> 15) & 1, e = (b >> 10) & 0x1F, m = b & 0x3FF;
    const float sign = s ? -1.0f : 1.0f;
    if (e == 0)  return sign * std::ldexp((float)m / 1024.0f, -14);
    if (e == 31) return m ? NAN : sign * INFINITY;
    return sign * (1.0f + (float)m / 1024.0f) * std::ldexp(1.0f, e - 15);
}

static uint16_t f32_to_bf16_cpu(float f) {            /* round-to-nearest-even */
    uint32_t u; std::memcpy(&u, &f, 4);
    const uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return (uint16_t)(u >> 16);
}

struct Case { int N, K, M, br, sdt; const char* name; };

static int run_case(const Case& c) {
    std::mt19937 rng(0xC0FFEE ^ (c.N * 31 + c.K * 7 + c.M * 3 + c.br * 17 + c.sdt));
    std::uniform_int_distribution<int> byte_d(0, 255);
    std::uniform_real_distribution<float> unit(-2.0f, 2.0f);

    const int srows = (c.N + c.br - 1) / c.br;
    const int scols = (c.K + 127) / 128;
    const int selt  = c.sdt == 2 ? 4 : 2;                 /* bytes per scale elem */

    std::vector<uint8_t>  codes((size_t)c.N * c.K);
    std::vector<uint16_t> x((size_t)c.M * c.K), s((size_t)srows * scols * 2);
    std::vector<float>    xf(c.M * c.K);                  /* the f32 x the kernel sees */
    std::vector<float>    y_cpu((size_t)c.M * c.N, 0.0f);
    std::vector<float>    y_gpu((size_t)c.M * c.N, -1.0f);

    for (auto& b : codes) b = (uint8_t)((byte_d(rng) & 0x7F) | (byte_d(rng) & 0x80));
    for (int i = 0; i < c.M * c.K; ++i) {
        const float v = unit(rng);
        x[i] = f32_to_bf16_cpu(v);
        xf[i] = bf16_cpu(x[i]);
    }
    std::vector<float> sf(srows * scols);
    for (int i = 0; i < srows * scols; ++i) {
        const float v = std::exp(unit(rng));               /* positive, no zeros */
        sf[i] = v;
        if (c.sdt == 0) s[i] = f32_to_bf16_cpu(v);
        else if (c.sdt == 1) {                             /* f16: nearest via float->half bits */
            /* encode by brute force over the quarter ulp grid */
            const float m = v / std::ldexp(1.0f, 0); (void)m;
            /* store through a real f16 conversion at test scale: keep it small */
            const float h = std::round(v * 1024.0f) / 1024.0f;   /* f16-exact enough */
            sf[i] = h;
            uint32_t u; float hv = h; std::memcpy(&u, &hv, 4);
            const int se = ((u >> 23) & 0xFF) - 127 + 15;
            const uint32_t sm = u & 0x7FFFFFu;
            s[i] = (uint16_t)(((u >> 16) & 0x8000u) | (se << 10) | (sm >> 13));
        } else {
            uint32_t u; std::memcpy(&u, &v, 4); std::memcpy(&s[i * 2], &u, 4);
            sf[i] = v;
        }
    }

    /* CPU oracle: the reference's own walk, ascending k in f32. */
    for (int m = 0; m < c.M; ++m)
        for (int n = 0; n < c.N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < c.K; ++k) {
                const size_t sidx = (size_t)(n / c.br) * scols + (k / 128);
                const float sv = c.sdt == 2 ? ((const float*)s.data())[sidx]
                              : c.sdt == 1 ? f16_cpu(s[sidx])
                              : bf16_cpu(s[sidx]);
                const float wv = e4m3_cpu(codes[(size_t)n * c.K + k]) * sv;
                acc = std::fma(xf[(size_t)m * c.K + k], wv, acc);
            }
            y_cpu[(size_t)m * c.N + n] = acc;
        }

    /* Device buffers. */
    uint8_t *d_codes, *d_s; uint16_t *d_x; float *d_y;
    hipMalloc(&d_codes, codes.size());
    hipMalloc(&d_s, s.size() * 2);
    hipMalloc(&d_x, x.size() * 2);
    hipMalloc(&d_y, y_gpu.size() * 4);
    hipMemcpy(d_codes, codes.data(), codes.size(), hipMemcpyHostToDevice);
    hipMemcpy(d_s, s.data(), s.size() * 2, hipMemcpyHostToDevice);
    hipMemcpy(d_x, x.data(), x.size() * 2, hipMemcpyHostToDevice);

    const int mt = c.M >= 8 ? 8 : c.M >= 4 ? 4 : c.M >= 2 ? 2 : 1;
    const dim3 grid((c.N + 7) / 8, (c.M + mt - 1) / mt);
    const int scols_i = scols, br_i = c.br;
    const uint32_t sdt = c.sdt == 0 ? RAD_BF16 : c.sdt == 1 ? RAD_F16 : RAD_F32;
    float* yp = d_y;
    auto launch = [&](auto kern) {
        kern<<<grid, 256>>>(d_x, c.K, d_codes, c.K, d_s, scols_i, br_i, sdt,
                            yp, c.N, c.N, c.K, c.M, 0);
    };
    if      (mt == 1) launch(logits_e4m3_kernel<1>);
    else if (mt == 2) launch(logits_e4m3_kernel<2>);
    else if (mt == 4) launch(logits_e4m3_kernel<4>);
    else              launch(logits_e4m3_kernel<8>);
    const hipError_t le = hipGetLastError();
    hipDeviceSynchronize();
    if (le != hipSuccess) { printf("%-28s LAUNCH FAIL %s\n", c.name, hipGetErrorString(le)); return 1; }
    hipMemcpy(y_gpu.data(), d_y, y_gpu.size() * 4, hipMemcpyDeviceToHost);

    double worst = 0.0; int bad = -1;
    for (size_t i = 0; i < y_gpu.size(); ++i) {
        const double ref = std::fabs(y_cpu[i]);
        const double err = std::fabs((double)y_gpu[i] - (double)y_cpu[i]);
        const double rel = err / (ref > 1e-6 ? ref : 1e-6);
        if (rel > worst) { worst = rel; if (rel > 1e-4) bad = (int)i; }
    }
    printf("%-28s worst rel = %.3e  %s\n", c.name, worst, worst < 1e-4 ? "OK" : "FAIL");
    if (bad >= 0) printf("   first bad idx %d: gpu %g cpu %g\n", bad, y_gpu[bad], y_cpu[bad]);
    hipFree(d_codes); hipFree(d_s); hipFree(d_x); hipFree(d_y);
    return worst < 1e-4 ? 0 : 1;
}

int main() {
    if (hipSetDevice(0) != hipSuccess) { printf("no card\n"); return 2; }
    const Case cases[] = {
        { 256, 512,  1, 128, 0, "N256 K512 M1 br128 bf16" },
        { 256, 512,  3, 128, 0, "N256 K512 M3 br128 bf16" },
        { 130, 300,  5, 128, 0, "N130 K300 M5 ragged br128" },
        { 256, 512,  9, 128, 0, "N256 K512 M9 two y-groups" },
        { 200, 512,  2,   1, 0, "N200 K512 M2 br1 bf16" },
        { 256, 384,  4, 128, 1, "N256 K384 M4 br128 f16" },
        { 256, 512,  8, 128, 2, "N256 K512 M8 br128 f32" },
        {   8,   4,  1, 128, 0, "tiny N8 K4" },
    };
    int fails = 0;
    for (const auto& c : cases) fails += run_case(c);
    printf("%d/%d passed\n", (int)(sizeof cases / sizeof *cases) - fails,
                             (int)(sizeof cases / sizeof *cases));
    return fails != 0;
}
