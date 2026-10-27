// q52_probe: minimal OLD-vs-OLD determinism probe for the SHIPPED
// gdn_recurrent_update_kernel. One case, env-driven geometry, per-word flip
// classification (v-column band, row), optional sentinel pad to catch
// out-of-bounds reads/writes. Built against the kernel passed in src/.
#include "../src/r3d_gdn.hip"
void r3d_dump_begin(R3dDumpSlot*, const char*, const void*, const RadArgs*, RadStream) {}
void r3d_dump_end(R3dDumpSlot*, int) {}

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) {                        \
    std::printf("HIP FAIL %s @%d: %s\n", #x, __LINE__, hipGetErrorString(e_));          \
    std::exit(1); } } while (0)

static uint32_t g_seed = 1;
static uint32_t xs() { uint32_t x = g_seed; x ^= x << 13; x ^= x >> 17; x ^= x << 5; g_seed = x; return x; }
static float u01() { return (xs() >> 8) * (1.0f / 16777216.0f); }

static void fill_f32(std::vector<uint8_t>& v, float amp) {
    float* f = (float*)v.data();
    for (size_t i = 0; i < v.size() / 4; ++i) f[i] = (u01() * 2.0f - 1.0f) * amp;
}
static void fill_bf(std::vector<uint8_t>& v, float amp) {
    uint16_t* b = (uint16_t*)v.data();
    for (size_t i = 0; i < v.size() / 2; ++i) {
        float f = (u01() * 2.0f - 1.0f) * amp;
        uint32_t u; std::memcpy(&u, &f, 4);
        uint32_t lsb = (u >> 16) & 1u;
        u = u + 0x7fffu + lsb;
        b[i] = (uint16_t)(u >> 16);
    }
}

struct Pack {
    const uint8_t *q; uint32_t dtQ; int64_t q_s0b, q_s1b;
    const uint8_t *k; uint32_t dtK; int64_t k_s0b, k_s1b;
    const uint8_t *v; uint32_t dtV; int64_t v_s0b, v_s1b;
    const uint8_t *av; uint32_t dtA; int64_t a_s0b;
    const uint8_t *bv; uint32_t dtB; int64_t b_s0b;
    const uint8_t *alog; uint32_t dtAL; int64_t al_s1b;
    const uint8_t *dtb; uint32_t dtDB; int64_t db_s1b;
    uint8_t *st; uint32_t dtST; int64_t st_s0b, st_s1b, st_s2b;
    const void *sidx; uint32_t dtS; int64_t si_s0b, si_s1b, sw;
    const void *nacc; uint32_t dtN;
    const void *cu; uint32_t dtC;
    const uint8_t *z; uint32_t dtZ; int64_t z_s0b, z_s1b;
    const uint8_t *nw; uint32_t dtW; int64_t w_s1b;
    uint8_t *o; uint32_t dtO; int64_t o_s0b, o_s1b;
    uint8_t *oq; R3dRow voq; uint8_t *osc; R3dRow vosc;
    int64_t V, grp; float scale, l2eps, neps, spthr; int sigm;
    int64_t st_eb, al_groups, al_fstr, al_vb, al_gb, al_mark, al_tmax;
};
#define PACK_ARGS                                                               \
    P.q, P.dtQ, P.q_s0b, P.q_s1b, P.k, P.dtK, P.k_s0b, P.k_s1b,                 \
    P.v, P.dtV, P.v_s0b, P.v_s1b, P.av, P.dtA, P.a_s0b, P.bv, P.dtB, P.b_s0b,   \
    P.alog, P.dtAL, P.al_s1b, P.dtb, P.dtDB, P.db_s1b,                          \
    P.st, P.dtST, P.st_s0b, P.st_s1b, P.st_s2b,                                \
    P.sidx, P.dtS, P.si_s0b, P.si_s1b, P.sw, P.nacc, P.dtN, P.cu, P.dtC,        \
    P.z, P.dtZ, P.z_s0b, P.z_s1b, P.nw, P.dtW, P.w_s1b,                         \
    P.o, P.dtO, P.o_s0b, P.o_s1b, P.oq, P.voq, P.osc, P.vosc,                   \
    P.V, P.grp, P.scale, P.l2eps, P.neps, P.spthr, P.sigm,                      \
    P.st_eb, P.al_groups, P.al_fstr, P.al_vb, P.al_gb, P.al_mark, P.al_tmax

static void launch_old(const Pack& P, uint32_t Hv, uint32_t N, bool anchor) {
    if (anchor) gdn_recurrent_update_kernel<true , 1><<<dim3(Hv, N), dim3(kRecThreads), 0>>> (PACK_ARGS);
    else        gdn_recurrent_update_kernel<false, 1><<<dim3(Hv, N), dim3(kRecThreads), 0>>> (PACK_ARGS);
    CK(hipGetLastError());
}

static int g_poison = 0;   // Q52_POISON: sentinel in guard pads
static void* dup_dev(const std::vector<uint8_t>& h) {
    const size_t pad = 16384;
    void* d = nullptr; CK(hipMalloc(&d, (h.empty() ? 16 : h.size()) + pad));
    CK(hipMemset(d, g_poison, (h.empty() ? 16 : h.size()) + pad));
    if (!h.empty()) CK(hipMemcpy(d, h.data(), h.size(), hipMemcpyHostToDevice));
    return d;
}
static uint64_t fnv(const std::vector<uint8_t>& v) {
    uint64_t h = 1469598103934665603ull;
    for (uint8_t c : v) { h ^= c; h *= 1099511628211ull; }
    return h;
}

int main(int argc, char** argv) {
    const uint32_t V   = argc > 1 ? std::atoi(argv[1]) : 40;
    const uint32_t Hv  = argc > 2 ? std::atoi(argv[2]) : 32;
    const uint32_t N   = argc > 3 ? std::atoi(argv[3]) : 8;
    const int T        = argc > 4 ? std::atoi(argv[4]) : 22;   // total tokens, row N=0
    const int anchor   = argc > 5 ? std::atoi(argv[5]) : 0;
    const int z_on     = argc > 6 ? std::atoi(argv[6]) : 1;
    const int oq_on    = argc > 7 ? std::atoi(argv[7]) : 0;
    const uint32_t seed= argc > 8 ? (uint32_t)std::atoi(argv[8]) : 11u;
    const int iters    = argc > 9 ? std::atoi(argv[9]) : 1;
    const char* pv = std::getenv("Q52_POISON"); g_poison = pv ? (int)std::strtol(pv, nullptr, 0) : 0;
    const char* pa = std::getenv("Q52_AMP"); float AMP = pa ? (float)std::atof(pa) : 1.0f;

    const uint32_t Hg = Hv > 1 ? Hv / 2 : 1, K = kRecK, SLOTS = 96, sw = 12;
    g_seed = seed * 2654435761u + 17u; if (!g_seed) g_seed = 1;

    // ragged rows: first N-1 get slen=1 unless single-row mode
    std::vector<int> slens(N, 1);
    if (N == 1) slens[0] = T; else slens[0] = T - (int)(N - 1);
    if (anchor && slens[0] == 0) slens[0] = 1;
    const int Tot = [&]{ int s = 0; for (int x : slens) s += x; return s; }();

    const size_t nq = (size_t)Tot * Hg * K * 2, nv = (size_t)Tot * Hv * V * 2;
    std::vector<uint8_t> hq(nq), hk(nq), hv(nv), hz(nv), hav((size_t)Tot*Hv*2),
        hbv(hav.size()), hal(Hv*4), hdb(Hv*4), hnw(V*4),
        hst((size_t)SLOTS * Hv * V * K * 4),
        hsidx((size_t)N*sw*4), hna(N*4), hcu((N+1)*4),
        hoq(oq_on ? nv/2 : 0), hos(std::max((size_t)Tot*Hv*4, (size_t)256)), ho(nv);
    fill_bf(hq, 1.0f*AMP); fill_bf(hk, 1.0f*AMP); fill_bf(hv, 2.0f*AMP);
    if (z_on) fill_bf(hz, 2.0f*AMP);
    fill_bf(hav, 4.0f*AMP); fill_bf(hbv, 6.0f*AMP);
    fill_f32(hal, 3.0f*AMP); fill_f32(hdb, 1.0f*AMP); fill_f32(hnw, 1.0f);
    std::memset(hst.data(), 0, hst.size());   // zeroed pool like a live boot
    std::memset(ho.data(), 0, ho.size());
    if (oq_on) std::memset(hoq.data(), 0xAA, hoq.size());
    int32_t* cu = (int32_t*)hcu.data(); cu[0] = 0;
    for (uint32_t n = 0; n < N; ++n) cu[n+1] = cu[n] + slens[n];
    int32_t* sidx = (int32_t*)hsidx.data();
    for (uint32_t n = 0; n < N; ++n) for (uint32_t c = 0; c < sw; ++c) sidx[n*sw+c] = (int)((n*sw+c) % SLOTS);
    int32_t* nn = (int32_t*)hna.data();
    for (uint32_t n = 0; n < N; ++n) nn[n] = anchor ? 2 : 1;

    Pack P{};
    P.dtQ = P.dtK = P.dtV = P.dtA = P.dtB = RAD_BF16; P.dtZ = z_on ? RAD_BF16 : 0;
    P.dtAL = P.dtDB = P.dtW = RAD_F32; P.dtST = RAD_F32; P.dtO = RAD_BF16;
    P.dtS = P.dtN = P.dtC = RAD_I32;
    P.q_s1b = K*2; P.q_s0b = (int64_t)Hg*K*2; P.k_s1b = K*2; P.k_s0b = P.q_s0b;
    P.v_s1b = V*2; P.v_s0b = (int64_t)Hv*V*2; P.z_s1b = V*2; P.z_s0b = P.v_s0b;
    P.o_s1b = V*2; P.o_s0b = (int64_t)Hv*V*2; P.a_s0b = (int64_t)Hv*2; P.b_s0b = P.a_s0b;
    P.al_s1b = 4; P.db_s1b = 4; P.w_s1b = 4;
    P.st_s2b = K*4; P.st_s1b = (int64_t)V*K*4; P.st_s0b = (int64_t)Hv*V*K*4;
    P.si_s1b = 4; P.si_s0b = sw*4; P.sw = sw;
    P.scale = 0.08838834764831845f; P.l2eps = 1e-6f; P.neps = 1e-6f; P.spthr = 20.0f;
    P.sigm = 0; P.V = V; P.grp = Hv/Hg; P.st_eb = 4;
    P.al_groups = (V + 31) / 32; P.al_fstr = P.al_groups*K + V + 2*P.al_groups;
    P.al_vb = P.al_groups*K; P.al_gb = P.al_groups*K + V;
    P.al_tmax = (V*K - 1) / P.al_fstr; if (P.al_tmax > 16) P.al_tmax = 16;
    P.al_mark = P.al_tmax * P.al_fstr;
    P.voq = R3dRow{ 0, {0,0,0}, {0,0,0}, (int64_t)(Hv*V), 0 };
    P.vosc = R3dRow{ 0, {0,0,0}, {0,0,0}, (int64_t)Hv, 0 };

    void* dq = dup_dev(hq); void* dk = dup_dev(hk); void* dv = dup_dev(hv);
    void* dav = dup_dev(hav); void* dbv = dup_dev(hbv); void* dal = dup_dev(hal);
    void* ddb = dup_dev(hdb); void* dnw = dup_dev(hnw); void* dsidx = dup_dev(hsidx);
    void* dna = dup_dev(hna); void* dcu = dup_dev(hcu); void* dz = dup_dev(hz);
    void* dstA = dup_dev(hst); void* dstB = dup_dev(hst);
    void* doA = dup_dev(ho); void* doB = dup_dev(ho);
    void* dqA = dup_dev(hoq); void* dqB = dup_dev(hoq);
    void* dsA = dup_dev(hos); void* dsB = dup_dev(hos);

    int fails = 0;
    for (int it = 0; it < iters; ++it) {
        P.q=(const uint8_t*)dq; P.k=(const uint8_t*)dk; P.v=(const uint8_t*)dv;
        P.av=(const uint8_t*)dav; P.bv=(const uint8_t*)dbv; P.alog=(const uint8_t*)dal;
        P.dtb=(const uint8_t*)ddb; P.z=z_on?(const uint8_t*)dz:nullptr; P.nw=(const uint8_t*)dnw;
        P.sidx=dsidx; P.nacc=dna; P.cu=dcu;
        P.st=(uint8_t*)dstA; P.o=(uint8_t*)doA; P.oq=oq_on?(uint8_t*)dqA:nullptr; P.osc=(uint8_t*)dsA;
        launch_old(P, Hv, N, anchor);
        P.st=(uint8_t*)dstB; P.o=(uint8_t*)doB; P.oq=oq_on?(uint8_t*)dqB:nullptr; P.osc=(uint8_t*)dsB;
        launch_old(P, Hv, N, anchor);
        CK(hipDeviceSynchronize());
        std::vector<uint8_t> oA(ho.size()), oB(ho.size()), tA(hst.size()), tB(hst.size());
        CK(hipMemcpy(oA.data(), doA, oA.size(), hipMemcpyDeviceToHost));
        CK(hipMemcpy(oB.data(), doB, oB.size(), hipMemcpyDeviceToHost));
        CK(hipMemcpy(tA.data(), dstA, tA.size(), hipMemcpyDeviceToHost));
        CK(hipMemcpy(tB.data(), dstB, tB.size(), hipMemcpyDeviceToHost));
        size_t diffo = 0, diffst = 0, first = (size_t)-1;
        for (size_t i = 0; i < oA.size(); ++i) if (oA[i] != oB[i]) { ++diffo; if (first == (size_t)-1) first = i; }
        for (size_t i = 0; i < tA.size(); ++i) if (tA[i] != tB[i]) ++diffst;
        if (diffo || diffst) {
            ++fails;
            // classify: word = first/2 in row-major (tok,head,v)
            const size_t w = first / 2;
            const size_t vcol = w % V, hh = (w / V) % Hv, tok = w / (V * Hv);
            // band census: flips per v column (v < 32 vs v >= 32 for V <= 64)
            int lo = 0, hi = 0;
            for (size_t i = 0; i + 1 < oA.size(); i += 2) {
                if (oA[i] != oB[i] || oA[i+1] != oB[i+1]) {
                    if ((i/2) % V < 32) ++lo; else ++hi;
                }
            }
            std::printf("iter%d DIFF o=%zu st=%zu first@word%zu (tok=%zu h=%zu v=%zu)\n",
                        it, diffo/2, diffst, w, tok, hh, vcol);
            std::printf("  flipped-words v<32:%d v>=32:%d\n", lo, hi);
            uint16_t a, b; std::memcpy(&a, &oA[first], 2); std::memcpy(&b, &oB[first], 2);
            uint32_t ua = (uint32_t)a << 16, ub = (uint32_t)b << 16; float fa, fb;
            std::memcpy(&fa, &ua, 4); std::memcpy(&fb, &ub, 4);
            // row-constancy test: ratios A/B across the first flipped (tok,h) row
            {
                const size_t rowbase = (w / V) * V;   // word index of row start
                int shown = 0; double r0 = 0.0; int constn = 1;
                for (size_t c = 0; c < V && shown < 6; ++c, ++shown) {
                    size_t idx = (rowbase + c) * 2; uint16_t x, y;
                    std::memcpy(&x, &oA[idx], 2); std::memcpy(&y, &oB[idx], 2);
                    uint32_t ux = (uint32_t)x << 16, uy = (uint32_t)y << 16;
                    float fx, fy; std::memcpy(&fx, &ux, 4); std::memcpy(&fy, &uy, 4);
                    double rr = (fy != 0.0f) ? (double)fx / (double)fy : 0.0;
                    if (shown == 0) r0 = rr; else if (rr != r0) constn = 0;
                    std::printf("  row c%zu A=%a B=%a ratio=%.9g\n", c, fx, fy, rr);
                }
                std::printf("  row-constancy: %s\n", constn ? "CONSTANT-RATIO" : "VARIES");
            }
            std::printf("  first A=%04x (%a) B=%04x (%a) fnv %016llx/%016llx stfnv %016llx/%016llx\n",
                        a, fa, b, fb, (unsigned long long)fnv(oA), (unsigned long long)fnv(oB),
                        (unsigned long long)fnv(tA), (unsigned long long)fnv(tB));
        } else if (it == 0 || it == iters - 1) {
            std::printf("iter%d SAME o=%016llx\n", it, (unsigned long long)fnv(oA));
        }
    }
    std::printf("RESULT V=%u Hv=%u N=%d T=%d anchor=%d z=%d oq=%d seed=%u: iters=%d fails=%d -- %s\n",
                V, Hv, N, T, anchor, z_on, oq_on, seed, iters, fails, fails ? "NONDET" : "DETERMINISTIC");
    return fails ? 1 : 0;
}
