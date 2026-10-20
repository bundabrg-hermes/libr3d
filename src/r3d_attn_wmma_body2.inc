/* r3d_attn_wmma_body2.inc -- the v2 schedule of the WMMA flash-attention tile.

 * Same arithmetic, same visibility rules, same arena contract as v1
 * (r3d_attn_wmma_body.inc); everything that moved here is SCHEDULING:
 *   - 8 wave32s (256 threads). Wave w owns query rows 16*(w&3); the upper waves
 *     (w>>2) split COLUMNS: S-mma gets one 16-wide column half per wave instead
 *     of two per wave at 4 waves, O-mma splits the head_dim into 4-column groups.
 *   - Q is staged once, loaded into persistent fragments (one fa per k-slice),
 *     and its staging LDS is then reused by the K buffers -- the S/P/K/V carve
 *     fits 48 KiB with a real double buffer, so every block gets 8 resident
 *     waves per CU for latency hiding (v1: 4) at the same occupancy count.
 *   - The next key slice is prefetched GLOBAL->REGISTERS before the current
 *     slice's mmas and stored AFTER them, overlapping the paged gather's latency
 *     with the matrix work (the gemm m64 DBUF lesson, applied to the gather).
 *   - Barriers per key step: 2 in pass one, 3 in pass two (v1: 4 + 6). K's two
 *     buffers make the slice store safe without its own barrier; the addressing
 *     barrier disappears entirely -- each staging warp recomputes the one
 *     page-table lookup its rows need (uniform in the warp -> a broadcast), and
 *     the column-badness byte for the NEXT tile rides the barrier that already
 *     closes the softmax / P phase, written strictly AFTER it (a CB write racing
 *     the softmax/P read in the same phase is the bug that barrier bought).
 * The two-pass order, the exp2 domain, the fragment-bit staging rules, and the
 * SPLITK partial-store layout are v1's verbatim, so the decode rows keep their
 * merge identity: a splits=1 launch is exactly this tile under a different final
 * store, and R3D_ATTN_V2=1 selects v2 over the default v1 for postmortem benches.
 * (v2 measured 0.43-0.80x at deep ctx against bit-identical v1 output, 2026-10-20:
 * one 256-thread block with 40.7 KB LDS per CU serializes every barrier on the CU.)
 *
 * Included by r3d_attn_paged_wmma.hip (SPLITK=0) and r3d_attn_paged_decode.hip
 * (SPLITK=1), after the helper block, inside each TU's second anonymous
 * namespace, behind the gfx1100..gfx1103 gate. The includer defines kBM, kBN,
 * kLog2e (shared with v1); this file owns its own thread count kT.
 */

/* kT (=256) is declared by each includer beside its v1 kThreads, outside the chip
 * gate, so the launchers (which live outside the gate) can hand block2(kT). */

/* v2 LDS carve (D=128 figures): S 10240 + P 4608 + K x2 16896 + V 8448 + meta
 * ~560 = ~40.8 KB static. Q pre-stages into the K region (exactly 2 x 8448 bytes
 * for 64 x 132 bf16) and dies when the fragments land; the O epilogue reuses
 * K0+K1+V (25344 >= 32 x 136 f32). At D=64: ~28.4 KB -- room for two 8-wave
 * blocks per CU once the engine asks for them (occupancy falls out of the carve,
 * which is the per-head_dim budget v1 never had). */
template <int D> struct Carve2 {
    static constexpr int kQP = D + 4, kVP = D + 4, kSP = kBN + 8, kPP = kBN + 4, kOP = D + 8;
    static constexpr int kK = kBN * kVP * 2;                     /* one K slice */
    static constexpr int offS  = 0;
    static constexpr int offP  = (offS + kBM * kSP * 4 + 15) & ~15;
    static constexpr int offK0 = (offP + kBM * kPP * 2 + 15) & ~15;   /* also Q preload */
    static constexpr int offK1 = offK0 + kK;
    static constexpr int offV  = offK1 + kK;
    static constexpr int offM  = offV + kK;                        /* m[64] l[64] f32 */
    static constexpr int offCB = offM + 2 * kBM * 4;
    static constexpr int total = offCB + kBN + 16;
};

/* one key's KV row base, or -1 when unmapped/out of range. A staging warp calls
 * this for its own rows only -- the load it performs is warp-uniform, so the
 * v1 RB[]/CB[] LDS phase and its barrier are simply gone from the hot loop. */
__device__ __forceinline__ int64_t kv_row2(const R3dAttnParams& P, int s, int64_t kh,
                                           int64_t j, int64_t jend) {
    if (j >= jend) return -1;
    const int64_t blk = (int64_t)ldi_w(P.bt_dt, P.bt,
                                       s * P.bt_rs + (j / P.block_size) * P.bt_cs);
    return blk >= 0 ? blk * P.kv_sb + kh * P.kv_sh + (j % P.block_size) * P.kv_so : -1;
}

template <int D, int KVDT, int SPLITK>
__global__ __launch_bounds__(kT) void attn_paged_wmma_kernel2(R3dAttnParams P) {
    constexpr int DSH = (D == 128) ? 7 : 6, DM1 = D - 1;
    constexpr int PFN = kBN * D / kT;         /* prefetch slots per thread (16 at D=128) */
    typedef Carve2<D> C;
    __shared__ unsigned char raw[C::total];

    float*  Ssp = reinterpret_cast<float*>(raw + C::offS);
    bf16*   Psp = reinterpret_cast<bf16*>(raw + C::offP);
    bf16*   K0  = reinterpret_cast<bf16*>(raw + C::offK0);
    bf16*   K1  = reinterpret_cast<bf16*>(raw + C::offK1);
    bf16*   Vp  = reinterpret_cast<bf16*>(raw + C::offV);
    float*  MSP = reinterpret_cast<float*>(raw + C::offM);
    float*  LSP = MSP + kBM;
    unsigned char* CB = raw + C::offCB;

    const int h = blockIdx.x, s = blockIdx.y;
    const int spl = (SPLITK && P.splits > 0) ? P.splits : 1;
    const int qt  = SPLITK ? (int)blockIdx.z / spl : (int)blockIdx.z;
    const int sp  = SPLITK ? (int)blockIdx.z % spl : 0;
    const int tid = threadIdx.x, w = tid >> 5;
    const int rb_own = w & 3;                 /* this wave's 16-row block        */
    const int cq     = w >> 2;                /* this wave's column quarter (0/1) */

    int64_t start, qlen;
    if (P.cus) {
        start = (int64_t)ldi_w(P.cus_dt, P.cus, s);
        qlen  = (int64_t)ldi_w(P.cus_dt, P.cus, s + 1) - start;
    } else {
        qlen  = P.q_len_uniform;
        start = (int64_t)s * qlen;
    }
    const int64_t i0 = (int64_t)qt * kBM;
    if (i0 >= qlen) return;
    const int64_t tok_base = start + i0;
    const int64_t valid    = min<int64_t>(qlen - i0, kBM);
    const int64_t ctx      = (int64_t)ldi_w(P.su_dt, P.su, s);
    const int64_t pos_base = ctx - qlen + i0;
    const int64_t kh       = h / P.gqa;
    const float   kds      = P.kdes ? ldt_w(P.kdes, s * P.kv_heads + kh) : 1.0f;
    const float   vds      = P.vdes ? ldt_w(P.vdes, s * P.kv_heads + kh) : 1.0f;
    const float   scale2   = P.scale * kds * kLog2e;
    const int64_t jend     = min<int64_t>(ctx, P.jcap);

    const int causal = P.causal; const int64_t window = P.window;
    auto visible = [&](int64_t rowpos, int64_t j) -> bool {
        if (causal && j > rowpos) return false;
        if (window > 0) {
            const int64_t d = rowpos - j;
            if (causal ? (d >= window) : (d >= window || d <= -window)) return false;
        }
        return true;
    };

    const int64_t pos_last = pos_base + valid - 1;
    int64_t jmax = jend;
    if (causal && pos_last + 1 < jmax) jmax = pos_last + 1;
    if (!causal && window > 0 && pos_last + window < jmax) jmax = pos_last + window;
    int64_t jmin = 0;
    if (window > 0) jmin = max<int64_t>(0, pos_base - window + 1);
    const int t0 = (int)(jmin / kBN);
    const int t1 = (int)((jmax + kBN - 1) / kBN);

    int et0 = t0, et1 = t1;
    if (SPLITK) {
        const int nt = t1 - t0;
        const int a  = t0 + (int)(((long long)nt * sp) / spl);
        const int b  = t0 + (int)(((long long)nt * (sp + 1)) / spl);
        et0 = a; et1 = b;
        if (et1 <= et0) {
            float* pso = reinterpret_cast<float*>(P.pf);
            const int64_t pn = P.rows * P.q_heads;
            float* psm = pso + pn * spl * D, * psl = psm + pn * spl;
            for (int64_t row = tid; row < valid; row += kT) {
                const int64_t idx = ((tok_base + row) * P.q_heads + h) * spl + sp;
                psm[idx] = -INFINITY; psl[idx] = 0.0f;
            }
            return;
        }
    }

    /* ---- Q preload: stage the 64-row tile into the K region, load each wave's
     * persistent A fragments, and only then let K slices overwrite that region.
     * Two barriers, once per block, not per tile. (Bit-pattern staging rules: the
     * bf16_store contract of the helpers above -- never construct a bf16.) ---- */
    {
        bf16* Qsp = K0;                        /* region doubles as Q staging */
        const int64_t qbase = tok_base * P.q_rs + (int64_t)h * P.q_hs;
        for (int e = tid; e < kBM * D; e += kT) {
            const int r = e >> DSH, d = e & DM1;
            bf16_store(Qsp + r * C::kQP + d, (r < valid)
                ? ((const uint16_t*)P.q)[qbase + (int64_t)r * P.q_rs + (int64_t)d * P.q_ds]
                : (uint16_t)0);
        }
        if (tid < kBM) { MSP[tid] = -INFINITY; LSP[tid] = 0.0f; }
        __syncthreads();
        roCW::fragment<roCW::matrix_a, 16, 16, 16, bf16, roCW::row_major> fa[D / 16];
        #pragma unroll
        for (int k0 = 0; k0 < D / 16; ++k0)
            roCW::load_matrix_sync(fa[k0], Qsp + (rb_own * 16) * C::kQP + k0 * 16,
                                   C::kQP, roCW::mem_row_major);

        roCW::fragment<roCW::matrix_b, 16, 16, 16, bf16, roCW::col_major> fb0, fbv;
        roCW::fragment<roCW::accumulator, 16, 16, 16, float> acc;
        roCW::fragment<roCW::accumulator, 16, 16, 16, float> oa[D / 32];   /* 4 at D=128 */
        #pragma unroll
        for (int i = 0; i < D / 32; ++i) roCW::fill_fragment(oa[i], 0.0f);

        /* ---- slice staging: global -> registers (issue) and registers -> LDS
         * (land). The (r,d) mapping is a pure function of the slot, so land() may
         * run arbitrarily far from issue() and still write the slice it read. ---- */
        uint16_t pf[PFN];
        auto issue = [&](int64_t jb, const int isV) {
            #pragma unroll
            for (int p = 0; p < PFN; ++p) {
                const int e = tid + p * kT;
                const int r = e >> DSH, d = e & DM1;
                const int64_t a = kv_row2(P, s, kh, jb + r, jend);
                /* slot walk exactly as v1: K at d*kv_sd, V at (D+d)*kv_sd (the slot
                 * is 2*head_dim ELEMENTS wide -- kv_sd scales the D too, raw-D was
                 * the draft bug that a kv_sd==1 fixture would have hidden). */
                const int64_t eoff = (int64_t)((isV ? D : 0) + d) * P.kv_sd;
                pf[p] = a < 0 ? (uint16_t)0
                      : (KVDT == RAD_BF16) ? ((const uint16_t*)P.kv)[a + eoff]
                      : fp8_to_bf16_w(((const uint8_t*)P.kv)[a + eoff]);
            }
        };
        auto land = [&](bf16* buf) {
            #pragma unroll
            for (int p = 0; p < PFN; ++p) {
                const int e = tid + p * kT;
                bf16_store(buf + (e >> DSH) * C::kVP + (e & DM1), pf[p]);
            }
        };

        /* ================= pass one: running max m and exp-sum l ================= */
        /* prologue: slice et0 + its column-badness bytes land before the loop */
        issue((int64_t)et0 * kBN, 0);
        land(K0);
        if (tid < kBN) CB[tid] = kv_row2(P, s, kh, (int64_t)et0 * kBN + tid, jend) < 0;
        __syncthreads();

        for (int t = et0; t < et1; ++t) {
            const int cur = (t - et0) & 1;
            bf16* Kcur = cur ? K1 : K0;
            const bool has_next = (t + 1 < et1);
            const int64_t jb_n = has_next ? (int64_t)(t + 1) * kBN : 0;
            if (has_next) issue(jb_n, 0);          /* gather in flight during mma */

            const int64_t jb = (int64_t)t * kBN;
            roCW::fill_fragment(acc, 0.0f);
            #pragma unroll
            for (int k0 = 0; k0 < D / 16; ++k0) {
                roCW::load_matrix_sync(fb0, Kcur + (cq * 16) * C::kVP + k0 * 16, C::kVP);
                roCW::mma_sync(acc, fa[k0], fb0, acc);
            }
            roCW::store_matrix_sync(Ssp + (rb_own * 16) * C::kSP + cq * 16, acc,
                                    C::kSP, roCW::mem_row_major);
            __syncthreads();                     /* Ssp visible; K(t) fragment reads done */

            if (tid < kBM && tid < valid) {      /* softmax, v1's arithmetic verbatim */
                const int64_t rowpos = pos_base + tid;
                const float* srow = Ssp + tid * C::kSP;
                float mx = MSP[tid];
                for (int jj = 0; jj < kBN; ++jj) {
                    if (CB[jj] || !visible(rowpos, jb + jj)) continue;
                    const float v = srow[jj] * scale2;
                    if (v > mx) mx = v;
                }
                if (mx != -INFINITY) {
                    if (mx > MSP[tid]) {
                        if (MSP[tid] != -INFINITY) LSP[tid] *= exp2f(MSP[tid] - mx);
                        MSP[tid] = mx;
                    }
                    float sum = 0.0f;
                    for (int jj = 0; jj < kBN; ++jj) {
                        if (CB[jj] || !visible(rowpos, jb + jj)) continue;
                        sum += exp2f(srow[jj] * scale2 - mx);
                    }
                    LSP[tid] += sum;
                }
            }
            if (has_next) land(cur ^ 1 ? K1 : K0);   /* the OTHER buffer: no barrier needed */
            __syncthreads();                     /* K(t+1) landed; softmax(t) readers done */
            /* The bad-byte write rides AFTER this barrier: softmax(t) read CB(t) and is
             * done at the barrier; softmax(t+1) sees the new bytes past the next S-store
             * barrier. A same-phase read/write on CB is exactly the race v1's dedicated
             * addressing barrier prevented -- it may not reappear as a "freed" barrier. */
            if (has_next && tid < kBN) CB[tid] = kv_row2(P, s, kh, jb_n + tid, jend) < 0;
        }

        /* ===================== pass two: O = P.V at the final m ===================== */
        /* K(et0) re-gathers behind this barrier (pass one's last slice sits in the
         * buffer pass two's first store targets); V(t) stages between P and the O mma
         * with the same register-prefetch depth. 2 barriers per step, as pass one. */
        issue((int64_t)et0 * kBN, 0);
        __syncthreads();                         /* Ssp/K readers of pass one's tail done */
        land(K0);
        if (tid < kBN) CB[tid] = kv_row2(P, s, kh, (int64_t)et0 * kBN + tid, jend) < 0;
        __syncthreads();

        for (int t = et0; t < et1; ++t) {
            const int cur = (t - et0) & 1;
            bf16* Kcur = cur ? K1 : K0;
            const bool has_next = (t + 1 < et1);
            const int64_t jb_n = has_next ? (int64_t)(t + 1) * kBN : 0;
            if (has_next) issue(jb_n, 0);

            const int64_t jb = (int64_t)t * kBN;
            roCW::fill_fragment(acc, 0.0f);       /* S again: same fragments, same order */
            #pragma unroll
            for (int k0 = 0; k0 < D / 16; ++k0) {
                roCW::load_matrix_sync(fb0, Kcur + (cq * 16) * C::kVP + k0 * 16, C::kVP);
                roCW::mma_sync(acc, fa[k0], fb0, acc);
            }
            roCW::store_matrix_sync(Ssp + (rb_own * 16) * C::kSP + cq * 16, acc,
                                    C::kSP, roCW::mem_row_major);
            __syncthreads();                     /* Ssp done; V(t-1)'s O-mma done everywhere */

            for (int e = tid; e < kBM * kBN; e += kT) {   /* S -> P, all eight waves */
                const int r = e >> 5, jj = e & 31;
                float pv = 0.0f;
                if (r < valid && !CB[jj] && visible(pos_base + r, jb + jj)) {
                    const float m = MSP[r];
                    if (m != -INFINITY) pv = exp2f(Ssp[r * C::kSP + jj] * scale2 - m);
                }
                bf16_store(Psp + r * C::kPP + jj, f32_bf16_w(pv));
            }
            if (has_next) land(cur ^ 1 ? K1 : K0);
            issue(jb, 1);                        /* V(t): gather issued after P's Ssp reads */
            __syncthreads();                     /* Psp done (CB(t) readers included); V(t) may land */
            if (has_next && tid < kBN) CB[tid] = kv_row2(P, s, kh, jb_n + tid, jend) < 0;
            land(Vp);
            __syncthreads();                     /* Vp visible to every O-mma */

            #pragma unroll
            for (int jk = 0; jk < kBN / 16; ++jk) {
                roCW::fragment<roCW::matrix_a, 16, 16, 16, bf16, roCW::row_major> fap;
                roCW::load_matrix_sync(fap, Psp + (rb_own * 16) * C::kPP + jk * 16,
                                       C::kPP, roCW::mem_row_major);
                #pragma unroll
                for (int i = 0; i < D / 32; ++i) {
                    const int d0 = (D / 32) * cq + i;
                    roCW::load_matrix_sync(fbv, Vp + (jk * 16) * C::kVP + d0 * 16,
                                           C::kVP, roCW::mem_row_major);
                    roCW::mma_sync(oa[i], fap, fbv, oa[i]);
                }
            }
            /* no barrier here: Psp's next write is guarded by pass two's step-head
             * barrier, and K(t+1) landed into a buffer nobody reads past it. */
        }

        /* ===================== out, v1's halves on 8 waves ===================== */
        float* Osp = reinterpret_cast<float*>(raw + C::offK0);   /* K0+K1+V: 25344 >= 17408 */

        if (SPLITK) {
            float* pso = reinterpret_cast<float*>(P.pf);
            const int64_t pn = P.rows * P.q_heads;
            float* psm = pso + pn * spl * D, * psl = psm + pn * spl;
            for (int half = 0; half < 2; ++half) {
                __syncthreads();
                if (((rb_own < 2) ? 0 : 1) == half) {           /* rows 0-31 / 32-63 */
                    const int wr = rb_own - 2 * half;
                    #pragma unroll
                    for (int i = 0; i < D / 32; ++i)
                        roCW::store_matrix_sync(Osp + (wr * 16) * C::kOP + ((D / 32) * cq + i) * 16,
                                                oa[i], C::kOP, roCW::mem_row_major);
                }
                __syncthreads();
                for (int e = tid; e < 32 * D; e += kT) {
                    const int r = e >> DSH, d = e & DM1;
                    const int row = half * 32 + r;
                    if (row >= valid) continue;
                    const int64_t idx = ((tok_base + row) * P.q_heads + h) * spl + sp;
                    pso[idx * D + d] = Osp[r * C::kOP + d];
                }
                if (tid < 32) {
                    const int row = half * 32 + tid;
                    if (row < valid) {
                        const int64_t idx = ((tok_base + row) * P.q_heads + h) * spl + sp;
                        psm[idx] = MSP[row];
                        psl[idx] = LSP[row];
                    }
                }
            }
            return;
        }
        for (int half = 0; half < 2; ++half) {
            __syncthreads();
            if (((rb_own < 2) ? 0 : 1) == half) {               /* PREDICATED, never continued */
                const int wr = rb_own - 2 * half;
                #pragma unroll
                for (int i = 0; i < D / 32; ++i)
                    roCW::store_matrix_sync(Osp + (wr * 16) * C::kOP + ((D / 32) * cq + i) * 16,
                                            oa[i], C::kOP, roCW::mem_row_major);
            }
            __syncthreads();
            for (int e = tid; e < 32 * D; e += kT) {
                const int r = e >> DSH, d = e & DM1;
                const int row = half * 32 + r;
                if (row >= valid) continue;
                const float ll = LSP[row];
                const float o = Osp[r * C::kOP + d] * (ll > 0.0f ? 1.0f / ll : 0.0f) * vds;
                const int64_t ob = (tok_base + row) * P.o_rs + (int64_t)h * P.o_hs
                                 + (int64_t)d * P.o_ds;
                if (P.out_f32) ((float*)P.out)[ob] = o;
                else           ((uint16_t*)P.out)[ob] = f32_bf16_w(o);
            }
        }
    }
}
