/* r3d_attn_wmma_body.inc -- the WMMA flash-attention tile, shared body text.

 * Included (inside the gfx1100..gfx1103 gate, after <rocwmma/rocwmma.hpp> and the
 * roCW/bf16 aliases, inside each TU's second anonymous namespace) by:
 *   r3d_attn_paged_wmma.hip    -- SPLITK=0, the prefill/any-length tile, out direct;
 *   r3d_attn_paged_decode.hip  -- SPLITK=1, the split-KV decode slice, partials to the
 *                                 engine scratch arena for the merge kernel.
 * Each TU instantiates only its own SPLITK, so the .so holds each instantiation once.
 * The body arithmetic -- fragment staging, the two-pass order, the visibility rules,
 * the O staging -- is then shared by construction, which is the point: a slice's
 * partials merge to what one full pass would have produced, and a splits=1 launch is
 * the SPLITK=0 tail under a different final store, not a re-implementation of it.
 *
 * The includer must define first: the dtype helpers and kBM/kBN/kThreads/kLog2e (the
 * verbatim helper block of r3d_attn_paged_wmma.hip, duplicated into the decode TU under
 * the same "most-reviewed lines" contract), R3dAttnParams (r3d_attn.h), rad_abi.h.

 * SPLITK=1 entry contract (r3d_attn_paged_decode.hip sets these): P.splits >= 1,
 * P.pf = arena base; grid z = tiles*splits; blockIdx.z = qt*splits + sp. Partials, per
 * live query row (tok global row index): see the tail's comment for the layout. An
 * empty slice stores m=-inf, l=0 and no O; a dead query tile (i0 >= qlen) stores
 * nothing, exactly as the SPLITK=0 tile -- the merge grid covers live rows only.
 */

/* bf16(x) on rocwmma::bfloat16_t CONVERTS NUMERICALLY (the same trap as the layout
 * probes: casting bf16 fragments to integer truncates). Fragment staging carries raw
 * bf16 BIT patterns -- write them, never a constructed bf16. */
__device__ __forceinline__ void bf16_store(bf16* p, uint16_t bits) {
    *reinterpret_cast<uint16_t*>(p) = bits;
}

/* LDS carve, byte offsets into one buffer (16-aligned), per head_dim:
 *   Q   64 x (D+4) bf16   query tile, staged once per workgroup
 *   S   64 x 40    f32    score tile (pad 8: 32 real columns, row-bank spread)
 *   P   64 x 36    bf16   exp'd probabilities, pass two only
 *   KV  32 x (D+4) bf16   key tile, THEN the value tile (aliased: V stages after the
 *                         tile's S-fragments have left the LDS, one sync apart)
 *   meta m/l f32[64], rowbase i64[32], colbad u8[32]
 * O-store staging (32 x (D+8) f32, halves of the output) aliases Q+S after pass two's
 * last fragment read. D=128 total: 40992 B static -- under the 48 KiB static ceiling,
 * no hipFuncSetAttribute dance. */
template <int D> struct Carve {
    static constexpr int kQP = D + 4, kVP = D + 4, kSP = kBN + 8, kPP = kBN + 4;
    static constexpr int kOP = D + 8;
    static constexpr int offQ  = 0;
    static constexpr int offS  = (offQ + kBM * kQP * 2 + 15) & ~15;
    static constexpr int offP  = (offS + kBM * kSP * 4 + 15) & ~15;
    static constexpr int offKV = (offP + kBM * kPP * 2 + 15) & ~15;
    static constexpr int offM  = (offKV + kBN * kVP * 2 + 15) & ~15;   /* + m[64] l[64] */
    static constexpr int offRB = offM + 2 * kBM * 4;
    static constexpr int offCB = offRB + kBN * 8;
    static constexpr int total = offCB + kBN + 16;
};

template <int D, int KVDT, int SPLITK>
__global__ __launch_bounds__(kThreads) void attn_paged_wmma_kernel(R3dAttnParams P) {
    constexpr int DSH = (D == 128) ? 7 : 6, DM1 = D - 1;
    typedef Carve<D> C;
    __shared__ unsigned char raw[C::total];

    bf16*   Qsp = reinterpret_cast<bf16*>(raw + C::offQ);
    float*  Ssp = reinterpret_cast<float*>(raw + C::offS);
    bf16*   Psp = reinterpret_cast<bf16*>(raw + C::offP);
    bf16*   KVp = reinterpret_cast<bf16*>(raw + C::offKV);
    float*  MSP = reinterpret_cast<float*>(raw + C::offM);
    float*  LSP = MSP + kBM;
    int64_t* RB = reinterpret_cast<int64_t*>(raw + C::offRB);
    unsigned char* CB = raw + C::offCB;
#ifdef R3D_DBG
    /* Debug wire: kdes carries a double[64] dump buffer (the repro's REPRO_DBG=1 arm);
     * descales are forced off on such runs (REPRO_NOP=1 pairs with it). Device printf is
     * NOT an option in this kernel: printf + a >32 KB static-LDS launch fails with
     * hipErrorIllegalState(401) on ROCm 10.1 -- that detour cost a session. */
    double* dbg = nullptr;
    {
        const bool dbg_wg = (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0);
        if (dbg_wg) { dbg = (threadIdx.x < 32 && P.kdes)
                                ? reinterpret_cast<double*>(const_cast<RadTensor*>(P.kdes)->data)
                                : nullptr;
                      P.kdes = nullptr; P.vdes = nullptr; }   /* NOP arith in the dump wg */
    }
#endif

    const int h = blockIdx.x, s = blockIdx.y;
    /* SPLITK=0: blockIdx.z is the query tile, as always. SPLITK=1: the z axis
     * carries (query tile, KV slice) -- one block per (head, seq, tile, slice). */
    const int spl = (SPLITK && P.splits > 0) ? P.splits : 1;
    const int qt  = SPLITK ? (int)blockIdx.z / spl : (int)blockIdx.z;
    const int sp  = SPLITK ? (int)blockIdx.z % spl : 0;
    const int tid = threadIdx.x, w = tid >> 5;

    /* ---- per-block bookkeeping. blockIdx carries (head, seq, query tile) directly, so
     * none of v1's linear seq_of scan runs on device; every thread reads the two or three
     * uniform scalars itself -- cheaper than a thread-0 broadcast at 128 threads. ---- */
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

    /* ref's key_at visibility rules, minus the addressing (key_at's mask half). */
    const int causal = P.causal; const int64_t window = P.window;
    auto visible = [&](int64_t rowpos, int64_t j) -> bool {
        if (causal && j > rowpos) return false;
        if (window > 0) {
            const int64_t d = rowpos - j;
            if (causal ? (d >= window) : (d >= window || d <= -window)) return false;
        }
        return true;
    };

    /* Tile range that any row can see (skip the dead head of a window, stop past the
     * last position any row may attend to). */
    const int64_t pos_last = pos_base + valid - 1;
    int64_t jmax = jend;
    if (causal && pos_last + 1 < jmax) jmax = pos_last + 1;
    if (!causal && window > 0 && pos_last + window < jmax) jmax = pos_last + window;
    int64_t jmin = 0;
    if (window > 0) jmin = max<int64_t>(0, pos_base - window + 1);
    const int t0 = (int)(jmin / kBN);
    const int t1 = (int)((jmax + kBN - 1) / kBN);

    /* ---- split-KV: this block's effective tile range is the [a,b) slice of [t0,t1)
     * cut by (sp, spl). The cut is pure integer arithmetic on (t1-t0, sp, spl), so the
     * merge kernel and the arena-sizing hook can recompute it exactly. A block whose
     * slice is empty stores the inert partial -- m=-inf, l=0, and NO O bytes (the merge
     * skips the O read when it sees m==-inf) -- and leaves; everything above this point
     * read only block-uniform values, so no thread is stranded at a barrier. ---- */
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
            for (int64_t row = tid; row < valid; row += kThreads) {
                const int64_t idx = ((tok_base + row) * P.q_heads + h) * spl + sp;
                psm[idx] = -INFINITY; psl[idx] = 0.0f;
            }
            return;
        }
    }

    /* ---- stage Q once: rows tok_base..+63 (invalid rows zeroed; they mask everywhere
     * and never store), columns 0..head_dim-1. ---- */
    const int64_t qbase = tok_base * P.q_rs + (int64_t)h * P.q_hs;
    for (int e = tid; e < kBM * D; e += kThreads) {
        const int r = e >> DSH, d = e & DM1;
        bf16_store(Qsp + r * C::kQP + d, (r < valid)
            ? ((const uint16_t*)P.q)[qbase + (int64_t)r * P.q_rs + (int64_t)d * P.q_ds]
            : (uint16_t)0);
    }
    if (tid < kBM) { MSP[tid] = -INFINITY; LSP[tid] = 0.0f; }
    __syncthreads();
#ifdef R3D_DBG
    if (dbg && tid == 0) {
        dbg[54] = (double)bf16_f32_w(((const uint16_t*)Qsp)[0]);
        dbg[57] = (double)(unsigned)((const uint16_t*)Qsp)[0];   /* raw LDS bits */
        dbg[58] = (double)(unsigned)((const uint16_t*)P.q)[qbase]; /* global truth */
    }
#endif

    roCW::fragment<roCW::matrix_a, 16, 16, 16, bf16, roCW::row_major> fa;
    roCW::fragment<roCW::matrix_b, 16, 16, 16, bf16, roCW::col_major> fb0, fb1;
    roCW::fragment<roCW::accumulator, 16, 16, 16, float> a0, a1;

    /* ================= pass one: per-row running max m and exp-sum l ================= */
    for (int t = et0; t < et1; ++t) {
        const int64_t jb = (int64_t)t * kBN;

        /* key addressing for the 32 columns (one thread per key, then broadcast) */
        if (tid < kBN) {
            const int64_t j = jb + tid;
            int64_t rb = -1;
            if (j < jend) {
                const int64_t blk = (int64_t)ldi_w(P.bt_dt, P.bt,
                                                   s * P.bt_rs + (j / P.block_size) * P.bt_cs);
                if (blk >= 0)
                    rb = blk * P.kv_sb + kh * P.kv_sh + (j % P.block_size) * P.kv_so;
            }
            RB[tid] = rb; CB[tid] = (rb < 0) ? 1 : 0;
        }
        __syncthreads();

        for (int e = tid; e < kBN * D; e += kThreads) {
            const int r = e >> DSH, d = e & DM1;
            uint16_t u = 0;
            if (RB[r] >= 0) {
                const int64_t a = RB[r] + (int64_t)d * P.kv_sd;
                u = (KVDT == RAD_BF16) ? ((const uint16_t*)P.kv)[a]
                                       : fp8_to_bf16_w(((const uint8_t*)P.kv)[a]);
#ifdef R3D_DBG
                if (dbg && t == t0 && e == 0) {
                    dbg[0] = (double)RB[r];
                    dbg[1] = (double)(KVDT == RAD_BF16 ? ((const uint16_t*)P.kv)[a]
                                                      : (int)((const uint8_t*)P.kv)[a]);
                    dbg[2] = (double)u;
                }
#endif
            }
            bf16_store(KVp + r * C::kVP + d, u);
        }
        __syncthreads();
#ifdef R3D_DBG
        if (dbg && t == t0 && tid == 0) {
            dbg[55] = (double)bf16_f32_w(((const uint16_t*)KVp)[0]);
            dbg[56] = (double)bf16_f32_w(((const uint16_t*)Qsp)[0]);
            dbg[59] = (double)(unsigned)((const uint16_t*)KVp)[0];   /* raw LDS bits */
            dbg[60] = (double)(unsigned)((const uint16_t*)Qsp)[0];
        }
#endif

        /* S = Q.K^T: matrix_a row-major at &Qsp[16w][k0]; the [k][n] B tile is K^T, and
         * col-major at &KVp[n0][k0] means element (k,n) = ptr[n*VP + k] = KVp[n][k]. */
        roCW::fill_fragment(a0, 0.0f);
        roCW::fill_fragment(a1, 0.0f);
        for (int k0 = 0; k0 < D; k0 += 16) {
            roCW::load_matrix_sync(fa,  Qsp + (w * 16) * C::kQP + k0, C::kQP, roCW::mem_row_major);
            roCW::load_matrix_sync(fb0, KVp + k0, C::kVP);
            roCW::load_matrix_sync(fb1, KVp + 16 * C::kVP + k0, C::kVP);
            roCW::mma_sync(a0, fa, fb0, a0);
            roCW::mma_sync(a1, fa, fb1, a1);
#ifdef R3D_DBG
            if (dbg && t == t0 && k0 == 0 && tid == 0) {   /* what the first mma saw */
                for (int i = 0; i < 8; ++i) {
                    dbg[18 + i] = (double)fa.x[i];
                    dbg[26 + i] = (double)fb0.x[i];
                    dbg[42 + i] = (double)a0.x[i];
                }
                dbg[50] = (double)bf16_f32_w(((const uint16_t*)KVp)[0]);
                dbg[51] = (double)bf16_f32_w(((const uint16_t*)KVp)[1]);
                dbg[52] = (double)bf16_f32_w(((const uint16_t*)Qsp)[0]);
                dbg[53] = (double)bf16_f32_w(((const uint16_t*)Qsp)[1]);
            }
#endif
        }
        roCW::store_matrix_sync(Ssp + (w * 16) * C::kSP,      a0, C::kSP, roCW::mem_row_major);
        roCW::store_matrix_sync(Ssp + (w * 16) * C::kSP + 16, a1, C::kSP, roCW::mem_row_major);
        __syncthreads();

        /* one thread per query row: visible max, then the online update of (m, l) */
        if (tid < kBM && tid < valid) {
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
#ifdef R3D_DBG
        if (dbg && t == t0 && tid == 0) {
            dbg[3] = (double)Ssp[0];
            dbg[4] = (double)Ssp[16];
            dbg[5] = (double)MSP[0];
            dbg[6] = (double)LSP[0];
            dbg[7] = (double)scale2;
        }
#endif
        __syncthreads();
    }

    /* ===================== pass two: O = P.V, P at the final m ====================== */
    roCW::fragment<roCW::accumulator, 16, 16, 16, float> oa[D / 16];
    #pragma unroll
    for (int i = 0; i < D / 16; ++i) roCW::fill_fragment(oa[i], 0.0f);

    for (int t = et0; t < et1; ++t) {
        const int64_t jb = (int64_t)t * kBN;

        if (tid < kBN) {                       /* key addressing, as pass one */
            const int64_t j = jb + tid;
            int64_t rb = -1;
            if (j < jend) {
                const int64_t blk = (int64_t)ldi_w(P.bt_dt, P.bt,
                                                   s * P.bt_rs + (j / P.block_size) * P.bt_cs);
                if (blk >= 0)
                    rb = blk * P.kv_sb + kh * P.kv_sh + (j % P.block_size) * P.kv_so;
            }
            RB[tid] = rb; CB[tid] = (rb < 0) ? 1 : 0;
        }
        __syncthreads();
        for (int e = tid; e < kBN * D; e += kThreads) {
            const int r = e >> DSH, d = e & DM1;
            uint16_t u = 0;
            if (RB[r] >= 0) {
                const int64_t a = RB[r] + (int64_t)d * P.kv_sd;
                u = (KVDT == RAD_BF16) ? ((const uint16_t*)P.kv)[a]
                                       : fp8_to_bf16_w(((const uint8_t*)P.kv)[a]);
            }
            bf16_store(KVp + r * C::kVP + d, u);
        }
        __syncthreads();

        roCW::fill_fragment(a0, 0.0f);         /* S again: same fragments, same order, */
        roCW::fill_fragment(a1, 0.0f);         /* so P here matches pass one's (m, l)  */
        for (int k0 = 0; k0 < D; k0 += 16) {
            roCW::load_matrix_sync(fa,  Qsp + (w * 16) * C::kQP + k0, C::kQP, roCW::mem_row_major);
            roCW::load_matrix_sync(fb0, KVp + k0, C::kVP);
            roCW::load_matrix_sync(fb1, KVp + 16 * C::kVP + k0, C::kVP);
            roCW::mma_sync(a0, fa, fb0, a0);
            roCW::mma_sync(a1, fa, fb1, a1);
        }
        roCW::store_matrix_sync(Ssp + (w * 16) * C::kSP,      a0, C::kSP, roCW::mem_row_major);
        roCW::store_matrix_sync(Ssp + (w * 16) * C::kSP + 16, a1, C::kSP, roCW::mem_row_major);
        __syncthreads();

        for (int e = tid; e < kBM * kBN; e += kThreads) {   /* S -> P, all four waves */
            const int r = e >> 5, jj = e & 31;
            float pv = 0.0f;
            if (r < valid && !CB[jj] && visible(pos_base + r, jb + jj)) {
                const float m = MSP[r];
                if (m != -INFINITY) pv = exp2f(Ssp[r * C::kSP + jj] * scale2 - m);
            }
            bf16_store(Psp + r * C::kPP + jj, f32_bf16_w(pv));
        }
#ifdef R3D_DBG
        if (dbg && t == t0 && tid == 0) {
            dbg[8] = (double)Ssp[0];
            dbg[9] = (double)bf16_f32_w(((const uint16_t*)Psp)[0]);
            dbg[10] = (double)MSP[0];
        }
#endif
        __syncthreads();                       /* this tile's S is consumed: KV is free */

        for (int e = tid; e < kBN * D; e += kThreads) {     /* V tile, aliasing K */
            const int r = e >> DSH, d = e & DM1;
            uint16_t u = 0;
            if (RB[r] >= 0) {
                const int64_t a = RB[r] + ((int64_t)D + d) * P.kv_sd;
                u = (KVDT == RAD_BF16) ? ((const uint16_t*)P.kv)[a]
                                       : fp8_to_bf16_w(((const uint8_t*)P.kv)[a]);
#ifdef R3D_DBG
                if (dbg && t == t0 && e == 0) {
                    dbg[11] = (double)a;
                    dbg[12] = (double)(KVDT == RAD_BF16 ? ((const uint16_t*)P.kv)[a]
                                                        : (int)((const uint8_t*)P.kv)[a]);
                    dbg[13] = (double)u;
                }
#endif
            }
            bf16_store(KVp + r * C::kVP + d, u);
        }
        __syncthreads();

        /* O += P.V: A = P row-major [16w][jk]; the V tile IS [key][d] = row-major B. */
        for (int jk = 0; jk < kBN / 16; ++jk) {
            roCW::fragment<roCW::matrix_a, 16, 16, 16, bf16, roCW::row_major> fap;
            roCW::load_matrix_sync(fap, Psp + (w * 16) * C::kPP + jk * 16, C::kPP,
                                   roCW::mem_row_major);
            #pragma unroll
            for (int d0 = 0; d0 < D / 16; ++d0) {
                roCW::fragment<roCW::matrix_b, 16, 16, 16, bf16, roCW::row_major> fbv;
                roCW::load_matrix_sync(fbv, KVp + (jk * 16) * C::kVP + d0 * 16, C::kVP,
                                       roCW::mem_row_major);
                roCW::mma_sync(oa[d0], fap, fbv, oa[d0]);
            }
        }
        __syncthreads();
    }

    /* ===================== out = (O / l) * vds, in 32-row halves ===================== */
    float* Osp = reinterpret_cast<float*>(raw + C::offQ);   /* aliases Q+S, both dead */

    if (SPLITK) {
        /* Partials, UN-normalized, to the engine scratch arena:
         *   so[((tok*q_heads + h)*spl + sp)*D + d] = O column d
         *   sm[(tok*q_heads + h)*spl + sp] = row max m   (exp2 domain, as tracked here)
         *   sl[...]                        = exp-sum l
         * The merge divides once per output row; with one live split it degenerates to
         * exactly the direct-out arithmetic below (w = exp2(m-m) = 1.0 exactly, den = l,
         * num = O), which is why a splits=1 A/B arm measures the tile, not the merge.
         * The halves reuse the proven O staging -- one wave's accumulators at a time
         * through the Osp alias -- with the same predicate-not-`continue` discipline
         * around the syncthreads as the direct-out halves below. */
        float* pso = reinterpret_cast<float*>(P.pf);
        const int64_t pn = P.rows * P.q_heads;
        float* psm = pso + pn * spl * D, * psl = psm + pn * spl;
        for (int half = 0; half < 2; ++half) {
            __syncthreads();
            if ((half == 0) ? (w < 2) : (w >= 2)) {
                const int wr = w - 2 * half;        /* relative tile-row: 0 or 1 */
                #pragma unroll
                for (int d0 = 0; d0 < D / 16; ++d0)
                    roCW::store_matrix_sync(Osp + (wr * 16) * C::kOP + d0 * 16, oa[d0], C::kOP,
                                            roCW::mem_row_major);
            }
            __syncthreads();
            for (int e = tid; e < 32 * D; e += kThreads) {
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
        /* wave owns rows [16w,16w+16): waves 0/1 fill half 0, waves 2/3 fill half 1.
         * PREDICATED, never `continue`d: every thread must reach both syncthreads. */
        if ((half == 0) ? (w < 2) : (w >= 2)) {
            const int wr = w - 2 * half;        /* relative tile-row: 0 or 1 */
            #pragma unroll
            for (int d0 = 0; d0 < D / 16; ++d0)
                roCW::store_matrix_sync(Osp + (wr * 16) * C::kOP + d0 * 16, oa[d0], C::kOP,
                                        roCW::mem_row_major);
        }
        __syncthreads();
        for (int e = tid; e < 32 * D; e += kThreads) {
            const int r = e >> DSH, d = e & DM1;
            const int row = half * 32 + r;
            if (row >= valid) continue;
            const float ll = LSP[row];
            const float o = Osp[r * C::kOP + d] * (ll > 0.0f ? 1.0f / ll : 0.0f) * vds;
#ifdef R3D_DBG
            if (dbg && half == 0 && e == 0) {
                dbg[14] = (double)ll;
                dbg[15] = (double)Osp[0];
                dbg[16] = (double)o;
                dbg[17] = (double)vds;
            }
#endif
            const int64_t ob = (tok_base + row) * P.o_rs + (int64_t)h * P.o_hs
                             + (int64_t)d * P.o_ds;
            if (P.out_f32) ((float*)P.out)[ob] = o;
            else           ((uint16_t*)P.out)[ob] = f32_bf16_w(o);
        }
    }
}
