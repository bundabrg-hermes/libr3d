/* r3d_attn_wmma_body3.inc -- the v3 prefill tile: v1's schedule on a THIN block.
 *
 * The post-mortem (docs/COVERAGE.md, 2026-10-20, v2 refutation): barrier COUNT is not
 * the serialization term -- OCCUPANCY is. v1's 64-row query tile carves ~41 KB LDS at
 * head_dim=128 (Carve<128>::total = 41008 B), which fits ONE block per 64 KB CU, so at
 * hd=128 every barrier already stalls a CU with nothing else resident; v2 tried to fix
 * the schedule by making the block BIGGER and lost 0.43-0.80x. The axis the split-KV
 * decode win rode is MORE, THINNER blocks, and 32-row query tiles are the shrink the
 * arithmetic survives unchanged.
 *
 * This body is v1's (r3d_attn_wmma_body.inc) text with kBM 64 -> 32 and kThreads
 * 128 -> 64. Everything per-row is deliberately untouched -- fragment order, the
 * two-pass recompute, the visibility rules, the fp8/bf16 staging, the exp2 domain --
 * so a row's bits are identical to v1's for the same input, and the bench's FNV hash
 * gate (tools/attn_bench.hip) can demand bit-for-bit v1/v3 equality as v2's did.
 * What changes is only the block's shape:
 *   - one wave per 16-row half (2 waves, not 4): the O store loops the same 2 phases
 *     as v1's halves, but each phase is ONE wave storing its own 16 rows;
 *   - LDS at D=128: Q 8448 + S 5120 + P 2304 + KV 8448 + meta ~512 = ~24.9 KB ->
 *     2 blocks/CU (was 1); at D=64 ~16.9 KB -> 3 blocks/CU (was 2);
 *   - the KV read doubles per query row (twice as many query tiles re-walk the KV) --
 *     accepted because the deep-ctx cases run at 15 GB/s against a 500 GB/s card: the
 *     term being bought out is 30x under its bandwidth ceiling, and the bench's
 *     kv_gbs column reports exactly what that trade cost if v3 wins on time.
 * SPLITK does not belong here: the split-KV band is the decode row's, and this body
 * carries only the direct-out tail. If v3's schedule ever wants slices, it takes
 * them through the decode TU's instantiation of body.inc, not by growing this text.
 *
 * Included (inside the chip gate, after the roCW aliases) ONLY by
 * r3d_attn_paged_wmma.hip, which instantiates attn_paged_wmma_kernel3<D, KVDT> and
 * dispatches it behind R3D_ATTN_V3=1 (opt-in, v2's promotion pattern inverted).
 * Requires the includer's helper block, kBN/kLog2e, R3dAttnParams, rad_abi.h.
 * The R3D_DBG dump arm lives in v1's body only -- v3 is a perf arm; forensics stay
 * on the text the serve actually runs.
 */

template <int D> struct Carve3 {
    static constexpr int kQP = D + 4, kVP = D + 4, kSP = kBN + 8, kPP = kBN + 4;
    static constexpr int kOP = D + 8;
    static constexpr int offQ  = 0;
    static constexpr int offS  = (offQ + kBM3 * kQP * 2 + 15) & ~15;
    static constexpr int offP  = (offS + kBM3 * kSP * 4 + 15) & ~15;
    static constexpr int offKV = (offP + kBM3 * kPP * 2 + 15) & ~15;
    static constexpr int offM  = (offKV + kBN * kVP * 2 + 15) & ~15;   /* + m[32] l[32] */
    static constexpr int offRB = offM + 2 * kBM3 * 4;
    static constexpr int offCB = offRB + kBN * 8;
    static constexpr int total = offCB + kBN + 16;
};

template <int D, int KVDT>
__global__ __launch_bounds__(kThreads3) void attn_paged_wmma_kernel3(R3dAttnParams P) {
    constexpr int DSH = (D == 128) ? 7 : 6, DM1 = D - 1;
    typedef Carve3<D> C;
    __shared__ unsigned char raw[C::total];

    bf16*   Qsp = reinterpret_cast<bf16*>(raw + C::offQ);
    float*  Ssp = reinterpret_cast<float*>(raw + C::offS);
    bf16*   Psp = reinterpret_cast<bf16*>(raw + C::offP);
    bf16*   KVp = reinterpret_cast<bf16*>(raw + C::offKV);
    float*  MSP = reinterpret_cast<float*>(raw + C::offM);
    float*  LSP = MSP + kBM3;
    int64_t* RB = reinterpret_cast<int64_t*>(raw + C::offRB);
    unsigned char* CB = raw + C::offCB;

    const int h = blockIdx.x, s = blockIdx.y, qt = blockIdx.z;
    const int tid = threadIdx.x, w = tid >> 5;         /* 2 waves, 16 rows each */

    int64_t start, qlen;
    if (P.cus) {
        start = (int64_t)ldi_w(P.cus_dt, P.cus, s);
        qlen  = (int64_t)ldi_w(P.cus_dt, P.cus, s + 1) - start;
    } else {
        qlen  = P.q_len_uniform;
        start = (int64_t)s * qlen;
    }
    const int64_t i0 = (int64_t)qt * kBM3;
    if (i0 >= qlen) return;
    const int64_t tok_base = start + i0;
    const int64_t valid    = min<int64_t>(qlen - i0, kBM3);
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
    const int et0 = (int)(jmin / kBN);
    const int et1 = (int)((jmax + kBN - 1) / kBN);

    /* ---- stage Q: 32 rows, invalid rows zeroed (mask everywhere, never store) ---- */
    const int64_t qbase = tok_base * P.q_rs + (int64_t)h * P.q_hs;
    for (int e = tid; e < kBM3 * D; e += kThreads3) {
        const int r = e >> DSH, d = e & DM1;
        bf16_store(Qsp + r * C::kQP + d, (r < valid)
            ? ((const uint16_t*)P.q)[qbase + (int64_t)r * P.q_rs + (int64_t)d * P.q_ds]
            : (uint16_t)0);
    }
    if (tid < kBM3) { MSP[tid] = -INFINITY; LSP[tid] = 0.0f; }
    __syncthreads();

    roCW::fragment<roCW::matrix_a, 16, 16, 16, bf16, roCW::row_major> fa;
    roCW::fragment<roCW::matrix_b, 16, 16, 16, bf16, roCW::col_major> fb0, fb1;
    roCW::fragment<roCW::accumulator, 16, 16, 16, float> a0, a1;

    /* ================= pass one: per-row running max m and exp-sum l ================= */
    for (int t = et0; t < et1; ++t) {
        const int64_t jb = (int64_t)t * kBN;

        if (tid < kBN) {                       /* key addressing (one thread per key) */
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

        for (int e = tid; e < kBN * D; e += kThreads3) {
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

        roCW::fill_fragment(a0, 0.0f);
        roCW::fill_fragment(a1, 0.0f);
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

        if (tid < kBM3 && tid < valid) {       /* one thread per query row */
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
        for (int e = tid; e < kBN * D; e += kThreads3) {
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

        roCW::fill_fragment(a0, 0.0f);         /* S again: same fragments, same order */
        roCW::fill_fragment(a1, 0.0f);
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

        for (int e = tid; e < kBM3 * kBN; e += kThreads3) {  /* S -> P, both waves */
            const int r = e >> 5, jj = e & 31;
            float pv = 0.0f;
            if (r < valid && !CB[jj] && visible(pos_base + r, jb + jj)) {
                const float m = MSP[r];
                if (m != -INFINITY) pv = exp2f(Ssp[r * C::kSP + jj] * scale2 - m);
            }
            bf16_store(Psp + r * C::kPP + jj, f32_bf16_w(pv));
        }
        __syncthreads();                       /* this tile's S is consumed: KV is free */

        for (int e = tid; e < kBN * D; e += kThreads3) {     /* V tile, aliasing K */
            const int r = e >> DSH, d = e & DM1;
            uint16_t u = 0;
            if (RB[r] >= 0) {
                const int64_t a = RB[r] + ((int64_t)D + d) * P.kv_sd;
                u = (KVDT == RAD_BF16) ? ((const uint16_t*)P.kv)[a]
                                       : fp8_to_bf16_w(((const uint8_t*)P.kv)[a]);
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

    /* ============== out = (O / l) * vds: one 16-row phase per wave =================
     * v1's halves, re-purposed: phase `half` is wave `half` staging its own 16 rows
     * through the Osp alias (16 x (D+8) f32 = 8704 B at D=128, inside Q+S's 13568),
     * then all 64 threads write global. PREDICATED stores, never `continue`d around
     * the barriers -- the standing rule for this family. */
    float* Osp = reinterpret_cast<float*>(raw + C::offQ);   /* aliases Q+S, both dead */

    for (int half = 0; half < 2; ++half) {
        __syncthreads();
        if (w == half) {
            #pragma unroll
            for (int d0 = 0; d0 < D / 16; ++d0)
                roCW::store_matrix_sync(Osp + d0 * 16, oa[d0], C::kOP, roCW::mem_row_major);
        }
        __syncthreads();
        for (int e = tid; e < 16 * D; e += kThreads3) {
            const int r = e >> DSH, d = e & DM1;
            const int row = half * 16 + r;
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
