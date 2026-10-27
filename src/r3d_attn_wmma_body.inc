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

/* Q11 (R3D_ATT_FS): fragment-side softmax folds over the pinned accumulator lane
 * map (COVERAGE window 17): lane l of a 16x16 accumulator holds row 2k+(l>>4),
 * col l&15 -- so one tile row's 32 score columns live in the 16 lanes of ONE
 * half-wave (even rows lanes 0..15, odd 16..31): col c in a0.x[k], col 16+c in
 * a1.x[k], same lane, same register. The row reduction is a 4-step butterfly
 * whose XORs 1/2/4/8 never cross a 16-lane DPP row, so it rides the exact
 * row_xmask primitive Q6/Q22 shipped (ISA table 94, 0x160|xor). The DPP control
 * word must be a compile-time constant (the Q6 probe law): literal masks, no
 * template. fmaxf matches the shipped sweep's `v > mx` semantics exactly: a NaN
 * operand is skipped, and the running max is never itself NaN. */
__device__ __forceinline__ float fs_rowmax(float p) {
    p = fmaxf(p, __uint_as_float(__builtin_amdgcn_update_dpp(__float_as_uint(p), __float_as_uint(p), 0x161, 0xf, 0xf, 0)));
    p = fmaxf(p, __uint_as_float(__builtin_amdgcn_update_dpp(__float_as_uint(p), __float_as_uint(p), 0x162, 0xf, 0xf, 0)));
    p = fmaxf(p, __uint_as_float(__builtin_amdgcn_update_dpp(__float_as_uint(p), __float_as_uint(p), 0x164, 0xf, 0xf, 0)));
    p = fmaxf(p, __uint_as_float(__builtin_amdgcn_update_dpp(__float_as_uint(p), __float_as_uint(p), 0x168, 0xf, 0xf, 0)));
    return p;
}
__device__ __forceinline__ float fs_rowsum(float s) {
    s += __uint_as_float(__builtin_amdgcn_update_dpp(__float_as_uint(s), __float_as_uint(s), 0x161, 0xf, 0xf, 0));
    s += __uint_as_float(__builtin_amdgcn_update_dpp(__float_as_uint(s), __float_as_uint(s), 0x162, 0xf, 0xf, 0));
    s += __uint_as_float(__builtin_amdgcn_update_dpp(__float_as_uint(s), __float_as_uint(s), 0x164, 0xf, 0xf, 0));
    s += __uint_as_float(__builtin_amdgcn_update_dpp(__float_as_uint(s), __float_as_uint(s), 0x168, 0xf, 0xf, 0));
    return s;
}

/* Q10 (R3D_F8ST=1): branchless e4m3 -> bf16 decode, BIT-FOR-BIT the shipped
 * fp8_to_bf16_w. Both sides of the shipped ternary and the shipped NaN select are
 * pure functions of the byte, so computing both and selecting is the same
 * function -- the select is not an approximation of the branch, it IS the branch's
 * truth table. The shipped form lowered to an exec-mask branch DANCE inside the
 * staging loop (v_cmpx_ne + s_xor/s_and_not1_saveexec + s_cbranch_execz + the
 * re-merge s_or), which per divergent byte costs both paths plus the mask
 * surgery; per the v129 disassembly that dance is the fp8 staging loop's largest
 * single cost after the byte gather itself. */
__device__ __forceinline__ uint16_t fp8_to_bf16_bl(unsigned b) {
    const unsigned e = (b >> 3) & 15u, m = b & 7u;
    const uint32_t nrm = ((e + 120u) << 23) | (m << 20);
    const uint32_t sub = __float_as_uint((float)m * 0.001953125f);
    uint32_t u = e ? nrm : sub;
    if (e == 15u && m == 7u) u = 0x7fc00000u;                  /* E4M3 NaN codes */
    u |= (b & 0x80u) << 24;
    return (uint16_t)(u >> 16);
}

/* LDS carve, byte offsets into one buffer (16-aligned), per head_dim:
 *   Q   64 x (D+8) bf16   query tile, staged once per workgroup (pitch 8-aligned:
 *                         the vector staging path stores uint4 straight into LDS)
 *   S   64 x 40    f32    score tile (pad 8: 32 real columns, row-bank spread)
 *   P   64 x 36    bf16   exp'd probabilities, pass two only
 *   KV  32 x (D+8) bf16   key tile, THEN the value tile (aliased: V stages after the
 *                         tile's S-fragments have left the LDS, one sync apart)
 *   meta m/l f32[64], rowbase i64[32], colbad u8[32]; the DBUF arm keeps a SECOND
 *        rowbase/colbad pair so tile t+1's addressing can run while tile t computes.
 * O-store staging (32 x (D+8) f32, halves of the output) aliases Q+S after pass two's
 * last fragment read. D=128 total: 41776 B static (DBUF +304 B) -- under the 48 KiB
 * static ceiling, no hipFuncSetAttribute dance; the +304 B does not change blocks/CU
 * at either head_dim (that is the m64 B-buffer lesson honored: no LDS double buffer).
 *
 * D=256 (issue #11 milestone 2) does NOT fit that static law: the shipped pads give
 * 66640 B against the 65536 ceiling (milestone-1 sizing). The 256 specialization below
 * fits by shaving the two pads that are bank-spread only -- S to kBN+4, P to kBN+2 --
 * while keeping the +8 on Q/KV/O, which is load-bearing addressing (uint4 row stores
 * need 16 B-aligned row starts; D+4=520 B rows break them). Fitted total 65360 B at
 * PAD=0 -- still one block/CU, the same occupancy D=128 already runs at, but over the
 * 48 KiB STATIC ceiling, so the kernel switches to extern __shared__ for D=256 and the
 * launcher sizes it (see r3d_attn256_smem: the mirrored formula the launcher uses,
 * pinned to C::total by the static_assert in the kernel). PAD is an addressing-only
 * A/B lever for the bank pads (0: S kBN+4; 1: S kBN+1 for row-bank spread); every PAD
 * is bit-identical in output, the bench picks the speed. */
template <int D, int PAD = 0> struct Carve {
    static constexpr int kQP = D + 8, kVP = D + 8, kSP = kBN + 8, kPP = kBN + 4;
    static constexpr int kOP = D + 8;
    static constexpr int offQ  = 0;
    static constexpr int offS  = (offQ + kBM * kQP * 2 + 15) & ~15;
    static constexpr int offP  = (offS + kBM * kSP * 4 + 15) & ~15;
    static constexpr int offKV = (offP + kBM * kPP * 2 + 15) & ~15;
    static constexpr int offM  = (offKV + kBN * kVP * 2 + 15) & ~15;   /* + m[64] l[64] */
    static constexpr int offRB = offM + 2 * kBM * 4;
    static constexpr int offCB = offRB + kBN * 8;
    static constexpr int offRB2 = offCB + kBN;         /* DBUF addressing pair (8-aligned) */
    static constexpr int offCB2 = offRB2 + kBN * 8;
    static constexpr int total = offCB2 + kBN + 16;
};

template <int PAD> struct Carve<256, PAD> {
    static constexpr int kQP = 256 + 8, kVP = 256 + 8;   /* +8 row pad: uint4 staging */
    static constexpr int kSP = PAD ? kBN + 1 : kBN + 4;  /* fitted: 36 (PAD1: 33) f32 */
    static constexpr int kPP = kBN + 2;                  /* fitted: 34 bf16           */
    static constexpr int kOP = 256 + 8;
    static constexpr int offQ  = 0;
    static constexpr int offS  = (offQ + kBM * kQP * 2 + 15) & ~15;
    static constexpr int offP  = (offS + kBM * kSP * 4 + 15) & ~15;
    static constexpr int offKV = (offP + kBM * kPP * 2 + 15) & ~15;
    static constexpr int offM  = (offKV + kBN * kVP * 2 + 15) & ~15;
    static constexpr int offRB = offM + 2 * kBM * 4;
    static constexpr int offCB = offRB + kBN * 8;
    static constexpr int offRB2 = offCB + kBN;
    static constexpr int offCB2 = offRB2 + kBN * 8;
    static constexpr int total = offCB2 + kBN + 16;      /* PAD0: 65360, PAD1: 64592 */
};

/* log2(D): every strided walk in the body and the helpers splits an element index as
 * (e >> DSH, e & (D-1)). 6/7 for the shipped head_dims, 8 for 256 -- the same value
 * for a given D everywhere, or the staging slots disagree with the readers. */
constexpr int r3d_dlog2(int d) { return d == 64 ? 6 : d == 128 ? 7 : d == 256 ? 8 : 0; }

/* Gather the 32 key rows RB[] (bad rows zero-filled) into LDS at pitch kVP. The
 * vector path walks EIGHT bf16 per iteration -- the same bits, 8x fewer scalar
 * load/store instructions per tile. It is the throughput fix the v2/v3 pair
 * localized: deep-ctx cases sustain ~90 GB/s on a 500+ GB/s card because the
 * elementwise loop, the barriers, and the bytes are all fine; only the
 * 2-byte-at-a-time walk is slow. `vec` arrives as the dtype/stride/base gate
 * (contiguous rows, 16B-aligned bases); the live row bases RB[] are re-checked
 * for 8-element alignment here (serve KV slots are multiples of 2*head_dim, so
 * yes on every shape the engine serves; the scalar loop keeps any odd shape
 * bit-identical too). fp8 stays scalar: fp8_to_bf16_w is per-byte and the
 * unpack costs more than it saves. koff selects K (0) or V (D) within the slot.
 * Must be called after the addressing barrier (it reads RB[]). */
template <int D, int KVDT, int PAD = 0, int F8ST = 0>
__device__ __forceinline__ void stage_kv_tile(bf16* KVp, const int64_t* RB,
                                              const R3dAttnParams& P, int64_t koff,
                                              bool vec, const uint16_t* F8Lut = nullptr) {
    constexpr int DSH = r3d_dlog2(D);
    typedef Carve<D, PAD> C;
    if (vec) {
        for (int r = 0; r < kBN; ++r)
            if (RB[r] >= 0 && (RB[r] & 7)) { vec = false; break; }
    }
    if (vec) {
        for (int e = threadIdx.x; e < kBN * D / 8; e += kThreads) {
            const int r = e >> (DSH - 3), d = (e & ((1 << (DSH - 3)) - 1)) << 3;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (RB[r] >= 0)
                v = *reinterpret_cast<const uint4*>((const uint16_t*)P.kv + RB[r] + koff + d);
            *reinterpret_cast<uint4*>(KVp + r * C::kVP + d) = v;
        }
    } else if constexpr (KVDT == RAD_F8E4M3 && F8ST >= 3) {
        /* Q10 R3D_F8ST=3: LUT + 8-byte global gather + 16B LDS stores. The bf16
         * bits are the LUT's, built from fp8_to_bf16_w over all 256 codes --
         * identical to the verbatim walk by construction. Gates mirror kvvec:
         * element-contiguous rows (kv_sd == 1) and 8-BYTE-aligned row bases --
         * in fp8 elements are bytes, so RB[] & 7. Row starts land on multiples
         * of 16B because the KV pitch is (D+8) elements; the Q1 pad law holds:
         * no repadding anywhere. A row that fails the gate reruns the whole
         * tile on the scalar LUT walk (block-uniform flag, one barrier
         * regime as before). */
        bool wvec = (P.kv_sd == 1);
        if (wvec)
            for (int r = 0; r < kBN; ++r)
                if (RB[r] >= 0 && (RB[r] & 7)) { wvec = false; break; }
        if (wvec) {
            for (int e = threadIdx.x; e < kBN * D / 8; e += kThreads) {
                const int r = e >> (DSH - 3), d = (e & ((1 << (DSH - 3)) - 1)) << 3;
                uint4 v = make_uint4(0, 0, 0, 0);
                if (RB[r] >= 0) {
                    const uint2 w = *reinterpret_cast<const uint2*>(
                        (const uint8_t*)P.kv + RB[r] + koff + d);
                    v.x = F8Lut[(w.x      ) & 0xff] | (F8Lut[(w.x >>  8) & 0xff] << 16);
                    v.y = F8Lut[(w.x >> 16) & 0xff] | (F8Lut[ w.x >> 24        ] << 16);
                    v.z = F8Lut[(w.y      ) & 0xff] | (F8Lut[(w.y >>  8) & 0xff] << 16);
                    v.w = F8Lut[(w.y >> 16) & 0xff] | (F8Lut[ w.y >> 24        ] << 16);
                }
                *reinterpret_cast<uint4*>(KVp + r * C::kVP + d) = v;
            }
        } else {
            for (int e = threadIdx.x; e < kBN * D; e += kThreads) {
                const int r = e >> DSH, d = e & (D - 1);
                uint16_t u = 0;
                if (RB[r] >= 0)
                    u = F8Lut[((const uint8_t*)P.kv)[RB[r] + (koff + (int64_t)d) * P.kv_sd]];
                bf16_store(KVp + r * C::kVP + d, u);
            }
        }
    } else if constexpr (KVDT == RAD_F8E4M3 && F8ST >= 1) {
        /* Q10 R3D_F8ST=1/2: the verbatim walk's slot map and addressing, decode
         * swapped -- branchless VALU (1) or the LDS LUT (2), both bit-for-bit
         * fp8_to_bf16_w. */
        for (int e = threadIdx.x; e < kBN * D; e += kThreads) {
            const int r = e >> DSH, d = e & (D - 1);
            uint16_t u = 0;
            if (RB[r] >= 0) {
                const int64_t a = RB[r] + (koff + (int64_t)d) * P.kv_sd;
                const uint8_t c = ((const uint8_t*)P.kv)[a];
                u = (F8ST >= 2) ? F8Lut[c] : fp8_to_bf16_bl(c);
            }
            bf16_store(KVp + r * C::kVP + d, u);
        }
    } else {
        for (int e = threadIdx.x; e < kBN * D; e += kThreads) {
            const int r = e >> DSH, d = e & (D - 1);
            uint16_t u = 0;
            if (RB[r] >= 0) {
                const int64_t a = RB[r] + (koff + (int64_t)d) * P.kv_sd;
                u = (KVDT == RAD_BF16) ? ((const uint16_t*)P.kv)[a]
                                       : fp8_to_bf16_w(((const uint8_t*)P.kv)[a]);
            }
            bf16_store(KVp + r * C::kVP + d, u);
        }
    }
}

/* Register-prefetch staging for the DBUF arm (see the kernel's DBUF branch).
 * ld walks the tile's uint4 items GLOBAL -> REGISTERS (the array is the in-flight
 * buffer); st walks REGISTERS -> LDS with the same e->(r,d) mapping stage_kv_tile's
 * vector path uses, so the bytes land in identical slots. The per-item alignment
 * gate is thread-local (RB[r]&7), so no block agreement is needed mid-pipeline;
 * misaligned rows walk scalars into the same registers (bit-identical, never hit
 * by serve layouts -- kv_sd==1 is a launcher gate for this arm). DBUF arm is
 * bf16-only: fp8 unpacks per byte and stays on the scalar arm. */
template <int D, int KVDT>
__device__ __forceinline__ void ld_kv_regs(uint4* dst, const int64_t* RB,
                                           const R3dAttnParams& P, int64_t koff,
                                           bool vec) {
    constexpr int DSH = r3d_dlog2(D);
    constexpr int KNU = kBN * D / 8 / kThreads;
    (void)vec;    /* DBUF launches only with the vec gate true; kept for symmetry */
    #pragma unroll
    for (int i = 0; i < KNU; ++i) {
        const int e = threadIdx.x + i * kThreads;
        const int r = e >> (DSH - 3), d = (e & ((1 << (DSH - 3)) - 1)) << 3;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (RB[r] >= 0) {
            if (((RB[r] + koff + d) & 7) == 0) {
                v = *reinterpret_cast<const uint4*>((const uint16_t*)P.kv + RB[r] + koff + d);
            } else {                                  /* rare: odd slot base */
                uint16_t t8[8];
                #pragma unroll
                for (int u = 0; u < 8; ++u)
                    t8[u] = ((const uint16_t*)P.kv)[RB[r] + koff + d + u];
                v = *reinterpret_cast<const uint4*>(t8);
            }
        }
        dst[i] = v;
    }
}

template <int D, int PAD = 0>
__device__ __forceinline__ void st_kv_regs(const uint4* src, bf16* KVp) {
    constexpr int DSH = r3d_dlog2(D);
    constexpr int KNU = kBN * D / 8 / kThreads;
    typedef Carve<D, PAD> C;
    #pragma unroll
    for (int i = 0; i < KNU; ++i) {
        const int e = threadIdx.x + i * kThreads;
        const int r = e >> (DSH - 3), d = (e & ((1 << (DSH - 3)) - 1)) << 3;
        *reinterpret_cast<uint4*>(KVp + r * C::kVP + d) = src[i];
    }
}

/* FOLD (GQA fold, llama.cpp's tile-kernel fold ported to this body): blockIdx.x is a
 * KV HEAD and the 64 tile rows pack that group's P.gqa q-heads token-major -- row r is
 * token (i0+r)/gqa, head kh*gqa + (i0+r)%gqa. The decode band (qlen <= 8) reads the
 * whole KV history per block, so folding divides that traffic by gqa; each row's
 * arithmetic is verbatim v1's, so outputs are bit-identical per row. FOLD=0 collapses
 * every helper below to the original expressions at compile time -- the prefill
 * instantiation is unchanged by construction (a prefill fold would stream the same
 * bytes: its tiles already carry 64 tokens of ONE head and skip dead KV heads). */
/* Q23 (R3D_HD256_SB): sched_barrier fence for the DBUF in-flight windows, the Q20
 * lever ported to attention. __builtin_amdgcn_sched_barrier(0) forbids memory ops
 * crossing it both ways, capping the live range of the staged in-flight batch. */
__device__ __forceinline__ void r3d_attn_sched_stop() { __builtin_amdgcn_sched_barrier(0); }

/* Q45a (R3D_ATTN256_STAGEBAR): lgkmcnt-scoped staging barrier. gfx1100 S_BARRIER
 * (ISA p.254) "do[es] not wait for any counters to go to zero before issuing" --
 * the full vmcnt(0)+lgkmcnt(0) drain the compiler pins on llvm.amdgcn.s.barrier is
 * fence habit, not a hardware requirement. Where a staging barrier's cross-lane
 * dependency is LDS-only (ds_store is lgkmcnt-counted) AND the guarded window
 * carries no global store (the touched sites' shipped lowering carries NO vscnt;
 * publish sites keep the compiler's full triple -- Q31 store law), the minimal
 * sufficient lowering is the Strata donor form: s_waitcnt lgkmcnt(0) for staging
 * visibility + s_barrier for wave-correct arrival; the in-flight global prefetch
 * stays in flight and lands behind the NEXT staging barrier's own vmcnt gate
 * (present in the shipped lowering). Inline asm = codegen contract (Q42 law):
 * re-prove the ISA site census after every toolchain bump. */
__device__ __forceinline__ void r3d_stagebar_scoped() {
    __asm__ __volatile__("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
}

/* barrier selector: SBZ=0 -> shipped __syncthreads() verbatim (withdrawal arm);
 * SBZ=1 -> scoped form at the Q45a-named staging sites only. Compile-time constant
 * in every instantiation: no runtime branch, no register cost. */
template <int SBZ>
__device__ __forceinline__ void r3d_stage_bar() {
    if (SBZ) r3d_stagebar_scoped(); else __syncthreads();
}

template <int D, int KVDT, int SPLITK, int DBUF = 0, int FOLD = 0, int PAD = 0, int F8ST = 0, int SB = 0, int FS = 0, int SBZ = 0>
__global__ __launch_bounds__(kThreads) void attn_paged_wmma_kernel(R3dAttnParams P) {
    static_assert(FS == 0 || DBUF == 1, "fragment-side softmax rides the DBUF schedule only");
    constexpr int DSH = r3d_dlog2(D), DM1 = D - 1;
    typedef Carve<D, PAD> C;
    static_assert(D == 64 || D == 128 || D == 256, "WMMA tile: head_dim outside {64,128,256}");
    /* Issue #11 milestone 3: the split arm at 256 opened. Everything the SPLITK=1
     * paths touch is D-generic by construction -- the slice cut is pure integer
     * arithmetic on (t1-t0, sp, spl), the partial store reuses the proven 32-row
     * O staging through the Q+S alias (32*(D+8)*4 = 33792 B fits the 42240 B
     * Q+S span at D=256), and the merge/arena hooks recompute the same cut.
     * FOLD rides the split arm only (fold+direct-out is never launched: the
     * prefill launcher keeps FOLD=0 on its 256 tile). */
    static_assert(D != 256 || FOLD == 0 || SPLITK == 1,
                  "256 tile: the fold arm rides the split-KV row only (issue #11 m3)");
    /* The launcher (outside the chip gate) sizes the 256 launch from the mirrored
     * host-side formula; pin the two together so a Carve edit that moves total
     * breaks the build, not the serve. */
    static_assert(D != 256 || C::total == r3d_attn256_smem(PAD), "carve mirror drift");
    /* 256 breaks the 48 KiB static ceiling (fitted total 65360 > 49152), so its LDS
     * is dynamic: extern __shared__, sized by the launcher, with the driver attribute
     * raised once before the first launch (see r3d_attn_paged_wmma.hip). 64/128 keep
     * the historical static array -- the discarded branch below is not instantiated
     * for them, and the 256 branch never allocates a static buffer. */
    unsigned char* raw;
    if constexpr (D == 256) { extern __shared__ unsigned char dyn[]; raw = dyn; }
    else                    { __shared__ unsigned char stat[C::total]; raw = stat; }

    bf16*   Qsp = reinterpret_cast<bf16*>(raw + C::offQ);
    float*  Ssp = reinterpret_cast<float*>(raw + C::offS);
    bf16*   Psp = reinterpret_cast<bf16*>(raw + C::offP);
    bf16*   KVp = reinterpret_cast<bf16*>(raw + C::offKV);
    float*  MSP = reinterpret_cast<float*>(raw + C::offM);
    float*  LSP = MSP + kBM;
    int64_t* RB = reinterpret_cast<int64_t*>(raw + C::offRB);
    unsigned char* CB = raw + C::offCB;
    int64_t* RB2 = reinterpret_cast<int64_t*>(raw + C::offRB2);
    unsigned char* CB2 = raw + C::offCB2;
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

    /* Q10 (R3D_F8ST>=2): per-block e4m3 LUT, fp8 byte -> exact bf16 bits, built
     * from the SHIPPED fp8_to_bf16_w over all 256 codes -- identical decode by
     * construction. Two stores per thread (kThreads == 128); visibility rides
     * the existing addressing barrier that already precedes every stage call
     * (RB[] itself is LDS), so the barrier count is untouched. For every
     * F8ST == 0 instantiation the array is never referenced (constexpr kills the
     * uses below) and the arm is the shipped code path verbatim. */
    constexpr bool kF8Lut = (KVDT == RAD_F8E4M3) && F8ST >= 2;
    __shared__ uint16_t F8Lut[kF8Lut ? 256 : 1];
    if (kF8Lut) {
        F8Lut[tid] = fp8_to_bf16_w(tid);
        F8Lut[tid + kThreads] = fp8_to_bf16_w(tid + kThreads);
    }

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
    const int64_t nrows_q = FOLD ? qlen * P.gqa : qlen;   /* live tile rows, fold units */
    const int64_t i0 = (int64_t)qt * kBM;
    if (i0 >= nrows_q) return;
    const int64_t tok_base = start + i0;
    const int64_t valid    = min<int64_t>(nrows_q - i0, kBM);
    const int64_t ctx      = (int64_t)ldi_w(P.su_dt, P.su, s);
    const int64_t pos_base = ctx - qlen + i0;
    const int64_t pos_f0   = ctx - qlen;              /* position of token 0 (fold rows) */
    const int64_t kh       = FOLD ? (int64_t)h : h / P.gqa;
    /* row -> (token, head) maps; FOLD=0 collapses to tok_base+r / h / pos_base+r */
    auto tokabs    = [&](int64_t r) { return FOLD ? start + (i0 + r) / P.gqa : tok_base + r; };
    auto headabs   = [&](int64_t r) { return FOLD ? kh * P.gqa + (i0 + r) % P.gqa : (int64_t)h; };
    auto rowpos_of = [&](int64_t r) { return FOLD ? pos_f0 + (i0 + r) / P.gqa : pos_base + r; };
    auto paridx    = [&](int64_t r) { return (tokabs(r) * P.q_heads + headabs(r)) * spl + sp; };
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

    /* Gather the 32 key rows for tile t into an addressing slot. One text for the
     * DBUF arm's prologue and loop tail (the non-DBUF arm keeps its historical
     * inline copy: the arithmetic here is that copy, moved verbatim). */
    auto address_tile = [&](int t, int64_t* RBx, unsigned char* CBx) {
        if (tid < kBN) {
            const int64_t j = (int64_t)t * kBN + tid;
            int64_t rb = -1;
            if (j < jend) {
                const int64_t blk = (int64_t)ldi_w(P.bt_dt, P.bt,
                                                   s * P.bt_rs + (j / P.block_size) * P.bt_cs);
                if (blk >= 0)
                    rb = blk * P.kv_sb + kh * P.kv_sh + (j % P.block_size) * P.kv_so;
            }
            RBx[tid] = rb; CBx[tid] = (rb < 0) ? 1 : 0;
        }
    };

    /* Tile range that any row can see (skip the dead head of a window, stop past the
     * last position any row may attend to). */
    const int64_t pos_last = rowpos_of(valid - 1);
    int64_t jmax = jend;
    if (causal && pos_last + 1 < jmax) jmax = pos_last + 1;
    if (!causal && window > 0 && pos_last + window < jmax) jmax = pos_last + window;
    int64_t jmin = 0;
    if (window > 0) jmin = max<int64_t>(0, rowpos_of(0) - window + 1);
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
                const int64_t idx = paridx(row);
                psm[idx] = -INFINITY; psl[idx] = 0.0f;
            }
            return;
        }
    }

    /* ---- stage Q once: rows tok_base..+63 (invalid rows zeroed; they mask everywhere
     * and never store), columns 0..head_dim-1. Vector path when the q walk is
     * contiguous and 8-aligned on both sides (bit-identical copy either way). ---- */
    const int64_t qbase = tok_base * P.q_rs + (int64_t)h * P.q_hs;
    /* per-tile-row Q base; FOLD gathers across the kv group's heads (row-major q walk
     * becomes a gather, but the vec gate is row-local: (q_rs&7)==0 and (q_hs&7)==0
     * keep every fold row 8-aligned on its own). */
    auto qrow = [&](int64_t r) { return FOLD
        ? tokabs(r) * P.q_rs + headabs(r) * P.q_hs : qbase + r * P.q_rs; };
    const bool qvec = (P.q_ds == 1) && ((P.q_rs & 7) == 0) && ((P.q_hs & 7) == 0) &&
                      (FOLD || (qbase & 7) == 0) && ((((uintptr_t)P.q) & 15) == 0);
    if (qvec) {
        for (int e = tid; e < kBM * D / 8; e += kThreads) {
            const int r = e >> (DSH - 3), d = (e & ((1 << (DSH - 3)) - 1)) << 3;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (r < valid)
                v = *reinterpret_cast<const uint4*>((const uint16_t*)P.q
                                                    + qrow(r) + d);
            *reinterpret_cast<uint4*>(Qsp + r * C::kQP + d) = v;
        }
    } else {
        for (int e = tid; e < kBM * D; e += kThreads) {
            const int r = e >> DSH, d = e & DM1;
            bf16_store(Qsp + r * C::kQP + d, (r < valid)
                ? ((const uint16_t*)P.q)[qrow(r) + (int64_t)d * P.q_ds]
                : (uint16_t)0);
        }
    }
    /* Per-kernel gate for the KV staging vector path (per-tile RB alignment is
     * re-checked inside stage_kv_tile). fp8 is excluded by dtype, not alignment. */
    const bool kvvec = (KVDT == RAD_BF16) && (P.kv_sd == 1) &&
                       ((((uintptr_t)P.kv) & 15) == 0);
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

    /* DBUF in-flight staging registers: kx carries the next tile's K rows while the
     * current tile computes; vc carries the current tile's V rows, reloaded to the
     * next tile's V right after its store (one in-flight V per thread block, and K
     * is consumed same-iteration so it needs no swap). */
    constexpr int KNU = kBN * D / 8 / kThreads;
    uint4 kx[KNU], vc[KNU];

    /* The O accumulators live above the arm branch: both schedules fill them the
     * same way, and the shared tail normalizes them into out/partials. */
    roCW::fragment<roCW::accumulator, 16, 16, 16, float> oa[D / 16];
    #pragma unroll
    for (int i = 0; i < D / 16; ++i) roCW::fill_fragment(oa[i], 0.0f);

    if (DBUF) {
        /* ---- DBUF schedule (gemm m64's lesson, applied to paged attention): the
         * next tile's global gather is issued BEFORE the current tile's WMMA and
         * lands in registers, then walks to LDS when the single KV buffer frees.
         * Barriers per tile drop pass one 4->3, pass two 6->4: addressing runs one
         * tile ahead in the double-buffered RB/CB pair, staging runs inside the
         * compute window. Arithmetic is v1's verbatim (same fragments, same order,
         * same LDS slots), so outputs are bit-identical and the arm needs no KLD. */
        int64_t* RBc = RB; int64_t* RBn = RB2;
        unsigned char* CBc = CB; unsigned char* CBn = CB2;

        /* pass one, pipelined */
        address_tile(et0, RBc, CBc);
        address_tile(et0 + 1, RBn, CBn);
        __syncthreads();
        stage_kv_tile<D, KVDT, PAD, F8ST>(KVp, RBc, P, 0, kvvec, F8Lut);
        __syncthreads();
        for (int t = et0; t < et1; ++t) {
            const int64_t jb = (int64_t)t * kBN;
            ld_kv_regs<D, KVDT>(kx, RBn, P, 0, kvvec);       /* K(t+1) in flight */
            if (SB) r3d_attn_sched_stop();                   /* Q23: cap the in-flight window */
            roCW::fill_fragment(a0, 0.0f);
            roCW::fill_fragment(a1, 0.0f);
            for (int k0 = 0; k0 < D; k0 += 16) {
                roCW::load_matrix_sync(fa,  Qsp + (w * 16) * C::kQP + k0, C::kQP, roCW::mem_row_major);
                roCW::load_matrix_sync(fb0, KVp + k0, C::kVP);
                roCW::load_matrix_sync(fb1, KVp + 16 * C::kVP + k0, C::kVP);
                roCW::mma_sync(a0, fa, fb0, a0);
                roCW::mma_sync(a1, fa, fb1, a1);
            }
            if (FS) {
                /* Q11 fragment-side online softmax: the row reductions ride the
                 * accumulator registers -- no Ssp round trip, no LDS read-back
                 * sweep. Row r's 32 columns live in half-wave (r&1): lane
                 * half*16+c holds col c (a0.x[k]) and col 16+c (a1.x[k]) of row
                 * w*16+2k+half. Per row: pair, butterfly over the 16 lanes, and
                 * the designated lanes (c==0) own the (m,l) update -- the sweep's
                 * arithmetic verbatim, only the sum's tree differs.
                 * NOTE: the lane map's l is the lane WITHIN THE WAVE (tid & 31):
                 * hf must be masked to the wave, or waves 1-3 misown rows. */
                __syncthreads();                               /* the mma's KVp reads are done */
                const int l15 = tid & 15, hf = (tid >> 4) & 1;
                float rmax[8], rsum[8];
                #pragma unroll
                for (int k = 0; k < 8; ++k) {
                    const int64_t rp = rowpos_of(w * 16 + 2 * k + hf);
                    const float v0 = (CBc[l15]      || !visible(rp, jb + l15))      ? -INFINITY : a0.x[k] * scale2;
                    const float v1 = (CBc[16 + l15] || !visible(rp, jb + 16 + l15)) ? -INFINITY : a1.x[k] * scale2;
                    rmax[k] = fs_rowmax(fmaxf(v0, v1));
                }
                #pragma unroll
                for (int k = 0; k < 8; ++k) {
                    const int rr = w * 16 + 2 * k + hf;
                    const int64_t rp = rowpos_of(rr);
                    const float mx = fmaxf(MSP[rr], rmax[k]);  /* the sweep's running max, same bits */
                    const float s0 = (CBc[l15]      || !visible(rp, jb + l15))      ? 0.0f : exp2f(a0.x[k] * scale2 - mx);
                    const float s1 = (CBc[16 + l15] || !visible(rp, jb + 16 + l15)) ? 0.0f : exp2f(a1.x[k] * scale2 - mx);
                    rsum[k] = fs_rowsum(s0 + s1);
                }
                if (l15 == 0) {                    /* designated lane per row: two per wave, 8 rows each */
                    #pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        const int rr = w * 16 + 2 * k + hf;
                        if (rr >= valid) continue;
                        float mx = rmax[k];
                        const float old = MSP[rr];
                        if (mx != mx) mx = old;      /* sweep semantics: an all-NaN row keeps the running max */
                        if (old > mx) mx = old;
                        if (mx != -INFINITY) {
                            if (mx > old) {
                                if (old != -INFINITY) LSP[rr] *= exp2f(old - mx);
                                MSP[rr] = mx;
                            }
                            LSP[rr] += rsum[k];
                        }
                    }
                }
                __syncthreads();                               /* CBc reads and the m/l updates complete */
            } else {
            roCW::store_matrix_sync(Ssp + (w * 16) * C::kSP,      a0, C::kSP, roCW::mem_row_major);
            roCW::store_matrix_sync(Ssp + (w * 16) * C::kSP + 16, a1, C::kSP, roCW::mem_row_major);
            r3d_stage_bar<SBZ>();  /* K(t)'s fragments have left. Q45a NAMED SITE: the 0x4c00-class staging barrier -- the drain here swallows the 16 b128 K/V register prefetch; cross-lane dep at this barrier is the 8 preceding LDS staging stores only, and the shipped lowering carries no vscnt. The only such site in the current binary; the vscnt-bearing 0xc7b4/0xd234-class barriers stay compiler-emitted verbatim (Q31 store law). */
            if (tid < kBM && tid < valid) {
                const int64_t rowpos = rowpos_of(tid);
                const float* srow = Ssp + tid * C::kSP;
                float mx = MSP[tid];
                for (int jj = 0; jj < kBN; ++jj) {
                    if (CBc[jj] || !visible(rowpos, jb + jj)) continue;
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
                        if (CBc[jj] || !visible(rowpos, jb + jj)) continue;
                        sum += exp2f(srow[jj] * scale2 - mx);
                    }
                    LSP[tid] += sum;
                }
            }
            __syncthreads();                                   /* CBc reads complete [Q45a-site2:OFF] */
            }
            st_kv_regs<D, PAD>(kx, KVp);                            /* K(t+1) -> LDS */
            address_tile(t + 2, RBc, CBc);                     /* refill the freed slot */
            { int64_t* tmp = RBc; RBc = RBn; RBn = tmp; }
            { unsigned char* tmp = CBc; CBc = CBn; CBn = tmp; }
            __syncthreads();
        }

        /* pass two, pipelined */   /* [Q45a-site3 pass1 close:OFF] */
        address_tile(et0, RBc, CBc);
        address_tile(et0 + 1, RBn, CBn);
        __syncthreads();
        stage_kv_tile<D, KVDT, PAD, F8ST>(KVp, RBc, P, 0, kvvec, F8Lut);   /* K(t0) */
        ld_kv_regs<D, KVDT>(vc, RBc, P, D, kvvec);             /* V(t0) in flight */
        __syncthreads();
        for (int t = et0; t < et1; ++t) {
            const int64_t jb = (int64_t)t * kBN;
            ld_kv_regs<D, KVDT>(kx, RBn, P, 0, kvvec);         /* K(t+1) in flight */
            if (SB) r3d_attn_sched_stop();                     /* Q23: cap the in-flight window */
            roCW::fill_fragment(a0, 0.0f);
            roCW::fill_fragment(a1, 0.0f);
            for (int k0 = 0; k0 < D; k0 += 16) {
                roCW::load_matrix_sync(fa,  Qsp + (w * 16) * C::kQP + k0, C::kQP, roCW::mem_row_major);
                roCW::load_matrix_sync(fb0, KVp + k0, C::kVP);
                roCW::load_matrix_sync(fb1, KVp + 16 * C::kVP + k0, C::kVP);
                roCW::mma_sync(a0, fa, fb0, a0);
                roCW::mma_sync(a1, fa, fb1, a1);
            }
            if (FS) {
                /* Q11 pass two: P = exp2(S*scale2 - m) computed on the fragments and
                 * stored straight to Psp -- Ssp never exists. The exp2 operand bits
                 * equal the e-loop's (Ssp is a fragment store round trip, S is never
                 * NaN-perturbed by it), m comes from the same MSP words, so P bits
                 * match the historical arm exactly; only pass one's l tree differs. */
                __syncthreads();                               /* b1': the mma's KVp reads are done; K(t) dead */
                const int l15 = tid & 15, hf = (tid >> 4) & 1;
                #pragma unroll
                for (int k = 0; k < 8; ++k) {
                    const int rr = w * 16 + 2 * k + hf;
                    const int64_t rp = rowpos_of(rr);
                    const float m = MSP[rr];
                    float p0 = 0.0f, p1 = 0.0f;
                    if (rr < valid && m != -INFINITY) {
                        p0 = (CBc[l15]      || !visible(rp, jb + l15))      ? 0.0f : exp2f(a0.x[k] * scale2 - m);
                        p1 = (CBc[16 + l15] || !visible(rp, jb + 16 + l15)) ? 0.0f : exp2f(a1.x[k] * scale2 - m);
                    }
                    bf16_store(Psp + rr * C::kPP + l15,      f32_bf16_w(p0));
                    bf16_store(Psp + rr * C::kPP + 16 + l15, f32_bf16_w(p1));
                }
            } else {
            roCW::store_matrix_sync(Ssp + (w * 16) * C::kSP,      a0, C::kSP, roCW::mem_row_major);
            roCW::store_matrix_sync(Ssp + (w * 16) * C::kSP + 16, a1, C::kSP, roCW::mem_row_major);
            __syncthreads();                                   /* b1: K(t)'s fragments left; S visible [Q45a-site4:OFF] */
            for (int e = tid; e < kBM * kBN; e += kThreads) {  /* S -> P, all four waves */
                const int r = e >> 5, jj = e & 31;
                float pv = 0.0f;
                if (r < valid && !CBc[jj] && visible(rowpos_of(r), jb + jj)) {
                    const float m = MSP[r];
                    if (m != -INFINITY) pv = exp2f(Ssp[r * C::kSP + jj] * scale2 - m);
                }
                bf16_store(Psp + r * C::kPP + jj, f32_bf16_w(pv));
            }
            }
            st_kv_regs<D, PAD>(vc, KVp);                            /* V(t) -> LDS (K(t) dead per b1) */
            ld_kv_regs<D, KVDT>(vc, RBn, P, D, kvvec);         /* V(t+1) in flight */
            if (SB) r3d_attn_sched_stop();                     /* Q23: cap the in-flight window */
            __syncthreads();                                   /* b2: Psp + V(t) visible [Q45a-site5:OFF] */
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
            __syncthreads();                                   /* b3: V(t)'s fragments left [Q45a-site6:OFF] */
            st_kv_regs<D, PAD>(kx, KVp);                            /* K(t+1) -> LDS */
            address_tile(t + 2, RBc, CBc);
            { int64_t* tmp = RBc; RBc = RBn; RBn = tmp; }
            { unsigned char* tmp = CBc; CBc = CBn; CBn = tmp; }
            __syncthreads();                                   /* b4: K(t+1)+addressing visible [Q45a-site7:OFF] */
        }
    } else {
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

#ifdef R3D_DBG
        /* the debug wire keeps its own instrumented scalar loop: the R3D_DBG build
         * is forensics, never a perf reference (see the printf+LDS 401 trap). */
        for (int e = tid; e < kBN * D; e += kThreads) {
            const int r = e >> DSH, d = e & DM1;
            uint16_t u = 0;
            if (RB[r] >= 0) {
                const int64_t a = RB[r] + (int64_t)d * P.kv_sd;
                u = (KVDT == RAD_BF16) ? ((const uint16_t*)P.kv)[a]
                                       : fp8_to_bf16_w(((const uint8_t*)P.kv)[a]);
                if (dbg && t == t0 && e == 0) {
                    dbg[0] = (double)RB[r];
                    dbg[1] = (double)(KVDT == RAD_BF16 ? ((const uint16_t*)P.kv)[a]
                                                      : (int)((const uint8_t*)P.kv)[a]);
                    dbg[2] = (double)u;
                }
            }
            bf16_store(KVp + r * C::kVP + d, u);
        }
#else
        stage_kv_tile<D, KVDT, PAD, F8ST>(KVp, RB, P, 0, kvvec, F8Lut);
#endif
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
            const int64_t rowpos = rowpos_of(tid);
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
    /* oa was filled above the arm branch; do NOT redeclare it here (a shadow would
     * leave the outer accumulators unfilled for the tail). */
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
        stage_kv_tile<D, KVDT, PAD, F8ST>(KVp, RB, P, 0, kvvec, F8Lut); /* K tile, as pass one */
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
            if (r < valid && !CB[jj] && visible(rowpos_of(r), jb + jj)) {
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

#ifdef R3D_DBG
        for (int e = tid; e < kBN * D; e += kThreads) {     /* V tile, aliasing K */
            const int r = e >> DSH, d = e & DM1;
            uint16_t u = 0;
            if (RB[r] >= 0) {
                const int64_t a = RB[r] + ((int64_t)D + d) * P.kv_sd;
                u = (KVDT == RAD_BF16) ? ((const uint16_t*)P.kv)[a]
                                       : fp8_to_bf16_w(((const uint8_t*)P.kv)[a]);
                if (dbg && t == t0 && e == 0) {
                    dbg[11] = (double)a;
                    dbg[12] = (double)(KVDT == RAD_BF16 ? ((const uint16_t*)P.kv)[a]
                                                        : (int)((const uint8_t*)P.kv)[a]);
                    dbg[13] = (double)u;
                }
            }
            bf16_store(KVp + r * C::kVP + d, u);
        }
#else
        stage_kv_tile<D, KVDT, PAD, F8ST>(KVp, RB, P, D, kvvec, F8Lut); /* V tile, aliasing K */
#endif
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

    }  /* end non-DBUF arm */

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
                const int64_t idx = paridx(row);
                pso[idx * D + d] = Osp[r * C::kOP + d];
            }
            if (tid < 32) {
                const int row = half * 32 + tid;
                if (row < valid) {
                    const int64_t idx = paridx(row);
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
            const int64_t ob = tokabs(row) * P.o_rs + headabs(row) * P.o_hs
                             + (int64_t)d * P.o_ds;
            if (P.out_f32) ((float*)P.out)[ob] = o;
            else           ((uint16_t*)P.out)[ob] = f32_bf16_w(o);
        }
    }
}
