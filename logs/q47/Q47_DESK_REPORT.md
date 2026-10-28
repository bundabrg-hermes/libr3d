# Q47 desk pass — multi-column register-tile GEMV for the verify band (no card)

**Session:** card-free desk window at main d53fcec (v162 line). Donor trees read
locally; census re-measured on cat from the SHIPPED `build_v162` artifact
(md5 351ca1033fc7e5b47419193e382d3d1b, verified before carving).

---

## 0. VERDICT: RESHAPE

The row's headline mechanism — NCOLS activation columns fed from ONE weight-word
load stream — **is already shipped**: the Q44 lane-map walk is, structurally, a
`tmp[NCOLS][1]` register-tile kernel (the `for m < MT` loop IS the NCOLS column
loop; the DPP-kit row-end butterfly IS the donor tail at NW=1, zero LDS). The
measured per-column VGPR margin on the shipped artifact is ~1.4–4 regs
(MT=1→8: 31→41), so the NCOLS axis is budget-trivial and novelty-free.

What the band is still short of is the **Q50-attributed** residual: walk
load→fold dependency latency across *iterations* (chain ~ MT), which more
columns cannot break (columns are independent and already interleaved; the
serialization is weight-word load→use between iterations). The mechanism that
attacks it is the shipped **Q54 bwin cur/nxt window** riding the same walk.

- **BUILD (reshaped arm A):** `gemm_nt_mxfp4_q8x_lm_kernel<MT, WIN=2>` —
  shape **NCOLS ∈ {4, 8} (= the MT ladder shapes), ROWS = 1, WIN = 2**,
  Q7-style parity ping-pong on the 2×uint4 weight pair + scale byte,
  16B-alignment-gated. Passes the paper gate with the largest margin of any
  candidate: realistic ~53–57 VGPR (worst-case full-live model ≤ 123), SGPR
  ~88–90 of the 106 fixed/wave, LDS 0 of 4096 B. FNV class expected (window
  is schedule-only; precedent Q54 WIN=1 byte-identical + Q44 EQUIV 504/504).
- **REJECT (ROWS ≥ 2, the "third band member" reading):** multi-*weight*-row
  per wave amortizes **nothing on the weight stream** (each row carries its
  own weight-word stream — the stream is per output row); it amortizes only
  L2-hot activation traffic and wave slots — the exact axis Q54's grid-
  rebalance falsification says loses on this card. It pays SGPR from the
  measured 84/106 baseline, halves the grid (S3-tail class risk), and the
  RDNA3-pinned donor table (llama.cpp, §3) itself retires to one row per
  block on RDNA3 hardware. Budgets: RW=2 passes VGPR on the realistic model
  only at WIN=1; RW=4 busts SGPR margin and the 4096 B LDS wall (donor
  partials) or the 128-VGPR cap (worst case); RW=8 busts everything and would
  compile into `launch_bounds(32,16)`-forced spill (the Q18b 132-VGPR/46-ssp
  class). Not buildable *and* not aimed at the attributed residual.

**Gate correction on the row's own terms:** the occupancy-16 per-wave VGPR
budget on gfx1100 is **128, not 256** (§2). The "256 hard cap" is the per-thread
allocator ceiling (true at 1–2 waves/SIMD — the sk family is LDS-bound, not
wave-bound, so 207 is legal for it). The row's fear "NCOLS>=4 may not coexist
with WIN=2" does not materialize at ROWS=1: the family-headroom anxiety is an
sk-kernel fact (207/256), not a band-kernel fact (41/128).

---

## 1. Census ground truth — measured THIS window (v162 shipped artifact)

Tool: cat `~/scratch/r3d/q54b_isa.sh` (fatbin carve EM_AMDGPU=224 + AMDHSA
notes via `/opt/rocm/lib/llvm/bin/llvm-readelf`), coverage guards satisfied:
`KERNELS_TOTAL=894`, sk rows reproduce the window-59 record exactly
(WIN=2 max 207, `<1,1>`=152). Band variants via `q47b_isa.sh`/`q47c_isa.sh`
(sed-filters of the same tool, kept on cat scratch).

fp8a8 sk family (the Q54 lineage donor of the window pattern):

| kernel `<SLABS,WIN>` | VGPR | vsp | SGPR | ssp | LDS | priv |
|---|---|---|---|---|---|---|
| sk `<1,1>` / `<1,2>` | 152 / 183 | 0 | 62 / 60 | 0 | 39680 | 0 |
| sk `<2,1>` / `<2,2>` | 162 / 183 | 0 | 64 / 62 | 0 | 39680 | 0 |
| sk `<3,1>` / `<3,2>` | 178 / 207 | 0 | 70 | 0 | 39680 | 128 |
| sk `<4,1>` / `<4,2>` | 178 / 207 | 0 | 70–74 | 0 | 39680 / 512 | 0 |

Window cost measured: **+29..+31 VGPR** for WIN 1→2 at equal SLABS.

TARGET — `gemm_nt_mxfp4_gemv_q8x_lm_kernel<MT>` (Q44 lane-map, the verify-band
arm, default-ON at mt≥4):

| MT | VGPR | vsp | SGPR | ssp | LDS | priv |
|---|---|---|---|---|---|---|
| 1 | 31 | 0 | 28 | 0 | 0 | 0 |
| 2 | 35 | 0 | 36 | 0 | 0 | 0 |
| 4 | 43 | 0 | 49 | 0 | 0 | 0 |
| 8 | **41** | 0 | **84** | 0 | **0** | 0 |

(matches the context-supplied 31/35/43/41 — independently re-measured.)

Withdrawal walk `gemm_nt_mxfp4_gemv_q8x_kernel<MT>` (for contrast): VGPR
27/29/32/36, SGPR 26/41/49/73, LDS 0/0/0/64 at MT 1/2/4/8.

Tool note: a third carved image reports all-zero notes rows (img026/img029);
the two live images agree field-for-field, and sk rows match the window-59
promoted record — parse trustworthy; zero rows are an image-selection artifact
of the carve, not kernel resources.

Measured payload decomposition, lm MT=8 (fit that lands exactly on 41):
`acc[8]` 8 + `wa/wbv` (2×uint4) 8 + X uint4-pairs live ≈16 (the walk keeps
~2 columns' activation pairs in flight) + `p` int32 1 + f32 fold temporaries
≈8 = 41. Marginal per column (MT2→MT4): (43−35)/2 = **+4**. SGPR 84 at MT=8
is the tightest shipped resource of the family (against the fixed 106/wave, §2).

---

## 2. Register/LDS ground rules — arithmetic line by line (ISA, not folklore)

`docs/isa/sections/3-3-2-vgprs.md` (§3.3.2.1, pp. 26–27), verbatim:
> "VGPRs are allocated in blocks of 16 for wave32 … and a shader may have up to
> 256 VGPRs. … Devices which have 1536 VGPRs per SIMD allocate in blocks of 24
> for wave32 …"

`docs/isa/sections/3-3-1-sgprs.md` (§3.3.1.1), verbatim:
> "Every wave is allocated a fixed number of SGPRs: • 106 normal SGPRs • VCC_HI
> and VCC_LO … • 16 Trap-temporary SGPRs …"

Derivation for wave32, launch_bounds(32,16), gfx1100:

```
waves/CU                  = 4 SIMD32 × 4 waves/SIMD                = 16
gfx1100 SIMD class        = the 512-VGPR/SIMD device (block 16)
   — falsifier: the 1536/SIMD class would allow 6 waves/SIMD at
     225 VGPR; the Q11b measured datum is 225 VGPR → 8 waves/CU
     (2 × 4-wave WG), which only the 512/SIMD class permits.
per-SIMD VGPR array       = 512 regs/lane × 32 lanes = 16384 DWORD
per-wave VGPR at 16 waves = 512 regs / 4 waves-per-SIMD            = 128
   (equivalently the task formula: floor(512 × 4 SIMD / 16 waves)   = 128;
    16-granularity: 128 is a whole number of blocks)
256 per thread            = the ARCHITECTURAL max, reachable only at
                            ≤ 2 waves/SIMD (8 waves/CU). The GDN/Q20
                            "256-pinned+spill" rows and sk's 207 are low-
                            occupancy/LDS-bound arms — NOT counterexamples.
                            Repo law already spells the 128 wall out: Q18b
                            "WC=4 busts the launch_bounds 128-VGPR cap
                            (132 VGPR + 46 SGPR spills)" — minBlocks=16 is
                            an allocator contract, so shapes over 128 spill
                            instead of dropping occupancy.
SGPR                      = fixed 106 normal/wave (not an occupancy knob);
                            spill wall at >106 (WC=4 ssp=46 precedent).
                            (Task preamble's "800/CU" must read 800/SIMD:
                            16 waves × 106 = 1696 > 800 would contradict
                            every shipped kernel at 16 waves.)
LDS/CU                    = 64 KiB  (Q1 measured law: 32768 B/WG = exactly
                            half occupancy) → per-block at 16 blocks/CU:
                            65536 / 16                      = 4096 B
```

So the Q47 gate is: **VGPR ≤ 128 (16-blocks), SGPR ≤ 106, LDS ≤ 4096 B/block**,
where "VGPR" counts the *template arm at its MT*, and any claim of ≥16-wave
residency must respect the 16-reg allocation granularity (e.g. 41 → allocates
as 48).

---

## 3. Donor sketches — verified against pinned trees on this host

Both trees present under `~/projects/libr3d/upstream/`; SHAs = this-host
checkouts, load-bearing regions read this window (no citation below is from
memory).

### 3a. Strata `native_mmvq_multi` — VERIFIED (tree @ 4b9e05a2)

`src/kernels/cuda/native_mmvq.cu`:
- Plan comment @ :820–830 (verbatim): "plan v0.3 P3: ncols = 2..8 (speculative
  verify, small batches) … Column j reads activation blocks x + j*(n_in/32)
  … every column is BITWISE equal to a single-column call … `constexpr int
  MAX_NCOLS = 8;`"
- Kernel @ :1170–1268: `template<typename F, int NCOLS, int NW, int ROWS, bool
  TS, bool PAIR>` , `__launch_bounds__(NW*WARP, (ROWS <= 2 ? 4 : 1))`, block
  `dim3(32, NW)`. Register tile `float tmp[NCOLS][ROWS]`. Inner loop @ :1216–
  1228: `for i<ROWS { wv = F::load(block)  // once per (row, block)
  ; for j<NCOLS { tmp[j][i] += F::apply(wv, x + j*x_stride + kby) } }` — the
  one-weight-word-many-columns shape.
- Tail @ :1233: `__shared__ float partial[NW-1>0?NW-1:1][NCOLS][ROWS][WARP]`,
  warps 1..NW-1 store, warp 0 sums; **at NW=1 the array degenerates and is
  unused — a single-wave block needs zero LDS partials.**
- TS tail @ :1240–1261 + `s26_tsum.cuh`: "warp 0's NCOLS x ROWS sums as one
  transposed butterfly … bitwise the same" — pure registers: `xmov16` =
  `__builtin_amdgcn_permlanex16(..., 0x76543210, 0xfedcba98, ...)` (**exactly
  our Q24-pinned selector law**) + DPP `0x160|O` = our Q6/Q22 kit;
  `pow2_ceil` caps at **P=32 values per pass** (NC·RW>32 ⇒ two passes).
- Their dispatch @ :1482–1504: multi at NCOLS 2..8; `ROWS=1` for the exact-
  layout rows, table ROWS for others; gfx12 note @ :1272+ already shows the
  multi-row arm losing to one-row-per-block where measured (R9700 Q8_0 rows1
  carve-out) — the donor itself demotes ROWS where it has card data.
- **HIP-portability:** the kernel compiles under HIPCC (PDL legs are `#if
  !defined(__HIPCC__)`); the tail kit is gfx10+ intrinsics we already ship.

### 3b. llama.cpp two-rows-per-block table — VERIFIED, WITH AN RDNA3-SPECIFIC
CORRECTION (tree @ de5122d87)

`ggml/src/ggml-cuda/mmvq.cu`:
- Register tile @ :737 `float tmp[ncols_dst][rows_per_cuda_block]`; per-warp
  LDS partials @ :786 `__shared__ float tmp_shared[nwarps-1>0?nwarps-1:1]
  [ncols_dst][rows_per_cuda_block][warp_size]` — same donor shape.
- `calc_rows_per_block` @ :595–616: ncols 2..8 → **2 rows** ONLY for the
  GENERIC / GCN / TURING / GB10 tables; every other table (incl. RDNA3_0)
  falls through to **1 row**. `mmvq_two_rows_ncols1` (:582–591) is a
  low-bit-type × GENERIC thing, not RDNA3.
- `calc_nwarps` @ :451–530, RDNA3_0 branch (comment verbatim "RDNA3 (W7900):
  stricter whitelist than RDNA4"): **ncols==1** → 8 warps/row for simple
  vec_dot types, 2 for Q6_K; **ncols 2..8 → `return 1` warp.**
- Consequence for this row: the "three engines converge on NCOLS×2-rows" claim
  is **dented for our card class** — the only RDNA3-pinned tuning in this
  donor is *one row per block*, 8 warps at NCOLS=1 (and that wide-block
  variant is Q51(a)'s already-booked experiment, not Q47).

### 3c. hipfire `_multi` co-launch — NOT ACTIONABLE / donor-unverified

Engine-gated per the row's own Q28-class disposition; not read this window,
no citation claimed.

---

## 4. Interaction with the shipped Q44 lane-map walk

Current arm (`r3d_gemm_mxfp4.hip:1090–1207`, dispatch :1452/:1638/:1654–1659,
grid = one 32-thread block per weight row `r`, `mt = M>=8?8:M>=4?4:M>=2?2:1`,
lm default at mt≥4): lane `t` owns one 32-code group per iteration (`g =
(it<<5)+t`); weight = 2×16B loads (`wa`,`wbv`, 8 VGPR); per column m: X 2×16B,
8 chained in-lane `sudot4` (exact int, no fold), per-lane `SG/AS/sc`
containment, `acc[m] += c`; tail = one 5-op DPP-kit butterfly per m at full
exec; ragged peel fold-free.

- **Does NCOLS replace the m-loop or wrap it? Neither: it IS the m-loop.**
  `acc[MT]` ≡ donor `tmp[NCOLS][1]`; the statically-unrolled `#pragma unroll`
  m-loop with `live` guard is the NCOLS column loop, and the weight pair is
  already loaded once per iteration for all columns. Arm A therefore adds only
  the iteration dimension: `WIN=2` cur/nxt staging of `(wa,wbv,sc)` on Q7
  parity ping-pong locals (the pattern at :998–1016 is the in-file precedent;
  Q54 sk `<SLABS,WIN>` is the shipped one). No new dynamic indices — Q18
  static-index law satisfied by construction (ping-pong on named locals, not
  arrays).
- **ROWS>1 would wrap** the walk in an outer static row loop over independent
  weight streams (`crow + row·rowbytes` per stream). Scale containment
  survives per (row, col); in-walk cross-lane stays 0; the tail grows to
  5×NC×RW ops (linear; or the s26ts transposed form — same op count, P≤32 per
  pass, zero LDS either way).
- **Precision class:** Arm A is schedule-only (window never reorders dot legs)
  → FNV vs shipped expected (Q54 precedent; battery zero-diff gate). A ROWS>1
  arm is FNV-attemptable per (row, col) pair (each row's walk byte-identical
  to that row's ladder walk, tail order preserved) — Q53's parity law is
  WMMA-only and does not apply to this SIMT family. Q44/Q49 NaN-sign law: no
  product is reassociated or commuted in either arm; if a build ever factors
  `sc` across columns it needs the `+v` pin (NaN-sign class).
- **m-row gap:** live = min(M, MT) row guards must be kept per row×column
  (the latent M 5..7 @ mt=4 gap stays mirrored, not widened); a ROWS=2 grid
  halving additionally changes row-ownership, adding a second gap surface —
  one more reason it is last.
- **Grid/tail interaction:** band grid = N ≫ 1536 resident blocks at the
  drafter shapes, so 16-wave residency is continuously saturated and the
  128-VGPR cap is binding; ROWS=2 halves N toward the S3 small-N tail class
  (Q50's exact sensitivity), while Arm A leaves the grid untouched.

---

## 5. NCOLS × ROWS budget table (arm = lm-walk structure at (NC, RW, WIN))

Models: **realistic** = measured 41 (MT=8) plus payload deltas (acc +1/pair,
weight stream +8, window +12..16 from the measured sk +29..31 halved for the
single-stream band window, X shared across rows); **worst** = full-live
count: `acc NC·RW + w 8·RW·WIN + x 8·NC + p 1 + scale/temps 2·NC + RW + ovh 16`.
Allocation granularity 16 applied in the "block" column. Gates: VGPR ≤ 128,
SGPR ≤ 106, LDS ≤ 4096 B/block (register-only tail ⇒ LDS 0 in all rows below;
donor-LDS-partial figures in the LDS sub-table).

| shape (NC, RW, WIN) | realistic VGPR | worst VGPR | alloc block | ≤128? | SGPR est | LDS (reg-tail) | verdict |
|---|---|---|---|---|---|---|---|
| **(8, 1, 1)** = shipped MT=8 | 41 (measured) | 114 | 48 | ✓ | 84 (measured) | 0 | shipped |
| **(8, 1, 2) = ARM A** | ~53–57 | 122 → blk 128 | 64–128 | ✓ both (worst at the cap, not over) | ~88–90 | 0 | **BUILD** |
| (4, 1, 2) | ~47–51 | 78 → blk 80 | 48–80 | ✓ | ~60 | 0 | BUILD (same arm, MT=4) |
| (8, 2, 1) | ~57–65 | 131 → blk 144 | 64–144 | ⚠ worst busts | ~88–92 | 0 | park |
| (8, 2, 2) | ~69–77 | 147 → blk 160 | 80–160 | ✗ worst busts | ~90–96 | 0 | REJECT |
| (4, 2, 2) | ~57–65 | 99 → blk 112 | 64–112 | ✓ | ~90–96 | 0 | passes, pointless (no weight amortization, grid ÷2) |
| (8, 4, 1) | ~89–97 | 165 → blk 176 | 96–176 | ✗ worst busts | ~96–102 ⚠ | 0 | REJECT |
| (8, 4, 2) | ~101–109 | 197 → blk 208 | 112–208 | ✗ | ≥100 ✗ margin gone | 0 | REJECT |
| (8, 8, 2) | ~165+ | 297 → ≥256 spill | ≥176 | ✗ spill-forced (Q18b class) | ≥110 ✗ ssp | — | REJECT |

LDS sub-table (budget 4096 B/block at 16 blocks/CU; register-only s26ts-form
tail = **0 B at every shape**):

| tail form | (8,1) | (8,2) | (8,4) | (8,8) |
|---|---|---|---|---|
| register butterfly (shipped / s26ts port) | 0 | 0 | 0 | 0 |
| 16-lane staged partials, NC·RW·16·4 B | 512 ✓ | 1024 ✓ | 2048 ✓ | 4096 = wall ⚠ (Q1 half-occ precedent — forbidden) |
| donor partials @ NW=1 | n/a (0) | 0 | 0 | 0 |
| donor partials if block widened to NW=4 (4 blk/CU, budget 16384) | 3072 ✓ | 6144 ✓ | 12288 ✓ | 24576 ✗ |

Sanity anchors: realistic model at (8,1,1) reproduces the measured 41 by
construction; the WIN delta 12–16 is the sk-measured +29..31 ÷ ~2 for the
band's single (dword-pair + byte) stream vs sk's multi-stream staging; the
worst column is ~2.8× the measured value at (8,1,1) — use it as the
allocator-adversarial bound, not the expectation.

---

## 6. Verdict detail + bench plan sketch (card window, Arm A only)

**RESHAPE:** build the window, not the wide tile. Passing shape: **NCOLS =
MT ∈ {4, 8} (band contract M ≤ 8 keeps NCOLS ≤ 8), ROWS = 1, WIN = 2**, as a
third template dimension on the existing lm kernel (lever `R3D_Q8X_BWIN`,
0 = verbatim withdrawal arm; the "third band member" is a third dispatch
shape of the lm arm, not a new kernel; m64 and mt∈{1,2} legs untouched).
Reject every ROWS ≥ 2 shape (§5). Flat outcome reading (row-adopted): if the
window is flat, the band residual is not MLP-class → close and feed Q36.

Bench plan (card window, per standing protocol):
1. **Desk pre-build census (card-free):** standalone compile `<MT,WIN>` matrix
   (verbatim flags + control instantiation, Q40 step-0 pattern): control
   WIN=1 must reproduce 41/84/0/0 exactly (fidelity gate before trusting any
   delta); ship-eligibility = VGPR ≤ 64–80 block, 0 vsp/ssp at MT=4/8, WIN=2.
2. **Harness:** bench_w2a8-shape (blob operands, L2 rotation past the 96 MB
   IC; regimes labelled; liveness print per distinct `(mt, walk, win)` tuple).
   M ∈ {4,5,6,7,8}, drafter K shapes; M=1/2 legs as flat controls.
3. **A/B:** same binary, `R3D_Q8X_BWIN=0` vs `2`, ≥3 crossed rounds, disjoint-
   spread adjudication via `tools/ab_parse.py`. Expectation: Q54-class lift on
   cold band legs (post-Q44 band streams 209–359 GB/s, 2–4× under spec;
   Q50 says chain~MT serialization binds — prior +15..35% on cold legs;
   cache-fit regime may sit under the roofline — adjudicate per regime).
4. **Numeric gate:** EQUIV bit-compare WIN=0 vs 2 across the matrix (FNV
   class expected; any drift = window reordering dots = bug, not tolerance);
   row-write-count assert per leg (live = min(M,MT) — the M 5..7 gap must not
   widen); battery 7 packs zero-diff + bwin liveness in the packs that
   dispatch the band.
5. **Census after:** shipped-.so carve (q54b/q47b tools, guards) + protocol
   #5 spill census law.
6. **Promotion:** q139 template; serve leg only if kernel A/B disjoint
   (band is serve-live through the mtp draft path at mt≥4 — the Q44 serve
   datum shows the band's serve sensitivity).

---

## 7. Ledger corrections to carry back

1. **The 256 number in the Q47 framing is the per-thread allocator cap, not
   the occupancy-16 budget; the true gate value is 128** (ISA §3.3.2.1 +
   512-VGPR/SIMD class falsified against 1536 by the Q11b 225-VGPR datum +
   the Q18b repo-law "launch_bounds 128-VGPR cap" spill event).
2. **llama.cpp RDNA3 tables do NOT endorse multi-row multi-col:** 2-rows-per-
   block is GENERIC/GCN/TURING/GB10-only; RDNA3_0 runs NB=1/NW=1 at ncols 2..8
   and 8-warp/1-row at ncols 1 (@ de5122d87). Donor convergence claim: 1.5
   engines, not 3, for our card class.
3. **Strata's own gfx12 measurement already demotes two-rows-per-block where
   measured** (native_mmvq.cu MODEL-UPDATE comment @ :1272+) — the donor's
   ROWS arm is a CUDA-card design point, not a portable law.
4. At NW=1 (our wave32 block family) the donor "per-warp LDS partials" term is
   identically zero — the LDS half of this row's design space does not bind;
   the tail is register-only in every shipped or donor-transposed form.

No repo files modified. Artifacts created this window: this report; cat
`~/scratch/r3d/q47b_isa.sh`, `q47c_isa.sh` + `q54b_isa_q47.txt` (census runs,
read-only on `build_v162`). Card, containers, and both repos untouched.
