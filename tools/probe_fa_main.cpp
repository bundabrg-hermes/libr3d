// probe_fa_main.cpp -- launch the REAL attn_paged_wmma tile through its own launcher,
// standalone (no plugin .so, no dlopen). If this returns err=401 at launch, the kernel
// itself is the problem, not the module. Buffers are real hipMalloc regions with
// plausible sizes; a device fault at sync is a DIFFERENT answer from a launch 401.
#include "rad_abi.h"
#include "r3d_attn.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>

int main() {
    const int64_t TOKS = 64, QH = 8, HD = 128, KVH = 1, NB = 8, BS = 16;
    void *q, *kv, *out, *bt, *su;
    hipMalloc(&q,  TOKS * QH * HD * 2);
    hipMalloc(&kv, NB * KVH * BS * 2 * HD * 2);          /* fp8 planes: 2*HD bytes/slot */
    hipMalloc(&out, TOKS * QH * HD * 4);
    hipMalloc(&bt, TOKS * 64 * 4);
    hipMalloc(&su, 1 << 20);
    hipMemset(q, 0x3f, TOKS * QH * HD * 2);
    hipMemset(kv, 0x30, NB * KVH * BS * 2 * HD * 2);
    hipMemset(bt, 0, TOKS * 64 * 4);
    hipMemset(su, 0, 1 << 20);

    R3dAttnParams P{};
    P.q = q; P.kv = kv; P.out = out; P.bt = bt; P.su = su; P.cus = nullptr;
    P.kdes = nullptr; P.vdes = nullptr;
    P.qdt = RAD_BF16; P.kvdt = RAD_F8E4M3;
    P.q_rs = QH * HD; P.q_hs = HD; P.q_ds = 1;
    P.o_rs = QH * HD; P.o_hs = HD; P.o_ds = 1;
    P.kv_sb = KVH * BS * 2 * HD; P.kv_sh = BS * 2 * HD; P.kv_so = 2 * HD; P.kv_sd = 1;
    P.bt_rs = 64; P.bt_cs = 1;
    P.head_dim = HD; P.gqa = 8; P.kv_heads = KVH; P.q_heads = QH; P.n_seq = 1;
    P.block_size = BS; P.max_blocks = NB; P.jcap = NB * BS;
    P.rows = 8; P.causal = 1; P.window = 0; P.scale = 0.0884f; P.q_len_uniform = 8;

    int rc = r3d_attn_paged_wmma_launch(P, 0);
    printf("launcher rc=%d\n", rc);
    printf("post-launch lasterr=%d %s\n", (int)hipGetLastError(), hipGetErrorString(hipGetLastError()));
    hipDeviceSynchronize();
    printf("post-sync   lasterr=%d %s\n", (int)hipGetLastError(), hipGetErrorString(hipGetLastError()));
    return 0;
}
