/* r3d_dump.hip -- implementation of the env-gated launch capture. Host code only;
 * it lives in a .hip TU so hipMemcpy needs no extra link plumbing. See r3d_dump.h. */
#include "r3d_dump.h"

#include <hip/hip_runtime.h>

#include <sys/stat.h>
#include <sys/types.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace {

struct Cfg {
    bool on = false;
    char dir[512] = "";
    char ops[512] = "";
    int per_row = 2;
    size_t cap = 32u << 20;
};

Cfg& cfg() {
    static Cfg c = [] {
        Cfg x;
        const char* d = std::getenv("R3D_DUMP");
        if (!d || !*d) return x;
        std::snprintf(x.dir, sizeof x.dir, "%s", d);
        const char* f = std::getenv("R3D_DUMP_OPS");
        std::snprintf(x.ops, sizeof x.ops, "%s", f && *f ? f :
                      "gdn_conv_prep,gdn_conv_recurrent,gdn_recurrent,gdn_kkt,gdn_chunk,logits_gemm");
        if (const char* n = std::getenv("R3D_DUMP_MAX")) { int v = std::atoi(n); if (v >= 0) x.per_row = v; }
        if (const char* n = std::getenv("R3D_DUMP_CAP")) { long v = std::atol(n); if (v > 0) x.cap = (size_t)v; }
        x.on = true;
        return x;
    }();
    return c;
}

std::mutex g_mu;
char g_used_names[48][72];
int  g_used[48];
int  g_used_n = 0;

size_t esz(uint32_t dt) {
    switch (dt) {
        case RAD_F32: case RAD_I32: case RAD_U32: return 4;
        case RAD_I64:                              return 8;
        case RAD_F16: case RAD_BF16: case RAD_I16:               return 2;
        case RAD_F8E4M3: case RAD_F8E5M2: case RAD_I8: case RAD_U8: case RAD_BOOL: return 1;
        default: return 4;  /* sub-byte codes never reach these rows; replay rejects odd dtypes anyway */
    }
}

void mkdirs(const char* path) {
    char buf[768]; std::snprintf(buf, sizeof buf, "%s", path);
    for (char* p = buf + 1; *p; ++p)
        if (*p == '/') { *p = '\0'; mkdir(buf, 0777); *p = '/'; }
    mkdir(buf, 0777);
}

/* Engine operands live in device arenas whose VA classification this runtime
 * misroutes for host-side hipMemcpy (the copy takes a host-memcpy shortcut and faults).
 * A device-side bounce is immune: the copy kernel reads the span exactly like the
 * row's own kernel does, into a plain hipMalloc staging buffer the host can read. */
__global__ void r3d_bounce_copy(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
                                size_t n) {
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        dst[i] = src[i];
}

/* Poison: fill an output operand's span with 0xDEADBEEF after the pre-state is saved.
 * Whatever the launch window actually rewrote is then visible byte-exactly in post.bin:
 * survivors mean NOBODY wrote that element between begin() and end() -- which distinguishes
 * "our kernel ran" from "an engine-side fused pass wrote it earlier / our dispatch never
 * executed before the capture synced" without trusting either side's numbers. */
__global__ void r3d_poison_fill(uint32_t* __restrict__ dst, size_t n32) {
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n32; i += stride)
        dst[i] = 0xdeadbeefu;
}

struct Staging {
    uint8_t* buf = nullptr;
    size_t cap = 0;
    bool ensure(size_t bytes) {
        if (bytes <= cap) return buf != nullptr;
        if (buf) hipFree(buf);
        cap = 0; buf = nullptr;
        if (hipMalloc(&buf, bytes) != hipSuccess) return false;
        cap = bytes; return true;
    }
};
Staging& staging() { static Staging s; return s; }

/* The engine records a pass's dispatch stream and replays it straight to the card,
 * bypassing our entries entirely ("draft pass issued differently from its recording").
 * So a capture that ADDS A DISPATCH is fatal the moment it stops adding it: the replay
 * no longer matches. Bounce kernels are therefore a last resort, never the first path.
 * First paths cost no dispatch: an explicit DeviceToHost (SDMA, classified by us --
 * the faulting shortcut was hipMemcpyDefault's own classification), or a host memcpy
 * for operands the runtime says are host/managed. The pointer attributes are queried
 * and logged, so when the runtime lies about an operand the log says who and where. */
bool device_to_host(void* host, const void* dev, size_t n, hipStream_t s, const char* who) {
    if (!n) return true;
    hipPointerAttribute_t at{};
    hipError_t qa = hipPointerGetAttributes(&at, dev);
    std::fprintf(stderr, "R3DDBG copy %s n=%zu q=%d type=%d dev=%d\n",
                 who, n, (int)qa, qa == hipSuccess ? (int)at.type : -1, at.device);
    std::fflush(stderr);
    if (qa == hipSuccess && (at.type == hipMemoryTypeHost || at.type == hipMemoryTypeManaged)) {
        std::memcpy(host, dev, n);   /* CPU-addressable per the runtime; no dispatch */
        return true;
    }
    /* Device (or unclassifiable): there is NO dispatch-free path. An explicit
     * DeviceToHost on the engine's arena pointers segfaults even when the pointer
     * attributes say DEVICE (proven on serve27dmp4: q=0 type=2 -> SIGSEGV). The
     * bounce is the only copy that works -- which is why capture is confined to
     * prefill rows (R3D_DUMP_OPS): prefill dispatches are never recorded/replayed,
     * so bounces there cannot poison a pass recording. NEVER widen this to rows
     * that run in the 1-token decode pass. */
    std::fprintf(stderr, "R3DDBG copy %s: bouncing (prefill-only rows)\n", who);
    if (!staging().ensure(n)) return false;
    r3d_bounce_copy<<<512, 256, 0, s>>>((const uint8_t*)dev, staging().buf, n);
    if (hipGetLastError() != hipSuccess) return false;
    if (hipStreamSynchronize(s) != hipSuccess) return false;
    return hipMemcpy(host, staging().buf, n, hipMemcpyDeviceToHost) == hipSuccess;
}

struct Capt {
    const RadArgs* a = nullptr;
    char row[72] = "";
    int seq = 0, n = 0;
    const void* impl = nullptr;
    std::vector<std::vector<uint8_t>> pre, post;
    std::vector<size_t> bytes;
    std::vector<int> trunc, nullp;
    std::vector<int> poison;   /* operand indices filled with DEADBEEF after pre was saved */
};

}  // namespace

void r3d_dump_begin(R3dDumpSlot* slot, const char* row, const void* impl, const RadArgs* a,
                    RadStream s) {
    slot->priv = nullptr;
    Cfg& c = cfg();
    if (!c.on) return;
    {   /* R3D_DUMP_OPS is a comma list of substrings; the row matches if any is in it */
        bool want = false;
        for (const char* p = c.ops; *p && !want; ) {
            const char* e = std::strchr(p, ',');
            size_t len = e ? (size_t)(e - p) : std::strlen(p);
            if (len && len < 72) {
                char tok[72]; std::memcpy(tok, p, len); tok[len] = '\0';
                if (std::strstr(row, tok)) want = true;
            }
            if (!e) break;
            p = e + 1;
        }
        if (!want) return;
    }
    std::lock_guard<std::mutex> lk(g_mu);
    int* used = nullptr;
    for (int i = 0; i < g_used_n; ++i) if (!std::strcmp(g_used_names[i], row)) { used = &g_used[i]; break; }
    if (!used) {
        if (g_used_n < 48) {
            std::snprintf(g_used_names[g_used_n], 72, "%s", row);
            g_used[g_used_n] = 0; used = &g_used[g_used_n]; ++g_used_n;
        }
    }
    if (!used || *used >= c.per_row) return;
    const int seq = *used;

    if (hipStreamSynchronize((hipStream_t)(uintptr_t)s) != hipSuccess) return;

    Capt* cp = new Capt();
    cp->a = a; cp->seq = seq; cp->impl = impl;
    std::snprintf(cp->row, sizeof cp->row, "%s", row);
    cp->n = a->n_t;
    cp->pre.resize(a->n_t); cp->post.resize(a->n_t);
    cp->bytes.assign(a->n_t, 0); cp->trunc.assign(a->n_t, 0); cp->nullp.assign(a->n_t, 0);
    for (int i = 0; i < a->n_t; ++i) {
        const RadTensor& t = a->t[i];
        if (!t.data) { cp->nullp[i] = 1; continue; }
        size_t els = 1;
        for (uint32_t r = 0; r < t.rank; ++r) els *= (size_t)t.shape[r];
        /* Copy the SPAN the strided kernel actually touches, not the product: a view's
         * product-sized contiguous read can cross the mapping and fault. */
        size_t span = 1;
        for (uint32_t r = 0; r < t.rank; ++r)
            if (t.shape[r] > 0) span = (size_t)((long long)span + (t.shape[r] - 1) * t.stride[r]);
        span = (size_t)((long long)span - 1) + 1;   /* last element inclusive */
        size_t nb = span * esz(t.dtype);
        if (nb > c.cap) { cp->trunc[i] = 1; nb = c.cap; }
        cp->bytes[i] = nb;
        cp->pre[i].resize(nb);
        /* Name the operand before touching it: a faulting copy's last stderr line is
         * then the culprit (the engine's arena does not guarantee shape-sized mappings). */
        std::fprintf(stderr, "R3DDBG %s op%d dt=%u rank=%u els=%zu span=%zu bytes=%zu ptr=%p sh=",
                     row, i, t.dtype, t.rank, els, span, nb, t.data);
        for (uint32_t r = 0; r < t.rank; ++r) std::fprintf(stderr, "%lld,", (long long)t.shape[r]);
        std::fprintf(stderr, " st=");
        for (uint32_t r = 0; r < t.rank; ++r) std::fprintf(stderr, "%lld,", (long long)t.stride[r]);
        std::fprintf(stderr, "\n");
        std::fflush(stderr);
        if (!device_to_host(cp->pre[i].data(), t.data, nb, (hipStream_t)(uintptr_t)s,
                            (std::string(row) + " pre op" + std::to_string(i)).c_str())) {
            delete cp; return;
        }
    }

    /* Poison the output operands of the prefill rows (see r3d_poison_fill): only rows whose
     * dispatches are never recorded/replayed, so the fill's own dispatch is harmless. */
    auto poison_list = [&](const char* r) -> std::vector<int> {
        if (std::strstr(r, "gdn_kkt"))       return {4};
        if (std::strstr(r, "chunk_scan"))    return {8, 9};
        if (std::strstr(r, "conv_prep"))     return {11, 12, 13, 14, 15};
        return {};
    };
    /* Poison rounds are now opt-in: a CLEAN capture round is the baseline every poison claim must be re-confirmed against (the fill corrupts its own round's replay diffs). Set R3D_DUMP_POISON=1 to arm it. */
    static const bool poison_on = std::getenv("R3D_DUMP_POISON") != nullptr;
    cp->poison = poison_on ? poison_list(row) : std::vector<int>{};
    for (int pi : cp->poison) {
        if (pi < 0 || pi >= (int)a->n_t) continue;
        const RadTensor& t = a->t[pi];
        if (!t.data || cp->nullp[pi] || cp->trunc[pi]) continue;
        size_t n32 = cp->bytes[pi] / 4;
        if (!n32) continue;
        r3d_poison_fill<<<512, 256, 0, (hipStream_t)(uintptr_t)s>>>((uint32_t*)t.data, n32);
        std::fprintf(stderr, "R3DDBG %s: poisoned op%d n32=%zu\n", row, pi, n32);
    }
    std::fflush(stderr);

    slot->priv = cp;
}

void r3d_dump_end(R3dDumpSlot* slot, int rc) {
    if (!slot->priv) return;
    Capt* cp = (Capt*)slot->priv; slot->priv = nullptr;
    const RadArgs* a = cp->a;

    if (rc == RAD_OK) {
        if (hipDeviceSynchronize() != hipSuccess) { delete cp; return; }
        for (int i = 0; i < cp->n; ++i) {
            if (cp->nullp[i] || !cp->bytes[i]) continue;
            cp->post[i].resize(cp->bytes[i]);
            std::fprintf(stderr, "R3DDBG %s POST op%d bytes=%zu ptr=%p\n",
                         cp->row, i, cp->bytes[i], a->t[i].data);
            std::fflush(stderr);
            if (!device_to_host(cp->post[i].data(), a->t[i].data, cp->bytes[i],
                                (hipStream_t)nullptr,
                                (std::string(cp->row) + " post op" + std::to_string(i)).c_str())) {
                delete cp; return;
            }
        }
    }

    char dir[768];
    std::snprintf(dir, sizeof dir, "%s/%04d_%s", cfg().dir, cp->seq, cp->row);
    mkdirs(dir);
    {
        FILE* f = std::fopen((std::string(dir) + "/manifest.tsv").c_str(), "w");
        if (f) {
            std::fprintf(f, "H\t%s\t%d\t%d\t%d\n", cp->row, rc, (int)a->n_t, (int)a->n_p);
            for (size_t pi : cp->poison) std::fprintf(f, "Z\t%zu\n", pi);
            for (int i = 0; i < a->n_p; ++i) {
                const RadParam& p = a->p[i];
                std::fprintf(f, "P\t%d\t%s\t%lld\t%lld\t%s\t%.17g\n", p.kind, p.key,
                             p.ival, p.ihi, p.sval ? p.sval : "-", p.dval);
            }
            size_t off = 0;
            for (int i = 0; i < cp->n; ++i) {
                const RadTensor& t = a->t[i];
                std::fprintf(f, "T\t%d\t%u\t%u", i, t.dtype, t.rank);
                for (uint32_t r = 0; r < t.rank; ++r) std::fprintf(f, "\t%lld", (long long)t.shape[r]);
                for (uint32_t r = 0; r < t.rank; ++r) std::fprintf(f, "\t%lld", (long long)t.stride[r]);
                std::fprintf(f, "\t%zu\t%d\t%d\t0x%llx\n", cp->bytes[i], cp->trunc[i],
                             cp->nullp[i], (unsigned long long)(uintptr_t)t.data);
                off += cp->bytes[i];
            }
            std::fclose(f);
        }
        for (int pass = 0; pass < 2; ++pass) {
            FILE* g = std::fopen((std::string(dir) + (pass ? "/post.bin" : "/pre.bin")).c_str(), "wb");
            if (!g) continue;
            for (int i = 0; i < cp->n; ++i) {
                const std::vector<uint8_t>& v = pass ? cp->post[i] : cp->pre[i];
                if (!v.empty()) std::fwrite(v.data(), 1, v.size(), g);
            }
            std::fclose(g);
        }
    }
    std::lock_guard<std::mutex> lk(g_mu);
    for (int i = 0; i < g_used_n; ++i) if (!std::strcmp(g_used_names[i], cp->row)) { ++g_used[i]; break; }
    delete cp;
}
