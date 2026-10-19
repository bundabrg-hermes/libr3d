/* Probe: dump gemm_nt / logits_gemm fixture cases -- params and operand shapes,
 * to see what numel(Y)/N is per case (the M the ref derives) vs the recorded M param.
 * (Earlier revision: dumped weight-plane encodings for act_mul/kv_store/cast; that
 * use is served by git history.) */
#include "kfixture.h"
#include <cstdio>
#include <cstring>

int main(int argc, char** argv) {
    using namespace rad;
    rad::Kfixture fx;
    if (fx.load(argv[1]) != 0) { printf("load failed\n"); return 1; }
    const char* want_n = argc > 2 ? argv[2] : nullptr;
    for (const KfCase& c : fx.cases) {
        if (c.op != "gemm_nt" && c.op != "logits_gemm") continue;
        long long N = -1, K = -1, M = -1;
        for (const KfParam& p : c.params) {
            if (p.key == "N" || p.key == "n_vocab") N = p.i;
            if (p.key == "K" || p.key == "n_embd")  K = p.i;
            if (p.key == "M")                        M = p.i;
        }
        char nbuf[24];
        if (want_n) { snprintf(nbuf, sizeof nbuf, "%lld", N); if (strcmp(nbuf, want_n)) continue; }
        printf("== %s M=%lld N=%lld K=%lld kf.m=%lld band=%s\n", c.op.c_str(), M, N, K,
               (long long)c.m, c.band.c_str());
        for (size_t k = 0; k < c.opd.size(); ++k) {
            const KfOperand& d = c.opd[k];
            printf("   opd[%zu] role=%d dtype=%u shape=[", k, d.role, d.dtype);
            long long numel = 1;
            for (size_t i = 0; i < d.shape.size(); ++i) { printf("%lld,", (long long)d.shape[i]); numel *= d.shape[i]; }
            printf("] numel=%lld%s\n", numel, d.has_enc ? " ENCODED" : "");
        }
    }
    return 0;
}
