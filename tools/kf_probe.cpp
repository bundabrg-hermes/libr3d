/* Probe: dump the logits_gemm (and gemm_nt) weight operands of the fixture —
 * how many planes the recorded weight encoding spans. */
#include "kfixture.h"
#include <cstdio>
#include <map>

int main(int argc, char** argv) {
    using namespace rad;
    rad::Kfixture fx;
    std::map<std::string,int> seen;
    if (fx.load(argv[1]) != 0) { printf("load failed\n"); return 1; }
    for (const KfCase& c : fx.cases) {
        if (c.op == "logits_gemm") continue;
        if (c.op != "act_mul" && c.op != "kv_store" && c.op != "cast") continue;
        if (seen[c.op]++ >= 2) continue;
        printf("== case op=%s band=%s seed=%llu\n", c.op.c_str(), c.band.c_str(),
               (unsigned long long)c.seed);
        for (const KfParam& p : c.params)
            printf("   param %s = %s%lld%s%g %s\n", p.key.c_str(),
                   p.kind == 0 ? "i" : "", p.i, p.kind == 1 ? "d" : "", p.d,
                   p.kind == 2 ? p.s.c_str() : "");
        for (size_t k = 0; k < c.opd.size(); ++k) {
            const KfOperand& d = c.opd[k];
            printf("   opd[%zu] role=%d dtype=%u absent=%d has_enc=%d n_sel=%zu",
                   k, d.role, d.dtype, (int)d.absent, (int)d.has_enc, d.sel.size());
            if (d.has_enc)
                printf(" scheme=%s planes=%d [", d.enc.scheme, d.enc.n_planes);
            for (int j = 0; j < d.enc.n_planes; ++j)
                printf("%s:%u ", d.enc.plane[j].role, d.enc.plane[j].dtype);
            printf("]\n");
        }
    }
    return 0;
}
