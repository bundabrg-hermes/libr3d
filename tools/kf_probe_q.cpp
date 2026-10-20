/* Probe: dump gemm_nt_q fixture cases -- every param (int and string) and per-operand
 * dtype/shape/absence, so an mxfp4a8 row can be written against the recorded geometries
 * without touching a model. Sibling of kf_probe.cpp (gemm_nt / logits_gemm). */
#include "kfixture.h"
#include "rad_types.h"
#include <cstdio>
#include <cstring>

static const char* dt_name(uint32_t d) {
    for (unsigned i = 0; i < sizeof(rad_dt_table) / sizeof(rad_dt_table[0]); ++i)
        if (rad_dt_table[i].dt == d) return rad_dt_table[i].name;
    return "?";
}

int main(int argc, char** argv) {
    using namespace rad;
    rad::Kfixture fx;
    if (fx.load(argv[1]) != 0) { printf("load failed\n"); return 1; }
    const char* want_enc = argc > 2 ? argv[2] : nullptr;   /* dtype-string filter */
    int shown = 0;
    for (const KfCase& c : fx.cases) {
        if (c.op != "gemm_nt_q") continue;
        const char* enc = nullptr;
        for (const KfParam& p : c.params)
            if (p.key == "dtype" && p.kind == RAD_P_STR) enc = p.s.c_str();
        if (want_enc && (!enc || strcmp(want_enc, enc))) continue;
        ++shown;
        printf("== %s band=%s", c.op.c_str(), c.band.c_str());
        for (const KfParam& p : c.params) {
            if (p.kind == RAD_P_INT)      printf(" %s=%lld", p.key.c_str(), p.i);
            else if (p.kind == RAD_P_STR) printf(" %s=%s", p.key.c_str(), p.s.c_str());
            else                          printf(" %s=%g", p.key.c_str(), p.d);
        }
        printf(" kf.m=%lld\n", (long long)c.m);
        for (size_t k = 0; k < c.opd.size(); ++k) {
            const KfOperand& d = c.opd[k];
            printf("   opd[%zu] dtype=%-10s role=%d", k, dt_name(d.dtype), d.role);
            if (d.absent) { printf(" ABSENT\n"); continue; }
            printf(" shape=[");
            long long numel = 1;
            for (size_t i = 0; i < d.shape.size(); ++i) {
                printf("%s%lld", i ? "," : "", (long long)d.shape[i]);
                numel *= (long long)d.shape[i];
            }
            printf("] numel=%lld%s\n", numel, d.has_enc ? " ENCODED" : "");
        }
    }
    printf("cases: %d\n", shown);
    return 0;
}
