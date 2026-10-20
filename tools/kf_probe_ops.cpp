/* Probe: dump fixture cases for a LIST of ops (argv[2..]), one sample per distinct
 * param digest, so rows for mul / sigmoid / rope_table / the GDN family can be written
 * against the recorded geometries without touching a model. Sibling of kf_probe.cpp and
 * kf_probe_q.cpp; those hardcode one op (or one dtype) and this does not. */
#include "kfixture.h"
#include "rad_types.h"
#include <cstdio>
#include <cstring>
#include <set>
#include <string>

static const char* dt_name(uint32_t d) {
    for (unsigned i = 0; i < sizeof(rad_dt_table) / sizeof(rad_dt_table[0]); ++i)
        if (rad_dt_table[i].dt == d) return rad_dt_table[i].name;
    return "?";
}

int main(int argc, char** argv) {
    using namespace rad;
    if (argc < 3) { printf("usage: kf_probe_ops FILE op [op...] [shape]\n"); return 2; }
    const bool want_shape = !strcmp(argv[argc - 1], "shape");
    rad::Kfixture fx;
    if (fx.load(argv[1]) != 0) { printf("load failed\n"); return 1; }
    std::set<std::string> done;
    for (const KfCase& c : fx.cases) {
        bool want = false;
        for (int i = 2; i < argc - (want_shape ? 1 : 0); ++i)
            if (c.op == argv[i]) want = true;
        if (!want) continue;
        std::string key = c.op;
        for (const KfParam& p : c.params) {
            char buf[128];
            if (p.kind == RAD_P_INT)      snprintf(buf, sizeof buf, " %s=%lld", p.key.c_str(), p.i);
            else if (p.kind == RAD_P_STR) snprintf(buf, sizeof buf, " %s=%s", p.key.c_str(), p.s.c_str());
            else                          snprintf(buf, sizeof buf, " %s=%.7g", p.key.c_str(), p.d);
            key += buf;
        }
        if (!done.insert(key).second) continue;
        printf("== %s\n", key.c_str());
        if (want_shape)
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
    printf("distinct: %zu\n", done.size());
    return 0;
}
