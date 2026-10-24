// rows_dump -- enumerate gemm_nt_q rows in a plugin .so: domain, priority, constraints.
// Build: g++ -std=c++20 -I/opt/radiance/include/radiance -w tools/rows_dump.cpp -o /tmp/rd -ldl
#include <dlfcn.h>
#include <cstdio>
#include <cstring>
#include "rad_abi.h"

typedef const RadKernelInfo* (*AtFn)(int);
typedef int (*CountFn)();

static const char* ckind(int k) {
    switch (k) { case RAD_C_EQ: return "EQ"; case RAD_C_LE: return "LE"; case RAD_C_GE: return "GE";
                 case RAD_C_DIV: return "DIV"; case RAD_C_IN: return "IN"; default: return "?"; }
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: rows_dump <lib.so> [op-filter]\n"); return 1; }
    void* dl = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!dl) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }
    auto count = (CountFn)dlsym(dl, "rad_kernel_count");
    auto at    = (AtFn)dlsym(dl, "rad_kernel_at");
    if (!count || !at) { fprintf(stderr, "no table\n"); return 1; }
    const int n = count();
    for (int i = 0; i < n; ++i) {
        const RadKernelInfo* k = at(i);
        if (!k || !k->op) continue;
        if (argc > 2 && strcmp(k->op, argv[2]) != 0) continue;
        printf("%s | %s | %s | dom=%d prio=%d | %d constraints\n",
               argv[1], k->op, k->name ? k->name : "?", (int)k->domain, k->priority,
               k->n_constraints);
        for (int c = 0; c < k->n_constraints; ++c)
            printf("    %s %s %lld %s\n", k->constraints[c].key, ckind(k->constraints[c].op),
                   (long long)k->constraints[c].ival,
                   k->constraints[c].sval ? k->constraints[c].sval : "");
    }
    return 0;
}
