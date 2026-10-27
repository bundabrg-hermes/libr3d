// q46_policy_check.cpp -- host-side differential check of the Q46 split policy.
// REPLICATES decode_splits() from r3d_attn_paged_decode.hip (both arms) against the
// v153 closed form over the shape grid. Arm 0 MUST equal the closed form everywhere;
// arm 1 prints the policy picks (the sweep table's expected values). Pure host code:
//   g++ -O2 -std=c++17 tools/q46_policy_check.cpp -o /tmp/q46pc && /tmp/q46pc [CU]
// (the shipped arm reads multiProcessorCount; the XTX answers 96 -- pass any other
// value here to inspect the rule's behavior on a hypothetical device).
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <algorithm>

static int64_t old_form(int64_t q_heads, int64_t n_seq, int64_t tiles_kv, bool fold) {
    const int64_t units = q_heads * (n_seq > 0 ? n_seq : 1);
    int64_t s = (fold ? 128 : 192) / units;
    const int64_t cap = fold ? tiles_kv : tiles_kv / 2;
    if (s > cap) s = cap;
    if (s > tiles_kv) s = tiles_kv;
    if (s > 64) s = 64;
    if (s < 1) s = 1;
    return s;
}

static int64_t wave_form(int64_t q_heads, int64_t n_seq, int64_t tiles_kv, bool fold, int cu) {
    const int64_t units = q_heads * (n_seq > 0 ? n_seq : 1);
    const int64_t cap = fold ? tiles_kv : tiles_kv / 2;
    int64_t smax = std::min(tiles_kv, (int64_t)64);
    if (cap < smax) smax = cap;
    if (smax < 1) smax = 1;
    const int64_t fill = (int64_t)cu * 8 / 10;
    int64_t s = 0;
    double best = 0.0;
    for (int64_t t = 1; t <= smax; ++t) {
        const int64_t blocks = units * t;
        if (blocks >= fill) { s = t; break; }
        const int64_t waves = (blocks + cu - 1) / cu;
        const double eff = (double)blocks / (double)(waves * cu);
        if (eff > best) best = eff;
    }
    if (s <= 0) {
        for (int64_t t = 1; t <= smax; ++t) {
            const int64_t blocks = units * t;
            const int64_t waves = (blocks + cu - 1) / cu;
            const double eff = (double)blocks / (double)(waves * cu);
            if (eff >= 0.85 * best) { s = t; break; }
        }
    }
    if (s <= 0) s = 1;
    if (s > smax) s = smax;
    if (s < 1) s = 1;
    return s;
}

int main(int argc, char** argv) {
    const int cu = argc > 1 ? atoi(argv[1]) : 96;
    int fails = 0, rows = 0;
    printf("== arm0-vs-v153 differential (must be identical) + arm1 picks, CU=%d ==\n", cu);
    const int64_t heads[] = {1, 2, 4, 8, 16, 32, 64, 128};
    const int64_t seqs[]  = {1, 2, 3, 4, 8, 16, 32, 64};
    const int64_t tkvs[]  = {0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512};
    for (int64_t h : heads) for (int64_t n : seqs) for (int64_t tk : tkvs) for (int f = 0; f < 2; ++f) {
        const int64_t a = old_form(h, n, tk, f), b = wave_form(h, n, tk, f, cu);
        ++rows;
        // Arm 1 must never exceed the old arm's envelope (cap floors to 1 exactly as
        // v153 does: spl==1 means "no split arm at all", the entry takes the tile):
        const int64_t env = f ? std::max(tk, (int64_t)1) : std::max(tk / 2, (int64_t)1);
        if (b > 64 || b > std::min(env, std::max(tk, (int64_t)1))) { ++fails;
            printf("CAPFAIL h=%ld n=%ld tk=%ld fold=%d wave=%ld\n", (long)h, (long)n, (long)tk, f, (long)b); }
        if (f == 0 && (h * std::max(n, (int64_t)1)) >= 64) continue;  // sample-print head
        if (n == 1 && (tk == 32 || tk == 128 || tk == 256))
            printf("h=%3ld n=%2ld tk=%3ld fold=%d  fixed=%2ld wave=%2ld\n", (long)h, (long)n, (long)tk, f, (long)a, (long)b);
    }
    // exhaustive arm0 equality on the FULL grid (shipped arm0 as coded):
    for (int64_t h : heads) for (int64_t n : seqs) for (int64_t tk : tkvs) for (int f = 0; f < 2; ++f) {
        // shipped arm0 (as coded): smax=min(cap,tkv,64); s=min(target,smax); >=1
        const int64_t units = h * std::max(n, (int64_t)1);
        const int64_t cap = f ? tk : tk / 2;
        int64_t smax = std::min(tk, (int64_t)64); if (cap < smax) smax = cap; if (smax < 1) smax = 1;
        int64_t s = (f ? 128 : 192) / units; if (s > smax) s = smax; if (s < 1) s = 1;
        if (s != old_form(h, n, tk, f)) { ++fails;
            printf("ARM0DIFF h=%ld n=%ld tk=%ld fold=%d shipped=%ld v153=%ld\n", (long)h, (long)n, (long)tk, f, (long)s, (long)old_form(h, n, tk, f)); }
    }
    printf("grid rows=%d fails=%d %s\n", rows, fails, fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}