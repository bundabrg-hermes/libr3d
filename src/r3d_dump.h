/* r3d_dump.h -- env-gated operand capture for suspect rows, so a live serve's real
 * launches can be replayed against the reference host rows offline (tools/replay_dump).
 *
 * Why this exists: every recorded kbench case compares our device row to libref on
 * inputs kbench itself built. A bug that lives only on the engine's runtime inputs --
 * dtypes, views, params no pack recorded -- is invisible to every gate we own. The
 * serve walk is where such a bug surfaces, and by then the inputs are gone. This
 * captures them: for the rows named in R3D_DUMP_OPS, the first R3D_DUMP_MAX launches
 * per row save every operand's bytes before and after the real launch, plus the whole
 * parameter list. The replay tool rebuilds the launch on host buffers, runs libref's
 * host row, and diffs it against what the card actually wrote.
 *
 * Cost when R3D_DUMP is unset: one static branch per wrapped entry. When set, this is
 * a debug path: it synchronises and mallocs, which a production launch must never do
 * (RadKernelInfo::launch's contract) -- hence the gate, and hence it ships disabled.
 *
 * Files per captured launch:  <dir>/<seq>_<row>/manifest.tsv, pre.bin, post.bin.
 */
#ifndef R3D_DUMP_H
#define R3D_DUMP_H

#include "rad_abi.h"

struct R3dDumpSlot { void* priv; };

void r3d_dump_begin(R3dDumpSlot* slot, const char* row, const void* impl, const RadArgs* a, RadStream s);
void r3d_dump_end(R3dDumpSlot* slot, int rc);

/* Wrap an extern "C" entry: the real implementation becomes `impl`, this keeps the
 * exported name and captures the launch when the environment asks for it. */
#define R3D_DUMP_WRAP(rowname, impl)                                        \
    { R3dDumpSlot ds_; r3d_dump_begin(&ds_, rowname, (const void*)(impl), a, s); \
      int rc_ = impl(a, s); r3d_dump_end(&ds_, rc_); return rc_; }

#endif /* R3D_DUMP_H */
