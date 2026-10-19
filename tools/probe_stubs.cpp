/* Stub definitions for kf_probe's link: kf_probe reads fixture metadata only, so
 * the rad_core draw/fill helpers kfixture.cpp references are never executed. */
#include "opshapes.h"
#include <functional>
#include <string>

namespace rad {

const int64_t RAD_FILL_BLOCK = 0;

void rad_for_blocks(long, const std::function<void(long)>&) {}
void rad_fill_normal(float*, long, unsigned long, float) {}
void rad_fill_normal_block(float*, long, unsigned long, long, float) {}
void rad_narrow(const float*, unsigned, long, void*) {}
bool enc_selection_logical(const RadEncoding&, const int*, int, const long*, const long*,
                           const void* const*, unsigned, long, void*, std::string*) { return false; }

} /* namespace rad */
