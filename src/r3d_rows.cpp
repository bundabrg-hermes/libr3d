1|/* r3d_rows.cpp -- libr3d's declaration: plugin identity and the kernel row table.
2| *
3| * Pure host C++ including only the ABI, so it compiles without ROCm, exactly like libr4d's
4| * r4d_rows.cpp. This skeleton exports the full kernel-plugin contract with an empty table:
5| * the loader takes it (ABI matches), finds no device rows for gfx11, and selection behaves as
6| * if libr3d were absent. Real rows land here as kernels are implemented; docs/COVERAGE.md
7| * tracks the order.
8| */
9|
10|#include "rad_abi.h"
11|#include <cstdlib>   // getenv, for the R3D_NO_GDN registration filter
12|#include <cstring>   // strcmp, same
13|
14|extern "C" {
15|
16|uint32_t rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }
17|
18|const RadPluginInfo* rad_plugin_info(void) {
19|    static const RadPluginInfo info = {
20|        RAD_PLUGIN_KERNEL,
21|        "libr3d",
22|        "0.1.0",
23|        "radiance kernel library for AMD RDNA3 (gfx11)",
24|        "gfx1100 gfx1101 gfx1102 gfx1103 gfx11-generic",
25|    };
26|    return &info;
27|}
28|
29|/* Kernel rows and their op schemas. Every op a plugin implements must have its schema
30| * declared by some *loaded kernel plugin* (loader.cpp stages rad_kernel_schema_count/at per
31| * plugin; libref is a kbench oracle, not a search-path plugin, so it fixes nothing). On RDNA3
32| * libr3d is that declarer, and a kernel whose op has no declared schema is an error:
33| *   "implements op 'rmsnorm', which no loaded plugin declares a schema for"
34| * The schema below is copied verbatim from libr4d's kSchemas entry, so where libr4d is also
35| * present (an RDNA4 box or a mixed rig) the loader's schema_same() check passes and first
36| * declarer wins with no refusal. Any future op libr3d implements must likewise arrive with
37| * its schema declared here, identical to libr4d's where one exists. */
38|
39|/* rmsnorm: verbatim libr4d (r4d_rows.cpp pRmsNorm/oRmsNorm). `wadd` is GemmaRMSNorm's `1 +`;
40| * `mult` scales the row before its one rounding. `w` optional: absent, the gain is one. */
41|static const RadParamSpec pRmsNorm[] = {
42|    { "M",     RAD_P_INT, RAD_REQUIRED },
43|    { "n",     RAD_P_INT, RAD_REQUIRED },
44|    { "eps",   RAD_P_F64, RAD_REQUIRED },
45|    { "dtype", RAD_P_STR, RAD_REQUIRED },
46|    { "wadd",  RAD_P_F64, RAD_OPTIONAL },
47|    { "mult",  RAD_P_F64, RAD_OPTIONAL },
48|};
49|static const RadOperandSpec oRmsNorm[] = {
50|    { "x", RAD_OPD_IN,     0 },
51|    { "w", RAD_OPD_WEIGHT, 1 },
52|    { "y", RAD_OPD_OUT,    0 },
53|};
54|
55|/* embed_lookup: verbatim libr4d (r4d_rows.cpp pEmbedM/oEmbed). `mult` is Gemma's sqrt(hidden);
56| * `vocab_offset` shards the table across TP ranks; `wscale` belongs to embed_lookup_q's
57| * quantised table and is unused by this bf16 row. */
58|static const RadParamSpec pEmbedM[] = {
59|    { "M",       RAD_P_INT, RAD_REQUIRED },
60|    { "n_embd",  RAD_P_INT, RAD_REQUIRED },
61|    { "n_vocab", RAD_P_INT, RAD_REQUIRED },
62|    { "dtype",   RAD_P_STR, RAD_REQUIRED },
63|    { "vocab_offset", RAD_P_INT, RAD_OPTIONAL },
64|    { "wscale",  RAD_P_F64, RAD_OPTIONAL },
65|    { "mult",    RAD_P_F64, RAD_OPTIONAL },
66|};
67|static const RadOperandSpec oEmbed[] = {
68|    { "tokens", RAD_OPD_IN,     0 },
69|    { "wte",    RAD_OPD_WEIGHT, 0 },
70|    { "x",      RAD_OPD_OUT,    0 },
71|};
72|
73|/* kv_store: verbatim libr4d (r4d_rows.cpp pKvStore/oKvStore). The scales are the reader's
74| * descales and unused for the bf16 cache this row serves. */
75|static const RadParamSpec pKvStore[] = {
76|    { "M",          RAD_P_INT, RAD_REQUIRED },
77|    { "head_dim",   RAD_P_INT, RAD_REQUIRED },
78|    { "n_head_kv",  RAD_P_INT, RAD_REQUIRED },
79|    { "block_size", RAD_P_INT, RAD_REQUIRED },
80|    { "kv_dtype",   RAD_P_STR, RAD_REQUIRED },
81|    { "k_scale",    RAD_P_F64, RAD_OPTIONAL },
82|    { "v_scale",    RAD_P_F64, RAD_OPTIONAL },
83|};
84|static const RadOperandSpec oKvStore[] = {
85|    { "k",            RAD_OPD_IN,    0 },
86|    { "v",            RAD_OPD_IN,    0 },
87|    { "slot_mapping", RAD_OPD_IN,    0 },
88|    { "kv_cache",     RAD_OPD_INOUT, 0 },
89|};
90|
91|/* act_mul: verbatim libr4d (r4d_rows.cpp pActMul/oBinary). The gated MLP pair: act applied to
92| * the gate operand, rounded to y's dtype, times the up operand. */
93|static const RadParamSpec pActMul[] = {
94|    { "M",    RAD_P_INT, RAD_REQUIRED },
95|    { "n",    RAD_P_INT, RAD_REQUIRED },
96|    { "act",  RAD_P_STR, RAD_REQUIRED },
97|    { "dtype", RAD_P_STR, RAD_REQUIRED },
98|};
99|static const RadOperandSpec oBinary[] = {
100|    { "a", RAD_OPD_IN,  0 },
101|    { "b", RAD_OPD_IN,  0 },
102|    { "y", RAD_OPD_OUT, 0 },
103|};
104|
105|/* cast: verbatim libr4d (r4d_rows.cpp pCast/oUnary). from/to are for the selector; the
106| * tensors' own dtypes are what the kernel reads. */
107|static const RadParamSpec pCast[] = {
108|    { "M",    RAD_P_INT, RAD_REQUIRED },
109|    { "n",    RAD_P_INT, RAD_REQUIRED },
110|    { "from", RAD_P_STR, RAD_REQUIRED },
111|    { "to",   RAD_P_STR, RAD_REQUIRED },
112|};
113|static const RadOperandSpec oUnary[] = {
114|    { "x", RAD_OPD_IN,  0 },
115|    { "y", RAD_OPD_OUT, 0 },
116|};
117|
118|/* rope: verbatim libr4d (r4d_rows.cpp pRope/oRope). The yarn and llama3 frequency-scaling
119| * parameters are declared (the schema must match libr4d's wherever both plugins load) but
120| * this plugin's row refuses them; the bf16 text models that are libr3d's scope do not use them. */
121|static const RadParamSpec pRope[] = {
122|    { "M",          RAD_P_INT, RAD_REQUIRED },
123|    { "head_dim",   RAD_P_INT, RAD_REQUIRED },
124|    { "n_head",     RAD_P_INT, RAD_REQUIRED },
125|    { "n_head_kv",  RAD_P_INT, RAD_REQUIRED },
126|    { "theta",      RAD_P_F64, RAD_REQUIRED },
127|    { "scale",      RAD_P_F64, RAD_REQUIRED },
128|    { "mode",       RAD_P_STR, RAD_REQUIRED },
129|    { "rotary_dim", RAD_P_INT, RAD_OPTIONAL },
130|    { "sections",   RAD_P_STR, RAD_OPTIONAL },
131|    { "yarn_factor", RAD_P_F64, RAD_OPTIONAL },
132|    { "yarn_low", RAD_P_F64, RAD_OPTIONAL },
133|    { "yarn_high", RAD_P_F64, RAD_OPTIONAL },
134|    { "yarn_magnitude", RAD_P_F64, RAD_OPTIONAL },
135|    { "ladder",     RAD_P_STR, RAD_OPTIONAL },   /* the base ladder's width: rotary|head */
136|    /* Llama 3's frequency scaling (abi/rad_yarn.h, RadLlama3). */
137|    { "llama3_factor", RAD_P_F64, RAD_OPTIONAL },
138|    { "llama3_low",    RAD_P_F64, RAD_OPTIONAL },
139|    { "llama3_high",   RAD_P_F64, RAD_OPTIONAL },
140|    { "llama3_orig",   RAD_P_F64, RAD_OPTIONAL },
141|};
142|static const RadOperandSpec oRope[] = {
143|    { "qkv",       RAD_OPD_INOUT, 0 },
144|    { "positions", RAD_OPD_IN,    0 },
145|};
146|
147|/* gemm_nt: verbatim libr4d (r4d_rows.cpp pGemmNt/oGemmNt). The quantised gemm_nt_q is
148| * declared next; its mxfp4a8 encoding has a row (r3d_gemm_mxfp4.hip, bf16 boundary
149| * conversion -- RDNA3 has no fp8/fp4 hardware), the int grids are later tiers, and the
150| * _bias/gated variants have no rows yet. */
151|static const RadParamSpec pGemmNt[] = {
152|    { "M",     RAD_P_INT, RAD_REQUIRED },
153|    { "N",     RAD_P_INT, RAD_REQUIRED },
154|    { "K",     RAD_P_INT, RAD_REQUIRED },
155|    { "dtype", RAD_P_STR, RAD_REQUIRED },
156|};
157|static const RadOperandSpec oGemmNt[] = {
158|    { "a", RAD_OPD_IN,     0 },
159|    { "b", RAD_OPD_WEIGHT, 0 },
160|    { "y", RAD_OPD_OUT,    0 },
161|};
162|
163|/* gemm_nt_q: verbatim libr4d (r4d_rows.cpp pGemmNtQ/oGemmNtQ) so schema_same() passes
164| * when the plugins co-load. a_sum/b_ref are the seventh/sixth operand tail RadArgs
165| * cannot carry as weight aux (the ABI gap libr4d's comment spells out); the mxfp4a8
166| * entry refuses a_sum, and accepts-and-ignores b_ref exactly as both host libraries. */
167|static const RadParamSpec pGemmNtQ[] = {
168|    { "M",     RAD_P_INT, RAD_REQUIRED },
169|    { "N",     RAD_P_INT, RAD_REQUIRED },
170|    { "K",     RAD_P_INT, RAD_REQUIRED },
171|    { "group", RAD_P_INT, RAD_REQUIRED },
172|    { "dtype", RAD_P_STR, RAD_REQUIRED },
173|    { "pb1",   RAD_P_INT, RAD_OPTIONAL },
174|    { "pb2",   RAD_P_INT, RAD_OPTIONAL },
175|};
176|static const RadOperandSpec oGemmNtQ[] = {
177|    { "a",       RAD_OPD_IN,     0 },
178|    { "a_scale", RAD_OPD_IN,     1 },
179|    { "b",       RAD_OPD_WEIGHT, 0 },
180|    { "b_scale", RAD_OPD_WEIGHT, 0 },
181|    { "y",       RAD_OPD_OUT,    0 },
182|    { "a_sum",   RAD_OPD_IN,     1 },
183|    { "b_ref",   RAD_OPD_WEIGHT, 1 },
184|};
185|
186|/* logits_gemm: verbatim libr4d (r4d_rows.cpp pLogits/oLogits). Same three operands in the
187| * same order as gemm_nt, the extents spelled n_vocab / n_embd. */
188|static const RadParamSpec pLogits[] = {
189|    { "M",       RAD_P_INT, RAD_REQUIRED },
190|    { "n_vocab", RAD_P_INT, RAD_REQUIRED },
191|    { "n_embd",  RAD_P_INT, RAD_REQUIRED },
192|    { "dtype",   RAD_P_STR, RAD_REQUIRED },
193|};
194|static const RadOperandSpec oLogits[] = {
195|    { "x",       RAD_OPD_IN,     0 },
196|    { "lm_head", RAD_OPD_WEIGHT, 0 },
197|    { "logits",  RAD_OPD_OUT,    0 },
198|};
199|
200|/* attn_paged, the parameter and operand lists copied verbatim from libr4d's pAttnPaged /
201| * oAttnPaged (r4d_rows.cpp) so co-load schema_same() passes. block_size is RAD_DERIVED --
202| * read off the cache tensor, its constraint admits any value -- and the trailing
203| * n_head / max_seqs exist for libr4d's split-KV scratch hook; ref ignores them and so does
204| * our row, but both plugins must agree on the list. */
205|static const RadParamSpec pAttnPaged[] = {
206|    { "q_len",      RAD_P_INT, RAD_REQUIRED },
207|    { "head_dim",   RAD_P_INT, RAD_REQUIRED },
208|    { "gqa",        RAD_P_INT, RAD_REQUIRED },
209|    { "block_size", RAD_P_INT, RAD_DERIVED  },
210|    { "causal",     RAD_P_INT, RAD_REQUIRED },
211|    { "window",     RAD_P_INT, RAD_REQUIRED },
212|    { "q_dtype",    RAD_P_STR, RAD_REQUIRED },
213|    { "kv_dtype",   RAD_P_STR, RAD_REQUIRED },
214|    { "scale",      RAD_P_F64, RAD_OPTIONAL },
215|    { "max_ctx",    RAD_P_INT, RAD_OPTIONAL },
216|    { "n_head",     RAD_P_INT, RAD_OPTIONAL },
217|    { "max_seqs",   RAD_P_INT, RAD_OPTIONAL },
218|};
219|static const RadOperandSpec oAttnPaged[] = {
220|    { "q",           RAD_OPD_IN,  0 },
221|    { "kv_cache",    RAD_OPD_IN,  0 },
222|    { "block_table", RAD_OPD_IN,  0 },
223|    { "seqused",     RAD_OPD_IN,  0 },
224|    { "k_descale",   RAD_OPD_IN,  1 },
225|    { "v_descale",   RAD_OPD_IN,  1 },
226|    { "cu_seqlens",  RAD_OPD_IN,  1 },
227|    { "out",         RAD_OPD_OUT, 0 },
228|};
229|
230|/* ---- sampling chain specs (verbatim from libr4d: pMVocab/pTopk/pMCand, oTemp/oMask/
231| * oTopk/oArgmax/oNarrow/oPick). The engine's `params` operand is the RadSampleParams byte
232| * blob (abi/rad_sample.h); n_vocab/n_cand name the leading dimension, vocab_off is the
233| * shard origin and stays 0 in this single-rank scope. */
234|static const RadParamSpec pMVocab[] = {
235|    { "M",       RAD_P_INT, RAD_REQUIRED },
236|    { "n_vocab", RAD_P_INT, RAD_REQUIRED },
237|};
238|static const RadParamSpec pTopk[] = {
239|    { "M",         RAD_P_INT, RAD_REQUIRED },
240|    { "n_vocab",   RAD_P_INT, RAD_REQUIRED },
241|    { "vocab_off", RAD_P_INT, RAD_OPTIONAL },
242|};
243|static const RadParamSpec pMCand[] = {
244|    { "M",      RAD_P_INT, RAD_REQUIRED },
245|    { "n_cand", RAD_P_INT, RAD_REQUIRED },
246|};
247|/* ---- the history/summary stages (verbatim from libr4d: pPenal/oPenalties/oDry and
248| * pLogprobs/oLogprobs). typical and xtc reuse pMCand/oNarrow -- same three operands,
249| * narrowed in place. ---- */
250|static const RadParamSpec pPenal[] = {
251|    { "M",         RAD_P_INT, RAD_REQUIRED },
252|    { "n_vocab",   RAD_P_INT, RAD_REQUIRED },
253|    { "vocab_off", RAD_P_INT, RAD_OPTIONAL },
254|};
255|static const RadOperandSpec oPenalties[] = {
256|    { "logits",  RAD_OPD_INOUT, 0 },
257|    { "params",  RAD_OPD_IN,    0 },
258|    { "history", RAD_OPD_IN,    0 },
259|};
260|static const RadOperandSpec oDry[] = {
261|    { "logits",   RAD_OPD_INOUT, 0 },
262|    { "params",   RAD_OPD_IN,    0 },
263|    { "history",  RAD_OPD_IN,    0 },
264|    { "breakers", RAD_OPD_IN,    1 },
265|};
266|static const RadParamSpec pLogprobs[] = {
267|    { "M",         RAD_P_INT, RAD_REQUIRED },
268|    { "n_vocab",   RAD_P_INT, RAD_REQUIRED },
269|    { "k",         RAD_P_INT, RAD_REQUIRED },
270|    { "vocab_off", RAD_P_INT, RAD_OPTIONAL },
271|};
272|static const RadOperandSpec oLogprobs[] = {
273|    { "logits", RAD_OPD_IN,  0 },
274|    { "params", RAD_OPD_IN,  0 },
275|    { "stats",  RAD_OPD_OUT, 0 },
276|};
277|/* ---- band-op specs (verbatim from libr4d, the plugin that fixes these schemas). ---- */
278|static const RadParamSpec pAdd[] = {
279|    { "M",     RAD_P_INT, RAD_REQUIRED },
280|    { "n",     RAD_P_INT, RAD_REQUIRED },
281|    { "dtype", RAD_P_STR, RAD_REQUIRED },
282|    { "div",   RAD_P_F64, RAD_OPTIONAL },
283|};
284|/* oBinary (a, b, y) is already declared above for the act/cast batch -- identical table. */
285|static const RadParamSpec pMnDtype[] = {
286|    { "M",     RAD_P_INT, RAD_REQUIRED },
287|    { "n",     RAD_P_INT, RAD_REQUIRED },
288|    { "dtype", RAD_P_STR, RAD_REQUIRED },
289|};
290|static const RadOperandSpec oGather[] = {
291|    { "x",   RAD_OPD_IN,  0 },
292|    { "idx", RAD_OPD_IN,  0 },
293|    { "y",   RAD_OPD_OUT, 0 },
294|};
295|/* rope_table: verbatim libref (ref_registry.cpp pRopeTable/oRopeTable). The 27B graph
296| * issues it for its partial rotary (rot 64 of head_dim 256); libr4d declares the same
297| * table, so a mixed rig dedupes it with first-declarer-wins. */
298|static const RadParamSpec pRopeTable[] = {
299|    { "M",             RAD_P_INT, RAD_REQUIRED },
300|    { "rot",           RAD_P_INT, RAD_REQUIRED },
301|    { "theta",         RAD_P_F64, RAD_REQUIRED },
302|    { "scale",         RAD_P_F64, RAD_REQUIRED },
303|    { "dtype",         RAD_P_STR, RAD_REQUIRED },
304|    { "yarn_factor",   RAD_P_F64, RAD_OPTIONAL },
305|    { "yarn_low",      RAD_P_F64, RAD_OPTIONAL },
306|    { "yarn_high",     RAD_P_F64, RAD_OPTIONAL },
307|    { "yarn_magnitude",RAD_P_F64, RAD_OPTIONAL },
308|    { "head_dim",      RAD_P_INT, RAD_OPTIONAL },
309|    { "ladder",        RAD_P_STR, RAD_OPTIONAL },
310|};
311|static const RadOperandSpec oRopeTable[] = {
312|    { "positions", RAD_OPD_IN,  0 },
313|    { "cos_sin",   RAD_OPD_OUT, 0 },
314|};
315|/* gdn_gated_rmsnorm: verbatim libref (pGatedNorm/oGatedNorm). */
316|static const RadParamSpec pGatedNorm[] = {
317|    { "M",        RAD_P_INT, RAD_REQUIRED },
318|    { "channels", RAD_P_INT, RAD_REQUIRED },
319|    { "eps",      RAD_P_F64, RAD_REQUIRED },
320|    { "act",      RAD_P_STR, RAD_REQUIRED },
321|};
322|static const RadOperandSpec oGatedNorm[] = {
323|    { "x", RAD_OPD_IN,     0 },
324|    { "z", RAD_OPD_IN,     0 },
325|    { "w", RAD_OPD_WEIGHT, 0 },
326|    { "o", RAD_OPD_OUT,    0 },
327|};
328|/* gdn_conv_update: verbatim libref (pConvUpdate/oConvUpdate). num_accepted is required by
329| * the schema -- ref's comment: an op that lets the caller omit it is an op whose rollback
330| * contract is optional. */
331|static const RadParamSpec pConvUpdate[] = {
332|    { "q_len",        RAD_P_INT, RAD_REQUIRED },
333|    { "head_k",       RAD_P_INT, RAD_REQUIRED },
334|    { "head_v",       RAD_P_INT, RAD_REQUIRED },
335|    { "conv_width",   RAD_P_INT, RAD_REQUIRED },
336|    { "max_query_len",RAD_P_INT, RAD_OPTIONAL },
337|};
338|static const RadOperandSpec oConvUpdate[] = {
339|    { "x",            RAD_OPD_IN,     0 },
340|    { "w",            RAD_OPD_WEIGHT, 0 },
341|    { "b",            RAD_OPD_WEIGHT, 1 },
342|    { "conv_state",   RAD_OPD_INOUT,  0 },
343|    { "state_idx",    RAD_OPD_IN,     0 },
344|    { "num_accepted", RAD_OPD_IN,     0 },
345|    { "cu",           RAD_OPD_IN,     0 },
346|    { "q",            RAD_OPD_OUT,    0 },
347|    { "k",            RAD_OPD_OUT,    0 },
348|    { "v",            RAD_OPD_OUT,    0 },
349|};
350|/* gdn_kkt_solve: verbatim libref (pGdnKktSolve/oGdnKktSolve). chunk is DERIVED with the
351| * seq-chunk role, exactly libr4d's P_CHUNK spelling. */
352|static const RadParamSpec pKktSolve[] = {
353|    { "M",      RAD_P_INT, RAD_REQUIRED },
354|    { "head_k", RAD_P_INT, RAD_REQUIRED },
355|    { "chunk",  RAD_P_INT, RAD_DERIVED, RAD_PROLE_SEQ_CHUNK },
356|};
357|static const RadOperandSpec oKktSolve[] = {
358|    { "k",    RAD_OPD_IN,  0 },
359|    { "beta", RAD_OPD_IN,  0 },
360|    { "g",    RAD_OPD_IN,  0 },
361|    { "cu",   RAD_OPD_IN,  0 },
362|    { "A",    RAD_OPD_OUT, 0 },
363|};
364|/* gdn_chunk_scan: verbatim libref (pChunkScan/oChunkScan), including the optional h0/ht/
365| * state_idx operands and the optional f64 `scale` and `state_fp16` parameters (ref reads the
366| * first, ignores the second). */
367|static const RadParamSpec pChunkScan[] = {
368|    { "M",          RAD_P_INT,   RAD_REQUIRED },
369|    { "head_k",     RAD_P_INT,   RAD_REQUIRED },
370|    { "head_v",     RAD_P_INT,   RAD_REQUIRED },
371|    { "chunk",      RAD_P_INT,   RAD_DERIVED, RAD_PROLE_SEQ_CHUNK },
372|    { "scale",      RAD_P_F64,   RAD_OPTIONAL },
373|    { "state_fp16", RAD_P_INT,   RAD_OPTIONAL },
374|};
375|static const RadOperandSpec oChunkScan[] = {
376|    { "q",         RAD_OPD_IN,  0 },
377|    { "k",         RAD_OPD_IN,  0 },
378|    { "v",         RAD_OPD_IN,  0 },
379|    { "A",         RAD_OPD_IN,  0 },
380|    { "g",         RAD_OPD_IN,  0 },
381|    { "beta",      RAD_OPD_IN,  0 },
382|    { "h0",        RAD_OPD_IN,  1 },
383|    { "cu",        RAD_OPD_IN,  0 },
384|    { "o",         RAD_OPD_OUT, 0 },
385|    { "ht",        RAD_OPD_OUT, 1 },
386|    { "state_idx", RAD_OPD_IN,  1 },
387|};
388|/* gdn_recurrent_update: verbatim libref/libr4d (pGdnRecurrentUpdate/oGdnRecurrentUpdate).
389| * q_len is required (the 27B pack records it, 1..128); the rest optional with the launcher
390| * defaults matching ref's. norm_w is ROLE_WEIGHT like the oracle's, and its plain bf16 single
391| * plane (this container's storage) needs no layout hook -- the row reads what it is given. */
392|static const RadParamSpec pRecurrent[] = {
393|    { "q_len",        RAD_P_INT, RAD_REQUIRED },
394|    { "head_k",       RAD_P_INT, RAD_REQUIRED },
395|    { "head_v",       RAD_P_INT, RAD_REQUIRED },
396|    { "n_head_v",     RAD_P_INT, RAD_OPTIONAL },
397|    { "scale",        RAD_P_F64, RAD_OPTIONAL },
398|    { "eps",          RAD_P_F64, RAD_OPTIONAL },
399|    { "l2_eps",       RAD_P_F64, RAD_OPTIONAL },
400|    { "softplus_thr", RAD_P_F64, RAD_OPTIONAL },
401|    { "act",          RAD_P_STR, RAD_OPTIONAL },
402|    { "state_fp16",   RAD_P_INT, RAD_OPTIONAL },
403|    { "state_form",   RAD_P_STR, RAD_OPTIONAL },
404|};
405|static const RadOperandSpec oRecurrent[] = {
406|    { "q",            RAD_OPD_IN,     0 },
407|    { "k",            RAD_OPD_IN,     0 },
408|    { "v",            RAD_OPD_IN,     0 },
409|    { "a",            RAD_OPD_IN,     0 },
410|    { "b",            RAD_OPD_IN,     0 },
411|    { "A_log",        RAD_OPD_WEIGHT, 0 },
412|    { "dt_bias",      RAD_OPD_WEIGHT, 0 },
413|    { "state",        RAD_OPD_INOUT,  0 },
414|    { "state_idx",    RAD_OPD_IN,     0 },
415|    { "num_accepted", RAD_OPD_IN,     0 },
416|    { "cu",           RAD_OPD_IN,     0 },
417|    { "z",            RAD_OPD_IN,     1 },
418|    { "norm_w",       RAD_OPD_WEIGHT, 1 },
419|    { "o",            RAD_OPD_OUT,    0 },
420|    { "o_q",          RAD_OPD_OUT,    1 },
421|    { "o_scale",      RAD_OPD_OUT,    1 },
422|};
423|static const RadParamSpec pQuantActFp8[] = {
424|    { "M",     RAD_P_INT, RAD_REQUIRED },
425|    { "n",     RAD_P_INT, RAD_REQUIRED },
426|    { "group", RAD_P_INT, RAD_REQUIRED },
427|    { "dtype", RAD_P_STR, RAD_REQUIRED },
428|};
429|static const RadOperandSpec oQuantActFp8[] = {
430|    { "x",     RAD_OPD_IN,  0 },
431|    { "q",     RAD_OPD_OUT, 0 },
432|    { "scale", RAD_OPD_OUT, 0 },
433|};
434|static const RadParamSpec pGatedQuantFp8[] = {
435|    { "M",     RAD_P_INT, RAD_REQUIRED },
436|    { "n",     RAD_P_INT, RAD_REQUIRED },
437|    { "group", RAD_P_INT, RAD_REQUIRED },
438|    { "dtype", RAD_P_STR, RAD_REQUIRED },
439|    { "act",   RAD_P_STR, RAD_OPTIONAL },
440|};
441|static const RadOperandSpec oGatedQuantFp8[] = {
442|    { "gate_up",  RAD_OPD_IN,  0 },
443|    { "q",        RAD_OPD_OUT, 0 },
444|    { "scale",    RAD_OPD_OUT, 0 },
445|    { "out_bf16", RAD_OPD_OUT, 1 },
446|};
447|static const RadOperandSpec oTemp[] = {
448|    { "logits", RAD_OPD_INOUT, 0 },
449|    { "params", RAD_OPD_IN,    0 },
450|};
451|static const RadOperandSpec oMask[] = {
452|    { "logits",  RAD_OPD_INOUT, 0 },
453|    { "params",  RAD_OPD_IN,    0 },
454|    { "bitmask", RAD_OPD_IN,    0 },
455|};
456|static const RadOperandSpec oTopk[] = {
457|    { "logits",   RAD_OPD_IN,  0 },
458|    { "params",   RAD_OPD_IN,  0 },
459|    { "cand_idx", RAD_OPD_OUT, 0 },
460|    { "cand_val", RAD_OPD_OUT, 0 },
461|    { "pairs",    RAD_OPD_OUT, 1 },
462|};
463|static const RadOperandSpec oArgmax[] = {
464|    { "logits", RAD_OPD_IN,  0 },
465|    { "params", RAD_OPD_IN,  0 },
466|    { "token",  RAD_OPD_OUT, 0 },
467|};
468|static const RadOperandSpec oNarrow[] = {
469|    { "cand_idx", RAD_OPD_INOUT, 0 },
470|    { "cand_val", RAD_OPD_INOUT, 0 },
471|    { "params",   RAD_OPD_IN,    0 },
472|};
473|static const RadOperandSpec oPick[] = {
474|    { "cand_idx", RAD_OPD_IN,  0 },
475|    { "cand_val", RAD_OPD_IN,  0 },
476|    { "params",   RAD_OPD_IN,    0 },
477|    { "token",    RAD_OPD_OUT, 0 },
478|};
479|
480|/* silu_mul: pMnDtype is the same table gather_rows declares (verbatim from libr4d's, as
481| * is oGateUp) -- the loader's schema check compares the fixed grammar, not pointers. */
482|static const RadOperandSpec oGateUp[] = {
483|    { "gate_up", RAD_OPD_IN,  0 },
484|    { "y",       RAD_OPD_OUT, 0 },
485|};
486|/* quant_act_i8: params identical to quant_act_fp8's; the optional fourth output selects
487| * the _asum form (verbatim libr4d oQuantAct). */
488|static const RadOperandSpec oQuantAct[] = {
489|    { "x",     RAD_OPD_IN,  0 },
490|    { "q",     RAD_OPD_OUT, 0 },
491|    { "scale", RAD_OPD_OUT, 0 },
492|    { "asum",  RAD_OPD_OUT, 1 },
493|};
494|/* row_topk (verbatim libr4d pRowTopk / oRowTopk). */
495|static const RadParamSpec pRowTopk[] = {
496|    { "M",         RAD_P_INT, RAD_REQUIRED },
497|    { "N",         RAD_P_INT, RAD_REQUIRED },
498|    { "R",         RAD_P_INT, RAD_REQUIRED },
499|    { "vocab_off", RAD_P_INT, RAD_OPTIONAL },
500|    { "dtype",     RAD_P_STR, RAD_REQUIRED },
501|};
502|static const RadOperandSpec oRowTopk[] = {
503|    { "x",     RAD_OPD_IN,  0 },
504|    { "idx",   RAD_OPD_OUT, 0 },
505|    { "val",   RAD_OPD_OUT, 0 },
506|    { "pairs", RAD_OPD_OUT, 1 },
507|};
508|/* rmsnorm_quant_fp8 (verbatim libr4d pRmsNormQuantFp8 / oRmsNormQuantFp8). */
509|static const RadParamSpec pRmsNormQuantFp8[] = {
510|    { "M",     RAD_P_INT, RAD_REQUIRED },
511|    { "n",     RAD_P_INT, RAD_REQUIRED },
512|    { "eps",   RAD_P_F64, RAD_REQUIRED },
513|    { "group", RAD_P_INT, RAD_REQUIRED },
514|    { "dtype", RAD_P_STR, RAD_REQUIRED },
515|    { "wadd",  RAD_P_F64, RAD_OPTIONAL },
516|};
517|static const RadOperandSpec oRmsNormQuantFp8[] = {
518|    { "x",        RAD_OPD_IN,     0 },
519|    { "residual", RAD_OPD_INOUT,  1 },   /* present = the fused_add_rms_norm form */
520|    { "w",        RAD_OPD_WEIGHT, 0 },
521|    { "q",        RAD_OPD_OUT,    0 },
522|    { "scale",    RAD_OPD_OUT,    0 },
523|    { "out_bf16", RAD_OPD_OUT,    1 },   /* layers with a bf16 consumer of the post-norm tensor */
524|};
525|
526|static const RadOpSchema kSchemas[] = {
527|    { "rmsnorm", pRmsNorm, (int)(sizeof(pRmsNorm)/sizeof(pRmsNorm[0])),
528|      oRmsNorm,  (int)(sizeof(oRmsNorm)/sizeof(oRmsNorm[0])),
529|      "y = x * rsqrt(mean(x^2) + eps) * (w + wadd). Declared verbatim from libr4d so both "
530|      "plugins agree wherever they co-load." },
531|    { "embed_lookup", pEmbedM, (int)(sizeof(pEmbedM)/sizeof(pEmbedM[0])),
532|      oEmbed,  (int)(sizeof(oEmbed)/sizeof(oEmbed[0])),
533|      "the embedding gather. A negative token id writes a zero row, and so does one outside "
534|      "[vocab_offset, vocab_offset + n_vocab) -- which is what makes the table shardable "
535|      "across the vocabulary, the symmetry lm_head already had. Ids stay GLOBAL; summing the "
536|      "ranks reconstructs x, so the caller follows this with an exact all_reduce." },
537|    { "kv_store", pKvStore, (int)(sizeof(pKvStore)/sizeof(pKvStore[0])),
538|      oKvStore,  (int)(sizeof(oKvStore)/sizeof(oKvStore[0])),
539|      "scatter k and v into kv_cache[n_blocks, kv_heads, block_size, 2*head_dim] -- K at "
540|      "[.., d] and V at [.., head_dim + d]. A negative slot stores nothing." },
541|    { "act_mul", pActMul, (int)(sizeof(pActMul)/sizeof(pActMul[0])),
542|      oBinary, (int)(sizeof(oBinary)/sizeof(oBinary[0])),
543|      "y = act(a) * b, the activation (silu, gelu, gelu_tanh) rounded to y's dtype before "
544|      "the product: silu_mul with its two halves in two operands." },
545|    { "cast", pCast, (int)(sizeof(pCast)/sizeof(pCast[0])),
546|      oUnary, (int)(sizeof(oUnary)/sizeof(oUnary[0])),
547|      "y = x, converted. `from` and `to` are for the selector; the bytes are described by the "
548|      "tensors' own dtypes and those are what the kernel reads." },
549|    { "rope", pRope, (int)(sizeof(pRope)/sizeof(pRope[0])),
550|      oRope,  (int)(sizeof(oRope)/sizeof(oRope[0])),
551|      "rotary embedding in place over the Q and K heads of a fused projection; the V heads "
552|      "that follow are untouched. The multi-component modes (`mrope`, `imrope`, `axial`) "
553|      "take [C, M] component-major positions and `sections`; with [M] positions every "
554|      "component is that one, so mrope and imrope over a text batch are neox exactly." },
555|    { "gemm_nt", pGemmNt, (int)(sizeof(pGemmNt)/sizeof(pGemmNt[0])),
556|      oGemmNt, (int)(sizeof(oGemmNt)/sizeof(oGemmNt[0])),
557|      "Y[M,N] = A[M,K] @ B[N,K]^T. B is stored transposed (the layout transformer weights "
558|      "already have), so both operands are k-contiguous and the inner loop is a dot product "
559|      "over k." },
560|    { "logits_gemm", pLogits, (int)(sizeof(pLogits)/sizeof(pLogits[0])),
561|      oLogits, (int)(sizeof(oLogits)/sizeof(oLogits[0])),
562|      "the lm_head projection. gemm_nt's arithmetic under the vocabulary edge's parameter "
563|      "spelling; M is one row per SAMPLED position, `gather_rows` having run first." },
564|    { "gemm_nt_q", pGemmNtQ, (int)(sizeof(pGemmNtQ)/sizeof(pGemmNtQ[0])),
565|      oGemmNtQ, (int)(sizeof(oGemmNtQ)/sizeof(oGemmNtQ[0])),
566|      "C[M,N] = A[M,K] @ dequant(Bq)[N,K]^T. `dtype` names the operand COMBINATION, so one "
567|      "op covers w4a16 through mxfp4a8 and an architecture plugin that changes quant changes "
568|      "one string." },
569|    { "attn_paged", pAttnPaged, (int)(sizeof(pAttnPaged)/sizeof(pAttnPaged[0])),
570|      oAttnPaged, (int)(sizeof(oAttnPaged)/sizeof(oAttnPaged[0])),
571|      "Paged attention over kv_cache[num_blocks, kv_heads, block_size, 2*head_dim] (K then "
572|      "V per slot), f32 softmax with the max subtracted, causal and sliding-window masking, "
573|      "q_len >= 1 so one kernel serves decode, speculative verify and prefill. seqused is "
574|      "the TOTAL key count including this step's query rows. k_descale / v_descale are "
575|      "optional and apply to an fp8 cache. `scale` is not in docs/OPS.md and defaults to "
576|      "1/sqrt(head_dim). Declared verbatim from libr4d so both plugins agree wherever they "
577|      "co-load." },
578|
579|      /* ---- the sampling chain (classic stages only; see r3d_sample.hip's scope note). ---- */
580|      /* Param/operand tables verbatim from libr4d (pMVocab/pTopk/pMCand and the o* tables),
581|      * so the loader's schema-duplication check passes wherever both plugins co-load. */
582|      { "sample_temp", pMVocab, (int)(sizeof(pMVocab)/sizeof(pMVocab[0])),
583|       oTemp,   (int)(sizeof(oTemp)/sizeof(oTemp[0])),
584|       "per-request temperature. temp <= 0 leaves the row untouched; greedy is sample_argmax." },
585|      { "sample_mask", pMVocab, (int)(sizeof(pMVocab)/sizeof(pMVocab[0])),
586|       oMask,   (int)(sizeof(oMask)/sizeof(oMask[0])),
587|       "A SET BIT MEANS ALLOWED. Word w bit b covers token 32w + b; a cleared bit sets "
588|       "-infinity. The row's mask_row selects the plane row, or -1 when unconstrained." },
589|      { "sample_topk", pTopk, (int)(sizeof(pTopk)/sizeof(pTopk[0])),
590|       oTopk,   (int)(sizeof(oTopk)/sizeof(oTopk[0])),
591|       "the exact top-k candidate set, descending, ties to the lower token id. The width is "
592|       "the buffer's; a row's own top_k may be narrower and leaves holes." },
593|      { "sample_argmax", pMVocab, (int)(sizeof(pMVocab)/sizeof(pMVocab[0])),
594|       oArgmax, (int)(sizeof(oArgmax)/sizeof(oArgmax[0])),
595|       "greedy, ties to the lower token id." },
596|      { "sample_topp", pMCand, (int)(sizeof(pMCand)/sizeof(pMCand[0])),
597|       oNarrow, (int)(sizeof(oNarrow)/sizeof(oNarrow[0])),
598|       "nucleus: keep the shortest prefix whose cumulative probability reaches top_p, always "
599|       "at least one. The candidates are assumed descending, as sample_topk leaves them." },
600|      { "sample_minp", pMCand, (int)(sizeof(pMCand)/sizeof(pMCand[0])),
601|       oNarrow, (int)(sizeof(oNarrow)/sizeof(oNarrow[0])),
602|       "keep candidates whose probability is at least min_p times the most likely one's." },
603|      { "sample_pick", pMCand, (int)(sizeof(pMCand)/sizeof(pMCand[0])),
604|        oPick,   (int)(sizeof(oPick)/sizeof(oPick[0])),
605|        "the inverse-CDF draw over the candidate set, from the specified splitmix64 stream." },
606|      { "sample_penalties", pPenal, (int)(sizeof(pPenal)/sizeof(pPenal[0])),
607|        oPenalties, (int)(sizeof(oPenalties)/sizeof(oPenalties[0])),
608|        "repetition, frequency and presence penalties over the logits row, from the row's "
609|        "history window: distinct tokens only, the repetition divides above zero and "
610|        "multiplies at or below." },
611|      { "sample_dry", pPenal, (int)(sizeof(pPenal)/sizeof(pPenal[0])),
612|        oDry, (int)(sizeof(oDry)/sizeof(oDry[0])),
613|        "the DRY repetition-suffix penalty: the longest suffix of the window that repeats "
614|        "before each continuation, multiplier times base^(length - allowed), capped at the "
615|        "float overflow boundary. An optional breaker plane exempts positions." },
616|      { "sample_typical", pMCand, (int)(sizeof(pMCand)/sizeof(pMCand[0])),
617|        oNarrow, (int)(sizeof(oNarrow)/sizeof(oNarrow[0])),
618|        "locally typical narrowing: keep the candidates whose |surprisal - entropy| mass "
619|        "at-or-below their distance reaches typical_p; every candidate tied at the cut "
620|        "survives together." },
621|      { "sample_xtc", pMCand, (int)(sizeof(pMCand)/sizeof(pMCand[0])),
622|        oNarrow, (int)(sizeof(oNarrow)/sizeof(oNarrow[0])),
623|        "exclude-top-changes: with the row's own coin, drop every candidate at or above "
624|        "xtc_threshold except the last one; fires only when two or more are above." },
625|      { "sample_logprobs", pLogprobs, (int)(sizeof(pLogprobs)/sizeof(pLogprobs[0])),
626|        oLogprobs, (int)(sizeof(oLogprobs)/sizeof(oLogprobs[0])),
627|        "the raw distribution's summary: max, sum of exp, and the k largest (id, logit) "
628|        "pairs over the UNSOFTMAXED row; reads nothing the other stages wrote." },
629|
630|      /* ---- the band ops that gate the quantising bf16 trunk (see r3d_band.hip). ---- */
631|      { "add", pAdd, (int)(sizeof(pAdd)/sizeof(pAdd[0])),
632|        oBinary, (int)(sizeof(oBinary)/sizeof(oBinary[0])),
633|        "y = a + b. `b` carrying exactly n elements against an output of M rows broadcasts "
634|        "along rows, and nothing else broadcasts at all. With `div`, y = rd(rd(a + b) / div), "
635|        "rd y's dtype." },
636|      { "mul", pMnDtype, (int)(sizeof(pMnDtype)/sizeof(pMnDtype[0])),
637|        oBinary, (int)(sizeof(oBinary)/sizeof(oBinary[0])),
638|        "y = a * b, with the same one broadcast rule as `add`." },
639|      { "sigmoid", pMnDtype, (int)(sizeof(pMnDtype)/sizeof(pMnDtype[0])),
640|        oUnary, (int)(sizeof(oUnary)/sizeof(oUnary[0])),
641|        "y = 1 / (1 + exp(-x))." },
642|      { "rope_table", pRopeTable, (int)(sizeof(pRopeTable)/sizeof(pRopeTable[0])),
643|        oRopeTable, (int)(sizeof(oRopeTable)/sizeof(oRopeTable[0])),
644|        "the cos/sin plane a fused prologue reads instead of theta: cos then sin of "
645|        "`rope`'s ladder at each position's own row of a [max_ctx, rot] table; a negative "
646|        "position writes nothing." },
647|      { "gdn_gated_rmsnorm", pGatedNorm, (int)(sizeof(pGatedNorm)/sizeof(pGatedNorm[0])),
648|        oGatedNorm, (int)(sizeof(oGatedNorm)/sizeof(oGatedNorm[0])),
649|        "o = x * rsqrt(mean_channels(x^2) + eps) * w * act(z). `act` is \"silu\" or "
650|        "\"sigmoid\"." },
651|      { "gdn_conv_update", pConvUpdate, (int)(sizeof(pConvUpdate)/sizeof(pConvUpdate[0])),
652|        oConvUpdate, (int)(sizeof(oConvUpdate)/sizeof(oConvUpdate[0])),
653|        "The same convolution for a decode step against the rolling window: reads at "
654|        "num_accepted-1 and rewrites the buffer shifted down by one, so a speculative "
655|        "rejection is a change of read offset rather than a recompute. `num_accepted` is "
656|        "REQUIRED, because an op that lets the caller omit it is an op whose rollback "
657|        "contract is optional (spec \u00a710); a non-speculative decode passes ones." },
658|      { "gdn_kkt_solve", pKktSolve, (int)(sizeof(pKktSolve)/sizeof(pKktSolve[0])),
659|        oKktSolve, (int)(sizeof(oKktSolve)/sizeof(oKktSolve[0])),
660|        "A = (I + strict_lower(diag(beta) K K^T e^dg))^-1 per chunk." },
661|      { "gdn_recurrent_update", pRecurrent, (int)(sizeof(pRecurrent)/sizeof(pRecurrent[0])),
662|        oRecurrent, (int)(sizeof(oRecurrent)/sizeof(oRecurrent[0])),
663|        "The decode recurrence against the paged state cache, per-candidate form, gated RMS "
664|        "norm folded. Verbatim libref/libr4d so co-loaded schema_same() passes." },
665|      { "gather_rows", pMnDtype, (int)(sizeof(pMnDtype)/sizeof(pMnDtype[0])),
666|        oGather, (int)(sizeof(oGather)/sizeof(oGather[0])),
667|        "y[i, :] = x[idx[i], :]. A negative index writes a zero row -- the padding convention a "
668|        "step with fewer sampled rows than the buffer's extent leaves behind." },
669|      { "gated_quant_fp8", pGatedQuantFp8, (int)(sizeof(pGatedQuantFp8)/sizeof(pGatedQuantFp8[0])),
670|        oGatedQuantFp8, (int)(sizeof(oGatedQuantFp8)/sizeof(oGatedQuantFp8[0])),
671|        "silu(a)*b or sigmoid(a)*b, then the same quantiser, in one pass." },
672|      { "quant_act_fp8", pQuantActFp8, (int)(sizeof(pQuantActFp8)/sizeof(pQuantActFp8[0])),
673|        oQuantActFp8, (int)(sizeof(oQuantActFp8)/sizeof(oQuantActFp8[0])),
674|        "per-(row, `group`-column) symmetric E4M3 quantisation of a token-major activation, in "
675|        "PLAIN ROW-MAJOR order rather than quant_act_i8's fragment order. The scale is "
676|        "amax/448 -- a DEQUANT multiplier, the same sense as the checkpoint's "
677|        "weight_scale_inv, so the fp8 GEMMs multiply both scales and never divide." },
678|      { "silu_mul", pMnDtype, (int)(sizeof(pMnDtype)/sizeof(pMnDtype[0])),
679|        oGateUp, (int)(sizeof(oGateUp)/sizeof(oGateUp[0])),
680|        "y[.., i] = silu(gate_up[.., i]) * gate_up[.., n + i]. The GATE IS THE FIRST HALF, "
681|        "which is the order rad-convert fuses gate_proj and up_proj in." },
682|      { "quant_act_i8", pQuantActFp8, (int)(sizeof(pQuantActFp8)/sizeof(pQuantActFp8[0])),
683|        oQuantAct, (int)(sizeof(oQuantAct)/sizeof(oQuantAct[0])),
684|        "per-row symmetric int8 quantisation, written in the byte order the r4d GEMMs' A "
685|        "fragment reads. A third output selects the form that also emits the per-group row "
686|        "sums an asymmetric weight grid needs." },
687|      { "row_topk", pRowTopk, (int)(sizeof(pRowTopk)/sizeof(pRowTopk[0])),
688|        oRowTopk, (int)(sizeof(oRowTopk)/sizeof(oRowTopk[0])),
689|        "exact per-row top-R, descending, ties to the lower column. `vocab_off` shifts every "
690|        "column reported so a rank's slice reports global ids, and `pairs` is the [M, R, 2] "
691|        "(int32 id, float value) plane an all_gather moves; a slot the row could not fill "
692|        "reports id -1 there and column 0 in `idx`, which is the unsharded contract unchanged." },
693|      { "rmsnorm_quant_fp8", pRmsNormQuantFp8,
694|        (int)(sizeof(pRmsNormQuantFp8)/sizeof(pRmsNormQuantFp8[0])),
695|        oRmsNormQuantFp8, (int)(sizeof(oRmsNormQuantFp8)/sizeof(oRmsNormQuantFp8[0])),
696|        "RMS norm and the fp8 activation quantiser in one pass over the row. The bf16 norm "
697|        "output the unfused pair wrote is still written, and the codes are taken over it." },
698|      };
699|
700|/* The rmsnorm row. Field order is RadKernelInfo's positional form (rad_abi.h): after the prose,
701| * constraints and (absent) tunables come init, fini, launch, scratch, layout, relayout, and
702| * opd_shape -- null because a cold-calling tool takes operand shapes from libref's for the op.
703| * `replaces` stays zero-initialised: this row is the unfused op, not a fusion. */
704|
705|static const RadConstraint cBf16Row[] = {
706|    { "dtype", RAD_C_IN, 0, "bf16" },
707|};
708|
709|/* logits_gemm bf16: verbatim libr4d cLogitsGemm = { C_DIV("n_embd", 8), C_IN("dtype", "bf16") }. */
710|static const RadConstraint cLogitsGemm[] = {
711|    { "n_embd", RAD_C_DIV, 8,    nullptr },
712|    { "dtype",  RAD_C_IN,  0,    "bf16" },
713|};
714|
715|/* gemm_nt_q mxfp4a8: the format gate IS the dtype string (every operand dtypes follow
716| * from it; the entry re-checks them). Other gemm_nt_q encodings are refused inside the
717| * entry with RAD_E_UNSUPPORTED, so this constraint only keeps the selector honest. */
718|static const RadConstraint cMxfp4Row[] = {
719|    { "dtype", RAD_C_IN, 0, "mxfp4a8" },
720|};
721|
722|/* kv_store constraints, the positional form of libr4d's cKvStore = { C_GE("block_size", 1),
723| * C_IN("kv_dtype", "bf16") }. */
724|static const RadConstraint cKvStore[] = {
725|    { "block_size", RAD_C_GE, 1, nullptr },
726|    { "kv_dtype",   RAD_C_IN, 0, "bf16" },
727|};
728|
729|/* libr3d is the only device library on RDNA3, so no in-plugin arbitration; the value only
730| * needs to sit above a future fallback row. */
731|enum { R3D_PRIO_ONLY = 1000 };
732|enum { R3D_PRIO_ATTN_DECODE = 1100 };   /* split-KV rows win the decode band from the
733|                                           any-length tile; nothing else is ranked */
734|
735|/* act_mul: the bf16 gated MLP, all four oracle activations this kernel ports. */
736|static const RadConstraint cActMul[] = {
737|    { "dtype", RAD_C_IN, 0, "bf16" },
738|    { "act",   RAD_C_IN, 0, "silu gelu gelu_tanh sigmoid" },
739|};
740|
741|/* The cast pairs this plugin serves, one row each (libr4d spells the same four; the f16
742| * narrowing is not yet ported and its pair simply has no row here). */
743|static const RadConstraint cCastBf16[] = {
744|    { "from", RAD_C_IN, 0, "bf16" }, { "to", RAD_C_IN, 0, "bf16" },
745|};
746|static const RadConstraint cCastWiden[] = {
747|    { "from", RAD_C_IN, 0, "bf16" }, { "to", RAD_C_IN, 0, "f32" },
748|};
749|static const RadConstraint cCastNarrow[] = {
750|    { "from", RAD_C_IN, 0, "f32" }, { "to", RAD_C_IN, 0, "bf16" },
751|};
752|
753|/* attn_paged: the row admits what the kernel serves. head_dim is a thread count here, so
754| * the entry point refuses > 1024; the constraint stays open above that and a query for a
755| * bigger head resolves past the row to nothing -- an honest miss. f32 q is served by the
756| * kernel but not advertised: the model path is bf16, and a constraint the fixtures never
757| * probe is a constraint never checked. */
758|static const RadConstraint cAttnPaged[] = {
759|    { "head_dim", RAD_C_GE, 1, nullptr },
760|    { "q_dtype",  RAD_C_IN, 0, "bf16" },
761|    { "kv_dtype", RAD_C_IN, 0, "bf16 f32 fp8_e4m3" },
762|};
763|
764|/* Split-KV decode: the tuning axis is the split count -- 0 is the kernel's own
765| * parallelism law (slices cut to fill the CUs), 8..64 measured overrides. 0 is
766| * launchable by construction (the law's own floor is 1, and splits=1 runs the plain
767| * tile through the same entry), which is all the loader requires of a default. */
768|static const int64_t kSplits[] = { 0, 8, 16, 32, 64 };
769|static const RadTunable tAttnSplit = { "splits", kSplits, 5, 0,
770|    "KV slices per (head, seq) computed in parallel; 0 = kernel law (fill the CUs)" };
771|
772|/* The decode band these rows take from attn_paged_bf16: q_len <= 8 (decode and
773| * speculative verify -- libr4d's decode-band bound, same spelling), window 0 (the row
774| * serves a full KV walk; windowed decode stays with the tile), bf16 q, bf16/fp8-e4m3
775| * cache. head_dim is one row per value, libr4d's convention: each is exactly a WMMA
776| * template instantiation, and no IN set exists for integer keys. */
777|static const RadConstraint cAttnDecodeH128[] = {
778|    RAD_CLE("q_len", 8),
779|    RAD_CEQ("window", 0),
780|    RAD_CEQ("head_dim", 128),
781|    RAD_CIN("q_dtype", "bf16"),
782|    RAD_CIN("kv_dtype", "bf16 fp8_e4m3"),
783|};
784|static const RadConstraint cAttnDecodeH64[] = {
785|    RAD_CLE("q_len", 8),
786|    RAD_CEQ("window", 0),
787|    RAD_CEQ("head_dim", 64),
788|    RAD_CIN("q_dtype", "bf16"),
789|    RAD_CIN("kv_dtype", "bf16 fp8_e4m3"),
790|};
791|
792|/* rope: neox and gptj over [M] positions, plus imrope over [3, M] component-major positions
793| * (the 27B trunk's attention) -- a subset of libr4d's five modes. A query for mrope/axial
794| * resolves to no device row on this card, which is an honest failure at issue, not an
795| * approximation. */
796|static const RadConstraint cRopeRow[] = {
797|    { "head_dim", RAD_C_GE, 1, nullptr },
798|    { "mode",     RAD_C_IN, 0, "neox gptj imrope" },
799|};
800|
801|/* Sampler row constraints, mirroring libr4d's cSampleVoc / cSampleCand. */
802|static const RadConstraint cSampleVoc[]  = { { "n_vocab", RAD_C_GE, 1, nullptr } };
803|static const RadConstraint cSampleCand[] = { { "n_cand", RAD_C_GE, 1, nullptr },
804|                                             { "n_cand", RAD_C_LE, 4096, nullptr } };
805|
806|/* Band-op constraints: the widths the kernels stride, and the activation dtypes
807| * r3d_band.hip actually reads (f32/bf16) -- an f16 band resolves to the reference
808| * instead of being refused at launch. */
809|static const RadConstraint cBandMn[] = {
810|    { "n",     RAD_C_GE, 1, nullptr },
811|    { "dtype", RAD_C_IN, 0, "f32 bf16" },
812|};
813|/* rope_table: the kernel's ladder and refusal cover every rot >= 2 even (odd refused at
814| * launch, >1024 by the shared-array bound); yarn and the `head` ladder are refused there,
815| * so nothing else is constrained here. */
816|static const RadConstraint cRopeTable[] = {
817|    { "rot", RAD_C_GE, 2, nullptr },
818|};
819|static const RadConstraint cGatedNorm[] = {
820|    { "channels", RAD_C_GE, 1, nullptr },
821|    { "act",      RAD_C_IN, 0, "silu sigmoid" },
822|};
823|static const RadConstraint cConvUpdate[] = {
824|    { "head_k",     RAD_C_GE, 1, nullptr },
825|    { "head_v",     RAD_C_GE, 1, nullptr },
826|    { "conv_width", RAD_C_GE, 1, nullptr },
827|};
828|/* kkt_solve: chunk 64 is the kernel's LDS bound (libr4d pins the same value; the derived
829| * parameter resolves from A's shape). head_k stays open -- the dot chain walks any K. */
830|static const RadConstraint cKktSolve[] = {
831|    { "head_k", RAD_C_GE, 1, nullptr },
832|    RAD_CEQ("chunk", 64),
833|};
834|/* chunk_scan: head_k 128 is the register bound (the state column is a thread's register
835| * array), head_v <= 128 the one-column-per-thread block; chunk 64 is the shared c64 value. */
836|static const RadConstraint cChunkScan[] = {
837|    RAD_CEQ("head_k", 128),
838|    { "head_v", RAD_C_GE, 1,    nullptr },
839|    { "head_v", RAD_C_LE, 128,  nullptr },
840|    RAD_CEQ("chunk", 64),
841|};
842|/* recurrent_update: the same register bounds as chunk_scan (r4d declares 128/128 outright;
843| * the <= spelling keeps head_v under the one-column-per-thread block honest). The state_form
844| * constraint is the scope gate: per_candidate is the form this kernel implements and the form
845| * a non-speculative batch issues, and a constraint on a key the query lacks does not hold --
846| * so anchor batches (speculation) and unnamed callers stay with libr4d's anchor kernel. */
847|static const RadConstraint cRecurrent[] = {
848|    RAD_CEQ("head_k", 128),
849|    { "head_v", RAD_C_GE, 1,    nullptr },
850|    { "head_v", RAD_C_LE, 128,  nullptr },
851|    RAD_CIN("state_form", "per_candidate"),
852|};
853|
854|extern "C" int r3d_rad_rmsnorm(const RadArgs*, RadStream);
855|extern "C" int r3d_rad_embed_lookup(const RadArgs*, RadStream);
856|extern "C" int r3d_rad_kv_store(const RadArgs*, RadStream);
857|extern "C" int r3d_rad_act_mul(const RadArgs*, RadStream);
858|extern "C" int r3d_rad_cast(const RadArgs*, RadStream);
859|extern "C" int r3d_rad_rope(const RadArgs*, RadStream);
860|extern "C" int r3d_rad_rope_table(const RadArgs*, RadStream);
861|extern "C" int r3d_rad_gemm_nt(const RadArgs*, RadStream);
862|extern "C" int r3d_rad_logits_gemm(const RadArgs*, RadStream);
863|extern "C" int r3d_rad_gemm_nt_q(const RadArgs*, RadStream);
864|extern "C" int r3d_rad_logits_gemm_e4m3(const RadArgs*, RadStream);
865|extern "C" int r3d_layout_head(const RadParam*, int, int, const RadEncoding*, const int*,
866|                               const RadTensor*, int, RadLayout*);
867|extern "C" int r3d_relayout_head(const RadParam*, int, int, const RadEncoding*, const int*,
868|                                 const RadTensor*, int, void*, int64_t);
869|extern "C" int r3d_unrelayout_head(const RadParam*, int, int, const RadEncoding*, const int*,
870|                                   const void*, int64_t, const RadTensor*, int);
871|extern "C" int r3d_rad_attn_paged(const RadArgs*, RadStream);
872|extern "C" int r3d_rad_attn_paged_decode(const RadArgs*, RadStream);
873|extern "C" int64_t r3d_rad_attn_decode_scratch(const RadArgs*);
874|extern "C" int r3d_rad_sample_temp(const RadArgs*, RadStream);
875|extern "C" int r3d_rad_sample_mask(const RadArgs*, RadStream);
876|extern "C" int r3d_rad_sample_argmax(const RadArgs*, RadStream);
877|extern "C" int r3d_rad_sample_topk(const RadArgs*, RadStream);
878|extern "C" int r3d_rad_sample_topp(const RadArgs*, RadStream);
879|extern "C" int r3d_rad_sample_minp(const RadArgs*, RadStream);
880|extern "C" int r3d_rad_sample_pick(const RadArgs*, RadStream);
881|extern "C" int r3d_rad_sample_penalties(const RadArgs*, RadStream);
882|extern "C" int r3d_rad_sample_dry(const RadArgs*, RadStream);
883|extern "C" int r3d_rad_sample_typical(const RadArgs*, RadStream);
884|extern "C" int r3d_rad_sample_xtc(const RadArgs*, RadStream);
885|extern "C" int r3d_rad_sample_logprobs(const RadArgs*, RadStream);
886|extern "C" int r3d_rad_quant_act_fp8(const RadArgs*, RadStream);
887|extern "C" int r3d_rad_gated_quant_fp8(const RadArgs*, RadStream);
888|extern "C" int r3d_rad_add(const RadArgs*, RadStream);
889|extern "C" int r3d_rad_mul(const RadArgs*, RadStream);
890|extern "C" int r3d_rad_sigmoid(const RadArgs*, RadStream);
891|extern "C" int r3d_rad_gather_rows(const RadArgs*, RadStream);
892|extern "C" int r3d_rad_gdn_gated_rmsnorm(const RadArgs*, RadStream);
893|extern "C" int r3d_rad_gdn_conv_update(const RadArgs*, RadStream);
894|extern "C" int r3d_rad_gdn_kkt_solve(const RadArgs*, RadStream);
895|extern "C" int r3d_rad_gdn_chunk_scan(const RadArgs*, RadStream);
896|extern "C" int r3d_rad_gdn_recurrent_update(const RadArgs*, RadStream);
897|extern "C" int r3d_rad_silu_mul(const RadArgs*, RadStream);
898|extern "C" int r3d_rad_quant_act_i8(const RadArgs*, RadStream);
899|extern "C" int r3d_rad_row_topk(const RadArgs*, RadStream);
900|extern "C" int r3d_rad_rmsnorm_quant_fp8(const RadArgs*, RadStream);
901|
902|/* Constraints for the r3d_more rows, in libr4d's shape so a query that resolves there
903| * resolves here identically: silu_mul is the plain bf16 row; quant_act_i8 walks the
904| * sixteen-byte WMMA step so `n DIV 16`; row_topk admits N a multiple of 8 and R <= 32. */
905|static const RadConstraint cQuantActI8[] = {
906|    { "n",     RAD_C_DIV, 16,   nullptr },
907|    { "dtype", RAD_C_IN,  0,    "bf16" },
908|};
909|static const RadConstraint cRowTopK[] = {
910|    { "N",     RAD_C_DIV, 8,    nullptr },
911|    { "M",     RAD_C_GE,  1,    nullptr },
912|    { "R",     RAD_C_GE,  1,    nullptr },
913|    { "R",     RAD_C_LE,  32,   nullptr },
914|    { "dtype", RAD_C_IN,  0,    "bf16" },
915|};
916|static const RadConstraint cRmsNormQuantFp8[] = {
917|    { "n",     RAD_C_DIV, 4,    nullptr },
918|    { "group", RAD_C_EQ,  128,  nullptr },   /* R4D_FP8_BLOCK */
919|    { "dtype", RAD_C_IN,  0,    "bf16" },
920|};
921|
922|static const RadKernelInfo kRows[] = {
923|
924|{ "rmsnorm_bf16", "rmsnorm", "norm",
925|  "root-mean-square norm with a gain, two passes over the row",
926|  "any n; the sum of squares accumulates in f32 and only the store narrows",
927|  "bf16 x and y, bf16 or f32 gain, fp32 reduction",
928|  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
929|  cBf16Row, 1, nullptr, 0, nullptr,
930|  nullptr, nullptr, r3d_rad_rmsnorm },
931|
932|{ "embed_lookup_bf16", "embed_lookup", "vocab",
933|  "embedding gather, one workgroup a token",
934|  "any n_embd and n_vocab; a negative or out-of-range id writes a zero row",
935|  "i32 tokens, bf16 table and output",
936|  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
937|  cBf16Row, 1, nullptr, 0, nullptr,
938|  nullptr, nullptr, r3d_rad_embed_lookup },
939|
940|{ "kv_store_bf16", "kv_store", "attn",
941|  "scatter k and v into the paged cache, one workgroup a token",
942|  "cache exactly [n_blocks, kv_heads, block_size, 2*head_dim]; a negative or out-of-range slot "
943|  "stores nothing",
944|  "bf16 k / v / cache, i32 slot mapping",
945|  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
946|  cKvStore, 2, nullptr, 0, nullptr,
947|  nullptr, nullptr, r3d_rad_kv_store },
948|
949|{ "act_mul_bf16", "act_mul", "elem",
950|  "gated activation over two operands: act(a) rounded, times b",
951|  "any n; a, b and y each at their own row pitch, so both may be column slices of one buffer",
952|  "bf16 in and out, f32 arithmetic, double-width silu / erf / tanh like the oracle",
953|  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
954|  cActMul, 2, nullptr, 0, nullptr,
955|  nullptr, nullptr, r3d_rad_act_mul },
956|
957|{ "cast_bf16_bf16", "cast", "elem",
958|  "y = x, bf16 to bf16 -- a strided row copy",
959|  "any M and n; both row pitches are read off the operands, so a column slice of a wider "
960|  "buffer is expressible",
961|  "bf16 in, bf16 out, byte-for-byte",
962|  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
963|  cCastBf16, 2, nullptr, 0, nullptr,
964|  nullptr, nullptr, r3d_rad_cast },
965|
966|{ "cast_bf16_f32", "cast", "elem",
967|  "y = x, bf16 widened to f32",
968|  "any M and n; both row pitches are read off the operands",
969|  "bf16 in, f32 out, exact",
970|  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
971|  cCastWiden, 2, nullptr, 0, nullptr,
972|  nullptr, nullptr, r3d_rad_cast },
973|
974|{ "cast_f32_bf16", "cast", "elem",
975|  "y = x, f32 narrowed to bf16",
976|  "any M and n; both row pitches are read off the operands",
977|  "f32 in, bf16 out, rounded to nearest even",
978|  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
979|  cCastNarrow, 2, nullptr, 0, nullptr,
980|  nullptr, nullptr, r3d_rad_cast },
981|
982|{ "rope_bf16", "rope", "attn",
983|  "rotary embedding in place, one thread a rotated pair, the ladder built once per token",
984|  "any head_dim; rotary_dim even and no wider than head_dim; neox and gptj pairings over "
985|  "[M] positions (the multi-component modes and yarn/llama3 scaling have no row here)",
986|  "bf16 qkv, i32 positions, fp32 angles through the accurate sincosf",
987|  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
988|  cRopeRow, 2, nullptr, 0, nullptr,
989|  nullptr, nullptr, r3d_rad_rope },
990|
991|{ "rope_table_f32", "rope_table", "attn",
992|  "cos/sin of the rope ladder per position row, one block per position, the ladder "
993|  "built once in shared per row",
994|  "rot even in [2, 1024]; [max_ctx, rot] table with unit last stride; a position "
995|  "outside the table writes its row nothing (ref skips it the same); yarn and the "
996|  "`head` ladder have no row here",
997|  "i32/i64 positions, f32 or bf16 table, fp32 angles through the accurate sincosf",
998|  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
999|  cRopeTable, 1, nullptr, 0, nullptr,
1000|  nullptr, nullptr, r3d_rad_rope_table },
1001|
1002|{ "gemm_nt_bf16", "gemm_nt", "gemm",
1003|  "dense bf16 GEMM, 64x64 output tile per workgroup, K in 16-wide LDS slices",
1004|  "any M, N and K (to 2^27); ragged tiles load as zero and are never stored; operand row "
1005|  "pitches follow the trailing-dimension fold, so a rank-3 qkv output is writable",
1006|  "bf16 A and B, bf16 or f32 Y, f32 accumulation",
1007|  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
1008|  cBf16Row, 1, nullptr, 0, nullptr,
1009|  nullptr, nullptr, r3d_rad_gemm_nt },
1010|
1011|/* logits_gemm: the e4m3 head with one float scale per 128 columns of K (the
1021| * layout hook admits exactly that joined form; the launcher pins the scale
1022| * grid's row block to 1 or 128). Ragged K without a four-byte row walk, and
1023| * int8 heads, are refused -- the host libraries keep those. */
1024|static const RadConstraint cLogitsRow[] = {
1025|    { "n_vocab", RAD_C_GE, 1,   nullptr },
1026|    { "n_embd",  RAD_C_GE, 128, nullptr },
1027|    { "dtype",   RAD_C_IN, 0, "bf16" },
1028|};
1029|
1030|{ "logits_gemm_e4m3", "logits_gemm", "vocab",
1031|  "bandwidth GEMV over the joined lm_head: warp per head row, four codes per "
1032|  "lane per step, x staged per 128-column block through LDS; the decoded e4m3 "
1033|  "times the block scale is one fma chain per output in f32, and the wavefront "
1034|  "folds its 32 partials through a butterfly",
1035|  "codes [n_vocab, n_embd] e4m3 dense, scale per (1 or 128 rows, 128 cols); M "
1036|  "derives from numel(y)/n_vocab, groups of up to 8 rows ride grid.y",
1037|  "bf16 x, joined e4m3+bf16/f16/f32 head, f32 logits",
1038|  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
1039|  cLogitsRow, 3, nullptr, 0, nullptr,
1040|  nullptr, nullptr, r3d_rad_logits_gemm_e4m3,
1041|  nullptr, r3d_layout_head, r3d_relayout_head, nullptr, r3d_unrelayout_head },
1042|
1043|{ "logits_gemm_bf16", "logits_gemm", "vocab",
1012|  "the lm_head projection: same tile as gemm_nt_bf16, the vocabulary edge's parameter "
1013|  "spelling; logits go out F32",
1014|  "any M and any vocabulary; n_embd a multiple of 8",
1015|  "bf16 x and lm_head, F32 LOGITS OUT, fp32 accumulate",
1016|  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
1017|  cLogitsGemm, 2, nullptr, 0, nullptr,
1018|  nullptr, nullptr, r3d_rad_logits_gemm },
1019|
1020|{ "gemm_nt_q_mxfp4", "gemm_nt_q", "gemm",
1044|  "mxfp4a8 WMMA tile: e2m1 nibbles decoded and E8M0-folded into bf16 B fragments per "
1045|  "32-K group, fp8_e4m3 activations decoded exactly, fp32 accumulators flushed and "
1046|  "row-scaled by a_scale per 128-K group through LDS",
1047|  "b:32 / a:128 scale grids only; nibble rows byte-packed; ragged tiles load as zero "
1048|  "and are never stored; M derives from numel(y)/N",
1049|  "fp8_e4m3 a with f32 a_scale, e2m1 b with E8M0 b_scale, bf16 y; b_ref accepted and "
1050|  "ignored (both host libraries do); a_sum refused",
1051|  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
1052|  cMxfp4Row, 1, nullptr, 0, nullptr,
1053|  nullptr, nullptr, r3d_rad_gemm_nt_q },
1054|
1055|{ "attn_paged_bf16", "attn_paged", "attn",
1056|  "paged causal / sliding-window attention, one workgroup per (query row, head), any "
1057|  "query length, varlen, split-free (no scratch)",
1058|  "any head_dim to 1024, any gqa, any block size, window and causal masking, ragged "
1059|  "batches via cu_seqlens, unmapped (negative) block table entries skipped",
1060|  "bf16 query, bf16 / f32 / fp8-e4m3 kv cache with optional per-(seq,head) descales, "
1061|  "bf16 or f32 out, f32 softmax with the max subtracted",
1062|  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
1063|  cAttnPaged, 3, nullptr, 0, nullptr,
1064|  nullptr, nullptr, r3d_rad_attn_paged },
1065|
1066|{ "attn_paged_decode_bf16kv_h128", "attn_paged", "attn",
1067|  "split-KV decode attention: the WMMA tile's two passes over each of `splits` KV slices "
1068|  "in parallel, un-normalized partials (O, m, l) in the engine arena, one merge kernel "
1069|  "rescaling them per output row; splits=1 IS the plain tile, same body, same tail math",
1070|  "decode and speculative verify (q_len <= 8), window 0, head_dim 128, any gqa, any "
1071|  "block size, ragged batches via cu_seqlens, unmapped block table entries skipped",
1072|  "bf16 query, bf16 / fp8-e4m3 kv cache with optional per-(seq,head) descales, bf16 or "
1073|  "f32 out, f32 partials and merge",
1074|  RAD_DOMAIN_DEVICE, R3D_PRIO_ATTN_DECODE,
1075|  cAttnDecodeH128, 5, &tAttnSplit, 1, nullptr,
1076|  nullptr, nullptr, r3d_rad_attn_paged_decode, r3d_rad_attn_decode_scratch },
1077|
1078|{ "attn_paged_decode_bf16kv_h64", "attn_paged", "attn",
1079|  "split-KV decode attention, head_dim 64: as attn_paged_decode_bf16kv_h128",
1080|  "decode and speculative verify (q_len <= 8), window 0, head_dim 64, any gqa, any "
1081|  "block size, ragged batches via cu_seqlens, unmapped block table entries skipped",
1082|  "bf16 query, bf16 / fp8-e4m3 kv cache with optional per-(seq,head) descales, bf16 or "
1083|  "f32 out, f32 partials and merge",
1084|  RAD_DOMAIN_DEVICE, R3D_PRIO_ATTN_DECODE,
1085|  cAttnDecodeH64, 5, &tAttnSplit, 1, nullptr,
1086|  nullptr, nullptr, r3d_rad_attn_paged_decode, r3d_rad_attn_decode_scratch },
1087|
1088|/* ---- the sampling chain. The candidate-width cap on the cand-plane rows mirrors
1089| * libr4d's: the engine's declare-time bound is the buffer width, and 4096 covers it.
1090| * sample_topk here is k parallel argmax rounds under the (v desc, id asc) order --
1091| * exact, deterministic, no radix sort; the vocab widths the fixtures record make the
1092| * re-reads L2-resident, and a correctness-first pass is the point of this library. */
1093|
1094|{ "sample_temp_f32", "sample_temp", "sample",
1095|  "per-request temperature over the logits row",
1096|  "any vocabulary; a row with temp <= 0 or exactly 1 is skipped whole",
1097|  "f32 logits in place, per-row params",
1098|  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
1099|  cSampleVoc, 1, nullptr, 0, nullptr,
1100|  nullptr, nullptr, r3d_rad_sample_temp },
1101|
1102|{ "sample_mask_f32", "sample_mask", "sample",
1103|  "the grammar bitmask -- a cleared bit sets -infinity",
1104|  "any vocabulary; the plane is [rows, ceil(n_vocab/32)], addressed by the row's mask_row",
1105|  "f32 logits in place, u32 bitmask, per-row params",
1106|  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
1107|  cSampleVoc, 1, nullptr, 0, nullptr,
1108|  nullptr, nullptr, r3d_rad_sample_mask },
1109|
1110|{ "sample_argmax_f32", "sample_argmax", "sample",
1111|  "greedy, ties to the lower token id",
1112|  "any vocabulary; the block reduction uses the same (value desc, id asc) order as top-k, "
1113|  "so greedy and top-1 agree",
1114|  "f32 logits, i32 token",
1115|  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
1116|  cSampleVoc, 1, nullptr, 0, nullptr,
1117|  nullptr, nullptr, r3d_rad_sample_argmax },
1118|
1119|{ "sample_topk_f32", "sample_topk", "sample",
1120|  "the exact top-k candidate set, descending, ties to the lower token id",
1121|  "any vocabulary; n_cand <= 4096; k rounds of block argmax under the (value desc, id asc) "
1122|  "order reproduce the reference selection exactly, including the tie window after the "
1123|  "previous pick; the optional vocab-parallel pairs plane is packed when present",
1124|  "f32 logits, i32 candidate ids, f32 candidate LOGITS (not probabilities)",
1125|  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
1126|  cSampleVoc, 1, nullptr, 0, nullptr,
1127|  nullptr, nullptr, r3d_rad_sample_topk },
1128|
1129|{ "sample_topp_f32", "sample_topp", "sample",
1130|  "nucleus narrowing in place",
1131|  "n_cand <= 4096; the prefix sum is SEQUENTIAL in f32 in candidate order -- a parallel "
1132|  "prefix sums in a different order and a candidate at the cut would survive in one and "
1133|  "not the other",
1134|  "i32 candidate ids and f32 candidate logits, both in place",
1135|  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
1136|  cSampleCand, 2, nullptr, 0, nullptr,
1137|  nullptr, nullptr, r3d_rad_sample_topp },
1138|
1139|{ "sample_minp_f32", "sample_minp", "sample",
1140|  "min-p narrowing in place",
1141|  "n_cand <= 4096; exp ratios against the max in candidate order, the max always survives",
1142|  "i32 candidate ids and f32 candidate logits, both in place",
1143|  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
1144|  cSampleCand, 2, nullptr, 0, nullptr,
1145|  nullptr, nullptr, r3d_rad_sample_minp },
1146|
1147|{ "sample_pick_f32", "sample_pick", "sample",
1148|  "the inverse-CDF draw over the candidate set, from the spec splitmix64 stream",
1149|  "n_cand <= 4096; one thread per row accumulates in candidate order so the draw "
1150|  "reproduces the reference token for token; a fall-through takes the last candidate "
1151|  "ABOVE -infinity, never a masked tail entry",
1152|  "i32 candidate ids, f32 candidate logits, i32 token out",
1153|  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
1154|  cSampleCand, 2, nullptr, 0, nullptr,
1155|  nullptr, nullptr, r3d_rad_sample_pick },
1156|
1157|{ "sample_penalties_f32", "sample_penalties", "sample",
1158|  "repetition / frequency / presence penalties from the row's history window",
1159|  "any vocabulary; the walk is per DISTINCT token of the window, so the cost is the "
1160|  "window, not the vocabulary; the i32 history plane must be contiguous",
1161|  "f32 logits in place, i32 history plane, per-row params",
1162|  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
1163|  cSampleVoc, 1, nullptr, 0, nullptr,
1164|  nullptr, nullptr, r3d_rad_sample_penalties },
1165|
1166|{ "sample_dry_f32", "sample_dry", "sample",
1167|  "the DRY repetition-suffix penalty, the naive longest-suffix scan of the oracle",
1168|  "the window is bounded by what fits LDS (hmax <= 6144); the penalty is the "
1169|  "repeated-multiplication form with the FLT_MAX/base stop, computed identically by every "
1170|  "implementation",
1171|  "f32 logits in place, i32 history plane, optional i32 breaker plane, per-row params",
1172|  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
1173|  cSampleVoc, 1, nullptr, 0, nullptr,
1174|  nullptr, nullptr, r3d_rad_sample_dry },
1175|
1176|{ "sample_typical_f32", "sample_typical", "sample",
1177|  "locally typical narrowing in place",
1178|  "n_cand <= 4096; the entropy and the cut scan are SEQUENTIAL in candidate order on one "
1179|  "thread -- the entropy sum and the mass scan are order-sensitive sums, and the tie rule "
1180|  "(everything at the cut survives) is the llama.cpp one the oracle documents",
1181|  "i32 candidate ids and f32 candidate logits, both in place",
1182|  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
1183|  cSampleCand, 2, nullptr, 0, nullptr,
1184|  nullptr, nullptr, r3d_rad_sample_typical },
1185|
1186|{ "sample_xtc_f32", "sample_xtc", "sample",
1187|  "exclude-top-changes narrowing in place, from the row's own XTC coin",
1188|  "n_cand <= 4096; the coin is the spec stream at RAD_SAMPLE_KEY_XTC, so the firing "
1189|  "decision is reproducible; ref fires on `>=` and a draw exactly on the boundary fires",
1190|  "i32 candidate ids and f32 candidate logits, both in place",
1191|  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
1192|  cSampleCand, 2, nullptr, 0, nullptr,
1193|  nullptr, nullptr, r3d_rad_sample_xtc },
1194|
1195|{ "sample_logprobs_f32", "sample_logprobs", "sample",
1196|  "the raw distribution's summary: max, double sum of exp, and the k largest pairs",
1197|  "any vocabulary; k rounds of the block argmax under the one total order reproduce the "
1198|  "reference's insertion-sorted top-k; a vocabulary narrower than k leaves the remaining "
1199|  "slots as (-1, -inf) holes",
1200|  "f32 logits (read, not written), f32 stats row [M, 2 + 2k]",
1201|  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
1202|  cSampleVoc, 1, nullptr, 0, nullptr,
1203|  nullptr, nullptr, r3d_rad_sample_logprobs },
1204|
1205|/* ---- the band ops. Until these exist, a bf16 trunk that quantises its MLP activations
1206| * cannot START on this card at all: quant_act_fp8 / gated_quant_fp8 / add / gather_rows
1207| * carry no weights, so the planner cannot place them on a host kernel and the graph is
1208| * unsupported at resolve. fp8 is storage here, not arithmetic -- no E4M3 hardware is
1209| * touched, only the ABI encoder. */
1210|
1211|{ "quant_act_fp8_act", "quant_act_fp8", "band",
1212|  "per-group symmetric E4M3 quantisation of an activation, plain row-major",
1213|  "f32/bf16 inputs; the scale is amax/448, a dequant multiplier; group clamped to n",
1214|  "activation in, f8e4m3 codes and f32 group scales out",
1215|  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
1216|  cBandMn, 2, nullptr, 0, nullptr,
1217|  nullptr, nullptr, r3d_rad_quant_act_fp8 },
1218|
1219|{ "gated_quant_fp8_act", "gated_quant_fp8", "band",
1220|  "silu(gate)*up (or sigmoid(gate)*up) fused with the same quantiser",
1221|  "f32/bf16 gate_up [M, 2n]; the gated value narrows to bf16 once and the quantiser sees "
1222|  "the widened narrow; the gate is recomputed per pass rather than staged, deterministically",
1223|  "f32/bf16 gate_up in, f8e4m3 codes, f32 scales, optional bf16/f32 gated output",
1224|  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
1225|  cBandMn, 2, nullptr, 0, nullptr,
1226|  nullptr, nullptr, r3d_rad_gated_quant_fp8 },
1227|
1228|{ "add_act", "add", "band",
1229|  "elementwise add with the double-rounded div form",
1230|  "f32/bf16 operands; b broadcasts along rows only at exactly n elements; with div, "
1231|  "y = rd(rd(a+b)/div)",
1232|  "f32/bf16 in and out",
1233|  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
1234|  cBandMn, 2, nullptr, 0, nullptr,
1235|  nullptr, nullptr, r3d_rad_add },
1236|
1237|{ "mul_act", "mul", "band",
1238|  "elementwise product over dense [rows, n] rows, one thread a lane",
1239|  "f32/bf16 operands; b broadcasts along rows only at exactly n elements; holey row "
1240|  "pitches are refused (the kernel walks the range linearly, ref's offlin not)",
1241|  "f32/bf16 in and out",
1242|  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
1243|  cBandMn, 2, nullptr, 0, nullptr,
1244|  nullptr, nullptr, r3d_rad_mul },
1245|
1246|{ "sigmoid_act", "sigmoid", "band",
1247|  "the double-rounded sigmoid band, walked linearly over numel(Y)",
1248|  "f32/bf16; dense layouts only (see r3d_band.hip on why a pitch wider than n is "
1249|  "refused here and not in `add`)",
1250|  "f32/bf16 in and out, exp through the same double-rounded form the quantizers use",
1251|  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
1252|  cBandMn, 2, nullptr, 0, nullptr,
1253|  nullptr, nullptr, r3d_rad_sigmoid },
1254|
1255|{ "gather_rows_act", "gather_rows", "band",
1256|  "row gather with the zero-row padding convention",
1257|  "f32/bf16 rows, i32 indices; a negative index writes a zero row; an index >= the "
1258|  "source rows ALSO writes zeros rather than failing the launch (a device kernel cannot "
1259|  "pre-validate like the host reference; documented deviation)",
1260|  "f32/bf16 x and y, i32 idx",
1261|  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
1262|  cBandMn, 2, nullptr, 0, nullptr,
1263|  nullptr, nullptr, r3d_rad_gather_rows },
1264|
1265|/* ---- the Gated DeltaNet family (see r3d_gdn.hip): 48 of the qwen3.5-27B trunk's 64
1266| * layers run on these. Op schemas are verbatim libref, so libr4d (gfx12) and libr3d
1267| * dedupe them cleanly on a mixed rig -- and once the last GDN row lands, a 27B program
1268| * splits per-op between the two libraries instead of refusing to load. */
1269|
1270|{ "gdn_gated_rmsnorm_bf16", "gdn_gated_rmsnorm", "gdn",
1271|  "channel-RMS norm gated by silu/sigmoid of a companion plane, one block per "
1272|  "(token, value head) row",
1273|  "channels up to 65535; x and o dense-row [rows, channels] (a rank-2 pitch wider than "
1274|  "the row is fine, wider unit-strides are not); z may be a column slice of a fused "
1275|  "[rows, heads*channels] row -- the engine's own gate_rowoff walk",
1276|  "f32/bf16 x/z/o, f32-or-bf16 weight read as stored, f32 row sum and scale chain",
1277|  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
1278|  cGatedNorm, 2, nullptr, 0, nullptr,
1279|  nullptr, nullptr, r3d_rad_gdn_gated_rmsnorm },
1280|
1281|{ "gdn_conv_update_bf16", "gdn_conv_update", "gdn",
1282|  "decode-step depthwise conv with silu over the rolling window, one block per "
1283|  "(sequence, channel); the window reads at num_accepted-1 and rewrites shifted down "
1284|  "by one, so rejection is a read-offset change, not a recompute",
1285|  "conv_width up to 17 taps; x dense-row [T, conv_dim] (wider pitch fine, wider "
1286|  "unit-stride not); conv_state [slots, conv_dim, state_len] or flat-tight; index "
1287|  "tensors (state_idx, num_accepted, cu) must be fully dense; a window check that "
1288|  "fails skips its sequence instead of failing the op",
1289|  "f32/bf16 x/w/b/conv_state/q/k/v; i32/u32/i64 index reads; the width-tap f32 chain "
1290|  "runs in ref's ascending order over exact loads -- bit-exact by construction",
1291|  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
1292|  cConvUpdate, 3, nullptr, 0, nullptr,
1293|  nullptr, nullptr, r3d_rad_gdn_conv_update },
1294|
1295|{ "gdn_kkt_solve_bf16", "gdn_kkt_solve", "gdn",
1296|  "chunk preamble A = (I + strict_lower(diag(beta) K K^T e^dg))^-1, one block per "
1297|  "(chunk, sequence, v head); the inverse is ref's forward substitution with threads "
1298|  "owning columns, so every f32 chain runs in ref's order",
1299|  "chunk 64 (LDS bound; the derived parameter resolves from A's shape, other chunks are "
1300|  "refused); any head_k; k/A dense last stride with a wider row pitch fine; beta/g dense "
1301|  "within the row; cu fully dense; blocks past a sequence's end exit empty",
1302|  "f32/bf16 k and A out, f32-or-bf16 beta/g read as stored; the gram dot is a d-ascending "
1303|  "f32 chain over exact widening and the substitution an l-ascending one -- bit-exact by "
1304|  "construction, the scalar sibling of libr4d's WMMA blocked merge",
1305|  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
1306|  cKktSolve, 2, nullptr, 0, nullptr,
1307|  nullptr, nullptr, r3d_rad_gdn_kkt_solve },
1308|
1309|{ "gdn_chunk_scan_bf16", "gdn_chunk_scan", "gdn",
1310|  "the chunked delta-rule scan W = beta(V - e^g K S^T), V' = A W, O = scale e^g Q S^T + sA V', "
1311|  "S = e^{g_last} S + V'^T K', one block per (sequence, v head); the chunk recurrence is "
1312|  "serial by nature, so the transposed f32 state lives in one-column-per-thread register "
1313|  "arrays and every chain walks the oracle's order",
1314|  "head_k 128 (register bound: the state column IS a thread's register array), head_v up to "
1315|  "128 (one value column per thread), chunk 64 (the pack's and the c64 family's; derived from "
1316|  "A's shape when the parameter is absent); q/k/v/A/o dense last stride with wider row pitch "
1317|  "fine; h0/ht strides read off the tensor (a paged state cache is page-padded); state_idx "
1318|  "rows may be wider than one slot (column 0 is the slot), a negative slot skips the row's "
1319|  "outputs and state exactly as the reference's continue does",
1320|  "f32/bf16 q/k/v/A/o, f32-or-bf16 g/beta read as stored, f32 h0/ht state planes; every "
1321|  "ascending-d dot, the descending-in-place V' = A W, and the state update are f32 chains in "
1322|  "the oracle's order over exact widening -- bit-exact by construction, the scalar sibling of "
1323|  "libr4d's k128_c64 row",
1324|  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
1325|  cChunkScan, 4, nullptr, 0, nullptr,
1326|  nullptr, nullptr, r3d_rad_gdn_chunk_scan },
1327|
1328|{ "gdn_recurrent_update_bf16", "gdn_recurrent_update", "gdn",
1329|  "the decode delta rule S = e^g S + beta(v - S k) k^T against the paged state cache, the qk l2 "
1330|  "norms, the q readout and the gated rms norm in one launch; one block per (sequence, v head), "
1331|  "the transposed f32 state in one-column-per-thread register arrays, every token walking the "
1332|  "oracle's order",
1333|  "head_k 128 (register bound, as on chunk_scan), head_v up to 128; state_form per_candidate "
1334|  "only -- anchor batches stay with libr4d's anchor kernel, and the constraint says so; the "
1335|  "state cache's slot/head/column strides are read off the tensor (a paged cache is page-"
1336|  "padded); state_idx rows may be wider than one column (the accepted-prefix walk clamps to the "
1337|  "last), a negative input slot skips the (sequence, head) row whole and a negative output slot "
1338|  "skips only that token's state write, both as the reference's guards do; the o_q/o_scale "
1339|  "quantiser pair is refused, never silently dropped",
1340|  "f32/bf16 q/k/v/a/b/o and z, f32-or-bf16 A_log/dt_bias/norm_w read as stored (plain single-"
1341|  "plane weights -- the 27B container stores them bf16 and the oracle's dtype-driven ldt reads "
1342|  "the same bytes); the delta-rule chains, the serial-order rms and qk reductions, the double-"
1343|  "rounded exp/softplus/sigmoid and the three-step gated norm are bit-exact by construction, "
1344|  "the scalar sibling of libr4d's recurrent row",
1345|  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
1346|  cRecurrent, 4, nullptr, 0, nullptr,
1347|  nullptr, nullptr, r3d_rad_gdn_recurrent_update },
1348|
1349|/* ---- the three bands MiniCPM5-2B's graph names at resolve (see r3d_more.hip). The
1350| * engine refused to START without them: weightless device bands have no host stand-in. */
1351|
1352|{ "silu_mul_bf16", "silu_mul", "elem",
1353|  "silu-gated product over a fused [M, 2n] gate_up, one f32 expression chain per element",
1354|  "any n; the gate is the first half",
1355|  "bf16 in and out, fp32 arithmetic, double-width sigmoid like the oracle",
1356|  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
1357|  cBf16Row, 1, nullptr, 0, nullptr,
1358|  nullptr, nullptr, r3d_rad_silu_mul },
1359|
1360|{ "quant_act_i8", "quant_act_i8", "quant",
1361|  "per-row symmetric int8 quantisation of a bf16 activation, written in the byte order "
1362|  "the r4d GEMMs' A fragment reads; a third output selects the form that also emits the "
1363|  "per-group row sums",
1364|  "n a multiple of 16; one workgroup per row",
1365|  "bf16 in, int8 out with an f32 per-row scale",
1366|  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
1367|  cQuantActI8, 2, nullptr, 0, nullptr,
1368|  nullptr, nullptr, r3d_rad_quant_act_i8 },
1369|
1370|{ "rowtopk_bf16", "row_topk", "select",
1371|  "exact per-row top-R of a bf16 matrix, descending, ties to the lower column; R rounds "
1372|  "of the (value desc, id asc) block argmax reproduce topk_next's window exactly",
1373|  "any M, N a multiple of 8, R <= 32",
1374|  "bf16 in, f32 values + i32 columns out, optional f32 pairs plane",
1375|  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
1376|  cRowTopK, 5, nullptr, 0, nullptr,
1377|  nullptr, nullptr, r3d_rad_row_topk },
1378|
1379|{ "rmsnorm_quant_fp8", "rmsnorm_quant_fp8", "quant",
1380|  "RMS norm and the fp8 group quantiser in one pass; the bf16 norm output is written and "
1381|  "the codes are taken over it, exactly as ref stores the narrowed values before reading "
1382|  "them; with a residual the bf16-narrowed sum is stored before the variance reads it",
1383|  "n a multiple of 4, group 128; one workgroup per row, the row staged in LDS (<= 30720)",
1384|  "bf16 in, fp8_e4m3 codes with an f32 per-group scale, optional bf16/f32 out_bf16",
1385|  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
1386|  cRmsNormQuantFp8, 3, nullptr, 0, nullptr,
1387|  nullptr, nullptr, r3d_rad_rmsnorm_quant_fp8 },
1388|
1389|};
1390|
1391|/* R3D_NO_GDN=1 withdraws the five GDN rows/schemas from the registration, which is the
1392| * serve's A/B lever for "GDN arithmetic on the host" without rebuilding a second binary:
1393| * the same .so, one env difference between arms. The band planner then places those bands
1394| * on the host exactly as it does for gdn_conv_prep today (they have weights), so startup
1395| * still completes. Unset is untouched behaviour. */
1396|static bool r3d_gdn_hidden(void) { static const int v = []{ return getenv("R3D_NO_GDN") ? 1 : 0; }(); return v != 0; }
1397|static bool r3d_row_hidden(const char* op) {
1398|    if (!r3d_gdn_hidden() || !op) return false;
1399|    static const char* const kGdn[] = { "gdn_gated_rmsnorm", "gdn_conv_update", "gdn_kkt_solve",
1400|                                        "gdn_chunk_scan", "gdn_recurrent_update" };
1401|    for (const char* n : kGdn) if (strcmp(op, n) == 0) return true;
1402|    return false;
1403|}
1404|
1405|int rad_kernel_schema_count(void) {
1406|    int n = 0;
1407|    for (const RadOpSchema& s : kSchemas) if (!r3d_row_hidden(s.op)) ++n;
1408|    return n;
1409|}
1410|const RadOpSchema* rad_kernel_schema_at(int i) {
1411|    if (i < 0) return nullptr;
1412|    for (const RadOpSchema& s : kSchemas) if (!r3d_row_hidden(s.op) && i-- == 0) return &s;
1413|    return nullptr;
1414|}
1415|
1416|int rad_kernel_count(void) {
1417|    int n = 0;
1418|    for (const RadKernelInfo& k : kRows) if (!r3d_row_hidden(k.op)) ++n;
1419|    return n;
1420|}
1421|const RadKernelInfo* rad_kernel_at(int i) {
1422|    if (i < 0) return nullptr;
1423|    for (const RadKernelInfo& k : kRows) if (!r3d_row_hidden(k.op) && i-- == 0) return &k;
1424|    return nullptr;
1425|}
1426|
1427|} /* extern "C" */