# DeepSeek-V4-Flash on gfx1151 (Strix Halo): why performance is capped

Notes from benchmarking and profiling DeepSeek-V4-Flash (284.33B, `deepseek4` arch:
43 layers, MLA attention kv_lora=512, 256 experts / 6 used + 1 shared, sparse-attention
indexer top_k=512) on AMD Strix Halo (Ryzen AI Max+ 395, gfx1151, RDNA 3.5 40 CU,
122 GB unified LPDDR5x). All numbers measured with llama.cpp HIP backend, flash
attention on, f16 KV cache.

## Hardware envelope

- Memory bandwidth: 256 GB/s theoretical, ~200 GB/s realistic (shared between CPU and GPU)
- Compute: 40 CU RDNA 3.5 iGPU. The MMA fast paths on RDNA are extended beyond
  head size 128 by this branch (see the prefill section)
- Memory capacity: 122 GB total, model is 84-96 GB depending on quant

## Decode (tg ~15 t/s): bandwidth-capped at ~27 t/s

Every generated token must physically read:

- ~2.5 GB of active expert weights (7 of 256 experts per token)
- ~5 GB of dense weights (attention projections, embeddings, output head)

Total ~7.5 GB/token. At ~200 GB/s the absolute ceiling is ~27 t/s; we measure ~15 t/s.
The gap is not one big thing:

- The MoE expert GEMV reaches ~220 GB/s (near ceiling).
- The dense Q8_0 attention GEMV reaches only ~110 GB/s. Per token there are ~320
  small GEMV kernel launches (MLA LoRA factorization wq_a/wq_b, 8 output groups,
  indexer projections, compressor updates). These small kernels are launch and
  occupancy bound, not bandwidth bound.
- The sparse indexer + CSA/HCA compressor add dozens of small ops per layer per token,
  scaling with context depth (tg 15.3 @ d=0 -> 13.4 @ d=16k). The lightning indexer
  score kernel now has an RDNA3 WMMA path (16 heads x 32 KV vectors per MMA step,
  quantized K dequantized to half in shared memory) instead of the scalar vector
  kernel only.

Things that do NOT help (measured):

- CPU MoE offload (`-ncmoe 43`): same physical RAM, ~22 GB/s effective, plus 86
  CPU<->GPU transitions per token. Strictly worse.
- GPU zero-copy of host weights (`GGML_OP_OFFLOAD_MIN_BATCH=1`): GART reads from
  the GPU side are no faster than CPU reads. Same speed.
- MMVQ nwarps tuning (1/2/4/8): flat, 8 regresses (register pressure on the small LLC).
- MTP speculative decoding (PR #25784, verified working): the draft model shares the
  same bandwidth; embedding + output-head reads per draft step eat the win. +3%.

## Load-time merged dense GEMV

To attack the ~320 small GEMV launches per token, projections that share the same
input are concatenated along the output dimension at model load time into one
synthetic weight tensor, so the graph issues a single `ggml_mul_mat` per group and
slices the result with materialized views (`ggml_cont`; downstream norm/reshape
ops require contiguous inputs). The GGUF file is unchanged; this is a pure runtime
transform. Groups per layer (for the UD-IQ3_XXS quant types):

- Group A (input = attn_norm output): HCA layers `[wkv, comp_wkv, comp_wgate]`
  (3 -> 1 GEMV); CSA adds `[lid_comp_wkv, lid_comp_wgate]` (5 -> 1). wq_a (Q6_K)
  and indexer_proj (F32) use different types than the Q8_0 members and stay
  separate; raw layers have only wkv, so nothing to merge.
- Group B (input = rms_norm(wq_a) output, CSA only): `[wq_b, indexer_attn_q_b]`
  (2 -> 1, both Q8_0).
- Group C (input = ffn_norm output): `[ffn_gate_shexp, ffn_up_shexp]` (2 -> 1,
  both Q6_K); the gate/up slices keep their distinct swiglu clamp limits.

The merge is per-group and falls back silently to the separate tensors when any
precondition fails: mmap load mode (the merge needs writable staging, so `-lm none`
/`dio` — already required on this platform), a source tensor missing or having a
different quant type / row length, per-source NVFP4 sidecar scales, or a user `-ot`
override matching one of the original tensor names. Loading an old GGUF therefore
behaves identically, just with fewer kernels.

## Speculative decoding (DSpark / MTP drafts)

The DSV4 DSpark draft works: tg 16.0 -> 19.8 t/s (+24%, ctx 8192, temp 0).
Notes:

- Some DSV4 DSpark GGUFs lack `token_embd.weight`/`output.weight` (incomplete
  conversion; upstream `conversion/` only has a Qwen3-backbone DSpark path).
  The draft then references the target's tensors via `ctx_other`: without
  `--fit on` the ggml scheduler aborts ("pre-allocated tensor ... cannot run
  the operation"); with `--fit on` it works with no performance cost (fit only
  changes buffer placement; baseline and DSpark measure identically with fit
  on/off). The dflash loader in this branch also accepts self-contained GGUFs
  carrying those two tensors (optional; old files behave as before) — inject
  them from the MTP GGUF with `scripts/gguf-inject-tensors.py` and `--fit`
  is no longer required.
- Keep `--spec-draft-n-max` at the default 3: n_max=4 drops to 17.0 t/s and
  n_max=5 to 14.1 t/s (the draft block_size is 5; larger blocks waste draft
  compute).
- The draft can run on a second GPU (measured: RTX 2080 Ti / CUDA sm_75) at
  the same speed as on-device (19.8 vs 19.9 t/s — hidden states cross via host
  memory, negligible cost), freeing ~12 GB of unified memory. Requires a
  CUDA+HIP dual-backend build with `GGML_BACKEND_DL=ON` (both libs export the
  same symbol, so runtime dlopen isolation is mandatory), a self-contained
  draft GGUF, and `--spec-draft-device CUDA0`. The MTP-Q8_0 draft works the
  same way: tg 16.0 -> 18.2 t/s.

## Prefill (pp ~230-240 t/s): compute-capped

Prefill reads each weight once per batch, so bandwidth amortizes; the limit is the
40 CU compute. Two specifics:

- MMQ config matters a lot on gfx1151: 128 threads / I=64 beats the upstream
  256 threads / I=128 table by +46% at pp512 (register pressure on the small LLC).
  For MoE expert dispatch, capping the MMQ tile J at 48 avoids a regression.
- MLA attention has head size 512. The MMA flash-attention kernels on RDNA only
  cover head size <= 128, so D=512 runs the scalar TILE kernel (cost grows with
  context depth, pp512 231 @ d=0 -> 136 @ d=16k). Enabling the RDNA WMMA path for
  DKQ=512 was tested (Q_in_reg=false, several config variants) and measured
  *slower* than the TILE kernel on gfx1151 (register spills with 512-wide
  accumulators), so it stays disabled.
  The final solution is a dedicated RDNA3 WMMA kernel for DKQ=DV=512 with K==V
  (`fattn-mma-d512-rdna.cuh`): warps split along DV for P*V (each warp holds
  only a 256-wide accumulator slice, no spills) + P materialized in shared
  memory + the K tile reused as V. Attention sinks are supported and the kernel
  is dispatched for the MLA prefill shape. Measured: pp512 218 -> 230 (+5.5%),
  pp512@16k 148 -> 176 (+18.7%).

## Memory capacity constraint

Full GPU offload fails with the default `mmap` load mode: the mmap'd file pages
(~96 GB) and the GTT copy (~107 GB) coexist and exceed the KFD resident system
memory limit, causing an SVM driver livelock. `-lm none` (plain read()) avoids the
duplication and is required. With the 84.7 GB IQ2_M quant there is ~25 GB of headroom
for KV cache and compute buffers.

## Measured results (final configuration)

Config: full GPU offload, `-lm none`, FA on, RDNA3.5 MMQ tuning
(128 threads / I=64, MoE J<=48).

| quant | pp512 | pp2048 | tg128 | tg128@16k |
|---|---|---|---|---|
| IQ3_XXS (96 GB) | 211 | 223 | 15.3 | 13.4 |
| IQ2_M (85 GB) | 231 | 238 | 14.9 | 13.2 |

Reference points: naive mmap config does not run at all (SVM livelock);
`-ncmoe 43` CPU-MoE config: tg 8.7, pp512 105.

With the three follow-up optimizations of this branch on top (lightning indexer
RDNA3 WMMA path + load-time merged dense GEMV + dedicated D=512 RDNA3 WMMA
fattn kernel), IQ3_XXS measures:

| test | baseline | optimized | change |
|---|---|---|---|
| pp512 | 211 | 230 | +9.0% |
| tg128 | 15.3 | 16.0 | +4.8% |
| pp512 @ d=16k | 136 | 176 | +29.1% |
| tg128 @ d=16k | 13.4 | 14.1 | +5.2% |

## Bottom line

Decode is bounded by bytes-per-token (7.5 GB) against ~200 GB/s; prefill is bounded
by 40 CU of compute. Only two structural levers remain: a smaller quant for the
dense (not just expert) tensors, and serving multiple sequences in parallel, which
shares the weight reads across sequences.
