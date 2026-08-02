# DeepSeek-V4-Flash on gfx1151 (Strix Halo): why performance is capped

Notes from benchmarking and profiling DeepSeek-V4-Flash (284.33B, `deepseek4` arch:
43 layers, MLA attention kv_lora=512, 256 experts / 6 used + 1 shared, sparse-attention
indexer top_k=512) on AMD Strix Halo (Ryzen AI Max+ 395, gfx1151, RDNA 3.5 40 CU,
122 GB unified LPDDR5x). All numbers measured with llama.cpp HIP backend, flash
attention on, f16 KV cache.

## Hardware envelope

- Memory bandwidth: 256 GB/s theoretical, ~200 GB/s realistic (shared between CPU and GPU)
- Compute: 40 CU RDNA 3.5 iGPU, no fast MMA path for head size > 128 on RDNA
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
  scaling with context depth (tg 15.3 @ d=0 -> 13.4 @ d=16k).

Things that do NOT help (measured):

- CPU MoE offload (`-ncmoe 43`): same physical RAM, ~22 GB/s effective, plus 86
  CPU<->GPU transitions per token. Strictly worse.
- GPU zero-copy of host weights (`GGML_OP_OFFLOAD_MIN_BATCH=1`): GART reads from
  the GPU side are no faster than CPU reads. Same speed.
- MMVQ nwarps tuning (1/2/4/8): flat, 8 regresses (register pressure on the small LLC).
- MTP speculative decoding (PR #25784, verified working): the draft model shares the
  same bandwidth; embedding + output-head reads per draft step eat the win. +3%.

## Prefill (pp ~230-240 t/s): compute-capped

Prefill reads each weight once per batch, so bandwidth amortizes; the limit is the
40 CU compute. Two specifics:

- MMQ config matters a lot on gfx1151: 128 threads / I=64 beats the upstream
  256 threads / I=128 table by +46% at pp512 (register pressure on the small LLC).
  For MoE expert dispatch, capping the MMQ tile J at 48 avoids a regression.
- MLA attention has head size 512. On RDNA, MMA flash-attention kernels only cover
  head size <= 128, so D=512 runs the scalar TILE kernel. Cost grows with context
  depth (pp512 231 @ d=0 -> 136 @ d=16k).

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

## Bottom line

Decode is bounded by bytes-per-token (7.5 GB) against ~200 GB/s; prefill is bounded
by 40 CU of compute. Only two structural levers remain: a smaller quant for the
dense (not just expert) tensors, and serving multiple sequences in parallel, which
shares the weight reads across sequences.
