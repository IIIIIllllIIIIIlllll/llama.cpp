# DeepSeek-V4-Flash 在 gfx1151 (Strix Halo) 上性能受限的原因

本文基于对 DeepSeek-V4-Flash（284.33B，`deepseek4` 架构：43 层，MLA 注意力
kv_lora=512，256 个 expert / 激活 6+1，稀疏注意力 indexer top_k=512）在
AMD Strix Halo（Ryzen AI Max+ 395，gfx1151，RDNA 3.5 40 CU，122 GB 统一
LPDDR5x 内存）上的实测与 profiling。数据均来自 llama.cpp HIP 后端、
flash attention 开、f16 KV cache 的实测。

## 硬件包线

- 内存带宽：理论 256 GB/s，实际可用 ~200 GB/s（CPU 与 GPU 共享）
- 算力：40 CU RDNA 3.5 核显。本分支把 RDNA 的 MMA 快路径扩展到了 head
  size 128 以上（见 Prefill 一节）
- 内存容量：共 122 GB，模型按量化不同 84-96 GB

## Decode（tg ~15 t/s）：带宽封顶，理论 ~27 t/s

每生成一个 token 都必须物理读取：

- ~2.5 GB 激活 expert 权重（256 选 7）
- ~5 GB dense 权重（注意力投影、embedding、output head）

合计 ~7.5 GB/token。按 ~200 GB/s 计算，绝对天花板约 27 t/s，实测 ~15 t/s。
差距不是单一原因：

- MoE expert 的 GEMV 已跑到 ~220 GB/s（接近封顶）。
- dense Q8_0 注意力 GEMV 只到 ~110 GB/s。每 token 有 ~320 次小 GEMV kernel
  调用（MLA 的 LoRA 分解 wq_a/wq_b、8 组输出投影、indexer 投影、压缩器更新），
  这些小 kernel 是 launch/占用率受限，不是带宽受限。
- 稀疏 indexer + CSA/HCA 压缩器每层每 token 还有几十个小算子，并随上下文
  深度增长（tg 15.3 @ d=0 -> 13.4 @ d=16k）。lightning indexer 打分 kernel
  新增 RDNA3 WMMA 路径（每步 MMA 处理 16 heads x 32 KV 向量，量化 K 在
  shared memory 中反量化为 half），不再只有标量 vector kernel。

实测无效的方向：

- CPU MoE offload（`-ncmoe 43`）：同一块物理内存，CPU 侧有效只有 ~22 GB/s，
  还引入每 token 86 次 CPU<->GPU 切换，全面更差。
- GPU 零拷贝读 host 权重（`GGML_OP_OFFLOAD_MIN_BATCH=1`）：GPU 走 GART 读
  host 内存并不比 CPU 读快，速度相同。
- MMVQ nwarps 扫描（1/2/4/8）：打平，8 回退（小 LLC 上的寄存器压力）。
- MTP 投机解码（PR #25784，已验证可用）：draft 模型共享同一带宽，每步起草
  都要读 embedding + output head，收益被吃光，净 +3%。

## 加载时合并 dense GEMV

针对每 token ~320 次小 GEMV launch 的问题：把共享同一输入的多个投影矩阵在
模型加载时沿输出维拼接成一个合成权重张量，图里每组只发一次 `ggml_mul_mat`，
再用物化 view（`ggml_cont`，下游 norm/reshape 要求连续输入）切出各块结果。
GGUF 文件不变，纯运行时变换。每层的分组（以 UD-IQ3_XXS 的实际量化类型为准）：

- 组 A（输入 = attn_norm 输出）：HCA 层 `[wkv, comp_wkv, comp_wgate]`（3 -> 1 次
  GEMV）；CSA 层再加 `[lid_comp_wkv, lid_comp_wgate]`（5 -> 1）。wq_a（Q6_K）与
  indexer_proj（F32）类型与其他成员（Q8_0）不一致，保持分离张量；raw 层仅
  wkv 一个成员，不合并。
- 组 B（输入 = rms_norm(wq_a) 输出，仅 CSA 层）：`[wq_b, indexer_attn_q_b]`
  （2 -> 1，均为 Q8_0）。
- 组 C（输入 = ffn_norm 输出）：`[ffn_gate_shexp, ffn_up_shexp]`（2 -> 1，均为
  Q6_K）；切出的 gate/up 仍按各自区间做 swiglu clamp。

合并以组为单位，任一前提不满足就静默回退到分离张量：mmap 加载模式（合并
需要可写 staging，必须 `-lm none`/`dio`——本平台本来就要求）、源张量缺失或
量化类型/行长不一致、源张量带 NVFP4 sidecar scale、或用户 `-ot` 正则命中了
原始张量名。加载旧 GGUF 行为完全一致，只是 kernel 更少。

## 投机解码（DSpark / MTP 草稿）

DSV4 的 DSpark 草稿实测可用：tg 16.0 -> 19.8（+24%，ctx 8192，temp 0）。
要点：

- 部分 DSV4 DSpark GGUF 缺 `token_embd.weight`/`output.weight`（转换不完整；
  上游 `conversion/` 只有 Qwen3 backbone 的 DSpark 转换路径）。此时草稿经
  `ctx_other` 引用主模型张量：不加 `--fit on` 会触发 ggml 调度器断言
  （pre-allocated tensor ... cannot run the operation），加 `--fit on` 则正常
  且性能无损（fit 只改 buffer 布局，基线与 DSpark 实测 fit on/off 均一致）。
  本分支的 dflash 加载器支持 GGUF 自带这两个张量（可选张量，缺失时维持
  原行为），可用 `scripts/gguf-inject-tensors.py` 从 MTP GGUF 注入补齐——
  自包含后不依赖 `--fit`。
- `--spec-draft-n-max` 保持默认 3：n_max=4 掉到 17.0，n_max=5 掉到 14.1
  （草稿 block_size=5，更大的块浪费草稿算力）。
- 草稿可放第二块显卡（实测 RTX 2080 Ti / CUDA sm_75）：速度与本机相同
  （19.8 vs 19.9 t/s，隐状态经 host 内存交换，开销可忽略），但省下 ~12 GB
  统一内存。需要 `GGML_BACKEND_DL=ON` 的 CUDA+HIP 双后端构建
  （两库导出同名符号，必须运行时 dlopen 隔离）、自包含草稿 GGUF、
  `--spec-draft-device CUDA0`。MTP-Q8_0 草稿同法可用：tg 16.0 -> 18.2。

## Prefill（pp ~230-240 t/s）：算力封顶

Prefill 每批只读一遍权重，带宽被摊薄，瓶颈在 40 CU 的算力。两个要点：

- MMQ 配置在 gfx1151 上影响很大：128 线程 / I=64 比上游的 256 线程 / I=128
  表快 46%（pp512，小 LLC 的寄存器压力）。MoE expert dispatch 还需要把
  MMQ tile 的 J 上限压到 48 避免回退。
- MLA 注意力 head size 为 512。RDNA 上 MMA flash-attention 只覆盖
  head size <= 128，D=512 只能走标量 TILE kernel（开销随深度增长，
  pp512 231 @ d=0 -> 136 @ d=16k）。曾实验性放开 DKQ=512 的 RDNA WMMA
  路径（Q_in_reg=false，多组配置），实测在 gfx1151 上比 TILE 更慢
  （512 宽累加器导致寄存器 spill），已整体回退。
  最终方案是针对 DKQ=DV=512 且 K==V 的 RDNA3 专用 WMMA kernel
  （`fattn-mma-d512-rdna.cuh`）：PV 阶段 warp 沿 DV 分工（每 warp 只持有
  256 宽累加器切片，避免 spill）+ P 物化 shared + K==V 复用 tile，
  支持 attention sinks，对 MLA prefill 形状启用 dispatch。
  实测 pp512 218 -> 230（+5.5%），pp512@16k 148 -> 176（+18.7%）。

## 内存容量约束

默认 `mmap` 加载无法全量 GPU：mmap 的文件页（~96 GB）与 GTT 副本（~107 GB）
同时存在，超过 KFD 驻留内存限额，触发 SVM 驱动层死锁。必须用 `-lm none`
（普通 read()）消除重复。84.7 GB 的 IQ2_M 量化可留 ~25 GB 余量给 KV 和
计算 buffer。

## 实测结果（最终配置）

配置：全量 GPU，`-lm none`，FA 开，RDNA3.5 MMQ 调优
（128 线程 / I=64，MoE J<=48）。

| 量化 | pp512 | pp2048 | tg128 | tg128@16k |
|---|---|---|---|---|
| IQ3_XXS (96 GB) | 211 | 223 | 15.3 | 13.4 |
| IQ2_M (85 GB) | 231 | 238 | 14.9 | 13.2 |

参照：朴素 mmap 配置完全无法运行（SVM 死锁）；`-ncmoe 43` CPU-MoE 配置：
tg 8.7，pp512 105。

叠加本分支后续三项优化（lightning indexer RDNA3 WMMA 路径 + 加载时合并
dense GEMV + D=512 RDNA3 专用 WMMA fattn kernel）后，IQ3_XXS 的实测为：

| 测试 | 基线 | 优化后 | 变化 |
|---|---|---|---|
| pp512 | 211 | 230 | +9.0% |
| tg128 | 15.3 | 16.0 | +4.8% |
| pp512@16k | 136 | 176 | +29.1% |
| tg128@16k | 13.4 | 14.1 | +5.2% |

## 结论

Decode 由"每 token 字节数（7.5 GB）对 ~200 GB/s"决定，prefill 由 40 CU
算力决定。仅剩两个结构性杠杆：dense 部分也更小的量化（不仅是 expert），
以及多序列并行服务（权重读取在序列间共享）。
