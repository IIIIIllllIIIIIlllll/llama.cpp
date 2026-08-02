# DeepSeek-V4-Flash 在 gfx1151 (Strix Halo) 上性能受限的原因

本文基于对 DeepSeek-V4-Flash（284.33B，`deepseek4` 架构：43 层，MLA 注意力
kv_lora=512，256 个 expert / 激活 6+1，稀疏注意力 indexer top_k=512）在
AMD Strix Halo（Ryzen AI Max+ 395，gfx1151，RDNA 3.5 40 CU，122 GB 统一
LPDDR5x 内存）上的实测与 profiling。数据均来自 llama.cpp HIP 后端、
flash attention 开、f16 KV cache 的实测。

## 硬件包线

- 内存带宽：理论 256 GB/s，实际可用 ~200 GB/s（CPU 与 GPU 共享）
- 算力：40 CU RDNA 3.5 核显，RDNA 上 head size > 128 没有可用的 MMA 快路径
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
  深度增长（tg 15.3 @ d=0 -> 13.4 @ d=16k）。

实测无效的方向：

- CPU MoE offload（`-ncmoe 43`）：同一块物理内存，CPU 侧有效只有 ~22 GB/s，
  还引入每 token 86 次 CPU<->GPU 切换，全面更差。
- GPU 零拷贝读 host 权重（`GGML_OP_OFFLOAD_MIN_BATCH=1`）：GPU 走 GART 读
  host 内存并不比 CPU 读快，速度相同。
- MMVQ nwarps 扫描（1/2/4/8）：打平，8 回退（小 LLC 上的寄存器压力）。
- MTP 投机解码（PR #25784，已验证可用）：draft 模型共享同一带宽，每步起草
  都要读 embedding + output head，收益被吃光，净 +3%。

## Prefill（pp ~230-240 t/s）：算力封顶

Prefill 每批只读一遍权重，带宽被摊薄，瓶颈在 40 CU 的算力。两个要点：

- MMQ 配置在 gfx1151 上影响很大：128 线程 / I=64 比上游的 256 线程 / I=128
  表快 46%（pp512，小 LLC 的寄存器压力）。MoE expert dispatch 还需要把
  MMQ tile 的 J 上限压到 48 避免回退。
- MLA 注意力 head size 为 512，而 RDNA 上 MMA flash-attention 只覆盖
  head size <= 128，D=512 只能走标量 TILE kernel，开销随深度增长
  （pp512 231 @ d=0 -> 136 @ d=16k）。

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

## 结论

Decode 由"每 token 字节数（7.5 GB）对 ~200 GB/s"决定，prefill 由 40 CU
算力决定。仅剩两个结构性杠杆：dense 部分也更小的量化（不仅是 expert），
以及多序列并行服务（权重读取在序列间共享）。
