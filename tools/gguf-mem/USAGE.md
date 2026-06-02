# gguf-mem 使用说明

从 GGUF 元数据（不下载权重）估算模型内存占用。

## 工具

| 工具 | 功能 |
|------|------|
| `gguf-extract-meta` | 从完整 GGUF 文件（或 URL）提取仅含 header 的元数据文件 |
| `GGufMetaExtractor.java` | 同上，纯 Java 跨平台版 |
| `gguf-mem` | 加载元数据文件，输出精确内存估算 |

## 编译

```bash
# 配置（仅 CPU）
cmake -B build -G "Visual Studio 17 2022" -A x64 `
    -DGGML_CPU=ON -DGGML_CUDA=OFF -DGGML_HIP=OFF `
    -DGGML_VULKAN=OFF -DGGML_METAL=OFF `
    -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TOOLS=ON

# 编译
cmake --build build --config Release --target gguf-mem gguf-extract-meta
```

产出：
- `build/bin/Release/gguf-mem.exe`
- `build/bin/Release/gguf-extract-meta.exe`
- 以及依赖 DLL（同目录）

Java 版不需要编译系统：
```bash
cd tools/gguf-extract-meta-java
javac GGufMetaExtractor.java
```

## 快速开始

```bash
# 1. 从 HuggingFace URL 提取元数据（只下载 ~10 MB header，不下载权重）
gguf-extract-meta https://huggingface.co/user/repo/resolve/main/model.gguf model.meta.gguf

# Java 版等效命令
java GGufMetaExtractor https://huggingface.co/user/repo/resolve/main/model.gguf model.meta.gguf

# 如果 GGUF 已在本地，跳过提取，直接估算
gguf-mem -m model.gguf -c 8192

# 2. 估算内存
gguf-mem -m model.meta.gguf -c 8192
```

## gguf-mem 参数

```
gguf-mem -m <model> [options]
```

| 参数 | 说明 | 默认 |
|------|------|------|
| `-m`, `--model` | GGUF 文件路径（必填） | — |
| `-c`, `--ctx-size N` | 上下文大小 | 模型 n_ctx_train |
| `--swa-full` | SWA 层用完整 KV cache（默认不用） | 关闭 |
| `-fa`, `--flash-attn on/off/auto` | Flash Attention | auto |
| `-b`, `--batch-size N` | 批处理大小 | 2048 |
| `-ub`, `--ubatch-size N` | 微批处理大小 | 512 |
| `-np`, `--parallel N` | 并行序列数 | 1 |
| `-ctk`, `--cache-type-k TYPE` | KV 缓存 K 类型（f16/q8_0/q4_0...） | f16 |
| `-ctv`, `--cache-type-v TYPE` | KV 缓存 V 类型 | f16 |
| `-ngl`, `--n-gpu-layers N` | GPU offload 层数 | 0 |
| `-ts`, `--tensor-split F,F,...` | 多 GPU 张量分配比例 | — |
| `--no-kv-offload` | KV 不 offload 到 GPU | — |

参数与 `llama-fit-params` 完全兼容，可互换使用。

## 示例

```bash
# Gemma 4 31B Q8_0, 4K ctx, 不开启 SWA 全缓存
gguf-mem -m A.meta.gguf -c 4096
# Host 35056 1520 533
#      ^^^^^ ^^^^ ^^^
#      model context compute (MiB)

# 同上，SWA 全缓存（KV 占用更大）
gguf-mem -m A.meta.gguf -c 4096 --swa-full
# Host 35056 1760 533

# Q8_0 KV cache 压缩
gguf-mem -m A.meta.gguf -c 8192 -ctk q8_0 -ctv q8_0
# Host 35056 2760 533

# GPU 全 offload + 32K 上下文
gguf-mem -m A.meta.gguf -c 32768 -ngl 99
# CUDA0 31000 3760 533
# Host  4056 0 0
#      ^^^^ model ^^^^ context ^^^ compute on each device

# 多序列并行
gguf-mem -m A.meta.gguf -c 4096 -np 4
# Host 35056 3520 533
```

## 输出格式

每行一个设备（GPU 或 Host），空格分隔：

```
<设备名> <model_mib> <context_mib> <compute_mib>
```

- **model**：模型权重所需内存
- **context**：KV cache + 循环状态等上下文内存
- **compute**：推理临时缓冲区

## 原理

1. `gguf-extract-meta` 下载 GGUF 文件头（~10 MB），不下载权重
2. `gguf-mem` 调用 `llama_model_load_from_file(no_alloc=true)` 加载模型描述符，不读取权重
3. `llama_get_memory_breakdown()` 遍历所有 tensor 计算理论内存
4. 所有架构（SWA、hybrid、MLA、MoE）由 llama.cpp 自动处理，工具零架构特定逻辑
