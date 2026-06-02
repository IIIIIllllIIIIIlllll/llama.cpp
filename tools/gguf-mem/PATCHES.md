# gguf-mem: llama.cpp 补丁记录

## 概述

为支持「从 GGUF 元数据（不读取权重）估算内存」，需对 llama.cpp 源码做 3 处修改。
这些补丁让 `llama_model_load_from_file(no_alloc=true)` 能接受**截断的元数据文件**
（仅含 header，不含 tensor weight data）。

## 补丁清单

| # | 文件 | 改动 |
|---|------|------|
| 1 | `ggml/src/gguf.cpp` | `gguf_init_from_reader` — 两处 |
| 2 | `src/llama-model-loader.h` | `llama_tensor_weight` 构造 — 文件边界检查 |
| 3 | `src/llama-model-loader.cpp` | 三处调用传参 |

---

## 补丁 1：`ggml/src/gguf.cpp`

### 改动 A：跳过 seek 到 data section

**位置**：`gguf_init_from_reader` 函数，原第 752 行附近

```cpp
// 原代码
if (n_tensors > 0 && !gr.seek(GGML_PAD(gr.tell(), ctx->alignment))) {
    GGML_LOG_ERROR("%s: failed to seek to beginning of data section\n", __func__);
    gguf_free(ctx);
    return nullptr;
}
ctx->offset = gr.tell();

// 改为
if (params.no_alloc) {
    ctx->offset = GGML_PAD(gr.tell(), ctx->alignment);
} else if (n_tensors > 0 && !gr.seek(GGML_PAD(gr.tell(), ctx->alignment))) {
    GGML_LOG_ERROR("%s: failed to seek to beginning of data section\n", __func__);
    gguf_free(ctx);
    return nullptr;
} else {
    ctx->offset = gr.tell();
}
```

**原因**：元数据文件没有 weight data section，seek 会失败。
`no_alloc=true` 时不读 tensor data，所以不需要真正定位到 data section。

### 改动 B：跳过 tensor offset 连续性校验

**位置**：同一函数，原第 766 行附近

```cpp
// 原代码
if (ti.offset != ctx->size) {

// 改为
if (ti.offset != ctx->size && !params.no_alloc) {
```

**原因**：元数据文件的 tensor offset 指向文件外的 GB 级位置，不连续是预期行为。
`no_alloc=true` 时不使用这些 offset，跳过校验。

---

## 补丁 2：`src/llama-model-loader.h`

### 改动：`llama_tensor_weight` 构造函数

**位置**：`struct llama_tensor_weight`，原第 39 行附近

```cpp
// 原签名
llama_tensor_weight(const llama_file * file, uint16_t idx,
    const struct gguf_context * gguf_ctx, ggml_tensor * tensor)

// 改为
llama_tensor_weight(const llama_file * file, uint16_t idx,
    const struct gguf_context * gguf_ctx, ggml_tensor * tensor,
    bool no_alloc = false)

// 构造函数内部的文件边界检查
// 原代码
offs = gguf_get_data_offset(gguf_ctx) + gguf_get_tensor_offset(gguf_ctx, tensor_idx);
if (offs + ggml_nbytes(tensor) < offs || offs + ggml_nbytes(tensor) > file->size()) {
    throw std::runtime_error("tensor data is not within the file bounds...");
}

// 改为
offs = gguf_get_data_offset(gguf_ctx) + gguf_get_tensor_offset(gguf_ctx, tensor_idx);
if (!no_alloc) {
    if (offs + ggml_nbytes(tensor) < offs || offs + ggml_nbytes(tensor) > file->size()) {
        throw std::runtime_error("tensor data is not within the file bounds...");
    }
}
```

**原因**：元数据文件只有 ~10 MB，而 tensor offset 指向 GB 级位置，
`offs + nbytes > file->size()` 必然触发。`no_alloc` 模式下不会调用 `load_all_data()`，
offset 不会被使用，边界检查无意义。

**注意**：`bool no_alloc = false` 默认值保证现有调用方不受影响。

---

## 补丁 3：`src/llama-model-loader.cpp`

### 改动：三处 `llama_tensor_weight` 构造调用

**位置**：`llama_model_loader` 构造函数中

```cpp
// 原代码（3 处，约第 582 / 648 / 692 行）
weights_map.emplace(tensor_name, llama_tensor_weight(files.back().get(), 0, metadata, cur));

// 改为
weights_map.emplace(tensor_name, llama_tensor_weight(files.back().get(), 0, metadata, cur, no_alloc));
```

`no_alloc` 是 `llama_model_loader` 构造函数的成员变量，在这三处都可直接使用。

---

## 移植检查清单

当 llama.cpp 更新后，按以下步骤重新应用补丁：

### 步骤 1：定位新代码位置

```bash
# 检查 gguf.cpp 改动是否仍在
git diff <旧tag> <新tag> -- ggml/src/gguf.cpp | grep "no_alloc"

# 检查 llama_tensor_weight 是否变化
git diff <旧tag> <新tag> -- src/llama-model-loader.h

# 检查三处调用是否变化
git diff <旧tag> <新tag> -- src/llama-model-loader.cpp
```

### 步骤 2：检查上下文是否变化

- `gguf_init_params` 是否还有 `no_alloc` 成员？
- `llama_model_loader` 是否还有 `no_alloc` 成员变量？
- `gguf_init_from_reader` 函数签名是否变化？

### 步骤 3：重新应用三处补丁

按上述 #1 → #2 → #3 顺序。

### 步骤 4：验证

```bash
# 用已知的 meta 文件验证
./build/bin/Release/gguf-mem.exe -m known-model.meta.gguf -c 8192
# 应该输出正常的内存估算值
```

### 步骤 5：检查 swa_full 默认值

每次更新需确认 `common/common.h` 中 `swa_full` 的默认值是否仍为 `false`，
以及 `src/llama-context.cpp` 中 `llama_context_default_params()` 的默认值。
如果新版修改了 SWA 逻辑，可能需要调整 `gguf-mem.cpp` 中的显式覆盖。

---

## 相关文件

```
tools/gguf-mem/
├── gguf-mem.cpp          # 内存估算工具（C++）
├── CMakeLists.txt        # 链接 llama-common + llama
├── README.md             # 开发指南（本文的超集）
├── USAGE.md              # 使用说明
└── PATCHES.md            # 本文 — 补丁记录 + 移植指引

tools/gguf-extract-meta/
├── gguf-extract-meta.cpp # 元数据提取工具（C++ / WinHTTP + cpp-httplib）
└── CMakeLists.txt        # 链接 ggml + winhttp
```
