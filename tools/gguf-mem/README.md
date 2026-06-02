# gguf-mem 开发指南

## 目标

从 GGUF 元数据（不读取权重）估算模型内存占用。
通过修改 llama.cpp 使其 `no_alloc` 模式能接受截断的元数据文件，
然后直接走 `llama_model_load_from_file` → `llama_get_memory_breakdown()`
这条真实路径，让所有架构的内存计算由 llama.cpp 自身完成，
工具层零架构特定逻辑。

---

## 一、为什么必须改 llama.cpp 源码

`llama_model_load_from_file(no_alloc=true)` 本意是"创建 tensor 描述符但不分配显存"，
但它在加载过程中做了两个假设：

1. **GGUF 文件结构完整**：读完 tensor info 后 seek 到 data section 起始位置，
   校验 tensor offset 连续递增——元数据文件没有权重数据，这两步会失败。

2. **tensor 数据在文件范围内**：`llama_tensor_weight` 构造时检查
   `tensor_offset + nbytes <= file_size`——元数据文件只有 ~10 MB，
   而 offset 指向 GB 级位置。

所以需要在这两处开口子：让 `no_alloc` 模式跳过这些校验。

---

## 二、llama.cpp 源码补丁（3 处）

### 补丁 1：`ggml/src/gguf.cpp` — `gguf_init_from_reader` 函数

**问题 1**：读完 tensor info 后 seek 到 data section 起始位置失败。

```cpp
// 原始代码（约第 751 行）：
if (n_tensors > 0 && !gr.seek(GGML_PAD(gr.tell(), ctx->alignment))) {
    GGML_LOG_ERROR("%s: failed to seek to beginning of data section\n", __func__);
    gguf_free(ctx);
    return nullptr;
}
ctx->offset = gr.tell();
```

```cpp
// 改为：
if (params.no_alloc) {
    // no_alloc 模式无需读 tensor data，跳过 seek
    ctx->offset = GGML_PAD(gr.tell(), ctx->alignment);
} else if (n_tensors > 0 && !gr.seek(GGML_PAD(gr.tell(), ctx->alignment))) {
    GGML_LOG_ERROR("%s: failed to seek to beginning of data section\n", __func__);
    gguf_free(ctx);
    return nullptr;
} else {
    ctx->offset = gr.tell();
}
```

**问题 2**：校验 tensor offset 连续递增失败。

```cpp
// 原始代码（约第 773 行）：
if (ti.offset != ctx->size && params.ctx != nullptr) {
    GGML_LOG_ERROR("tensor '%s' has offset %" PRIu64 ", expected %zu\n",
        ti.t.name, ti.offset, ctx->size);
    gguf_free(ctx);
    return nullptr;
}

// 改为：
if (ti.offset != ctx->size && !params.no_alloc) {
    // 同上错误信息
    gguf_free(ctx);
    return nullptr;
}
```

> 为什么原来是 `params.ctx != nullptr`？因为最初只考虑了 `gguf_init_from_file(ctx=NULL)`
> 场景。但 `llama_model_loader` 调用时 `ctx` 永远不为 NULL（它需要创建 tensor 对象），
> 所以必须改用 `!params.no_alloc` 覆盖模型加载路径。

---

### 补丁 2：`src/llama-model-loader.h` — `llama_tensor_weight` 构造函数

```cpp
// 原始签名（约第 39 行）：
llama_tensor_weight(const llama_file * file, uint16_t idx,
    const struct gguf_context * gguf_ctx, ggml_tensor * tensor)

// 改为：
llama_tensor_weight(const llama_file * file, uint16_t idx,
    const struct gguf_context * gguf_ctx, ggml_tensor * tensor,
    bool no_alloc = false)   // ← 新增参数，默认 false 保持兼容
```

构造函数内部的文件边界检查：

```cpp
// 原始：
offs = gguf_get_data_offset(gguf_ctx) + gguf_get_tensor_offset(gguf_ctx, tensor_idx);
if (offs + ggml_nbytes(tensor) < offs || offs + ggml_nbytes(tensor) > file->size()) {
    throw std::runtime_error("tensor data is not within the file bounds...");
}

// 改为：
offs = gguf_get_data_offset(gguf_ctx) + gguf_get_tensor_offset(gguf_ctx, tensor_idx);
if (!no_alloc) {
    if (offs + ggml_nbytes(tensor) < offs || offs + ggml_nbytes(tensor) > file->size()) {
        throw std::runtime_error("tensor data is not within the file bounds...");
    }
}
```

> 原理：`no_alloc` 模式下 `load_all_data()` 永远不会被调用，tensor offset 不会被实际使用，
> 所以边界检查没有意义。

---

### 补丁 3：`src/llama-model-loader.cpp` — 三处构造调用

`llama_model_loader` 构造函数中，有三处创建 `llama_tensor_weight`：

```cpp
// 约第 582 行（单文件无分片）：
weights_map.emplace(tensor_name, llama_tensor_weight(files.back().get(), 0, metadata, cur));
// 改为：
weights_map.emplace(tensor_name, llama_tensor_weight(files.back().get(), 0, metadata, cur, no_alloc));

// 约第 648 行（分片文件）：
weights_map.emplace(tensor_name, llama_tensor_weight(files.back().get(), idx, ctx_gguf.get(), cur));
// 改为：
weights_map.emplace(tensor_name, llama_tensor_weight(files.back().get(), idx, ctx_gguf.get(), cur, no_alloc));

// 约第 692 行（从 FILE* 加载）：
weights_map.emplace(tensor_name, llama_tensor_weight(files.back().get(), 0, metadata, cur));
// 改为：
weights_map.emplace(tensor_name, llama_tensor_weight(files.back().get(), 0, metadata, cur, no_alloc));
```

> `no_alloc` 是 `llama_model_loader` 构造函数的参数，在这三处都可用。

---

## 三、`gguf-mem` 工具（内存估算）

### 设计思路

不自己算——直接调 llama.cpp 的 `llama_get_memory_breakdown()`，
让所有架构的 SWA/hybrid/MLA/MoE 由 llama.cpp 自身处理。

### 关键入口

```cpp
// common_get_device_memory_data 是已有函数（common/fit.cpp:29），它做了：
//   1. llama_model_load_from_file(mparams.no_alloc=true)
//   2. llama_init_from_model(model, cparams)
//   3. llama_get_memory_breakdown(ctx)
//   4. 按 device 汇总并返回
//
// common_fit_print 是对它的薄封装（common/fit.cpp:934）

common_fit_print(path_model, &mparams, &cparams);
```

### 源码：`tools/gguf-mem/gguf-mem.cpp`

```cpp
int main(int argc, char ** argv) {
    common_params params;

    // Override defaults: always print, never fit
    params.fit_params       = false;
    params.fit_params_print = true;
    params.swa_full         = false;   // ← 核心！SWA 层用小缓存。
                                       //   llama_context_default_params() 默认 true，
                                       //   会导致 Gemma 等模型的 SWA 层分配完整的 n_ctx KV cache。
                                       //   必须显式覆盖为 false。

    // Pre-parse --parallel / -np: 该参数注册在 LLAMA_EXAMPLE_PARALLEL 下，
    // 不被 LLAMA_EXAMPLE_COMMON 继承，所以 common_params_parse(...FIT_PARAMS) 不会处理。
    // 在此手动预解析。
    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "-np") == 0 || strcmp(argv[i], "--parallel") == 0) && i + 1 < argc) {
            params.n_parallel = atoi(argv[++i]);
        }
    }

    common_init();
    common_params_parse(argc, argv, params, LLAMA_EXAMPLE_FIT_PARAMS);

    llama_backend_init();
    auto mparams = common_model_params_to_llama(params);
    auto cparams = common_context_params_to_llama(params);

    // common_fit_print 内部自动设置 no_alloc=true, use_mmap=false
    common_fit_print(params.model.path.c_str(), &mparams, &cparams);

    llama_backend_free();
}
```

### CMakeLists.txt

```cmake
# tools/gguf-mem/CMakeLists.txt
# 依赖：llama-common（提供 common_fit_print） + llama（提供模型加载和 memory_breakdown）

set(TARGET gguf-mem)

add_executable(${TARGET} gguf-mem.cpp)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
```

### 与 `llama-fit-params` 的参数兼容性

二者共用 `common_params_parse(_, _, _, LLAMA_EXAMPLE_FIT_PARAMS)`，
因此 `llama-fit-params` 的所有参数可直接传给 `gguf-mem`。

参数来源规则（`arg.cpp:1078`）：
```
参数可见 = arg.in_example(当前example) || arg.in_example(LLAMA_EXAMPLE_COMMON)
```

| 参数 | 来源 | 影响 |
|------|------|------|
| `-m` `-c` `-b` `-ub` `--swa-full` `-fa` `-ctk` `-ctv` `-ngl` `-ts` `--no-kv-offload` | `LLAMA_EXAMPLE_COMMON` | model / KV cache / compute 大小 |
| `--fit on/off` `--fit-params-print` | `LLAMA_EXAMPLE_FIT_PARAMS` | 被 `gguf-mem` 忽略（永远 print 模式） |
| `-np` / `--parallel` | `LLAMA_EXAMPLE_PARALLEL` | **gguf-mem 手动预解析支持** |

`--parallel` 是唯一的例外——它注册在 `LLAMA_EXAMPLE_PARALLEL` 下，
不被 `LLAMA_EXAMPLE_COMMON` 继承。`gguf-mem` 在 `common_params_parse`
之前手动扫描 argv 处理此参数。

### 内存计算流程

```
common_fit_print
  └─ common_get_device_memory_data(path, mparams, cparams)
       │
       ├─ llama_model_load_from_file(path, {.no_alloc=true, .use_mmap=false})
       │    ├─ gguf_init_from_file(no_alloc=true)  → 解析 header
       │    ├─ load_arch_hparams()  → 读架构特定参数（SWA pattern、hybrid interval…）
       │    ├─ load_tensors()  → 创建 ggml_tensor，分配 dummy buffer (size=0)
       │    └─ load_data()  → no_alloc=true，直接返回（√ 不读权重）
       │
       ├─ llama_init_from_model(model, cparams)
       │    ├─ 创建 memory 对象（llama_kv_cache / llama_kv_cache_iswa /
       │    │   llama_memory_hybrid / llama_memory_recurrent）
       │    ├─ graph_reserve()  → 构建推理图，scheduler 估算 compute buffer
       │    └─ backend_buf_exp_size[]  → 临时 buffer 大小
       │
       └─ llama_get_memory_breakdown(ctx)
            ├─ model.memory_breakdown()
            │    └─ ggml_backend_alloc_ctx_tensors_from_buft_size()
            │       遍历所有 tensor，按 buffer type 汇总 nbytes
            ├─ memory->memory_breakdown()  → KV cache / recurrent state
            └─ compute: backend_buf_exp_size[i]
```

---

## 四、`gguf-extract-meta` 工具（元数据提取）

### 功能

从完整 GGUF 文件（或 URL）提取仅包含 header 的 meta 文件。
meta 文件 tensor offset 连续正确，结构完整，可以被 `llama_model_load_from_file` 接受。

### 为什么 URL 下载可行

GGUF 是前缀式二进制格式：

```
┌──────────────────────────────────────────────┐ offset 0
│ Magic "GGUF"        (4 bytes)                │
│ Version             (4 bytes)                │
│ n_tensors           (8 bytes)                │
│ n_kv                (8 bytes)                │
├──────────────────────────────────────────────┤
│ KV pair 0: key(str) + type(int32) + value    │  ← 元数据（架构名、超参、tokenizer）
│ KV pair 1: ...                               │
│ ...（57~70 条）                               │
├──────────────────────────────────────────────┤
│ Tensor 0: name(str) + n_dims + dims[] + type + offset │
│ Tensor 1: ...                                │
│ ...（290~833 条）                              │
├──────────────────────────────────────────────┤
│ [对齐填充到 32 字节]                           │
├══════════════════════════════════════════════┤ ← data section
│ Tensor 0 权重数据                             │  ← 从不下载
│ Tensor 1 权重数据                             │
│ ...（几百 GB）                                 │
└──────────────────────────────────────────────┘
```

**关键**：header 在文件开头，大小与模型总参数无关，只取决于 tensor 数量。
对任何规模的模型都是 **~1-20 MB**。

### 渐进式 Range 下载

```
第 1 次: Range: bytes=0-65535     (64 KiB)  → gguf_init_from_buffer → NULL（header 不完整）
第 2 次: Range: bytes=0-131071    (128 KiB) → NULL
第 3 次: Range: bytes=0-262143    (256 KiB) → NULL
...
第 8 次: Range: bytes=0-8388607   (8 MiB)   → 成功（所有 tensor info 已读入）
```

每次翻倍，最多 32 MiB。

### C++ 版：`tools/gguf-extract-meta/gguf-extract-meta.cpp`

```cpp
// Windows：WinHTTP API（零外部依赖）
// 非 Windows：当前 stub，扩展时用 libcurl 或平台 HTTP

#ifdef _WIN32
#include <windows.h>
#include <winhttp.h>
#pragma comment(lib, "winhttp.lib")

static bool winhttp_range_download(const std::string & url,
                                   int64_t range_start, int64_t range_end,
                                   std::vector<uint8_t> & out_body, int & out_status) {
    // 1. WinHttpOpen → WinHttpConnect → WinHttpOpenRequest
    // 2. WinHttpSetOption(REDIRECT_POLICY_ALWAYS)  // 跟随 302
    // 3. WinHttpSendRequest("Range: bytes=A-B")
    // 4. WinHttpReceiveResponse
    // 5. 循环 WinHttpQueryDataAvailable / WinHttpReadData
}
#endif
```

### CMakeLists.txt

```cmake
# tools/gguf-extract-meta/CMakeLists.txt
# 依赖：ggml（提供 gguf_init_from_file / gguf_write_to_file）
# Windows：额外链接 winhttp（HTTPS Range 下载，零外部依赖）

set(TARGET gguf-extract-meta)

add_executable(${TARGET} gguf-extract-meta.cpp)
target_link_libraries(${TARGET} PRIVATE ggml)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if (WIN32)
    target_link_libraries(${TARGET} PRIVATE winhttp)
endif()

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
```

### Java 版：`tools/gguf-extract-meta-java/GGufMetaExtractor.java`

```java
// JDK 8+, 零依赖，单文件
// java GGufMetaExtractor.java <URL|file> [output]

// 核心类：
class GGufReader  { byte[] data; int pos; ... }  // 小端二进制读取
class GGufWriter  { ByteArrayOutputStream ... }   // 小端二进制写入

// 渐进下载：
byte[] downloadHeader(String url) {
    for (long chunk = 65536; chunk <= 33554432; chunk *= 2) {
        HttpURLConnection conn = (HttpURLConnection) new URL(url).openConnection();
        conn.setRequestProperty("Range", "bytes=0-" + (chunk - 1));
        conn.setInstanceFollowRedirects(true);
        byte[] body = readAllBytes(conn.getInputStream());
        if (validateHeader(body)) return body;   // 成功则返回
    }
}

// 写出 meta 文件：
void writeMetaFile(byte[] data, String outPath) {
    // 1. 读原始 header → 2. 逐字节拷贝 KV pairs
    // 3. 重算 tensor offset 为连续递增（从 0 开始）
    // 4. 写对齐填充，不写 tensor data
}
```

---

## 五、移植检查清单

llama.cpp 发新版后，按以下步骤重建：

1. **重新应用三个源码补丁**（第二部分）
   - `ggml/src/gguf.cpp`：两处条件分支
   - `src/llama-model-loader.h`：`llama_tensor_weight` 加 `no_alloc` 参数
   - `src/llama-model-loader.cpp`：三处传参

2. **检查 `swa_full` 的默认值和逻辑**
   - `common/common.h` ~545 行：`bool swa_full = false`
   - `src/llama-context.cpp` ~3365 行：`llama_context_default_params()` 中的默认值
   - 如果新版改了默认值或 ISWA 逻辑，可能需要调整 `gguf-mem.cpp` 中的显式覆盖

3. **检查 `common_fit_print` 接口是否变化**
   - `common/fit.cpp:934` 和 `common/fit.h:32`
   - 它内部调用 `common_get_device_memory_data`，后者设置 `no_alloc=true`

4. **编译** → 见第六节编译指南

5. **验证**：用已知模型对比上次结果
   ```bash
   ./build/bin/Release/gguf-mem.exe -m known-model.meta.gguf -c 8192
   ```

6. **Java 版不需要移植**：纯 Java 实现，不依赖 llama.cpp。
   仅当 GGUF 格式版本升级时才需要更新 `GGUF_MAGIC` 常量。

---

## 六、编译指南

### 前提

- CMake ≥ 3.14
- C++17 编译器（MSVC 2019+ / GCC 9+ / Clang 10+）
- JDK 8+（仅 Java 版 extract-meta 需要）

### 注册到构建系统

在 `tools/CMakeLists.txt` 中，`add_subdirectory(fit-params)` 之后插入：

```cmake
    add_subdirectory(fit-params)
    add_subdirectory(gguf-extract-meta)    # ← 新增
    add_subdirectory(gguf-mem)             # ← 新增
    add_subdirectory(results)
```

完整文件结构（第 15-45 行）：
```cmake
if (EMSCRIPTEN)
else()
    add_subdirectory(batched-bench)
    add_subdirectory(gguf-split)
    # ... 其他工具 ...
    add_subdirectory(fit-params)
    add_subdirectory(gguf-extract-meta)    # GGUF 元数据提取
    add_subdirectory(gguf-mem)             # 内存估算
    add_subdirectory(results)
endif()
```

### CPU-only 编译（Windows / MSVC）

```powershell
# 配置（首次）
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
- 以及依赖 DLL（`ggml.dll`, `ggml-base.dll`, `ggml-cpu.dll`, `llama.dll`, `llama-common.dll`）

### CPU-only 编译（Linux / GCC 或 Clang）

```bash
cmake -B build \
    -DGGML_CPU=ON -DGGML_CUDA=OFF \
    -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TOOLS=ON

cmake --build build --config Release -j$(nproc) --target gguf-mem gguf-extract-meta
```

### 带 CUDA 支持的编译

如果需要估算 GPU 显存占用：

```bash
cmake -B build \
    -DGGML_CUDA=ON \
    -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TOOLS=ON

cmake --build build --config Release -j$(nproc) --target gguf-mem gguf-extract-meta
```

`gguf-mem` 会输出每个 GPU 的显存分配（model/context/compute），
格式为 `DeviceName model_mib context_mib compute_mib`。

### Java 版编译

```bash
cd tools/gguf-extract-meta-java
javac GGufMetaExtractor.java
```

JDK 11+ 也可直接运行源码：
```bash
java GGufMetaExtractor.java <URL> [output]
```

### 验证编译结果

```bash
# C++ 工具
./build/bin/Release/gguf-extract-meta.exe
# 输出: Usage: gguf-extract-meta.exe <input.gguf | https://.../model.gguf> [output.meta.gguf]

./build/bin/Release/gguf-mem.exe --help
# 输出: 完整参数列表（与 llama.cpp 其他工具一致）

# Java 工具
cd tools/gguf-extract-meta-java
java GGufMetaExtractor
# 输出: Usage: java GGufMetaExtractor.java <input.gguf | https://.../model.gguf> [output.meta.gguf]
```

---

## 七、工具文件清单

```
tools/
├── gguf-mem/
│   ├── gguf-mem.cpp          # 内存估算（C++ / llama runtime）
│   ├── CMakeLists.txt        # 链接 llama-common + llama
│   └── README.md             # 本文档
│
├── gguf-extract-meta/
│   ├── gguf-extract-meta.cpp # 元数据提取（C++ / WinHTTP）
│   └── CMakeLists.txt        # 链接 ggml + winhttp
│
└── gguf-extract-meta-java/
    └── GGufMetaExtractor.java # 元数据提取（Java / 跨平台）

tools/CMakeLists.txt          # 注册两个 C++ 工具
```
