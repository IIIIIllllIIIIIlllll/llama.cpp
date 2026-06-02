// gguf-mem: estimate host/device memory of a GGUF model using llama.cpp runtime.
//
// Supports all context/model parameters that affect memory:
//   -c N, --ctx-size N          context size
//   -sf 0/1, --swa-full 0/1     full-size SWA cache (default: 0)
//   -fa 0/1, --flash-attn 0/1   flash attention (affects compute buffers)
//   -b N, --batch-size N        batch size
//   -ub N, --ubatch-size N      micro-batch size
//   -np N, --parallel N         number of parallel sequences
//   --cache-type-k TYPE         KV cache key type (f16/q8_0/q4_0/...)
//   --cache-type-v TYPE         KV cache value type
//   -ngl N, --n-gpu-layers N    GPU offload layers
//   -ts F,F,... --tensor-split  per-GPU tensor split
//
// Usage:
//   gguf-mem --model model.gguf -c 8192 -sf 0
//   gguf-mem -m model.gguf -c 32768 --cache-type-k q8_0 -ngl 99

#include "llama.h"
#include "../src/llama-ext.h"

#include "arg.h"
#include "common.h"
#include "fit.h"
#include "log.h"

#include <stdio.h>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267)
#endif

int main(int argc, char ** argv) {
    common_params params;

    // Override defaults: always print, never fit
    params.fit_params       = false;
    params.fit_params_print = true;
    params.swa_full         = false;  // default: small SWA cache

    // Pre-parse --parallel / -np: this flag is registered under
    // LLAMA_EXAMPLE_PARALLEL which is not inherited via LLAMA_EXAMPLE_COMMON,
    // so common_params_parse(..., LLAMA_EXAMPLE_FIT_PARAMS) won't see it.
    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "-np") == 0 || strcmp(argv[i], "--parallel") == 0) && i + 1 < argc) {
            params.n_parallel = atoi(argv[++i]);
        }
    }

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_FIT_PARAMS)) {
        return 1;
    }

    if (params.model.path.empty()) {
        fprintf(stderr, "ERROR: --model is required\n");
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    auto mparams = common_model_params_to_llama(params);
    auto cparams = common_context_params_to_llama(params);

    common_fit_print(params.model.path.c_str(), &mparams, &cparams);

    llama_backend_free();
    return 0;
}
