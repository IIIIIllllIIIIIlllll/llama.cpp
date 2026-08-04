// Radix-select top-k, ported from vektorprime/working_ds4_speed
// (commit 5997de23, "cuda: replace cub::DeviceTopK with custom radix-select top-k kernel").
// CUB-free: works on HIP builds where GGML_CUDA_USE_CUB is disabled and TOP_K previously
// fell back to a full bitonic argsort per row (or the CPU backend for wide rows).

#include "common.cuh"
#include "top-k.cuh"

__device__ __forceinline__ uint32_t top_k_f32_ord(uint32_t bits) {
    return bits ^ ((uint32_t)((int32_t) bits >> 31) | 0x80000000u);
}

struct top_k_f_hist_l1 {
    uint32_t * __restrict__ hist;

    __device__ __forceinline__ void operator()(int, uint32_t u) const {
        atomicAdd(&hist[u >> 21], 1);
    }
};

struct top_k_f_hist_l2 {
    uint32_t * __restrict__ hist;
    const int b1;

    __device__ __forceinline__ void operator()(int, uint32_t u) const {
        if ((int) (u >> 21) == b1) {
            atomicAdd(&hist[(u >> 10) & 0x7FF], 1);
        }
    }
};

struct top_k_f_hist_l3 {
    uint32_t * __restrict__ hist;
    const int b1;
    const int b2;

    __device__ __forceinline__ void operator()(int, uint32_t u) const {
        if ((int) (u >> 21) == b1 && (int) ((u >> 10) & 0x7FF) == b2) {
            atomicAdd(&hist[u & 0x3FF], 1);
        }
    }
};

struct top_k_f_emit_above {
    int * __restrict__ dst;
    float * __restrict__ vals;
    int * __restrict__ claim;
    const int * __restrict__ idx_map;
    const int idx_base;
    const int b1;
    const int b2;
    const int b3;

    __device__ __forceinline__ void operator()(int i, uint32_t u) const {
        const uint32_t key1 = u >> 21;
        const uint32_t key2 = (u >> 10) & 0x7FF;
        const uint32_t key3 = u & 0x3FF;

        const bool above = key1 > (uint32_t) b1 || (key1 == (uint32_t) b1 && (key2 > (uint32_t) b2 || (key2 == (uint32_t) b2 && key3 > (uint32_t) b3)));

        if (above) {
            const int slot = atomicAdd(claim, 1);
            dst[slot] = (idx_map != nullptr ? idx_map[i] : i) + idx_base;
            if (vals != nullptr) {
                const uint32_t bits = (u & 0x80000000u) != 0 ? (u ^ 0x80000000u) : ~u;
                vals[slot] = __uint_as_float(bits);
            }
        }
    }
};

struct top_k_f_emit_boundary {
    int * __restrict__ dst;
    float * __restrict__ vals;
    int * __restrict__ claim;
    const int * __restrict__ idx_map;
    const int idx_base;
    const int b1;
    const int b2;
    const int b3;
    const int k;

    __device__ __forceinline__ void operator()(int i, uint32_t u) const {
        if ((int) (u >> 21) == b1 && (int) ((u >> 10) & 0x7FF) == b2 && (int) (u & 0x3FF) == b3) {
            const int slot = atomicAdd(claim, 1);
            if (slot < k) {
                dst[slot] = (idx_map != nullptr ? idx_map[i] : i) + idx_base;
                if (vals != nullptr) {
                    const uint32_t bits = (u & 0x80000000u) != 0 ? (u ^ 0x80000000u) : ~u;
                    vals[slot] = __uint_as_float(bits);
                }
            }
        }
    }
};

template <int NTHREADS, typename OP>
__device__ void top_k_scan(const float * __restrict__ src, const int ncols, OP op) {
    const int tid = threadIdx.x;

    const uintptr_t addr = (uintptr_t) src;

    if ((addr & 3) != 0) {
        for (int i = tid; i < ncols; i += NTHREADS) {
            op(i, top_k_f32_ord(__float_as_uint(src[i])));
        }
        return;
    }

    int aligned = 0;
    if ((addr & 15) != 0) {
        aligned = (int) ((16 - (int) (addr & 15)) >> 2);
        if (aligned > ncols) {
            aligned = ncols;
        }
    }

    const int tail  = (ncols - aligned) & 3;
    const int v_end = ncols - tail;

    for (int i = tid; i < aligned; i += NTHREADS) {
        op(i, top_k_f32_ord(__float_as_uint(src[i])));
    }

    for (int i = aligned + 4*tid; i + 3 < v_end; i += 4*NTHREADS) {
        const float4 v = *(const float4 *) (src + i);
        op(i + 0, top_k_f32_ord(__float_as_uint(v.x)));
        op(i + 1, top_k_f32_ord(__float_as_uint(v.y)));
        op(i + 2, top_k_f32_ord(__float_as_uint(v.z)));
        op(i + 3, top_k_f32_ord(__float_as_uint(v.w)));
    }

    for (int i = v_end + tid; i < ncols; i += NTHREADS) {
        op(i, top_k_f32_ord(__float_as_uint(src[i])));
    }
}

__device__ void top_k_select_boundary(uint32_t * __restrict__ hist, const int n_buckets, const int target, int & b, int & S) {
    const int lane  = threadIdx.x & 31;
    const int block = (n_buckets + 31) / 32;

    int partial = 0;
    for (int j = 0; j < block; ++j) {
        const int i = lane * block + j;
        if (i < n_buckets) {
            partial += (int) hist[i];
        }
    }

    const int incl = warp_prefix_inclusive_sum(partial);
    const int excl = incl - partial;

    int acc = excl;
    for (int j = 0; j < block; ++j) {
        const int i = lane * block + j;
        if (i < n_buckets) {
            const uint32_t old = hist[i];
            hist[i] = (uint32_t) acc + old;
            acc += (int) old;
        }
    }

    int found = n_buckets;
    for (int j = 0; j < block; ++j) {
        const int i = lane * block + j;
        if (i < n_buckets && (int) hist[i] >= target && i < found) {
            found = i;
        }
    }

#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        found = min(found, __shfl_xor_sync(0xffffffff, found, off, WARP_SIZE));
    }

    if (found == n_buckets) {
        found = n_buckets - 1;
    }

    b = found;
    S = (int) hist[n_buckets - 1] - (int) hist[b];
}

template <int NTHREADS>
__device__ __noinline__ void top_k_pass_hist_l1(const float * __restrict__ src, const int ncols, uint32_t * __restrict__ hist) {
    for (int i = threadIdx.x; i < 2048; i += NTHREADS) {
        hist[i] = 0;
    }
    __syncthreads();

    top_k_scan<NTHREADS>(src, ncols, top_k_f_hist_l1{ hist });
    __syncthreads();
}

template <int NTHREADS>
__device__ __noinline__ void top_k_pass_hist_l2(const float * __restrict__ src, const int ncols, const int b1, uint32_t * __restrict__ hist) {
    for (int i = threadIdx.x; i < 2048; i += NTHREADS) {
        hist[i] = 0;
    }
    __syncthreads();

    top_k_scan<NTHREADS>(src, ncols, top_k_f_hist_l2{ hist, b1 });
    __syncthreads();
}

template <int NTHREADS>
__device__ __noinline__ void top_k_pass_hist_l3(const float * __restrict__ src, const int ncols, const int b1, const int b2, uint32_t * __restrict__ hist) {
    for (int i = threadIdx.x; i < 1024; i += NTHREADS) {
        hist[i] = 0;
    }
    __syncthreads();

    top_k_scan<NTHREADS>(src, ncols, top_k_f_hist_l3{ hist, b1, b2 });
    __syncthreads();
}

template <int NTHREADS>
__device__ __noinline__ void top_k_pass_emit(
        const float * __restrict__ src,
        int * __restrict__ dst,
        float * __restrict__ vals,
        int * __restrict__ claim,
        const int * __restrict__ idx_map,
        const int idx_base,
        const int ncols,
        const int k,
        const int b1,
        const int b2,
        const int b3) {
    top_k_scan<NTHREADS>(src, ncols, top_k_f_emit_above{ dst, vals, claim, idx_map, idx_base, b1, b2, b3 });
    __syncthreads();

    top_k_scan<NTHREADS>(src, ncols, top_k_f_emit_boundary{ dst, vals, claim, idx_map, idx_base, b1, b2, b3, k });
}

template <int NTHREADS>
__device__ void top_k_radix_select(
        const float * __restrict__ src,
        const int *   __restrict__ idx_map,
        int *         __restrict__ dst,
        float *       __restrict__ vals,
        const int ncols,
        const int k,
        const int idx_base,
        uint32_t * __restrict__ hist,
        int * __restrict__ claim) {
    const int tid = threadIdx.x;

    __shared__ int s_b1, s_b2, s_b3;
    __shared__ int s_S, s_S2, s_S3;

    top_k_pass_hist_l1<NTHREADS>(src, ncols, hist);

    if (tid < 32) {
        int b1, S1;
        top_k_select_boundary(hist, 2048, ncols - k + 1, b1, S1);
        s_b1 = b1;
        s_S  = S1;
    }
    __syncthreads();

    top_k_pass_hist_l2<NTHREADS>(src, ncols, s_b1, hist);

    if (tid < 32) {
        int total2 = 0;
        for (int i = tid; i < 2048; i += 32) {
            total2 += (int) hist[i];
        }
        total2 = warp_reduce_sum(total2);

        int b2, S2;
        top_k_select_boundary(hist, 2048, total2 - (k - s_S) + 1, b2, S2);
        s_b2 = b2;
        s_S2 = S2;
    }
    __syncthreads();

    top_k_pass_hist_l3<NTHREADS>(src, ncols, s_b1, s_b2, hist);

    if (tid < 32) {
        int total3 = 0;
        for (int i = tid; i < 1024; i += 32) {
            total3 += (int) hist[i];
        }
        total3 = warp_reduce_sum(total3);

        int b3, S3;
        top_k_select_boundary(hist, 1024, total3 - (k - s_S - s_S2) + 1, b3, S3);
        s_b3 = b3;
        s_S3 = S3;
    }
    __syncthreads();

    if (tid == 0) {
        claim[0] = 0;
    }
    __syncthreads();

    top_k_pass_emit<NTHREADS>(src, dst, vals, claim, idx_map, idx_base, ncols, k, s_b1, s_b2, s_b3);
}

template <int NTHREADS>
__global__ void __launch_bounds__(512, 2) top_k_chunk_kernel(
        const float * __restrict__ src,
        float * __restrict__ vals,
        int * __restrict__ dst,
        const int ncols,
        const int k,
        const int bpr,
        const int chunk_size) {
    const int row   = blockIdx.x / bpr;
    const int chunk = blockIdx.x % bpr;
    const int idx_base = chunk * chunk_size;
    const int n_chunk  = min(chunk_size, ncols - idx_base);

    const float * src_c = src + (int64_t) row * ncols + idx_base;
    int * dst_c = dst + (int64_t) row * bpr * k + chunk * k;
    float * vals_c = vals != nullptr ? vals + (int64_t) row * bpr * k + chunk * k : nullptr;

    __shared__ uint32_t hist[2048];
    __shared__ int claim;

    top_k_radix_select<NTHREADS>(src_c, nullptr, dst_c, vals_c, n_chunk, k, idx_base, hist, &claim);
}

template <int NTHREADS>
__global__ void __launch_bounds__(512, 2) top_k_merge_kernel(
        const float * __restrict__ vals,
        const int *   __restrict__ idxs,
        int * __restrict__ dst,
        const int n_candidates,
        const int k) {
    const int row = blockIdx.x;

    __shared__ uint32_t hist[2048];
    __shared__ int claim;

    top_k_radix_select<NTHREADS>(vals + (int64_t) row * n_candidates, idxs + (int64_t) row * n_candidates,
                                 dst + (int64_t) row * k, nullptr, n_candidates, k, 0, hist, &claim);
}

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);
    const int64_t k     = dst->ne[0];

    if (k == 0) {
        return;
    }

    GGML_ASSERT(ncols >= k);

    int64_t bpr = std::min<int64_t>(128, std::max<int64_t>(1, 512 / std::max<int64_t>(nrows, 1)));
    bpr = std::min(bpr, ncols / k);

    int64_t chunk_size = (ncols + bpr - 1) / bpr;

    while (bpr > 1 && ncols - (bpr - 1) * chunk_size < k) {
        --bpr;
        chunk_size = (ncols + bpr - 1) / bpr;
    }

    constexpr int NTHREADS = 512;

    if (bpr == 1) {
        top_k_chunk_kernel<NTHREADS><<<(uint32_t) nrows, NTHREADS, 0, stream>>>(
                src0_d, nullptr, dst_d, (int) ncols, (int) k, 1, (int) chunk_size);
    } else {
        ggml_cuda_pool & pool = ctx.pool();

        ggml_cuda_pool_alloc<float> vals_alloc(pool, nrows * bpr * k);
        ggml_cuda_pool_alloc<int>   idxs_alloc(pool, nrows * bpr * k);

        top_k_chunk_kernel<NTHREADS><<<(uint32_t) (nrows * bpr), NTHREADS, 0, stream>>>(
                src0_d, vals_alloc.get(), idxs_alloc.get(), (int) ncols, (int) k, (int) bpr, (int) chunk_size);
        top_k_merge_kernel<NTHREADS><<<(uint32_t) nrows, NTHREADS, 0, stream>>>(
                vals_alloc.get(), idxs_alloc.get(), dst_d, (int) (bpr * k), (int) k);
    }
}
