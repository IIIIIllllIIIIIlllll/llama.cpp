#include "common.cuh"
#include "mma.cuh"
#include "fattn-common.cuh"

using namespace ggml_cuda_mma;

// RDNA3-only WMMA flash-attention kernel for DKQ = DV = 512 with V being a view of K
// (DeepSeek-V4 MLA prefill shapes: large GQA ratio, f16 mask required, K/V padded to FATTN_KQ_STRIDE).
//
// The generic MMA kernel in fattn-mma-f16.cuh cannot handle DKQ = 512 on AMD WMMA: every warp would
// need a full DV-wide VKQ accumulator (512 floats / 128 registers per thread) and spills.
// This kernel avoids the problem by splitting the warps along DV for the P*V phase:
//
//   - The block computes ncols Q columns x nbatch_fa KV rows per iteration.
//   - Phase 1 (Q*K^T): warps are grouped as cg = threadIdx.y/np (16 Q columns per group) x
//     kvs = threadIdx.y%np. The np warps of a group each compute a partial 16x16 KQ tile over a
//     DKQ/np slice of the contraction dimension. The partial tiles are exchanged via shared memory
//     and summed, then a single warp per group applies the mask, computes the block-wide online
//     softmax (one maximum per KV block, no per-warp maxima) and writes P as half2 to shared memory
//     in a layout that load_ldmatrix can read directly as a B tile.
//   - Phase 2 (P*V): the same warp now owns a DV/np slice of the output for its Q column group.
//     The B tile is P from shared memory, the A tiles are the DV slices of the SAME staged K data
//     (K == V) read via load_ldmatrix_trans. The VKQ accumulator per thread is only
//     (DV/np)/16 * 8 floats (128 for np = 2). Before accumulating, the accumulator is rescaled by
//     expf(KQ_max_prev - KQ_max_new), which is mathematically equivalent to the np end-combine of
//     the generic kernel.
//   - Phase 3: the DV slices of the warps do not overlap, so no weighted combine is needed;
//     every warp divides its slice by the rowsum and writes it straight to dstk.
//
// Grid: blockIdx.x enumerates all output tiles (jt, zt_gqa, z_KV, sequence); every block loops over
// the entire KV cache (no stream-k / fixup support).
//
// Configurations (select at compile time with GGML_CUDA_FATTN_D512_RDNA_CONFIG):
//   R1 (default): ncols = 32 (ncols1 = 4, ncols2 = 8), nwarps = 4, np = 2. ~53 KiB shared, 1 block/SM.
//   R2:           ncols = 16 (ncols1 = 2, ncols2 = 8), nwarps = 2, np = 2. ~35 KiB shared.

#define GGML_CUDA_FATTN_D512_RDNA_R1 1
#define GGML_CUDA_FATTN_D512_RDNA_R2 2

#ifndef GGML_CUDA_FATTN_D512_RDNA_CONFIG
#define GGML_CUDA_FATTN_D512_RDNA_CONFIG GGML_CUDA_FATTN_D512_RDNA_R1
#endif // GGML_CUDA_FATTN_D512_RDNA_CONFIG

template<int ncols1, int ncols2, int nwarps>
__launch_bounds__(nwarps*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void flash_attn_ext_d512_rdna(
        const char * Q_ptr,
        const char * K_ptr,
        const char * V_ptr,
        const char * mask_ptr,
        const char * sinks_ptr,
        const int  * KV_max_ptr,
        float      * dst_ptr,
        float2     * dst_meta_ptr,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33) {
    ggml_cuda_pdl_sync();
#if defined(FLASH_ATTN_AVAILABLE) && defined(AMD_WMMA_AVAILABLE) && defined(RDNA3)
    constexpr int DKQ = 512;
    constexpr int DV  = 512;

    using T_A_KQ  = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // K rows x DKQ, row-major
    using T_B_KQ  = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // Q cols x DKQ, column-major
    using T_C_KQ  = tile<16, 16, float, DATA_LAYOUT_I_MAJOR>;          // KQ: i = Q col, j = KV row
    using T_A_VKQ = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // DV slice x KV, transposed load
    using T_B_VKQ = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // P: Q cols x KV, column-major
    using T_C_VKQ = tile<16, 16, float, DATA_LAYOUT_I_MAJOR>;          // VKQ: i = Q col, j = DV

    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int ncols     = ncols1 * ncols2;
    constexpr int nbatch_fa = 16;                                      // KV rows per iteration
    constexpr int cg_count  = ncols / 16;                              // 16-column Q groups per block
    constexpr int np        = nwarps / cg_count;                       // warps per Q group: DKQ split in phase 1, DV split in phase 2
    constexpr int dv_slice  = DV / np;                                 // DV columns per warp in phase 2
    constexpr int dv_tiles  = dv_slice / 16;

    static_assert(ncols  % 16 == 0,             "bad ncols");
    static_assert(nwarps % cg_count == 0,       "bad nwarps");
    static_assert(np == 1 || np == 2 || np == 4, "bad np");
    static_assert(ncols2 > 1,                   "kernel requires GQA tiling / a mask");

    constexpr int stride_tile_Q = DKQ/2 + 4;
    constexpr int stride_tile_K = DKQ/2 + 4;
    constexpr int stride_tile_P = nbatch_fa/2 + 4;

    extern __shared__ half2 tile_Q[];
    half2  * tile_K     = tile_Q + ncols*stride_tile_Q;
    half2  * tile_P     = tile_K + nbatch_fa*stride_tile_K;
    half   * tile_mask  = (half *) (tile_P + ncols*stride_tile_P);
    float  * KQ_partial = (float  *) (tile_mask + ncols1*(nbatch_fa + 8)); // (np-1)*cg_count partial 16x16 KQ tiles
    float2 * KQ_meta    = (float2 *) (KQ_partial + (np > 1 ? (np-1)*cg_count*256 : 0)); // per column: running max, rowsum
    float  * KQ_fscale  = (float  *) (KQ_meta + ncols);                    // per column: rescale factor for phase 2

    const char * GGML_CUDA_RESTRICT Q     = Q_ptr;
    const char * GGML_CUDA_RESTRICT K     = K_ptr;
    const char * GGML_CUDA_RESTRICT mask  = mask_ptr;
    const char * GGML_CUDA_RESTRICT sinks = sinks_ptr;
    float      * GGML_CUDA_RESTRICT dst   = dst_ptr;

    GGML_UNUSED_VARS(V_ptr, KV_max_ptr, dst_meta_ptr, max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne10, ne13, nb21, nb22, nb23, ne31, ne32, nb32, ne03);

    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.

    const int stride_Q1   = nb01 / sizeof(float2);
    const int stride_Q2   = nb02 / sizeof(float2);
    const int stride_K    = nb11 / sizeof(half2);
    const int stride_mask = nb31 / sizeof(half);

    const int iter_k     = ne11 / nbatch_fa; // Dispatch guarantees ne11 % FATTN_KQ_STRIDE == 0.
    const int iter_j     = (ne01.z    + ncols1 - 1) / ncols1;
    const int iter_z_gqa = (gqa_ratio + ncols2 - 1) / ncols2;

    // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index.
    int bx = blockIdx.x;
    const int jt       = bx % iter_j;     bx /= iter_j;
    const int zt_gqa   = bx % iter_z_gqa; bx /= iter_z_gqa;
    const int z_KV     = bx % ne12;
    const int sequence = bx / ne12;

    const int zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2; // Global Q head start index.

    const float2 * Q_f2   = (const float2 *) (Q + nb03*sequence + nb02*zt_Q);
    const half2  * K_h2   = (const half2  *) (K + nb13*sequence + nb12*z_KV);
    const half   * mask_h = (const half *) (mask + nb33*(sequence % ne33));
    float2       * dstk   = ((float2 *) dst) + (sequence*ne01.z*ne02 + zt_Q) * (DV/2);

    const int cg  = threadIdx.y / np; // Q column group of this warp.
    const int kvs = threadIdx.y % np; // DKQ slice in phase 1, DV slice in phase 2.

    // Load Q data into tile_Q, multiplied by scale (Q_in_reg = false).
    // The loading is done with decreasing granularity for D for better memory bandwidth.
    const half2 scale_h2 = make_half2(scale, scale);
#pragma unroll
    for (int stride_k : {warp_size, warp_size/2, warp_size/4, warp_size/8}) {
        const int k0_start  = stride_k == warp_size ? 0 : DKQ/2 - (DKQ/2) % (2*stride_k);
        const int k0_stop   =                             DKQ/2 - (DKQ/2) % (1*stride_k);
        const int stride_jc = warp_size / stride_k;

        if (k0_start == k0_stop) {
            continue;
        }

#pragma unroll
        for (int jc0 = 0; jc0 < ncols; jc0 += nwarps*stride_jc) {
            const int jc = jc0 + threadIdx.y*stride_jc + (stride_k == warp_size ? 0 : threadIdx.x / stride_k);

            if (jc0 + nwarps*stride_jc > ncols && jc >= ncols) {
                break;
            }

            const int j = jc / ncols2;
            const int c = jc % ncols2;

            if ((ncols1 == 1 || jt*ncols1 + j < int(ne01.z)) && (ncols2 == 1 || zt_gqa*ncols2 + c < gqa_ratio)) {
#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == warp_size ? threadIdx.x : threadIdx.x % stride_k);

                    const float2 tmp = Q_f2[(jt*ncols1 + j)*stride_Q1 + c*stride_Q2 + k];
                    tile_Q[jc*stride_tile_Q + k] = scale_h2 * make_half2(tmp.x, tmp.y);
                }
            } else {
#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == warp_size ? threadIdx.x : threadIdx.x % stride_k);

                    tile_Q[jc*stride_tile_Q + k] = make_half2(0.0f, 0.0f);
                }
            }
        }
    }

    // Initialize the per-column running max / rowsum:
    if (threadIdx.y == 0 && threadIdx.x < ncols) {
        KQ_meta[threadIdx.x] = make_float2(-FLT_MAX/2.0f, 0.0f);
    }

    __syncthreads();

    T_C_VKQ VKQ_C[dv_tiles]; // Accumulator for the DV slice of this warp, zero-initialized.

    for (int kb0 = 0; kb0 < iter_k; ++kb0) {
        const int k_VKQ_0 = kb0*nbatch_fa;

        // Stage K (== V) rows and the mask into shared memory:
        constexpr bool use_cp_async = false;
        constexpr bool oob_check    = false;
        flash_attn_ext_f16_load_tile<stride_tile_K, nwarps, nbatch_fa, use_cp_async, oob_check>
            (K_h2 + int64_t(k_VKQ_0)*stride_K, tile_K, DKQ/2, stride_K, nbatch_fa);
        flash_attn_ext_f16_load_mask<ncols1, nwarps, nbatch_fa, use_cp_async, oob_check>
            (mask_h + k_VKQ_0, tile_mask, stride_mask, nbatch_fa, jt*ncols1, ne01);
        __syncthreads();

        // Phase 1: Q*K^T. Every warp computes a partial 16x16 KQ tile for its Q column group,
        //     the np warps of a group split the DKQ contraction dimension.
        T_C_KQ KQ_C[1];
        constexpr int ksteps = DKQ/(2*T_A_KQ::J)/np;
        static_assert(ksteps*np*T_A_KQ::J == DKQ/2, "bad ksteps");
#pragma unroll
        for (int ks = 0; ks < ksteps; ++ks) {
            const int k_KQ_0 = (kvs*ksteps + ks)*T_A_KQ::J;

            T_B_KQ Q_B;
            load_ldmatrix(Q_B, tile_Q + cg*(T_B_KQ::I*stride_tile_Q) + k_KQ_0, stride_tile_Q);

            T_A_KQ K_A;
            load_ldmatrix(K_A, tile_K + k_KQ_0, stride_tile_K);

            mma(KQ_C[0], K_A, Q_B);
        }

        // Exchange the partial KQ tiles via shared memory and sum them up.
        // Only the kvs == 0 warp of each group needs the full tile (for the softmax and P).
        if constexpr (np > 1) {
            if (kvs > 0) {
                float * buf = KQ_partial + ((kvs - 1)*cg_count + cg)*256;
#pragma unroll
                for (int l = 0; l < T_C_KQ::ne; ++l) {
                    buf[T_C_KQ::get_i(l)*16 + T_C_KQ::get_j(l)] = KQ_C[0].x[l];
                }
            }
            __syncthreads();
            if (kvs == 0) {
#pragma unroll
                for (int p = 0; p < np - 1; ++p) {
                    const float * buf = KQ_partial + (p*cg_count + cg)*256;
#pragma unroll
                    for (int l = 0; l < T_C_KQ::ne; ++l) {
                        KQ_C[0].x[l] += buf[T_C_KQ::get_i(l)*16 + T_C_KQ::get_j(l)];
                    }
                }
            }
        }

        // Softmax with a single block-wide maximum per KV block and write P to shared memory.
        // Only the kvs == 0 warp of each group is active, its 2 threads per Q column hold the full
        //     16 KV rows of the column (j = 2*l + threadIdx.x/16 covers rows 0-15).
        if (kvs == 0) {
            const int q = cg*16 + threadIdx.x % 16;
            const int j = q / ncols2;

            // Add the mask (slope == 1.0f for ncols2 > 1):
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                const int i = T_C_KQ::get_j(l);
                KQ_C[0].x[l] += __half2float(tile_mask[j*(nbatch_fa + 8) + i]);
            }

            float2 meta = KQ_meta[q]; // x = running max, y = running rowsum.

            float KQ_max_new = meta.x;
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                KQ_max_new = fmaxf(KQ_max_new, KQ_C[0].x[l] + FATTN_KQ_MAX_OFFSET);
            }
            // Values per KQ column are spread across 2 threads:
            KQ_max_new = fmaxf(KQ_max_new, __shfl_xor_sync(0xFFFFFFFF, KQ_max_new, 16, warp_size));

            const float KQ_max_diff = meta.x - KQ_max_new;
            float KQ_max_scale = expf(KQ_max_diff);
            *((uint32_t *) &KQ_max_scale) *= KQ_max_diff >= SOFTMAX_FTZ_THRESHOLD;

            float KQ_rowsum_add = 0.0f;
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                KQ_C[0].x[l] = expf(KQ_C[0].x[l] - KQ_max_new);
                KQ_rowsum_add += KQ_C[0].x[l];
            }
            KQ_rowsum_add += __shfl_xor_sync(0xFFFFFFFF, KQ_rowsum_add, 16, warp_size);

            // Convert the KQ C tile to a B tile (P) and store it to shared memory in row-major
            //     [Q col][KV half2] layout so phase 2 can read it back with load_ldmatrix.
            // After the shuffle inside get_half2 both threads of a pair hold the same half2 values.
            const T_B_VKQ P_B = get_half2(KQ_C[0]);
            if (threadIdx.x < 16) {
#pragma unroll
                for (int l = 0; l < T_B_VKQ::ne; ++l) {
                    tile_P[q*stride_tile_P + l] = P_B.x[l];
                }
                KQ_meta[q]   = make_float2(KQ_max_new, KQ_max_scale*meta.y + KQ_rowsum_add);
                KQ_fscale[q] = KQ_max_scale;
            }
        }
        __syncthreads();

        // Phase 2: P*V with V == K reused from shared memory.
        // Every warp owns a dv_slice-wide slice of DV for its Q column group.
        {
            const int q = cg*16 + threadIdx.x % 16;
            const float f = KQ_fscale[q]; // Same Q column for all l, uniform per thread.

            // Online rescaling of the accumulator, equivalent to the np end-combine of the generic kernel:
#pragma unroll
            for (int t = 0; t < dv_tiles; ++t) {
#pragma unroll
                for (int l = 0; l < T_C_VKQ::ne; ++l) {
                    VKQ_C[t].x[l] *= f;
                }
            }

            T_B_VKQ P_B;
            load_ldmatrix(P_B, tile_P + (cg*16)*stride_tile_P, stride_tile_P);

#pragma unroll
            for (int t = 0; t < dv_tiles; ++t) {
                T_A_VKQ V_A; // Transposed in SRAM but not in registers, gets transposed on load.
                load_ldmatrix_trans(V_A, tile_K + kvs*(dv_slice/2) + t*(T_A_VKQ::I/2), stride_tile_K);
                mma(VKQ_C[t], V_A, P_B);
            }
        }
        __syncthreads();
    }

    // Phase 3: the DV slices of the warps do not overlap, no weighted combine is needed.
    // Every thread owns one Q column (threadIdx.x % 16) of its warp's group and writes
    //     dv_tiles*8 float2 values; the 2 threads of a column pair hold DV offsets 2*l and 2*l+1.
    {
        const int q = cg*16 + threadIdx.x % 16;
        const int j_dst = q / ncols2;
        const int c_dst = q % ncols2;

        const bool valid = (ncols1 == 1 || jt*ncols1 + j_dst < int(ne01.z)) &&
                           (ncols2 == 1 || zt_gqa*ncols2 + c_dst < gqa_ratio);

        // Attention sink: the sink logit acts like an extra key position that only contributes to the
        //     softmax denominator. Fold it into the per-column max/rowsum once per block (every block
        //     loops over the entire KV cache, so unlike the tile kernel no blockIdx.y == 0 restriction
        //     is needed) and rescale the accumulator like an extra online softmax step. The 2 threads
        //     of a column pair redundantly compute the same correction, no shared memory is written.
        float rowsum = KQ_meta[q].y;
        if (sinks) {
            const float2 meta = KQ_meta[q];
            const float sink  = ncols2 == 1 || zt_gqa*ncols2 + c_dst < gqa_ratio ?
                ((const float *) sinks)[zt_Q + c_dst] : -FLT_MAX/2.0f;

            const float KQ_max_new   = fmaxf(meta.x, sink);
            const float KQ_max_scale = expf(meta.x - KQ_max_new);
            rowsum = rowsum*KQ_max_scale + expf(sink - KQ_max_new);

#pragma unroll
            for (int t = 0; t < dv_tiles; ++t) {
#pragma unroll
                for (int l = 0; l < T_C_VKQ::ne; ++l) {
                    VKQ_C[t].x[l] *= KQ_max_scale;
                }
            }
        }

        float2 * dstk_col = dstk + ((jt*ncols1 + j_dst)*ne02 + c_dst)*(DV/2) + kvs*(dv_slice/2);

#pragma unroll
        for (int t = 0; t < dv_tiles; ++t) {
#pragma unroll
            for (int l = 0; l < T_C_VKQ::ne; ++l) {
                const float v0 =                 VKQ_C[t].x[l]              / rowsum;
                const float v1 = __shfl_xor_sync(0xFFFFFFFF, VKQ_C[t].x[l], 16, warp_size) / rowsum;
                if (valid && threadIdx.x < 16) {
                    dstk_col[t*(T_A_VKQ::I/2) + l] = make_float2(v0, v1);
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(Q_ptr, K_ptr, V_ptr, mask_ptr, sinks_ptr, KV_max_ptr, dst_ptr, dst_meta_ptr, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
        ne31, ne32, ne33,
              nb31, nb32, nb33);
    NO_DEVICE_CODE;
#endif // defined(FLASH_ATTN_AVAILABLE) && defined(AMD_WMMA_AVAILABLE) && defined(RDNA3)
}

template <int ncols1, int ncols2, int nwarps>
static void ggml_cuda_flash_attn_ext_mma_d512_rdna_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    constexpr int DKQ = 512;
    constexpr int ncols     = ncols1 * ncols2;
    constexpr int nbatch_fa = 16;
    constexpr int cg_count  = ncols / 16;
    constexpr int np        = nwarps / cg_count;

    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_ASSERT(Q->nb[0] == ggml_element_size(Q));
    GGML_ASSERT(K->nb[0] == ggml_element_size(K));

    GGML_ASSERT(mask && mask->type == GGML_TYPE_F16);
    GGML_ASSERT(K->ne[1] % nbatch_fa == 0); // Guaranteed by the dispatch logic in fattn.cu (FATTN_KQ_STRIDE).

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t main_stream = ctx.stream();
    const int id = ggml_cuda_get_device();

    // V is a view of K and never read by the kernel, only K may need conversion to f16:
    const ggml_cuda_flash_attn_ext_f16_extra_data f16_extra =
        ggml_cuda_flash_attn_ext_get_f16_extra_data(dst, true, false);

    const char * K_data = (const char *) K->data;
    size_t nb11 = K->nb[1];
    size_t nb12 = K->nb[2];
    size_t nb13 = K->nb[3];

    if (K->type != GGML_TYPE_F16) {
        const size_t bs = ggml_blck_size(K->type);
        const size_t ts = ggml_type_size(K->type);

        GGML_ASSERT(f16_extra.K != 0);
        half * K_f16 = (half *) f16_extra.K;
        if (ggml_is_contiguously_allocated(K)) {
            to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(K->type);
            to_fp16(K_data, K_f16, ggml_nelements(K), main_stream);

            nb11 = nb11*bs*sizeof(half)/ts;
            nb12 = nb12*bs*sizeof(half)/ts;
            nb13 = nb13*bs*sizeof(half)/ts;
        } else {
            GGML_ASSERT(K->nb[0] == ts);
            to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(K->type);
            const int64_t s01 = nb11 / ts;
            const int64_t s02 = nb12 / ts;
            const int64_t s03 = nb13 / ts;
            to_fp16(K_data, K_f16, K->ne[0], K->ne[1], K->ne[2], K->ne[3], s01, s02, s03, main_stream);

            nb11 = K->ne[0] * sizeof(half);
            nb12 = K->ne[1] * nb11;
            nb13 = K->ne[2] * nb12;
        }
        K_data = (char *) K_f16;
    }

    constexpr int stride_tile_Q = DKQ/2 + 4;
    constexpr int stride_tile_K = DKQ/2 + 4;
    constexpr int stride_tile_P = nbatch_fa/2 + 4;

    const size_t nbytes_shared =
          size_t(ncols)                           * stride_tile_Q   * sizeof(half2)
        + size_t(nbatch_fa)                       * stride_tile_K   * sizeof(half2)
        + size_t(ncols)                           * stride_tile_P   * sizeof(half2)
        + size_t(ncols1)                          * (nbatch_fa + 8) * sizeof(half)
        + size_t(np > 1 ? (np-1)*cg_count*256 : 0)                  * sizeof(float)
        + size_t(ncols)                                           * sizeof(float2)
        + size_t(ncols)                                           * sizeof(float);

    fattn_kernel_t fattn_kernel = flash_attn_ext_d512_rdna<ncols1, ncols2, nwarps>;

#if !defined(GGML_USE_MUSA)
    static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
    if (!shared_memory_limit_raised[id]) {
#if defined(GGML_USE_HIP)
        using fattn_kernel_ptr_t = const void*;
#else
        using fattn_kernel_ptr_t = fattn_kernel_t;
#endif // defined(GGML_USE_HIP)
        CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared));
        shared_memory_limit_raised[id] = true;
    }
#endif // !defined(GGML_USE_MUSA)

    const int gqa_ratio    = Q->ne[2] / K->ne[2];
    const int ntiles_x     = (Q->ne[1] + ncols1 - 1) / ncols1;
    const int ntiles_z_gqa = (gqa_ratio + ncols2 - 1) / ncols2;
    const int ntiles_dst   = ntiles_x * ntiles_z_gqa * K->ne[2] * Q->ne[3];

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    const uint3 ne01 = init_fastdiv_values(Q->ne[1]);

    const dim3 blocks_num(ntiles_dst, 1, 1);
    const dim3 block_dim(WARP_SIZE, nwarps, 1);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num, block_dim, nbytes_shared, main_stream);
    ggml_cuda_kernel_launch(fattn_kernel, launch_params,
        (const char *) Q->data,
        K_data,
        nullptr,                    // V: unused, V is a view of K
        (const char *) mask->data,
        sinks ? (const char *) sinks->data : nullptr,
        nullptr,                    // KV_max: unsupported
        (float *) dst->data,
        nullptr,                    // dst_meta: no stream-k / fixup
        scale, 0.0f, 0.0f, 0.0f, 0, 0.0f,
        Q->ne[0], ne01,     Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
        K->ne[0], K->ne[1], K->ne[2], K->ne[3], nb11, nb12, nb13,
        0, 0, 0,
        mask->ne[1], mask->ne[2], mask->ne[3],
        mask->nb[1], mask->nb[2], mask->nb[3]);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_flash_attn_ext_mma_d512_rdna(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
#if GGML_CUDA_FATTN_D512_RDNA_CONFIG == GGML_CUDA_FATTN_D512_RDNA_R2
    ggml_cuda_flash_attn_ext_mma_d512_rdna_case<2, 8, 2>(ctx, dst); // R2: ncols = 16, nwarps = 2
#else
    ggml_cuda_flash_attn_ext_mma_d512_rdna_case<4, 8, 4>(ctx, dst); // R1: ncols = 32, nwarps = 4
#endif // GGML_CUDA_FATTN_D512_RDNA_CONFIG == GGML_CUDA_FATTN_D512_RDNA_R2
}
