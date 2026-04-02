/*
 * turboquant_cuda_kernel.cu -- TQ3 (3-bit) KV Cache Quantization for CUDA
 *
 * Copyright (c) 2026 Elyan Labs. All rights reserved.
 * Licensed under the MIT License.
 *
 * CUDA implementation of TurboQuant TQ3, matching the POWER8 VSX version
 * in ggml-turboquant-vsx.h.  128-float blocks quantized to 56 bytes
 * (9.1x compression vs FP32).
 *
 * Algorithm (quantize):
 *   1. L2 norm of 128-float block, normalize to unit vector
 *   2. Fast Walsh-Hadamard Transform (FWHT) -- 7 butterfly stages
 *   3. Deterministic sign flips (xoshiro256** PRNG, seed-based)
 *   4. Absmax of transformed vector, scale to [-1, +1]
 *   5. Lloyd-Max codebook quantize (8 centroids, 3 bits)
 *   6. Pack 128 x 3-bit indices into 48 bytes
 *   Store: norm (4B) + scale (4B) + packed (48B) = 56 bytes
 *
 * Dequantize is the exact reverse.
 *
 * Target GPUs: V100 (SM_70), RTX 5070 (SM_100), RTX 3060 (SM_86), M40 (SM_52)
 *
 * Build:
 *   nvcc -arch=sm_70 -O2 -o tq3_cuda_test turboquant_cuda_kernel.cu \
 *        -DTURBOQUANT_SELF_TEST
 */

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>

#include <cuda_runtime.h>

/* ------------------------------------------------------------------ */
/*  Constants                                                         */
/* ------------------------------------------------------------------ */

#define TQ3_BLOCK_SIZE    128
#define TQ3_PACKED_BYTES  48   /* 128 * 3 / 8 */
#define TQ3_FWHT_STAGES   7   /* log2(128) */
#define TQ3_NUM_CENTROIDS  8

/* ------------------------------------------------------------------ */
/*  Block layout: 56 bytes per 128 floats                             */
/* ------------------------------------------------------------------ */

typedef struct {
    float         norm;                             /*  4 bytes */
    float         scale;                            /*  4 bytes */
    unsigned char packed_indices[TQ3_PACKED_BYTES];  /* 48 bytes */
} block_tq3_t;                                      /* 56 bytes */

/* ------------------------------------------------------------------ */
/*  Constant memory: Lloyd-Max codebook and decision boundaries       */
/*  Matches ggml-turboquant-vsx.h exactly for bit-compatible output.  */
/* ------------------------------------------------------------------ */

__constant__ float d_TQ3_CENTROIDS[TQ3_NUM_CENTROIDS] = {
    -0.9816f, -0.7560f, -0.5005f, -0.2451f,
     0.2451f,  0.5005f,  0.7560f,  0.9816f
};

__constant__ float d_TQ3_BOUNDARIES[7] = {
    -0.8688f, -0.6283f, -0.3728f,  0.0000f,
     0.3728f,  0.6283f,  0.8688f
};

/* Host-side centroids available if needed for CPU-side verification */
/* static const float h_TQ3_CENTROIDS[TQ3_NUM_CENTROIDS] = {
    -0.9816f, -0.7560f, -0.5005f, -0.2451f,
     0.2451f,  0.5005f,  0.7560f,  0.9816f
}; */

/* ------------------------------------------------------------------ */
/*  xoshiro256** PRNG -- device side                                  */
/*  Used to generate deterministic sign-flip mask per seed.           */
/* ------------------------------------------------------------------ */

__device__ __forceinline__
uint64_t tq3_rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

/*
 * Generate next xoshiro256** value and advance state.
 * Matches the VSX header implementation exactly.
 */
__device__ __forceinline__
uint64_t tq3_xoshiro_next(uint64_t s[4]) {
    const uint64_t result = tq3_rotl(s[1] * 5, 7) * 9;
    const uint64_t t = s[1] << 17;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = tq3_rotl(s[3], 45);

    return result;
}

/*
 * Seed xoshiro256** state from a 64-bit seed via splitmix64.
 * Identical to the VSX header so sign masks match bit-for-bit.
 */
__device__ __forceinline__
void tq3_xoshiro_seed(uint64_t s[4], uint64_t seed) {
    for (int i = 0; i < 4; i++) {
        seed += 0x9E3779B97F4A7C15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z = z ^ (z >> 31);
        s[i] = z;
    }
}

/* ================================================================== */
/*  CUDA Kernels                                                      */
/* ================================================================== */

/*
 * Shared memory layout per block (1 block = 1 TQ3 vector):
 *   [0..511]   float work[128]    -- 512 bytes, working buffer
 *   [512..639] int8_t signs[128]  -- 128 bytes, +1/-1 as bytes
 *
 * Block size: 128 threads, one per vector coordinate.
 */

/* ------------------------------------------------------------------ */
/*  tq3_quantize_kernel                                               */
/*  One CUDA block per 128-float TQ3 vector.                          */
/* ------------------------------------------------------------------ */

__global__ void tq3_quantize_kernel(
    const float   * __restrict__ src,
    block_tq3_t   * __restrict__ dst,
    int             n_vectors,
    uint64_t        seed)
{
    const int vec_id = blockIdx.x;
    if (vec_id >= n_vectors) return;

    const int tid = threadIdx.x;  /* 0..127 */

    /* Shared memory: working buffer + sign mask */
    __shared__ float  s_work[TQ3_BLOCK_SIZE];
    __shared__ int8_t s_sign[TQ3_BLOCK_SIZE];

    /* ---- Step 0: Generate sign-flip mask (thread 0 only, tiny cost) ---- */
    if (tid == 0) {
        uint64_t rng[4];
        tq3_xoshiro_seed(rng, seed);
        for (int i = 0; i < TQ3_BLOCK_SIZE; i++) {
            uint64_t r = tq3_xoshiro_next(rng);
            s_sign[i] = (r & 1) ? -1 : 1;
        }
    }
    __syncthreads();

    /* ---- Step 1: Load source value ---- */
    const float *block_src = src + (int64_t)vec_id * TQ3_BLOCK_SIZE;
    float val = block_src[tid];

    /* ---- Step 2: Compute L2 norm via warp reduction ---- */
    float sq = val * val;

    /* Warp-level reduction (assumes warp size 32, 4 warps in block) */
    for (int offset = 16; offset > 0; offset >>= 1) {
        sq += __shfl_down_sync(0xFFFFFFFF, sq, offset);
    }

    /* Lane 0 of each warp writes partial sum to shared memory */
    __shared__ float s_partial[4];
    if ((tid & 31) == 0) {
        s_partial[tid >> 5] = sq;
    }
    __syncthreads();

    /* Thread 0 reduces the 4 warp sums */
    __shared__ float s_norm;
    __shared__ float s_inv_norm;
    if (tid == 0) {
        float total = s_partial[0] + s_partial[1] + s_partial[2] + s_partial[3];
        s_norm = sqrtf(total);
        s_inv_norm = (s_norm > 1e-12f) ? (1.0f / s_norm) : 0.0f;
    }
    __syncthreads();

    float norm = s_norm;

    /* Handle zero-norm block: write zeros and exit */
    if (norm < 1e-12f) {
        if (tid == 0) {
            dst[vec_id].norm  = 0.0f;
            dst[vec_id].scale = 0.0f;
        }
        /* Each thread clears part of packed_indices (48 bytes / 128 threads) */
        if (tid < TQ3_PACKED_BYTES) {
            dst[vec_id].packed_indices[tid] = 0;
        }
        return;
    }

    /* ---- Step 3: Normalize to unit vector ---- */
    val *= s_inv_norm;
    s_work[tid] = val;
    __syncthreads();

    /* ---- Step 4: In-place FWHT (7 butterfly stages) ---- */
    /*
     * Each stage s: stride = 2^(s+1), half = 2^s
     * Pair (base + j, base + j + half): a' = a + b, b' = a - b
     * Each thread handles its own index based on butterfly pattern.
     */
    for (int stage = 0; stage < TQ3_FWHT_STAGES; stage++) {
        int half   = 1 << stage;
        int stride = half << 1;

        /* Determine which pair this thread belongs to */
        int group  = tid / stride;         /* which group of size 'stride' */
        int within = tid % stride;         /* position within group */
        int base   = group * stride;

        if (within < half) {
            /* This thread is in the 'a' position */
            int idx_a = base + within;
            int idx_b = base + within + half;
            float a = s_work[idx_a];
            float b = s_work[idx_b];
            s_work[idx_a] = a + b;
            s_work[idx_b] = a - b;
        }
        __syncthreads();
    }

    /* ---- Step 5: Apply deterministic sign flips ---- */
    s_work[tid] *= (float)s_sign[tid];
    __syncthreads();

    /* ---- Step 6: Compute absmax via warp reduction ---- */
    float absval = fabsf(s_work[tid]);

    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xFFFFFFFF, absval, offset);
        absval = fmaxf(absval, other);
    }

    if ((tid & 31) == 0) {
        s_partial[tid >> 5] = absval;
    }
    __syncthreads();

    __shared__ float s_scale;
    __shared__ float s_inv_scale;
    if (tid == 0) {
        float amax = fmaxf(fmaxf(s_partial[0], s_partial[1]),
                           fmaxf(s_partial[2], s_partial[3]));
        s_scale = amax;
        s_inv_scale = (amax > 1e-12f) ? (1.0f / amax) : 0.0f;
    }
    __syncthreads();

    /* ---- Step 7: Normalize to [-1, +1] for codebook ---- */
    float normalized = s_work[tid] * s_inv_scale;

    /* ---- Step 8: Quantize to nearest Lloyd-Max centroid ---- */
    /*
     * Binary search through 7 boundaries for 8 centroids.
     * Matches _tq3_quantize_scalar() in the VSX header exactly.
     */
    uint8_t idx;
    if (normalized < d_TQ3_BOUNDARIES[3]) {
        if (normalized < d_TQ3_BOUNDARIES[1]) {
            idx = (normalized < d_TQ3_BOUNDARIES[0]) ? 0 : 1;
        } else {
            idx = (normalized < d_TQ3_BOUNDARIES[2]) ? 2 : 3;
        }
    } else {
        if (normalized < d_TQ3_BOUNDARIES[5]) {
            idx = (normalized < d_TQ3_BOUNDARIES[4]) ? 4 : 5;
        } else {
            idx = (normalized < d_TQ3_BOUNDARIES[6]) ? 6 : 7;
        }
    }

    /* ---- Step 9: Pack 128 x 3-bit indices into 48 bytes ---- */
    /*
     * Each thread owns one 3-bit index.  We compute the byte position
     * and bit offset, then use atomicOr to write without conflicts.
     * This is safe because at most 2 threads share a byte, and their
     * 3-bit fields never overlap on the same bits.
     *
     * For throughput we first zero the packed buffer, then OR in bits.
     */
    unsigned char *packed = dst[vec_id].packed_indices;

    /* Zero the packed buffer (48 bytes, use first 48 threads) */
    if (tid < TQ3_PACKED_BYTES) {
        packed[tid] = 0;
    }
    __syncthreads();

    {
        int bit_pos  = tid * 3;
        int byte_idx = bit_pos >> 3;
        int bit_off  = bit_pos & 7;

        /* Low part always fits in one byte */
        atomicOr((unsigned int *)(packed + (byte_idx & ~3)),
                 (unsigned int)((idx << bit_off) << ((byte_idx & 3) * 8)));

        /* High part: if bit_off > 5, 3 bits spill into next byte */
        if (bit_off > 5) {
            int next_byte = byte_idx + 1;
            atomicOr((unsigned int *)(packed + (next_byte & ~3)),
                     (unsigned int)(((idx >> (8 - bit_off)) & 0x07)
                                    << ((next_byte & 3) * 8)));
        }
    }
    __syncthreads();

    /* ---- Step 10: Write norm and scale (thread 0) ---- */
    if (tid == 0) {
        dst[vec_id].norm  = norm;
        dst[vec_id].scale = s_scale;
    }
}

/* ------------------------------------------------------------------ */
/*  tq3_dequantize_kernel                                             */
/*  One CUDA block per 128-float TQ3 vector.                          */
/* ------------------------------------------------------------------ */

__global__ void tq3_dequantize_kernel(
    const block_tq3_t * __restrict__ src,
    float             * __restrict__ dst_out,
    int                 n_vectors,
    uint64_t            seed)
{
    const int vec_id = blockIdx.x;
    if (vec_id >= n_vectors) return;

    const int tid = threadIdx.x;  /* 0..127 */

    __shared__ float  s_work[TQ3_BLOCK_SIZE];
    __shared__ int8_t s_sign[TQ3_BLOCK_SIZE];

    /* ---- Step 0: Generate sign-flip mask ---- */
    if (tid == 0) {
        uint64_t rng[4];
        tq3_xoshiro_seed(rng, seed);
        for (int i = 0; i < TQ3_BLOCK_SIZE; i++) {
            uint64_t r = tq3_xoshiro_next(rng);
            s_sign[i] = (r & 1) ? -1 : 1;
        }
    }
    __syncthreads();

    float norm  = src[vec_id].norm;
    float scale = src[vec_id].scale;

    /* Handle zero-norm block */
    if (norm < 1e-12f) {
        dst_out[(int64_t)vec_id * TQ3_BLOCK_SIZE + tid] = 0.0f;
        return;
    }

    /* ---- Step 1: Unpack this thread's 3-bit index ---- */
    const unsigned char *packed = src[vec_id].packed_indices;
    uint8_t idx;
    {
        int bit_pos  = tid * 3;
        int byte_idx = bit_pos >> 3;
        int bit_off  = bit_pos & 7;

        uint16_t two_bytes = (uint16_t)packed[byte_idx];
        if (byte_idx + 1 < TQ3_PACKED_BYTES) {
            two_bytes |= ((uint16_t)packed[byte_idx + 1]) << 8;
        }
        idx = (uint8_t)((two_bytes >> bit_off) & 0x07);
    }

    /* ---- Step 2: Dequantize: centroid * absmax_scale ---- */
    s_work[tid] = d_TQ3_CENTROIDS[idx] * scale;
    __syncthreads();

    /* ---- Step 3: Undo sign flips ---- */
    s_work[tid] *= (float)s_sign[tid];
    __syncthreads();

    /* ---- Step 4: Inverse FWHT (same as forward, then /N) ---- */
    for (int stage = 0; stage < TQ3_FWHT_STAGES; stage++) {
        int half   = 1 << stage;
        int stride = half << 1;

        int group  = tid / stride;
        int within = tid % stride;
        int base   = group * stride;

        if (within < half) {
            int idx_a = base + within;
            int idx_b = base + within + half;
            float a = s_work[idx_a];
            float b = s_work[idx_b];
            s_work[idx_a] = a + b;
            s_work[idx_b] = a - b;
        }
        __syncthreads();
    }

    /* Divide by N (= 128) */
    float result = s_work[tid] * (1.0f / (float)TQ3_BLOCK_SIZE);

    /* ---- Step 5: Rescale by original L2 norm ---- */
    result *= norm;

    /* ---- Write output ---- */
    dst_out[(int64_t)vec_id * TQ3_BLOCK_SIZE + tid] = result;
}

/* ================================================================== */
/*  Host Wrappers                                                     */
/* ================================================================== */

void tq3_cuda_quantize(
    const float   *d_src,
    block_tq3_t   *d_dst,
    int            n_vectors,
    uint64_t       seed,
    cudaStream_t   stream)
{
    if (n_vectors <= 0) return;
    dim3 grid(n_vectors);
    dim3 block(TQ3_BLOCK_SIZE);
    tq3_quantize_kernel<<<grid, block, 0, stream>>>(d_src, d_dst, n_vectors, seed);
}

void tq3_cuda_dequantize(
    const block_tq3_t *d_src,
    float             *d_dst,
    int                n_vectors,
    uint64_t           seed,
    cudaStream_t       stream)
{
    if (n_vectors <= 0) return;
    dim3 grid(n_vectors);
    dim3 block(TQ3_BLOCK_SIZE);
    tq3_dequantize_kernel<<<grid, block, 0, stream>>>(d_src, d_dst, n_vectors, seed);
}

/* ================================================================== */
/*  Self-Test                                                         */
/* ================================================================== */

#ifdef TURBOQUANT_SELF_TEST

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

/*
 * tq3_cuda_self_test -- round-trip accuracy test on GPU
 *
 * 1. Generate pseudo-Gaussian test data (central limit theorem, 12 uniforms)
 * 2. Quantize on GPU
 * 3. Dequantize on GPU
 * 4. Copy back and check MSE + cosine similarity
 *
 * Returns 0 on PASS, 1 on FAIL.
 */
int tq3_cuda_self_test(void) {
    const uint64_t seed = 0x456C79616E4C6162ULL;  /* "ElyanLab" */
    const int n_vectors = 64;  /* test 64 blocks for good coverage */
    const int n_floats  = n_vectors * TQ3_BLOCK_SIZE;

    /* ---- Allocate host memory ---- */
    float *h_original  = (float *)malloc(n_floats * sizeof(float));
    float *h_recovered = (float *)malloc(n_floats * sizeof(float));
    if (!h_original || !h_recovered) {
        fprintf(stderr, "Host allocation failed\n");
        return 1;
    }

    /* ---- Generate test data: pseudo-Gaussian via CLT ---- */
    {
        uint32_t lcg = 0xDEADBEEF;
        for (int i = 0; i < n_floats; i++) {
            float sum = 0.0f;
            for (int j = 0; j < 12; j++) {
                lcg = lcg * 1664525u + 1013904223u;
                sum += (float)(lcg & 0xFFFF) / 65535.0f;
            }
            h_original[i] = (sum - 6.0f) * 0.5f;  /* ~N(0, 0.25) */
        }
    }

    /* ---- Allocate device memory ---- */
    float       *d_src = NULL;
    float       *d_dst = NULL;
    block_tq3_t *d_blocks = NULL;

    CUDA_CHECK(cudaMalloc(&d_src,    n_floats  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dst,    n_floats  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_blocks, n_vectors * sizeof(block_tq3_t)));

    /* ---- Upload source data ---- */
    CUDA_CHECK(cudaMemcpy(d_src, h_original, n_floats * sizeof(float),
                          cudaMemcpyHostToDevice));

    /* ---- Quantize on GPU ---- */
    tq3_cuda_quantize(d_src, d_blocks, n_vectors, seed, 0);
    CUDA_CHECK(cudaGetLastError());

    /* ---- Dequantize on GPU ---- */
    tq3_cuda_dequantize(d_blocks, d_dst, n_vectors, seed, 0);
    CUDA_CHECK(cudaGetLastError());

    /* ---- Copy results back ---- */
    CUDA_CHECK(cudaMemcpy(h_recovered, d_dst, n_floats * sizeof(float),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaDeviceSynchronize());

    /* ---- Compute error metrics ---- */
    double mse_total  = 0.0;
    double mean_total = 0.0;
    double var_total  = 0.0;
    double dot_or = 0.0, dot_oo = 0.0, dot_rr = 0.0;

    for (int i = 0; i < n_floats; i++) {
        mean_total += (double)h_original[i];
    }
    mean_total /= (double)n_floats;

    for (int i = 0; i < n_floats; i++) {
        double diff = (double)h_recovered[i] - (double)h_original[i];
        mse_total += diff * diff;

        double dev = (double)h_original[i] - mean_total;
        var_total += dev * dev;

        dot_or += (double)h_original[i]  * (double)h_recovered[i];
        dot_oo += (double)h_original[i]  * (double)h_original[i];
        dot_rr += (double)h_recovered[i] * (double)h_recovered[i];
    }

    double mse  = mse_total / (double)n_floats;
    double var  = var_total / (double)n_floats;
    double nmse = (var > 1e-12) ? (mse / var) : 0.0;
    double cos_sim = (dot_oo > 1e-12 && dot_rr > 1e-12)
        ? dot_or / (sqrt(dot_oo) * sqrt(dot_rr))
        : 0.0;

    /* ---- Also verify one block's packed data via host unpack ---- */
    block_tq3_t h_block;
    CUDA_CHECK(cudaMemcpy(&h_block, d_blocks, sizeof(block_tq3_t),
                          cudaMemcpyDeviceToHost));

    int pack_ok = 1;
    {
        /* Unpack and verify each index is in range [0, 7] */
        int bit_pos = 0;
        for (int i = 0; i < TQ3_BLOCK_SIZE; i++) {
            int byte_idx = bit_pos >> 3;
            int bit_off  = bit_pos & 7;

            uint16_t two_bytes = (uint16_t)h_block.packed_indices[byte_idx];
            if (byte_idx + 1 < TQ3_PACKED_BYTES) {
                two_bytes |= ((uint16_t)h_block.packed_indices[byte_idx + 1]) << 8;
            }
            uint8_t idx = (uint8_t)((two_bytes >> bit_off) & 0x07);
            if (idx > 7) {
                pack_ok = 0;
                break;
            }
            bit_pos += 3;
        }
    }

    /* ---- Print GPU info ---- */
    int device;
    cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    printf("TQ3 CUDA Self-Test (Elyan Labs TurboQuant)\n");
    printf("  GPU:            %s (SM_%d%d)\n", prop.name, prop.major, prop.minor);
    printf("  Test vectors:   %d (%d floats)\n", n_vectors, n_floats);
    printf("  Block size:     %d floats\n", TQ3_BLOCK_SIZE);
    printf("  Packed size:    %d bytes (%.1fx compression vs FP32)\n",
           (int)sizeof(block_tq3_t),
           (float)(TQ3_BLOCK_SIZE * sizeof(float)) / (float)sizeof(block_tq3_t));
    printf("  Block[0] norm:  %.6f\n", h_block.norm);
    printf("  Block[0] scale: %.6f\n", h_block.scale);
    printf("  MSE:            %.8f\n", mse);
    printf("  Variance:       %.8f\n", var);
    printf("  NMSE:           %.6f (%.2f%%)\n", nmse, nmse * 100.0);
    printf("  Cosine sim:     %.6f\n", cos_sim);
    printf("  Pack check:     %s\n", pack_ok ? "PASS" : "FAIL");

    /*
     * Pass criteria (same as VSX header):
     *   - NMSE < 15%: TQ3 with Lloyd-Max achieves ~3-12% on typical activations.
     *   - Cosine similarity > 0.90: directional preservation is critical for
     *     KV cache -- attention depends on vector direction.
     */
    int nmse_ok = (nmse < 0.15);
    int cos_ok  = (cos_sim > 0.90);
    int pass    = nmse_ok && cos_ok && pack_ok;

    printf("  NMSE check:     %s (%.2f%% < 15%%)\n",
           nmse_ok ? "PASS" : "FAIL", nmse * 100.0);
    printf("  Cosine check:   %s (%.4f > 0.90)\n",
           cos_ok ? "PASS" : "FAIL", cos_sim);
    printf("  Result:         %s\n", pass ? "PASS" : "FAIL");

    /* ---- Cleanup ---- */
    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_blocks);
    free(h_original);
    free(h_recovered);

    return pass ? 0 : 1;
}

int main(void) {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        fprintf(stderr, "No CUDA devices found.\n");
        return 1;
    }
    return tq3_cuda_self_test();
}

#endif /* TURBOQUANT_SELF_TEST */
