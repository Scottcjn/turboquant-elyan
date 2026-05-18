#include "ggml-turboquant-vsx.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static void test_block_layout(void) {
    assert(TQ3_BLOCK_SIZE == 128);
    assert(TQ3_PACKED_BYTES == 48);
    assert(sizeof(block_tq3_t) == 56);
}

static void test_pack_unpack_pattern_round_trip(void) {
    uint8_t indices[TQ3_BLOCK_SIZE];
    uint8_t unpacked[TQ3_BLOCK_SIZE];
    uint8_t packed[TQ3_PACKED_BYTES];

    for (int i = 0; i < TQ3_BLOCK_SIZE; i++) {
        indices[i] = (uint8_t)((i * 3) % TQ3_NUM_CENTROIDS);
    }

    _tq3_pack_indices(indices, packed);
    _tq3_unpack_indices(packed, unpacked);

    for (int i = 0; i < TQ3_BLOCK_SIZE; i++) {
        assert(unpacked[i] == indices[i]);
    }
}

static void test_quantize_scalar_boundaries(void) {
    assert(_tq3_quantize_scalar(-1.0f) == 0);
    assert(_tq3_quantize_scalar(-0.80f) == 1);
    assert(_tq3_quantize_scalar(-0.50f) == 2);
    assert(_tq3_quantize_scalar(-0.10f) == 3);
    assert(_tq3_quantize_scalar(0.00f) == 4);
    assert(_tq3_quantize_scalar(0.50f) == 5);
    assert(_tq3_quantize_scalar(0.80f) == 6);
    assert(_tq3_quantize_scalar(1.00f) == 7);
}

static void test_sign_mask_is_deterministic_for_seed(void) {
    tq3_context_t a;
    tq3_context_t b;
    tq3_vsx_init(&a, 0x1234ULL);
    tq3_vsx_init(&b, 0x1234ULL);

    assert(a.seed == b.seed);
    for (int i = 0; i < TQ3_BLOCK_SIZE; i++) {
        assert(a.sign_mask[i] == b.sign_mask[i]);
        assert(a.sign_mask[i] == 1.0f || a.sign_mask[i] == -1.0f);
    }
}

static void test_zero_block_round_trip_dequantizes_to_zero(void) {
    float input[TQ3_BLOCK_SIZE];
    float output[TQ3_BLOCK_SIZE];
    block_tq3_t block;

    memset(input, 0, sizeof(input));
    memset(output, 0x7f, sizeof(output));
    memset(&block, 0x7f, sizeof(block));

    quantize_row_tq3_vsx(input, &block, TQ3_BLOCK_SIZE);
    dequantize_row_tq3_vsx(&block, output, TQ3_BLOCK_SIZE);

    assert(block.norm == 0.0f);
    assert(block.scale == 0.0f);
    for (int i = 0; i < TQ3_BLOCK_SIZE; i++) {
        assert(output[i] == 0.0f);
    }
}

int main(void) {
    (void)tq3_self_test;
    test_block_layout();
    test_pack_unpack_pattern_round_trip();
    test_quantize_scalar_boundaries();
    test_sign_mask_is_deterministic_for_seed();
    test_zero_block_round_trip_dequantizes_to_zero();
    printf("turboquant VSX scalar fallback tests passed\n");
    return 0;
}
