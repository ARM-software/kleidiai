//
// SPDX-FileCopyrightText: Copyright 2022, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//


#include "kai/ops/gemm/kai_ops.hpp"
#include <cstddef>
#include <cstdint>
#include <arm_neon.h>

namespace kai {
namespace ops {
namespace depthwise {

void a64_u8q_nhwc_generic_output4_mla_depthfirst_intrinsics_impl(
  const uint8_t *const *const inptrs,
  uint8_t *const *const outptrs,
  const void *params,
  const kai::ops::Requantize32& qp,
  const unsigned int n_points,
  const unsigned int n_channels
)
{
  auto weights = reinterpret_cast<const uint8_t *>(params);
  auto bias = qp.bias;

  const uint8x16_t a_offset = vdupq_n_u8(qp.a_offset);
  const uint8x16_t b_offset = vdupq_n_u8(qp.b_offset);
  const int16x8_t c_offset = vdupq_n_s16(qp.c_offset);
  const int16x8_t clamp_min = vdupq_n_s16(qp.minval);
  const int16x8_t clamp_max = vdupq_n_s16(qp.maxval);

  int32x4_t rq_left_shift_0123, rq_left_shift_4567, rq_left_shift_89ab, rq_left_shift_cdef;
  rq_left_shift_0123 = rq_left_shift_4567 = rq_left_shift_89ab = rq_left_shift_cdef = vdupq_n_s32(qp.per_layer_left_shift);
  auto rq_per_channel_left_shifts = qp.per_channel_left_shifts;

  int32x4_t rq_mul_0123, rq_mul_4567, rq_mul_89ab, rq_mul_cdef;
  rq_mul_0123 = rq_mul_4567 = rq_mul_89ab = rq_mul_cdef = vdupq_n_s32(qp.per_layer_mul);
  auto rq_per_channel_muls = qp.per_channel_muls;

  int32x4_t rq_right_shift_0123, rq_right_shift_4567, rq_right_shift_89ab, rq_right_shift_cdef;
  rq_right_shift_0123 = rq_right_shift_4567 = rq_right_shift_89ab = rq_right_shift_cdef = vdupq_n_s32(qp.per_layer_right_shift);
  auto rq_per_channel_right_shifts = qp.per_channel_right_shifts;

  for (auto c = 0u; c < n_channels; c += 16u)
  {
    // Load the bias
    int32x4_t bias_0123, bias_4567, bias_89ab, bias_cdef;
    if (bias != nullptr)
    {
      bias_0123 = vld1q_s32(bias);
      bias_4567 = vld1q_s32(bias + 4);
      bias_89ab = vld1q_s32(bias + 8);
      bias_cdef = vld1q_s32(bias + 12);
      bias += 16;
    }
    else
    {
      bias_0123 = bias_4567 = bias_89ab = bias_cdef = vdupq_n_s32(0);
    }

    // Use the bias to initialise the accumulators
    int32x4_t acc0_0123, acc1_0123, acc2_0123, acc3_0123;
    int32x4_t acc0_4567, acc1_4567, acc2_4567, acc3_4567;
    int32x4_t acc0_89ab, acc1_89ab, acc2_89ab, acc3_89ab;
    int32x4_t acc0_cdef, acc1_cdef, acc2_cdef, acc3_cdef;
    acc0_0123 = acc1_0123 = acc2_0123 = acc3_0123 = bias_0123;
    acc0_4567 = acc1_4567 = acc2_4567 = acc3_4567 = bias_4567;
    acc0_89ab = acc1_89ab = acc2_89ab = acc3_89ab = bias_89ab;
    acc0_cdef = acc1_cdef = acc2_cdef = acc3_cdef = bias_cdef;

    auto inptr = inptrs;
    for (auto n = 0u; n < n_points; n++)
    {
      // Load the weight corresponding to this point
      const uint8x16_t w_all = vld1q_u8(weights);
      const int16x8_t w_01234567 = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(w_all), vget_low_u8(b_offset)));
      const int16x8_t w_89abcdef = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(w_all), vget_high_u8(b_offset)));
      weights += 16;

      // Load input points
#define LOAD_INPUT(N) \
      const uint8x16_t i ## N ## _all = vld1q_u8(*(inptr++) + c); \
      const int16x8_t i ## N ## _01234567 = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(i ## N ##_all), vget_low_u8(a_offset))); \
      const int16x8_t i ## N ## _89abcdef = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(i ## N ##_all), vget_high_u8(a_offset)));

#define PERFORM_MULS(N) \
      acc ## N ##_0123 = vmlal_s16(acc ## N ##_0123, vget_low_s16(i ## N ##_01234567), vget_low_s16(w_01234567)); \
      acc ## N ##_4567 = vmlal_s16(acc ## N ##_4567, vget_high_s16(i ## N ##_01234567), vget_high_s16(w_01234567)); \
      acc ## N ##_89ab = vmlal_s16(acc ## N ##_89ab, vget_low_s16(i ## N ##_89abcdef), vget_low_s16(w_89abcdef)); \
      acc ## N ##_cdef = vmlal_s16(acc ## N ##_cdef, vget_high_s16(i ## N ##_89abcdef), vget_high_s16(w_89abcdef));

      // Load consecutive input points and perform the multiplications.
      LOAD_INPUT(0)
      PERFORM_MULS(0)

      LOAD_INPUT(1)
      PERFORM_MULS(1)

      LOAD_INPUT(2)
      PERFORM_MULS(2)

      LOAD_INPUT(3)
      PERFORM_MULS(3)
    }

    // Perform the requantisation
    if (rq_per_channel_left_shifts != nullptr)
    {
      rq_left_shift_0123 = vld1q_s32(rq_per_channel_left_shifts);
      rq_left_shift_4567 = vld1q_s32(rq_per_channel_left_shifts + 4);
      rq_left_shift_89ab = vld1q_s32(rq_per_channel_left_shifts + 8);
      rq_left_shift_cdef = vld1q_s32(rq_per_channel_left_shifts + 12);
      rq_per_channel_left_shifts += 16;
    }

#define PERFORM_LEFT_SHIFT(CHANNELS) \
    acc0_##CHANNELS = vshlq_s32(acc0_##CHANNELS, rq_left_shift_##CHANNELS); \
    acc1_##CHANNELS = vshlq_s32(acc1_##CHANNELS, rq_left_shift_##CHANNELS); \
    acc2_##CHANNELS = vshlq_s32(acc2_##CHANNELS, rq_left_shift_##CHANNELS); \
    acc3_##CHANNELS = vshlq_s32(acc3_##CHANNELS, rq_left_shift_##CHANNELS);

    PERFORM_LEFT_SHIFT(0123)
    PERFORM_LEFT_SHIFT(4567)
    PERFORM_LEFT_SHIFT(89ab)
    PERFORM_LEFT_SHIFT(cdef)

    if (rq_per_channel_muls != nullptr)
    {
      rq_mul_0123 = vld1q_s32(rq_per_channel_muls);
      rq_mul_4567 = vld1q_s32(rq_per_channel_muls + 4);
      rq_mul_89ab = vld1q_s32(rq_per_channel_muls + 8);
      rq_mul_cdef = vld1q_s32(rq_per_channel_muls + 12);
      rq_per_channel_muls += 16;
    }

#define PERFORM_RQ_MUL(CHANNELS) \
    acc0_##CHANNELS = vqdmulhq_s32(acc0_##CHANNELS, rq_mul_##CHANNELS); \
    acc1_##CHANNELS = vqdmulhq_s32(acc1_##CHANNELS, rq_mul_##CHANNELS); \
    acc2_##CHANNELS = vqdmulhq_s32(acc2_##CHANNELS, rq_mul_##CHANNELS); \
    acc3_##CHANNELS = vqdmulhq_s32(acc3_##CHANNELS, rq_mul_##CHANNELS);

    PERFORM_RQ_MUL(0123)
    PERFORM_RQ_MUL(4567)
    PERFORM_RQ_MUL(89ab)
    PERFORM_RQ_MUL(cdef)

    if (rq_per_channel_right_shifts != nullptr)
    {
      rq_right_shift_0123 = vld1q_s32(rq_per_channel_right_shifts);
      rq_right_shift_4567 = vld1q_s32(rq_per_channel_right_shifts + 4);
      rq_right_shift_89ab = vld1q_s32(rq_per_channel_right_shifts + 8);
      rq_right_shift_cdef = vld1q_s32(rq_per_channel_right_shifts + 12);
      rq_per_channel_right_shifts += 16;
    }

#define PERFORM_RIGHT_SHIFT(CHANNELS) \
    acc0_##CHANNELS = vrshlq_s32(acc0_##CHANNELS, rq_right_shift_##CHANNELS); \
    acc1_##CHANNELS = vrshlq_s32(acc1_##CHANNELS, rq_right_shift_##CHANNELS); \
    acc2_##CHANNELS = vrshlq_s32(acc2_##CHANNELS, rq_right_shift_##CHANNELS); \
    acc3_##CHANNELS = vrshlq_s32(acc3_##CHANNELS, rq_right_shift_##CHANNELS);

    PERFORM_RIGHT_SHIFT(0123)
    PERFORM_RIGHT_SHIFT(4567)
    PERFORM_RIGHT_SHIFT(89ab)
    PERFORM_RIGHT_SHIFT(cdef)

    // Narrow prior to adding the C offset
#define NARROW_ACC(N) \
    int16x8_t out ## N ## _01234567 = vqmovn_high_s32(vqmovn_s32(acc ## N ## _0123), acc ## N ## _4567); \
    int16x8_t out ## N ## _89abcdef = vqmovn_high_s32(vqmovn_s32(acc ## N ## _89ab), acc ## N ## _cdef); \

    NARROW_ACC(0)
    NARROW_ACC(1)
    NARROW_ACC(2)
    NARROW_ACC(3)

#define RQ_OFFSET(CHANNELS) \
    out0_##CHANNELS = vqaddq_s16(out0_##CHANNELS, c_offset); \
    out1_##CHANNELS = vqaddq_s16(out1_##CHANNELS, c_offset); \
    out2_##CHANNELS = vqaddq_s16(out2_##CHANNELS, c_offset); \
    out3_##CHANNELS = vqaddq_s16(out3_##CHANNELS, c_offset);

    RQ_OFFSET(01234567)
    RQ_OFFSET(89abcdef)

#define CLAMP_ACC(CHANNELS) \
    out0_##CHANNELS = vmaxq_s16(vminq_s16(out0_##CHANNELS, clamp_max), clamp_min); \
    out1_##CHANNELS = vmaxq_s16(vminq_s16(out1_##CHANNELS, clamp_max), clamp_min); \
    out2_##CHANNELS = vmaxq_s16(vminq_s16(out2_##CHANNELS, clamp_max), clamp_min); \
    out3_##CHANNELS = vmaxq_s16(vminq_s16(out3_##CHANNELS, clamp_max), clamp_min);

    CLAMP_ACC(01234567)
    CLAMP_ACC(89abcdef)

    // Narrow and store
#define NARROW_AND_STORE(N) \
    const uint8x16_t out##N##_all = vuzp1q_u8((uint8x16_t) out##N##_01234567, (uint8x16_t) out##N##_89abcdef); \
    vst1q_u8(*(outptr++) + c, out##N##_all);

    auto outptr = outptrs; // Copy so we can increment
    NARROW_AND_STORE(0)
    NARROW_AND_STORE(1)
    NARROW_AND_STORE(2)
    NARROW_AND_STORE(3)
  }
}

}  // namespace depthwise
}  // namespace ops
}  // namespace kai
