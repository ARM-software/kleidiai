//
// SPDX-FileCopyrightText: Copyright 2022, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "kai/ops/gemm/kai_ops.hpp"

#include <arm_neon.h>
#include <cstddef>
#include <cstdint>

#if defined(__aarch64__)

namespace kai {
namespace ops {
namespace depthwise {

void a64_u8qa_nhwc_3x3_s1_output2x2_mla_depthfirst_intrinsics_impl(
  unsigned int n_channels,
  const uint8_t *const *const inptrs,
  const uint8_t *weights,
  const int32_t *bias,
  const kai::ops::Requantize32 &qp,
  const int32_t *requant_muls,
  const int32_t *requant_shifts,
  uint8_t *const *const outptrs
)
{
  const uint8x16_t b_offset = vdupq_n_u8((uint8_t) qp.b_offset);
  const int16x8_t c_offset = vdupq_n_s16((int16_t) qp.c_offset);
  const int16x8_t clamp_min = vdupq_n_s16(qp.minval);
  const int16x8_t clamp_max = vdupq_n_s16(qp.maxval);
  auto c = 0u;

  do
  {
    // Prepare the bias
    int32x4_t bias0123, bias4567, bias89ab, biascdef;
    if (bias != nullptr)
    {
      bias0123 = vld1q_s32(bias);
      bias4567 = vld1q_s32(bias + 4);
      bias89ab = vld1q_s32(bias + 8);
      biascdef = vld1q_s32(bias + 12);
      bias += 16;
    }
    else
    {
      bias0123 = bias4567 = bias89ab = biascdef = vdupq_n_s32(0);
    }

#define LOAD_WEIGHT(I, J) \
    const uint8x16_t w ## I ## J ## _all = vld1q_u8(weights); \
    weights += 16; \
    const int16x8_t w ## I ## J ## _01234567 = (int16x8_t) vsubl_u8(vget_low_u8(w ## I ## J ## _all), vget_low_u8(b_offset)); \
    const int16x8_t w ## I ## J ## _89abcdef = (int16x8_t) vsubl_u8(vget_high_u8(w ## I ## J ## _all), vget_high_u8(b_offset));

#define LOAD_INPUT_NO_OFFSET(I, J) \
    const uint8x16_t i ## I ## J ## _all = vld1q_u8(inptrs[I*4 + J] + c); \
    const int16x8_t i ## I ## J ## _01234567 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(i ## I ## J ## _all))); \
    const int16x8_t i ## I ## J ## _89abcdef = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(i ## I ## J ## _all)));

#define MUL_ACC(OI, OJ, WI, WJ, II, IJ) \
    acc ## OI ## OJ ## _0123 = vmlal_s16(acc ## OI ## OJ ## _0123, vget_low_s16(w ## WI ## WJ ## _01234567), vget_low_s16(i ## II ## IJ ## _01234567)); \
    acc ## OI ## OJ ## _4567 = vmlal_s16(acc ## OI ## OJ ## _4567, vget_high_s16(w ## WI ## WJ ## _01234567), vget_high_s16(i ## II ## IJ ## _01234567)); \
    acc ## OI ## OJ ## _89ab = vmlal_s16(acc ## OI ## OJ ## _89ab, vget_low_s16(w ## WI ## WJ ## _89abcdef), vget_low_s16(i ## II ## IJ ## _89abcdef)); \
    acc ## OI ## OJ ## _cdef = vmlal_s16(acc ## OI ## OJ ## _cdef, vget_high_s16(w ## WI ## WJ ## _89abcdef), vget_high_s16(i ## II ## IJ ## _89abcdef));

    LOAD_WEIGHT(0, 0)
    LOAD_INPUT_NO_OFFSET(0, 0)

    int32x4_t acc00_0123 = vmlal_s16(bias0123, vget_low_s16(w00_01234567), vget_low_s16(i00_01234567));
    int32x4_t acc00_4567 = vmlal_s16(bias4567, vget_high_s16(w00_01234567), vget_high_s16(i00_01234567));
    int32x4_t acc00_89ab = vmlal_s16(bias89ab, vget_low_s16(w00_89abcdef), vget_low_s16(i00_89abcdef));
    int32x4_t acc00_cdef = vmlal_s16(biascdef, vget_high_s16(w00_89abcdef), vget_high_s16(i00_89abcdef));

    LOAD_INPUT_NO_OFFSET(0, 1)

    int32x4_t acc01_0123 = vmlal_s16(bias0123, vget_low_s16(w00_01234567), vget_low_s16(i01_01234567));
    int32x4_t acc01_4567 = vmlal_s16(bias4567, vget_high_s16(w00_01234567), vget_high_s16(i01_01234567));
    int32x4_t acc01_89ab = vmlal_s16(bias89ab, vget_low_s16(w00_89abcdef), vget_low_s16(i01_89abcdef));
    int32x4_t acc01_cdef = vmlal_s16(biascdef, vget_high_s16(w00_89abcdef), vget_high_s16(i01_89abcdef));

    LOAD_INPUT_NO_OFFSET(1, 0)

    int32x4_t acc10_0123 = vmlal_s16(bias0123, vget_low_s16(w00_01234567), vget_low_s16(i10_01234567));
    int32x4_t acc10_4567 = vmlal_s16(bias4567, vget_high_s16(w00_01234567), vget_high_s16(i10_01234567));
    int32x4_t acc10_89ab = vmlal_s16(bias89ab, vget_low_s16(w00_89abcdef), vget_low_s16(i10_89abcdef));
    int32x4_t acc10_cdef = vmlal_s16(biascdef, vget_high_s16(w00_89abcdef), vget_high_s16(i10_89abcdef));

    LOAD_INPUT_NO_OFFSET(1, 1)

    int32x4_t acc11_0123 = vmlal_s16(bias0123, vget_low_s16(w00_01234567), vget_low_s16(i11_01234567));
    int32x4_t acc11_4567 = vmlal_s16(bias4567, vget_high_s16(w00_01234567), vget_high_s16(i11_01234567));
    int32x4_t acc11_89ab = vmlal_s16(bias89ab, vget_low_s16(w00_89abcdef), vget_low_s16(i11_89abcdef));
    int32x4_t acc11_cdef = vmlal_s16(biascdef, vget_high_s16(w00_89abcdef), vget_high_s16(i11_89abcdef));

    LOAD_WEIGHT(0, 1)

    MUL_ACC(0, 0, 0, 1, 0, 1)
    LOAD_INPUT_NO_OFFSET(0, 2)
    MUL_ACC(0, 1, 0, 1, 0, 2)
    MUL_ACC(1, 0, 0, 1, 1, 1)
    LOAD_INPUT_NO_OFFSET(1, 2)
    MUL_ACC(1, 1, 0, 1, 1, 2)

    LOAD_WEIGHT(0, 2)

    MUL_ACC(0, 0, 0, 2, 0, 2)
    LOAD_INPUT_NO_OFFSET(0, 3)
    MUL_ACC(0, 1, 0, 2, 0, 3)
    MUL_ACC(1, 0, 0, 2, 1, 2)
    LOAD_INPUT_NO_OFFSET(1, 3)
    MUL_ACC(1, 1, 0, 2, 1, 3)

    LOAD_WEIGHT(1, 0)

    MUL_ACC(0, 0, 1, 0, 1, 0)
    MUL_ACC(0, 1, 1, 0, 1, 1)
    LOAD_INPUT_NO_OFFSET(2, 0)
    MUL_ACC(1, 0, 1, 0, 2, 0)

    LOAD_INPUT_NO_OFFSET(2, 1)
    MUL_ACC(1, 1, 1, 0, 2, 1)

    LOAD_WEIGHT(1, 1)

    MUL_ACC(0, 0, 1, 1, 1, 1)
    MUL_ACC(0, 1, 1, 1, 1, 2)
    MUL_ACC(1, 0, 1, 1, 2, 1)
    LOAD_INPUT_NO_OFFSET(2, 2)
    MUL_ACC(1, 1, 1, 1, 2, 2)

    LOAD_WEIGHT(1, 2)

    MUL_ACC(0, 0, 1, 2, 1, 2)
    MUL_ACC(0, 1, 1, 2, 1, 3)
    MUL_ACC(1, 0, 1, 2, 2, 2)
    LOAD_INPUT_NO_OFFSET(2, 3)
    MUL_ACC(1, 1, 1, 2, 2, 3)

    LOAD_WEIGHT(2, 0)

    MUL_ACC(0, 0, 2, 0, 2, 0)
    MUL_ACC(0, 1, 2, 0, 2, 1)
    LOAD_INPUT_NO_OFFSET(3, 0)
    MUL_ACC(1, 0, 2, 0, 3, 0)
    LOAD_INPUT_NO_OFFSET(3, 1)
    MUL_ACC(1, 1, 2, 0, 3, 1)

    LOAD_WEIGHT(2, 1)

    MUL_ACC(0, 0, 2, 1, 2, 1)
    MUL_ACC(0, 1, 2, 1, 2, 2)
    MUL_ACC(1, 0, 2, 1, 3, 1)
    LOAD_INPUT_NO_OFFSET(3, 2)
    MUL_ACC(1, 1, 2, 1, 3, 2)

    LOAD_WEIGHT(2, 2)

    MUL_ACC(0, 0, 2, 2, 2, 2)
    MUL_ACC(0, 1, 2, 2, 2, 3)
    MUL_ACC(1, 0, 2, 2, 3, 2)
    LOAD_INPUT_NO_OFFSET(3, 3)
    MUL_ACC(1, 1, 2, 2, 3, 3)

    // Perform requantization and store
#define RQ_MUL(CHANNELS) \
    const int32x4_t rq_mul_ ## CHANNELS = vld1q_s32(requant_muls); \
    requant_muls += 4; \
    acc00_ ## CHANNELS = vqdmulhq_s32(acc00_ ## CHANNELS, rq_mul_ ## CHANNELS); \
    acc01_ ## CHANNELS = vqdmulhq_s32(acc01_ ## CHANNELS, rq_mul_ ## CHANNELS); \
    acc10_ ## CHANNELS = vqdmulhq_s32(acc10_ ## CHANNELS, rq_mul_ ## CHANNELS); \
    acc11_ ## CHANNELS = vqdmulhq_s32(acc11_ ## CHANNELS, rq_mul_ ## CHANNELS);

    RQ_MUL(0123);
    RQ_MUL(4567);
    RQ_MUL(89ab);
    RQ_MUL(cdef);

#define RQ_SHIFT(CHANNELS) \
    const int32x4_t rq_shift_ ## CHANNELS = vld1q_s32(requant_shifts); \
    requant_shifts += 4; \
    acc00_ ## CHANNELS = vrshlq_s32(acc00_ ## CHANNELS, rq_shift_ ## CHANNELS); \
    acc01_ ## CHANNELS = vrshlq_s32(acc01_ ## CHANNELS, rq_shift_ ## CHANNELS); \
    acc10_ ## CHANNELS = vrshlq_s32(acc10_ ## CHANNELS, rq_shift_ ## CHANNELS); \
    acc11_ ## CHANNELS = vrshlq_s32(acc11_ ## CHANNELS, rq_shift_ ## CHANNELS);

    RQ_SHIFT(0123);
    RQ_SHIFT(4567);
    RQ_SHIFT(89ab);
    RQ_SHIFT(cdef);

    // At this point we narrow before adding the C offset
#define NARROW_ACC(I, J) \
    int16x8_t out ## I ## J ##_01234567 = vqmovn_high_s32(vqmovn_s32(acc ## I ## J ##_0123), acc ## I ## J ##_4567); \
    int16x8_t out ## I ## J ##_89abcdef = vqmovn_high_s32(vqmovn_s32(acc ## I ## J ##_89ab), acc ## I ## J ##_cdef);

    NARROW_ACC(0, 0)
    NARROW_ACC(0, 1)
    NARROW_ACC(1, 0)
    NARROW_ACC(1, 1)

#define RQ_OFFSET(CHANNELS) \
    out00_ ## CHANNELS = vqaddq_s16(out00_ ## CHANNELS, c_offset); \
    out01_ ## CHANNELS = vqaddq_s16(out01_ ## CHANNELS, c_offset); \
    out10_ ## CHANNELS = vqaddq_s16(out10_ ## CHANNELS, c_offset); \
    out11_ ## CHANNELS = vqaddq_s16(out11_ ## CHANNELS, c_offset);

    RQ_OFFSET(01234567);
    RQ_OFFSET(89abcdef);

#define RQ_CLAMP(CHANNELS) \
    out00_ ## CHANNELS = vmaxq_s16(out00_ ## CHANNELS, clamp_min); \
    out00_ ## CHANNELS = vminq_s16(out00_ ## CHANNELS, clamp_max); \
    out01_ ## CHANNELS = vmaxq_s16(out01_ ## CHANNELS, clamp_min); \
    out01_ ## CHANNELS = vminq_s16(out01_ ## CHANNELS, clamp_max); \
    out10_ ## CHANNELS = vmaxq_s16(out10_ ## CHANNELS, clamp_min); \
    out10_ ## CHANNELS = vminq_s16(out10_ ## CHANNELS, clamp_max); \
    out11_ ## CHANNELS = vmaxq_s16(out11_ ## CHANNELS, clamp_min); \
    out11_ ## CHANNELS = vminq_s16(out11_ ## CHANNELS, clamp_max);

    RQ_CLAMP(01234567);
    RQ_CLAMP(89abcdef);

    // Narrow and store
#define NARROW_AND_STORE(OI, OJ) \
    const int8x16_t out ## OI ## OJ ## _all = vuzp1q_s8((int8x16_t) out ## OI ## OJ ##_01234567, (int8x16_t) out ## OI ## OJ ##_89abcdef); \
    vst1q_u8(outptrs[OI * 2 + OJ] + c, (uint8x16_t) out ## OI ## OJ ##_all);

    NARROW_AND_STORE(0, 0)
    NARROW_AND_STORE(0, 1)
    NARROW_AND_STORE(1, 0)
    NARROW_AND_STORE(1, 1)

    n_channels -= 16;
    c += 16;
  } while (n_channels);
}


void a64_u8q_nhwc_3x3_s1_output2x2_mla_depthfirst_intrinsics_impl(
  unsigned int n_channels,
  const uint8_t *const *const inptrs,
  const uint8_t *weights,
  const int32_t *bias,
  const kai::ops::Requantize32 &qp,
  const int32_t *requant_muls,
  const int32_t *requant_shifts,
  uint8_t *const *const outptrs
)
{
  const uint8x16_t a_offset = vdupq_n_u8((uint8_t) qp.a_offset);
  const uint8x16_t b_offset = vdupq_n_u8((uint8_t) qp.b_offset);
  const int16x8_t c_offset = vdupq_n_s16((int16_t) qp.c_offset);
  const int16x8_t clamp_min = vdupq_n_s16(qp.minval);
  const int16x8_t clamp_max = vdupq_n_s16(qp.maxval);
  auto c = 0u;

  do
  {
    // Prepare the bias
    int32x4_t bias0123, bias4567, bias89ab, biascdef;
    if (bias != nullptr)
    {
      bias0123 = vld1q_s32(bias);
      bias4567 = vld1q_s32(bias + 4);
      bias89ab = vld1q_s32(bias + 8);
      biascdef = vld1q_s32(bias + 12);
      bias += 16;
    }
    else
    {
      bias0123 = bias4567 = bias89ab = biascdef = vdupq_n_s32(0);
    }

#define LOAD_INPUT_WITH_OFFSET(I, J) \
    const uint8x16_t i ## I ## J ## _all = vld1q_u8(inptrs[I*4 + J] + c); \
    const int16x8_t i ## I ## J ## _01234567 = (int16x8_t) vsubl_u8(vget_low_u8(i ## I ## J ## _all), vget_low_u8(a_offset)); \
    const int16x8_t i ## I ## J ## _89abcdef = (int16x8_t) vsubl_u8(vget_high_u8(i ## I ## J ## _all), vget_high_u8(a_offset));

    LOAD_WEIGHT(0, 0)
    LOAD_INPUT_WITH_OFFSET(0, 0)

    int32x4_t acc00_0123 = vmlal_s16(bias0123, vget_low_s16(w00_01234567), vget_low_s16(i00_01234567));
    int32x4_t acc00_4567 = vmlal_s16(bias4567, vget_high_s16(w00_01234567), vget_high_s16(i00_01234567));
    int32x4_t acc00_89ab = vmlal_s16(bias89ab, vget_low_s16(w00_89abcdef), vget_low_s16(i00_89abcdef));
    int32x4_t acc00_cdef = vmlal_s16(biascdef, vget_high_s16(w00_89abcdef), vget_high_s16(i00_89abcdef));

    LOAD_INPUT_WITH_OFFSET(0, 1)

    int32x4_t acc01_0123 = vmlal_s16(bias0123, vget_low_s16(w00_01234567), vget_low_s16(i01_01234567));
    int32x4_t acc01_4567 = vmlal_s16(bias4567, vget_high_s16(w00_01234567), vget_high_s16(i01_01234567));
    int32x4_t acc01_89ab = vmlal_s16(bias89ab, vget_low_s16(w00_89abcdef), vget_low_s16(i01_89abcdef));
    int32x4_t acc01_cdef = vmlal_s16(biascdef, vget_high_s16(w00_89abcdef), vget_high_s16(i01_89abcdef));

    LOAD_INPUT_WITH_OFFSET(1, 0)

    int32x4_t acc10_0123 = vmlal_s16(bias0123, vget_low_s16(w00_01234567), vget_low_s16(i10_01234567));
    int32x4_t acc10_4567 = vmlal_s16(bias4567, vget_high_s16(w00_01234567), vget_high_s16(i10_01234567));
    int32x4_t acc10_89ab = vmlal_s16(bias89ab, vget_low_s16(w00_89abcdef), vget_low_s16(i10_89abcdef));
    int32x4_t acc10_cdef = vmlal_s16(biascdef, vget_high_s16(w00_89abcdef), vget_high_s16(i10_89abcdef));

    LOAD_INPUT_WITH_OFFSET(1, 1)

    int32x4_t acc11_0123 = vmlal_s16(bias0123, vget_low_s16(w00_01234567), vget_low_s16(i11_01234567));
    int32x4_t acc11_4567 = vmlal_s16(bias4567, vget_high_s16(w00_01234567), vget_high_s16(i11_01234567));
    int32x4_t acc11_89ab = vmlal_s16(bias89ab, vget_low_s16(w00_89abcdef), vget_low_s16(i11_89abcdef));
    int32x4_t acc11_cdef = vmlal_s16(biascdef, vget_high_s16(w00_89abcdef), vget_high_s16(i11_89abcdef));

    LOAD_WEIGHT(0, 1)

    MUL_ACC(0, 0, 0, 1, 0, 1)
    LOAD_INPUT_WITH_OFFSET(0, 2)
    MUL_ACC(0, 1, 0, 1, 0, 2)
    MUL_ACC(1, 0, 0, 1, 1, 1)
    LOAD_INPUT_WITH_OFFSET(1, 2)
    MUL_ACC(1, 1, 0, 1, 1, 2)

    LOAD_WEIGHT(0, 2)

    MUL_ACC(0, 0, 0, 2, 0, 2)
    LOAD_INPUT_WITH_OFFSET(0, 3)
    MUL_ACC(0, 1, 0, 2, 0, 3)
    MUL_ACC(1, 0, 0, 2, 1, 2)
    LOAD_INPUT_WITH_OFFSET(1, 3)
    MUL_ACC(1, 1, 0, 2, 1, 3)

    LOAD_WEIGHT(1, 0)

    MUL_ACC(0, 0, 1, 0, 1, 0)
    MUL_ACC(0, 1, 1, 0, 1, 1)
    LOAD_INPUT_WITH_OFFSET(2, 0)
    MUL_ACC(1, 0, 1, 0, 2, 0)

    LOAD_INPUT_WITH_OFFSET(2, 1)
    MUL_ACC(1, 1, 1, 0, 2, 1)

    LOAD_WEIGHT(1, 1)

    MUL_ACC(0, 0, 1, 1, 1, 1)
    MUL_ACC(0, 1, 1, 1, 1, 2)
    MUL_ACC(1, 0, 1, 1, 2, 1)
    LOAD_INPUT_WITH_OFFSET(2, 2)
    MUL_ACC(1, 1, 1, 1, 2, 2)

    LOAD_WEIGHT(1, 2)

    MUL_ACC(0, 0, 1, 2, 1, 2)
    MUL_ACC(0, 1, 1, 2, 1, 3)
    MUL_ACC(1, 0, 1, 2, 2, 2)
    LOAD_INPUT_WITH_OFFSET(2, 3)
    MUL_ACC(1, 1, 1, 2, 2, 3)

    LOAD_WEIGHT(2, 0)

    MUL_ACC(0, 0, 2, 0, 2, 0)
    MUL_ACC(0, 1, 2, 0, 2, 1)
    LOAD_INPUT_WITH_OFFSET(3, 0)
    MUL_ACC(1, 0, 2, 0, 3, 0)
    LOAD_INPUT_WITH_OFFSET(3, 1)
    MUL_ACC(1, 1, 2, 0, 3, 1)

    LOAD_WEIGHT(2, 1)

    MUL_ACC(0, 0, 2, 1, 2, 1)
    MUL_ACC(0, 1, 2, 1, 2, 2)
    MUL_ACC(1, 0, 2, 1, 3, 1)
    LOAD_INPUT_WITH_OFFSET(3, 2)
    MUL_ACC(1, 1, 2, 1, 3, 2)

    LOAD_WEIGHT(2, 2)

    MUL_ACC(0, 0, 2, 2, 2, 2)
    MUL_ACC(0, 1, 2, 2, 2, 3)
    MUL_ACC(1, 0, 2, 2, 3, 2)
    LOAD_INPUT_WITH_OFFSET(3, 3)
    MUL_ACC(1, 1, 2, 2, 3, 3)

    // Perform requantization and store
    RQ_MUL(0123);
    RQ_MUL(4567);
    RQ_MUL(89ab);
    RQ_MUL(cdef);

    RQ_SHIFT(0123);
    RQ_SHIFT(4567);
    RQ_SHIFT(89ab);
    RQ_SHIFT(cdef);

    // At this point we narrow before adding the C offset
    NARROW_ACC(0, 0)
    NARROW_ACC(0, 1)
    NARROW_ACC(1, 0)
    NARROW_ACC(1, 1)

    RQ_OFFSET(01234567);
    RQ_OFFSET(89abcdef);

    RQ_CLAMP(01234567);
    RQ_CLAMP(89abcdef);

    // Narrow and store
    NARROW_AND_STORE(0, 0)
    NARROW_AND_STORE(0, 1)
    NARROW_AND_STORE(1, 0)
    NARROW_AND_STORE(1, 1)

    n_channels -= 16;
    c += 16;
  } while (n_channels);
}

}  // namespace depthwise
}  // namespace ops
}  // namespace kai

#endif  // defined(__aarch64__)
