//
// SPDX-FileCopyrightText: Copyright 2021-2022, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace kai {
namespace ops {
namespace depthwise {

struct interleave_sme_u8q_3x3_dot
{
  static void pack_parameters(unsigned int, void *, const int32_t *, const uint8_t *, const kai::ops::Requantize32 &, size_t, size_t);
  static size_t get_packed_size(const DepthwiseArgs &);
};

struct interleave_sme2_u8q_3x3_dot
{
  static void pack_parameters(unsigned int, void *, const int32_t *, const uint8_t *, const kai::ops::Requantize32 &, size_t, size_t);
  static size_t get_packed_size(const DepthwiseArgs &);
};

struct interleave_sme_s8q_3x3_dot
{
  static void pack_parameters(unsigned int, void *, const int32_t *, const int8_t *, const kai::ops::Requantize32 &, size_t, size_t);
  static size_t get_packed_size(const DepthwiseArgs &);
};

struct interleave_sme2_s8q_3x3_dot
{
  static void pack_parameters(unsigned int, void *, const int32_t *, const int8_t *, const kai::ops::Requantize32 &, size_t, size_t);
  static size_t get_packed_size(const DepthwiseArgs &);
};

struct interleave_sve_u8q_3x3_dot
{
  static void pack_parameters(unsigned int, void *, const int32_t *, const uint8_t *, const kai::ops::Requantize32 &, size_t, size_t);
  static size_t get_packed_size(const DepthwiseArgs &);
};

struct interleave_sve_s8q_3x3_dot
{
  static void pack_parameters(unsigned int, void *, const int32_t *, const int8_t *, const kai::ops::Requantize32 &, size_t, size_t);
  static size_t get_packed_size(const DepthwiseArgs &);
};


struct interleave_a64_u8q_3x3_dot
{
  static void pack_parameters(unsigned int, void *, const int32_t *, const uint8_t *, const kai::ops::Requantize32 &, size_t, size_t);
  static size_t get_packed_size(const DepthwiseArgs &);
};

struct interleave_a64_s8q_3x3_dot
{
  static void pack_parameters(unsigned int, void *, const int32_t *, const int8_t *, const kai::ops::Requantize32 &, size_t, size_t);
  static size_t get_packed_size(const DepthwiseArgs &);
};

}  // namespace depthwise
}  // namespace ops
}  // namespace kai
