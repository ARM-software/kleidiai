#
# SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.16)

project(dwconv_clamp_f32_f32_f32p_planar_sme2 LANGUAGES C CXX ASM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(KLEIDIAI_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../)
include(FetchContent)

set(KLEIDIAI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(KLEIDIAI_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)

fetchcontent_declare(
    kleidiai
    SOURCE_DIR ${KLEIDIAI_PATH}
)

fetchcontent_makeavailable(kleidiai)

add_executable(
    dwconv_clamp_f32_f32_f32p_planar_sme2
    dwconv_clamp_f32_f32_f32p_planar_sme2.cpp
)

target_link_libraries(dwconv_clamp_f32_f32_f32p_planar_sme2 PRIVATE kleidiai)

target_compile_options(dwconv_clamp_f32_f32_f32p_planar_sme2
    PRIVATE "-march=armv8.2-a+sve+sve2;-fno-tree-vectorize"
)

target_compile_definitions(dwconv_clamp_f32_f32_f32p_planar_sme2
    PRIVATE $<$<CONFIG:Debug>:KAI_DEBUG>
)
