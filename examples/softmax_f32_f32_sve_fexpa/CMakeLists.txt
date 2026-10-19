#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.16)

project(softmax_f32_f32_sve_fexpa LANGUAGES C CXX ASM)

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

add_executable(softmax_f32_f32_sve_fexpa
    softmax_f32_f32_sve_fexpa.cpp
)

target_link_libraries(softmax_f32_f32_sve_fexpa PRIVATE kleidiai)

target_compile_options(softmax_f32_f32_sve_fexpa
    PRIVATE "-march=armv8-a+sve;-fno-tree-vectorize"
)

target_compile_definitions(softmax_f32_f32_sve_fexpa
    PRIVATE $<$<CONFIG:Debug>:KAI_DEBUG>
)
