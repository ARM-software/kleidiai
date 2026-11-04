#
# SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#
include(FetchContent)

# Set timestamp of extracted contents to time of extraction.
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

fetchcontent_declare(googletest
    URL ${CMAKE_CURRENT_SOURCE_DIR}/third_party/googletest-v1.18.0.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

fetchcontent_makeavailable(googletest)
