#
# SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

# -------------------------------------------------------------------------------------------------
# cmake-format: off

set(CMAKE_SYSTEM_NAME Android)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

if(NOT DEFINED CMAKE_ANDROID_NDK)
    if(DEFINED ENV{ANDROID_NDK_HOME})
        set(CMAKE_ANDROID_NDK "$ENV{ANDROID_NDK_HOME}" CACHE PATH "" FORCE)
    elseif(DEFINED ENV{ANDROID_NDK})
        set(CMAKE_ANDROID_NDK "$ENV{ANDROID_NDK}" CACHE PATH "" FORCE)
    else()
        message(FATAL_ERROR "Set CMAKE_ANDROID_NDK, ANDROID_NDK_HOME, or ANDROID_NDK to your NDK root.")
    endif()
endif()

set(ANDROID_ABI arm64-v8a CACHE STRING "" FORCE)
set(ANDROID_PLATFORM android-30 CACHE STRING "" FORCE) # for clang++ 30

# See: https://developer.android.com/ndk/guides/cmake#the_new_toolchain_file
include("${CMAKE_ANDROID_NDK}/build/cmake/android.toolchain.cmake")

set(KLEIDIAI_OPS_ENABLE_NATIVE OFF CACHE BOOL "" FORCE)
set(KLEIDIAI_OPS_ENABLE_BARE_METAL OFF CACHE BOOL "" FORCE)
set(KLEIDIAI_OPS_ENABLE_AARCH32 OFF CACHE BOOL "" FORCE)
set(KLEIDIAI_OPS_ENABLE_ANDROID ON CACHE BOOL "" FORCE)

set(KLEIDIAI_OPS_CCXX_FLAGS_INIT -Wno-overlength-strings CACHE STRING "")
set(KLEIDIAI_OPS_CCXX_FLAGS_DEBUG -Wall -pedantic CACHE STRING "")
set(KLEIDIAI_OPS_CCXX_FLAGS_RELEASE -O3 CACHE STRING "")

set(KLEIDIAI_OPS_LINKER_FLAGS_INIT -fPIE -pie -static-libstdc++ CACHE STRING "")
set(KLEIDIAI_OPS_LINKER_FLAGS_DEBUG "" CACHE STRING "")
set(KLEIDIAI_OPS_LINKER_FLAGS_RELEASE "" CACHE STRING "")

# cmake-format: on
# -------------------------------------------------------------------------------------------------
