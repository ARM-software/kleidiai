#
# SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

# -------------------------------------------------------------------------------------------------
# cmake-format: off

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

if(NOT DEFINED CMAKE_C_COMPILER)
    set(CMAKE_C_COMPILER gcc)
endif()
if(NOT DEFINED CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER g++)
endif()

set(KLEIDIAI_OPS_ENABLE_NATIVE ON CACHE BOOL "Enable native build" FORCE)
set(KLEIDIAI_OPS_ENABLE_BARE_METAL OFF CACHE BOOL "Enable bare-metal mode" FORCE)
set(KLEIDIAI_OPS_ENABLE_AARCH32 OFF CACHE BOOL "Enable AArch32 mode" FORCE)
set(KLEIDIAI_OPS_ENABLE_ANDROID OFF CACHE BOOL "Enable Android build" FORCE)

set(KLEIDIAI_OPS_CCXX_FLAGS_INIT "" CACHE STRING "Base C/CXX flags.")
set(KLEIDIAI_OPS_CCXX_FLAGS_DEBUG "" CACHE STRING "Debug config specific C/CXX flags.")
set(KLEIDIAI_OPS_CCXX_FLAGS_RELEASE "" CACHE STRING "Release config specific C/CXX flags.")

set(KLEIDIAI_OPS_LINKER_FLAGS_INIT -static CACHE STRING "Base linker flags.")
set(KLEIDIAI_OPS_LINKER_FLAGS_DEBUG "" CACHE STRING "Debug config specific linker flags.")
set(KLEIDIAI_OPS_LINKER_FLAGS_RELEASE "" CACHE STRING "Release config specific linker flags.")

# cmake-format: on
# -------------------------------------------------------------------------------------------------
