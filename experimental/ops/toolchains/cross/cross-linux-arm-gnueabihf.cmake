#
# SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

# -------------------------------------------------------------------------------------------------
# cmake-format: off

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

if(NOT DEFINED COMPILER_TRIPLET)
    set(COMPILER_TRIPLET arm-none-linux-gnueabihf)
endif()
if(NOT DEFINED CMAKE_C_COMPILER)
    set(CMAKE_C_COMPILER ${COMPILER_TRIPLET}-gcc)
endif()
if(NOT DEFINED CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER ${COMPILER_TRIPLET}-g++)
endif()

set(KLEIDIAI_OPS_ENABLE_NATIVE OFF CACHE BOOL "Enable native build" FORCE)
set(KLEIDIAI_OPS_ENABLE_BARE_METAL OFF CACHE BOOL "Enable bare-metal mode" FORCE)
set(KLEIDIAI_OPS_ENABLE_AARCH32 ON CACHE BOOL "Enable AArch32 mode" FORCE)
set(KLEIDIAI_OPS_ENABLE_ANDROID OFF CACHE BOOL "Enable Android build" FORCE)

set(KLEIDIAI_OPS_CCXX_FLAGS_INIT -mfloat-abi=hard -mfpu=neon-fp16 -mfp16-format=ieee -Wno-psabi CACHE STRING "Base C/CXX flags.")
set(KLEIDIAI_OPS_CCXX_FLAGS_DEBUG "" CACHE STRING "Debug C/CXX flags.")
set(KLEIDIAI_OPS_CCXX_FLAGS_RELEASE "" CACHE STRING "Release C/CXX flags.")

set(KLEIDIAI_OPS_LINKER_FLAGS_INIT -static CACHE STRING "Base linker flags.")
set(KLEIDIAI_OPS_LINKER_FLAGS_DEBUG "" CACHE STRING "Debug linker flags.")
set(KLEIDIAI_OPS_LINKER_FLAGS_RELEASE "" CACHE STRING "Release linker flags.")

# cmake-format: on
# -------------------------------------------------------------------------------------------------
