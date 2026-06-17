#
# SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

# -------------------------------------------------------------------------------------------------
# cmake-format: off

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR arm)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

if(NOT DEFINED COMPILER_TRIPLET)
    set(COMPILER_TRIPLET arm-none-eabi)
endif()
if(NOT DEFINED CMAKE_C_COMPILER)
    set(CMAKE_C_COMPILER ${COMPILER_TRIPLET}-gcc)
endif()
if(NOT DEFINED CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER ${COMPILER_TRIPLET}-g++)
endif()

# Try to locate the specs file using the compiler
set(_KLEIDIAI_OPS_SPECS_NAME aprofile-validation.specs)
set(_KLEIDIAI_OPS_FULL_SPECS_PATH)

execute_process(
    COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=${_KLEIDIAI_OPS_SPECS_NAME}
    OUTPUT_VARIABLE _KLEIDIAI_OPS_SPECS_SEARCH_OUTPUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

if(_KLEIDIAI_OPS_SPECS_SEARCH_OUTPUT AND (NOT _KLEIDIAI_OPS_SPECS_SEARCH_OUTPUT STREQUAL "${_KLEIDIAI_OPS_SPECS_NAME}") AND (EXISTS "${_KLEIDIAI_OPS_SPECS_SEARCH_OUTPUT}"))
    set(_KLEIDIAI_OPS_FULL_SPECS_PATH "${_KLEIDIAI_OPS_SPECS_SEARCH_OUTPUT}")
else()
    set(_KLEIDIAI_OPS_FULL_SPECS_PATH "${_KLEIDIAI_OPS_SPECS_NAME}")
    message(WARNING "Could not resolve '${_KLEIDIAI_OPS_SPECS_NAME}' via -print-file-name.")
endif()

set(KLEIDIAI_OPS_ENABLE_NATIVE OFF CACHE BOOL "Enable native build" FORCE)
set(KLEIDIAI_OPS_ENABLE_BARE_METAL ON CACHE BOOL "Enable bare-metal mode" FORCE)
set(KLEIDIAI_OPS_ENABLE_AARCH32 ON CACHE BOOL "Enable AArch32 mode" FORCE)
set(KLEIDIAI_OPS_ENABLE_ANDROID OFF CACHE BOOL "Enable Android build" FORCE)

set(KLEIDIAI_OPS_CCXX_FLAGS_INIT -specs=${_KLEIDIAI_OPS_FULL_SPECS_PATH} -mfloat-abi=hard -mfpu=neon-fp16 -mfp16-format=ieee CACHE STRING "Base C/CXX flags.")
set(KLEIDIAI_OPS_CCXX_FLAGS_DEBUG "" CACHE STRING "Debug config specific C/CXX flags.")
set(KLEIDIAI_OPS_CCXX_FLAGS_RELEASE "" CACHE STRING "Release config specific C/CXX flags.")

set(KLEIDIAI_OPS_LINKER_FLAGS_INIT -specs=${_KLEIDIAI_OPS_FULL_SPECS_PATH} -mfloat-abi=hard -mfpu=neon-fp16 CACHE STRING "Base linker flags.")
set(KLEIDIAI_OPS_LINKER_FLAGS_DEBUG "" CACHE STRING "Debug config specific linker flags.")
set(KLEIDIAI_OPS_LINKER_FLAGS_RELEASE "" CACHE STRING "Release config specific linker flags.")

# cmake-format: on
# -------------------------------------------------------------------------------------------------
