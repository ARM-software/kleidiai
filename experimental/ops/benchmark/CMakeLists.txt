#
# SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

# -------------------------------------------------------------------------------------------------

message(STATUS "Entered benchmark directory")
list(APPEND CMAKE_MESSAGE_INDENT " ")

# Define GIT_HASH and GIT_BRANCH variables
include(KleidiAIOpsGetGitInfo)

# Generate version.cpp using GIT_HASH and GIT_BRANCH and stick it in the build directory
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/version.cpp.in
               ${CMAKE_CURRENT_BINARY_DIR}/src/version.cpp @ONLY)

set(REQUIRE_PERF_CPP FALSE)
if(NOT KLEIDIAI_OPS_ENABLE_BARE_METAL)
    if(KLEIDIAI_OPS_ENABLE_THREADS)
        if(KLEIDIAI_OPS_ENABLE_CYCLE_PROFILING)
            set(REQUIRE_PERF_CPP TRUE)
        endif()
    elseif(KLEIDIAI_OPS_ENABLE_CYCLE_TIMING OR KLEIDIAI_OPS_ENABLE_CYCLE_PROFILING)
        set(REQUIRE_PERF_CPP TRUE)
    endif()
endif()

set(GEMM_SOURCES
    # Main GEMM sources
    ${CMAKE_CURRENT_LIST_DIR}/gemm.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/misc.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/test_suite.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/scheduler.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/memory_allocator.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/file_dumper.cpp
    $<$<BOOL:${REQUIRE_PERF_CPP}>:${CMAKE_CURRENT_LIST_DIR}/src/perf.cpp>
    # Wrapper sources
    ${CMAKE_CURRENT_LIST_DIR}/src/gemm_bf16_new.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/gemm_s16.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/gemm_s8.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/gemm_u16.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/gemm_u8.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/gemm_u8s8.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/hgemm_new.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sgemm_new.cpp
    # Auto-generated version file
    ${CMAKE_CURRENT_BINARY_DIR}/src/version.cpp)

set(GEMM_EXEC_INCLUDE_DIRS
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    # Private library sources
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../kai/ops>
    # Public include headers
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)

# The 'gemm' benchmarking executable
add_executable(gemm ${GEMM_SOURCES})
target_link_libraries(gemm PRIVATE kleidiai_ops)
target_compile_options(gemm PRIVATE "${KLEIDIAI_OPS_ARCH};${KLEIDIAI_OPS_CCXX_FLAGS}")
target_compile_definitions(gemm PRIVATE "${KLEIDIAI_OPS_COMPILE_DEFINES}")
target_include_directories(gemm PRIVATE ${GEMM_EXEC_INCLUDE_DIRS})
set_target_properties(
    gemm
    PROPERTIES C_STANDARD ${KLEIDIAI_OPS_C_STANDARD}
               C_STANDARD_REQUIRED ON
               C_EXTENSIONS OFF
               CXX_STANDARD ${KLEIDIAI_OPS_CXX_STANDARD}
               CXX_STANDARD_REQUIRED ON
               CXX_EXTENSIONS OFF)

# Suppress warnings
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-Wno-format HAS_WNO_FORMAT)
check_cxx_compiler_flag(-Wno-psabi HAS_WNO_PSABI)
check_cxx_compiler_flag(-Wno-unused-result HAS_WNO_UNUSED_RESULT)
target_compile_options(
    gemm PRIVATE $<$<BOOL:${HAS_WNO_PSABI}>:-Wno-psabi> $<$<BOOL:${HAS_WNO_FORMAT}>:-Wno-format>
                 $<$<BOOL:${HAS_WNO_UNUSED_RESULT}>:-Wno-unused-result>)

# Link to threading library
if(KLEIDIAI_OPS_ENABLE_THREADS)
    find_package(Threads REQUIRED)
    if(NOT KLEIDIAI_OPS_ENABLE_ANDROID)
        if(KLEIDIAI_OPS_ENABLE_NATIVE)
            target_link_libraries(gemm PRIVATE Threads::Threads)
        else()
            target_link_libraries(gemm PRIVATE $<LINK_LIBRARY:WHOLE_ARCHIVE,Threads::Threads>)
        endif()
    endif()
endif()

# Set the output directory to the project build directory
set_target_properties(gemm PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

message(STATUS "Leaving benchmark directory")
# -------------------------------------------------------------------------------------------------
