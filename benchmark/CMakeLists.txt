#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

# https://github.com/google/benchmark/issues/351
if(NOT (${CMAKE_SYSTEM_PROCESSOR} MATCHES "ARM"))
    set(CMAKE_CROSSCOMPILING TRUE CACHE INTERNAL "")
    set(HAVE_POSIX_REGEX FALSE CACHE INTERNAL "")
endif()

set(BENCHMARK_DOWNLOAD_DEPENDENCIES "OFF")
set(BENCHMARK_ENABLE_TESTING "OFF")
set(BENCHMARK_ENABLE_GTEST_TESTS "OFF")

include(FetchGBench)

# Single benchmark binary combining matmul, imatmul and dwconv
add_executable(kleidiai_benchmark
    benchmark/main.cpp
    benchmark/dwconv/dwconv_registry.cpp
    benchmark/matmul/matmul_registry.cpp
    benchmark/imatmul/imatmul_registry.cpp
    test/common/cpu_info.cpp
    test/common/data_type.cpp
)

set_source_files_properties(benchmark/matmul/matmul_registry.cpp
    PROPERTIES COMPILE_OPTIONS "-fno-tree-vectorize;-fno-tree-slp-vectorize;-march=armv8-a+sve+sve2${KLEIDIAI_INTERNAL_EXTRA_ARCH}")
set_source_files_properties(benchmark/imatmul/imatmul_registry.cpp
    PROPERTIES COMPILE_OPTIONS "-fno-tree-vectorize;-fno-tree-slp-vectorize;-march=armv8-a+sve+sve2${KLEIDIAI_INTERNAL_EXTRA_ARCH}")

target_link_libraries(kleidiai_benchmark
    PRIVATE
        kleidiai
        benchmark::benchmark
)

if(APPLE)
    find_library(KPERF_FRAMEWORK kperf
        PATHS /System/Library/PrivateFrameworks
        NO_DEFAULT_PATH
    )
    find_library(KPERFDATA_FRAMEWORK kperfdata
        PATHS /System/Library/PrivateFrameworks
        NO_DEFAULT_PATH
    )

    if(KPERF_FRAMEWORK AND KPERFDATA_FRAMEWORK)
        target_link_libraries(kleidiai_benchmark
            PRIVATE
                "${KPERF_FRAMEWORK}"
                "${KPERFDATA_FRAMEWORK}"
        )
        target_compile_definitions(kleidiai_benchmark PRIVATE KAI_BENCHMARK_KPERF=1)
    else()
        target_compile_definitions(kleidiai_benchmark PRIVATE KAI_BENCHMARK_KPERF=0)
    endif()
endif()
