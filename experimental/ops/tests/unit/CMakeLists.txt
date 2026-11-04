#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

# Unit tests on Linux AArch64 systems
if(CMAKE_SYSTEM_NAME STREQUAL "Linux"
   AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")

    foreach(TEST_NAME IN ITEMS sve_vector_length sme_vector_length)
        set(test_target "${TEST_NAME}_test")

        # Define the test executable
        add_executable(${test_target} "${TEST_NAME}.cpp")
        target_include_directories(${test_target} PRIVATE "${PROJECT_SOURCE_DIR}")
        target_link_libraries(${test_target} PRIVATE kleidiai_ops)

        # Need to be able to compile without SVE or SME enabled
        target_compile_options(${test_target} PRIVATE ${KLEIDIAI_OPS_CCXX_FLAGS} -march=armv8-a)

        set_target_properties(
            ${test_target}
            PROPERTIES CXX_STANDARD ${KLEIDIAI_OPS_CXX_STANDARD}
                       CXX_STANDARD_REQUIRED ON
                       CXX_EXTENSIONS OFF)

        # Define the test to be recognised by ctest
        add_test(NAME ${TEST_NAME} COMMAND ${test_target})
        set_tests_properties(${TEST_NAME} PROPERTIES SKIP_RETURN_CODE 77)
    endforeach()
endif()
