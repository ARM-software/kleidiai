
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

set(triple aarch64-linux-gnu)
set(TOOLS_PATH /prj/qct/chips/swarch/tools)
set(CROSS_COMPILE_LLVM_PATH ${TOOLS_PATH}/clang-16.0.0)

set(CMAKE_C_COMPILER "${CROSS_COMPILE_LLVM_PATH}/bin/clang")
set(CMAKE_C_COMPILER_TARGET ${triple})
set(CMAKE_ASM_COMPILER_TARGET ${triple})
set(CMAKE_C_FLAGS "-march=armv8.5-a+sme+sve2")
set(CMAKE_ASM_FLAGS "-march=armv8.5-a+sme+sve2")

set(CMAKE_CXX_COMPILER "${CROSS_COMPILE_LLVM_PATH}/bin/clang++")
set(CMAKE_ASM_COMPILER "${CROSS_COMPILE_LLVM_PATH}/bin/clang++")
set(CMAKE_CXX_COMPILER_TARGET ${triple})
set(CMAKE_CXX_FLAGS "-march=armv8.5-a+sve2+sme")

set(CMAKE_SHARED_LINKER_FLAGS "-static")
set(CMAKE_MODULE_LINKER_FLAGS "-static")
set(CMAKE_EXE_LINKER_FLAGS "-static")
