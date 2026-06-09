# Copyright (c) 2025 Wenchao Huang <physhuangwenchao@gmail.com>
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

cmake_minimum_required(VERSION 3.18)

# Fetch GoogleTest
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Target name
set(TARGET_NAME nucleus-tests)

# Source files
file(GLOB TEST_CPP_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB TEST_CUDA_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")

# Create the executable target
add_executable(${TARGET_NAME} ${TEST_CPP_SOURCES} ${TEST_CUDA_SOURCES})

# Link directories
target_link_libraries(${TARGET_NAME} PRIVATE nucleus GTest::gtest GTest::gtest_main)

# MSVC settings
if(MSVC)
    target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/MP /W3 /WX /utf-8>)
endif()

# CMake properties
set_target_properties(${TARGET_NAME} PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75"
    FOLDER "Tests"
)

# Enable CTest integration
include(GoogleTest)
gtest_discover_tests(${TARGET_NAME})