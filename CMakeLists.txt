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

# Project name: Nucleus
project(Nucleus LANGUAGES CXX CUDA)

# Find CUDA Toolkit (needed to expose include dirs to C++ compilation units)
find_package(CUDAToolkit REQUIRED)

# Organize targets into folders
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Library options
option(NUCLEUS_BUILD_TESTS "Build tests for Nucleus library" OFF)
option(NUCLEUS_BUILD_EXAMPLES "Build examples for Nucleus library" OFF)
option(NUCLEUS_BUILD_SHARED_LIB "Build Nucleus as shared library" ON)

# C++ standard options (adjust as needed)
set(NUCLEUS_CPP_STANDARD "20" CACHE STRING "C++ standard (20/23/26)")
set_property(CACHE NUCLEUS_CPP_STANDARD PROPERTY STRINGS 20 23 26)

# Target name
set(TARGET_NAME nucleus)

# Set C++ standard
set(CMAKE_CXX_STANDARD ${NUCLEUS_CPP_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD ${NUCLEUS_CPP_STANDARD})
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Source files
file(GLOB NUCLEUS_SOURCES CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/include/nucleus/*.h"
    "${PROJECT_SOURCE_DIR}/include/nucleus/*.cuh"
    "${PROJECT_SOURCE_DIR}/src/*.cpp"
    "${PROJECT_SOURCE_DIR}/src/*.cu"
)

# Build library
if(NUCLEUS_BUILD_SHARED_LIB)
    # Create the shared library
    add_library(${TARGET_NAME} SHARED ${NUCLEUS_SOURCES})

    # Link directories
    target_link_libraries(${TARGET_NAME} PUBLIC CUDA::cudart)

    # Export macro
    target_compile_definitions(${TARGET_NAME} PRIVATE NS_API_BUILD)

    # Export header
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/${TARGET_NAME}_export_shared.h
        ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}_export.h
        COPYONLY
    )
else()
    # Create the static library
    add_library(${TARGET_NAME} STATIC ${NUCLEUS_SOURCES})

    # Export header
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/${TARGET_NAME}_export_static.h
        ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}_export.h
        COPYONLY
    )
endif()

# Include directories
target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/nucleus)

# MSVC settings
if(MSVC)
    target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/MP /W4 /WX /utf-8>)

    # CUDA 13.2+ (CCCL) requires the conforming preprocessor; propagate to all consumers via PUBLIC
    target_compile_options(${TARGET_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)

    # Ensure UTF-8 encoding for CUDA compilation as well; propagate to all consumers via PUBLIC
    target_compile_options(${TARGET_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
endif()

# Load the custom helper that adds a dummy.cpp source file
if(MSVC)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/add_dummy_cpp.cmake)
endif()

# Optional: Add tests if enabled
if(NUCLEUS_BUILD_TESTS)
    add_subdirectory(tests)
endif()

# Optional: Add examples if enabled
if(NUCLEUS_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()