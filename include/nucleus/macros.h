/**
 *	Copyright (c) 2025 Wenchao Huang <physhuangwenchao@gmail.com>
 *
 *	Permission is hereby granted, free of charge, to any person obtaining a copy
 *	of this software and associated documentation files (the "Software"), to deal
 *	in the Software without restriction, including without limitation the rights
 *	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *	copies of the Software, and to permit persons to whom the Software is
 *	furnished to do so, subject to the following conditions:
 *
 *	The above copyright notice and this permission notice shall be included in all
 *	copies or substantial portions of the Software.
 *
 *	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *	SOFTWARE.
 */
#pragma once

#include <cstdio>
#include <assert.h>
#include "nucleus_export.h"

#pragma warning(disable: 4201)		//!	Nonstandard extension used: nameless struct/union.

/*********************************************************************************
*******************************    C++ Version     *******************************
*********************************************************************************/

#if defined(_MSVC_LANG)
	#define NS_CPLUSPLUS			_MSVC_LANG
#else
	#define NS_CPLUSPLUS			__cplusplus
#endif

#if NS_CPLUSPLUS > 202002L
	#define NS_CPP_VERSION 23
#elif NS_CPLUSPLUS > 201703L
	#define NS_CPP_VERSION 20
#elif NS_CPLUSPLUS > 201402L
	#define NS_CPP_VERSION 17
#else
	#error "Requires at least c++ standard version 17"
#endif

#define NS_HAS_CXX_20				(NS_CPP_VERSION >= 20)
#define NS_HAS_CXX_17				(NS_CPP_VERSION >= 17)

/*********************************************************************************
****************************    Compiling Numbers     ****************************
*********************************************************************************/

#define NS_MIN(a, b)				(((a) < (b)) ? (a) : (b))
#define NS_MAX(a, b)				(((a) > (b)) ? (a) : (b))

/*********************************************************************************
***************************    CUDA Compatibilities    ***************************
*********************************************************************************/

#if defined(__CUDACC__)
	#define	NS_INLINE						__inline__
	#define NS_ALIGN(n)						__align__(n)
	#define NS_FORCE_INLINE					__forceinline__
	#define	NS_CUDA_CALLABLE				__host__ __device__
	#define	NS_CUDA_CALLABLE_INLINE			__host__ __device__ __inline__
#else
	#define	NS_INLINE						inline
	#define NS_ALIGN(n)						alignas(n)
	#if defined(_MSC_VER)
		#define NS_FORCE_INLINE				__forceinline
	#else
		#define NS_FORCE_INLINE				__attribute__((always_inline)) inline
	#endif
	#define	NS_CUDA_CALLABLE
	#define	NS_CUDA_CALLABLE_INLINE			inline

	#ifndef __device__
		#define __device__
	#endif
#endif

#if defined(__CUDA_ARCH__)
	#define	NS_ASSERT(expression)
#else
	#define	NS_ASSERT(expression)			assert(expression)
#endif

/*********************************************************************************
**********************************    Utils    ***********************************
*********************************************************************************/

#define NS_NODISCARD						[[nodiscard]]
#if defined(_MSC_VER)
	#define NS_NOVTABLE						__declspec(novtable)
#else
	#define NS_NOVTABLE
#endif

#if defined(DEBUG) || defined(_DEBUG)
	#define NS_DEBUG
#endif

/*********************************************************************************
*******************************    Noncopyable    ********************************
*********************************************************************************/

#define NS_NONCOPYABLE(ClassName)												\
																				\
	ClassName(const ClassName&) = delete;										\
																				\
	ClassName & operator=(const ClassName&) = delete;							\

/*********************************************************************************
********************************    Namespace    *********************************
*********************************************************************************/

#ifndef NS_NAMESPACE
	#define NS_NAMESPACE				ns
#endif

#define NS_USING_NAMESPACE				using namespace NS_NAMESPACE;

namespace NS_NAMESPACE { namespace dev {} }
namespace dev = NS_NAMESPACE::dev;