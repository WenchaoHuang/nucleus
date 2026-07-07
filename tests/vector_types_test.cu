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

#include <cuda_runtime.h>
#include <nucleus/vector_types.h>

/*********************************************************************************
****************************    vector_types_test    *****************************
*********************************************************************************/

static_assert(ns::BinaryCompatible<ns::int2, ::int2>::value);
static_assert(ns::BinaryCompatible<ns::int3, ::int3>::value);
static_assert(ns::BinaryCompatible<ns::int4, ::int4>::value);

static_assert(ns::BinaryCompatible<ns::uint2, ::uint2>::value);
static_assert(ns::BinaryCompatible<ns::uint3, ::uint3>::value);
static_assert(ns::BinaryCompatible<ns::uint4, ::uint4>::value);

static_assert(ns::BinaryCompatible<ns::char2, ::char2>::value);
static_assert(ns::BinaryCompatible<ns::char3, ::char3>::value);
static_assert(ns::BinaryCompatible<ns::char4, ::char4>::value);

static_assert(ns::BinaryCompatible<ns::uchar2, ::uchar2>::value);
static_assert(ns::BinaryCompatible<ns::uchar3, ::uchar3>::value);
static_assert(ns::BinaryCompatible<ns::uchar4, ::uchar4>::value);

static_assert(ns::BinaryCompatible<ns::short2, ::short2>::value);
static_assert(ns::BinaryCompatible<ns::short3, ::short3>::value);
static_assert(ns::BinaryCompatible<ns::short4, ::short4>::value);

static_assert(ns::BinaryCompatible<ns::float2, ::float2>::value);
static_assert(ns::BinaryCompatible<ns::float3, ::float3>::value);
static_assert(ns::BinaryCompatible<ns::float4, ::float4>::value);

static_assert(ns::BinaryCompatible<ns::double2, ::double2>::value);
static_assert(ns::BinaryCompatible<ns::double3, ::double3>::value);
#if __CUDACC_VER_MAJOR__ >= 13
static_assert(ns::BinaryCompatible<ns::double4, ::double4>::value);
static_assert(ns::BinaryCompatible<ns::double4_16a, ::double4_16a>::value);
static_assert(ns::BinaryCompatible<ns::double4_32a, ::double4_32a>::value);
#else
static_assert(ns::BinaryCompatible<ns::double4, ::double4>::value);
static_assert(ns::BinaryCompatible<ns::double4_16a, ::double4>::value);
#endif

static_assert(ns::BinaryCompatible<ns::ushort2, ::ushort2>::value);
static_assert(ns::BinaryCompatible<ns::ushort3, ::ushort3>::value);
static_assert(ns::BinaryCompatible<ns::ushort4, ::ushort4>::value);

static_assert(ns::BinaryCompatible<ns::longlong2, ::longlong2>::value);
static_assert(ns::BinaryCompatible<ns::longlong3, ::longlong3>::value);
#if __CUDACC_VER_MAJOR__ >= 13
static_assert(ns::BinaryCompatible<ns::longlong4, ::longlong4>::value);
static_assert(ns::BinaryCompatible<ns::longlong4_16a, ::longlong4_16a>::value);
static_assert(ns::BinaryCompatible<ns::longlong4_32a, ::longlong4_32a>::value);
#else
static_assert(ns::BinaryCompatible<ns::longlong4, ::longlong4>::value);
static_assert(ns::BinaryCompatible<ns::longlong4_16a, ::longlong4>::value);
#endif

static_assert(ns::BinaryCompatible<ns::ulonglong2, ::ulonglong2>::value);
static_assert(ns::BinaryCompatible<ns::ulonglong3, ::ulonglong3>::value);
#if __CUDACC_VER_MAJOR__ >= 13
static_assert(ns::BinaryCompatible<ns::ulonglong4, ::ulonglong4>::value);
static_assert(ns::BinaryCompatible<ns::ulonglong4_16a, ::ulonglong4>::value);
#else
static_assert(ns::BinaryCompatible<ns::ulonglong4, ::ulonglong4>::value);
static_assert(ns::BinaryCompatible<ns::ulonglong4_16a, ::ulonglong4>::value);
#endif