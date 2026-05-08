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

#include "fwd.h"
#include <type_traits>

#ifdef _MSC_VER
	#pragma warning(push)
	#pragma warning(disable : 4324)		// structure was padded due to alignment specifier
#endif

namespace NS_NAMESPACE
{
	/*****************************************************************************
	****************************    Vector Types     *****************************
	*****************************************************************************/

	/**
	 *	@brief	Type aliases for commonly used vector types, conditionally defined.
	 * 
	 *	In non-CUDA environments (__CUDACC__ not defined), custom Vec2, Vec3,
	 *	and Vec4 templates are used to define vector types, ensuring no dependency on
	 *	CUDA headers. This provides a consistent interface for scalar and vector types
	 *	across platforms, enhancing modularity and portability.
	 * 
	 *	This approach avoids exposing CUDA headers in the interface, allowing the
	 *	header to be included in non-CUDA projects.
	 */
	template<typename Type, int Align> struct NS_ALIGN(Align) Vec2 { Type x, y; };
	template<typename Type, int Align> struct NS_ALIGN(Align) Vec3 { Type x, y, z; };
	template<typename Type, int Align> struct NS_ALIGN(Align) Vec4 { Type x, y, z, w; };
#ifndef __CUDACC__
	using int2 = Vec2<int, 8>;
	using int3 = Vec3<int, 4>;
	using int4 = Vec4<int, 16>;
	using int3_16a = Vec3<int, 16>;

	using char2 = Vec2<char, 2>;
	using char3 = Vec3<char, 1>;
	using char4 = Vec4<char, 4>;

	using short2 = Vec2<short, 4>;
	using short3 = Vec3<short, 2>;
	using short4 = Vec4<short, 8>;

	using float2 = Vec2<float, 8>;
	using float3 = Vec3<float, 4>;
	using float4 = Vec4<float, 16>;
	using float3_16a = Vec3<float, 16>;

	using double2 = Vec2<double, 16>;
	using double3 = Vec3<double,  8>;
	using double4 = Vec4<double, 16>;
	using double4_16a = Vec4<double, 16>;
	using double4_32a = Vec4<double, 32>;

	using uint = unsigned int;
	using uint2 = Vec2<unsigned int, 8>;
	using uint3 = Vec3<unsigned int, 4>;
	using uint4 = Vec4<unsigned int, 16>;
	using uint3_16a = Vec3<unsigned int, 16>;

	using longlong = long long;
	using longlong2 = Vec2<long long, 16>;
	using longlong3 = Vec3<long long,  8>;
	using longlong4 = Vec4<long long, 16>;
	using longlong4_16a = Vec4<long long, 16>;
	using longlong4_32a = Vec4<long long, 32>;

	using uchar = unsigned char;
	using uchar2 = Vec2<unsigned char, 2>;
	using uchar3 = Vec3<unsigned char, 1>;
	using uchar4 = Vec4<unsigned char, 4>;

	using ushort = unsigned short;
	using ushort2 = Vec2<unsigned short, 4>;
	using ushort3 = Vec3<unsigned short, 2>;
	using ushort4 = Vec4<unsigned short, 8>;

	using ulonglong = unsigned long long;
	using ulonglong2 = Vec2<unsigned long long, 16>;
	using ulonglong3 = Vec3<unsigned long long,  8>;
	using ulonglong4 = Vec4<unsigned long long, 16>;
	using ulonglong4_16a = Vec4<unsigned long long, 16>;
	using ulonglong4_32a = Vec4<unsigned long long, 32>;
#else
	using int2 = ::int2;									static_assert(BinaryCompatible<Vec2<int, 8>, int2>::value);
	using int3 = ::int3;									static_assert(BinaryCompatible<Vec3<int, 4>, int3>::value);
	using int4 = ::int4;									static_assert(BinaryCompatible<Vec4<int, 16>, int4>::value);
	using int3_16a = Vec3<int, 16>;							static_assert(BinaryCompatible<Vec3<int, 16>, int3_16a>::value);

	using uint = unsigned int;
	using uint2 = ::uint2;									static_assert(BinaryCompatible<Vec2<unsigned int, 8>, uint2>::value);
	using uint3 = ::uint3;									static_assert(BinaryCompatible<Vec3<unsigned int, 4>, uint3>::value);
	using uint4 = ::uint4;									static_assert(BinaryCompatible<Vec4<unsigned int, 16>, uint4>::value);
	using uint3_16a = Vec3<unsigned int, 16>;				static_assert(BinaryCompatible<Vec3<unsigned int, 16>, uint3_16a>::value);
	
	using char2 = ::char2;									static_assert(BinaryCompatible<Vec2<char, 2>, char2>::value);
	using char3 = ::char3;									static_assert(BinaryCompatible<Vec3<char, 1>, char3>::value);
	using char4 = ::char4;									static_assert(BinaryCompatible<Vec4<char, 4>, char4>::value);

	using uchar = unsigned char;
	using uchar2 = ::uchar2;								static_assert(BinaryCompatible<Vec2<unsigned char, 2>, uchar2>::value);
	using uchar3 = ::uchar3;								static_assert(BinaryCompatible<Vec3<unsigned char, 1>, uchar3>::value);
	using uchar4 = ::uchar4;								static_assert(BinaryCompatible<Vec4<unsigned char, 4>, uchar4>::value);

	using short2 = ::short2;								static_assert(BinaryCompatible<Vec2<short, 4>, short2>::value);
	using short3 = ::short3;								static_assert(BinaryCompatible<Vec3<short, 2>, short3>::value);
	using short4 = ::short4;								static_assert(BinaryCompatible<Vec4<short, 8>, short4>::value);

	using float2 = ::float2;								static_assert(BinaryCompatible<Vec2<float, 8>, float2>::value);
	using float3 = ::float3;								static_assert(BinaryCompatible<Vec3<float, 4>, float3>::value);
	using float4 = ::float4;								static_assert(BinaryCompatible<Vec4<float, 16>, float4>::value);
	using float3_16a = Vec3<float, 16>;						static_assert(BinaryCompatible<Vec3<float, 16>, float3_16a>::value);

	using double2 = ::double2;								static_assert(BinaryCompatible<Vec2<double, 16>, double2>::value);
	using double3 = ::double3;								static_assert(BinaryCompatible<Vec3<double,  8>, double3>::value);
#if __CUDACC_VER_MAJOR__ >= 13
	using double4 = ::double4_16a;							static_assert(BinaryCompatible<Vec4<double, 16>, double4>::value);
	using double4_16a = ::double4_16a;						static_assert(BinaryCompatible<Vec4<double, 16>, double4_16a>::value);
	using double4_32a = ::double4_32a;						static_assert(BinaryCompatible<Vec4<double, 32>, double4_32a>::value);
#else
	using double4 = ::double4;								static_assert(BinaryCompatible<Vec4<double, 16>, double4>::value);
	using double4_16a = ::double4;							static_assert(BinaryCompatible<Vec4<double, 16>, double4_16a>::value);
	using double4_32a = Vec4<double, 32>;					static_assert(BinaryCompatible<Vec4<double, 32>, double4_32a>::value);
#endif

	using ushort = unsigned short;
	using ushort2 = ::ushort2;								static_assert(BinaryCompatible<Vec2<unsigned short, 4>, ushort2>::value);
	using ushort3 = ::ushort3;								static_assert(BinaryCompatible<Vec3<unsigned short, 2>, ushort3>::value);
	using ushort4 = ::ushort4;								static_assert(BinaryCompatible<Vec4<unsigned short, 8>, ushort4>::value);

	using longlong = long long;
	using longlong2 = ::longlong2;							static_assert(BinaryCompatible<Vec2<long long, 16>, longlong2>::value);
	using longlong3 = ::longlong3;							static_assert(BinaryCompatible<Vec3<long long, 8>, longlong3>::value);
#if __CUDACC_VER_MAJOR__ >= 13
	using longlong4 = ::longlong4_16a;						static_assert(BinaryCompatible<Vec4<long long, 16>, longlong4>::value);
	using longlong4_16a = ::longlong4_16a;					static_assert(BinaryCompatible<Vec4<long long, 16>, longlong4_16a>::value);
	using longlong4_32a = ::longlong4_32a;					static_assert(BinaryCompatible<Vec4<long long, 32>, longlong4_32a>::value);
#else
	using longlong4 = ::longlong4;							static_assert(BinaryCompatible<Vec4<long long, 16>, longlong4>::value);
	using longlong4_16a = ::longlong4;						static_assert(BinaryCompatible<Vec4<long long, 16>, longlong4_16a>::value);
	using longlong4_32a = Vec4<long long, 32>;				static_assert(BinaryCompatible<Vec4<long long, 32>, longlong4_32a>::value);
#endif

	using ulonglong = unsigned long long;
	using ulonglong2 = ::ulonglong2;						static_assert(BinaryCompatible<Vec2<unsigned long long, 16>, ulonglong2>::value);
	using ulonglong3 = ::ulonglong3;						static_assert(BinaryCompatible<Vec3<unsigned long long,  8>, ulonglong3>::value);
#if __CUDACC_VER_MAJOR__ >= 13
	using ulonglong4 = ::ulonglong4_16a;					static_assert(BinaryCompatible<Vec4<unsigned long long, 16>, ulonglong4>::value);
	using ulonglong4_16a = ::ulonglong4_16a;				static_assert(BinaryCompatible<Vec4<unsigned long long, 16>, ulonglong4_16a>::value);
	using ulonglong4_32a = ::ulonglong4_32a;				static_assert(BinaryCompatible<Vec4<unsigned long long, 32>, ulonglong4_32a>::value);
#else
	using ulonglong4 = ::ulonglong4;						static_assert(BinaryCompatible<Vec4<unsigned long long, 16>, ulonglong4>::value);
	using ulonglong4_16a = ::ulonglong4;					static_assert(BinaryCompatible<Vec4<unsigned long long, 16>, ulonglong4_16a>::value);
	using ulonglong4_32a = Vec4<unsigned long long, 32>;	static_assert(BinaryCompatible<Vec4<unsigned long long, 32>, ulonglong4_32a>::value);
#endif
#endif
}

#ifdef _MSC_VER
	#pragma warning(pop)	// warning: 4324
#endif