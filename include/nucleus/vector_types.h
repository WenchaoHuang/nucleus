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
	 *	@brief	Type aliases for commonly used vector types.
	 *	@note	This approach avoids exposing CUDA headers in the interface, allowing the
	 *			header to be included in non-CUDA projects.
	 */
	template<typename Type, int Align> struct NS_ALIGN(Align) Vec2 { Type x, y; };
	template<typename Type, int Align> struct NS_ALIGN(Align) Vec3 { Type x, y, z; };
	template<typename Type, int Align> struct NS_ALIGN(Align) Vec4 { Type x, y, z, w; };

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
}

#ifdef _MSC_VER
	#pragma warning(pop)	// warning: 4324
#endif