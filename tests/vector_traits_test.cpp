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

#include <gtest/gtest.h>
#include <nucleus/vector_types.h>
#include <nucleus/vector_traits.h>

/*********************************************************************************
***************************    vector_traits_test    *****************************
*********************************************************************************/

#if NS_HAS_CXX_20

//	--- scalar_type_t ---
static_assert(std::is_same_v<ns::scalar_type_t<ns::int2>, int>);
static_assert(std::is_same_v<ns::scalar_type_t<ns::int3>, int>);
static_assert(std::is_same_v<ns::scalar_type_t<ns::int4>, int>);

static_assert(std::is_same_v<ns::scalar_type_t<ns::char2>, char>);
static_assert(std::is_same_v<ns::scalar_type_t<ns::char3>, char>);
static_assert(std::is_same_v<ns::scalar_type_t<ns::char4>, char>);

static_assert(std::is_same_v<ns::scalar_type_t<ns::short2>, short>);
static_assert(std::is_same_v<ns::scalar_type_t<ns::short3>, short>);
static_assert(std::is_same_v<ns::scalar_type_t<ns::short4>, short>);

static_assert(std::is_same_v<ns::scalar_type_t<ns::float2>, float>);
static_assert(std::is_same_v<ns::scalar_type_t<ns::float3>, float>);
static_assert(std::is_same_v<ns::scalar_type_t<ns::float4>, float>);

static_assert(std::is_same_v<ns::scalar_type_t<ns::double2>, double>);
static_assert(std::is_same_v<ns::scalar_type_t<ns::double3>, double>);
static_assert(std::is_same_v<ns::scalar_type_t<ns::double4>, double>);

static_assert(std::is_same_v<ns::scalar_type_t<ns::uint2>, unsigned int>);
static_assert(std::is_same_v<ns::scalar_type_t<ns::uint3>, unsigned int>);
static_assert(std::is_same_v<ns::scalar_type_t<ns::uint4>, unsigned int>);

static_assert(std::is_same_v<ns::scalar_type_t<ns::uchar2>, unsigned char>);
static_assert(std::is_same_v<ns::scalar_type_t<ns::uchar3>, unsigned char>);
static_assert(std::is_same_v<ns::scalar_type_t<ns::uchar4>, unsigned char>);

static_assert(std::is_same_v<ns::scalar_type_t<ns::ushort2>,unsigned short>);
static_assert(std::is_same_v<ns::scalar_type_t<ns::ushort3>,unsigned short>);
static_assert(std::is_same_v<ns::scalar_type_t<ns::ushort4>,unsigned short>);

//	--- scalar_type_t: cv-qualifiers and references are stripped ---
static_assert(std::is_same_v<ns::scalar_type_t<const ns::float2>, float>);
static_assert(std::is_same_v<ns::scalar_type_t<const ns::float2&>, float>);

//	--- vec2_like: positive cases ---
static_assert(ns::vec2_like<ns::int2>);
static_assert(ns::vec2_like<ns::uint2>);
static_assert(ns::vec2_like<ns::char2>);
static_assert(ns::vec2_like<ns::uchar2>);
static_assert(ns::vec2_like<ns::float2>);
static_assert(ns::vec2_like<ns::short2>);
static_assert(ns::vec2_like<ns::ushort2>);
static_assert(ns::vec2_like<ns::double2>);

//	--- vec2_like: negative cases ---
static_assert(!ns::vec2_like<int>);
static_assert(!ns::vec2_like<float>);
static_assert(!ns::vec2_like<ns::float3>);
static_assert(!ns::vec2_like<ns::float4>);

//	--- vec2_like: struct with extra fields must not match ---
struct ExtraField2 { float x, y, extra; };
static_assert(!ns::vec2_like<ExtraField2>);

//	--- vec3_like: positive cases ---
static_assert(ns::vec3_like<ns::int3>);
static_assert(ns::vec3_like<ns::uint3>);
static_assert(ns::vec3_like<ns::char3>);
static_assert(ns::vec3_like<ns::float3>);
static_assert(ns::vec3_like<ns::uchar3>);
static_assert(ns::vec3_like<ns::short3>);
static_assert(ns::vec3_like<ns::ushort3>);
static_assert(ns::vec3_like<ns::double3>);
static_assert(ns::vec3_like<ns::int3_16a>);
static_assert(ns::vec3_like<ns::uint3_16a>);
static_assert(ns::vec3_like<ns::float3_16a>);

//	--- vec3_like: negative cases ---
static_assert(!ns::vec3_like<int>);
static_assert(!ns::vec3_like<float>);
static_assert(!ns::vec3_like<ns::float2>);
static_assert(!ns::vec3_like<ns::float4>);

//	--- vec3_like: struct with extra fields must not match ---
struct ExtraField3 { float x, y, z, a, b; };
static_assert(!ns::vec3_like<ExtraField3>);

//	--- vec4_like: positive cases ---
static_assert(ns::vec4_like<ns::int4>);
static_assert(ns::vec4_like<ns::uint4>);
static_assert(ns::vec4_like<ns::char4>);
static_assert(ns::vec4_like<ns::uchar4>);
static_assert(ns::vec4_like<ns::short4>);
static_assert(ns::vec4_like<ns::float4>);
static_assert(ns::vec4_like<ns::ushort4>);
static_assert(ns::vec4_like<ns::double4>);

//	--- vec4_like: negative cases ---
static_assert(!ns::vec4_like<int>);
static_assert(!ns::vec4_like<float>);
static_assert(!ns::vec4_like<ns::float2>);
static_assert(!ns::vec4_like<ns::float3>);

//	--- vec4_like: struct with extra fields must not match ---
struct ExtraField4 { float x, y, z, w, extra; };
static_assert(!ns::vec4_like<ExtraField4>);

//	--- vec_like: positive cases ---
static_assert(ns::vec_like<ns::int2>);
static_assert(ns::vec_like<ns::int3>);
static_assert(ns::vec_like<ns::int4>);
static_assert(ns::vec_like<ns::float2>);
static_assert(ns::vec_like<ns::float3>);
static_assert(ns::vec_like<ns::float4>);
static_assert(ns::vec_like<ns::double2>);
static_assert(ns::vec_like<ns::double3>);
static_assert(ns::vec_like<ns::double4>);

//	--- vec_like: negative cases ---
static_assert(!ns::vec_like<int>);
static_assert(!ns::vec_like<float>);

//	--- vec2_like<Type, Scalar>: typed concept ---
static_assert(ns::vec2_like<ns::int2, int>);
static_assert(ns::vec2_like<ns::float2, float>);
static_assert(ns::vec2_like<ns::double2, double>);
static_assert(!ns::vec2_like<ns::int2, float>);
static_assert(!ns::vec2_like<ns::float2, int>);

//	--- vec3_like<Type, Scalar>: typed concept ---
static_assert(ns::vec3_like<ns::int3, int>);
static_assert(ns::vec3_like<ns::float3, float>);
static_assert(ns::vec3_like<ns::double3, double>);
static_assert(!ns::vec3_like<ns::int3, float>);
static_assert(!ns::vec3_like<ns::float3, int>);

//	--- vec4_like<Type, Scalar>: typed concept ---
static_assert(ns::vec4_like<ns::int4, int>);
static_assert(ns::vec4_like<ns::float4, float>);
static_assert(ns::vec4_like<ns::double4, double>);
static_assert(!ns::vec4_like<ns::int4, float>);
static_assert(!ns::vec4_like<ns::float4, int>);

//	--- common scalar-typed aliases ---
static_assert(ns::int2_like<ns::int2>);
static_assert(ns::int3_like<ns::int3>);
static_assert(ns::int4_like<ns::int4>);
static_assert(!ns::int2_like<ns::float2>);

static_assert(ns::uint2_like<ns::uint2>);
static_assert(ns::uint3_like<ns::uint3>);
static_assert(ns::uint4_like<ns::uint4>);
static_assert(!ns::uint2_like<ns::int2>);

static_assert(ns::float2_like<ns::float2>);
static_assert(ns::float3_like<ns::float3>);
static_assert(ns::float4_like<ns::float4>);
static_assert(!ns::float2_like<ns::double2>);

static_assert(ns::double2_like<ns::double2>);
static_assert(ns::double3_like<ns::double3>);
static_assert(ns::double4_like<ns::double4>);
static_assert(!ns::double2_like<ns::float2>);

static_assert(ns::char2_like<ns::char2>);
static_assert(ns::char3_like<ns::char3>);
static_assert(ns::char4_like<ns::char4>);
static_assert(!ns::char2_like<ns::uchar2>);

static_assert(ns::uchar2_like<ns::uchar2>);
static_assert(ns::uchar3_like<ns::uchar3>);
static_assert(ns::uchar4_like<ns::uchar4>);
static_assert(!ns::uchar2_like<ns::char2>);

static_assert(ns::short2_like<ns::short2>);
static_assert(ns::short3_like<ns::short3>);
static_assert(ns::short4_like<ns::short4>);
static_assert(!ns::short2_like<ns::ushort2>);

static_assert(ns::ushort2_like<ns::ushort2>);
static_assert(ns::ushort3_like<ns::ushort3>);
static_assert(ns::ushort4_like<ns::ushort4>);
static_assert(!ns::ushort2_like<ns::short2>);

//	--- scalar_type_t: positive cases ---
static_assert(std::is_same_v<ns::scalar_type_t<ns::int2>, int>);
static_assert(std::is_same_v<ns::scalar_type_t<ns::float2>, float>);
static_assert(std::is_same_v<ns::scalar_type_t<ns::float3>, float>);
static_assert(std::is_same_v<ns::scalar_type_t<ns::float4>, float>);
static_assert(std::is_same_v<ns::scalar_type_t<ns::double4>, double>);

TEST(VectorTraitsTest, CompileTimeChecks)
{
	// All checks are static_assert above; this test just confirms compilation
	SUCCEED();
}

#endif	//	NS_HAS_CXX_20