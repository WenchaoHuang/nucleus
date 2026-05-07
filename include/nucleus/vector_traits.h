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
#include "vector_types.h"
#include <type_traits>

#if NS_HAS_CXX_20
	#include <utility>
	#include <concepts>
#endif

namespace NS_NAMESPACE
{
	/*****************************************************************************
	****************************    Vector Concepts    ***************************
	*****************************************************************************/

#if NS_HAS_CXX_20

	/**
	 *	@brief	Concept that matches types with exactly two same-typed `x` and `y` components.
	 *	@note	Satisfied by types such as `float2`, `int2`, `double2`, etc.
	 *			sizeof(Type) must equal 2 * sizeof(scalar component) to exclude structs with extra fields.
	 *			When Scalar is not void, also requires that the component type equals Scalar.
	 */
	template<typename Type, typename Scalar = void> concept vec2_like = requires(Type v)
	{
		v.x;
		v.y;
		requires std::is_same_v<decltype(v.x), decltype(v.y)>;
		requires sizeof(Type) == 2 * sizeof(decltype(v.x));
		requires std::is_void_v<Scalar> || std::is_same_v<std::remove_cvref_t<decltype(v.x)>, Scalar>;
	};


	/**
	 *	@brief	Concept that matches types with exactly three same-typed components `x`, `y` and `z`.
	 *	@note	Satisfied by types such as `float3`, `int3`, `double3`, etc. Types with a
	 *			w member (e.g. `float4`) are explicitly excluded, and sizeof(Type) must equal
	 *			3 * or 4 * sizeof(scalar component) to also cover 16-byte-aligned variants
	 *			such as `float3_16a` while still excluding structs with too many fields.
	 *			When Scalar is not void, also requires that the component type equals Scalar.
	 */
	template<typename Type, typename Scalar = void> concept vec3_like = requires(Type v)
	{
		v.x;
		v.y;
		v.z;
		requires std::is_same_v<decltype(v.x), decltype(v.y)>;
		requires std::is_same_v<decltype(v.x), decltype(v.z)>;
		requires sizeof(Type) == 3 * sizeof(decltype(v.x)) || sizeof(Type) == 4 * sizeof(decltype(v.x));
		requires std::is_void_v<Scalar> || std::is_same_v<std::remove_cvref_t<decltype(v.x)>, Scalar>;
	} && !requires(Type v) { v.w; };


	/**
	 *	@brief	Concept that matches types with exactly four same-typed components `x`, `y`, `z` and `w`.
	 *	@note	Satisfied by types such as `float4`, `int4`, `double4`, etc. sizeof(Type) must equal
	 *			4 * sizeof(scalar component) to exclude structs with extra fields.
	 *			When Scalar is not void, also requires that the component type equals Scalar.
	 */
	template<typename Type, typename Scalar = void> concept vec4_like = requires(Type v)
	{
		v.x;
		v.y;
		v.z;
		v.w;
		requires std::is_same_v<decltype(v.x), decltype(v.y)>;
		requires std::is_same_v<decltype(v.x), decltype(v.z)>;
		requires std::is_same_v<decltype(v.x), decltype(v.w)>;
		requires sizeof(Type) == 4 * sizeof(decltype(v.x));
		requires std::is_void_v<Scalar> || std::is_same_v<std::remove_cvref_t<decltype(v.x)>, Scalar>;
	};


	//	--- Common scalar-typed aliases ---

	template<typename Type> concept int2_like    = vec2_like<Type, int>;
	template<typename Type> concept int3_like    = vec3_like<Type, int>;
	template<typename Type> concept int4_like    = vec4_like<Type, int>;

	template<typename Type> concept uint2_like   = vec2_like<Type, unsigned int>;
	template<typename Type> concept uint3_like   = vec3_like<Type, unsigned int>;
	template<typename Type> concept uint4_like   = vec4_like<Type, unsigned int>;

	template<typename Type> concept float2_like  = vec2_like<Type, float>;
	template<typename Type> concept float3_like  = vec3_like<Type, float>;
	template<typename Type> concept float4_like  = vec4_like<Type, float>;

	template<typename Type> concept double2_like = vec2_like<Type, double>;
	template<typename Type> concept double3_like = vec3_like<Type, double>;
	template<typename Type> concept double4_like = vec4_like<Type, double>;

#endif	//	NS_HAS_CXX_20

	/*****************************************************************************
	******************************    scalar_type    *****************************
	*****************************************************************************/

#if NS_HAS_CXX_20

	namespace details
	{
		template<typename Type>	 struct VecScalarType;
		template<vec2_like Type> struct VecScalarType<Type> { using type = std::remove_cvref_t<decltype(std::declval<Type>().x)>; };
		template<vec3_like Type> struct VecScalarType<Type> { using type = std::remove_cvref_t<decltype(std::declval<Type>().x)>; };
		template<vec4_like Type> struct VecScalarType<Type> { using type = std::remove_cvref_t<decltype(std::declval<Type>().x)>; };
	}

	/**
	 *	@brief	Extracts the scalar type from a vector type.
	 *	@note	For vector types with an `x` member (e.g. `float2`, `int3`, `double4`),
	 *			scalar_type_t<T> is the type of that member (e.g. `float`, `int`, `double`).
	 *			CV-qualifiers and references on T are stripped before deduction.
	 */
	template<typename T> using scalar_type_t = typename details::VecScalarType<std::remove_cvref_t<T>>::type;

#endif	//	NS_HAS_CXX_20
}