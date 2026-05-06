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
#include <utility>
#include <type_traits>

#if NS_HAS_CXX_20
	#include <concepts>
#endif

namespace NS_NAMESPACE
{
	/*****************************************************************************
	****************************    Vector Concepts    ***************************
	*****************************************************************************/

#if NS_HAS_CXX_20

	/**
	 *	@brief	Concept that matches types with exactly two same-typed x and y components.
	 *	@note	Satisfied by types such as float2, int2, double2, etc. Types with a
	 *			z member (e.g. float3) are explicitly excluded, and sizeof(T) must equal
	 *			2 * sizeof(scalar component) to exclude structs with extra fields.
	 */
	template<typename T> concept vec2_like = requires(T v)
	{
		v.x;
		v.y;
		requires std::is_same_v<decltype(v.x), decltype(v.y)>;
		requires sizeof(T) == 2 * sizeof(decltype(v.x));
	} && !requires(T v) { v.z; };

	/**
	 *	@brief	Concept that matches types with exactly three same-typed components x, y and z.
	 *	@note	Satisfied by types such as float3, int3, double3, etc. Types with a
	 *			w member (e.g. float4) are explicitly excluded, and sizeof(T) must equal
	 *			3 * or 4 * sizeof(scalar component) to also cover 16-byte-aligned variants
	 *			such as float3_16a while still excluding structs with too many fields.
	 */
	template<typename T> concept vec3_like = requires(T v)
	{
		v.x;
		v.y;
		v.z;
		requires std::is_same_v<decltype(v.x), decltype(v.y)>;
		requires std::is_same_v<decltype(v.x), decltype(v.z)>;
		requires sizeof(T) == 3 * sizeof(decltype(v.x)) || sizeof(T) == 4 * sizeof(decltype(v.x));
	} && !requires(T v) { v.w; };

	/**
	 *	@brief	Concept that matches types with exactly four same-typed components x, y, z and w.
	 *	@note	Satisfied by types such as float4, int4, double4, etc. sizeof(T) must equal
	 *			4 * sizeof(scalar component) to exclude structs with extra fields.
	 */
	template<typename T> concept vec4_like = requires(T v)
	{
		v.x;
		v.y;
		v.z;
		v.w;
		requires std::is_same_v<decltype(v.x), decltype(v.y)>;
		requires std::is_same_v<decltype(v.x), decltype(v.z)>;
		requires std::is_same_v<decltype(v.x), decltype(v.w)>;
		requires sizeof(T) == 4 * sizeof(decltype(v.x));
	};

#endif	//	NS_HAS_CXX_20

	/*****************************************************************************
	******************************    scalar_type    *****************************
	*****************************************************************************/

	namespace details
	{
		/**
		 *	@brief	Primary template: treats the type itself as its own scalar type.
		 *	@note	Used as the fallback for scalar (non-vector) types.
		 */
		template<typename T, typename = void>
		struct VecScalarType { using type = T; };

		/**
		 *	@brief	Specialization for types that expose an x member (vector types).
		 *	@note	Extracts the scalar type as the decayed type of the x member,
		 *			covering Vec2, Vec3, Vec4, and compatible CUDA vector types.
		 */
		template<typename T>
		struct VecScalarType<T, std::void_t<decltype(std::declval<T>().x)>>
		{
			using type = std::decay_t<decltype(std::declval<T>().x)>;
		};
	}

	/**
	 *	@brief	Extracts the scalar type from a vector or scalar type.
	 *	@note	For scalar types (e.g. float, int), scalar_type_t<T> is T itself.
	 *			For vector types with an x member (e.g. float2, int3, double4),
	 *			scalar_type_t<T> is the type of that member (e.g. float, int, double).
	 *			CV-qualifiers and references on T are stripped before deduction.
	 */
	template<typename T>
	using scalar_type_t = typename details::VecScalarType<
		std::remove_cv_t<std::remove_reference_t<T>>>::type;
}
