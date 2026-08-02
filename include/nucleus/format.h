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
#include "vector_traits.h"

namespace NS_NAMESPACE
{
	/*****************************************************************************
	********************************    Format    ********************************
	*****************************************************************************/

	/**
	 *	@brief	Available texel formats for CUDA texture objects.
	 *	@note	This enum class defines the supported data types and component counts
	 *			for texels in CUDA texture objects. Each format specifies a base type
	 *			(e.g., char, int, float) and the number of components (1, 2, or 4).
	 */
	enum class Format
	{
		Undefined,					// Undefined format
		Int, Int2, Int4,			// Signed 32-bit integer (1, 2, or 4 components)
		Uint, Uint2, Uint4,			// Unsigned 32-bit integer (1, 2, or 4 components)
		Char, Char2, Char4,			// Signed 8-bit integer (1, 2, or 4 components)
		Uchar, Uchar2, Uchar4,		// Unsigned 8-bit integer (1, 2, or 4 components)
		Short, Short2, Short4,		// Signed 16-bit integer (1, 2, or 4 components)
		Ushort, Ushort2, Ushort4,	// Unsigned 16-bit integer (1, 2, or 4 components)
		Float, Float2, Float4,		// 32-bit floating-point (1, 2, or 4 components)
	};

	/*****************************************************************************
	****************************    FormatOf<Type>    ****************************
	*****************************************************************************/

	//!	@brief		Maps C++ types to corresponding `Format` enum values.
	template<typename Type>		struct FormatOf			{ static_assert(false, "Invalid format!"); };

	template<>					struct FormatOf<int>	{ static constexpr Format value = Format::Int; };
	template<int2_like Type>	struct FormatOf<Type>	{ static constexpr Format value = Format::Int2; };
	template<int4_like Type>	struct FormatOf<Type>	{ static constexpr Format value = Format::Int4; };

	template<>					struct FormatOf<uint>	{ static constexpr Format value = Format::Uint; };
	template<uint2_like Type>	struct FormatOf<Type>	{ static constexpr Format value = Format::Uint2; };
	template<uint4_like Type>	struct FormatOf<Type>	{ static constexpr Format value = Format::Uint4; };

	template<>					struct FormatOf<char>	{ static constexpr Format value = Format::Char; };
	template<char2_like Type>	struct FormatOf<Type>	{ static constexpr Format value = Format::Char2; };
	template<char4_like Type>	struct FormatOf<Type>	{ static constexpr Format value = Format::Char4; };

	template<>					struct FormatOf<uchar>	{ static constexpr Format value = Format::Uchar; };
	template<uchar2_like Type>	struct FormatOf<Type>	{ static constexpr Format value = Format::Uchar2; };
	template<uchar4_like Type>	struct FormatOf<Type>	{ static constexpr Format value = Format::Uchar4; };

	template<>					struct FormatOf<short>	{ static constexpr Format value = Format::Short; };
	template<short2_like Type>	struct FormatOf<Type>	{ static constexpr Format value = Format::Short2; };
	template<short4_like Type>	struct FormatOf<Type>	{ static constexpr Format value = Format::Short4; };

	template<>					struct FormatOf<ushort>	{ static constexpr Format value = Format::Ushort; };
	template<ushort2_like Type>	struct FormatOf<Type>	{ static constexpr Format value = Format::Ushort2; };
	template<ushort4_like Type>	struct FormatOf<Type>	{ static constexpr Format value = Format::Ushort4; };

	template<>					struct FormatOf<float>	{ static constexpr Format value = Format::Float; };
	template<float2_like Type>	struct FormatOf<Type>	{ static constexpr Format value = Format::Float2; };
	template<float4_like Type>	struct FormatOf<Type>	{ static constexpr Format value = Format::Float4; };

	/*****************************************************************************
	**************************    FormatInfo<Format>    **************************
	*****************************************************************************/

	//!	@brief		Provides type information for a given `Format` enum value.
	template<Format format> struct FormatInfo		{ static_assert(false, "Invalid format!"); };
	template<Format format> using FormatInfo_t		= typename FormatInfo<format>::type;

	template<> struct FormatInfo<Format::Int>		{ using type = int;			using component_type = int;		static constexpr int component_count = 1; };
	template<> struct FormatInfo<Format::Int2>		{ using type = int2;		using component_type = int;		static constexpr int component_count = 2; };
	template<> struct FormatInfo<Format::Int4>		{ using type = int4;		using component_type = int;		static constexpr int component_count = 4; };

	template<> struct FormatInfo<Format::Uint>		{ using type = uint;		using component_type = uint;	static constexpr int component_count = 1; };
	template<> struct FormatInfo<Format::Uint2>		{ using type = uint2;		using component_type = uint;	static constexpr int component_count = 2; };
	template<> struct FormatInfo<Format::Uint4>		{ using type = uint4;		using component_type = uint;	static constexpr int component_count = 4; };

	template<> struct FormatInfo<Format::Char>		{ using type = char;		using component_type = char;	static constexpr int component_count = 1; };
	template<> struct FormatInfo<Format::Char2>		{ using type = char2;		using component_type = char;	static constexpr int component_count = 2; };
	template<> struct FormatInfo<Format::Char4>		{ using type = char4;		using component_type = char;	static constexpr int component_count = 4; };

	template<> struct FormatInfo<Format::Uchar>		{ using type = uchar;		using component_type = uchar;	static constexpr int component_count = 1; };
	template<> struct FormatInfo<Format::Uchar2>	{ using type = uchar2;		using component_type = uchar;	static constexpr int component_count = 2; };
	template<> struct FormatInfo<Format::Uchar4>	{ using type = uchar4;		using component_type = uchar;	static constexpr int component_count = 4; };

	template<> struct FormatInfo<Format::Short>		{ using type = short;		using component_type = short;	static constexpr int component_count = 1; };
	template<> struct FormatInfo<Format::Short2>	{ using type = short2;		using component_type = short;	static constexpr int component_count = 2; };
	template<> struct FormatInfo<Format::Short4>	{ using type = short4;		using component_type = short;	static constexpr int component_count = 4; };

	template<> struct FormatInfo<Format::Ushort>	{ using type = ushort;		using component_type = ushort;	static constexpr int component_count = 1; };
	template<> struct FormatInfo<Format::Ushort2>	{ using type = ushort2;		using component_type = ushort;	static constexpr int component_count = 2; };
	template<> struct FormatInfo<Format::Ushort4>	{ using type = ushort4;		using component_type = ushort;	static constexpr int component_count = 4; };

	template<> struct FormatInfo<Format::Float>		{ using type = float;		using component_type = float;	static constexpr int component_count = 1; };
	template<> struct FormatInfo<Format::Float2>	{ using type = float2;		using component_type = float;	static constexpr int component_count = 2; };
	template<> struct FormatInfo<Format::Float4>	{ using type = float4;		using component_type = float;	static constexpr int component_count = 4; };

	/*****************************************************************************
	*************************    CudaTexelType<Type>    **************************
	*****************************************************************************/

	//!	@brief		Extracts the CUDA texel type corresponding to a C++ type.
	template<typename Type> using CudaTexelType = CudaBuiltinType<FormatInfo_t<FormatOf<Type>::value>>;
	template<typename Type> using CudaTexelType_t = typename CudaTexelType<Type>::type;
}
