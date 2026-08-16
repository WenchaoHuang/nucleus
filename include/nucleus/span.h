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
#include <span>
#include <initializer_list>

namespace NS_NAMESPACE
{
	namespace detail
	{
		/*************************************************************************
		*********************    ConstSpan<Type, Extent>    **********************
		*************************************************************************/

		/**
		 *	@brief		A read-only `std::span` adapter with initializer-list construction.
		 *	@tparam		Type - The unqualified element type viewed by the span.
		 *	@tparam		Extent - The static extent, or `std::dynamic_extent`.
		 *
		 *	All constructors other than the base copy constructor are inherited from
		 *	`std::span<const Type, Extent>`. The explicit base bridge restores
		 *	construction from a base span object, including results produced by
		 *	standard span utilities.
		 *
		 *	@warning	A braced temporary initializer list normally owns its backing
		 *				array only until the end of the full expression. This constructor is
		 *				intended for calls such as `consume({1, 2, 3})`; storing that span
		 *				leaves it dangling.
		 */
		template<typename Type, size_t Extent> class ConstSpan : public std::span<const Type, Extent>
		{
			using Base = std::span<const Type, Extent>;
			using ListElement = std::remove_cv_t<Type>;

		public:

			//! Inherit pointer, iterator, array, range, and converting-span constructors.
			using Base::Base;

			//!	@brief		Preserve the default-construction rules of the underlying `std::span`.
			constexpr ConstSpan() noexcept = default;

			//!	@brief		Construct from the exact base specialization, whose copy constructor is not inherited.
			constexpr ConstSpan(Base span) noexcept : Base(span) {}

			/**
			 *	@brief		Constructs a dynamic read-only span over an initializer list.
			 *	@param[in]	list - Elements to view without taking ownership.
			 *	@warning	A span made from a braced temporary list becomes dangling at the end of the full expression.
			 */
			constexpr ConstSpan(std::initializer_list<ListElement> list) noexcept requires(Extent == std::dynamic_extent) : Base(list.begin(), list.size()) {}
		};

		//! Selects an exact `std::span` for mutable element types.
		template<typename Type, size_t Extent> struct SpanSelector { using type = std::span<Type, Extent>; };

		//! Selects the initializer-list-enabled adapter for const element types.
		template<typename Type, size_t Extent> struct SpanSelector<const Type, Extent> { using type = ConstSpan<Type, Extent>; };
	}

	/*****************************************************************************
	**************************    Span<Type, Extent>    **************************
	*****************************************************************************/

	/**
	 *	@brief		A host-only, non-owning view over a contiguous sequence.
	 *	@tparam		Type - Element type. Const element types accept initializer lists.
	 *	@tparam		Extent - Static element count, or `std::dynamic_extent`.
	 *	@note		For mutable `Type`, this alias is exactly `std::span<Type, Extent>`.
	 *	@note		For const `Type`, it selects `detail::ConstSpan`, which adds initializer-list
	 *				construction while retaining the standard span interface through public inheritance.
	 */
	template<typename Type, size_t Extent = std::dynamic_extent> using Span = typename detail::SpanSelector<Type, Extent>::type;
}
