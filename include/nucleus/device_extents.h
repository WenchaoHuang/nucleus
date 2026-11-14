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

#include "device_span.h"

namespace NS_NAMESPACE::dev::detail
{
	//!	@brief		Stores only the runtime dimensions of a multidimensional span.
	template<size_t Count> struct ExtentStorage
	{
		size_t	m_sizes[Count]{};
	};

	//!	@brief		Fixed dimensions do not require runtime storage.
	template<> struct ExtentStorage<0> {};

	/*****************************************************************************
	*******************************    Extents    *******************************
	*****************************************************************************/

	//!	@brief		Shared static/dynamic dimension storage for contiguous spans.
	template<size_t... Dimensions> class Extents : protected ExtentStorage<(size_t(0) + ... + (Dimensions == dynamic_extent))>
	{
	public:

		static constexpr size_t rank = sizeof...(Dimensions);
		static constexpr size_t rank_dynamic = (size_t(0) + ... + (Dimensions == dynamic_extent));
		static constexpr bool default_constructible = ((Dimensions == dynamic_extent || Dimensions == 0) || ...);

		//! @brief		Allows constant initialization of empty device views.
		constexpr Extents() noexcept = default;

		/**
		 *	@brief		Accepts all dimensions, or only the dynamic dimensions in order.
		 *	@note		Explicitly supplied static dimensions must match the template arguments.
		 */
		template<typename... Sizes> NS_CUDA_CALLABLE constexpr explicit Extents(Sizes... sizes) noexcept
			requires((std::is_integral_v<Sizes> && ...) && (sizeof...(Sizes) == rank || sizeof...(Sizes) == rank_dynamic))
		{
			[[maybe_unused]] const size_t values[] = { static_cast<size_t>(sizes)..., 0 };
			constexpr size_t extents[] = { Dimensions... };

			if constexpr (sizeof...(Sizes) == rank)
			{
				[[maybe_unused]] size_t index = 0;
				for (size_t i = 0; i < rank; ++i)
				{
					if (extents[i] == dynamic_extent)
					{
						if constexpr (rank_dynamic != 0)
							this->m_sizes[index++] = values[i];
					}
					else
						NS_ASSERT(values[i] == extents[i]);
				}
			}
			else if constexpr (rank_dynamic != 0)
			{
				for (size_t i = 0; i < rank_dynamic; ++i)
					this->m_sizes[i] = values[i];
			}
		}

		//!	@brief		Returns a dimension, using a compile-time constant when possible.
		template<size_t Index> NS_CUDA_CALLABLE constexpr size_t get() const noexcept
		{
			static_assert(Index < rank);
			constexpr size_t extents[] = { Dimensions... };

			if constexpr (extents[Index] != dynamic_extent)
				return extents[Index];
			else
			{
				size_t index = 0;
				for (size_t i = 0; i != Index; ++i)
					index += (extents[i] == dynamic_extent);

				return this->m_sizes[index];
			}
		}
	};

	//!	@brief		Whether two dimensions can describe the same shape.
	template<size_t Target, size_t Source> inline constexpr bool compatible_extent =
		Target == dynamic_extent || Source == dynamic_extent || Target == Source;
}
