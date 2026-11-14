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

#include "device_span_2d.h"

namespace NS_NAMESPACE::dev
{
	//!	@brief		A non-owning view over a contiguous 3D array, with x varying fastest.
	template<typename Type, size_t Width = dynamic_extent, size_t Height = dynamic_extent, size_t Depth = dynamic_extent> class Span3D;

	/*****************************************************************************
	*************************    Span3D<const Type>    **************************
	*****************************************************************************/

	/**
	 *	@brief		A non-owning, read-only view over a contiguous 3D array.
	 *	@tparam		Type - The element type (non-const).
	 *	@tparam		Width - The width of the array, or dynamic_extent.
	 *	@tparam		Height - The height of the array, or dynamic_extent.
	 *	@tparam		Depth - The depth of the array, or dynamic_extent.
	 *	@note		Dimensions are ordered width, height, depth; x varies fastest in memory.
	 *			Pitched allocations and non-contiguous subregions are not supported.
	 *			Like Span, mutable views inherit from this specialization to support
	 *			implicit conversion and template argument deduction for const views.
	 */
	template<typename Type, size_t Width, size_t Height, size_t Depth> class Span3D<const Type, Width, Height, Depth> : protected detail::Extents<Width, Height, Depth>
	{
		using _Extents = detail::Extents<Width, Height, Depth>;

	public: // Type definitions for the span.

		using size_type = size_t;
		using pointer = const Type *;
		using reference = const Type &;
		using element_type = const Type;
		using const_pointer = const Type *;
		using const_reference = const Type &;
		using difference_type = std::ptrdiff_t;
		using value_type = std::remove_cv_t<Type>;

		static constexpr size_t static_width = Width;
		static constexpr size_t static_height = Height;
		static constexpr size_t static_depth = Depth;

	public: // Constructors for the span.

		//!	@brief		Copy constructor for the span.
		constexpr Span3D(const Span3D &) noexcept = default;

		//! @brief		Dynamic initialization is not supported for a `__constant__` variable
		constexpr Span3D() noexcept requires(_Extents::default_constructible) = default;

		/**
		 *	@brief		Constructs a span with all dimensions, or just the dynamic dimensions.
		 *	@note		Dimensions are supplied in width, height, depth order. Fixed dimensions
		 *				must match when supplied; storage must cover the complete shape.
		 */
		template<typename... Sizes> NS_CUDA_CALLABLE constexpr explicit Span3D(const Type * data, Sizes... sizes) noexcept
			requires((std::is_integral_v<Sizes> && ...) && (sizeof...(Sizes) == 3 || sizeof...(Sizes) == _Extents::rank_dynamic)) :
			_Extents(sizes...), m_data(data) {}

		//!	@brief		Converts compatible shapes; dynamic-to-fixed dimensions require explicit construction.
		template<size_t OtherWidth, size_t OtherHeight, size_t OtherDepth> NS_CUDA_CALLABLE constexpr
		explicit((Width != dynamic_extent && OtherWidth == dynamic_extent) || (Height != dynamic_extent && OtherHeight == dynamic_extent) || (Depth != dynamic_extent && OtherDepth == dynamic_extent))
		Span3D(const Span3D<const Type, OtherWidth, OtherHeight, OtherDepth> & rhs) noexcept
			requires(detail::compatible_extent<Width, OtherWidth> && detail::compatible_extent<Height, OtherHeight> && detail::compatible_extent<Depth, OtherDepth>) : Span3D(rhs.data(), rhs.width(), rhs.height(), rhs.depth()) {}

		//!	@brief		Constructs a span from an array, requiring an exact match for fixed dimensions.
		explicit(_Extents::rank_dynamic != 3) Span3D(const Array3D<Type> & arr) noexcept :
			Span3D(arr.data(), arr.width(), arr.height(), arr.depth()) {}

	public: // Observers.

		//!	@brief		Returns a const pointer to the underlying data.
		NS_CUDA_CALLABLE constexpr const_pointer data() const noexcept { return m_data; }

		//!	@brief		Checks if the span has no bound storage, following Span.
		NS_CUDA_CALLABLE constexpr bool empty() const noexcept { return m_data == nullptr; }

		//!	@brief		Returns the width of the span.
		NS_CUDA_CALLABLE constexpr size_type width() const noexcept { return _Extents::template get<0>(); }

		//!	@brief		Returns the height of the span.
		NS_CUDA_CALLABLE constexpr size_type height() const noexcept { return _Extents::template get<1>(); }

		//!	@brief		Returns the depth of the span.
		NS_CUDA_CALLABLE constexpr size_type depth() const noexcept { return _Extents::template get<2>(); }

		//!	@brief		Returns the number of elements in the span.
		NS_CUDA_CALLABLE constexpr size_type size() const noexcept { return width() * height() * depth(); }

		//!	@brief		Returns the size of the span in bytes.
		NS_CUDA_CALLABLE constexpr size_type size_bytes() const noexcept { return size() * sizeof(Type); }

		//!	@brief		Returns the row pitch in bytes.
		NS_CUDA_CALLABLE constexpr size_type pitch() const noexcept { return width() * sizeof(Type); }

	public: // Element access.

		//!	@brief		Returns a const reference to the element at the given coordinates.
		NS_CUDA_CALLABLE constexpr const_reference operator()(size_t x, size_t y, size_t z) const noexcept
		{
			NS_ASSERT(x < width() && y < height() && z < depth());

			return m_data[(z * height() + y) * width() + x];
		}

		//!	@brief		Returns a read-only slice, preserving its static dimensions.
		NS_CUDA_CALLABLE constexpr Span2D<const Type, Width, Height> operator[](size_t z) const noexcept
		{
			NS_ASSERT(z < depth());

			return Span2D<const Type, Width, Height>(m_data ? m_data + z * width() * height() : nullptr, width(), height());
		}

	private:

		const Type *	m_data{ nullptr };
	};

	/*****************************************************************************
	****************************    Span3D<Type>    *****************************
	*****************************************************************************/

	/**
	 *	@brief		A non-owning, mutable view over a contiguous 3D array.
	 *	@note		Follows Span's const-view inheritance and writable-storage contract.
	 *			A const view object exposes only read-only elements and subviews.
	 *	@warning	Do not rebind the inherited const base to truly const storage and
	 *				then write through this span; the type system cannot enforce this.
	 */
	template<typename Type, size_t Width, size_t Height, size_t Depth> class Span3D : public Span3D<const Type, Width, Height, Depth>
	{
		using _ConstBase = Span3D<const Type, Width, Height, Depth>;
		using _Extents = detail::Extents<Width, Height, Depth>;

	public: // Type definitions for the mutable span.

		using pointer = Type *;
		using reference = Type &;
		using element_type = Type;

	public: // Constructors for the span.

		//!	@brief		Copy constructor for the span.
		constexpr Span3D(const Span3D &) noexcept = default;

		//! @brief		Dynamic initialization is not supported for a `__constant__` variable
		constexpr Span3D() noexcept requires(_Extents::default_constructible) = default;

		//!	@brief		Constructs a span with all dimensions, or just the dynamic dimensions in order.
		template<typename... Sizes> NS_CUDA_CALLABLE constexpr explicit Span3D(Type * data, Sizes... sizes) noexcept
			requires((std::is_integral_v<Sizes> && ...) && (sizeof...(Sizes) == 3 || sizeof...(Sizes) == _Extents::rank_dynamic)) :
			_ConstBase(data, sizes...) {}

		//!	@brief		Converts compatible shapes; dynamic-to-fixed dimensions require explicit construction.
		template<size_t OtherWidth, size_t OtherHeight, size_t OtherDepth> NS_CUDA_CALLABLE constexpr
		explicit((Width != dynamic_extent && OtherWidth == dynamic_extent) || (Height != dynamic_extent && OtherHeight == dynamic_extent) || (Depth != dynamic_extent && OtherDepth == dynamic_extent))
		Span3D(const Span3D<Type, OtherWidth, OtherHeight, OtherDepth> & rhs) noexcept
			requires(detail::compatible_extent<Width, OtherWidth> && detail::compatible_extent<Height, OtherHeight> && detail::compatible_extent<Depth, OtherDepth>) : _ConstBase(rhs.data(), rhs.width(), rhs.height(), rhs.depth()) {}

		//!	@brief		Constructs a writable span from an array, matching fixed dimensions exactly.
		explicit(_Extents::rank_dynamic != 3) Span3D(Array3D<Type> & arr) noexcept :
			_ConstBase(arr.data(), arr.width(), arr.height(), arr.depth()) {}

	public: // Element access.

		using _ConstBase::data;
		using _ConstBase::operator();
		using _ConstBase::operator[];

		//!	@brief		Returns a pointer to the underlying writable storage.
		NS_CUDA_CALLABLE constexpr pointer data() noexcept { return const_cast<pointer>(_ConstBase::data()); }

		//!	@brief		Returns a reference to the element at the given coordinates.
		NS_CUDA_CALLABLE constexpr reference operator()(size_t x, size_t y, size_t z) noexcept
		{
			return const_cast<reference>(_ConstBase::operator()(x, y, z));
		}

		//!	@brief		Returns a mutable slice, preserving its static dimensions.
		NS_CUDA_CALLABLE constexpr Span2D<Type, Width, Height> operator[](size_t z) noexcept
		{
			auto subview = _ConstBase::operator[](z);

			return Span2D<Type, Width, Height>(const_cast<pointer>(subview.data()), subview.width(), subview.height());
		}
	};
}
