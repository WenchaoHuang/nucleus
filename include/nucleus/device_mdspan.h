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

namespace NS_NAMESPACE::dev
{
	template<typename Type, size_t... Extents> class Mdspan;

	namespace detail
	{
		template<size_t N> struct MdspanShapeStorage { size_t m_extents[N]{}; };
		template<> struct MdspanShapeStorage<0> {};

		//! @brief		Stores only the dynamic dimensions of a row-major view.
		template<size_t... Extents> struct MdspanShape : MdspanShapeStorage<((Extents == dynamic_extent ? 1 : 0) + ... + 0)>
		{
			static_assert(sizeof...(Extents) > 0);
			static constexpr size_t m_rank = sizeof...(Extents);
			static constexpr size_t m_rank_dynamic = ((Extents == dynamic_extent ? 1 : 0) + ... + 0);
			static constexpr size_t m_static_size = m_rank_dynamic == 0 ? (Extents * ... * size_t{ 1 }) : dynamic_extent;
			static constexpr bool m_default_constructible = m_rank_dynamic > 0 || ((Extents == 0) || ...);

			constexpr MdspanShape() noexcept = default;

			NS_CUDA_CALLABLE static constexpr size_t static_extent(size_t dimension) noexcept
			{
				NS_ASSERT(dimension < m_rank/* dimension out of range */);
				const size_t values[]{ Extents... };
				return values[dimension];
			}

			template<typename... Sizes> NS_CUDA_CALLABLE explicit constexpr MdspanShape(Sizes... sizes) noexcept
				requires((sizeof...(Sizes) == m_rank || sizeof...(Sizes) == m_rank_dynamic) && (std::is_convertible_v<Sizes, size_t> && ...))
			{
				const size_t values[]{ static_cast<size_t>(sizes)..., 0 };
				[[maybe_unused]] size_t dynamic_index = 0;
				for (size_t i = 0; i < m_rank; ++i)
				{
					if constexpr (m_rank_dynamic > 0)
						if (static_extent(i) == dynamic_extent)
						{
							this->m_extents[dynamic_index] = values[sizeof...(Sizes) == m_rank ? i : dynamic_index];
							++dynamic_index;
						}
					if constexpr (sizeof...(Sizes) == m_rank)
						NS_ASSERT(static_extent(i) == dynamic_extent || static_extent(i) == values[i]/* static extent mismatch */);
				}
			}

			NS_CUDA_CALLABLE constexpr size_t extent(size_t dimension) const noexcept
			{
				const size_t value = static_extent(dimension);
				if constexpr (m_rank_dynamic > 0)
					if (value == dynamic_extent)
					{
						size_t dynamic_index = 0;
						for (size_t i = 0; i < dimension; ++i)
							dynamic_index += static_extent(i) == dynamic_extent;
						return this->m_extents[dynamic_index];
					}
				return value;
			}

			NS_CUDA_CALLABLE constexpr size_t size() const noexcept
			{
				size_t result = 1;
				for (size_t i = 0; i < m_rank; ++i)
					result *= extent(i);
				return result;
			}

			NS_CUDA_CALLABLE constexpr size_t stride(size_t dimension) const noexcept
			{
				NS_ASSERT(dimension < m_rank/* dimension out of range */);
				size_t result = 1;
				for (size_t i = dimension + 1; i < m_rank; ++i)
					result *= extent(i);
				return result;
			}

			template<typename... Indices> NS_CUDA_CALLABLE constexpr size_t offset(Indices... indices) const noexcept
				requires(sizeof...(Indices) == m_rank && (std::is_convertible_v<Indices, size_t> && ...))
			{
				const size_t values[]{ static_cast<size_t>(indices)... };
				size_t result = 0;
				for (size_t i = 0; i < m_rank; ++i)
				{
					NS_ASSERT(values[i] < extent(i)/* index out of range */);
					result = result * extent(i) + values[i];
				}
				return result;
			}

			template<size_t... OtherExtents> static constexpr bool compatible() noexcept
			{
				if constexpr (sizeof...(OtherExtents) != m_rank)
					return false;
				else
					return ((Extents == dynamic_extent || OtherExtents == dynamic_extent || Extents == OtherExtents) && ...);
			}
		};

		template<typename Type, size_t... Extents> struct MdspanBase : MdspanShape<Extents...>
		{
			constexpr MdspanBase() noexcept = default;

			template<typename... Sizes> NS_CUDA_CALLABLE constexpr MdspanBase(Type * data, Sizes... sizes) noexcept :
				MdspanShape<Extents...>(sizes...), m_data(data) {}

			Type * m_data{ nullptr };
		};
	}

	/*****************************************************************************
	*********************    Mdspan<const Type, Extents...>    ********************
	*****************************************************************************/

	/**
	 *	@brief		A non-owning, row-major multidimensional view over const objects.
	 *	@tparam		Type - The element type (non-const).
	 *	@tparam		Extents - Static dimensions or dynamic_extent values.
	 */
	template<typename Type, size_t... Extents> class Mdspan<const Type, Extents...> : protected detail::MdspanBase<const Type, Extents...>
	{
		using _Base = detail::MdspanBase<const Type, Extents...>;
		using _Shape = detail::MdspanShape<Extents...>;

		template<typename, size_t...> friend class Mdspan;

		template<typename OtherType, size_t... OtherExtents, size_t... I> NS_CUDA_CALLABLE constexpr
		Mdspan(const Mdspan<OtherType, OtherExtents...> & rhs, std::index_sequence<I...>) noexcept : _Base(rhs.data(), rhs.extent(I)...) {}

	public:	// Type definitions.

		using size_type = size_t;
		using pointer = const Type *;
		using reference = const Type &;
		using element_type = const Type;
		using const_pointer = const Type *;
		using const_reference = const Type &;
		using value_type = std::remove_cv_t<Type>;

	public:	// Constructors.

		//! @brief		Dynamic initialization is not supported for a `__constant__` variable.
		constexpr Mdspan() noexcept requires(_Shape::m_default_constructible) = default;
		constexpr Mdspan(const Mdspan &) noexcept = default;

		//! @brief		Constructor with all dimensions or only the dynamic dimensions.
		template<typename... Sizes> NS_CUDA_CALLABLE explicit constexpr Mdspan(const Type * data, Sizes... sizes) noexcept
			requires((sizeof...(Sizes) == _Shape::m_rank || sizeof...(Sizes) == _Shape::m_rank_dynamic)
				&& (std::is_convertible_v<Sizes, size_type> && ...)) : _Base(data, sizes...) {}

		//! @brief		Constructor from a view with compatible elements and dimensions.
		template<typename OtherType, size_t... OtherExtents> NS_CUDA_CALLABLE constexpr Mdspan(const Mdspan<OtherType, OtherExtents...> & rhs) noexcept
			requires(_Shape::template compatible<OtherExtents...>() && std::is_convertible_v<OtherType(*)[], const Type(*)[]>) :
			Mdspan(rhs, std::make_index_sequence<_Shape::m_rank>{}) {}

	public: // Observers.

		NS_CUDA_CALLABLE static constexpr size_type rank() noexcept { return _Shape::m_rank; }
		NS_CUDA_CALLABLE static constexpr size_type rank_dynamic() noexcept { return _Shape::m_rank_dynamic; }
		using _Shape::static_extent;
		using _Shape::extent;
		using _Shape::stride;

		NS_CUDA_CALLABLE constexpr const_pointer data() const noexcept { return _Base::m_data; }
		NS_CUDA_CALLABLE constexpr bool empty() const noexcept { return data() == nullptr; }
		NS_CUDA_CALLABLE constexpr size_type size() const noexcept requires(_Shape::m_rank_dynamic > 0) { return _Shape::size(); }
		NS_CUDA_CALLABLE static constexpr size_type size() noexcept requires(_Shape::m_rank_dynamic == 0) { return _Shape::m_static_size; }
		NS_CUDA_CALLABLE constexpr size_type size_bytes() const noexcept requires(_Shape::m_rank_dynamic > 0) { return size() * sizeof(Type); }
		NS_CUDA_CALLABLE static constexpr size_type size_bytes() noexcept requires(_Shape::m_rank_dynamic == 0) { return size() * sizeof(Type); }

	public: // Element access.

		//! @brief		Returns a const reference at the given multidimensional index.
		template<typename... Indices> NS_CUDA_CALLABLE constexpr const_reference operator()(Indices... indices) const noexcept
			requires(sizeof...(Indices) == rank() && (std::is_convertible_v<Indices, size_type> && ...))
		{
			return data()[_Shape::offset(indices...)];
		}

		NS_CUDA_CALLABLE constexpr const_reference operator[](size_type index) const noexcept requires(rank() == 1) { return (*this)(index); }

		//! @brief		Returns a one-dimensional span over the same storage.
		NS_CUDA_CALLABLE constexpr auto flatten() const noexcept { return Span<const Type, _Shape::m_static_size>(data(), size()); }
	};

	/*****************************************************************************
	************************    Mdspan<Type, Extents...>    ***********************
	*****************************************************************************/

	//! @brief		A mutable view inheriting the const view, as in dev::Span.
	template<typename Type, size_t... Extents> class Mdspan : public Mdspan<const Type, Extents...>
	{
		using _ConstBase = Mdspan<const Type, Extents...>;
		using _Shape = detail::MdspanShape<Extents...>;

	public:	// Type definitions.

		using pointer = Type *;
		using reference = Type &;
		using element_type = Type;
		using size_type = typename _ConstBase::size_type;

	public:	// Constructors.

		//! @brief		Dynamic initialization is not supported for a `__constant__` variable.
		constexpr Mdspan() noexcept requires(_Shape::m_default_constructible) = default;
		constexpr Mdspan(const Mdspan &) noexcept = default;

		template<typename... Sizes> NS_CUDA_CALLABLE explicit constexpr Mdspan(Type * data, Sizes... sizes) noexcept
			requires((sizeof...(Sizes) == _Shape::m_rank || sizeof...(Sizes) == _Shape::m_rank_dynamic)
				&& (std::is_convertible_v<Sizes, size_type> && ...)) : _ConstBase(data, sizes...) {}

		template<typename OtherType, size_t... OtherExtents> NS_CUDA_CALLABLE constexpr Mdspan(const Mdspan<OtherType, OtherExtents...> & rhs) noexcept
			requires(_Shape::template compatible<OtherExtents...>() && std::is_convertible_v<OtherType(*)[], Type(*)[]>) : _ConstBase(rhs) {}

	public: // Element access.

		using _ConstBase::data;
		using _ConstBase::operator();
		using _ConstBase::operator[];
		using _ConstBase::flatten;

		NS_CUDA_CALLABLE constexpr pointer data() noexcept { return const_cast<pointer>(_ConstBase::data()); }

		template<typename... Indices> NS_CUDA_CALLABLE constexpr reference operator()(Indices... indices) noexcept
			requires(sizeof...(Indices) == _Shape::m_rank && (std::is_convertible_v<Indices, size_type> && ...))
		{
			return data()[_Shape::offset(indices...)];
		}

		NS_CUDA_CALLABLE constexpr reference operator[](size_type index) noexcept requires(_Shape::m_rank == 1) { return (*this)(index); }

		NS_CUDA_CALLABLE constexpr auto flatten() noexcept { return Span<Type, _Shape::m_static_size>(data(), _ConstBase::size()); }
	};
}
