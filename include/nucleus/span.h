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
#include <array>
#include <vector>

namespace NS_NAMESPACE
{
	inline constexpr size_t dynamic_extent = static_cast<size_t>(-1);
	template<typename Type, size_t Extent = dynamic_extent> class Span;
	template<typename Type, size_t Extent> NS_CUDA_CALLABLE constexpr auto as_bytes(const Span<const Type, Extent> & span) noexcept;
	template<typename Type, size_t Extent> NS_CUDA_CALLABLE constexpr auto as_writeable_bytes(Span<Type, Extent> & span) noexcept;

	namespace details
	{
		/*************************************************************************
		*************************    SpanExtentType    ***************************
		*************************************************************************/

		/**
		 *	@brief		Helper class to store the data pointer and size of a span.
		 */
		template<typename Type, size_t Extent> struct SpanExtentType
		{
		#ifndef __CUDA_ARCH__
			//!	For host code which has initialization.
			constexpr SpanExtentType() noexcept = default;
		#else
			//! Dynamic initialization is not supported for a `__constant__` variable
			NS_CUDA_CALLABLE SpanExtentType() noexcept {}
		#endif

			//!	Constructor for a span with a fixed extent.
			NS_CUDA_CALLABLE SpanExtentType(Type * data, size_t) : m_data(data)
			{
				static_assert(m_size <= dynamic_extent / sizeof(Type), "size of span in bytes exceeds numeric_limits<size_t>::max()");
			}

			//!	Constant size of the span, which is equal to `Extent`.
			static constexpr size_t m_size = Extent;

			Type * m_data{ nullptr };
		};
		
		/*************************************************************************
		***************    SpanExtentType<Type, dynamic_extent>    ***************
		*************************************************************************/

		/**
		 *	@brief		Helper class to store the data pointer and size of a span with a dynamic extent.
		 */
		template<typename Type> struct SpanExtentType<Type, dynamic_extent>
		{
		#ifndef __CUDA_ARCH__
			//!	For host code which has initialization.
			constexpr SpanExtentType() noexcept = default;
		#else
			//! Dynamic initialization is not supported for a `__constant__` variable
			NS_CUDA_CALLABLE SpanExtentType() noexcept {}
		#endif

			//!	Constructor for a span with a dynamic extent.
			NS_CUDA_CALLABLE SpanExtentType(Type * data, size_t size) : m_data(data), m_size(size)
			{
				NS_ASSERT(m_size <= dynamic_extent / sizeof(Type)/* size of span in bytes exceeds numeric_limits<size_t>::max() */);
			}
			
			Type * m_data{ nullptr };
			size_t m_size{ 0 };
		};
	}

	/*****************************************************************************
	***********************    Span<const Type, Extent>    ***********************
	*****************************************************************************/

	/**
	 *	@brief	A non-owning view over a contiguous sequence of const objects.
	 *	@tparam	Type - The element type (non-const, e.g. int, not const int).
	 *	@tparam	Extent - The number of elements (static extent) or dynamic_extent.
	 *	@note	This is the const-view specialization. `Span<T>` inherits from this,
	 *			providing implicit conversion from mutable span to const span.
	 */
	template<typename Type, size_t Extent> class Span<const Type, Extent> : private details::SpanExtentType<const Type, Extent>
	{
		using _SpanBase = details::SpanExtentType<const Type, Extent>;

	public:

		//!	Type definitions for the span.
		using size_type = size_t;
		using element_type = const Type;
		using value_type = std::remove_cv_t<Type>;
		using difference_type = std::ptrdiff_t;
		using pointer = const Type *;
		using const_pointer = const Type *;
		using reference = const Type &;
		using const_reference = const Type &;
		using iterator = const Type *;
		using const_iterator = const Type *;

	public:	//!	Constructors for the span.
		
	#ifndef __CUDA_ARCH__
		//!	For host code which has initialization.
		constexpr Span() noexcept = default;
	#else
		//! Dynamic initialization is not supported for a `__constant__` variable
		NS_CUDA_CALLABLE Span() noexcept {}
	#endif
		//!	Constructor for a span with a pointer and size.
		NS_CUDA_CALLABLE explicit Span(const Type * data, size_t size) : _SpanBase(data, size) {}

		//!	Constructor for a span from `std::vector`.
		Span(const std::vector<Type> & data) requires(Extent == dynamic_extent) : _SpanBase(data.data(), data.size()) {}

		//!	Constructor for a span from `std::initializer_list`.
		Span(const std::initializer_list<Type> & data) requires(Extent == dynamic_extent) : _SpanBase(data.begin(), data.size()) {}

		//!	Constructor for a span with a fixed array.
		template<size_t N> NS_CUDA_CALLABLE Span(const Type (&data)[N]) requires(Extent == dynamic_extent || Extent == N) : _SpanBase(data, N) {}

		//!	Constructor for a span with fixed extent from `std::array`.
		template<size_t N> Span(const std::array<Type, N> & data) requires(Extent == dynamic_extent || Extent == N) : _SpanBase(data.data(), N) {}

	public: //!	Element access.

		//!	Returns a const pointer to the underlying data.
		NS_CUDA_CALLABLE constexpr pointer data() const noexcept { return _SpanBase::m_data; }

		//!	Returns a const reference to the element at the given index.
		NS_CUDA_CALLABLE constexpr const_reference operator[](size_t index) const
		{
			NS_ASSERT(index < size()/* index out of range */);

			return _SpanBase::m_data[index];
		}

		//!	Returns a const reference to the first element.
		NS_CUDA_CALLABLE constexpr const_reference front() const
		{
			NS_ASSERT(!empty()/* front() called on empty span */);

			return _SpanBase::m_data[0];
		}

		//!	Returns a const reference to the last element.
		NS_CUDA_CALLABLE constexpr const_reference back() const
		{
			NS_ASSERT(!empty()/* back() called on empty span */);

			return _SpanBase::m_data[size() - 1];
		}

	public: //!	Iterators.

		//!	Returns a const iterator to the beginning.
		NS_CUDA_CALLABLE constexpr const_iterator begin() const noexcept { return _SpanBase::m_data; }

		//!	Returns a const iterator to the end.
		NS_CUDA_CALLABLE constexpr const_iterator end() const noexcept { return _SpanBase::m_data + size(); }

		//!	Returns a const iterator to the beginning (explicit).
		NS_CUDA_CALLABLE constexpr const_iterator cbegin() const noexcept { return begin(); }

		//!	Returns a const iterator to the end (explicit).
		NS_CUDA_CALLABLE constexpr const_iterator cend() const noexcept { return end(); }

	public: //!	Observers.

		//!	Returns the number of elements in the span.
		NS_CUDA_CALLABLE constexpr size_type size() const noexcept { return _SpanBase::m_size; }

		//!	Returns the size of the span in bytes.
		NS_CUDA_CALLABLE constexpr size_type size_bytes() const noexcept { return size() * sizeof(Type); }

		//!	Checks if the span is empty.
		NS_NODISCARD NS_CUDA_CALLABLE constexpr bool empty() const noexcept { return size() == 0; }

		//!	Returns the number of elements in the span (static for fixed extent).
		static constexpr size_type size() noexcept requires(Extent != dynamic_extent) { return Extent; }

	public: //!	Subviews.

		/**
		 *	@brief		Returns a subspan of the first \p count elements.
		 *	@param[in]	count - Number of elements in the subspan.
		 */
		NS_CUDA_CALLABLE constexpr Span<const Type, dynamic_extent> first(size_type count) const
		{
			NS_ASSERT(count <= size()/* first() count exceeds span size */);

			return Span<const Type, dynamic_extent>(_SpanBase::m_data, count);
		}

		/**
		 *	@brief		Returns a subspan of the last \p count elements.
		 *	@param[in]	count - Number of elements in the subspan.
		 */
		NS_CUDA_CALLABLE constexpr Span<const Type, dynamic_extent> last(size_type count) const
		{
			NS_ASSERT(count <= size()/* last() count exceeds span size */);

			return Span<const Type, dynamic_extent>(_SpanBase::m_data + size() - count, count);
		}

		/**
		 *	@brief		Returns a subspan starting at \p offset with \p count elements.
		 *	@param[in]	offset - Start element index (relative to this span).
		 *	@param[in]	count - Number of elements in the subspan (dynamic_extent for rest).
		 */
		NS_CUDA_CALLABLE constexpr Span<const Type, dynamic_extent> subspan(size_type offset, size_type count = dynamic_extent) const
		{
			NS_ASSERT(offset <= size()/* subspan() offset exceeds span size */);

			if (count == dynamic_extent)
				count = size() - offset;
			else
				NS_ASSERT(offset + count <= size()/* subspan() count exceeds span bounds */);

			return Span<const Type, dynamic_extent>(_SpanBase::m_data + offset, count);
		}
	};

	/*****************************************************************************
	**************************    Span<Type, Extent>    **************************
	*****************************************************************************/

	/**
	 *	@brief	A non-owning view over a contiguous sequence of mutable objects.
	 *	@tparam	Type - The element type.
	 *	@tparam	Extent - The number of elements (static extent) or dynamic_extent.
	 *	@note	Inherits from `Span<const Type, Extent>` for implicit const conversion.
	 */
	template<typename Type, size_t Extent> class Span : public Span<const Type, Extent>
	{
		using _ConstBase = Span<const Type, Extent>;

	public:

		//!	Type definitions for the mutable span.
		using element_type = Type;
		using pointer = Type *;
		using reference = Type &;
		using iterator = Type *;

	public:	//!	Constructors for the span.

	#ifndef __CUDA_ARCH__
		//!	For host code which has initialization.
		constexpr Span() noexcept = default;
	#else
		//! Dynamic initialization is not supported for a `__constant__` variable
		NS_CUDA_CALLABLE Span() noexcept {}
	#endif
		//!	Constructor for a span with a pointer and size.
		NS_CUDA_CALLABLE explicit Span(Type * data, size_t size) : _ConstBase(data, size) {}

		//!	Constructor for a span from std::vector.
		Span(std::vector<Type> & data) requires(Extent == dynamic_extent) : _ConstBase(data.data(), data.size()) {}

		//!	Constructor for a span with a fixed array.
		template<size_t N> NS_CUDA_CALLABLE Span(Type (&data)[N]) requires(Extent == dynamic_extent || Extent == N) : _ConstBase(data, N) {}

		//!	Constructor for a span with fixed extent from std::array.
		template<size_t N> Span(std::array<Type, N> & data) requires(Extent == dynamic_extent || Extent == N) : _ConstBase(data.data(), N) {}

	public: //!	Element access.

		//!	Returns a pointer to the underlying data.
		NS_CUDA_CALLABLE constexpr pointer data() const noexcept { return const_cast<pointer>(_ConstBase::data()); }

		//!	Returns a reference to the element at the given index.
		NS_CUDA_CALLABLE constexpr reference operator[](size_t index) const
		{
			NS_ASSERT(index < this->size()/* index out of range */);

			return const_cast<pointer>(this->data())[index];
		}

		//!	Returns a reference to the first element.
		NS_CUDA_CALLABLE constexpr reference front() const
		{
			NS_ASSERT(!this->empty()/* front() called on empty span */);

			return const_cast<pointer>(this->data())[0];
		}

		//!	Returns a reference to the last element.
		NS_CUDA_CALLABLE constexpr reference back() const
		{
			NS_ASSERT(!this->empty()/* back() called on empty span */);

			return const_cast<pointer>(this->data())[this->size() - 1];
		}

	public: //!	Iterators.

		//!	Returns an iterator to the beginning.
		NS_CUDA_CALLABLE constexpr iterator begin() const noexcept { return data(); }

		//!	Returns an iterator to the end.
		NS_CUDA_CALLABLE constexpr iterator end() const noexcept { return data() + this->size(); }

	public: //!	Subviews.

		/**
		 *	@brief		Returns a subspan of the first \p count elements.
		 *	@param[in]	count - Number of elements in the subspan.
		 */
		NS_CUDA_CALLABLE constexpr Span<Type, dynamic_extent> first(size_t count) const
		{
			NS_ASSERT(count <= this->size()/* first() count exceeds span size */);

			return Span<Type, dynamic_extent>(data(), count);
		}

		/**
		 *	@brief		Returns a subspan of the last \p count elements.
		 *	@param[in]	count - Number of elements in the subspan.
		 */
		NS_CUDA_CALLABLE constexpr Span<Type, dynamic_extent> last(size_t count) const
		{
			NS_ASSERT(count <= this->size()/* last() count exceeds span size */);

			return Span<Type, dynamic_extent>(data() + this->size() - count, count);
		}

		/**
		 *	@brief		Returns a subspan starting at \p offset with \p count elements.
		 *	@param[in]	offset - Start element index (relative to this span).
		 *	@param[in]	count - Number of elements in the subspan (dynamic_extent for rest).
		 */
		NS_CUDA_CALLABLE constexpr Span<Type, dynamic_extent> subspan(size_t offset, size_t count = dynamic_extent) const
		{
			NS_ASSERT(offset <= this->size()/* subspan() offset exceeds span size */);

			if (count == dynamic_extent)
				count = this->size() - offset;
			else
				NS_ASSERT(offset + count <= this->size()/* subspan() count exceeds span bounds */);

			return Span<Type, dynamic_extent>(data() + offset, count);
		}
	};

	//!	Returns a span viewing the same data as `const byte` (non-modifiable).
	template<typename Type, size_t Extent> NS_CUDA_CALLABLE constexpr auto as_bytes(const Span<const Type, Extent> & span) noexcept
	{
		if constexpr (Extent != dynamic_extent)
			return Span<const byte, Extent * sizeof(Type)>(reinterpret_cast<const byte*>(span.data()), span.size_bytes());
		else
			return Span<const byte, dynamic_extent>(reinterpret_cast<const byte*>(span.data()), span.size_bytes());
	}

	//!	Returns a span viewing the same data as writable `byte`.
	template<typename Type, size_t Extent> NS_CUDA_CALLABLE constexpr auto as_writable_bytes(Span<Type, Extent> & span) noexcept
	{
		if constexpr (Extent != dynamic_extent)
			return Span<byte, Extent * sizeof(Type)>(reinterpret_cast<byte*>(span.data()), span.size_bytes());
		else
			return Span<byte, dynamic_extent>(reinterpret_cast<byte*>(span.data()), span.size_bytes());
	}
}