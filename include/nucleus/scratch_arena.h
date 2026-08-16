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

#include "buffer.h"
#include "utility.h"
#include "device_span.h"
#include "device_pointer.h"

namespace NS_NAMESPACE
{
	/*****************************************************************************
	*****************************    ScratchArena    *****************************
	*****************************************************************************/

	/**
	 *	@brief		Reusable temporary memory space backed by a single `Buffer`.
	 *	@note		A `ScratchArena` caches its largest allocation and reuses it for subsequent operations.
	 *	@note		Temporary pointers become invalid when the scratch arena is reused, cleared,
	 *				destroyed, or its buffer grows.
	 *	@warning	For asynchronous work, reuse is safe only when accesses are ordered on the same
	 *				stream or explicitly synchronized by the caller.
	 */
	class ScratchArena
	{
		NS_NONCOPYABLE(ScratchArena)

	public:

		//!	@brief		Constructs an empty scratch arena using the default allocator.
		ScratchArena() : ScratchArena(Runtime::defaultAllocator()) {}

		//!	@brief		Constructs a scratch arena with initial capacity using the default allocator.
		explicit ScratchArena(size_t capacity) : ScratchArena(Runtime::defaultAllocator(), capacity) {}

		//! @brief		Constructs an empty scratch arena using the specified allocator.
		explicit ScratchArena(std::shared_ptr<Allocator> allocator) : ScratchArena(std::move(allocator), 0) {}

		//! @brief		Constructs a scratch arena with the specified allocator and initial capacity.
		explicit ScratchArena(std::shared_ptr<Allocator> allocator, size_t capacity) : m_allocator(std::move(allocator)), m_offset(0) { this->reserve(capacity); }

	public:

		//!	@brief		Checks whether the scratch arena has no cached buffer.
		bool empty() const noexcept { return m_buffer.empty(); }

		//! @brief		Returns the cached buffer capacity in bytes.
		size_t capacity() const noexcept { return m_buffer.capacity(); }

		//! @brief		Returns the number of bytes used since the last reuse.
		size_t used() const noexcept { return m_offset; }

		//! @brief		Returns the number of unused bytes in the cached buffer.
		size_t remaining() const noexcept { return this->capacity() - m_offset; }

		//! @brief		Returns the scratch arena allocator.
		const std::shared_ptr<Allocator> & allocator() const noexcept { return m_allocator; }

	public:

		/**
		 *	@brief		Allocates a typed contiguous temporary array.
		 *	@param[in]	count - Number of elements to allocate.
		 *	@return		Non-owning device pointer to the temporary array.
		 *	@warning	The returned pointer becomes invalid when the scratch arena is reused,
		 *				cleared, destroyed, or its buffer grows.
		 */
		template<typename Type> dev::Span<Type> allocate(size_t count)
		{
			static_assert(!std::is_void_v<Type>, "ScratchArena cannot allocate void elements!");
			static_assert(!std::is_const_v<Type> && !std::is_volatile_v<Type>, "ScratchArena allocations must be mutable!");

			if (count == 0)
				return dev::Span<Type>(nullptr, 0);

			const size_t alignedOffset = align_up(m_offset, alignof(Type));
			const size_t allocationEnd = alignedOffset + sizeof(Type) * count;

			NS_ASSERT(allocationEnd <= m_buffer.capacity());

			m_offset = allocationEnd;

			auto address = m_buffer.address() + alignedOffset;

			return dev::Span<Type>(reinterpret_cast<Type*>(address), count);
		}


		/**
		 *	@brief		Allocates a packed 2D temporary array.
		 *	@param[in]	width - Number of columns.
		 *	@param[in]	height - Number of rows.
		 *	@return		Non-owning 2D device pointer to the temporary array.
		 */
		template<typename Type> dev::Ptr2<Type> allocate2D(uint32_t width, uint32_t height)
		{
			auto values = this->allocate<Type>(width * height);

			return dev::Ptr2<Type>(values.data(), width, height);
		}


		/**
		 *	@brief		Allocates a packed 3D temporary array.
		 *	@param[in]	width - Number of columns.
		 *	@param[in]	height - Number of rows.
		 *	@param[in]	depth - Number of layers.
		 *	@return		Non-owning 3D device pointer to the temporary array.
		 */
		template<typename Type> dev::Ptr3<Type> allocate3D(uint32_t width, uint32_t height, uint32_t depth)
		{
			auto values = this->allocate<Type>(width * height * depth);

			return dev::Ptr3<Type>(values.data(), width, height, depth);
		}


		/**
		 *	@brief		Ensures that the cached buffer has at least the requested capacity.
		 *	@param[in]	capacity - Minimum capacity in bytes.
		 *	@note		Existing temporary allocations are discarded when the buffer grows.
		 */
		void reserve(size_t capacity)
		{
			if (capacity > m_buffer.capacity())
			{
				m_buffer = Buffer(m_allocator, capacity);

				m_offset = 0;
			}
		}


		/**
		 *	@brief		Makes all temporary allocations available for reuse.
		 *	@warning	All previously returned pointers become invalid for subsequent operations.
		 */
		void reuse() noexcept { m_offset = 0; }


		/**
		 *	@brief		Releases the cached buffer and invalidates all temporary pointers.
		 */
		void clear() noexcept
		{
			m_buffer = Buffer();

			m_offset = 0;
		}

	private:

		std::shared_ptr<Allocator>		m_allocator;
		Buffer							m_buffer;
		size_t							m_offset;
	};
}
