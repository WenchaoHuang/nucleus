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
#include "buffer.h"
#include "logger.h"
#include "runtime.h"
#include <utility>

namespace NS_NAMESPACE
{
	/*****************************************************************************
	********************************    Array    *********************************
	*****************************************************************************/

	/**
	 *	@brief		Owns a contiguous allocation of typed memory through a `Buffer`.
	 *	@note		Elements are not initialized, constructed, or destroyed.
	 */
	template<typename Type> class Array
	{
		NS_NONCOPYABLE(Array)

	public:

		//!	@brief		Default constructor.
		Array() noexcept = default;

		//!	@brief		Allocates array with \p count elements using the default allocator.
		explicit Array(size_t count) : Array(Runtime::defaultAllocator(), count) {}

		//!	@brief		Allocates array with \p count elements.
		explicit Array(std::shared_ptr<Allocator> allocator, size_t count) : Array() { this->resize(std::move(allocator), count); }

		//!	@note		Transfers ownership from another array.
		void operator=(Array && rhs) noexcept { m_buffer = std::move(rhs.m_buffer); }

		//!	@brief		Move constructor. Transfers ownership from another array.
		Array(Array && rhs) noexcept : m_buffer(std::move(rhs.m_buffer)) {}

	public:

		//!	@brief		Checks if the array is empty.
		bool empty() const noexcept { return m_buffer.empty(); }

		//!	@brief		Returns the size of the allocation in bytes.
		size_t size_bytes() const noexcept { return m_buffer.capacity(); }

		//!	@brief		Returns the number of elements.
		size_t size() const noexcept { return m_buffer.capacity() / sizeof(Type); }

		//!	@brief		Returns a mutable pointer to the underlying memory.
		Type * data() noexcept { return reinterpret_cast<Type*>(m_buffer.data()); }

		//!	@brief		Returns a const pointer to the underlying memory.
		const Type * data() const noexcept { return reinterpret_cast<const Type*>(m_buffer.data()); }

	public:

		/**
		 *	@brief		Gets the allocator associated with.
		 */
		const std::shared_ptr<Allocator> & allocator() const noexcept { return m_buffer.allocator(); }


		/**
		 *	@brief		Resizes the array using a new allocator.
		 *	@param[in]	allocator - The new allocator to use.
		 *	@param[in]	count - The new number of elements.
		 *	@note		If the allocator or size changes, existing data will be lost.
		 */
		void resize(std::shared_ptr<Allocator> allocator, size_t count)
		{
			NS_ASSERT_LOG_IF(allocator == nullptr, "Empty allocator!");

			if ((this->allocator() != allocator) || (this->size() != count))
			{
				m_buffer = Buffer(std::move(allocator), sizeof(Type) * count);
			}
		}


		/**
		 *	@brief		Resizes the array using the default allocator.
		 *	@param[in]	count - The new number of elements.
		 *	@note		If the allocator or size changes, existing data will be lost.
		 */
		void resize(size_t count)
		{
			auto allocator = Runtime::defaultAllocator();

			NS_ASSERT_LOG_IF(!allocator, "No default allocator!");

			this->resize(allocator, count);
		}


		/**
		 *	@brief		Releases ownership of the internal buffer.
		 *	@return		The released buffer (empty if the array was empty).
		 *	@note		After this call, the array will be empty but still valid.
		 */
		Buffer releaseBuffer() noexcept { return std::exchange(m_buffer, Buffer()); }


		/**
		 *	@brief		Swaps contents with another array.
		 */
		void swap(Array & rhs) noexcept { std::swap(m_buffer, rhs.m_buffer); }


		/**
		 *	@brief		Clears the array and releases all allocated memory.
		 */
		void clear() noexcept { m_buffer = Buffer{}; }

	private:

		Buffer		m_buffer;
	};
}
