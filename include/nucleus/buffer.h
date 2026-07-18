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

#include "allocator.h"

namespace NS_NAMESPACE
{
	/*****************************************************************************
	********************************    Buffer    ********************************
	*****************************************************************************/

	/**
	 *	@brief		Shared handle for memory management.
	 *	@note		The allocated memory is owned by an internal body and remains valid until
	 *				the last Buffer handle that references it is destroyed.
	 */
	class Buffer
	{
	public:

		/**
		 *	@brief		Constructs an empty buffer.
		 *	@note		No memory allocation is performed.
		 */
		Buffer() noexcept = default;


		/**
		 *	@brief		Constructs a buffer with specified capacity.
		 *	@param[in]	allocator - Shared pointer to the memory allocator.
		 *	@param[in]	capacity - Requested buffer capacity in bytes.
		 *	@throws		cudaError_t - In case of failure.
		 *	@note		The actual allocation is delegated to the provided Allocator.
		 */
		explicit Buffer(std::shared_ptr<Allocator> allocator, size_t capacity)
		{
			if (capacity > 0) { m_storage = std::make_shared<Storage>(std::move(allocator), capacity); }
		}

	public:

		/**
		 *	@brief		Gets the associated memory allocator.
		 *	@return		Shared pointer to the Allocator instance.
		 */
		const std::shared_ptr<Allocator> & allocator() const noexcept
		{
			static const std::shared_ptr<Allocator> nullAllocator;

			return m_storage ? m_storage->allocator : nullAllocator;
		}


		/**
		 *	@brief		Checks if buffer is empty.
		 *	@return		Returns `true` if storage is empty (nullptr), `false` otherwise.
		 */
		bool empty() const noexcept { return m_storage == nullptr; }


		/**
		 *	@brief		Checks whether this handle references a buffer body.
		 */
		explicit operator bool() const noexcept { return m_storage != nullptr; }


		/**
		 *	@brief		Gets the buffer capacity.
		 *	@return		Returns the total number of bytes allocated for this buffer.
		 */
		size_t capacity() const noexcept { return m_storage ? m_storage->capacity : 0; }


		/**
		 *	@brief		Retrun logical address of the memory.
		 */
		std::uintptr_t address() const noexcept { return reinterpret_cast<std::uintptr_t>(this->data()); }


		/**
		 *	@brief		Gets constant pointer to the underlying data.
		 *	@return		Const pointer to the buffer data.
		 *	@note		The returned pointer remains valid until the last sharing Buffer handle is destroyed.
		 */
		const void * data() const noexcept { return m_storage ? m_storage->data : nullptr; }


		/**
		 *	@brief		Gets mutable pointer to the underlying data.
		 *	@return		Pointer to the buffer data.
		 *	@note		The returned pointer remains valid until the last sharing Buffer handle is destroyed.
		 */
		void * data() noexcept { return m_storage ? m_storage->data : nullptr; }

	private:

		/**
		 *	@brief		Internal storage structure for buffer data and allocator.
		 */
		struct Storage
		{
			const std::shared_ptr<Allocator>		allocator;
			const size_t							capacity;
			void * const							data;

			//!	@brief		Constructs a Storage instance with specified allocator and capacity.
			explicit Storage(std::shared_ptr<Allocator> allocator, size_t capacity)
				: allocator(std::move(allocator)), capacity(capacity), data(this->allocator->allocateMemory(capacity)) {}

			//!	@brief		Destructs the Storage instance and deallocates the associated memory.	
			~Storage() noexcept { if (data != nullptr) { allocator->deallocateMemory(data); } }
		};
		
		std::shared_ptr<Storage>		m_storage;
	};
}
