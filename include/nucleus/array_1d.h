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
#include "buffer_view.h"
#include "device_pointer.h"

namespace NS_NAMESPACE
{
	/*****************************************************************************
	********************************    Array    *********************************
	*****************************************************************************/

	/**
	 *	@brief		A 1D array template that provides device-accessible memory management.
	 */
	template<typename Type> class Array : public dev::Ptr<Type>
	{
		NS_NONCOPYABLE(Array)

	public:

		//!	@brief		Construct an empty array.
		Array() noexcept : dev::Ptr<Type>(nullptr) {}

		//!	@brief		Allocates array with \p width elements.
		explicit Array(std::shared_ptr<Allocator> alloctor, size_t width) : Array() { this->resize(alloctor, width); }

		//!	@brief		Move constructor. Transfers ownership from another array.
		Array(Array && rhs) : dev::Ptr<Type>(std::exchange(rhs.m_data, nullptr), std::exchange(rhs.m_width, 0)), m_buffer(std::move(rhs.m_buffer)) {}

	public:

		/**
		 *	@brief		Resizes the array using a new allocator.
		 *	@param[in]	allocator - The new allocator to use.
		 *	@param[in]	width - The new number of elements.
		 *	@note		If the allocator or size changes, existing data will be lost.
		 */
		void resize(std::shared_ptr<Allocator> allocator, size_t width)
		{
			NS_ASSERT_LOG_IF(allocator == nullptr, "Empty allocator!");

			if ((this->allocator() != allocator) || (this->size() != width))
			{
				m_buffer = Buffer(allocator, sizeof(Type) * width);
					
				dev::Ptr<Type>::m_data = reinterpret_cast<Type*>(m_buffer.data());

				dev::Ptr<Type>::m_width = width;
			}
		}


		/**
		 *	@brief		Resizes the array using the current allocator.
		 *	@param[in]	width - The new number of elements.
		 *	@note		If the size changes, existing data will be lost.
		 */
		void resize(size_t width)
		{
			NS_ASSERT_LOG_IF(!m_buffer, "Empty allocator!");

			this->resize(m_buffer.allocator(), width);
		}


		/**
		 *	@brief		Gets the allocator associated with.
		 */
		 std::shared_ptr<Allocator> allocator() const
		 {
			 return m_buffer.allocator();
		 }


		/**
		 *	@brief		Releases ownership of the internal buffer.
		 *	@return		The released buffer (nullptr if array was empty).
		 *	@note		After this call, the array will be empty but still valid.
		 */
		Buffer releaseBuffer() noexcept
		{
			dev::Ptr<Type>::m_width = 0;

			dev::Ptr<Type>::m_data = nullptr;

			return std::exchange(m_buffer, Buffer());
		}


		/**
		 *	@brief		Move assignment operator.
		 */
		void operator=(Array && rhs) noexcept
		{
			dev::Ptr<Type>::m_data = std::exchange(rhs.m_data, nullptr);

			dev::Ptr<Type>::m_width = std::exchange(rhs.m_width, 0);

			m_buffer = std::move(rhs.m_buffer);
		}


		/**
		 *	@brief		Swaps contents with another array.
		 */
		void swap(Array & rhs) noexcept
		{
			std::swap(dev::Ptr<Type>::m_width, rhs.m_width);

			std::swap(dev::Ptr<Type>::m_data, rhs.m_data);

			std::swap(m_buffer, rhs.m_buffer);
		}


		/**
		 *	@brief		Clears the array and releases all allocated memory.
		 */
		void clear() noexcept
		{
			if (m_buffer)
			{
				dev::Ptr<Type>::m_data = nullptr;

				dev::Ptr<Type>::m_width = 0;

				m_buffer = Buffer();
			}
		}


		/**
		 *	@brief		Returns a non-owning 1D view of the entire array.
		 */
		BufferView<const Type> view() const { return m_buffer ? BufferView<const Type>(m_buffer) : BufferView<const Type>(); }


		/**
		 *	@brief		Returns a non-owning 1D view of the entire array.
		 */
		BufferView<Type> view() { return m_buffer ? BufferView<Type>(m_buffer) : BufferView<Type>(); }


		/**
		 *	@brief		Return constant version of device pointer.
		 *	@note		Provides an explicit method to get device pointer. 
		 */
		dev::Ptr<const Type> ptr() const { return *this; }


		/**
		 *	@brief		Returns device pointer.
		 *	@note		Provides an explicit method to get device pointer. 
		 */
		dev::Ptr<Type> ptr() { return *this; }

	private:

		Buffer		m_buffer;
	};
}
