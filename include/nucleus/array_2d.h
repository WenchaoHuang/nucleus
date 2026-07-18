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
	*******************************    Array2D    ********************************
	*****************************************************************************/

	/**
	 *	@brief		A 2D array template that provides device-accessible memory management.
	 */
	template<typename Type> class Array2D : public dev::Ptr2<Type>
	{
		NS_NONCOPYABLE(Array2D)

	public:

		//!	@brief		Construct an empty array.
		Array2D() noexcept : dev::Ptr2<Type>(nullptr), m_buffer() {}

		//!	@brief		Constructs and allocates a 2D array with specified dimensions.
		explicit Array2D(std::shared_ptr<Allocator> alloctor, size_t width, size_t height) : Array2D() { this->resize(alloctor, width, height); }

		//!	@brief		Move constructor. Transfers ownership from another array.
		Array2D(Array2D && rhs) : dev::Ptr2<Type>(std::exchange(rhs.m_data, nullptr), std::exchange(rhs.m_width, 0), std::exchange(rhs.m_height, 0)),  m_buffer(std::move(rhs.m_buffer)) {}

	public:

		/**
		 *	@brief		Resizes the array with new dimensions and allocator.
		 *	@param[in]	allocator - New memory allocator (must not be null).
		 *	@param[in]	width - New column count.
		 *	@param[in]	height - New row count.
		 *	@note		If the allocator or size changes, existing data will be lost.
		 */
		void resize(std::shared_ptr<Allocator> allocator, size_t width, size_t height)
		{
			NS_ASSERT_LOG_IF(allocator == nullptr, "Empty allocator!");

			if ((this->allocator() != allocator) || (this->size() != width * height))
			{
				m_buffer = Buffer(allocator, sizeof(Type) * width * height);

				dev::Ptr2<Type>::m_data = reinterpret_cast<Type*>(m_buffer.data());

				dev::Ptr2<Type>::m_height = static_cast<uint32_t>(height);

				dev::Ptr2<Type>::m_width = static_cast<uint32_t>(width);
			}
			else if ((this->width() != width) || (this->height() != height))
			{
				dev::Ptr2<Type>::m_height = static_cast<uint32_t>(height);

				dev::Ptr2<Type>::m_width = static_cast<uint32_t>(width);
			}
		}


		/**
		 *	@brief		Resizes the array maintaining current allocator.
		 *	@param[in]	width - New column count
		 *	@param[in]	height - New row count
		 *	@note		If the size changes, existing data will be lost.
		 */
		void resize(size_t width, size_t height)
		{
			NS_ASSERT_LOG_IF(!m_buffer, "Empty allocator!");

			this->resize(m_buffer.allocator(), width, height);
		}


		/**
		 *	@brief		Changes array dimensions without reallocation.
		 *	@param[in]	width - New column count
		 *	@param[in]	height - New row count
		 */
		void reshape(size_t width, size_t height)
		{
			NS_ASSERT_LOG_IF(this->size() != width * height, "Size mismatch!");

			if (this->size() == width * height)
			{
				dev::Ptr2<Type>::m_height = static_cast<uint32_t>(height);

				dev::Ptr2<Type>::m_width = static_cast<uint32_t>(width);
			}
		}


		/**
		 *	@brief		Gets the allocator associated with.
		 */
		std::shared_ptr<Allocator> allocator() const
		{
			return m_buffer.allocator();
		}


		/**
		 *	@brief		Releases the ownership of the internal buffer and returns it.
		 *	@note		After this call, the Array2D will be in an empty state.
		 */
		Buffer releaseBuffer()
		{
			dev::Ptr2<Type>::m_width = 0;

			dev::Ptr2<Type>::m_height = 0;

			dev::Ptr2<Type>::m_data = nullptr;

			return std::exchange(m_buffer, Buffer());
		}


		/**
		 *	@brief		Move assignment operator.
		 */
		void operator=(Array2D && rhs) noexcept
		{
			dev::Ptr2<Type>::m_data = std::exchange(rhs.m_data, nullptr);

			dev::Ptr2<Type>::m_height = std::exchange(rhs.m_height, 0);

			dev::Ptr2<Type>::m_width = std::exchange(rhs.m_width, 0);

			m_buffer = std::move(rhs.m_buffer);
		}


		/**
		 *	@brief		Swaps the contents.
		 */
		void swap(Array2D & rhs) noexcept
		{
			std::swap(dev::Ptr2<Type>::m_height, rhs.m_height);

			std::swap(dev::Ptr2<Type>::m_width, rhs.m_width);

			std::swap(dev::Ptr2<Type>::m_data, rhs.m_data);

			std::swap(m_buffer, rhs.m_buffer);
		}


		/**
		 *	@brief		Clears the array and releases all allocated memory.
		 */
		void clear() noexcept
		{
			if (m_buffer)
			{
				dev::Ptr2<Type>::m_data = nullptr;

				dev::Ptr2<Type>::m_height = 0;
				
				dev::Ptr2<Type>::m_width = 0;

				m_buffer = Buffer();
			}
		}


		/**
		 *	@brief		Returns a non-owning 2D view of the entire array.
		 */
		BufferView2D<const Type> view() const { return m_buffer ? BufferView2D<const Type>(m_buffer, 0, this->width(), this->height()) : BufferView2D<const Type>(); }


		/**
		 *	@brief		Returns a non-owning 2D view of the entire array.
		 */
		BufferView2D<Type> view() { return m_buffer ? BufferView2D<Type>(m_buffer, 0, this->width(), this->height()) : BufferView2D<Type>(); }


		/**
		 *	@brief		Return constant version of device pointer.
		 *	@note		Provides an explicit method to get device pointer.
		 */
		dev::Ptr2<const Type> ptr() const { return *this; }


		/**
		 *	@brief		Returns device pointer.
		 *	@note		Provides an explicit method to get device pointer.
		 */
		dev::Ptr2<Type> ptr() { return *this; }

	private:

		Buffer		m_buffer;
	};
}
