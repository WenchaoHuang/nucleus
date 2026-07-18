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
#include "device_pointer.h"

namespace NS_NAMESPACE
{
	/*****************************************************************************
	****************************    BufferView<T>    *****************************
	*****************************************************************************/

	//	A class representing a 1D view of a buffer.
	template<typename Type> class BufferView : public dev::Ptr<Type>
	{

	public:

		//	Default constructor.
		BufferView() : dev::Ptr<Type>(nullptr), m_offset(0) {}

		//	Construct with nullptr.
		BufferView(std::nullptr_t) : dev::Ptr<Type>(nullptr), m_offset(0) {}

		//	Copy constructor, initializes from another BufferView of the same type.
		BufferView(const BufferView<std::remove_cv_t<Type>> & rhs) : dev::Ptr<Type>(rhs.data(), rhs.width()), m_buffer(rhs.buffer()), m_offset(rhs.offset()) {}

		//	Construct with a given Buffer.
		explicit BufferView(Buffer buffer) : dev::Ptr<Type>(static_cast<Type*>(buffer.data()), buffer.capacity() / sizeof(Type)), m_buffer(std::move(buffer)), m_offset(0) { NS_ASSERT(m_buffer); }

		/**
		 *	@brief		Constructor to initialize with a buffer handle, offset and width.
		 *	@param[in]	buffer - Buffer handle.
		 *	@param[in]	offset - Byte offset within the buffer where the 1D view starts.
		 *	@param[in]	width - Width of the 1D view.
		 *	@warning	Ensures the buffer is not null, the address is correctly aligned, and the size fits within the buffer.
		 */
		explicit BufferView(Buffer buffer, size_t offset, size_t width) : dev::Ptr<Type>(reinterpret_cast<Type*>(buffer.address() + offset), width), m_buffer(std::move(buffer)), m_offset(offset)
		{
			NS_ASSERT(m_buffer);
			NS_ASSERT((m_buffer.address() + offset) % alignof(Type) == 0);
			NS_ASSERT((offset + sizeof(Type) * width) <= m_buffer.capacity());
		}

	public:

		//	Returns the associated Buffer handle.
		Buffer buffer() const { return m_buffer; }

		//	Returns device pointer to the underlying array, explicitly.
		const dev::Ptr<Type> & ptr() const { return *this; }

		//	Returns the byte offset within the buffer.
		size_t offset() const { return m_offset; }

		/**
		 *	@brief		Returns a sub-view starting at element \p start with \p width elements.
		 *	@param[in]	start - Element index of the sub-view start (relative to this view).
		 *	@param[in]	width - Number of elements in the sub-view.
		 */
		BufferView<Type> subview(size_t start, size_t width) const
		{
			NS_ASSERT(start + width <= this->width());

			return BufferView<Type>(m_buffer, m_offset + start * sizeof(Type), width);
		}

	private:

		size_t						m_offset;
		Buffer						m_buffer;
	};

	/*****************************************************************************
	***************************    BufferView2D<T>    ****************************
	*****************************************************************************/

	//	A class representing a 2D view of a buffer.
	template<typename Type> class BufferView2D : public dev::Ptr2<Type>
	{

	public:

		//	Default constructor.
		BufferView2D() : dev::Ptr2<Type>(nullptr), m_offset(0) {}

		//	Construct with nullptr.
		BufferView2D(std::nullptr_t) : dev::Ptr2<Type>(nullptr), m_offset(0) {}

		//	Copy constructor, initializes from another BufferView2D of the same type.
		BufferView2D(const BufferView2D<std::remove_cv_t<Type>> & rhs) : dev::Ptr2<Type>(rhs.data(), rhs.width(), rhs.height()), m_buffer(rhs.buffer()), m_offset(rhs.offset()) {}

		//	Copy constructor, construct with a given BufferView.
		explicit BufferView2D(const BufferView<std::remove_cv_t<Type>> & rhs) : dev::Ptr2<Type>(rhs.data(), rhs.width(), 1), m_buffer(rhs.buffer()), m_offset(rhs.offset()) {}

		/**
		 *	@brief		Constructor to initialize with a buffer handle, offset, width, and height.
		 *	@param[in]	buffer - Buffer handle.
		 *	@param[in]	offset - offset within the buffer where the 2D view starts.
		 *	@param[in]	width - Width of the 2D view.
		 *	@param[in]	height - Height of the 2D view.
		 *	@warning	Ensures the buffer is not null, the address is correctly aligned, and the size fits within the buffer.
		 */
		explicit BufferView2D(Buffer buffer, size_t offset, uint32_t width, uint32_t height) : dev::Ptr2<Type>(reinterpret_cast<Type*>(buffer.address() + offset), width, height), m_buffer(std::move(buffer)), m_offset(offset)
		{
			NS_ASSERT(m_buffer);
			NS_ASSERT((m_buffer.address() + offset) % alignof(Type) == 0);
			NS_ASSERT((offset + sizeof(Type) * width * height) <= m_buffer.capacity());
		}

	public:

		//	Returns the associated Buffer handle.
		Buffer buffer() const { return m_buffer; }

		//	Returns device pointer to the underlying array, explicitly.
		const dev::Ptr2<Type> & ptr() const { return *this; }

		//	Returns the byte offset within the buffer.
		size_t offset() const { return m_offset; }

		/**
		 *	@brief		Returns a 2D sub-view with new dimensions, starting at element (\p x, \p y).
		 *	@param[in]	x - Column start index (relative to this view).
		 *	@param[in]	y - Row start index (relative to this view).
		 *	@param[in]	width - Number of columns in the sub-view.
		 *	@param[in]	height - Number of rows in the sub-view.
		 */
		BufferView2D<Type> subview(size_t x, size_t y, size_t width, size_t height) const
		{
			NS_ASSERT(x + width <= this->width() && y + height <= this->height());

			return BufferView2D<Type>(m_buffer, m_offset + (y * this->width() + x) * sizeof(Type), static_cast<uint32_t>(width), static_cast<uint32_t>(height));
		}

	private:

		size_t						m_offset;
		Buffer						m_buffer;
	};

	/*****************************************************************************
	***************************    BufferView3D<T>    ****************************
	*****************************************************************************/

	//	A class representing a 3D view of a buffer.
	template<typename Type> class BufferView3D : public dev::Ptr3<Type>
	{

	public:

		//	Default constructor.
		BufferView3D() : dev::Ptr3<Type>(nullptr), m_offset(0) {}

		//	Construct with nullptr.
		BufferView3D(std::nullptr_t) : dev::Ptr3<Type>(nullptr), m_offset(0) {}

		//	Copy constructor, initializes from another BufferView3D of the same type.
		BufferView3D(const BufferView3D<std::remove_cv_t<Type>> & rhs) : dev::Ptr3<Type>(rhs.data(), rhs.width(), rhs.height(), rhs.depth()), m_buffer(rhs.buffer()), m_offset(rhs.offset()) {}

		//	Copy constructor, construct with a given BufferView2D.
		explicit BufferView3D(const BufferView2D<std::remove_cv_t<Type>> & rhs) : dev::Ptr3<Type>(rhs.data(), rhs.width(), rhs.height(), 1), m_buffer(rhs.buffer()), m_offset(rhs.offset()) {}

		//	Copy constructor, construct with a given BufferView.
		explicit BufferView3D(const BufferView<std::remove_cv_t<Type>> & rhs) : dev::Ptr3<Type>(rhs.data(), rhs.width(), 1, 1), m_buffer(rhs.buffer()), m_offset(rhs.offset()) {}

		/**
		 *	@brief		Constructor to initialize with a buffer handle, offset, width, height and depth.
		 *	@param[in]	buffer - Buffer handle.
		 *	@param[in]	offset - offset within the buffer where the 3D view starts.
		 *	@param[in]	width - Width of the 3D view.
		 *	@param[in]	height - Height of the 3D view.
		 *	@param[in]	depth - Depth of the 3D view.
		 *	@warning	Ensures the buffer is not null, the address is correctly aligned, and the size fits within the buffer.
		 */
		explicit BufferView3D(Buffer buffer, size_t offset, uint32_t width, uint32_t height, uint32_t depth) : dev::Ptr3<Type>(reinterpret_cast<Type*>(buffer.address() + offset), width, height, depth), m_buffer(std::move(buffer)), m_offset(offset)
		{
			NS_ASSERT(m_buffer);
			NS_ASSERT((m_buffer.address() + offset) % alignof(Type) == 0);
			NS_ASSERT((offset + sizeof(Type) * width * height * depth) <= m_buffer.capacity());
		}

	public:

		//	Returns the associated Buffer handle.
		Buffer buffer() const { return m_buffer; }

		//	Returns device pointer to the underlying array, explicitly.
		const dev::Ptr3<Type> & ptr() const { return *this; }

		//	Returns the byte offset within the buffer.
		size_t offset() const { return m_offset; }

		/**
		 *	@brief		Returns a 3D sub-view with new dimensions, starting at element (\p x, \p y, \p z).
		 *	@param[in]	x - Column start index (relative to this view).
		 *	@param[in]	y - Row start index (relative to this view).
		 *	@param[in]	z - Layer start index (relative to this view).
		 *	@param[in]	width - Number of columns in the sub-view.
		 *	@param[in]	height - Number of rows in the sub-view.
		 *	@param[in]	depth - Number of layers in the sub-view.
		 */
		BufferView3D<Type> subview(size_t x, size_t y, size_t z, size_t width, size_t height, size_t depth) const
		{
			NS_ASSERT(x + width <= this->width() && y + height <= this->height() && z + depth <= this->depth());

			return BufferView3D<Type>(m_buffer, m_offset + (z * this->height() * this->width() + y * this->width() + x) * sizeof(Type), static_cast<uint32_t>(width), static_cast<uint32_t>(height), static_cast<uint32_t>(depth));
		}

	private:

		size_t						m_offset;
		Buffer						m_buffer;
	};

	/*****************************************************************************
	**************************    view_cast<T1, T2>    ***************************
	*****************************************************************************/

	//	Reinterpret a 1D buffer view as another element type, enforcing binary compatibility at compile time.
	template<typename DstType, typename SrcType> BufferView<DstType> view_cast(BufferView<SrcType> view)
	{
		static_assert(BinaryCompatible<DstType, SrcType>::value, "view_cast requires DstType and SrcType to be binary compatible");

		return view.empty() ? BufferView<DstType>() : BufferView<DstType>(view.buffer(), view.offset(), view.width());
	}

	//	Reinterpret a 2D buffer view as another element type, enforcing binary compatibility at compile time.
	template<typename DstType, typename SrcType> BufferView2D<DstType> view_cast(BufferView2D<SrcType> view)
	{
		static_assert(BinaryCompatible<DstType, SrcType>::value, "view_cast requires DstType and SrcType to be binary compatible");

		return view.empty() ? BufferView2D<DstType>() : BufferView2D<DstType>(view.buffer(), view.offset(), view.width(), view.height());
	}

	//	Reinterpret a 3D buffer view as another element type, enforcing binary compatibility at compile time.
	template<typename DstType, typename SrcType> BufferView3D<DstType> view_cast(BufferView3D<SrcType> view)
	{
		static_assert(BinaryCompatible<DstType, SrcType>::value, "view_cast requires DstType and SrcType to be binary compatible");

		return view.empty() ? BufferView3D<DstType>() : BufferView3D<DstType>(view.buffer(), view.offset(), view.width(), view.height(), view.depth());
	}

	//	Reinterpret a 1D buffer view as another element type, enforcing binary compatibility at compile time.
	template<typename DstType, typename SrcType> BufferView<const DstType> view_cast(BufferView<const SrcType> view)
	{
		static_assert(BinaryCompatible<DstType, SrcType>::value, "view_cast requires DstType and SrcType to be binary compatible");

		return view.empty() ? BufferView<const DstType>() : BufferView<const DstType>(view.buffer(), view.offset(), view.width());
	}

	//	Reinterpret a 2D buffer view as another element type, enforcing binary compatibility at compile time.
	template<typename DstType, typename SrcType> BufferView2D<const DstType> view_cast(BufferView2D<const SrcType> view)
	{
		static_assert(BinaryCompatible<DstType, SrcType>::value, "view_cast requires DstType and SrcType to be binary compatible");

		return view.empty() ? BufferView2D<const DstType>() : BufferView2D<const DstType>(view.buffer(), view.offset(), view.width(), view.height());
	}

	//	Reinterpret a 3D buffer view as another element type, enforcing binary compatibility at compile time.
	template<typename DstType, typename SrcType> BufferView3D<const DstType> view_cast(BufferView3D<const SrcType> view)
	{
		static_assert(BinaryCompatible<DstType, SrcType>::value, "view_cast requires DstType and SrcType to be binary compatible");

		return view.empty() ? BufferView3D<const DstType>() : BufferView3D<const DstType>(view.buffer(), view.offset(), view.width(), view.height(), view.depth());
	}
}
