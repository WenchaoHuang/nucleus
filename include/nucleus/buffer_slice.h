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
#include "device_span.h"
#include "device_pointer.h"

namespace NS_NAMESPACE
{
	/*****************************************************************************
	**************************    BufferSlice<Type>    ***************************
	*****************************************************************************/

	//!	@brief		A class representing a 1D slice of a buffer.
	template<typename Type> class BufferSlice : public dev::Span<Type>
	{
		using _Base = dev::Span<Type>;

	public:

		//!	@brief		Default constructor.
		BufferSlice() : _Base(nullptr, 0), m_offset(0) {}

		//!	@brief		Construct with nullptr.
		BufferSlice(std::nullptr_t) : _Base(nullptr, 0), m_offset(0) {}

		//!	@brief		Copy constructor, initializes from another `BufferSlice` of the same type.
		BufferSlice(const BufferSlice<std::remove_cv_t<Type>> & rhs) : _Base(rhs.data(), rhs.size()), m_buffer(rhs.buffer()), m_offset(rhs.offset()) {}

		//!	@brief		Construct with a given `Buffer`.
		explicit BufferSlice(Buffer buffer) : _Base(static_cast<Type*>(buffer.data()), buffer.capacity() / sizeof(Type)), m_buffer(std::move(buffer)), m_offset(0) { NS_ASSERT(m_buffer); }

		/**
		 *	@brief		Constructor to initialize with a buffer handle, offset and size.
		 *	@param[in]	buffer - Buffer handle.
		 *	@param[in]	offset - Byte offset within the buffer where the 1D slice starts.
		 *	@param[in]	size - Width of the 1D slice.
		 *	@warning	Ensures the buffer is not null, the address is correctly aligned, and the size fits within the buffer.
		 */
		explicit BufferSlice(Buffer buffer, size_t offset, size_t size) : _Base(reinterpret_cast<Type*>(buffer.address() + offset), size), m_buffer(std::move(buffer)), m_offset(offset)
		{
			NS_ASSERT(m_buffer);
			NS_ASSERT((m_buffer.address() + offset) % alignof(Type) == 0);
			NS_ASSERT((offset + sizeof(Type) * size) <= m_buffer.capacity());
		}

	public:

		/**
		 *	@brief		Returns a sub-slice starting at element \p start with \p size elements.
		 *	@param[in]	start - Element index of the sub-slice start (relative to this slice).
		 *	@param[in]	size - Number of elements in the sub-slice.
		 */
		BufferSlice subslice(size_t start, size_t size) const
		{
			NS_ASSERT(start + size <= this->size());

			return BufferSlice(m_buffer, m_offset + start * sizeof(Type), size);
		}

		//!	@brief		Returns constant device span to the underlying array, explicitly.
		const dev::Span<const Type> & span() const { return *this; }

		//!	@brief		Returns device span to the underlying array, explicitly.
		const dev::Span<Type> & span() { return *this; }

		//!	@brief		Returns the byte offset within the buffer.
		size_t offset() const { return m_offset; }

		//!	@brief		Returns the associated Buffer handle.
		Buffer buffer() const { return m_buffer; }

	private:

		size_t			m_offset;
		Buffer			m_buffer;
	};

	/*****************************************************************************
	*************************    BufferSlice2D<Type>    **************************
	*****************************************************************************/

	//!	@brief		A class representing a 2D slice of a buffer.
	template<typename Type> class BufferSlice2D : public dev::Ptr2<Type>
	{

	public:

		//!	@brief		Default constructor.
		BufferSlice2D() : dev::Ptr2<Type>(nullptr), m_offset(0) {}

		//!	@brief		Construct with nullptr.
		BufferSlice2D(std::nullptr_t) : dev::Ptr2<Type>(nullptr), m_offset(0) {}

		//!	@brief		Copy constructor, initializes from another 2D `BufferSlice` of the same type.
		BufferSlice2D(const BufferSlice2D<std::remove_cv_t<Type>> & rhs) : dev::Ptr2<Type>(rhs.data(), rhs.width(), rhs.height()), m_buffer(rhs.buffer()), m_offset(rhs.offset()) {}

		//!	@brief		Copy constructor, construct with a given 1D `BufferSlice`.
		explicit BufferSlice2D(const BufferSlice<std::remove_cv_t<Type>> & rhs) : dev::Ptr2<Type>(rhs.data(), rhs.width(), 1), m_buffer(rhs.buffer()), m_offset(rhs.offset()) {}

		/**
		 *	@brief		Constructor to initialize with a buffer handle, offset, width, and height.
		 *	@param[in]	buffer - Buffer handle.
		 *	@param[in]	offset - offset within the buffer where the 2D slice starts.
		 *	@param[in]	width - Width of the 2D slice.
		 *	@param[in]	height - Height of the 2D slice.
		 *	@warning	Ensures the buffer is not null, the address is correctly aligned, and the size fits within the buffer.
		 */
		explicit BufferSlice2D(Buffer buffer, size_t offset, uint32_t width, uint32_t height) : dev::Ptr2<Type>(reinterpret_cast<Type*>(buffer.address() + offset), width, height), m_buffer(std::move(buffer)), m_offset(offset)
		{
			NS_ASSERT(m_buffer);
			NS_ASSERT((m_buffer.address() + offset) % alignof(Type) == 0);
			NS_ASSERT((offset + sizeof(Type) * width * height) <= m_buffer.capacity());
		}

	public:

		/**
		 *	@brief		Returns a 2D sub-slice with new dimensions, starting at element (\p x, \p y).
		 *	@param[in]	x - Column start index (relative to this slice).
		 *	@param[in]	y - Row start index (relative to this slice).
		 *	@param[in]	width - Number of columns in the sub-slice.
		 *	@param[in]	height - Number of rows in the sub-slice.
		 */
		BufferSlice2D subslice(size_t x, size_t y, size_t width, size_t height) const
		{
			NS_ASSERT(x + width <= this->width() && y + height <= this->height());

			return BufferSlice2D(m_buffer, m_offset + (y * this->width() + x) * sizeof(Type), static_cast<uint32_t>(width), static_cast<uint32_t>(height));
		}

		//!	@brief		Returns device pointer to the underlying array, explicitly.
		const dev::Ptr2<Type> & ptr() const { return *this; }

		//!	@brief		Returns the byte offset within the buffer.
		size_t offset() const { return m_offset; }

		//!	@brief		Returns the associated Buffer handle.
		Buffer buffer() const { return m_buffer; }

	private:

		size_t			m_offset;
		Buffer			m_buffer;
	};

	/*****************************************************************************
	*************************    BufferSlice3D<Type>    **************************
	*****************************************************************************/

	//!	@brief		A class representing a 3D slice of a buffer.
	template<typename Type> class BufferSlice3D : public dev::Ptr3<Type>
	{

	public:

		//!	@brief		Default constructor.
		BufferSlice3D() : dev::Ptr3<Type>(nullptr), m_offset(0) {}

		//!	@brief		Construct with nullptr.
		BufferSlice3D(std::nullptr_t) : dev::Ptr3<Type>(nullptr), m_offset(0) {}

		//!	@brief		Copy constructor, initializes from another `BufferSlice` of the same type.
		BufferSlice3D(const BufferSlice3D<std::remove_cv_t<Type>> & rhs) : dev::Ptr3<Type>(rhs.data(), rhs.width(), rhs.height(), rhs.depth()), m_buffer(rhs.buffer()), m_offset(rhs.offset()) {}

		//!	@brief		Copy constructor, construct with a given 2D `BufferSlice`.
		explicit BufferSlice3D(const BufferSlice2D<std::remove_cv_t<Type>> & rhs) : dev::Ptr3<Type>(rhs.data(), rhs.width(), rhs.height(), 1), m_buffer(rhs.buffer()), m_offset(rhs.offset()) {}

		//!	@brief		Copy constructor, construct with a given 1D `BufferSlice`.
		explicit BufferSlice3D(const BufferSlice<std::remove_cv_t<Type>> & rhs) : dev::Ptr3<Type>(rhs.data(), rhs.width(), 1, 1), m_buffer(rhs.buffer()), m_offset(rhs.offset()) {}

		/**
		 *	@brief		Constructor to initialize with a buffer handle, offset, width, height and depth.
		 *	@param[in]	buffer - Buffer handle.
		 *	@param[in]	offset - offset within the buffer where the 3D slice starts.
		 *	@param[in]	width - Width of the 3D slice.
		 *	@param[in]	height - Height of the 3D slice.
		 *	@param[in]	depth - Depth of the 3D slice.
		 *	@warning	Ensures the buffer is not null, the address is correctly aligned, and the size fits within the buffer.
		 */
		explicit BufferSlice3D(Buffer buffer, size_t offset, uint32_t width, uint32_t height, uint32_t depth) : dev::Ptr3<Type>(reinterpret_cast<Type*>(buffer.address() + offset), width, height, depth), m_buffer(std::move(buffer)), m_offset(offset)
		{
			NS_ASSERT(m_buffer);
			NS_ASSERT((m_buffer.address() + offset) % alignof(Type) == 0);
			NS_ASSERT((offset + sizeof(Type) * width * height * depth) <= m_buffer.capacity());
		}

	public:

		/**
		 *	@brief		Returns a 3D sub-slice with new dimensions, starting at element (\p x, \p y, \p z).
		 *	@param[in]	x - Column start index (relative to this slice).
		 *	@param[in]	y - Row start index (relative to this slice).
		 *	@param[in]	z - Layer start index (relative to this slice).
		 *	@param[in]	width - Number of columns in the sub-slice.
		 *	@param[in]	height - Number of rows in the sub-slice.
		 *	@param[in]	depth - Number of layers in the sub-slice.
		 */
		BufferSlice3D subslice(size_t x, size_t y, size_t z, size_t width, size_t height, size_t depth) const
		{
			NS_ASSERT(x + width <= this->width() && y + height <= this->height() && z + depth <= this->depth());

			return BufferSlice<Type>(m_buffer, m_offset + (z * this->height() * this->width() + y * this->width() + x) * sizeof(Type), static_cast<uint32_t>(width), static_cast<uint32_t>(height), static_cast<uint32_t>(depth));
		}

		//!	@brief		Returns device pointer to the underlying array, explicitly.
		const dev::Ptr3<Type> & ptr() const { return *this; }

		//!	@brief		Returns the byte offset within the buffer.
		size_t offset() const { return m_offset; }

		//!	@brief		Returns the associated Buffer handle.
		Buffer buffer() const { return m_buffer; }

	private:

		size_t			m_offset;
		Buffer			m_buffer;
	};

	/*****************************************************************************
	**************************    slice_cast<T1, T2>    ***************************
	*****************************************************************************/

	//!	@brief		Reinterpret a 1D buffer slice as another element type, enforcing binary compatibility at compile time.
	template<typename DstType, typename SrcType> BufferSlice<DstType> slice_cast(BufferSlice<SrcType> slice)
	{
		static_assert(BinaryCompatible<DstType, SrcType>::value, "slice_cast requires DstType and SrcType to be binary compatible");

		return slice.empty() ? BufferSlice<DstType>() : BufferSlice<DstType>(slice.buffer(), slice.offset(), slice.size());
	}

	//!	@brief		Reinterpret a 2D buffer slice as another element type, enforcing binary compatibility at compile time.
	template<typename DstType, typename SrcType> BufferSlice2D<DstType> slice_cast(BufferSlice2D<SrcType> slice)
	{
		static_assert(BinaryCompatible<DstType, SrcType>::value, "slice_cast requires DstType and SrcType to be binary compatible");

		return slice.empty() ? BufferSlice2D<DstType>() : BufferSlice2D<DstType>(slice.buffer(), slice.offset(), slice.width(), slice.height());
	}

	//!	@brief		Reinterpret a 3D buffer slice as another element type, enforcing binary compatibility at compile time.
	template<typename DstType, typename SrcType> BufferSlice3D<DstType> slice_cast(BufferSlice3D<SrcType> slice)
	{
		static_assert(BinaryCompatible<DstType, SrcType>::value, "slice_cast requires DstType and SrcType to be binary compatible");

		return slice.empty() ? BufferSlice3D<DstType>() : BufferSlice3D<DstType>(slice.buffer(), slice.offset(), slice.width(), slice.height(), slice.depth());
	}

	//!	@brief		Reinterpret a 1D buffer slice as another element type, enforcing binary compatibility at compile time.
	template<typename DstType, typename SrcType> BufferSlice<const DstType> slice_cast(BufferSlice<const SrcType> slice)
	{
		static_assert(BinaryCompatible<DstType, SrcType>::value, "slice_cast requires DstType and SrcType to be binary compatible");

		return slice.empty() ? BufferSlice<const DstType>() : BufferSlice<const DstType>(slice.buffer(), slice.offset(), slice.width());
	}

	//!	@brief		Reinterpret a 2D buffer slice as another element type, enforcing binary compatibility at compile time.
	template<typename DstType, typename SrcType> BufferSlice2D<const DstType> slice_cast(BufferSlice2D<const SrcType> slice)
	{
		static_assert(BinaryCompatible<DstType, SrcType>::value, "slice_cast requires DstType and SrcType to be binary compatible");

		return slice.empty() ? BufferSlice2D<const DstType>() : BufferSlice2D<const DstType>(slice.buffer(), slice.offset(), slice.width(), slice.height());
	}

	//!	@brief		Reinterpret a 3D buffer slice as another element type, enforcing binary compatibility at compile time.
	template<typename DstType, typename SrcType> BufferSlice3D<const DstType> slice_cast(BufferSlice3D<const SrcType> slice)
	{
		static_assert(BinaryCompatible<DstType, SrcType>::value, "slice_cast requires DstType and SrcType to be binary compatible");

		return slice.empty() ? BufferSlice3D<const DstType>() : BufferSlice3D<const DstType>(slice.buffer(), slice.offset(), slice.width(), slice.height(), slice.depth());
	}
}
