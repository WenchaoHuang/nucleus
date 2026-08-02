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

namespace NS_NAMESPACE::dev
{
	/*****************************************************************************
	********************************    Ptr<T>    ********************************
	*****************************************************************************/

	//!	@brief		Template wrapper for 1D device pointer.
	template<typename Type> struct Ptr
	{
		//!	@brief		Default constructor, should not initialized.
		NS_CUDA_CALLABLE Ptr() noexcept {}

		//!	@brief		Constructor with zero initialization.
		NS_CUDA_CALLABLE Ptr(std::nullptr_t) : m_data(nullptr), m_width(0) {}

		//!	@brief		Construct with pointer, width.
		NS_CUDA_CALLABLE Ptr(Type * ptr, size_t width = SIZE_MAX) : m_data(ptr), m_width(width) {}

		//!	@brief		Copy constructor.
		NS_CUDA_CALLABLE Ptr(const Ptr<std::remove_cv_t<Type>> &rhs) : m_data(rhs.data()), m_width(rhs.width()) {}

		//!	@brief		Copy constructor, construct with a given 2D device pointer.
		NS_CUDA_CALLABLE explicit Ptr(const Ptr2<std::remove_cv_t<Type>> &rhs) : m_data(rhs.data()), m_width(rhs.size()) {}

		//!	@brief		Copy constructor, construct with a given 3D device pointer.
		NS_CUDA_CALLABLE explicit Ptr(const Ptr3<std::remove_cv_t<Type>> &rhs) : m_data(rhs.data()), m_width(rhs.size()) {}

		//!	@brief		Returns size of the array in bytes.
		NS_CUDA_CALLABLE size_t bytes() const { return sizeof(Type) * m_width; }

		//!	@brief		Return pitch the of array in bytes.
		NS_CUDA_CALLABLE size_t pitch() const { return sizeof(Type) * m_width; }

		// Tests if the array is empty.
		NS_CUDA_CALLABLE bool empty() const { return m_data == nullptr; }

		//!	@brief		Treat as a raw pointer. e.g. if (pX == nullptr). 
		NS_CUDA_CALLABLE operator Type*() const { return m_data; }

		//!	@brief		Return width the of array.
		NS_CUDA_CALLABLE size_t width() const { return m_width; }

		//!	@brief		Return size of the array.
		NS_CUDA_CALLABLE size_t size() const { return m_width; }

		//!	@brief		Returns the raw pointer explicitly.
		NS_CUDA_CALLABLE Type * data() const { return m_data; }

		//!	@brief		Get address to the first element at [i]-th row.
		NS_CUDA_CALLABLE Type & operator[](size_t i) const
		{
			NS_ASSERT((m_data != nullptr) && (i < m_width));

		#ifdef NS_CUDA_MEMORY_CHECK

			if (m_data == nullptr)
			{
				printf("Invalid memory address (nullptr)!\ndevPtr<%s> with size of %lld, width = %lld.\n", TypeName<Type>(), sizeof(Type), m_width);

			#ifdef __CUDA_ARCH__
				__trap();
			#endif
			}
			else if (i >= m_width)
			{
				printf("Illegal memory access!\ndevPtr<%s> with size of %lld, width = %lld, index = %d.\n", TypeName<Type>(), sizeof(Type), m_width, i);

			#ifdef __CUDA_ARCH__
				__trap();
			#endif
			}
			else if (std::is_const_v<Type> && Math::IsNaN(m_data[i]))
			{
				printf("Reading a NaN!\ndevPtr<%s> with size of %lld, width = %lld, index = %d.\n", TypeName<Type>(), sizeof(Type), m_width, i);

			#ifdef __CUDA_ARCH__
				__trap();
			#endif
			}
		#endif

			return m_data[i];
		}

	protected:

		Type *		m_data;
		size_t		m_width;
	};

	/*****************************************************************************
	*******************************    Ptr2<T>    ********************************
	*****************************************************************************/

	//!	@brief		Template wrapper for 2D device pointer.
	template<typename Type> struct Ptr2
	{
		//!	@brief		Default constructor, should not initialized.
		NS_CUDA_CALLABLE Ptr2() noexcept {}

		//!	@brief		Constructor with zero initialization.
		NS_CUDA_CALLABLE Ptr2(std::nullptr_t) : m_data(nullptr), m_width(0), m_height(0) {}

		//!	@brief		Construct with pointer, width and height.
		NS_CUDA_CALLABLE Ptr2(Type * ptr, uint32_t width, uint32_t height) : m_data(ptr), m_width(width), m_height(height) {}

		//!	@brief		Copy constructor.
		NS_CUDA_CALLABLE Ptr2(const Ptr2<std::remove_cv_t<Type>> &rhs) : m_data(rhs.data()), m_width(rhs.width()), m_height(rhs.height()) {}

		//!	@brief		Copy constructor, construct with a given 3D device pointer.
		NS_CUDA_CALLABLE explicit Ptr2(const Ptr3<std::remove_cv_t<Type>> &rhs) : m_data(rhs.data()), m_width(rhs.width()), m_height(rhs.height() * rhs.depth()) {}

		//!	@brief		Copy constructor, construct with a given 1D device pointer.
		NS_CUDA_CALLABLE explicit Ptr2(const Ptr<std::remove_cv_t<Type>> &rhs) : m_data(rhs.data()), m_width(rhs.width()), m_height(1) {}

		//!	@brief		Returns size of the array in bytes.
		NS_CUDA_CALLABLE size_t bytes() const { return sizeof(Type) * m_width * m_height; }

		//!	@brief		Returns pitch of the array in bytes.
		NS_CUDA_CALLABLE uint32_t pitch() const { return sizeof(Type) * m_width; }

		//!	@brief		Returns element count.
		NS_CUDA_CALLABLE size_t size() const { return m_height * m_width; }

		//!	@brief		Tests if the array is empty.
		NS_CUDA_CALLABLE bool empty() const { return m_data == nullptr; }

		//!	@brief		Returns height of the array.
		NS_CUDA_CALLABLE uint32_t height() const { return m_height; }

		//!	@brief		Returns width the of array.
		NS_CUDA_CALLABLE uint32_t width() const { return m_width; }

		//!	@brief		Treat as a raw pointer. e.g. if (pX == nullptr). 
		NS_CUDA_CALLABLE operator Type*() const { return m_data; }

		//!	@brief		Returns the raw pointer explicitly.
		NS_CUDA_CALLABLE Type * data() const { return m_data; }

		//!	@brief		Get address to the first element at [i]-th row.
		NS_CUDA_CALLABLE Ptr<Type> operator[](size_t i) const
		{
			NS_ASSERT((m_data != nullptr) && (i < m_height));

		#ifdef NS_CUDA_MEMORY_CHECK

			if (m_data == nullptr)
			{
				printf("Invalid memory address (nullptr)!\ndevPtr2<%s> with size of %lld, width = %d, height = %d.\n", TypeName<Type>(), sizeof(Type), m_width, m_height);

			#ifdef __CUDA_ARCH__
				__trap();
			#endif
			}
			else if (i >= m_height)
			{
				printf("Illegal memory access!\ndevPtr2<%s> with size of %lld, width = %d, height = %d, index = %d.\n", TypeName<Type>(), sizeof(Type), m_width, m_height, i);

			#ifdef __CUDA_ARCH__
				__trap();
			#endif
			}
		#endif

			return Ptr<Type>(m_data + i * m_width, m_width);
		}

	protected:

		Type *			m_data;
		uint32_t		m_width;
		uint32_t		m_height;
	};

	/*****************************************************************************
	*******************************    Ptr3<T>    ********************************
	*****************************************************************************/

	//!	@brief		Template wrapper for 3D device pointer.
	template<typename Type> struct Ptr3
	{
		//!	@brief		Default constructor, should not initialized.
		NS_CUDA_CALLABLE Ptr3() noexcept {}

		//!	@brief		Constructor with zero initialization.
		NS_CUDA_CALLABLE Ptr3(std::nullptr_t) : m_data(nullptr), m_width(0), m_height(0), m_depth(0) {}

		//!	@brief		Construct with pointer, width and height.
		NS_CUDA_CALLABLE Ptr3(Type * ptr, uint32_t width, uint32_t height, uint32_t depth) : m_data(ptr), m_width(width), m_height(height), m_depth(depth) {}

		//!	@brief		Copy constructor.
		NS_CUDA_CALLABLE Ptr3(const Ptr3<std::remove_cv_t<Type>> & rhs) : m_data(rhs.data()), m_width(rhs.width()), m_height(rhs.height()), m_depth(rhs.depth()) {}

		//!	@brief		Copy constructor, construct with a given 2D device pointer.
		NS_CUDA_CALLABLE explicit Ptr3(const Ptr2<std::remove_cv_t<Type>> & rhs) : m_data(rhs.data()), m_width(rhs.width()), m_height(rhs.height()), m_depth(1) {}

		//!	@brief		Copy constructor, construct with a given 1D device pointer.
		NS_CUDA_CALLABLE explicit Ptr3(const Ptr<std::remove_cv_t<Type>> & rhs) : m_data(rhs.data()), m_width(rhs.width()), m_height(1), m_depth(1) {}

		//!	@brief		Returns size of the array in bytes.
		NS_CUDA_CALLABLE size_t bytes() const { return sizeof(Type) * m_width * m_height * m_depth; }

		//!	@brief		Returns element count of the array.
		NS_CUDA_CALLABLE size_t size() const { return m_width * m_height * m_depth; }

		//!	@brief		Returns pitch of the array in bytes.
		NS_CUDA_CALLABLE uint32_t pitch() const { return sizeof(Type) * m_width; }

		// Tests if the array is empty.
		NS_CUDA_CALLABLE bool empty() const { return m_data == nullptr; }

		//!	@brief		Returns height of the array.
		NS_CUDA_CALLABLE uint32_t height() const { return m_height; }

		//!	@brief		Returns width the of array.
		NS_CUDA_CALLABLE uint32_t width() const { return m_width; }

		//!	@brief		Returns depth the of array.
		NS_CUDA_CALLABLE uint32_t depth() const { return m_depth; }

		//!	@brief		Treat as a raw pointer. e.g. if (pX == nullptr). 
		NS_CUDA_CALLABLE operator Type*() const { return m_data; }

		//!	@brief		Returns the raw pointer explicitly.
		NS_CUDA_CALLABLE Type * data() const { return m_data; }

		//!	@brief		Get address to the first element at [i]-th row.
		NS_CUDA_CALLABLE Ptr2<Type> operator[](size_t i) const
		{
			NS_ASSERT((m_data != nullptr) && (i < m_depth));

		#ifdef NS_CUDA_MEMORY_CHECK

			if (m_data == nullptr)
			{
				printf("Invalid memory address (nullptr)!\ndevPtr3<%s> with size of %lld, width = %d, height = %d, depth = %d.\n", TypeName<Type>(), sizeof(Type), m_width, m_height, m_depth);

			#ifdef __CUDA_ARCH__
				__trap();
			#endif
			}
			else if (i >= m_depth)
			{
				printf("Illegal memory access!\ndevPtr3<%s> with size of %lld, width = %d, height = %d, depth = %d, index = %d.\n", TypeName<Type>(), sizeof(Type), m_width, m_height, m_depth, i);

			#ifdef __CUDA_ARCH__
				__trap();
			#endif
			}
		#endif

			return Ptr2<Type>(m_data + i * (m_width * m_height), m_width, m_height);
		}

	protected:

		Type *			m_data;
		uint32_t		m_width;
		uint32_t		m_height;
		uint32_t		m_depth;
	};
}

namespace NS_NAMESPACE
{
	/*****************************************************************************
	***************************    ptr_cast<T1, T2>    ***************************
	*****************************************************************************/

	//!	@brief		Reinterpret a 1D device pointer as another element type, enforcing binary compatibility at compile time.
	template<typename DstType, typename SrcType> NS_CUDA_CALLABLE dev::Ptr<DstType> ptr_cast(dev::Ptr<SrcType> ptr)
	{
		static_assert(BinaryCompatible<DstType, SrcType>::value, "ptr_cast requires DstType and SrcType to be binary compatible");

		return dev::Ptr<DstType>(reinterpret_cast<DstType*>(ptr.data()), ptr.width());
	}

	//!	@brief		Reinterpret a 2D device pointer as another element type, enforcing binary compatibility at compile time.
	template<typename DstType, typename SrcType> NS_CUDA_CALLABLE dev::Ptr2<DstType> ptr_cast(dev::Ptr2<SrcType> ptr)
	{
		static_assert(BinaryCompatible<DstType, SrcType>::value, "ptr_cast requires DstType and SrcType to be binary compatible");

		return dev::Ptr2<DstType>(reinterpret_cast<DstType*>(ptr.data()), ptr.width(), ptr.height());
	}

	//!	@brief		Reinterpret a 3D device pointer as another element type, enforcing binary compatibility at compile time.
	template<typename DstType, typename SrcType> NS_CUDA_CALLABLE dev::Ptr3<DstType> ptr_cast(dev::Ptr3<SrcType> ptr)
	{
		static_assert(BinaryCompatible<DstType, SrcType>::value, "ptr_cast requires DstType and SrcType to be binary compatible");

		return dev::Ptr3<DstType>(reinterpret_cast<DstType*>(ptr.data()), ptr.width(), ptr.height(), ptr.depth());
	}

	//!	@brief		Reinterpret a 1D device pointer as another element type, enforcing binary compatibility at compile time.
	template<typename DstType, typename SrcType> NS_CUDA_CALLABLE dev::Ptr<const DstType> ptr_cast(dev::Ptr<const SrcType> ptr)
	{
		static_assert(BinaryCompatible<DstType, SrcType>::value, "ptr_cast requires DstType and SrcType to be binary compatible");

		return dev::Ptr<const DstType>(reinterpret_cast<const DstType*>(ptr.data()), ptr.width());
	}

	//!	@brief		Reinterpret a 2D device pointer as another element type, enforcing binary compatibility at compile time.
	template<typename DstType, typename SrcType> NS_CUDA_CALLABLE dev::Ptr2<const DstType> ptr_cast(dev::Ptr2<const SrcType> ptr)
	{
		static_assert(BinaryCompatible<DstType, SrcType>::value, "ptr_cast requires DstType and SrcType to be binary compatible");

		return dev::Ptr2<const DstType>(reinterpret_cast<const DstType*>(ptr.data()), ptr.width(), ptr.height());
	}

	//!	@brief		Reinterpret a 3D device pointer as another element type, enforcing binary compatibility at compile time.
	template<typename DstType, typename SrcType> NS_CUDA_CALLABLE dev::Ptr3<const DstType> ptr_cast(dev::Ptr3<const SrcType> ptr)
	{
		static_assert(BinaryCompatible<DstType, SrcType>::value, "ptr_cast requires DstType and SrcType to be binary compatible");

		return dev::Ptr3<const DstType>(reinterpret_cast<const DstType*>(ptr.data()), ptr.width(), ptr.height(), ptr.depth());
	}
}
