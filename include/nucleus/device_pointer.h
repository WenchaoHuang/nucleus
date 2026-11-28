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
		NS_CUDA_CALLABLE Ptr(std::nullptr_t) : data_(nullptr), width_(0) {}

		//!	@brief		Construct with pointer, width.
		NS_CUDA_CALLABLE Ptr(Type * ptr, size_t width = SIZE_MAX) : data_(ptr), width_(width) {}

		//!	@brief		Copy constructor.
		NS_CUDA_CALLABLE Ptr(const Ptr<std::remove_cv_t<Type>> &rhs) : data_(rhs.data()), width_(rhs.width()) {}

		//!	@brief		Copy constructor, construct with a given 2D device pointer.
		NS_CUDA_CALLABLE explicit Ptr(const Ptr2<std::remove_cv_t<Type>> &rhs) : data_(rhs.data()), width_(rhs.size()) {}

		//!	@brief		Copy constructor, construct with a given 3D device pointer.
		NS_CUDA_CALLABLE explicit Ptr(const Ptr3<std::remove_cv_t<Type>> &rhs) : data_(rhs.data()), width_(rhs.size()) {}

		//!	@brief		Returns size of the array in bytes.
		NS_CUDA_CALLABLE size_t bytes() const { return sizeof(Type) * width_; }

		//!	@brief		Return pitch the of array in bytes.
		NS_CUDA_CALLABLE size_t pitch() const { return sizeof(Type) * width_; }

		// Tests if the array is empty.
		NS_CUDA_CALLABLE bool empty() const { return data_ == nullptr; }

		//!	@brief		Treat as a raw pointer. e.g. if (x == nullptr). 
		NS_CUDA_CALLABLE operator Type*() const { return data_; }

		//!	@brief		Return width the of array.
		NS_CUDA_CALLABLE size_t width() const { return width_; }

		//!	@brief		Return size of the array.
		NS_CUDA_CALLABLE size_t size() const { return width_; }

		//!	@brief		Returns the raw pointer explicitly.
		NS_CUDA_CALLABLE Type * data() const { return data_; }

		//!	@brief		Get address to the first element at [i]-th row.
		NS_CUDA_CALLABLE Type & operator[](size_t i) const
		{
			NS_ASSERT((data_ != nullptr) && (i < width_));

		#ifdef NS_CUDA_MEMORY_CHECK

			if (data_ == nullptr)
			{
				printf("Invalid memory address (nullptr)!\ndevPtr<%s> with size of %lld, width = %lld.\n", TypeName<Type>(), sizeof(Type), width_);

			#ifdef __CUDA_ARCH__
				__trap();
			#endif
			}
			else if (i >= width_)
			{
				printf("Illegal memory access!\ndevPtr<%s> with size of %lld, width = %lld, index = %d.\n", TypeName<Type>(), sizeof(Type), width_, i);

			#ifdef __CUDA_ARCH__
				__trap();
			#endif
			}
			else if (std::is_const_v<Type> && Math::IsNaN(data_[i]))
			{
				printf("Reading a NaN!\ndevPtr<%s> with size of %lld, width = %lld, index = %d.\n", TypeName<Type>(), sizeof(Type), width_, i);

			#ifdef __CUDA_ARCH__
				__trap();
			#endif
			}
		#endif

			return data_[i];
		}

	protected:

		Type *		data_;
		size_t		width_;
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
		NS_CUDA_CALLABLE Ptr2(std::nullptr_t) : data_(nullptr), width_(0), height_(0) {}

		//!	@brief		Construct with pointer, width and height.
		NS_CUDA_CALLABLE Ptr2(Type * ptr, uint32_t width, uint32_t height) : data_(ptr), width_(width), height_(height) {}

		//!	@brief		Copy constructor.
		NS_CUDA_CALLABLE Ptr2(const Ptr2<std::remove_cv_t<Type>> &rhs) : data_(rhs.data()), width_(rhs.width()), height_(rhs.height()) {}

		//!	@brief		Copy constructor, construct with a given 3D device pointer.
		NS_CUDA_CALLABLE explicit Ptr2(const Ptr3<std::remove_cv_t<Type>> &rhs) : data_(rhs.data()), width_(rhs.width()), height_(rhs.height() * rhs.depth()) {}

		//!	@brief		Copy constructor, construct with a given 1D device pointer.
		NS_CUDA_CALLABLE explicit Ptr2(const Ptr<std::remove_cv_t<Type>> &rhs) : data_(rhs.data()), width_(rhs.width()), height_(1) {}

		//!	@brief		Returns size of the array in bytes.
		NS_CUDA_CALLABLE size_t bytes() const { return sizeof(Type) * width_ * height_; }

		//!	@brief		Returns pitch of the array in bytes.
		NS_CUDA_CALLABLE uint32_t pitch() const { return sizeof(Type) * width_; }

		//!	@brief		Returns element count.
		NS_CUDA_CALLABLE size_t size() const { return height_ * width_; }

		//!	@brief		Tests if the array is empty.
		NS_CUDA_CALLABLE bool empty() const { return data_ == nullptr; }

		//!	@brief		Returns height of the array.
		NS_CUDA_CALLABLE uint32_t height() const { return height_; }

		//!	@brief		Returns width the of array.
		NS_CUDA_CALLABLE uint32_t width() const { return width_; }

		//!	@brief		Treat as a raw pointer. e.g. if (x == nullptr). 
		NS_CUDA_CALLABLE operator Type*() const { return data_; }

		//!	@brief		Returns the raw pointer explicitly.
		NS_CUDA_CALLABLE Type * data() const { return data_; }

		//!	@brief		Get address to the first element at [i]-th row.
		NS_CUDA_CALLABLE Ptr<Type> operator[](size_t i) const
		{
			NS_ASSERT((data_ != nullptr) && (i < height_));

		#ifdef NS_CUDA_MEMORY_CHECK

			if (data_ == nullptr)
			{
				printf("Invalid memory address (nullptr)!\ndevPtr2<%s> with size of %lld, width = %d, height = %d.\n", TypeName<Type>(), sizeof(Type), width_, height_);

			#ifdef __CUDA_ARCH__
				__trap();
			#endif
			}
			else if (i >= height_)
			{
				printf("Illegal memory access!\ndevPtr2<%s> with size of %lld, width = %d, height = %d, index = %d.\n", TypeName<Type>(), sizeof(Type), width_, height_, i);

			#ifdef __CUDA_ARCH__
				__trap();
			#endif
			}
		#endif

			return Ptr<Type>(data_ + i * width_, width_);
		}

	protected:

		Type *			data_;
		uint32_t		width_;
		uint32_t		height_;
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
		NS_CUDA_CALLABLE Ptr3(std::nullptr_t) : data_(nullptr), width_(0), height_(0), depth_(0) {}

		//!	@brief		Construct with pointer, width and height.
		NS_CUDA_CALLABLE Ptr3(Type * ptr, uint32_t width, uint32_t height, uint32_t depth) : data_(ptr), width_(width), height_(height), depth_(depth) {}

		//!	@brief		Copy constructor.
		NS_CUDA_CALLABLE Ptr3(const Ptr3<std::remove_cv_t<Type>> & rhs) : data_(rhs.data()), width_(rhs.width()), height_(rhs.height()), depth_(rhs.depth()) {}

		//!	@brief		Copy constructor, construct with a given 2D device pointer.
		NS_CUDA_CALLABLE explicit Ptr3(const Ptr2<std::remove_cv_t<Type>> & rhs) : data_(rhs.data()), width_(rhs.width()), height_(rhs.height()), depth_(1) {}

		//!	@brief		Copy constructor, construct with a given 1D device pointer.
		NS_CUDA_CALLABLE explicit Ptr3(const Ptr<std::remove_cv_t<Type>> & rhs) : data_(rhs.data()), width_(rhs.width()), height_(1), depth_(1) {}

		//!	@brief		Returns size of the array in bytes.
		NS_CUDA_CALLABLE size_t bytes() const { return sizeof(Type) * width_ * height_ * depth_; }

		//!	@brief		Returns element count of the array.
		NS_CUDA_CALLABLE size_t size() const { return width_ * height_ * depth_; }

		//!	@brief		Returns pitch of the array in bytes.
		NS_CUDA_CALLABLE uint32_t pitch() const { return sizeof(Type) * width_; }

		// Tests if the array is empty.
		NS_CUDA_CALLABLE bool empty() const { return data_ == nullptr; }

		//!	@brief		Returns height of the array.
		NS_CUDA_CALLABLE uint32_t height() const { return height_; }

		//!	@brief		Returns width the of array.
		NS_CUDA_CALLABLE uint32_t width() const { return width_; }

		//!	@brief		Returns depth the of array.
		NS_CUDA_CALLABLE uint32_t depth() const { return depth_; }

		//!	@brief		Treat as a raw pointer. e.g. if (x == nullptr). 
		NS_CUDA_CALLABLE operator Type*() const { return data_; }

		//!	@brief		Returns the raw pointer explicitly.
		NS_CUDA_CALLABLE Type * data() const { return data_; }

		//!	@brief		Get address to the first element at [i]-th row.
		NS_CUDA_CALLABLE Ptr2<Type> operator[](size_t i) const
		{
			NS_ASSERT((data_ != nullptr) && (i < depth_));

		#ifdef NS_CUDA_MEMORY_CHECK

			if (data_ == nullptr)
			{
				printf("Invalid memory address (nullptr)!\ndevPtr3<%s> with size of %lld, width = %d, height = %d, depth = %d.\n", TypeName<Type>(), sizeof(Type), width_, height_, depth_);

			#ifdef __CUDA_ARCH__
				__trap();
			#endif
			}
			else if (i >= depth_)
			{
				printf("Illegal memory access!\ndevPtr3<%s> with size of %lld, width = %d, height = %d, depth = %d, index = %d.\n", TypeName<Type>(), sizeof(Type), width_, height_, depth_, i);

			#ifdef __CUDA_ARCH__
				__trap();
			#endif
			}
		#endif

			return Ptr2<Type>(data_ + i * (width_ * height_), width_, height_);
		}

	protected:

		Type *			data_;
		uint32_t		width_;
		uint32_t		height_;
		uint32_t		depth_;
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
