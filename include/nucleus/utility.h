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

namespace NS_NAMESPACE
{
	/*****************************************************************************
	*******************************    Utility    ********************************
	*****************************************************************************/

	/**
	 *	@brief		Computes the ceiling of integer division (x/y)
	 *	@example	stream.launch(kernel, ns::ceil_div(count, 256), 256)(...);
	 *	@note		Safe for unsigned integer arithmetic.
	 */
	constexpr uint32_t ceil_div(size_t x, size_t y) { return static_cast<uint32_t>((x + y - 1) / y); }


	/**
	 *	@brief		Rounds up a value to the nearest multiple of the given alignment.
	 *	@example	size_t pitch = align_up(width, 16);
	 *	@note		Useful for memory alignment, thread/block sizing, or buffer strides.
	 *				Alignment must be a non-zero positive integer.
	 */
	constexpr size_t align_up(size_t x, size_t alignment) { return ceil_div(x, alignment) * alignment; }

	/*****************************************************************************
	***************************    BinaryCompatible    ***************************
	*****************************************************************************/

	//!	Utility functions to reinterpret buffer views as another compatible element type.
	template<typename DstType, typename SrcType> BufferView<DstType> view_cast(BufferView<SrcType> view);
	template<typename DstType, typename SrcType> BufferView2D<DstType> view_cast(BufferView2D<SrcType> view);
	template<typename DstType, typename SrcType> BufferView3D<DstType> view_cast(BufferView3D<SrcType> view);

	template<typename DstType, typename SrcType> BufferView<const DstType> view_cast(BufferView<const SrcType> view);
	template<typename DstType, typename SrcType> BufferView2D<const DstType> view_cast(BufferView2D<const SrcType> view);
	template<typename DstType, typename SrcType> BufferView3D<const DstType> view_cast(BufferView3D<const SrcType> view);

	//! Utility functions to reinterpret device pointers as another compatible element type.
	template<typename DstType, typename SrcType> NS_CUDA_CALLABLE dev::Ptr<const DstType> ptr_cast(dev::Ptr<const SrcType> ptr);
	template<typename DstType, typename SrcType> NS_CUDA_CALLABLE dev::Ptr2<const DstType> ptr_cast(dev::Ptr2<const SrcType> ptr);
	template<typename DstType, typename SrcType> NS_CUDA_CALLABLE dev::Ptr3<const DstType> ptr_cast(dev::Ptr3<const SrcType> ptr);

	template<typename DstType, typename SrcType> NS_CUDA_CALLABLE dev::Ptr<DstType> ptr_cast(dev::Ptr<SrcType> ptr);
	template<typename DstType, typename SrcType> NS_CUDA_CALLABLE dev::Ptr2<DstType> ptr_cast(dev::Ptr2<SrcType> ptr);
	template<typename DstType, typename SrcType> NS_CUDA_CALLABLE dev::Ptr3<DstType> ptr_cast(dev::Ptr3<SrcType> ptr);
}