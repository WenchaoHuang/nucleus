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
#include "vector_types.h"

namespace NS_NAMESPACE
{
	/*****************************************************************************
	****************************    ImageAccessor    *****************************
	*****************************************************************************/

	/**
	 *	@brief		A generic image accessor template for CUDA arrays.
	 *	@note		This provides position-aware access to CUDA array memory with support for
	 *				1D, 2D and 3D offset operations. The void specialization serves as the base
	 *				implementation containing common functionality.
	 */
	template<> struct ImageAccessor<void>
	{
		//!	@brief		CUDA array handle (device memory pointer)
		cudaArray_t handle = nullptr;
		
		//!	@brief		Current 3D position in the array (x, y, z coordinates)
		ulonglong3 pos = { 0, 0, 0 };

	public:

		//!	@brief		Creates a new accessor with 1D offset applied.
		ImageAccessor operator+(ulonglong offset) const
		{
			return ImageAccessor{ handle, ulonglong3{ pos.x + offset, pos.y, pos.z } };
		}

		//!	@brief		Creates a new accessor with 2D offset applied.
		ImageAccessor operator+(ulonglong2 offset) const
		{
			return ImageAccessor{ handle, ulonglong3{ pos.x + offset.x, pos.y + offset.y, pos.z } };
		}

		//!	@brief		Creates a new accessor with 3D offset applied.
		ImageAccessor operator+(ulonglong3 offset) const
		{
			return ImageAccessor{ handle, ulonglong3{ pos.x + offset.x, pos.y + offset.y, pos.z + offset.z } };
		}

		//!	@brief		Implicit conversion to underlying CUDA array handle
		operator cudaArray_t() const { return handle; }
	};


	/**
	 *	@brief		Typed image accessor inheriting base functionality
	 *	@tparam		Type - The pixel/element type of the CUDA array
	 *	@note		This template specialization adds type safety while maintaining
	 *				all positional access capabilities of the base class.
	 */
	template<typename Type> struct ImageAccessor : public ImageAccessor<void> {};
}
