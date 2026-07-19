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

#include "image.h"

namespace NS_NAMESPACE
{
	/*****************************************************************************
	****************************    Image3D<void>    *****************************
	*****************************************************************************/

	/**
	 *	@brief		Represents a 3D texture memory.
	 */
	template<> class Image3D<void> : public Image
	{

	public:

		/**
		 *	@brief		Constructs a 3D image.
		 *	@param[in]	allocator - Pointer to the associated allocator.
		 *	@param[in]	format - Texel format of the image.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	height - height of the image.
		 *	@param[in]	depth - Depth of the image.
		 *	@throw		cudaError_t - In case of failure.
		 */
		NS_API explicit Image3D(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, size_t height, size_t depth);

	public:

		//	Returns the height of the image.
		uint32_t height() const { return m_height; }

		//	Returns the depth of the image.
		uint32_t depth() const { return m_depth; }
	};

	/*****************************************************************************
	****************************    Image3D<Type>    *****************************
	*****************************************************************************/

	/**
	 *	@brief		Represents a 3D texture memory with specified storing type.
	 */
	template<typename Type> class Image3D : public Image3D<void>
	{

	public:

		/**
		 *	@brief		Constructs a 3D image.
		 *	@param[in]	allocator - Pointer to the associated allocator.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	height - height of the image.
		 *	@param[in]	depth - Depth of the image.
		 *	@throw		cudaError_t - In case of failure.
		 */
		explicit Image3D(std::shared_ptr<DeviceAllocator> allocator, size_t width, size_t height, size_t depth) : Image3D<void>(allocator, FormatMapping<Type>::value, width, height, depth) {}

	public:

		//	Returns accessor to the data.
		ImageAccessor<Type> data() const { return ImageAccessor<Type>{ m_hImage }; }

		//	Return the texel format of the image at compile time.
		static constexpr Format format() { return FormatMapping<Type>::value; }
	};

	/*****************************************************************************
	***************************    Image3DLod<void>    ***************************
	*****************************************************************************/

	/**
	 *	@brief		Represents a 3D mipmapped texture memory.
	 */
	template<> class Image3DLod<void> : public ImageLod
	{

	public:

		/**
		 *	@brief		Constructs a 3D mipmapped image.
		 *	@param[in]	allocator - Pointer to the associated allocator.
		 *	@param[in]	format - Texel format of the image.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	height - height of the image.
		 *	@param[in]	depth - Depth of the image.
		 *	@param[in]	numLevels - Number of mipmap levels to allocated, is clamped to the range [1, 1 + floor(log2(max(width, height, depth)))].
		 *	@throw		cudaError_t - In case of failure.
		 */
		NS_API Image3DLod(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, size_t height, size_t depth, unsigned int numLevels);

		//	Returns the height of the image.
		uint32_t height() const { return m_height; }

		//	Returns the depth of the image.
		uint32_t depth() const { return m_depth; }
	};

	/*****************************************************************************
	***************************    Image3DLod<Type>    ***************************
	*****************************************************************************/

	/**
	 *	@brief		Represents a 3D mipmapped texture memory width specified storing type.
	 */
	template<typename Type> class Image3DLod : public Image3DLod<void>
	{

	public:

		/**
		 *	@brief		Constructs a 3D mipmapped image.
		 *	@param[in]	allocator - Pointer to the associated allocator.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	height - height of the image.
		 *	@param[in]	depth - Depth of the image.
		 *	@param[in]	numLevels - Number of mipmap levels to allocated, is clamped to the range [1, 1 + floor(log2(max(width, height, depth)))].
		 *	@throw		cudaError_t - In case of failure.
		 */
		Image3DLod(std::shared_ptr<DeviceAllocator> allocator, size_t width, size_t height, size_t depth, unsigned int numLevels) : Image3DLod<void>(allocator, FormatMapping<Type>::value, width, height, depth, numLevels) {}


		/**
		 *	@return		Texel format of the image at compile time.
		 */
		static constexpr Format format() { return FormatMapping<Type>::value; }
	};
}
