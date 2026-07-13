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
	****************************    Image1D<void>    *****************************
	*****************************************************************************/

	/**
	 *	@brief		Represents a 1D texture memory.
	 */
	template<> class Image1D<void> : public Image
	{
		friend class Image1DLod<void>;

	public:

		/**
		 *	@brief		Constructs a 1D image.
		 *	@param[in]	allocator - Pointer to the associated allocator.
		 *	@param[in]	format - Texel format of the image.
		 *	@param[in]	width - Width of the image.
		 */
		NS_API explicit Image1D(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width);

	private:

		/**
		 *	@brief		Constructs from Image1DLod.
		 *	@param[in]	hImage - Handle of texture memory (from cudaMipmappedArray_t).
		 *	@param[in]	format - Texel format of the image.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	height - height of the image.
		 *	@param[in]	depth - Depth of the image.
		 *	@param[in]	flags - Flags for image creation (interanl use).
		 *	@throw		cudaError_t - In case of failure.
		 *	@note		Created by class `Image1DLod<void>` only.
		 */
		explicit Image1D(cudaArray_t hImage, Format format, size_t width, size_t height, size_t depth, int flags) : Image(hImage, format, width, height, depth, flags) {}
	};

	/*****************************************************************************
	****************************    Image1D<Type>    *****************************
	*****************************************************************************/

	/**
	 *	@brief		Represents a 1D texture memory with specified storing type.
	 */
	template<typename Type> class Image1D : public Image1D<void>
	{

	public:

		/**
		 *	@brief		Constructs a 1D image.
		 *	@param[in]	allocator - Pointer to the associated allocator.
		 *	@param[in]	width - Width of the image.
		 */
		explicit Image1D(std::shared_ptr<DeviceAllocator> allocator, size_t width) : Image1D<void>(allocator, FormatMapping<Type>::value, width) {}

	public:

		//	Returns accessor to the data.
		ImageAccessor<Type> data() const { return ImageAccessor<Type>{ m_hImage }; }

		//	Returns the texel format of the image at compile time.
		static constexpr Format format() { return FormatMapping<Type>::value; }
	};

	/*****************************************************************************
	*************************    Image1DLayered<void>    *************************
	*****************************************************************************/

	/**
	 *	@brief		Represents a 1D layered texture memory.
	 */
	template<> class Image1DLayered<void> : public Image
	{
		friend class Image1DLayeredLod<void>;

	public:

		/**
		 *	@brief		Constructs a 1D layered image.
		 *	@param[in]	allocator - Pointer to the associated allocator.
		 *	@param[in]	format - Texel format of the image.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	numLayers - Layers of the image, is clamped down to 1.
		 */
		NS_API explicit Image1DLayered(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, size_t numLayers);

	private:

		/**
		 *	@brief		Constructs from MipmappedTextureMemory1DLayered.
		 *	@param[in]	hImage - Handle of texture memory (from cudaMipmappedArray_t).
		 *	@param[in]	format - Texel format of the image.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	height - height of the image.
		 *	@param[in]	depth - Depth of the image.
		 * 	@param[in]	flags - Flags for image creation (interanl use).
		 *	@throw		cudaError_t - In case of failure.
		 *	@note		Created by class `Image1DLayeredLod<void>` only.
		 */
		explicit Image1DLayered(cudaArray_t hImage, Format format, size_t width, size_t height, size_t depth, int flags) : Image(hImage, format, width, height, depth, flags) {}

	public:

		//	Returns the number of layers.
		uint32_t numLayers() const { return m_depth; }
	};

	/*****************************************************************************
	*************************    Image1DLayered<Type>    *************************
	*****************************************************************************/

	/**
	 *	@brief		Represents a 1D layered texture memory with specified storing type.
	 */
	template<typename Type> class Image1DLayered : public Image1DLayered<void>
	{

	public:

		/**
		 *	@brief		Constructs a 1D layered image.
		 *	@param[in]	allocator - Pointer to the associated allocator.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	numLayers - Number of layers.
		 */
		explicit Image1DLayered(std::shared_ptr<DeviceAllocator> allocator, size_t width, size_t numLayers) : Image1DLayered<void>(allocator, FormatMapping<Type>::value, width, numLayers) {}

	public:

		//	Returns accessor to the data.
		ImageAccessor<Type> data() const { return ImageAccessor<Type>{ m_hImage }; }

		//	Returns the texel format of the image at compile time.
		static constexpr Format format() { return FormatMapping<Type>::value; }
	};

	/*****************************************************************************
	***************************    Image1DLod<void>    ***************************
	*****************************************************************************/

	/**
	 *	@brief		Represents a 1D mipmapped texture memory.
	 */
	template<> class Image1DLod<void> : public ImageLod
	{

	public:

		/**
		 *	@brief		Constructs a 1D mipmapped image.
		 *	@param[in]	allocator - Pointer to the associated allocator.
		 *	@param[in]	format - Texel format of the image.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	numLevels - Number of mipmap levels to allocated, is clamped to the range [1, 1 + floor(log2(width))].
		 *	@throw		cudaError_t - In case of failure.
		 */
		NS_API Image1DLod(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, unsigned int numLevels);


		/**
		 *	@return		Reference to the specified level.
		 *	@warning	`level` should be in the range [0, numLevel).
		 */
		Image1D<void> & getLevel(size_t level) { return *m_mipmaps[level]; }

	private:

		std::vector<std::shared_ptr<Image1D<void>>>		m_mipmaps;
	};

	/*****************************************************************************
	***************************    Image1DLod<Type>    ***************************
	*****************************************************************************/

	/**
	 *	@brief		Represents a 1D mipmapped texture memory with specified storing type.
	 */
	template<typename Type> class Image1DLod : public Image1DLod<void>
	{

	public:

		/**
		 *	@brief		Constructs a 1D mipmapped image.
		 *	@param[in]	allocator - Pointer to the associated allocator.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	numLevels - Number of mipmap levels to allocated, is clamped to the range [1, 1 + floor(log2(width))].
		 *	@throw		cudaError_t - In case of failure.
		 */
		Image1DLod(std::shared_ptr<DeviceAllocator> allocator, size_t width, unsigned int numLevels) : Image1DLod<void>(allocator, FormatMapping<Type>::value, width, numLevels) {}


		/**
		 *	@return		Reference to the specified level.
		 *	@warning	`level` should be in the range [0, numLevel).
		 */
		Image1D<Type> & getLevel(size_t level) { return reinterpret_cast<Image1D<Type>&>(Image1DLod<void>::getLevel(level)); }


		/**
		 *	@return		Texel format of the image at compile time.
		 */
		static constexpr Format format() { return FormatMapping<Type>::value; }
	};

	/*****************************************************************************
	***********************    Image1DLayeredLod<void>    ************************
	*****************************************************************************/

	/**
	 *	@brief		Represents a 1D layered mipmapped texture memory.
	 */
	template<> class Image1DLayeredLod<void> : public ImageLod
	{

	public:

		/**
		 *	@brief		Constructs a 1D layered mipmapped image.
		 *	@param[in]	allocator - Pointer to the associated allocator.
		 *	@param[in]	format - Texel format of the image.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	numLayers - Layers of the image, is clamped down to 1.
		 *	@param[in]	numLevels - Number of mipmap levels to allocated, is clamped to the range [1, 1 + floor(log2(width))].
		 *	@throw		cudaError_t - In case of failure.
		 */
		NS_API Image1DLayeredLod(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, size_t numLayers, unsigned int numLevels);


		/**
		 *	@return		Reference to the specified level.
		 *	@warning	`level` should be in the range [0, numLevel).
		 */
		Image1DLayered<void> & getLevel(size_t level) { return *m_mipmaps[level]; }


		/**
		 *	@return		The number of layers.
		 */
		uint32_t numLayers() const { return m_depth; }

	private:

		std::vector<std::shared_ptr<Image1DLayered<void>>>		m_mipmaps;
	};

	/*****************************************************************************
	***********************    Image1DLayeredLod<Type>    ************************
	*****************************************************************************/

	/**
	 *	@brief		Represents a 1D layered mipmapped texture memory with specified storing type.
	 */
	template<typename Type> class Image1DLayeredLod : public Image1DLayeredLod<void>
	{

	public:

		/**
		 *	@brief		Constructs a 1D layered mipmapped image.
		 *	@param[in]	allocator - Pointer to the associated allocator.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	numLayers - Layers of the image, is clamped down to 1.
		 *	@param[in]	numLevels - Number of mipmap levels to allocated, is clamped to the range [1, 1 + floor(log2(width))].
		 *	@throw		cudaError_t - In case of failure.
		 */
		Image1DLayeredLod(std::shared_ptr<DeviceAllocator> allocator, size_t width, size_t numLayers, unsigned int numLevels) : Image1DLayeredLod<void>(allocator, FormatMapping<Type>::value, width, numLayers, numLevels) {}


		/**
		 *	@return		Reference to the specified level.
		 *	@warning	`level` should be in the range [0, numLevel).
		 */
		Image1DLayered<Type> & getLevel(size_t level) { return reinterpret_cast<Image1DLayered<Type>&>(Image1DLayeredLod<void>::getLevel(level)); }


		/**
		 *	@return		Texel format of the image at compile time.
		 */
		static constexpr Format format() { return FormatMapping<Type>::value; }
	};
}