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
	****************************    Image2D<void>    *****************************
	*****************************************************************************/

	/**
	 *	@brief		Represents a 2D texture memory.
	 */
	template<> class Image2D<void> : public Image
	{
		friend class Image2DLod<void>;

	public:

		/**
		 *	@brief		Constructs a 2D image.
		 *	@param[in]	allocator - Pointer to the associated allocator.
		 *	@param[in]	format - Texel format of the image.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	height - height of the image.
		 *	@throw		cudaError_t - In case of failure.
		 */
		NS_API explicit Image2D(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, size_t height);

	private:

		/**
		 *	@brief		Constructs from Image2DLod.
		 *	@param[in]	hImage - Handle of texture memory (from cudaMipmappedArray_t).
		 *	@param[in]	format - Texel format of the image.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	height - height of the image.
		 *	@param[in]	depth - Depth of the image.
		 *	@throw		cudaError_t - In case of failure.
		 *	@note		Created by class `Image2DLod<void>` only.
		 */
		explicit Image2D(cudaArray_t hImage, Format format, size_t width, size_t height, size_t depth) : Image(hImage, format, width, height, depth) {}

	public:

		//	Returns the height of the image.
		uint32_t height() const { return m_height; }
	};

	/*****************************************************************************
	****************************    Image2D<Type>    *****************************
	*****************************************************************************/

	/**
	 *	@brief		Represents a 2D texture memory with specified storing type.
	 */
	template<typename Type> class Image2D : public Image2D<void>
	{

	public:

		/**
		 *	@brief		Constructs a 2D image.
		 *	@param[in]	allocator - Pointer to the associated allocator.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	height - height of the image.
		 *	@param[in]	numLayers - Layers of the image, is clamped down to 1.
		 *	@throw		cudaError_t - In case of failure.
		 */
		explicit Image2D(std::shared_ptr<DeviceAllocator> allocator, size_t width, size_t height) : Image2D<void>(allocator, FormatMapping<Type>::value, width, height) {}

	public:

		//	Returns accessor to the data.
		ImageAccessor<Type> data() const { return ImageAccessor<Type>{ m_hImage }; }

		//	Returns the texel format of the image at compile time.
		static constexpr Format format() { return FormatMapping<Type>::value; }
	};

	/*****************************************************************************
	*************************    Image2DLayered<void>    *************************
	*****************************************************************************/

	/**
	 *	@brief		Represents a 2D layered texture memory.
	 */
	template<> class Image2DLayered<void> : public Image
	{
		friend class Image2DLayeredLod<void>;

	public:

		/**
		 *	@brief		Constructs a layered 2D image.
		 *	@param[in]	allocator - Pointer to the associated allocator.
		 *	@param[in]	format - Texel format of the image.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	height - height of the image.
		 *	@param[in]	numLayers - Layers of the image, is clamped down to 1.
		 *	@throw		cudaError_t - In case of failure.
		 */
		NS_API explicit Image2DLayered(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, size_t height, size_t numLayers);

	private:

		/**
		 *	@brief		Constructs from MipmappedTextureMemory2DLayered.
		 *	@param[in]	hImage - Handle of texture memory (from cudaMipmappedArray_t).
		 *	@param[in]	format - Texel format of the image.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	height - height of the image.
		 *	@param[in]	depth - Depth of the image.
		 *	@throw		cudaError_t - In case of failure.
		 *	@note		Created by class `Image2DLayeredLod<void>` only.
		 */
		explicit Image2DLayered(cudaArray_t hImage, Format format, size_t width, size_t height, size_t depth) : Image(hImage, format, width, height, depth) {}

	public:

		//	Returns the number of layers.
		uint32_t numLayers() const { return m_depth; }

		//	Returns the height of the image.
		uint32_t height() const { return m_height; }
	};

	/*****************************************************************************
	*************************    Image2DLayered<Type>    *************************
	*****************************************************************************/

	/**
	 *	@brief		Represents a 2D layered texture memory with specified storing type.
	 */
	template<typename Type> class Image2DLayered : public Image2DLayered<void>
	{

	public:

		/**
		 *	@brief		Constructs a layered 2D image.
		 *	@param[in]	allocator - Pointer to the associated allocator.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	height - height of the image.
		 *	@param[in]	numLayers - Layers of the image, is clamped down to 1.
		 *	@throw		cudaError_t - In case of failure.
		 */
		explicit Image2DLayered(std::shared_ptr<DeviceAllocator> allocator, size_t width, size_t height, size_t numLayers) : Image2DLayered<void>(allocator, FormatMapping<Type>::value, width, height, numLayers) {}
	
	public:

		//	Returns accessor to the data.
		ImageAccessor<Type> data() const { return ImageAccessor<Type>{ m_hImage }; }

		//	Returns the texel format of the image at compile time.
		static constexpr Format format() { return FormatMapping<Type>::value; }
	};

	/*****************************************************************************
	***************************    Image2DLod<void>    ***************************
	*****************************************************************************/

	/**
	 *	@brief		Represents a 2D mipmapped texture memory.
	 */
	template<> class Image2DLod<void> : public ImageLod
	{

	public:

		/**
		 *	@brief		Constructs a 2D mipmapped image.
		 *	@param[in]	allocator - Pointer to the associated allocator.
		 *	@param[in]	format - Texel format of the image.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	height - height of the image.
		 *	@param[in]	numLevels - Number of mipmap levels to allocated, is clamped to the range [1, 1 + floor(log2(max(width, height)))].
		 *	@throw		cudaError_t - In case of failure.
		 */
		NS_API Image2DLod(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, size_t height, unsigned int numLevels);


		/**
		 *	@return		Reference to the specified level.
		 *	@warning	`level` should be in the range [0, numLevel).
		 */
		Image2D<void> & getLevel(size_t level) { return *m_mipmaps[level]; }


		/**
		 *	@return		The height of the image.
		 */
		uint32_t height() const { return m_height; }

	private:

		std::vector<std::shared_ptr<Image2D<void>>>		m_mipmaps;
	};

	/*****************************************************************************
	***************************    Image2DLod<Type>    ***************************
	*****************************************************************************/

	/**
	 *	@brief		Represents a 2D mipmapped texture memory width specified storing type.
	 */
	template<typename Type> class Image2DLod : public Image2DLod<void>
	{

	public:

		/**
		 *	@brief		Constructs a 2D mipmapped image.
		 *	@param[in]	allocator - Pointer to the associated allocator.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	height - height of the image.
		 *	@param[in]	numLevels - Number of mipmap levels to allocated, is clamped to the range [1, 1 + floor(log2(max(width, height)))].
		 *	@throw		cudaError_t - In case of failure.
		 */
		Image2DLod(std::shared_ptr<DeviceAllocator> allocator, size_t width, size_t height, unsigned int numLevels) : Image2DLod<void>(allocator, FormatMapping<Type>::value, width, height, numLevels) {}


		/**
		 *	@return		Reference to the specified level.
		 *	@warning	`level` should be in the range [0, numLevel).
		 */
		Image2D<Type> & getLevel(size_t level) { return reinterpret_cast<Image2D<Type>&>(Image2DLod<void>::getLevel(level)); }


		/**
		 *	@return		Texel format of the image at compile time.
		 */
		static constexpr Format format() { return FormatMapping<Type>::value; }
	};

	/*****************************************************************************
	***********************    Image2DLayeredLod<void>    ************************
	*****************************************************************************/

	/**
	 *	@brief		Represents a 2D layered mipmapped texture memory.
	 */
	template<> class Image2DLayeredLod<void> : public ImageLod
	{

	public:

		/**
		 *	@brief		Constructs a 2D layered mipmapped image.
		 *	@param[in]	allocator - Pointer to the associated allocator.
		 *	@param[in]	format - Texel format of the image.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	height - height of the image.
		 *	@param[in]	numLayers - Layers of the image, is clamped down to 1.
		 *	@param[in]	numLevels - Number of mipmap levels to allocated, is clamped to the range [1, 1 + floor(log2(max(width, height)))].
		 *	@throw		cudaError_t - In case of failure.
		 */
		NS_API Image2DLayeredLod(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, size_t height, size_t numLayers, unsigned int numLevels);


		/**
		 *	@return		Reference to the specified level.
		 *	@warning	`level` should be in the range [0, numLevel).
		 */
		Image2DLayered<void> & getLevel(size_t level) { return *m_mipmaps[level]; }

	public:

		//	Returns the number of layers.
		uint32_t numLayers() const { return m_depth; }

		//	Returns the height of the image.
		uint32_t height() const { return m_height; }

	private:

		std::vector<std::shared_ptr<Image2DLayered<void>>>		m_mipmaps;
	};

	/*****************************************************************************
	***********************    Image2DLayeredLod<Type>    ************************
	*****************************************************************************/

	/**
	 *	@brief		Represents a 2D layered mipmapped texture memory width specified storing type.
	 */
	template<typename Type> class Image2DLayeredLod : public Image2DLayeredLod<void>
	{

	public:

		/**
		 *	@brief		Constructs a 2D layered mipmapped image.
		 *	@param[in]	allocator - Pointer to the associated allocator.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	height - height of the image.
		 *	@param[in]	numLayers - Layers of the image, is clamped down to 1.
		 *	@param[in]	numLevels - Number of mipmap levels to allocated, is clamped to the range [1, 1 + floor(log2(max(width, height)))].
		 *	@throw		cudaError_t - In case of failure.
		 */
		Image2DLayeredLod(std::shared_ptr<DeviceAllocator> allocator, size_t width, size_t height, size_t numLayers, unsigned int numLevels) : Image2DLayeredLod<void>(allocator, FormatMapping<Type>::value, width, height, numLayers, numLevels) {}


		/**
		 *	@return		Reference to the specified level.
		 *	@warning	`level` should be in the range [0, numLevel).
		 */
		Image2DLayered<Type> & getLevel(size_t level) { return reinterpret_cast<Image2DLayered<Type>&>(Image2DLayeredLod<void>::getLevel(level)); }


		/**
		 *	@return		Texel format of the image at compile time.
		 */
		static constexpr Format format() { return FormatMapping<Type>::value; }
	};
}