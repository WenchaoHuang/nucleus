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
	***************************    ImageCube<void>    ****************************
	*****************************************************************************/

	/**
	 *	@brief		Represents a cubemap texture memory.
	 */
	template<> class ImageCube<void> : public Image
	{
		friend class ImageCubeLod<void>;

	public:

		/**
		 *	@brief		Constructs a cubemap image.
		 *	@param[in]	allocator - Pointer to the associated allocator.
		 *	@param[in]	format - Texel format of the image.
		 *	@param[in]	width - Width of the image.
		 *	@throw		cudaError_t - In case of failure.
		 */
		NS_API explicit ImageCube(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width);

	private:

		/**
		 *	@brief		Constructs from ImageCubeLod.
		 *	@param[in]	hImage - Handle of texture memory (from cudaMipmappedArray_t).
		 *	@param[in]	format - Texel format of the image.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	height - height of the image.
		 *	@param[in]	depth - Depth of the image.
		 * 	@param[in]	flags - Flags for image creation (interanl use).
		 *	@throw		cudaError_t - In case of failure.
		 * 	@note		Created by class `ImageCubeLod<void>` only.
		 */
		explicit ImageCube(cudaArray_t hImage, Format format, size_t width, size_t height, size_t depth, int flags) : Image(hImage, format, width, height, depth, flags) {}
	};

	/*****************************************************************************
	***************************    ImageCube<Type>    ****************************
	*****************************************************************************/

	/**
	 *	@brief		Represents a cubemap texture memory with specified storing type.
	 */
	template<typename Type> class ImageCube : public ImageCube<void>
	{

	public:

		/**
		 *	@brief		Constructs a cubemap image.
		 *	@param[in]	allocator - Pointer to the associated allocator.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	numLayers - Layers of the image, is clamped down to 1.
		 *	@throw		cudaError_t - In case of failure.
		 */
		explicit ImageCube(std::shared_ptr<DeviceAllocator> allocator, size_t width) : ImageCube<void>(allocator, FormatMapping<Type>::value, width) {}

	public:

		//	Returns accessor to the data.
		ImageAccessor<Type> data() const { return ImageAccessor<Type>{ m_hImage }; }

		//	Return the texel format of the image at compile time.
		static constexpr Format format() { return FormatMapping<Type>::value; }
	};

	/*****************************************************************************
	************************    ImageCubeLayered<void>    ************************
	*****************************************************************************/

	/**
	 *	@brief		Represents a layered cubemap texture memory.
	 */
	template<> class ImageCubeLayered<void> : public Image
	{
		friend class ImageCubeLayeredLod<void>;

	public:

		/**
		 *	@brief		Constructs a layered cubemap image.
		 *	@param[in]	allocator - Pointer to the associated allocator.
		 *	@param[in]	format - Texel format of the image.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	numLayers - Layers of the image, is clamped down to 1.
		 *	@throw		cudaError_t - In case of failure.
		 */
		NS_API explicit ImageCubeLayered(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, size_t numLayers);

	private:

		/**
		 *	@brief		Constructs from MipmappedTextureMemoryCubemapLayered.
		 *	@param[in]	hImage - Handle of texture memory (from cudaMipmappedArray_t).
		 *	@param[in]	format - Texel format of the image.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	height - height of the image.
		 *	@param[in]	depth - Depth of the image.
		 * 	@param[in]	flags - Flags for image creation (interanl use).
		 *	@throw		cudaError_t - In case of failure.
		 * 	@note		Created by class `ImageCubeLayeredLod<void>` only.
		 */
		explicit ImageCubeLayered(cudaArray_t hImage, Format format, size_t width, size_t height, size_t depth, int flags) : Image(hImage, format, width, height, depth, flags) {}

	public:

		//	Returns the number of layers.
		uint32_t numLayers() const { return m_depth / 6; }
	};

	/*****************************************************************************
	************************    ImageCubeLayered<Type>    ************************
	*****************************************************************************/

	/**
	 *	@brief		Represents a layered cubemap texture memory with specified storing type.
	 */
	template<typename Type> class ImageCubeLayered : public ImageCubeLayered<void>
	{

	public:

		/**
		 *	@brief		Constructs a layered cubemap image.
		 *	@param[in]	allocator - Pointer to the associated allocator.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	numLayers - Layers of the image, is clamped down to 1.
		 *	@throw		cudaError_t - In case of failure.
		 */
		explicit ImageCubeLayered(std::shared_ptr<DeviceAllocator> allocator, size_t width, size_t numLayers) : ImageCubeLayered<void>(allocator, FormatMapping<Type>::value, width, numLayers) {}
	
	public:

		//	Returns accessor to the data.
		ImageAccessor<Type> data() const { return ImageAccessor<Type>{ m_hImage }; }

		//	Return the texel format of the image at compile time.
		static constexpr Format format() { return FormatMapping<Type>::value; }
	};

	/*****************************************************************************
	**************************    ImageCubeLod<void>    **************************
	*****************************************************************************/

	/**
	 *	@brief		Represents a cube mipmapped texture memory.
	 */
	template<> class ImageCubeLod<void> : public ImageLod
	{

	public:

		/**
		 *	@brief		Constructs a cube mipmapped image.
		 *	@param[in]	allocator - Pointer to the associated allocator.
		 *	@param[in]	format - Texel format of the image.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	numLevels - Number of mipmap levels to allocated, is clamped to the range [1, 1 + floor(log2(width))].
		 *	@throw		cudaError_t - In case of failure.
		 */
		NS_API ImageCubeLod(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, unsigned int numLevels);


		/**
		 *	@return		Reference to the specified level.
		 *	@warning	`level` should be in the range [0, numLevel).
		 */
		ImageCube<void> & getLevel(size_t level) { return *m_mipmaps[level]; }

	private:

		std::vector<std::shared_ptr<ImageCube<void>>>		m_mipmaps;
	};

	/*****************************************************************************
	**************************    ImageCubeLod<Type>    **************************
	*****************************************************************************/

	/**
	 *	@brief		Represents a cube mipmapped texture memory width specified storing type.
	 */
	template<typename Type> class ImageCubeLod : public ImageCubeLod<void>
	{

	public:

		/**
		 *	@brief		Constructs a cube mipmapped image.
		 *	@param[in]	allocator - Pointer to the associated allocator.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	numLevels - Number of mipmap levels to allocated, is clamped to the range [1, 1 + floor(log2(width))].
		 *	@throw		cudaError_t - In case of failure.
		 */
		ImageCubeLod(std::shared_ptr<DeviceAllocator> allocator, size_t width, unsigned int numLevels) : ImageCubeLod<void>(allocator, FormatMapping<Type>::value, width, numLevels) {}


		/**
		 *	@return		Reference to the specified level.
		 *	@warning	`level` should be in the range [0, numLevel).
		 */
		ImageCube<Type> & getLevel(size_t level) { return reinterpret_cast<ImageCube<Type>&>(ImageCubeLod<void>::getLevel(level)); }


		/**
		 *	@return		Texel format of the image at compile time.
		 */
		static constexpr Format format() { return FormatMapping<Type>::value; }
	};

	/*****************************************************************************
	**********************    ImageCubeLayeredLod<void>    ***********************
	*****************************************************************************/

	/**
	 *	@brief		Represents a cube layered mipmapped texture memory.
	 */
	template<> class ImageCubeLayeredLod<void> : public ImageLod
	{

	public:

		/**
		 *	@brief		Constructs a cube layered mipmapped image.
		 *	@param[in]	allocator - Pointer to the associated allocator.
		 *	@param[in]	format - Texel format of the image.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	numLayers - Layers of the image, is clamped down to 1.
		 *	@param[in]	numLevels - Number of mipmap levels to allocated, is clamped to the range [1, 1 + floor(log2(width))].
		 *	@throw		cudaError_t - In case of failure.
		 */
		NS_API ImageCubeLayeredLod(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, size_t numLayers, unsigned int numLevels);


		/**
		 *	@return		Reference to the specified level.
		 *	@warning	`level` should be in the range [0, numLevel).
		 */
		ImageCubeLayered<void> & getLevel(size_t level) { return *m_mipmaps[level]; }

		
		/**
		 *	@return		The number of layers.
		 */
		uint32_t numLayers() const { return m_depth / 6; }

	private:

		std::vector<std::shared_ptr<ImageCubeLayered<void>>>		m_mipmaps;
	};

	/*****************************************************************************
	**********************    ImageCubeLayeredLod<Type>    ***********************
	*****************************************************************************/

	/**
	 *	@brief		Represents a cube layered mipmapped texture memory width specified storing type.
	 */
	template<typename Type> class ImageCubeLayeredLod : public ImageCubeLayeredLod<void>
	{

	public:

		/**
		 *	@brief		Constructs a cube layered mipmapped image.
		 *	@param[in]	allocator - Pointer to the associated allocator.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	numLayers - Layers of the image, is clamped down to 1.
		 *	@param[in]	numLevels - Number of mipmap levels to allocated, is clamped to the range [1, 1 + floor(log2(width))].
		 *	@throw		cudaError_t - In case of failure.
		 */
		ImageCubeLayeredLod(std::shared_ptr<DeviceAllocator> allocator, size_t width, size_t numLayers, unsigned int numLevels) : ImageCubeLayeredLod<void>(allocator, FormatMapping<Type>::value, width, numLayers, numLevels) {}


		/**
		 *	@return		Reference to the specified level.
		 *	@warning	`level` should be in the range [0, numLevel).
		 */
		ImageCubeLayered<Type> & getLevel(size_t level) { return reinterpret_cast<ImageCubeLayered<Type>&>(ImageCubeLayeredLod<void>::getLevel(level)); }


		/**
		 *	@return		Texel format of the image at compile time.
		 */
		static constexpr Format format() { return FormatMapping<Type>::value; }
	};
}