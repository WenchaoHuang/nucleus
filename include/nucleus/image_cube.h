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

		//!	@brief		Default constructor.
		ImageCube() = default;

		/**
		 *	@brief		Constructs a cubemap image.
		 *	@param[in]	allocator - Pointer to the associated allocator.
		 *	@param[in]	format - Texel format of the image.
		 *	@param[in]	width - Width of the image.
		 *	@throw		cudaError_t - In case of failure.
		 */
		NS_API explicit ImageCube(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width);

	protected:

		//!	@brief		Copy constructor from `Image`.
		explicit ImageCube(const Image & image) : Image(image) {}
	};

	/*****************************************************************************
	***************************    ImageCube<Type>    ****************************
	*****************************************************************************/

	/**
	 *	@brief		Represents a cubemap texture memory with specified storing type.
	 */
	template<typename Type> class ImageCube : public ImageCube<void>
	{
		friend class ImageCubeLod<Type>;

	public:

		//!	@brief		Default constructor.
		ImageCube() = default;

		/**
		 *	@brief		Constructs a cubemap image.
		 *	@param[in]	allocator - Pointer to the associated allocator.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	numLayers - Layers of the image, is clamped down to 1.
		 *	@throw		cudaError_t - In case of failure.
		 */
		explicit ImageCube(std::shared_ptr<DeviceAllocator> allocator, size_t width) : ImageCube<void>(allocator, FormatOf<Type>::value, width) {}

	protected:

		//!	@brief		Copy constructor from `Image`.
		explicit ImageCube(const Image & image) : ImageCube<void>(image) {}

	public:

		//!	@brief		Returns accessor to the data.
		ImageAccessor<Type> data() const { return ImageAccessor<Type>{ m_hImage }; }

		//!	@brief		Return the texel format of the image at compile time.
		static constexpr Format format() { return FormatOf<Type>::value; }
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

		//!	@brief		Default constructor.
		ImageCubeLayered() = default;

		/**
		 *	@brief		Constructs a layered cubemap image.
		 *	@param[in]	allocator - Pointer to the associated allocator.
		 *	@param[in]	format - Texel format of the image.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	numLayers - Layers of the image, is clamped down to 1.
		 *	@throw		cudaError_t - In case of failure.
		 */
		NS_API explicit ImageCubeLayered(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, size_t numLayers);

	protected:

		//!	@brief		Copy constructor from `Image`.
		explicit ImageCubeLayered(const Image & image) : Image(image) {}

	public:

		//!	@brief		Returns the number of layers.
		uint32_t numLayers() const { return m_extent.depth / 6; }
	};

	/*****************************************************************************
	************************    ImageCubeLayered<Type>    ************************
	*****************************************************************************/

	/**
	 *	@brief		Represents a layered cubemap texture memory with specified storing type.
	 */
	template<typename Type> class ImageCubeLayered : public ImageCubeLayered<void>
	{
		friend class ImageCubeLayeredLod<Type>;

	public:

		//!	@brief		Default constructor.
		ImageCubeLayered() = default;

		/**
		 *	@brief		Constructs a layered cubemap image.
		 *	@param[in]	allocator - Pointer to the associated allocator.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	numLayers - Layers of the image, is clamped down to 1.
		 *	@throw		cudaError_t - In case of failure.
		 */
		explicit ImageCubeLayered(std::shared_ptr<DeviceAllocator> allocator, size_t width, size_t numLayers) : ImageCubeLayered<void>(allocator, FormatOf<Type>::value, width, numLayers) {}

	protected:

		//!	@brief		Copy constructor from `Image`.
		explicit ImageCubeLayered(const Image & image) : ImageCubeLayered<void>(image) {}

	public:

		//!	@brief		Returns accessor to the data.
		ImageAccessor<Type> data() const { return ImageAccessor<Type>{ m_hImage }; }

		//!	@brief		Return the texel format of the image at compile time.
		static constexpr Format format() { return FormatOf<Type>::value; }
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

		//!	@brief		Default constructor.
		ImageCubeLod() = default;

		/**
		 *	@brief		Constructs a cube mipmapped image.
		 *	@param[in]	allocator - Pointer to the associated allocator.
		 *	@param[in]	format - Texel format of the image.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	numLevels - Number of mipmap levels to allocated, is clamped to the range [1, 1 + floor(log2(width))].
		 *	@throw		cudaError_t - In case of failure.
		 */
		NS_API explicit ImageCubeLod(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, unsigned int numLevels);

	public:

		//!	@brief		Return the specified level.
		ImageCube<void> level(size_t i) const { return ImageCube<void>(m_mipmaps[i]); }
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

		//!	@brief		Default constructor.
		ImageCubeLod() = default;

		/**
		 *	@brief		Constructs a cube mipmapped image.
		 *	@param[in]	allocator - Pointer to the associated allocator.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	numLevels - Number of mipmap levels to allocated, is clamped to the range [1, 1 + floor(log2(width))].
		 *	@throw		cudaError_t - In case of failure.
		 */
		explicit ImageCubeLod(std::shared_ptr<DeviceAllocator> allocator, size_t width, unsigned int numLevels) : ImageCubeLod<void>(allocator, FormatOf<Type>::value, width, numLevels) {}

	public:

		//!	@brief		Return the specified level.
		ImageCube<Type> level(size_t i) const { return ImageCube<Type>(m_mipmaps[i]); }

		//!	@brief		Returns the texel format of the image at compile time.
		static constexpr Format format() { return FormatOf<Type>::value; }
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

		//!	@brief		Default constructor.
		ImageCubeLayeredLod() = default;

		/**
		 *	@brief		Constructs a cube layered mipmapped image.
		 *	@param[in]	allocator - Pointer to the associated allocator.
		 *	@param[in]	format - Texel format of the image.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	numLayers - Layers of the image, is clamped down to 1.
		 *	@param[in]	numLevels - Number of mipmap levels to allocated, is clamped to the range [1, 1 + floor(log2(width))].
		 *	@throw		cudaError_t - In case of failure.
		 */
		NS_API explicit ImageCubeLayeredLod(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, size_t numLayers, unsigned int numLevels);

	public:

		//!	@brief		Return the specified level.
		ImageCubeLayered<void> level(size_t i) const { return ImageCubeLayered<void>(m_mipmaps[i]); }

		//!	@brief		Returns the number of layers.
		uint32_t numLayers() const { return m_extent.depth / 6; }
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

		//!	@brief		Default constructor.
		ImageCubeLayeredLod() = default;

		/**
		 *	@brief		Constructs a cube layered mipmapped image.
		 *	@param[in]	allocator - Pointer to the associated allocator.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	numLayers - Layers of the image, is clamped down to 1.
		 *	@param[in]	numLevels - Number of mipmap levels to allocated, is clamped to the range [1, 1 + floor(log2(width))].
		 *	@throw		cudaError_t - In case of failure.
		 */
		explicit ImageCubeLayeredLod(std::shared_ptr<DeviceAllocator> allocator, size_t width, size_t numLayers, unsigned int numLevels) : ImageCubeLayeredLod<void>(allocator, FormatOf<Type>::value, width, numLayers, numLevels) {}

	public:

		//!	@brief		Return the specified level.
		ImageCubeLayered<Type> level(size_t i) const { return ImageCubeLayered<Type>(m_mipmaps[i]); }

		//!	@brief		Returns the texel format of the image at compile time.
		static constexpr Format format() { return FormatOf<Type>::value; }
	};
}
