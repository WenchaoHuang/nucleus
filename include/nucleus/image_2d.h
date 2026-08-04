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
#include "runtime.h"

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
		 *	@brief		Default constructor.
		 */
		Image2D() = default;


		/**
		 *	@brief		Constructs a 2D image.
		 *	@param[in]	allocator - Pointer to the associated allocator.
		 *	@param[in]	format - Texel format of the image.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	height - height of the image.
		 *	@throw		cudaError_t - In case of failure.
		 */
		NS_API explicit Image2D(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, size_t height);


		/**
		 *	@brief		Constructs a 2D image with default allocator.
		 *	@param[in]	format - Texel format of the image.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	height - height of the image.
		 *	@throw		cudaError_t - In case of failure.
		 */
		explicit Image2D(Format format, size_t width, size_t height) : Image2D(Runtime::defaultAllocator(), format, width, height) {}

	protected:

		//!	@brief		Copy constructor from `Image`.
		explicit Image2D(const Image & image) : Image(image) {}

	public:

		//!	@brief		Returns the height of the image.
		uint32_t height() const { return m_extent.height; }
	};

	/*****************************************************************************
	****************************    Image2D<Type>    *****************************
	*****************************************************************************/

	/**
	 *	@brief		Represents a 2D texture memory with specified storing type.
	 */
	template<typename Type> class Image2D : public Image2D<void>
	{
		friend class Image2DLod<Type>;

	public:

		/**
		 *	@brief		Default constructor.
		 */
		Image2D() = default;


		/**
		 *	@brief		Constructs a 2D image with default allocator.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	height - height of the image.
		 *	@param[in]	numLayers - Layers of the image, is clamped down to 1.
		 *	@throw		cudaError_t - In case of failure.
		 */
		explicit Image2D(size_t width, size_t height) : Image2D(Runtime::defaultAllocator(), width, height) {}


		/**
		 *	@brief		Constructs a 2D image.
		 *	@param[in]	allocator - Pointer to the associated allocator.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	height - height of the image.
		 *	@param[in]	numLayers - Layers of the image, is clamped down to 1.
		 *	@throw		cudaError_t - In case of failure.
		 */
		explicit Image2D(std::shared_ptr<DeviceAllocator> allocator, size_t width, size_t height) : Image2D<void>(allocator, FormatOf<Type>::value, width, height) {}

	protected:

		//!	@brief		Copy constructor from `Image`.
		explicit Image2D(const Image & image) : Image2D<void>(image) {}

	public:

		//!	@brief		Returns accessor to the data.
		ImageAccessor<Type> data() const { return ImageAccessor<Type>{ m_hImage }; }

		//!	@brief		Returns the texel format of the image at compile time.
		static constexpr Format format() { return FormatOf<Type>::value; }
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
		 *	@brief		Default constructor.
		 */
		Image2DLayered() = default;


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


		/**
		 *	@brief		Constructs a layered 2D image with default allocator.
		 *	@param[in]	format - Texel format of the image.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	height - height of the image.
		 *	@param[in]	numLayers - Layers of the image, is clamped down to 1.
		 *	@throw		cudaError_t - In case of failure.
		 */
		explicit Image2DLayered(Format format, size_t width, size_t height, size_t numLayers) : Image2DLayered(Runtime::defaultAllocator(), format, width, height, numLayers) {}

	protected:

		//!	@brief		Copy constructor from `Image`.
		explicit Image2DLayered(const Image & image) : Image(image) {}

	public:

		//!	@brief		Returns the number of layers.
		uint32_t numLayers() const { return m_extent.depth; }

		//!	@brief		Returns the height of the image.
		uint32_t height() const { return m_extent.height; }
	};

	/*****************************************************************************
	*************************    Image2DLayered<Type>    *************************
	*****************************************************************************/

	/**
	 *	@brief		Represents a 2D layered texture memory with specified storing type.
	 */
	template<typename Type> class Image2DLayered : public Image2DLayered<void>
	{
		friend class Image2DLayeredLod<Type>;

	public:

		/**
		 *	@brief		Default constructor.
		 */
		Image2DLayered() = default;


		/**
		 *	@brief		Constructs a layered 2D image with default allocator.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	height - height of the image.
		 *	@param[in]	numLayers - Layers of the image, is clamped down to 1.
		 *	@throw		cudaError_t - In case of failure.
		 */
		explicit Image2DLayered(size_t width, size_t height, size_t numLayers) : Image2DLayered(Runtime::defaultAllocator(), width, height, numLayers) {}


		/**
		 *	@brief		Constructs a layered 2D image.
		 *	@param[in]	allocator - Pointer to the associated allocator.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	height - height of the image.
		 *	@param[in]	numLayers - Layers of the image, is clamped down to 1.
		 *	@throw		cudaError_t - In case of failure.
		 */
		explicit Image2DLayered(std::shared_ptr<DeviceAllocator> allocator, size_t width, size_t height, size_t numLayers) : Image2DLayered<void>(allocator, FormatOf<Type>::value, width, height, numLayers) {}

	protected:

		//!	@brief		Copy constructor from `Image`.
		explicit Image2DLayered(const Image & image) : Image2DLayered<void>(image) {}

	public:

		//!	@brief		Returns accessor to the data.
		ImageAccessor<Type> data() const { return ImageAccessor<Type>{ m_hImage }; }

		//!	@brief		Returns the texel format of the image at compile time.
		static constexpr Format format() { return FormatOf<Type>::value; }
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
		 *	@brief		Default constructor.
		 */
		Image2DLod() = default;


		/**
		 *	@brief		Constructs a 2D mipmapped image.
		 *	@param[in]	allocator - Pointer to the associated allocator.
		 *	@param[in]	format - Texel format of the image.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	height - height of the image.
		 *	@param[in]	numLevels - Number of mipmap levels to allocated, is clamped to the range [1, 1 + floor(log2(max(width, height)))].
		 *	@throw		cudaError_t - In case of failure.
		 */
		NS_API explicit Image2DLod(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, size_t height, unsigned int numLevels);


		/**
		 *	@brief		Constructs a 2D mipmapped image with default allocator.
		 *	@param[in]	format - Texel format of the image.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	height - height of the image.
		 *	@param[in]	numLevels - Number of mipmap levels to allocated, is clamped to the range [1, 1 + floor(log2(max(width, height)))].
		 *	@throw		cudaError_t - In case of failure.
		 */
		explicit Image2DLod(Format format, size_t width, size_t height, unsigned int numLevels) : Image2DLod(Runtime::defaultAllocator(), format, width, height, numLevels) {}

	public:

		//!	@brief		Return the specified level.
		Image2D<void> level(size_t i) const { return Image2D<void>(m_mipmaps[i]); }

		//!	@brief		Returns the height of the image.
		uint32_t height() const { return m_extent.height; }
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
		 *	@brief		Default constructor.
		 */
		Image2DLod() = default;


		/**
		 *	@brief		Constructs a 2D mipmapped image with default allocator.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	height - height of the image.
		 *	@param[in]	numLevels - Number of mipmap levels to allocated, is clamped to the range [1, 1 + floor(log2(max(width, height)))].
		 *	@throw		cudaError_t - In case of failure.
		 */
		explicit Image2DLod(size_t width, size_t height, unsigned int numLevels) : Image2DLod(Runtime::defaultAllocator(), width, height, numLevels) {}


		/**
		 *	@brief		Constructs a 2D mipmapped image.
		 *	@param[in]	allocator - Pointer to the associated allocator.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	height - height of the image.
		 *	@param[in]	numLevels - Number of mipmap levels to allocated, is clamped to the range [1, 1 + floor(log2(max(width, height)))].
		 *	@throw		cudaError_t - In case of failure.
		 */
		explicit Image2DLod(std::shared_ptr<DeviceAllocator> allocator, size_t width, size_t height, unsigned int numLevels) : Image2DLod<void>(allocator, FormatOf<Type>::value, width, height, numLevels) {}

	public:

		//!	@brief		Return the specified level.
		Image2D<Type> level(size_t i) const { return Image2D<Type>(m_mipmaps[i]); }

		//!	@brief		Returns the texel format of the image at compile time.
		static constexpr Format format() { return FormatOf<Type>::value; }
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
		 *	@brief		Default constructor.
		 */
		Image2DLayeredLod() = default;


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
		NS_API explicit Image2DLayeredLod(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, size_t height, size_t numLayers, unsigned int numLevels);


		/**
		 *	@brief		Constructs a 2D layered mipmapped image with default allocator.
		 *	@param[in]	format - Texel format of the image.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	height - height of the image.
		 *	@param[in]	numLayers - Layers of the image, is clamped down to 1.
		 *	@param[in]	numLevels - Number of mipmap levels to allocated, is clamped to the range [1, 1 + floor(log2(max(width, height)))].
		 *	@throw		cudaError_t - In case of failure.
		 */
		explicit Image2DLayeredLod(Format format, size_t width, size_t height, size_t numLayers, unsigned int numLevels) : Image2DLayeredLod(Runtime::defaultAllocator(), format, width, height, numLayers, numLevels) {}

	public:

		//!	@brief		Return the specified level.
		Image2DLayered<void> level(size_t i) const { return Image2DLayered<void>(m_mipmaps[i]); }

		//!	@brief		Returns the number of layers.
		uint32_t numLayers() const { return m_extent.depth; }

		//!	@brief		Returns the height of the image.
		uint32_t height() const { return m_extent.height; }
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
		 *	@brief		Default constructor.
		 */
		Image2DLayeredLod() = default;


		/**
		 *	@brief		Constructs a 2D layered mipmapped image with default allocator.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	height - height of the image.
		 *	@param[in]	numLayers - Layers of the image, is clamped down to 1.
		 *	@param[in]	numLevels - Number of mipmap levels to allocated, is clamped to the range [1, 1 + floor(log2(max(width, height)))].
		 *	@throw		cudaError_t - In case of failure.
		 */
		explicit Image2DLayeredLod(size_t width, size_t height, size_t numLayers, unsigned int numLevels) : Image2DLayeredLod(Runtime::defaultAllocator(), width, height, numLayers, numLevels) {}


		/**
		 *	@brief		Constructs a 2D layered mipmapped image.
		 *	@param[in]	allocator - Pointer to the associated allocator.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	height - height of the image.
		 *	@param[in]	numLayers - Layers of the image, is clamped down to 1.
		 *	@param[in]	numLevels - Number of mipmap levels to allocated, is clamped to the range [1, 1 + floor(log2(max(width, height)))].
		 *	@throw		cudaError_t - In case of failure.
		 */
		explicit Image2DLayeredLod(std::shared_ptr<DeviceAllocator> allocator, size_t width, size_t height, size_t numLayers, unsigned int numLevels) : Image2DLayeredLod<void>(allocator, FormatOf<Type>::value, width, height, numLayers, numLevels) {}

	public:

		//!	@brief		Return the specified level.
		Image2DLayered<Type> level(size_t i) const { return Image2DLayered<Type>(m_mipmaps[i]); }

		//!	@brief		Returns the texel format of the image at compile time.
		static constexpr Format format() { return FormatOf<Type>::value; }
	};
}
