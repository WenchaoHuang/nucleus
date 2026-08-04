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
		 *	@brief		Default constructor.
		 */
		Image1D() = default;


		/**
		 *	@brief		Constructs a 1D image.
		 *	@param[in]	allocator - Pointer to the associated allocator.
		 *	@param[in]	format - Texel format of the image.
		 *	@param[in]	width - Width of the image.
		 */
		NS_API explicit Image1D(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width);


		/**
		 *	@brief		Constructs a 1D image with default allocator.
		 *	@param[in]	format - Texel format of the image.
		 *	@param[in]	width - Width of the image.
		 */
		explicit Image1D(Format format, size_t width) : Image1D(Runtime::defaultAllocator(), format, width) {}

	protected:

		//!	@brief		Copy constructor from `Image`.
		explicit Image1D(const Image & image) : Image(image) {}
	};

	/*****************************************************************************
	****************************    Image1D<Type>    *****************************
	*****************************************************************************/

	/**
	 *	@brief		Represents a 1D texture memory with specified storing type.
	 */
	template<typename Type> class Image1D : public Image1D<void>
	{
		friend class Image1DLod<Type>;

	public:

		/**
		 *	@brief		Default constructor.
		 */
		Image1D() = default;


		/**
		 *	@brief		Constructs a 1D image with default allocator.
		 * 	@param[in]	width - Width of the image.
		 */
		explicit Image1D(size_t width) : Image1D(Runtime::defaultAllocator(), width) {}


		/**
		 *	@brief		Constructs a 1D image.
		 *	@param[in]	allocator - Pointer to the associated allocator.
		 *	@param[in]	width - Width of the image.
		 */
		explicit Image1D(std::shared_ptr<DeviceAllocator> allocator, size_t width) : Image1D<void>(allocator, FormatOf<Type>::value, width) {}

	protected:

		//!	@brief		Copy constructor from `Image`.
		explicit Image1D(const Image & image) : Image1D<void>(image) {}

	public:

		//!	@brief		Returns accessor to the data.
		ImageAccessor<Type> data() const { return ImageAccessor<Type>{ m_hImage }; }

		//!	@brief		Returns the texel format of the image at compile time.
		static constexpr Format format() { return FormatOf<Type>::value; }
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
		 *	@brief		Default constructor.
		 */
		Image1DLayered() = default;


		/**
		 *	@brief		Constructs a 1D layered image.
		 *	@param[in]	allocator - Pointer to the associated allocator.
		 *	@param[in]	format - Texel format of the image.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	numLayers - Layers of the image, is clamped down to 1.
		 */
		NS_API explicit Image1DLayered(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, size_t numLayers);


		/**
		 *	@brief		Constructs a 1D layered image with default allocator.
		 *	@param[in]	format - Texel format of the image.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	numLayers - Layers of the image, is clamped down to 1.
		 */
		explicit Image1DLayered(Format format, size_t width, size_t numLayers) : Image1DLayered(Runtime::defaultAllocator(), format, width, numLayers) {}

	protected:

		//!	@brief		Copy constructor from `Image`.
		explicit Image1DLayered(const Image & image) : Image(image) {}

	public:

		//!	@brief		Returns the number of layers.
		uint32_t numLayers() const { return m_extent.depth; }
	};

	/*****************************************************************************
	*************************    Image1DLayered<Type>    *************************
	*****************************************************************************/

	/**
	 *	@brief		Represents a 1D layered texture memory with specified storing type.
	 */
	template<typename Type> class Image1DLayered : public Image1DLayered<void>
	{
		friend class Image1DLayeredLod<Type>;

	public:

		/**
		 *	@brief		Default constructor.
		 */
		Image1DLayered() = default;


		/**
		 *	@brief		Constructs a 1D layered image with default allocator.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	numLayers - Number of layers.
		 */
		explicit Image1DLayered(size_t width, size_t numLayers) : Image1DLayered(Runtime::defaultAllocator(), width, numLayers) {}


		/**
		 *	@brief		Constructs a 1D layered image.
		 *	@param[in]	allocator - Pointer to the associated allocator.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	numLayers - Number of layers.
		 */
		explicit Image1DLayered(std::shared_ptr<DeviceAllocator> allocator, size_t width, size_t numLayers) : Image1DLayered<void>(allocator, FormatOf<Type>::value, width, numLayers) {}
	
	protected:

		//!	@brief		Copy constructor from `Image`.
		explicit Image1DLayered(const Image & image) : Image1DLayered<void>(image) {}

	public:

		//!	@brief		Returns accessor to the data.
		ImageAccessor<Type> data() const { return ImageAccessor<Type>{ m_hImage }; }

		//!	@brief		Returns the texel format of the image at compile time.
		static constexpr Format format() { return FormatOf<Type>::value; }
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
		 *	@brief		Default constructor.
		 */
		Image1DLod() = default;

		/**
		 *	@brief		Constructs a 1D mipmapped image.
		 *	@param[in]	allocator - Pointer to the associated allocator.
		 *	@param[in]	format - Texel format of the image.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	numLevels - Number of mipmap levels to allocated, is clamped to the range [1, 1 + floor(log2(width))].
		 *	@throw		cudaError_t - In case of failure.
		 */
		NS_API explicit Image1DLod(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, unsigned int numLevels);


		/**
		 *	@brief		Constructs a 1D mipmapped image with default allocator.
		 *	@param[in]	format - Texel format of the image.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	numLevels - Number of mipmap levels to allocated, is clamped to the range [1, 1 + floor(log2(width))].
		 * 	@throw		cudaError_t - In case of failure.
		 */
		explicit Image1DLod(Format format, size_t width, unsigned int numLevels) : Image1DLod(Runtime::defaultAllocator(), format, width, numLevels) {}

	public:

		//!	@brief		Return the specified level.
		Image1D<void> level(size_t i) const { return Image1D<void>(m_mipmaps[i]); }
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
		 *	@brief		Default constructor.
		 */
		Image1DLod() = default;


		/**
		 *	@brief		Constructs a 1D mipmapped image with default allocator.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	numLevels - Number of mipmap levels to allocated, is clamped to the range [1, 1 + floor(log2(width))].
		 *	@throw		cudaError_t - In case of failure.
		 */
		explicit Image1DLod(size_t width, unsigned int numLevels) : Image1DLod(Runtime::defaultAllocator(), width, numLevels) {}


		/**
		 *	@brief		Constructs a 1D mipmapped image.
		 *	@param[in]	allocator - Pointer to the associated allocator.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	numLevels - Number of mipmap levels to allocated, is clamped to the range [1, 1 + floor(log2(width))].
		 *	@throw		cudaError_t - In case of failure.
		 */
		explicit Image1DLod(std::shared_ptr<DeviceAllocator> allocator, size_t width, unsigned int numLevels) : Image1DLod<void>(allocator, FormatOf<Type>::value, width, numLevels) {}

	public:

		//!	@brief		Return the specified level.
		Image1D<Type> level(size_t i) const { return Image1D<Type>(m_mipmaps[i]); }

		//!	@brief		Returns the texel format of the image at compile time.
		static constexpr Format format() { return FormatOf<Type>::value; }
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
		 *	@brief		Default constructor.
		 */
		Image1DLayeredLod() = default;


		/**
		 *	@brief		Constructs a 1D layered mipmapped image.
		 *	@param[in]	allocator - Pointer to the associated allocator.
		 *	@param[in]	format - Texel format of the image.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	numLayers - Layers of the image, is clamped down to 1.
		 *	@param[in]	numLevels - Number of mipmap levels to allocated, is clamped to the range [1, 1 + floor(log2(width))].
		 *	@throw		cudaError_t - In case of failure.
		 */
		NS_API explicit Image1DLayeredLod(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, size_t numLayers, unsigned int numLevels);


		/**
		 *	@brief		Constructs a 1D layered mipmapped image with default allocator.
		 *	@param[in]	format - Texel format of the image.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	numLayers - Layers of the image, is clamped down to 1.
		 *	@param[in]	numLevels - Number of mipmap levels to allocated, is clamped to the range [1, 1 + floor(log2(width))].
		 *	@throw		cudaError_t - In case of failure.
		 */
		explicit Image1DLayeredLod(Format format, size_t width, size_t numLayers, unsigned int numLevels) : Image1DLayeredLod(Runtime::defaultAllocator(), format, width, numLayers, numLevels) {}

	public:

		//!	@brief		Return the specified level.
		Image1DLayered<void> level(size_t i) const { return Image1DLayered<void>(m_mipmaps[i]); }

		//!	@brief		Returns the number of layers.
		uint32_t numLayers() const { return m_extent.depth; }
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
		 *	@brief		Default constructor.
		 */
		Image1DLayeredLod() = default;


		/**
		 *	@brief		Constructs a 1D layered mipmapped image with default allocator.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	numLayers - Layers of the image, is clamped down to 1.
		 *	@param[in]	numLevels - Number of mipmap levels to allocated, is clamped to the range [1, 1 + floor(log2(width))].
		 *	@throw		cudaError_t - In case of failure.
		 */
		explicit Image1DLayeredLod(size_t width, size_t numLayers, unsigned int numLevels) : Image1DLayeredLod(Runtime::defaultAllocator(), width, numLayers, numLevels) {}


		/**
		 *	@brief		Constructs a 1D layered mipmapped image.
		 *	@param[in]	allocator - Pointer to the associated allocator.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	numLayers - Layers of the image, is clamped down to 1.
		 *	@param[in]	numLevels - Number of mipmap levels to allocated, is clamped to the range [1, 1 + floor(log2(width))].
		 *	@throw		cudaError_t - In case of failure.
		 */
		explicit Image1DLayeredLod(std::shared_ptr<DeviceAllocator> allocator, size_t width, size_t numLayers, unsigned int numLevels) : Image1DLayeredLod<void>(allocator, FormatOf<Type>::value, width, numLayers, numLevels) {}

	public:

		//!	@brief		Return the specified level.
		Image1DLayered<Type> level(size_t i) const { return Image1DLayered<Type>(m_mipmaps[i]); }

		//!	@brief		Returns the texel format of the image at compile time.
		static constexpr Format format() { return FormatOf<Type>::value; }
	};
}
