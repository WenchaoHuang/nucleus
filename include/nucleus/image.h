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
#include "format.h"
#include "host_types.h"
#include <variant>
#include <vector>

namespace NS_NAMESPACE
{
	//!	@brief		Represents the dimensions of a 3D object (width, height, depth).
	struct Extent { unsigned int width, height, depth; };

	/*****************************************************************************
	******************************    ImageBase    *******************************
	*****************************************************************************/

	/**
	 *	@brief		Base class of `Image` and `ImageLod`
	 */
	class ImageBase
	{

	protected:

		/**
		 *	@brief		Constructor
		 *	@param[in]	format - Texel format of the image.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	height - height of the image.
		 *	@param[in]	depth - Depth of the image.
		 */
		NS_API explicit ImageBase(Format format, size_t width, size_t height, size_t depth);


		/**
		 *	@brief		Virtual destructor.
		 */
		virtual ~ImageBase() noexcept {}

	public:

		//!	@brief		Returns the texel format of the image.
		Format format() const { return m_format; }

		//!	@brief		Retruns the width of the image.
		uint32_t width() const { return m_extent.width; }

		//!	@brief		Returns pointer to the allocator associated with.
		NS_API const std::shared_ptr<DeviceAllocator> & allocator() const;

	protected:

		class Resource;
		std::shared_ptr<Resource>		m_resource;
		Format							m_format;
		Extent							m_extent;
	};

	/*****************************************************************************
	********************************    Image    *********************************
	*****************************************************************************/

	/**
	 *	@brief		Base class represents a arbitrary texture memory.
	 *	@note		Texture memory are opaque memory layouts optimized for texture fetching.
	 *	@note		Nucleus-created images always enable CUDA surface load/store support. This
	 *				removes a creation-time API choice so an image can be bound to a Surface
	 *				when needed; the additional capability has negligible performance impact.
	 *	@see		class `ImageBase`
	 */
	class Image : public ImageBase
	{
		
	protected:

		/**
		 *	@brief		Constructs a image.
		 *	@param[in]	allocator - Pointer to the associated allocator.
		 *	@param[in]	format - Texel format of the image.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	height - height of the image.
		 *	@param[in]	depth - Depth of the image.
		 *	@param[in]	flags - Flags for image creation (interanl use).
		 *	@throw		cudaError_t - In case of failure.
		 */
		NS_API explicit Image(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, size_t height, size_t depth, int flags);


		/**
		 *	@brief		Constructs from ImageLod.
		 *	@param[in]	hImage - Handle of texture memory (from cudaMipmappedArray_t).
		 *	@param[in]	format - Texel format of the image.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	height - height of the image.
		 *	@param[in]	depth - Depth of the image.
		 */
		NS_API explicit Image(cudaArray_t hImage, Format format, size_t width, size_t height, size_t depth);

	public:

		//!	@brief		Returns accessor to the data.
		ImageAccessor<void> data() const { return ImageAccessor<void>{ m_hImage }; }

	protected:
        
        cudaArray_t		m_hImage;
	};

	/*****************************************************************************
	*******************************    ImageLod    *******************************
	*****************************************************************************/

	/**
	 *	@brief		Base class represents a arbitrary mipmapped texture memory.
	 *  @note		Texture memory are opaque memory layouts optimized for texture fetching.
	 *	@note		Nucleus-created images always enable CUDA surface load/store support. This
	 *				removes a creation-time API choice so an image can be bound to a Surface
	 *				when needed; the additional capability has negligible performance impact.
	 *	@see		class `ImageBase` and `Image`
	 */
	class ImageLod : public ImageBase
	{

	protected:

		/**
		 *	@brief		Constructs a image with level of details.
		 *	@param[in]	allocator - Pointer to the associated allocator.
		 *	@param[in]	format - Texel format of the image.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	height - height of the image.
		 *	@param[in]	depth - Depth of the image.
		 *	@param[in]	numLevels - Number of mipmap levels to allocated.
		 *	@param[in]	flags - Flags for image creation (interanl use).
		 *	@throw		cudaError_t - In case of failure.
		 */
		NS_API explicit ImageLod(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, size_t height, size_t depth, unsigned int numLevels, int flags);

	public:

		//!	@brief		Returns CUDA type of this object.
		cudaMipmappedArray_t handle() const { return m_hImageLod; }

		//!	@brief		Returns the number of mipmap levels.
		unsigned int numLevels() const { return m_numLevels; }

	protected:

		cudaMipmappedArray_t		m_hImageLod;
		unsigned int				m_numLevels;
	};
}