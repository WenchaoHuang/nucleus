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

		//!	@brief		RAII object, which manages the underlying CUDA array or mipmapped array.
		class Resource;


		/**
		 *	@brief		Default constructor.
		 */
		ImageBase() : m_format(Format::Undefined), m_extent({ 0, 0, 0 }) {}


		/**
		 *	@brief		Constructor
		 *	@param[in]	resource - Shared pointer to the underlying resource.
		 *	@param[in]	format - Texel format of the image.
		 *	@param[in]	extent - Dimensions of the image.
		 */
		NS_API explicit ImageBase(std::shared_ptr<Resource> resource, Format format, Extent extent);


		/**
		 *	@brief		Virtual destructor.
		 */
		virtual ~ImageBase() noexcept {}

	public:

		//!	@brief		Returns pointer to the allocator associated with.
		NS_API const std::shared_ptr<DeviceAllocator> & allocator() const;

		//!	@brief		Tests if the image is valid (non-null).
		operator bool() const { return m_resource != nullptr; }

		//!	@brief		Tests if the image is empty (null).
		bool empty() const { return m_resource == nullptr; }

		//!	@brief		Retruns the width of the image.
		uint32_t width() const { return m_extent.width; }

		//!	@brief		Returns the texel format of the image.
		Format format() const { return m_format; }

	protected:

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
		friend class ImageLod;

	protected:

		/**
		 *	@brief		Default constructor.
		 */
		Image() : m_hImage(nullptr) {}


		/**
		 *	@brief		Constructs a image.
		 *	@param[in]	allocator - Pointer to the associated allocator.
		 *	@param[in]	format - Texel format of the image.
		 *	@param[in]	extent - Dimensions of the image.
		 *	@param[in]	flags - Flags for image creation (interanl use).
		 *	@throw		cudaError_t - In case of failure.
		 */
		NS_API explicit Image(std::shared_ptr<DeviceAllocator> allocator, Format format, Extent extent, int flags);


		/**
		 *	@brief		Constructs from ImageLod.
		 *	@param[in]	resource - Shared pointer to the underlying resource.
		 *	@param[in]	level - Level of detail to be used.
		 */
		NS_API explicit Image(std::shared_ptr<Resource> resource, unsigned int level);

	public:

		//!	@brief		Returns accessor to the data.
		ImageAccessor<void> data() const { return ImageAccessor<void>{ m_hImage }; }

		//!	@brief		Returns CUDA type of this object.
		cudaArray_t handle() const { return m_hImage; }

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
		 *	@brief		Default constructor.
		 */
		ImageLod() : m_hImageLod(nullptr), m_numLevels(0) {}


		/**
		 *	@brief		Constructs a image with level of details.
		 *	@param[in]	allocator - Pointer to the associated allocator.
		 *	@param[in]	format - Texel format of the image.
		 *	@param[in]	extent - Dimensions of the image.
		 *	@param[in]	numLevels - Number of mipmap levels to allocated.
		 *	@param[in]	flags - Flags for image creation (interanl use).
		 *	@throw		cudaError_t - In case of failure.
		 */
		NS_API explicit ImageLod(std::shared_ptr<DeviceAllocator> allocator, Format format, Extent extent, unsigned int numLevels, int flags);

	public:

		//!	@brief		Returns reference to the mipmap levels.
		const std::vector<Image> & mipmaps() const { return m_mipmaps; }

		//!	@brief		Return reference to the specified level.
		const Image & level(size_t i) const { return m_mipmaps[i]; }

		//!	@brief		Returns CUDA type of this object.
		cudaMipmappedArray_t handle() const { return m_hImageLod; }

		//!	@brief		Returns the number of mipmap levels.
		unsigned int numLevels() const { return m_numLevels; }

	protected:

		std::vector<Image>			m_mipmaps;
		cudaMipmappedArray_t		m_hImageLod;
		unsigned int				m_numLevels;
	};
}
