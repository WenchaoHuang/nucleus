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

#include "logger.h"
#include "image_1d.h"
#include "image_2d.h"
#include "image_3d.h"
#include "allocator.h"
#include "image_cube.h"

#include <cmath>
#include <algorithm>
#include <cuda_runtime_api.h>

NS_USING_NAMESPACE

/*********************************************************************************
*******************************    make_extent    ********************************
*********************************************************************************/

//!	@brief		Constructs an `Extent` object from width, height, and depth.
static inline Extent make_extent(size_t width, size_t height, size_t depth)
{
	return Extent{ static_cast<unsigned int>(width), static_cast<unsigned int>(height), static_cast<unsigned int>(depth) };
}

/*********************************************************************************
***************************    ImageBase::Resource    ****************************
*********************************************************************************/

/**
 *	@brief		RAII class for `ImageBase`, which manages the underlying CUDA array or mipmapped array.
 */
class ImageBase::Resource
{
	NS_NONCOPYABLE(Resource)

public:

	/**
	 *	@brief		Constructs a image.
	 *	@param[in]	allocator - Pointer to the associated allocator.
	 *	@param[in]	format - Texel format of the image.
	 *	@param[in]	extent - Dimensions of the image.
	 *	@param[in]	flags - Flags for image creation (interanl use).
	 *	@throw		ErrorCode - In case of failure.
	 */
	explicit Resource(std::shared_ptr<DeviceAllocator> allocator, Format format, Extent extent, int flags)
		: handle(allocator->allocate_texture_memory(format, extent.width, extent.height, extent.depth, flags | cudaArraySurfaceLoadStore)),
		allocator(allocator), format(format), extent(extent) {}


	/**
	 *	@brief		Constructs a image with level of details.
	 *	@param[in]	allocator - Pointer to the associated allocator.
	 *	@param[in]	format - Texel format of the image.
	 *	@param[in]	extent - Dimensions of the image.
	 *	@param[in]	num_levels - Number of mipmap levels to allocated.
	 *	@param[in]	flags - Flags for image creation (interanl use).
	 *	@throw		ErrorCode - In case of failure.
	 */
	explicit Resource(std::shared_ptr<DeviceAllocator> allocator, Format format, Extent extent, unsigned int num_levels, int flags)
		: handle(allocator->allocate_mipmap_texture_memory(format, extent.width, extent.height, extent.depth, num_levels, flags | cudaArraySurfaceLoadStore)),
		allocator(allocator), format(format), extent(extent)
	{
		try
		{
			this->mipmaps.resize(num_levels);

			auto image_lod = std::get<cudaMipmappedArray_t>(handle);

			for (unsigned int i = 0; i < num_levels; i++)
			{
				cudaError_t err = cudaGetMipmappedArrayLevel(&mipmaps[i].first, image_lod, i);

				if (err == cudaSuccess)
				{
					uint32_t				_flags = 0;
					cudaExtent				_extent = {};
					cudaChannelFormatDesc	channel_desc = {};

					err = cudaArrayGetInfo(&channel_desc, &_extent, &_flags, mipmaps[i].first);

					if (err != cudaSuccess) throw static_cast<ErrorCode>(err);

					mipmaps[i].second.height = static_cast<unsigned int>(_extent.height);
					mipmaps[i].second.width = static_cast<unsigned int>(_extent.width);
					mipmaps[i].second.depth = static_cast<unsigned int>(_extent.depth);
				}
				else
				{
					throw static_cast<ErrorCode>(err);
				}
			}
		}
		catch (...)
		{
			allocator->deallocate_mipmap_texture_memory(std::get<cudaMipmappedArray_t>(handle));

			throw;
		}
	}


	/**
	 *	@brief		Destroys the image storage.
	 */
	~Resource()
	{
		if (allocator != nullptr)
		{
			if (handle.index() == 0)
			{
				allocator->deallocate_texture_memory(std::get<cudaArray_t>(handle));
			}
			else if (handle.index() == 1)
			{
				allocator->deallocate_mipmap_texture_memory(std::get<cudaMipmappedArray_t>(handle));
			}
		}
	}

public:

	std::vector<std::pair<cudaArray_t, Extent>>					mipmaps;
	const std::variant<cudaArray_t, cudaMipmappedArray_t>		handle;
	const std::shared_ptr<DeviceAllocator>						allocator;
	const Format												format;
	const Extent												extent;
};

/*********************************************************************************
********************************    ImageBase    *********************************
*********************************************************************************/

ImageBase::ImageBase(std::shared_ptr<Resource> resource, Format format, Extent extent) : resource_(std::move(resource)), format_(format), extent_(extent)
{

}


const std::shared_ptr<DeviceAllocator> & ImageBase::allocator() const
{
	static const std::shared_ptr<DeviceAllocator> null_allocator = nullptr;

	return resource_ ? resource_->allocator : null_allocator;
}

/*********************************************************************************
**********************************    Image    ***********************************
*********************************************************************************/

Image::Image(std::shared_ptr<DeviceAllocator> allocator, Format format, Extent extent, int flags)
	: ImageBase(std::make_shared<Resource>(std::move(allocator), format, extent, flags), format, extent), image_(std::get<cudaArray_t>(resource_->handle))
{
	
}


Image::Image(std::shared_ptr<Resource> resource, unsigned int level) : ImageBase(resource, resource->format, resource->mipmaps[level].second), image_(resource->mipmaps[level].first)
{
	NS_ASSERT(level < resource_->mipmaps.size());
}

/*********************************************************************************
*********************************    ImageLod    *********************************
*********************************************************************************/

ImageLod::ImageLod(std::shared_ptr<DeviceAllocator> allocator, Format format, Extent extent, unsigned int num_levels, int flags)
	: ImageBase(std::make_shared<Resource>(std::move(allocator), format, extent, num_levels, flags), format, extent), num_levels_(num_levels), image_lod_(std::get<cudaMipmappedArray_t>(resource_->handle))
{
	mipmaps_.reserve(resource_->mipmaps.size());

	for (unsigned int level = 0; level < resource_->mipmaps.size(); level++)
	{
		mipmaps_.emplace_back(Image(resource_, level));
	}
}

/*********************************************************************************
*********************************    Image1D    **********************************
*********************************************************************************/

/**
 *	@details	A 1D array is allocated if the height and depth extents are both zero.
 *	@details	A 1D layered CUDA array is allocated if only the height extent is zero and the cudaArrayLayered flag is set.
 *				Each layer is a 1D array. The number of layers is determined by the depth extent.
 */
Image1D<void>::Image1D(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width)
	: Image(std::move(allocator), format, make_extent(width, 0, 0), cudaArrayDefault)
{
	NS_ASSERT(width > 0);
}


Image1DLayered<void>::Image1DLayered(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, size_t num_layers)
	: Image(std::move(allocator), format, make_extent(width, 0, std::max<size_t>(1, num_layers)), cudaArrayLayered)
{
	NS_ASSERT(width > 0);
}


Image1DLod<void>::Image1DLod(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, unsigned int num_levels)
	: ImageLod(allocator, format, make_extent(width, 0, 0), std::clamp(num_levels, 1u, 1u + static_cast<unsigned int>(std::floor(std::log2(width)))), cudaArrayDefault)
{
	NS_ASSERT(width > 0);
}


Image1DLayeredLod<void>::Image1DLayeredLod(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, size_t num_layers, unsigned int num_levels)
	: ImageLod(std::move(allocator), format, make_extent(width, 0, std::max<size_t>(1, num_layers)), std::clamp(num_levels, 1u, 1u + static_cast<uint32_t>(std::floor(std::log2(width)))), cudaArrayLayered)
{
	NS_ASSERT(width > 0);
}

/*********************************************************************************
*********************************    Image2D    **********************************
*********************************************************************************/

/**
 *	@details	A 2D array is allocated if only the depth extent is zero.
 *	@details	A 2D layered CUDA array is allocated if all three extents are non-zero and the cudaArrayLayered flag is set.
 *				Each layer is a 2D array. The number of layers is determined by the depth extent.
 */
Image2D<void>::Image2D(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, size_t height)
	: Image(std::move(allocator), format, make_extent(width, height, 0), cudaArrayDefault)
{
	NS_ASSERT(width * height > 0);
}


Image2DLayered<void>::Image2DLayered(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, size_t height, size_t num_layers)
	: Image(std::move(allocator), format, make_extent(width, height, std::max<size_t>(1, num_layers)), cudaArrayLayered)
{
	NS_ASSERT(width * height > 0);
}


Image2DLod<void>::Image2DLod(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, size_t height, unsigned int num_levels)
	: ImageLod(std::move(allocator), format, make_extent(width, height, 0), std::clamp(num_levels, 1u, 1u + static_cast<uint32_t>(std::floor(std::log2(std::max(width, height))))), cudaArrayDefault)
{
	NS_ASSERT(width * height > 0);
}


Image2DLayeredLod<void>::Image2DLayeredLod(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, size_t height, size_t num_layers, unsigned int num_levels)
	: ImageLod(std::move(allocator), format, make_extent(width, height, std::max<size_t>(1, num_layers)), std::clamp(num_levels, 1u, 1u + static_cast<uint32_t>(std::floor(std::log2(std::max(width, height))))), cudaArrayLayered)
{
	NS_ASSERT(width * height > 0);
}

/*********************************************************************************
*********************************    Image3D    **********************************
*********************************************************************************/

/**
 *	@details	A 3D array is allocated if all three extents are non-zero.
 */
Image3D<void>::Image3D(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, size_t height, size_t depth)
	: Image(std::move(allocator), format, make_extent(width, height, depth), cudaArrayDefault)
{
	NS_ASSERT(width * height * depth > 0);
}


Image3DLod<void>::Image3DLod(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, size_t height, size_t depth, unsigned int num_levels)
	: ImageLod(std::move(allocator), format, make_extent(width, height, depth), std::clamp(num_levels, 1u, 1u + static_cast<uint32_t>(std::floor(std::log2(std::max(std::max(width, height), depth))))), cudaArrayDefault)
{
	NS_ASSERT(width * height * depth > 0);
}

/*********************************************************************************
********************************    ImageCube    *********************************
*********************************************************************************/

/**
 *	@details	A cubemap CUDA array is allocated if all three extents are non-zero and the cudaArrayCubemap flag is set.
 *				Width must be equal to height, and depth must be six. A cubemap is a special type of 2D layered CUDA array, where the six layers represent the six faces of a cube.
 *				The order of the six layers in memory is the same as that listed in ::cudaGraphicsCubeFace.
 *	@details	A cubemap layered CUDA array is allocated if all three extents are non-zero, and both, cudaArrayCubemap and cudaArrayLayered flags are set.
 *				Width must be equal to height, and depth must be a multiple of six. A cubemap layered CUDA array is a special type of 2D layered CUDA array that consists of a collection of cubemaps.
 *				The first six layers represent the first cubemap, the next six layers form the second cubemap, and so on.
 */
ImageCube<void>::ImageCube(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width)
	: Image(std::move(allocator), format, make_extent(width, width, 6), cudaArrayCubemap)
{
	NS_ASSERT(width > 0);
}


ImageCubeLayered<void>::ImageCubeLayered(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, size_t num_layers)
	: Image(std::move(allocator), format, make_extent(width, width, 6 * std::max<size_t>(1, num_layers)), cudaArrayCubemap | cudaArrayLayered)
{
	NS_ASSERT(width > 0);
}


ImageCubeLod<void>::ImageCubeLod(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, unsigned int num_levels)
	: ImageLod(std::move(allocator), format, make_extent(width, width, 6), std::clamp(num_levels, 1u, 1u + static_cast<uint32_t>(std::floor(std::log2(width)))), cudaArrayCubemap)
{
	NS_ASSERT(width > 0);
}


ImageCubeLayeredLod<void>::ImageCubeLayeredLod(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, size_t num_layers, unsigned int num_levels)
	: ImageLod(std::move(allocator), format, make_extent(width, width, 6 * std::max<size_t>(1, num_layers)), std::clamp(num_levels, 1u, 1u + static_cast<uint32_t>(std::floor(std::log2(width)))), cudaArrayCubemap | cudaArrayLayered)
{
	NS_ASSERT(width > 0);
}
