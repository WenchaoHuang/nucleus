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
#include <variant>
#include <algorithm>
#include <cuda_runtime_api.h>

NS_USING_NAMESPACE

/*********************************************************************************
****************************    ImageBase::Storage    ****************************
*********************************************************************************/

/**
 *	@brief		Storage class for ImageBase, which manages the underlying CUDA array or mipmapped array.
 */
class ImageBase::Storage
{

public:

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
	explicit Storage(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, size_t height, size_t depth, int flags)
		: hImage(allocator->allocateTextureMemory(format, width, height, depth, flags | cudaArraySurfaceLoadStore)), numLevels(0),
		allocator(allocator), height(static_cast<uint32_t>(height)), width(static_cast<uint32_t>(width)), depth(static_cast<uint32_t>(depth)), format(format) {}


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
	explicit Storage(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, size_t height, size_t depth, unsigned int numLevels, int flags)
		: hImage(allocator->allocateMipmapTextureMemory(format, width, height, depth, numLevels, flags | cudaArraySurfaceLoadStore)), numLevels(numLevels),
		allocator(allocator), height(static_cast<uint32_t>(height)), width(static_cast<uint32_t>(width)), depth(static_cast<uint32_t>(depth)), format(format) {}


	/**
	 *	@brief		Destroys the image storage.
	 */
	~Storage()
	{
		if (allocator != nullptr)
		{
			if (hImage.index() == 0)
			{
				allocator->deallocateTextureMemory(std::get<cudaArray_t>(hImage));
			}
			else if (hImage.index() == 1)
			{
				allocator->deallocateMipmapTextureMemory(std::get<cudaMipmappedArray_t>(hImage));
			}
		}
	}

public:

	const std::variant<cudaArray_t, cudaMipmappedArray_t>		hImage;
	const std::shared_ptr<DeviceAllocator>						allocator;
	const unsigned int											numLevels;
	const unsigned int											height;
	const unsigned int											width;
	const unsigned int											depth;
	const Format												format;
};

/*********************************************************************************
********************************    ImageBase    *********************************
*********************************************************************************/

ImageBase::ImageBase(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, size_t height, size_t depth)
	: m_allocator(allocator), m_format(format), m_width(static_cast<uint32_t>(width)), m_height(static_cast<uint32_t>(height)), m_depth(static_cast<uint32_t>(depth)),
	m_hImage(nullptr), m_hImageLod(nullptr), m_numLevels(0)
{

}


ImageBase::ImageBase(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, size_t height, size_t depth, int flags)
	: ImageBase(allocator, format, width, height, depth)
{
	m_storage = std::make_shared<Storage>(allocator, format, width, height, depth, flags | cudaArraySurfaceLoadStore);
	m_hImage = std::get<cudaArray_t>(m_storage->hImage);

	NS_ASSERT(allocator != nullptr);
}


ImageBase::ImageBase(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, size_t height, size_t depth, unsigned int numLevels, int flags)
	: ImageBase(allocator, format, width, height, depth)
{
	m_storage = std::make_shared<Storage>(allocator, format, width, height, depth, numLevels, flags | cudaArraySurfaceLoadStore);
	m_hImageLod = std::get<cudaMipmappedArray_t>(m_storage->hImage);
	m_numLevels = m_storage->numLevels;

	NS_ASSERT(allocator != 0);
}


cudaArray_t ImageBase::subresource(unsigned int level) const
{
	cudaArray_t hImage = m_hImage;

	if (m_hImageLod != nullptr)
	{
		NS_ASSERT(level < m_numLevels);

		cudaError_t err = cudaGetMipmappedArrayLevel(&hImage, m_hImageLod, level);

		if (err != cudaSuccess)
		{
			NS_ERROR_LOG("%s.", cudaGetErrorString(err));

			cudaGetLastError();

			throw err;
		}
	}
	else
	{
		NS_ASSERT(level == 0);
	}

	return hImage;
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
	: Image(allocator, format, width, 0, 0, cudaArrayDefault)
{
	NS_ASSERT(width > 0);
}


Image1DLayered<void>::Image1DLayered(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, size_t numLayers)
	: Image(allocator, format, width, 0, std::max<size_t>(1, numLayers), cudaArrayLayered)
{
	NS_ASSERT(width > 0);
}


Image1DLod<void>::Image1DLod(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, unsigned int numLevels)
	: ImageLod(allocator, format, width, 0, 0, std::clamp(numLevels, 1u, 1u + static_cast<uint32_t>(std::floor(std::log2(width)))), cudaArrayDefault)
{
	NS_ASSERT(width > 0);
}


Image1DLayeredLod<void>::Image1DLayeredLod(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, size_t numLayers, unsigned int numLevels)
	: ImageLod(allocator, format, width, 0, std::max<size_t>(1, numLayers), std::clamp(numLevels, 1u, 1u + static_cast<uint32_t>(std::floor(std::log2(width)))), cudaArrayLayered)
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
	: Image(allocator, format, width, height, 0, cudaArrayDefault)
{
	NS_ASSERT(width * height > 0);
}


Image2DLayered<void>::Image2DLayered(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, size_t height, size_t numLayers)
	: Image(allocator, format, width, height, std::max<size_t>(1, numLayers), cudaArrayLayered)
{
	NS_ASSERT(width * height > 0);
}


Image2DLod<void>::Image2DLod(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, size_t height, unsigned int numLevels)
	: ImageLod(allocator, format, width, height, 0, std::clamp(numLevels, 1u, 1u + static_cast<uint32_t>(std::floor(std::log2(std::max(width, height))))), cudaArrayDefault)
{
	NS_ASSERT(width * height > 0);
}


Image2DLayeredLod<void>::Image2DLayeredLod(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, size_t height, size_t numLayers, unsigned int numLevels)
	: ImageLod(allocator, format, width, height, std::max<size_t>(1, numLayers), std::clamp(numLevels, 1u, 1u + static_cast<uint32_t>(std::floor(std::log2(std::max(width, height))))), cudaArrayLayered)
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
	: Image(allocator, format, width, height, depth, cudaArrayDefault)
{
	NS_ASSERT(width * height * depth > 0);
}


Image3DLod<void>::Image3DLod(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, size_t height, size_t depth, unsigned int numLevels)
	: ImageLod(allocator, format, width, height, depth, std::clamp(numLevels, 1u, 1u + static_cast<uint32_t>(std::floor(std::log2(std::max(std::max(width, height), depth))))), cudaArrayDefault)
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
	: Image(allocator, format, width, width, 6, cudaArrayCubemap)
{
	NS_ASSERT(width > 0);
}


ImageCubeLayered<void>::ImageCubeLayered(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, size_t numLayers)
	: Image(allocator, format, width, width, 6 * std::max<size_t>(1, numLayers), cudaArrayCubemap | cudaArrayLayered)
{
	NS_ASSERT(width > 0);
}


ImageCubeLod<void>::ImageCubeLod(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, unsigned int numLevels)
	: ImageLod(allocator, format, width, width, 6, std::clamp(numLevels, 1u, 1u + static_cast<uint32_t>(std::floor(std::log2(width)))), cudaArrayCubemap)
{
	NS_ASSERT(width > 0);
}


ImageCubeLayeredLod<void>::ImageCubeLayeredLod(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, size_t numLayers, unsigned int numLevels)
	: ImageLod(allocator, format, width, width, 6 * std::max<size_t>(1, numLayers), std::clamp(numLevels, 1u, 1u + static_cast<uint32_t>(std::floor(std::log2(width)))), cudaArrayCubemap | cudaArrayLayered)
{
	NS_ASSERT(width > 0);
}
