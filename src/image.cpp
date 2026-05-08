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

#include <cuda_runtime_api.h>
#include <algorithm>
#include <cmath>
#include "logger.h"
#include "image_1d.h"
#include "image_2d.h"
#include "image_3d.h"
#include "allocator.h"
#include "image_cube.h"

NS_USING_NAMESPACE

/*********************************************************************************
********************************    ImageBase    *********************************
*********************************************************************************/

ImageBase::ImageBase(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, size_t height, size_t depth, int flags)
	: m_allocator(allocator), m_format(format), m_width(static_cast<uint32_t>(width)), m_height(static_cast<uint32_t>(height)),
	  m_depth(static_cast<uint32_t>(depth)), m_flags(flags)
{

}

/*********************************************************************************
**********************************    Image    ***********************************
*********************************************************************************/

Image::Image(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, size_t height, size_t depth, int flags)
	: ImageBase(allocator, format, width, height, depth, flags), m_hImage(allocator->allocateTextureMemory(format, width, height, depth, flags))
{
	NS_ASSERT(allocator != nullptr);
}


Image::Image(cudaArray_t hImage, Format format, size_t width, size_t height, size_t depth, int flags)
	: ImageBase(nullptr, format, width, height, depth, flags), m_hImage(hImage)
{
	NS_ASSERT(hImage != nullptr);
}


bool Image::isSurfaceLoadStoreSupported() const
{
	return (m_flags & cudaArraySurfaceLoadStore);
}


Image::~Image() noexcept
{
	if ((m_allocator != nullptr) && (m_hImage != nullptr))
	{
		m_allocator->deallocateTextureMemory(m_hImage);
	}
}

/*********************************************************************************
*********************************    ImageLod    *********************************
*********************************************************************************/

ImageLod::ImageLod(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, size_t height, size_t depth, unsigned int numLevels, int flags)
	: ImageBase(allocator, format, width, height, depth, flags), m_numLevels(numLevels),
	  m_hImageLod(allocator->allocateMipmapTextureMemory(format, width, height, depth, numLevels, flags))
{
	NS_ASSERT(allocator != 0);
}


ImageLod::~ImageLod()
{
	if ((m_allocator != nullptr) && (m_hImageLod != nullptr))
	{
		m_allocator->deallocateMipmapTextureMemory(m_hImageLod);
	}
}

/*********************************************************************************
****************************    NS_CREATE_MIPMAPS    *****************************
*********************************************************************************/

static std::vector<cudaArray_t> getMipmapHandles(cudaMipmappedArray_t hImageLod, unsigned int numLevels)
{
	std::vector<cudaArray_t> hImages(numLevels);

	for (unsigned int i = 0; i < numLevels; i++)
	{
		cudaError_t err = cudaGetMipmappedArrayLevel(hImages.data() + i, hImageLod, i);

		if (err != cudaSuccess)
		{
			NS_ERROR_LOG("%s.", cudaGetErrorString(err));

			cudaGetLastError();

			throw err;
		}
	}

	return hImages;
}

#define NS_CREATE_MIPMAPS(ImageType)										\
																			\
	auto mipmapHandles = getMipmapHandles(m_hImageLod, m_numLevels);		\
																			\
	m_mipmaps.resize(mipmapHandles.size());									\
																			\
	for (size_t i = 0; i < mipmapHandles.size(); i++)						\
	{																		\
		uint32_t				flags = 0;									\
		cudaExtent				extent = {};								\
		cudaChannelFormatDesc	channelDesc = {};							\
																			\
		cudaArrayGetInfo(&channelDesc, &extent, &flags, mipmapHandles[i]);	\
																			\
		m_mipmaps[i] = std::shared_ptr<ImageType>(new ImageType(mipmapHandles[i], format, extent.width, extent.height, extent.depth, flags));	\
	}

/*********************************************************************************
*********************************    Image1D    **********************************
*********************************************************************************/

/**
 *	@details	A 1D array is allocated if the height and depth extents are both zero.
 *	@details	A 1D layered CUDA array is allocated if only the height extent is zero and the cudaArrayLayered flag is set.
 *				Each layer is a 1D array. The number of layers is determined by the depth extent.
 */
Image1D<void>::Image1D(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, bool bSurfaceLoadStore)
	: Image(allocator, format, width, 0, 0, bSurfaceLoadStore ? cudaArraySurfaceLoadStore : cudaArrayDefault)
{
	NS_ASSERT(width > 0);
}


Image1DLayered<void>::Image1DLayered(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, size_t numLayers, bool bSurfaceLoadStore)
	: Image(allocator, format, width, 0, std::max<size_t>(1, numLayers), bSurfaceLoadStore ? (cudaArrayLayered | cudaArraySurfaceLoadStore) : cudaArrayLayered)
{
	NS_ASSERT(width > 0);
}


Image1DLod<void>::Image1DLod(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, unsigned int numLevels)
	: ImageLod(allocator, format, width, 0, 0, std::clamp(numLevels, 1u, 1u + static_cast<uint32_t>(std::floor(std::log2(width)))), cudaArrayDefault)
{
	NS_ASSERT(width > 0);

	NS_CREATE_MIPMAPS(Image1D<void>);
}


Image1DLayeredLod<void>::Image1DLayeredLod(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, size_t numLayers, unsigned int numLevels)
	: ImageLod(allocator, format, width, 0, std::max<size_t>(1, numLayers), std::clamp(numLevels, 1u, 1u + static_cast<uint32_t>(std::floor(std::log2(width)))), cudaArrayLayered)
{
	NS_ASSERT(width > 0);

	NS_CREATE_MIPMAPS(Image1DLayered<void>);
}

/*********************************************************************************
*********************************    Image2D    **********************************
*********************************************************************************/

/**
 *	@details	A 2D array is allocated if only the depth extent is zero.
 *	@details	A 2D layered CUDA array is allocated if all three extents are non-zero and the cudaArrayLayered flag is set.
 *				Each layer is a 2D array. The number of layers is determined by the depth extent.
 */
Image2D<void>::Image2D(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, size_t height, bool bSurfaceLoadStore)
	: Image(allocator, format, width, height, 0, bSurfaceLoadStore ? cudaArraySurfaceLoadStore : cudaArrayDefault)
{
	NS_ASSERT(width * height > 0);
}


Image2DLayered<void>::Image2DLayered(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, size_t height, size_t numLayers, bool bSurfaceLoadStore)
	: Image(allocator, format, width, height, std::max<size_t>(1, numLayers), bSurfaceLoadStore ? (cudaArrayLayered | cudaArraySurfaceLoadStore) : cudaArrayLayered)
{
	NS_ASSERT(width * height > 0);
}


Image2DLod<void>::Image2DLod(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, size_t height, unsigned int numLevels)
	: ImageLod(allocator, format, width, height, 0, std::clamp(numLevels, 1u, 1u + static_cast<uint32_t>(std::floor(std::log2(std::max(width, height))))), cudaArrayDefault)
{
	NS_ASSERT(width * height > 0);

	NS_CREATE_MIPMAPS(Image2D<void>);
}


Image2DLayeredLod<void>::Image2DLayeredLod(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, size_t height, size_t numLayers, unsigned int numLevels)
	: ImageLod(allocator, format, width, height, std::max<size_t>(1, numLayers), std::clamp(numLevels, 1u, 1u + static_cast<uint32_t>(std::floor(std::log2(std::max(width, height))))), cudaArrayLayered)
{
	NS_ASSERT(width * height > 0);

	NS_CREATE_MIPMAPS(Image2DLayered<void>);
}

/*********************************************************************************
*********************************    Image3D    **********************************
*********************************************************************************/

/**
 *	@details	A 3D array is allocated if all three extents are non-zero.
 */
Image3D<void>::Image3D(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, size_t height, size_t depth, bool bSurfaceLoadStore)
	: Image(allocator, format, width, height, depth, bSurfaceLoadStore ? cudaArraySurfaceLoadStore : cudaArrayDefault)
{
	NS_ASSERT(width * height * depth > 0);
}


Image3DLod<void>::Image3DLod(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, size_t height, size_t depth, unsigned int numLevels)
	: ImageLod(allocator, format, width, height, depth, std::clamp(numLevels, 1u, 1u + static_cast<uint32_t>(std::floor(std::log2(std::max(std::max(width, height), depth))))), cudaArrayDefault)
{
	NS_ASSERT(width * height * depth > 0);

	NS_CREATE_MIPMAPS(Image3D<void>);
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
ImageCube<void>::ImageCube(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, bool bSurfaceLoadStore)
	: Image(allocator, format, width, width, 6, bSurfaceLoadStore ? (cudaArrayCubemap | cudaArraySurfaceLoadStore) : cudaArrayCubemap)
{
	NS_ASSERT(width > 0);
}


ImageCubeLayered<void>::ImageCubeLayered(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, size_t numLayers, bool bSurfaceLoadStore)
	: Image(allocator, format, width, width, 6 * std::max<size_t>(1, numLayers), bSurfaceLoadStore ? (cudaArrayCubemap | cudaArrayLayered | cudaArraySurfaceLoadStore) : (cudaArrayCubemap | cudaArrayLayered))
{
	NS_ASSERT(width > 0);
}


ImageCubeLod<void>::ImageCubeLod(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, unsigned int numLevels)
	: ImageLod(allocator, format, width, width, 6, std::clamp(numLevels, 1u, 1u + static_cast<uint32_t>(std::floor(std::log2(width)))), cudaArrayCubemap)
{
	NS_ASSERT(width > 0);

	NS_CREATE_MIPMAPS(ImageCube<void>);
}


ImageCubeLayeredLod<void>::ImageCubeLayeredLod(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, size_t numLayers, unsigned int numLevels)
	: ImageLod(allocator, format, width, width, 6 * std::max<size_t>(1, numLayers), std::clamp(numLevels, 1u, 1u + static_cast<uint32_t>(std::floor(std::log2(width)))), cudaArrayCubemap | cudaArrayLayered)
{
	NS_ASSERT(width > 0);

	NS_CREATE_MIPMAPS(ImageCubeLayered<void>);
}