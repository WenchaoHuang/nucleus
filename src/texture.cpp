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

#include "image.h"
#include "logger.h"
#include "texture.h"
#include "image_cube.h"
#include <cuda_runtime_api.h>

NS_USING_NAMESPACE

/*********************************************************************************
*******************************    Validations    ********************************
*********************************************************************************/

static_assert(static_cast<int>(FilterMode::Linear) == cudaFilterModeLinear);
static_assert(static_cast<int>(FilterMode::Nearest) == cudaFilterModePoint);

static_assert(static_cast<int>(AddressMode::Wrap) == cudaAddressModeWrap);
static_assert(static_cast<int>(AddressMode::Clamp) == cudaAddressModeClamp);
static_assert(static_cast<int>(AddressMode::Mirror) == cudaAddressModeMirror);
static_assert(static_cast<int>(AddressMode::Border) == cudaAddressModeBorder);

/*********************************************************************************
*******************************    TextureBase    ********************************
*********************************************************************************/

TextureBase::TextureBase(std::shared_ptr<ImageBase> image, Sampler sampler, Format viewFormat) : m_hTexture(0)
{
	cudaResourceDesc resDesc = {};

	Format format = image->format();

	if (image->numLevels() == 0)
	{
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = image->data().handle;

		if (!std::dynamic_pointer_cast<ImageCube<void>>(image) &&
			!std::dynamic_pointer_cast<ImageCubeLayered<void>>(image))
		{
			sampler.seamlessCubemap = false;	//	avoid error
		}
	}
	else
	{
		resDesc.resType = cudaResourceTypeMipmappedArray;
		resDesc.res.mipmap.mipmap = image->handle();

		if (!std::dynamic_pointer_cast<ImageCubeLod<void>>(image) &&
			!std::dynamic_pointer_cast<ImageCubeLayeredLod<void>>(image))
		{
			sampler.seamlessCubemap = false;	//	avoid error
		}
	}

	if ((viewFormat != Format::Float) && (viewFormat != Format::Float2) && (viewFormat != Format::Float4))
	{
		sampler.mipmapFilterMode = FilterMode::Nearest;
		sampler.filterMode = FilterMode::Nearest;
	}

	cudaTextureDesc								textureDesc = {};
	textureDesc.addressMode[0]					= static_cast<cudaTextureAddressMode>(sampler.addressMode[0]);
	textureDesc.addressMode[1]					= static_cast<cudaTextureAddressMode>(sampler.addressMode[1]);
	textureDesc.addressMode[2]					= static_cast<cudaTextureAddressMode>(sampler.addressMode[2]);
	textureDesc.filterMode						= static_cast<cudaTextureFilterMode>(sampler.filterMode);
	textureDesc.readMode						= (format == viewFormat) ? cudaReadModeElementType : cudaReadModeNormalizedFloat;
	textureDesc.sRGB							= sampler.sRGB;
	textureDesc.borderColor[0]					= sampler.borderColor[0];
	textureDesc.borderColor[1]					= sampler.borderColor[1];
	textureDesc.borderColor[2]					= sampler.borderColor[2];
	textureDesc.borderColor[3]					= sampler.borderColor[3];
	textureDesc.normalizedCoords				= sampler.normalizedCoords;
	textureDesc.maxAnisotropy					= sampler.maxAnisotropy;
	textureDesc.mipmapFilterMode				= static_cast<cudaTextureFilterMode>(sampler.mipmapFilterMode);
	textureDesc.mipmapLevelBias					= sampler.mipmapLevelBias;
	textureDesc.minMipmapLevelClamp				= sampler.minMipmapLevelClamp;
	textureDesc.maxMipmapLevelClamp				= sampler.maxMipmapLevelClamp;
	textureDesc.disableTrilinearOptimization	= sampler.disableTrilinearOptimization;
	textureDesc.seamlessCubemap					= sampler.seamlessCubemap;

	cudaError_t err = cudaCreateTextureObject(&m_hTexture, &resDesc, &textureDesc, nullptr);

	if (err == cudaSuccess)
	{
		m_sampler = sampler;

		m_image = image;
	}
	else
	{
		NS_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();

		throw err;
	}
}


TextureBase::~TextureBase()
{
	if (m_hTexture != 0)
	{
		cudaError_t err = cudaDestroyTextureObject(m_hTexture);

		if (err != cudaSuccess)
		{
			NS_ERROR_LOG("%s.", cudaGetErrorString(err));

			cudaGetLastError();
		}
	}
}
