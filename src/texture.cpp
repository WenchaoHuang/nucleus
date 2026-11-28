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

TextureBase::TextureBase(std::shared_ptr<ImageBase> source_image, Sampler sampler, Format view_format) : texture_(0)
{
	auto image = std::dynamic_pointer_cast<Image>(source_image);
	auto image_lod = std::dynamic_pointer_cast<ImageLod>(source_image);

	NS_ASSERT(image || image_lod);
	{
		cudaResourceDesc res_desc = {};

		Format format = Format::Char;

		if (image != nullptr)
		{
			format = image->format();

			res_desc.resType = cudaResourceTypeArray;
			res_desc.res.array.array = image->data().handle;

			if (!std::dynamic_pointer_cast<ImageCube<void>>(image) &&
				!std::dynamic_pointer_cast<ImageCubeLayered<void>>(image))
			{
				sampler.seamless_cubemap = false;	//	avoid error
			}
		}
		else if (image_lod != nullptr)
		{
			format = image_lod->format();

			res_desc.resType = cudaResourceTypeMipmappedArray;
			res_desc.res.mipmap.mipmap = image_lod->handle();

			if (!std::dynamic_pointer_cast<ImageCubeLod<void>>(image_lod) &&
				!std::dynamic_pointer_cast<ImageCubeLayeredLod<void>>(image_lod))
			{
				sampler.seamless_cubemap = false;	//	avoid error
			}
		}

		if ((view_format != Format::Float) && (view_format != Format::Float2) && (view_format != Format::Float4))
		{
			sampler.mipmap_filter_mode = FilterMode::Nearest;

			sampler.filter_mode = FilterMode::Nearest;
		}

		cudaTextureDesc								texture_desc = {};
		texture_desc.addressMode[0]					= static_cast<cudaTextureAddressMode>(sampler.address_mode[0]);
		texture_desc.addressMode[1]					= static_cast<cudaTextureAddressMode>(sampler.address_mode[1]);
		texture_desc.addressMode[2]					= static_cast<cudaTextureAddressMode>(sampler.address_mode[2]);
		texture_desc.filterMode						= static_cast<cudaTextureFilterMode>(sampler.filter_mode);
		texture_desc.readMode						= (format == view_format) ? cudaReadModeElementType : cudaReadModeNormalizedFloat;
		texture_desc.sRGB							= sampler.s_rgb;
		texture_desc.borderColor[0]					= sampler.border_color[0];
		texture_desc.borderColor[1]					= sampler.border_color[1];
		texture_desc.borderColor[2]					= sampler.border_color[2];
		texture_desc.borderColor[3]					= sampler.border_color[3];
		texture_desc.normalizedCoords				= sampler.normalized_coords;
		texture_desc.maxAnisotropy					= sampler.max_anisotropy;
		texture_desc.mipmapFilterMode				= static_cast<cudaTextureFilterMode>(sampler.mipmap_filter_mode);
		texture_desc.mipmapLevelBias					= sampler.mipmap_level_bias;
		texture_desc.minMipmapLevelClamp				= sampler.min_mipmap_level_clamp;
		texture_desc.maxMipmapLevelClamp				= sampler.max_mipmap_level_clamp;
		texture_desc.disableTrilinearOptimization	= sampler.disable_trilinear_optimization;
		texture_desc.seamlessCubemap					= sampler.seamless_cubemap;

		cudaError_t err = cudaCreateTextureObject(&texture_, &res_desc, &texture_desc, nullptr);

		if (err != cudaSuccess)
		{
			throw static_cast<ErrorCode>(err);
		}
		
		image_ = source_image;
		sampler_ = sampler;
	}
}


TextureBase::~TextureBase()
{
	if (texture_ != 0)
	{
		cudaError_t err = cudaDestroyTextureObject(texture_);

		if (err != cudaSuccess)
		{
			NS_ERROR_LOG("%s.", cudaGetErrorString(err));

			cudaGetLastError();
		}
	}
}
