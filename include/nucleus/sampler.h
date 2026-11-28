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
#include <limits>

namespace NS_NAMESPACE
{
	/*****************************************************************************
	*****************************    AddressMode    ******************************
	*****************************************************************************/

	enum class AddressMode		//	CUDA texture address modes
	{
		Wrap = 0,				//	Wrapping address mode
		Clamp = 1,				//	Clamp to edge address mode
		Mirror = 2,				//	Mirror address mode
		Border = 3,				//	Border address mode
	};

	/*****************************************************************************
	******************************    FilterMode    ******************************
	*****************************************************************************/
	
	enum class FilterMode		//	CUDA texture filter modes
	{
		Nearest = 0,			//	Point filter mode
		Linear = 1,				//	Linear filter mode
	};

	/*****************************************************************************
	*******************************    Sampler    ********************************
	*****************************************************************************/

	struct Sampler
	{
		/**
		 *	@brief		Specifies the addressing mode for each dimension of the texture data.
		 *	@details	If Sampler::normalized_coords is set to zero, AddressMode::eWrap and AddressMode::eMirror won't be supported and will be switched to AddressMode::eClamp.
		 */
		AddressMode address_mode[3] = { AddressMode::Wrap, AddressMode::Wrap, AddressMode::Wrap };


		/**
		 *	@brief		Specifies the filtering mode to be used when fetching from the texture.
		 */
		FilterMode filter_mode = FilterMode::Linear;


		/**
		 *	@brief		Specifies the filter mode when the calculated mipmap level lies between two defined mipmap levels.
		 */
		FilterMode mipmap_filter_mode = FilterMode::Linear;


		/**
		 *	@brief		Specifies the float values of color.
		 *	@note		Application using integer border color values will need to <reinterpret_cast> these values to float.
		 *				The values are set only when the addressing mode specified by Sampler::address_mode is AddressMode::eBorder.
		 */
		float border_color[4] = { 0.0f, 0.0f, 0.0f, 0.0f };


		/**
		 *	@brief		Specifies the maximum anistropy ratio to be used when doing anisotropic filtering.
		 *	@note		This value will be clamped to the range [1,16].
		 */
		unsigned int max_anisotropy = 1;


		/**
		 *	@brief		Specifies the offset to be applied to the calculated mipmap level.
		 */
		float mipmap_level_bias = 0.0f;


		/**
		 *	@brief		Specifies the lower end of the mipmap level range to clamp access to.
		 */
		float min_mipmap_level_clamp = 0.0f;


		/**
		 *	@brief		Specifies the upper end of the mipmap level range to clamp access to.
		 */
		float max_mipmap_level_clamp = std::numeric_limits<float>::max();


		/**
		 *	@brief		Specifies whether s_rgb to linear conversion should be performed during texture fetch.
		 *	@note		This applies only to 8-bit and 16-bit unsigned integer formats.
		 */
		bool s_rgb = true;


		/**
		 *	@brief		Specifies whether the texture coordinates will be normalized or not.
		 */
		bool normalized_coords = true;


		/**
		 *	@brief		Specifies whether the trilinear filtering optimizations will be disabled.
		 */
		bool disable_trilinear_optimization = false;


		/**
		 *	@brief		Specifies whether seamless cube map filtering is enabled.
		 *	@note		This flag can only be specified if the underlying resource is a ImageCube, ImageCubeLod or ImageCubeLayeredLod.
		 *	@note		When seamless cube map filtering is enabled, texture address modes specified by Sampler::address_mode are ignored.
		 *	@note		Instead, if the Sampler::filter_mode is set to FilterMode::eNearest the address mode AddressMode::eClamp will be applied for all dimensions.
		 *	@note		If the Sampler::filter_mode is set to FilterMode::eLinear seamless cube map filtering will be performed when sampling along the cube face borders.
		 */
		bool seamless_cubemap = true;
	};
}
