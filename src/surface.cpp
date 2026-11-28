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
#include "surface.h"
#include <cuda_runtime_api.h>

NS_USING_NAMESPACE

/*********************************************************************************
*********************************    Surface    **********************************
*********************************************************************************/

Surface::Surface() : surface_(0)
{

}


void Surface::bind_image(std::shared_ptr<Image> image)
{
	this->unbind();

	cudaResourceDesc res_desc = {};
	res_desc.resType = cudaResourceTypeArray;
	res_desc.res.array.array = image->handle();

	cudaError_t err = cudaCreateSurfaceObject(&surface_, &res_desc);

	if (err != cudaSuccess)
	{
		throw static_cast<ErrorCode>(err);
	}
	
	image_ = image;
}


void Surface::unbind()
{
	if (surface_ != 0)
	{
		cudaError_t err = cudaDestroySurfaceObject(surface_);

		if (err != cudaSuccess)
		{
			throw static_cast<ErrorCode>(err);
		}

		image_ = nullptr;

		surface_ = 0;
	}
}


Surface::~Surface()
{
	try
	{
		this->unbind();
	}
	catch (ErrorCode err)
	{
		NS_ERROR_LOG("%s.", cudaGetErrorString(static_cast<cudaError_t>(err)));
	}
}
