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

#include "format.h"
#include "logger.h"
#include "device.h"
#include "allocator.h"
#include <cuda_runtime.h>

NS_USING_NAMESPACE

/*********************************************************************************
******************************    HostAllocator    *******************************
*********************************************************************************/

void * HostAllocator::do_allocate_memory(size_t bytes)
{
	void * ptr = nullptr;

	cudaError_t err = cudaMallocHost(&ptr, bytes);

	if (err != cudaSuccess)
	{
		NS_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();

		throw err;
	}

#ifdef NS_CUDA_MEMORY_CHECK
	err = cudaMemset(ptr, -1, bytes);
#endif

	return ptr;
}


void HostAllocator::do_deallocate_memory(void * ptr)
{
	cudaError_t err = cudaFreeHost(ptr);

	if (err != cudaSuccess)
	{
		NS_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();
	}
}

/*********************************************************************************
*****************************    DeviceAllocator    ******************************
*********************************************************************************/

DeviceAllocator::DeviceAllocator(class Device * device) : device_(device)
{
	NS_ASSERT(device != nullptr);
}


static cudaChannelFormatDesc create_channel_desc(Format format)
{
	cudaChannelFormatDesc channel_desc = {};

	switch (format)
	{
		case Format::Int:		{ channel_desc = cudaCreateChannelDesc<::int1>();		break; }
		case Format::Int2:		{ channel_desc = cudaCreateChannelDesc<::int2>();		break; }
		case Format::Int4:		{ channel_desc = cudaCreateChannelDesc<::int4>();		break; }
		case Format::Uint:		{ channel_desc = cudaCreateChannelDesc<::uint1>();		break; }
		case Format::Uint2:		{ channel_desc = cudaCreateChannelDesc<::uint2>();		break; }
		case Format::Uint4:		{ channel_desc = cudaCreateChannelDesc<::uint4>();		break; }
		case Format::Char:		{ channel_desc = cudaCreateChannelDesc<::char1>();		break; }
		case Format::Char2:		{ channel_desc = cudaCreateChannelDesc<::char2>();		break; }
		case Format::Char4:		{ channel_desc = cudaCreateChannelDesc<::char4>();		break; }
		case Format::Uchar:		{ channel_desc = cudaCreateChannelDesc<::uchar1>();		break; }
		case Format::Uchar2:	{ channel_desc = cudaCreateChannelDesc<::uchar2>();		break; }
		case Format::Uchar4:	{ channel_desc = cudaCreateChannelDesc<::uchar4>();		break; }
		case Format::Short:		{ channel_desc = cudaCreateChannelDesc<::short1>();		break; }
		case Format::Short2:	{ channel_desc = cudaCreateChannelDesc<::short2>();		break; }
		case Format::Short4:	{ channel_desc = cudaCreateChannelDesc<::short4>();		break; }
		case Format::Ushort:	{ channel_desc = cudaCreateChannelDesc<::ushort1>();	break; }
		case Format::Ushort2:	{ channel_desc = cudaCreateChannelDesc<::ushort2>();	break; }
		case Format::Ushort4:	{ channel_desc = cudaCreateChannelDesc<::ushort4>();	break; }
		case Format::Float:		{ channel_desc = cudaCreateChannelDesc<::float1>();		break; }
		case Format::Float2:	{ channel_desc = cudaCreateChannelDesc<::float2>();		break; }
		case Format::Float4:	{ channel_desc = cudaCreateChannelDesc<::float4>();		break; }
		default:				{ NS_ERROR_LOG("Invalid format!");						break; }
	}

	return channel_desc;
}


cudaArray_t DeviceAllocator::allocate_texture_memory(Format format, size_t width, size_t height, size_t depth, int flags)
{
	device_->set_current();

	cudaArray_t array = nullptr;

	cudaChannelFormatDesc channel_desc = create_channel_desc(format);

	cudaError_t err = cudaMalloc3DArray(&array, &channel_desc, make_cudaExtent(width, height, depth), flags);

	if (err != cudaSuccess)
	{
		NS_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();

		throw err;
	}

	return array;
}


cudaMipmappedArray_t DeviceAllocator::allocate_mipmap_texture_memory(Format format, size_t width, size_t height, size_t depth, unsigned int num_levels, int flags)
{
	device_->set_current();

	cudaMipmappedArray_t mipmaped_array = nullptr;

	cudaChannelFormatDesc channel_desc = create_channel_desc(format);

	cudaError_t err = cudaMallocMipmappedArray(&mipmaped_array, &channel_desc, make_cudaExtent(width, height, depth), num_levels, flags);

	if (err != cudaSuccess)
	{
		NS_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();

		throw err;
	}

	return mipmaped_array;
}


void DeviceAllocator::deallocate_mipmap_texture_memory(cudaMipmappedArray_t mipmaped_array)
{
	cudaError_t err = cudaFreeMipmappedArray(mipmaped_array);

	if (err != cudaSuccess)
	{
		NS_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();
	}
}


void DeviceAllocator::deallocate_texture_memory(cudaArray_t array)
{
	cudaError_t err = cudaFreeArray(array);

	if (err != cudaSuccess)
	{
		NS_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();
	}
}


void * DeviceAllocator::do_allocate_memory(size_t bytes)
{
	void * ptr = nullptr;

	device_->set_current();

	cudaError_t err = cudaMalloc(&ptr, bytes);

	if (err != cudaSuccess)
	{
		NS_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();

		throw err;
	}

#ifdef NS_CUDA_MEMORY_CHECK
	err = cudaMemset(ptr, -1, bytes);

	cudaDeviceSynchronize();
#endif

	return ptr;
}


void DeviceAllocator::do_deallocate_memory(void * ptr)
{
	cudaError_t err = cudaFree(ptr);

	if (err != cudaSuccess)
	{
		NS_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();
	}
}
