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
#include "device.h"
#include "stream.h"
#include "allocator.h"
#include <cuda_runtime_api.h>

NS_USING_NAMESPACE

/*********************************************************************************
**********************************    Device    **********************************
*********************************************************************************/

Device::Device(int id, const cudaDeviceProp & devProp) : m_id(id), m_devProp(std::make_unique<cudaDeviceProp>(devProp)),
	m_defaultAlloc(std::make_shared<DeviceAllocator>(this)), m_defaultStream(new Stream(this, nullptr))
{

}


Error_t Device::init() noexcept
{
	this->setCurrent();

	cudaError_t err = cudaFree(nullptr);

	if (err != cudaSuccess)
	{
		NS_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();
	}

	return err;
}


size_t Device::freeMemorySize() const
{
	this->setCurrent();

	size_t freeMemInBytes = 0, totalMemInBytes = 0;

	cudaError_t err = cudaMemGetInfo(&freeMemInBytes, &totalMemInBytes);

	if (err != cudaSuccess)
	{
		NS_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();
	}

	return freeMemInBytes;
}


void Device::sync() const
{
	this->setCurrent();

	cudaError_t err = cudaDeviceSynchronize();

	if (err != cudaSuccess)
	{
		NS_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();
	}
}


void Device::setCurrent() const
{
	thread_local int currentDevice = 0;

	if (this->m_id != currentDevice)
	{
		cudaError_t err = cudaSetDevice(this->m_id);

		if (err != cudaSuccess)
		{
			NS_ERROR_LOG("%s.", cudaGetErrorString(err));

			cudaGetLastError();
		}
		else
		{
			currentDevice = this->m_id;
		}
	}
}


Device::~Device() noexcept
{

}