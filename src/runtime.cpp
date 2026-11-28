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
#include "runtime.h"
#include <cuda_runtime_api.h>

NS_USING_NAMESPACE

/*********************************************************************************
*********************************    Runtime    **********************************
*********************************************************************************/

Runtime::Runtime()
{
	int driver_version = 0;

	cudaError_t err = cudaDriverGetVersion(&driver_version);

	if (err != cudaSuccess)		throw static_cast<ErrorCode>(err);

	driver_version_.major = driver_version / 1000;
	driver_version_.minor = (driver_version % 1000) / 10;

	NS_INFO_LOG("CUDA driver version: %d.%d", driver_version_.major, driver_version_.minor);

	//////////////////////////////////////////////////////////////////////

	int runtime_version = 0;

	err = cudaRuntimeGetVersion(&runtime_version);

	if (err != cudaSuccess)		throw static_cast<ErrorCode>(err);

	runtime_version_.major = runtime_version / 1000;
	runtime_version_.minor = (runtime_version % 1000) / 10;

	NS_INFO_LOG("CUDA runtime version: %d.%d", runtime_version_.major, runtime_version_.minor);

	//////////////////////////////////////////////////////////////////////

	int device_count = 0;

	err = cudaGetDeviceCount(&device_count);

	if (err != cudaSuccess && err != cudaErrorNoDevice)	throw static_cast<ErrorCode>(err);

	NS_INFO_LOG_IF(err == cudaErrorNoDevice, "No CUDA-capable devices were detected.");

	//////////////////////////////////////////////////////////////////////

	try
	{
		for (int i = 0; i < device_count; i++)
		{
			cudaDeviceProp dev_prop = {};

			err = cudaGetDeviceProperties(reinterpret_cast<cudaDeviceProp*>(&dev_prop), i);

			if (err != cudaSuccess) throw static_cast<ErrorCode>(err);

			NS_INFO_LOG("CUDA device(%d): %s, compute capability: %d.%d", i, dev_prop.name, dev_prop.major, dev_prop.minor);

			cuda_devices_.emplace_back(new Device(i, dev_prop));

			if (i == 0)
			{
				default_alloc_ = cuda_devices_[0]->default_allocator();
			}
		}
	}
	catch (...)
	{
		for (auto device : cuda_devices_)
		{
			delete device;
		}

		throw;
	}
}


const char * Runtime::get_error_string(ErrorCode error) noexcept
{
	return cudaGetErrorString(static_cast<cudaError_t>(error));
}


const char * Runtime::get_error_name(ErrorCode error) noexcept
{
	return cudaGetErrorName(static_cast<cudaError_t>(error));
}


ErrorCode Runtime::get_last_error() noexcept
{
	return cudaGetLastError();
}


Runtime::~Runtime()
{
	for (auto device : cuda_devices_)
	{
		delete device;
	}
}
