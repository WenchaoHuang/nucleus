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
#include "context.h"
#include <cuda_runtime_api.h>

NS_USING_NAMESPACE

/*********************************************************************************
*********************************    Context    **********************************
*********************************************************************************/

Context::Context()
{
	cudaGetLastError();

	//////////////////////////////////////////////////////////////////////

	int driverVersion = 0;

	cudaDriverGetVersion(&driverVersion);

	m_driverVersion.Major = driverVersion / 1000;
	m_driverVersion.Minor = (driverVersion % 1000) / 10;

	NS_INFO_LOG("CUDA driver version: %d.%d", m_driverVersion.Major, m_driverVersion.Minor);

	//////////////////////////////////////////////////////////////////////

	int runtimeVersion = 0;

	cudaRuntimeGetVersion(&runtimeVersion);

	m_runtimeVersion.Major = runtimeVersion / 1000;
	m_runtimeVersion.Minor = (runtimeVersion % 1000) / 10;

	NS_INFO_LOG("CUDA runtime version: %d.%d", m_runtimeVersion.Major, m_runtimeVersion.Minor);

	//////////////////////////////////////////////////////////////////////

	cudaGetLastError();

	int deviceCount = 0;

	auto err = cudaGetDeviceCount(&deviceCount);

	m_cudaDevices.resize(deviceCount, nullptr);

	NS_INFO_LOG_IF(err == cudaErrorNoDevice, "No CUDA-capable devices were detected.");

	//////////////////////////////////////////////////////////////////////

	for (int i = 0; i < deviceCount; i++)
	{
		cudaDeviceProp devProp = {};

		cudaGetDeviceProperties(reinterpret_cast<cudaDeviceProp*>(&devProp), i);

		NS_INFO_LOG("CUDA device(%d): %s, compute capability: %d.%d", i, devProp.name, devProp.major, devProp.minor);

		m_cudaDevices[i] = new Device(i, devProp);
	}

	cudaGetLastError();
}


const char * Context::getErrorString(Error_t eValue) noexcept
{
	return cudaGetErrorString(static_cast<cudaError_t>(eValue));
}


const char * Context::getErrorName(Error_t eValue) noexcept
{
	return cudaGetErrorName(static_cast<cudaError_t>(eValue));
}


Error_t Context::getLastError() noexcept
{
	return cudaGetLastError();
}


Context::~Context()
{
	for (size_t i = 0; i < m_cudaDevices.size(); i++)
	{
		delete m_cudaDevices[i];
	}

	m_cudaDevices.clear();
}