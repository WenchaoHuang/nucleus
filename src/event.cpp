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
#include "event.h"
#include "device.h"
#include "logger.h"

NS_USING_NAMESPACE

/*********************************************************************************
**********************************    Event    ***********************************
*********************************************************************************/

Event::Event(Device * device, bool isBlockingSync, bool isDisableTiming)
	: m_device(device), m_hEvent(nullptr), m_isBlockingSync(isBlockingSync)
{
	NS_ASSERT(device != nullptr);

	device->setCurrent();

	unsigned int flags = cudaEventDefault;

	if (isBlockingSync)		flags |= cudaEventBlockingSync;
	if (isDisableTiming)	flags |= cudaEventDisableTiming;

	cudaError_t err = cudaEventCreateWithFlags(&m_hEvent, flags);

	if (err != cudaSuccess)
	{
		NS_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();

		throw err;
	}
}


void Event::sync() const
{
	cudaError_t err = cudaEventSynchronize(m_hEvent);

	if (err != cudaSuccess)
	{
		NS_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();
	}
}


bool Event::query() const
{
	cudaError_t err = cudaEventQuery(m_hEvent);

	if ((err != cudaSuccess) && (err != cudaErrorNotReady))
	{
		NS_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();
	}

	return err == cudaSuccess;
}


Event::~Event() noexcept
{
	if (m_hEvent != nullptr)
	{
		cudaError_t err = cudaEventDestroy(m_hEvent);

		if (err != cudaSuccess)
		{
			NS_ERROR_LOG("%s.", cudaGetErrorString(err));

			cudaGetLastError();
		}
	}
}

/*********************************************************************************
********************************    TimedEvent    ********************************
*********************************************************************************/

std::chrono::nanoseconds TimedEvent::elapsedTime(TimedEvent & eventStart, TimedEvent & eventEnd)
{
	float milliseconds = 0.0f;

	cudaError_t err = cudaEventElapsedTime(&milliseconds, eventStart.handle(), eventEnd.handle());

	if (err != cudaSuccess)
	{
		NS_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();
	}

	return std::chrono::nanoseconds(static_cast<long long>(milliseconds * 1e6));
}