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

#include "event.h"
#include "device.h"
#include "logger.h"
#include <cuda_runtime_api.h>

NS_USING_NAMESPACE

/*********************************************************************************
*********************************    EventRef    *********************************
*********************************************************************************/

void EventRef::sync() const
{
	cudaError_t err = cudaEventSynchronize(event_);

	if (err != cudaSuccess)
	{
		throw static_cast<ErrorCode>(err);
	}
}


bool EventRef::is_done() const
{
	cudaError_t err = cudaEventQuery(event_);

	if (err != cudaSuccess && err != cudaErrorNotReady)
	{
		throw static_cast<ErrorCode>(err);
	}

	return err == cudaSuccess;
}

/*********************************************************************************
**********************************    Event    ***********************************
*********************************************************************************/

Event::Event(Device * device, int flags) : EventRef(cudaEvent_t{}), device_(device)
{
	NS_ASSERT(device != nullptr);

	device->set_current();

	cudaError_t err = cudaEventCreateWithFlags(&event_, flags);

	if (err != cudaSuccess)
	{
		throw static_cast<ErrorCode>(err);
	}
}


Event::Event(Device * device, bool blocking_sync) : Event(device, blocking_sync ? cudaEventBlockingSync | cudaEventDisableTiming : cudaEventDisableTiming)
{

}


Event::~Event() noexcept
{
	if (event_ != nullptr)
	{
		cudaError_t err = cudaEventDestroy(event_);

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

TimedEvent::TimedEvent(Device * device, bool blocking_sync) : Event(device, blocking_sync ? cudaEventBlockingSync : cudaEventDefault)
{

}


std::chrono::nanoseconds NS_NAMESPACE::operator-(const TimedEvent & end, const TimedEvent & start)
{
	float milliseconds = 0.0f;

	cudaError_t err = cudaEventElapsedTime(&milliseconds, start.handle(), end.handle());

	if (err == cudaSuccess)
	{
		return std::chrono::nanoseconds(static_cast<long long>(milliseconds * 1e6));
	}
	else
	{
		throw static_cast<ErrorCode>(err);
	}
}
