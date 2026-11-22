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

#include <nucleus/event.h>
#include <nucleus/device.h>
#include <nucleus/stream.h>
#include <nucleus/runtime.h>
#include <nucleus/scoped_timer.h>
#include <cuda_runtime_api.h>

/*********************************************************************************
********************************    test_event    ********************************
*********************************************************************************/

void test_event()
{
	auto device = ns::Runtime::device(0);
	auto & stream = device->defaultStream();

	ns::Event		event0(device);
	ns::TimedEvent	event1(device);
	ns::TimedEvent	event2(device);
	ns::ScopedTimer	timer(stream, [](std::chrono::nanoseconds ns) { printf("ScopedTime: %fus\n", ns.count() * 1e-3); });

	ns::Event		releasable_event(device);
	cudaEvent_t		released_handle = releasable_event.release();
	assert(cudaEventDestroy(released_handle) == cudaSuccess);
	assert(released_handle != nullptr);
	assert(!releasable_event);

	{
		ns::Stream				producer(device);
		const ns::EventRef		borrowed = event0.handle();
		const ns::EventRef		copy = borrowed;
		assert(copy == borrowed && copy != event1);

		producer.record(borrowed);
		stream.waitFor(copy).sync();
		assert(copy.is_done());
	}

	// Test event recording and synchronization
	stream.record(event0.handle());
	event0.sync();
	stream.record(event1);
	stream.record(event2);
	stream.waitFor(event2).sync();
	assert(event0.is_done());


	// Test elapsed time
	auto elapsed = event2 - event1;
	assert(elapsed.count() >= 0);


	// Move assignment test
	auto event3 = std::move(event2);
	assert(!event2 && event3);


	// Move assignment test
	event3 = std::move(event1);
	assert(!event1 && event3);
}
