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
#include <type_traits>

/*********************************************************************************
********************************    test_event    ********************************
*********************************************************************************/

void test_event()
{
	static_assert(std::is_trivially_copyable_v<ns::EventRef>);
	static_assert(std::is_nothrow_copy_constructible_v<ns::EventRef>);
	static_assert(!std::is_copy_assignable_v<ns::EventRef>);
	static_assert(!std::is_move_assignable_v<ns::EventRef>);
	static_assert(std::is_convertible_v<cudaEvent_t, ns::EventRef>);
	static_assert(!std::is_constructible_v<ns::EventRef, std::nullptr_t>);
	static_assert(!std::is_constructible_v<ns::EventRef, int>);
	static_assert(!std::is_copy_constructible_v<ns::Event>);
	static_assert(!ns::EventRef(cudaEvent_t{}));

	auto device = ns::Runtime::device(0);
	auto & stream = device->defaultStream();

	ns::Event		event0(device);
	ns::TimedEvent	event1(device);
	ns::TimedEvent	event2(device);
	ns::ScopedTimer	timer(stream, [](std::chrono::nanoseconds ns) { printf("ScopedTime: %fus\n", ns.count() * 1e-3); });

	{
		ns::Stream		   producer(device);
		const ns::EventRef borrowed = event0.get();
		const ns::EventRef copy		= borrowed;
		assert(copy == borrowed && copy != event1);

		producer.record(borrowed);
		stream.waitFor(copy).sync();
		assert(copy.is_done());
	}
	stream.record(event0.get());
	event0.sync();
	assert(event0.is_done());

	const auto expectCudaError = [](auto operation)
	{
		try
		{
			operation();
		}
		catch ([[maybe_unused]] ns::ErrorCode error)
		{
			assert(error != 0);
			return;
		}
		assert(false && "Expected the operation to report a CUDA error");
	};
	const ns::EventRef empty(cudaEvent_t{});
	expectCudaError([&]
					{ empty.sync(); });
	expectCudaError([&]
					{ (void)empty.is_done(); });
	expectCudaError([&]
					{ (void)(event2 - event1); });

	stream.record(event1);
	stream.record(event2);
	stream.waitFor(event2).sync();

	[[maybe_unused]] auto elapsed = event2 - event1;
	assert(elapsed.count() >= 0);
}
