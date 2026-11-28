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

#include <nucleus/buffer.h>
#include <nucleus/device.h>
#include <nucleus/runtime.h>

class CountingAllocator : public ns::Allocator
{

public:

	unsigned int allocations = 0;
	unsigned int deallocations = 0;

private:

	void * do_allocate_memory(size_t bytes) override
	{
		allocations++;

		return ::operator new(bytes);
	}

	void do_deallocate_memory(void * ptr) override
	{
		deallocations++;

		::operator delete(ptr);
	}
};

/*********************************************************************************
*******************************    test_buffer    ********************************
*********************************************************************************/

void test_buffer()
{
	auto device = ns::Runtime::device(0);
	auto allocator = device->default_allocator();

	ns::Buffer buffer0;
	ns::Buffer buffer1(allocator, 1024);
	ns::Buffer buffer2 = buffer1;
	ns::Buffer buffer3 = std::move(buffer2);
	ns::Buffer buffer4;
	buffer4 = std::move(buffer3);
	std::unique_ptr<ns::Buffer> unique_buffer = std::make_unique<ns::Buffer>(allocator, 1024);

	static_assert(std::is_nothrow_copy_constructible_v<ns::Buffer>);
	static_assert(std::is_nothrow_move_constructible_v<ns::Buffer>);

	assert(buffer0.empty());
	assert(buffer0.capacity() == 0);
	assert(buffer0.data() == nullptr);
	assert(buffer0.allocator() == nullptr);
	assert(buffer1.data() == buffer4.data());
	assert(buffer1.address() == buffer4.address());
	assert(buffer1.capacity() == buffer4.capacity());
	assert(buffer1.allocator() == buffer4.allocator());
	assert(buffer2.empty());
	assert(buffer2.capacity() == 0);
	assert(!buffer2);
	assert(buffer3.empty());
	assert(buffer3.address() == 0);
	assert(buffer3.capacity() == 0);
	assert(!buffer3);

	auto counting_allocator = std::make_shared<CountingAllocator>();
	{
		ns::Buffer original(counting_allocator, 64);
		ns::Buffer copy = original;

		original = ns::Buffer();
		assert(counting_allocator->allocations == 1);
		assert(counting_allocator->deallocations == 0);
	}
	assert(counting_allocator->deallocations == 1);

	unique_buffer = nullptr;

	buffer1.allocator();
	buffer1.capacity();
	buffer1.empty();
	buffer1.data();
}
