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

#include <nucleus/logger.h>
#include <nucleus/format.h>
#include <nucleus/device.h>
#include <nucleus/runtime.h>
#include <nucleus/allocator.h>

/*********************************************************************************
******************************    test_allocator    ******************************
*********************************************************************************/

class MyHostAllocator : public ns::HostAllocator
{
	virtual void * do_allocate_memory(size_t bytes) override
	{
		NS_INFO_LOG("Allocate host memory: %lld.", bytes);

		return ns::HostAllocator::do_allocate_memory(bytes);
	}
	virtual void do_deallocate_memory(void * ptr) override
	{
		ns::HostAllocator::do_deallocate_memory(ptr);

		NS_INFO_LOG("Deallocate host memory.");
	}
};


void test_allocator()
{
	auto device = ns::Runtime::device(0);

	MyHostAllocator host_alloc;
	auto host_ptr = host_alloc.allocate_memory(110);
	host_alloc.deallocate_memory(host_ptr);

	ns::DeviceAllocator dev_alloc(device);
	auto Ptr = dev_alloc.allocate_memory(128);
	dev_alloc.deallocate_memory(Ptr);

	auto pAlloc = device->default_allocator();
	auto tex_mem = pAlloc->allocate_texture_memory(ns::Format::Float, 100, 100, 100);
	dev_alloc.deallocate_texture_memory(tex_mem);

	auto mip_tex_mem = pAlloc->allocate_mipmap_texture_memory(ns::Format::Int, 100, 100, 100, 5);
	dev_alloc.deallocate_mipmap_texture_memory(mip_tex_mem);
}
