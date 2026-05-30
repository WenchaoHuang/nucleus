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

#include <gtest/gtest.h>
#include <nucleus/logger.h>
#include <nucleus/format.h>
#include <nucleus/device.h>
#include <nucleus/context.h>
#include <nucleus/allocator.h>

/*********************************************************************************
******************************    allocator_test    ******************************
*********************************************************************************/

class MyHostAllocator : public ns::HostAllocator
{
	virtual void * doAllocateMemory(size_t bytes) override
	{
		NS_INFO_LOG("Allocate host memory: %lld.", bytes);

		return ns::HostAllocator::doAllocateMemory(bytes);
	}
	virtual void doDeallocateMemory(void * ptr) override
	{
		ns::HostAllocator::doDeallocateMemory(ptr);

		NS_INFO_LOG("Deallocate host memory.");
	}
};


TEST(AllocatorTest, BasicOperations)
{
	auto device = ns::Context::getInstance()->device(0);

	MyHostAllocator hostAlloc;
	auto hostPtr = hostAlloc.allocateMemory(110);
	hostAlloc.deallocateMemory(hostPtr);

	ns::DeviceAllocator devAlloc(device);
	auto Ptr = devAlloc.allocateMemory(128);
	devAlloc.deallocateMemory(Ptr);

	auto pAlloc = device->defaultAllocator();
	auto texMem = pAlloc->allocateTextureMemory(ns::Format::Float, 100, 100, 100);
	devAlloc.deallocateTextureMemory(texMem);

	auto mipTexMem = pAlloc->allocateMipmapTextureMemory(ns::Format::Int, 100, 100, 100, 5);
	devAlloc.deallocateMipmapTextureMemory(mipTexMem);
}