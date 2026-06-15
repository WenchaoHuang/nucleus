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

/*********************************************************************************
*******************************    buffer_test    ********************************
*********************************************************************************/

void buffer_test()
{
	auto device = ns::Runtime::device(0);
	auto allocator = device->defaultAllocator();

	ns::Buffer buffer0;
	ns::Buffer buffer1(allocator, 1024);
	ns::SharedBuffer sharedBuffer(allocator, 1024);
	std::unique_ptr<ns::Buffer> uniqueBuffer = std::make_unique<ns::Buffer>(allocator, 1024);

	uniqueBuffer = nullptr;
	sharedBuffer = nullptr;

	buffer1.allocator();
	buffer1.capacity();
	buffer1.empty();
	buffer1.data();
}