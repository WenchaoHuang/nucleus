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

#include <nucleus/device.h>
#include <nucleus/runtime.h>
#include <nucleus/buffer_slice.h>

 /*********************************************************************************
 ****************************    test_buffer_slice    *****************************
 *********************************************************************************/

void test_buffer_slice()
{
	auto device = ns::Runtime::device(0);
	auto allocator = device->defaultAllocator();
	ns::Buffer buffer(allocator, sizeof(int) * 1024);

	ns::BufferSlice<int> bufferSlice0;
	ns::BufferSlice<int> bufferSlice1 = nullptr;
	ns::BufferSlice<int> bufferSlice2(buffer);
	ns::BufferSlice<int> bufferSlice3(buffer, 0, sizeof(int) * 10);
	ns::BufferSlice<int> bufferSlice3b = bufferSlice3.subslice(0, 5);

	ns::BufferSlice2D<int> bufferSlice4;
	ns::BufferSlice2D<int> bufferSlice5 = nullptr;

	ns::BufferSlice3D<int> bufferSlice6;
	ns::BufferSlice3D<int> bufferSlice7 = nullptr;

	ns::slice_cast<int>(bufferSlice0);
	ns::slice_cast<float>(bufferSlice4);
	ns::slice_cast<unsigned int>(bufferSlice6);
}
