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
#include <nucleus/buffer_view.h>

 /*********************************************************************************
 *****************************    buffer_view_test    *****************************
 *********************************************************************************/

void buffer_view_test()
{
	auto device = ns::Runtime::device(0);
	auto allocator = device->defaultAllocator();
	auto buffer = std::make_shared<ns::Buffer>(allocator, sizeof(int) * 1024);

	ns::BufferView<int> bufferView0;
	ns::BufferView<int> bufferView1 = nullptr;
	ns::BufferView<int> bufferView2(buffer);
	ns::BufferView<int> bufferView3(buffer, 0, sizeof(int) * 10);
	ns::BufferView<int> bufferView3b = bufferView3.subview(0, 5);

	ns::BufferView2D<int> bufferView4;
	ns::BufferView2D<int> bufferView5 = nullptr;

	ns::BufferView3D<int> bufferView6;
	ns::BufferView3D<int> bufferView7 = nullptr;

	ns::view_cast<int>(bufferView0);
	ns::view_cast<float>(bufferView4);
	ns::view_cast<unsigned int>(bufferView6);
}