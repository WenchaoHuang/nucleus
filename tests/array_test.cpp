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

#include <vector>
#include <nucleus/device.h>
#include <nucleus/runtime.h>
#include <nucleus/array_1d.h>
#include <nucleus/array_2d.h>
#include <nucleus/array_3d.h>

/*********************************************************************************
********************************    array_test    ********************************
*********************************************************************************/

static void test(dev::Ptr<int> a, dev::Ptr2<const float> b, dev::Ptr3<float> c)
{

}


void array_test()
{
	auto device = ns::Runtime::device(0);
	auto allocator = device->defaultAllocator();

	ns::Array<int>		array0;
	ns::Array<int>		array1(allocator, 100);
	ns::Array<int>		array11 = std::move(array1);

	ns::Array2D<float>	array2;
	ns::Array2D<float>	array3(allocator, 100, 100);
	ns::Array2D<float>	array33 = std::move(array3);

	ns::Array3D<float>	array4;
	ns::Array3D<float>	array5(allocator, 100, 100, 100);
	ns::Array3D<float>	array55 = std::move(array5);

	std::vector<ns::Array<int>>		arrArray0;
	std::vector<ns::Array<int>>		arrArray1(200);

	arrArray1[0].resize(allocator, 10);
	arrArray1.resize(20);

	assert(array0.empty());
	assert(array1.empty());
	assert(!array11.empty());
	assert(arrArray1[0].size() == 10);

	if (!array11.empty())
	{
		assert(array11.size() == 100);
		assert(array11.width() == 100);
		assert(array11.bytes() == 100 * sizeof(int));
		assert(array11.pitch() == 100 * sizeof(int));
		assert(array11.releaseBuffer() != nullptr);
		assert(array11.allocator() == nullptr);
		array11.resize(allocator, 200);
		array11.resize(300);
		array11.resize(300);
		array11.clear();
		array11.ptr();
		assert(array11.data() == nullptr);
	}

	if (!array33.empty())
	{
		float * data = &array33[0][2];

		assert(array33.size() == 100 * 100);
		assert(array33.bytes() == 100 * 100 * sizeof(float));
		assert(array33.width() == 100);
		assert(array33.pitch() == 100 * sizeof(float));
		assert(array33.height() == 100);
		assert(array33.allocator() == allocator);
		assert(array33.releaseBuffer() != nullptr);
		array33.resize(allocator, 200, 400);
		array33.reshape(100, 800);
		array33.resize(400, 200);
		array33.clear();
		array33.ptr();
		assert(array33.data() == nullptr);
	}

	if (!array55.empty())
	{
		float * data = &array55[0][0][0];

		assert(array55.size() == 100 * 100 * 100);
		assert(array55.bytes() == 100 * 100 * 100 * sizeof(float));
		assert(array55.width() == 100);
		assert(array55.pitch() == 100 * sizeof(float));
		assert(array55.depth() == 100);
		assert(array55.height() == 100);
		assert(array55.allocator() == allocator);
		array55.resize(allocator, 200, 400, 500);
		array55.reshape(200, 500, 400);
		array55.resize(500, 200, 300);
		array55.clear();
		array55.data();
		array55.ptr();
		assert(array5.data() == nullptr);
	}

	test(array11, array33, array55);
}