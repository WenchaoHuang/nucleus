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
#include <nucleus/stream.h>
#include <nucleus/runtime.h>
#include <nucleus/surface.h>
#include <nucleus/array_2d.h>
#include <nucleus/allocator.h>
#include <nucleus/launch_utils.cuh>

/*********************************************************************************
******************************    pinned_memory    *******************************
*********************************************************************************/

struct NS_ALIGN(16) MyPixelType1
{
	float a, b, c, d;
};

struct NS_ALIGN(16) MyPixelType2
{
	float a;
	int b;
	short c;
	short d;
	char e[4];
};


template<> struct ns::FormatMapping<MyPixelType1> { static constexpr ns::Format value = ns::Format::Float4; };
template<> struct ns::FormatMapping<MyPixelType2> { static constexpr ns::Format value = ns::Format::Int4; };


__global__ void pixel_assign(dev::Surf2D<MyPixelType1> out1, dev::Surf2D<MyPixelType2> out2)
{
	auto i = blockDim.x * blockIdx.x + threadIdx.x;
	auto j = blockDim.y * blockIdx.y + threadIdx.y;

	MyPixelType1 value1 = {};
	value1.a = 1.0f;
	value1.b = 2.0f;
	value1.c = 3.0f;
	value1.d = 4.0f;

	MyPixelType2 value2 = {};
	value2.a = 1.0f;
	value2.b = 2;
	value2.c = 3;
	value2.d = 4;
	value2.e[0] = 5;
	value2.e[1] = 6;
	value2.e[2] = 7;
	value2.e[3] = 8;

	out1.write(value1, i, j);
	out2.write(value2, i, j);
}


int main()
{
	auto device = ns::Runtime::device(0);
	auto hostAlloc = std::make_shared<ns::HostAllocator>();
	auto allocator = device->defaultAllocator();
	auto & stream = device->defaultStream();
	
	auto image1 = std::make_shared<ns::Image2D<MyPixelType1>>(allocator, 16, 16);
	auto image2 = std::make_shared<ns::Image2D<MyPixelType2>>(allocator, 16, 16);

	ns::Surface2D<MyPixelType1>		surface1(image1);
	ns::Surface2D<MyPixelType2>		surface2(image2);
	ns::Array2D<MyPixelType1>		array1(hostAlloc, image1->width(), image1->height());
	ns::Array2D<MyPixelType2>		array2(hostAlloc, image2->width(), image2->height());

	stream.launch(pixel_assign, 1, { image1->width(), image1->height(), 1})(surface1, surface2);
	stream.memcpy2D(array1.data(), array1.pitch(), image1->data(), image1->width(), image1->height());
	stream.memcpy2D(array2.data(), array2.pitch(), image2->data(), image2->width(), image2->height());
	stream.sync();

	for (int i = 0; i < array1.height(); i++)
	{
		for (int j = 0; j < array1.width(); j++)
		{
			printf(" array1[%d][%d].a = %f\n", i, j, array1[i][j].a);
			printf(" array1[%d][%d].b = %f\n", i, j, array1[i][j].b);
			printf(" array1[%d][%d].c = %f\n", i, j, array1[i][j].c);
			printf(" array1[%d][%d].d = %f\n", i, j, array1[i][j].d);
			printf("\n");

			assert(array1[i][j].a == 1.0f);
			assert(array1[i][j].b == 2.0f);
			assert(array1[i][j].c == 3.0f);
			assert(array1[i][j].d == 4.0f);
		}
	}

	printf("\n\n");

	for (int i = 0; i < array2.height(); i++)
	{
		for (int j = 0; j < array2.width(); j++)
		{
			printf(" array1[%d][%d].a = %f\n", i, j, array2[i][j].a);
			printf(" array1[%d][%d].b = %d\n", i, j, array2[i][j].b);
			printf(" array1[%d][%d].c = %d\n", i, j, array2[i][j].c);
			printf(" array1[%d][%d].d = %d\n", i, j, array2[i][j].d);
			printf(" array1[%d][%d].e[0] = %d\n", i, j, array2[i][j].e[0]);
			printf(" array1[%d][%d].e[1] = %d\n", i, j, array2[i][j].e[1]);
			printf(" array1[%d][%d].e[2] = %d\n", i, j, array2[i][j].e[2]);
			printf(" array1[%d][%d].e[3] = %d\n", i, j, array2[i][j].e[3]);
			printf("\n");

			assert(array2[i][j].a == 1.0f);
			assert(array2[i][j].b == 2);
			assert(array2[i][j].c == 3);
			assert(array2[i][j].d == 4);
			assert(array2[i][j].e[0] == 5);
			assert(array2[i][j].e[1] == 6);
			assert(array2[i][j].e[2] == 7);
			assert(array2[i][j].e[3] == 8);
		}
	}

	system("pause");

	return 0;
}