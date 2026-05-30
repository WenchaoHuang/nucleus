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
#include <vector>
#include <functional>
#include <nucleus/stream.h>
#include <nucleus/device.h>
#include <nucleus/context.h>
#include <nucleus/array_1d.h>
#include <nucleus/array_2d.h>
#include <nucleus/array_3d.h>
#include <nucleus/image_1d.h>
#include <nucleus/image_2d.h>
#include <nucleus/image_3d.h>
#include <nucleus/launch_utils.cuh>
#include <device_launch_parameters.h>

/*********************************************************************************
*******************************    stream_test    ********************************
*********************************************************************************/

__constant__ int cache[100];

__global__ void test_kernel()
{
	CUDA_for(i, 1);

	printf("device: Happy Nucleus!\n");
}


TEST(StreamTest, BasicOperations)
{
	auto device = ns::Context::getInstance()->device(0);
	auto allocator = device->defaultAllocator();
	auto & stream = device->defaultStream();

	stream.sync();
	stream.isComplete();
	stream.setForceSync(true);
	stream.setForceSync(false);
	EXPECT_EQ(stream.device(), device);
	EXPECT_EQ(stream.handle(), nullptr);
}

TEST(StreamTest, HostFunc)
{
	auto device = ns::Context::getInstance()->device(0);
	auto & stream = device->defaultStream();

	int a;
	auto pfnTask = [](int*) { printf("host: Happy Nucleus!\n"); };
	stream.launchHostFunc<int>(pfnTask, &a);
	stream.launch(test_kernel, ns::ceil_div(15, 32), 32)();
}

TEST(StreamTest, Memcpy1D)
{
	auto device = ns::Context::getInstance()->device(0);
	auto allocator = device->defaultAllocator();
	auto & stream = device->defaultStream();

	std::vector<int>	host_data(100, 33);
	ns::Array<int>		dev_data1(allocator, 100);

	stream.fill(dev_data1.data(), 1, dev_data1.size());
	stream.memcpy(host_data.data(), dev_data1.data(), dev_data1.size());

	for (size_t i = 0; i < host_data.size(); i++)
	{
		EXPECT_EQ(host_data[i], 1);
	}
}

TEST(StreamTest, Memcpy2D)
{
	auto device = ns::Context::getInstance()->device(0);
	auto allocator = device->defaultAllocator();
	auto & stream = device->defaultStream();

	std::vector<int>	host_data(100, 33);
	ns::Array2D<int>	dev_data2(allocator, 10, 10);

	stream.fill(dev_data2.data(), 2, dev_data2.size());
	stream.memcpy2D(host_data.data(), dev_data2.pitch(), dev_data2.data(), dev_data2.pitch(), dev_data2.width(), dev_data2.height());

	for (size_t i = 0; i < host_data.size(); i++)
	{
		EXPECT_EQ(host_data[i], 2);
	}
}

TEST(StreamTest, Memcpy3D)
{
	auto device = ns::Context::getInstance()->device(0);
	auto allocator = device->defaultAllocator();
	auto & stream = device->defaultStream();

	std::vector<int>	host_data(100, 33);
	ns::Array3D<int>	dev_data3(allocator, 2, 5, 10);

	stream.fill(dev_data3.data(), 3, dev_data3.size());
	stream.memcpy3D(host_data.data(), dev_data3.pitch(), dev_data3.height(), dev_data3.data(), dev_data3.pitch(), dev_data3.height(), dev_data3.width(), dev_data3.height(), dev_data3.depth());

	for (size_t i = 0; i < host_data.size(); i++)
	{
		EXPECT_EQ(host_data[i], 3);
	}
}

TEST(StreamTest, MemcpyImage1D)
{
	auto device = ns::Context::getInstance()->device(0);
	auto allocator = device->defaultAllocator();
	auto & stream = device->defaultStream();

	std::vector<int>	host_data(100, 33);
	ns::Array<int>		dev_data1(allocator, 100);
	ns::Image1D<int>	dev_data4(allocator, 100);

	stream.fill(dev_data1.data(), 4, dev_data1.size());
	stream.memcpy(dev_data4.data(), dev_data1.data(), dev_data1.size());
	stream.memcpy(host_data.data(), dev_data4.data(), dev_data1.size());

	for (size_t i = 0; i < host_data.size(); i++)
	{
		EXPECT_EQ(host_data[i], 4);
	}
}

TEST(StreamTest, MemcpyImage2D)
{
	auto device = ns::Context::getInstance()->device(0);
	auto allocator = device->defaultAllocator();
	auto & stream = device->defaultStream();

	std::vector<int>	host_data(100, 33);
	ns::Array2D<int>	dev_data2(allocator, 10, 10);
	ns::Image2D<int>	dev_data5(allocator, 10, 10);

	stream.fill(dev_data2.data(), 5, dev_data2.size());
	stream.memcpy2D(dev_data5.data(), dev_data2.data(), dev_data2.pitch(), dev_data2.width(), dev_data2.height());
	stream.memcpy2D(host_data.data(), dev_data2.pitch(), dev_data5.data(), dev_data2.width(), dev_data2.height());

	for (size_t i = 0; i < host_data.size(); i++)
	{
		EXPECT_EQ(host_data[i], 5);
	}
}

TEST(StreamTest, MemcpyImage3D)
{
	auto device = ns::Context::getInstance()->device(0);
	auto allocator = device->defaultAllocator();
	auto & stream = device->defaultStream();

	std::vector<int>	host_data(100, 33);
	ns::Array3D<int>	dev_data3(allocator, 2, 5, 10);
	ns::Image3D<int>	dev_data6(allocator, 2, 5, 10);

	stream.fill(dev_data3.data(), 6, dev_data3.size());
	stream.memcpy3D(dev_data6.data(), dev_data3.data(), dev_data3.pitch(), dev_data3.height(), dev_data3.width(), dev_data3.height(), dev_data3.depth());
	stream.memcpy3D(host_data.data(), dev_data3.pitch(), dev_data3.height(), dev_data6.data(), dev_data3.width(), dev_data3.height(), dev_data3.depth());

	for (size_t i = 0; i < host_data.size(); i++)
	{
		EXPECT_EQ(host_data[i], 6);
	}
}

TEST(StreamTest, MemcpySymbol)
{
	auto device = ns::Context::getInstance()->device(0);
	auto & stream = device->defaultStream();

	std::vector<int> host_data(100, 7);
	stream.memcpyToSymbol(cache, host_data.data(), host_data.size());
	host_data.assign(100, 0);
	stream.memcpyFromSymbol(host_data.data(), cache, host_data.size());

	for (size_t i = 0; i < host_data.size(); i++)
	{
		EXPECT_EQ(host_data[i], 7);
	}
}