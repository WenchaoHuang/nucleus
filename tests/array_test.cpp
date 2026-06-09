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
#include <nucleus/device.h>
#include <nucleus/context.h>
#include <nucleus/array_1d.h>
#include <nucleus/array_2d.h>
#include <nucleus/array_3d.h>

/*********************************************************************************
********************************    array_test    ********************************
*********************************************************************************/

static void test(dev::Ptr<int> a, dev::Ptr2<const float> b, dev::Ptr3<float> c)
{

}


TEST(ArrayTest, Array1D)
{
	auto device = ns::Context::getInstance()->device(0);
	auto allocator = device->defaultAllocator();

	ns::Array<int>		array0;
	ns::Array<int>		array1(allocator, 100);
	ns::Array<int>		array11 = std::move(array1);

	std::vector<ns::Array<int>>		arrArray0;
	std::vector<ns::Array<int>>		arrArray1(200);

	arrArray1[0].resize(allocator, 10);
	arrArray1.resize(20);

	EXPECT_TRUE(array0.empty());
	EXPECT_TRUE(array1.empty());
	EXPECT_FALSE(array11.empty());
	EXPECT_EQ(arrArray1[0].size(), 10u);

	ASSERT_FALSE(array11.empty());
	EXPECT_EQ(array11.size(), 100u);
	EXPECT_EQ(array11.width(), 100u);
	EXPECT_EQ(array11.bytes(), 100 * sizeof(int));
	EXPECT_EQ(array11.pitch(), 100 * sizeof(int));
	EXPECT_NE(array11.releaseBuffer(), nullptr);
	EXPECT_EQ(array11.allocator(), nullptr);
	array11.resize(allocator, 200);
	array11.resize(300);
	array11.resize(300);
	array11.clear();
	array11.ptr();
	EXPECT_EQ(array11.data(), nullptr);
}

TEST(ArrayTest, Array2D)
{
	auto device = ns::Context::getInstance()->device(0);
	auto allocator = device->defaultAllocator();

	ns::Array2D<float>	array2;
	ns::Array2D<float>	array3(allocator, 100, 100);
	ns::Array2D<float>	array33 = std::move(array3);

	ASSERT_FALSE(array33.empty());
	float * data = &array33[0][2];

	EXPECT_EQ(array33.size(), 100u * 100u);
	EXPECT_EQ(array33.bytes(), 100 * 100 * sizeof(float));
	EXPECT_EQ(array33.width(), 100u);
	EXPECT_EQ(array33.pitch(), 100 * sizeof(float));
	EXPECT_EQ(array33.height(), 100u);
	EXPECT_EQ(array33.allocator(), allocator);
	EXPECT_NE(array33.releaseBuffer(), nullptr);
	array33.resize(allocator, 200, 400);
	array33.reshape(100, 800);
	array33.resize(400, 200);
	array33.clear();
	array33.ptr();
	EXPECT_EQ(array33.data(), nullptr);
}

TEST(ArrayTest, Array3D)
{
	auto device = ns::Context::getInstance()->device(0);
	auto allocator = device->defaultAllocator();

	ns::Array3D<float>	array4;
	ns::Array3D<float>	array5(allocator, 100, 100, 100);
	ns::Array3D<float>	array55 = std::move(array5);

	ASSERT_FALSE(array55.empty());
	float * data = &array55[0][0][0];

	EXPECT_EQ(array55.size(), 100u * 100u * 100u);
	EXPECT_EQ(array55.bytes(), 100 * 100 * 100 * sizeof(float));
	EXPECT_EQ(array55.width(), 100u);
	EXPECT_EQ(array55.pitch(), 100 * sizeof(float));
	EXPECT_EQ(array55.depth(), 100u);
	EXPECT_EQ(array55.height(), 100u);
	EXPECT_EQ(array55.allocator(), allocator);
	array55.resize(allocator, 200, 400, 500);
	array55.reshape(200, 500, 400);
	array55.resize(500, 200, 300);
	array55.clear();
	array55.data();
	array55.ptr();
	EXPECT_EQ(array5.data(), nullptr);
}

TEST(ArrayTest, PtrConversion)
{
	auto device = ns::Context::getInstance()->device(0);
	auto allocator = device->defaultAllocator();

	ns::Array<int>		array11(allocator, 100);
	ns::Array2D<float>	array33(allocator, 100, 100);
	ns::Array3D<float>	array55(allocator, 100, 100, 100);

	test(array11, array33, array55);
}