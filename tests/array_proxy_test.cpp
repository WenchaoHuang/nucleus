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
#include <nucleus/array_proxy.h>

/*********************************************************************************
*****************************    array_proxy_test    *****************************
*********************************************************************************/

TEST(ArrayProxyTest, CArray)
{
	int a[3] = { 1, 2, 3 };
	ns::ArrayProxy<int> x0(a);

	EXPECT_EQ(x0[0], a[0]);
	EXPECT_EQ(x0[1], a[1]);
	EXPECT_EQ(x0[2], a[2]);
	EXPECT_EQ(x0.data(), a);
	EXPECT_EQ(x0.size(), 3u);
	EXPECT_FALSE(x0.empty());
}

TEST(ArrayProxyTest, StdArray)
{
	std::array<int, 5> b = { 0,1,2,3,4 };
	ns::ArrayProxy<int> x1(b);

	EXPECT_EQ(x1[0], b[0]);
	EXPECT_EQ(x1[1], b[1]);
	EXPECT_EQ(x1[2], b[2]);
	EXPECT_EQ(x1[3], b[3]);
	EXPECT_EQ(x1[4], b[4]);
	EXPECT_EQ(x1.data(), b.data());
	EXPECT_EQ(x1.size(), b.size());
	EXPECT_FALSE(x1.empty());
}

TEST(ArrayProxyTest, StdVector)
{
	std::vector<int> c = { 0,1 };
	ns::ArrayProxy<int> x2(c);

	EXPECT_EQ(x2[0], c[0]);
	EXPECT_EQ(x2[1], c[1]);
	EXPECT_EQ(x2.data(), c.data());
	EXPECT_EQ(x2.size(), c.size());
	EXPECT_FALSE(x2.empty());
}

TEST(ArrayProxyTest, Nullptr)
{
	ns::ArrayProxy<int> x3 = nullptr;

	EXPECT_EQ(x3.size(), 0u);
	EXPECT_EQ(x3.data(), nullptr);
	EXPECT_TRUE(x3.empty());
}

TEST(ArrayProxyTest, InitializerList)
{
	ns::ArrayProxy<int> x4({ 1,2,3 });

	ASSERT_FALSE(x4.empty());
	auto x5 = x4.end();
	auto x6 = x4.data();
	auto x7 = x4.begin();
	auto x8 = x4.front();
	auto x9 = x4.back();
	EXPECT_EQ(x4.size(), 3u);
	EXPECT_EQ(x4[0], 1);
	EXPECT_EQ(x4[1], 2);
	EXPECT_EQ(x4[2], 3);

	for (auto val : x4)
	{

	}
}