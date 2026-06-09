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
#include <nucleus/context.h>

/*********************************************************************************
*******************************    context_test    *******************************
*********************************************************************************/

TEST(ContextTest, VersionComparison)
{
	EXPECT_LT(ns::Version(10, 2), ns::Version(10, 3));
	EXPECT_GT(ns::Version(10, 4), ns::Version(10, 3));
	EXPECT_EQ(ns::Version(10, 5), ns::Version(10, 5));
	EXPECT_GE(ns::Version(10, 5), ns::Version(10, 5));
	EXPECT_LE(ns::Version(10, 5), ns::Version(10, 5));
}

TEST(ContextTest, GetInstance)
{
	auto context = ns::Context::getInstance();
	auto driverVersion = context->driverVersion();
	auto runtimVersion = context->runtimeVersion();
	auto devices = context->getDevices();
	auto device = context->device(0);
}