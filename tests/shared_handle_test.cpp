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
#include <nucleus/buffer.h>
#include <nucleus/device.h>
#include <nucleus/context.h>

/*********************************************************************************
****************************    shared_handle_test    ****************************
*********************************************************************************/

TEST(SharedHandleTest, BasicOperations)
{
	auto device = ns::Context::getInstance()->device(0);
	auto allocator = device->defaultAllocator();

	ns::SharedBuffer sharedBuffer0;
	ns::SharedBuffer sharedBuffer1 = nullptr;
	ns::SharedBuffer sharedBuffer2(allocator, 100);

	EXPECT_EQ(sharedBuffer2->capacity(), 100u);
	ns::SharedBuffer sharedBuffer3 = std::move(sharedBuffer2);
	EXPECT_EQ(sharedBuffer3->capacity(), 100u);

	ns::SharedBuffer sharedBuffer4 = ns::SharedBuffer{ allocator, 200 };
	EXPECT_EQ(sharedBuffer4->capacity(), 200u);

	ns::SharedBuffer sharedBuffer5 = std::make_unique<ns::Buffer>(allocator, 300);
	EXPECT_EQ(sharedBuffer5->capacity(), 300u);

	ns::SharedBuffer sharedBuffer6 = std::make_shared<ns::Buffer>(allocator, 400);
	EXPECT_EQ(sharedBuffer6->capacity(), 400u);
	EXPECT_TRUE(static_cast<bool>(sharedBuffer6));

	sharedBuffer6.reset();
	EXPECT_EQ(sharedBuffer6, nullptr);
}