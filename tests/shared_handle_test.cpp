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

#include <nucleus/fwd.h>

/*********************************************************************************
****************************    shared_handle_test    ****************************
*********************************************************************************/

void shared_handle_test()
{
	struct TestObject
	{
		explicit TestObject(size_t value) : value(value) {}

		size_t value;
	};

	using TestHandle = ns::SharedHandle<TestObject>;

	TestHandle handle0;
	TestHandle handle1 = nullptr;
	TestHandle handle2(100);

	assert(handle2->value == 100);
	TestHandle handle3 = std::move(handle2);
	assert(handle3->value == 100);

	TestHandle handle4 = TestHandle{ 200 };
	assert(handle4->value == 200);

	TestHandle handle5 = std::make_unique<TestObject>(300);
	assert(handle5->value == 300);

	TestHandle handle6 = std::make_shared<TestObject>(400);
	assert(handle6->value == 400);
	assert(handle6);

	handle6.reset();
	assert(handle6 == nullptr);
}
