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

#include <nucleus/device_pointer.h>
#include <device_launch_parameters.h>

/*********************************************************************************
*******************************    test_dev_ptr    *******************************
*********************************************************************************/

NS_CUDA_CALLABLE void test(dev::Ptr<int> arr0, dev::Ptr<const int> arr1,
						   dev::Ptr2<short> arr2, dev::Ptr2<const short> arr3,
						   dev::Ptr3<float> arr4, dev::Ptr3<const float> arr5)
{
//	arr0[0] = 0;
	arr0.data();
	arr0.size();
	arr0.width();
	arr0.empty();
	arr0.pitch();
	arr0 = nullptr;
	arr0 = dev::Ptr<int>(nullptr, 10);
	NS_ASSERT(arr0.width() == 10);
	arr0 = dev::Ptr<int>(nullptr);
	arr0 = nullptr;
	NS_ASSERT(arr0.width() == 0);

//	arr1[0] = 0;
//	auto x = arr1[0];
	arr1.data();
	arr1.size();
	arr1.width();
	arr1.empty();
	arr1.pitch();
	arr1 = nullptr;
	arr1 = dev::Ptr<int>(nullptr, 10);
	NS_ASSERT(arr1.width() == 10);
	arr1 = dev::Ptr<int>(nullptr);
	arr1 = nullptr;
	NS_ASSERT(arr1.width() == 0);

//	arr2[0];
//	arr2[0][0] = 0;
	arr2.data();
	arr2.size();
	arr2.empty();
	arr2.pitch();
	arr2.width();
	arr2.height();
	arr2 = nullptr;
	arr2 = dev::Ptr2<short>(nullptr, 10, 20);
	NS_ASSERT(arr2.width() == 10);
	NS_ASSERT(arr2.height() == 20);
	arr2 = dev::Ptr2<short>(nullptr);
	arr2 = nullptr;
	NS_ASSERT(arr2.size() == 0);

//	arr3[0];
//	arr3[0][0] = 0;
	arr3.data();
	arr3.size();
	arr3.empty();
	arr3.pitch();
	arr3.width();
	arr3.height();
	arr3 = nullptr;
	arr3 = dev::Ptr2<short>(nullptr, 10, 20);
	NS_ASSERT(arr3.width() == 10);
	NS_ASSERT(arr3.height() == 20);
	arr3 = dev::Ptr2<short>(nullptr);
	arr3 = nullptr;
	NS_ASSERT(arr3.size() == 0);

//	arr4[0];
//	arr4[0][0];
//	arr4[0][0][0] = 0;
	arr4.data();
	arr4.size();
	arr4.empty();
	arr4.pitch();
	arr4.width();
	arr4.height();
	arr4 = nullptr;
	arr4 = dev::Ptr3<float>(nullptr, 10, 20, 30);
	NS_ASSERT(arr4.width() == 10);
	NS_ASSERT(arr4.depth() == 30);
	NS_ASSERT(arr4.height() == 20);
	arr4 = dev::Ptr3<float>(nullptr);
	arr4 = nullptr;
	NS_ASSERT(arr4.size() == 0);

//	arr5[0];
//	arr5[0][0];
//	arr5[0][0][0] = 0;
	arr5.data();
	arr5.size();
	arr5.empty();
	arr5.pitch();
	arr5.width();
	arr5.height();
	arr5 = nullptr;
	arr5 = dev::Ptr3<float>(nullptr, 10, 20, 30);
	NS_ASSERT(arr5.width() == 10);
	NS_ASSERT(arr5.depth() == 30);
	NS_ASSERT(arr5.height() == 20);
	arr5 = dev::Ptr3<float>(nullptr);
	arr5 = nullptr;
	NS_ASSERT(arr5.size() == 0);
}


void test_dev_ptr()
{
	dev::Ptr<int> devPtr0;
	dev::Ptr<int> devPtr1 = nullptr;
	dev::Ptr<int> devPtr2(nullptr, 1024);
	ns::ptr_cast<float>(devPtr0);

	dev::Ptr2<short> devPtr3;
	dev::Ptr2<short> devPtr4 = nullptr;
	dev::Ptr2<short> devPtr5(nullptr, 100, 200);
	ns::ptr_cast<short>(devPtr3);

	dev::Ptr3<float> devPtr6;
	dev::Ptr3<float> devPtr7 = nullptr;
	dev::Ptr3<float> devPtr8(nullptr, 100, 200, 300);
	ns::ptr_cast<float>(devPtr6);

	if (devPtr2.empty())
	{
		assert(devPtr2.size() == 1024);
		assert(devPtr2.empty() == true);
		assert(devPtr2.bytes() == 1024 * sizeof(int));
		assert(devPtr2.width() == 1024);
		assert(devPtr2.pitch() == 1024 * sizeof(int));
		assert(devPtr2.data() == nullptr);
		devPtr2 = nullptr;
	}

	if (devPtr3 != devPtr4)
	{
		assert(devPtr4.size() == 0);
		assert(devPtr4.bytes() == 0);
		assert(devPtr4.width() == 0);
		assert(devPtr4.pitch() == 0);
		assert(devPtr4.height() == 0);
		assert(devPtr4.empty() == true);
		assert(devPtr4.data() == nullptr);
		devPtr4 = nullptr;
	}

	if (devPtr8.empty())
	{
		assert(devPtr8.empty() == true);
		assert(devPtr8.size() == 100 * 200 * 300);
		assert(devPtr8.bytes() == 100 * 200 * 300 * sizeof(float));
		assert(devPtr8.width() == 100);
		assert(devPtr8.height() == 200);
		assert(devPtr8.depth() == 300);
		assert(devPtr8.data() == nullptr);
		devPtr8 = nullptr;
	}

	test(devPtr2, devPtr2, devPtr4, devPtr4, devPtr8, devPtr8);
}