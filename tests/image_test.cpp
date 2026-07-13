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
#include <nucleus/image_1d.h>
#include <nucleus/image_2d.h>
#include <nucleus/image_3d.h>
#include <nucleus/image_cube.h>

/*********************************************************************************
********************************    image_test    ********************************
*********************************************************************************/

void image_test()
{
	auto device = ns::Runtime::device(0);
	auto allocator = device->defaultAllocator();

	//=========================    1D    =========================

	ns::Image1D<int> image1(allocator, 128);
	static_assert(image1.format() == ns::Format::Int);
	assert(image1.allocator() == allocator);
	assert(image1.data() != nullptr);
	assert(image1.width() == 128);

	ns::Image1DLayered<float> image2(allocator, 1024, 5);
	static_assert(image2.format() == ns::Format::Float);
	assert(image2.allocator() == allocator);
	assert(image2.data() != nullptr);
	assert(image2.numLayers() == 5);
	assert(image2.width() == 1024);

	ns::Image1DLod<short> image3(allocator, 1024, 3);
	static_assert(image3.format() == ns::Format::Short);
	assert(image3.getLevel(0).width() == 1024);
	assert(image3.getLevel(1).width() == 512);
	assert(image3.getLevel(2).width() == 256);
	assert(image3.allocator() == allocator);
	assert(image3.handle() != nullptr);
	assert(image3.numLevels() == 3);
	assert(image3.width() == 1024);

	ns::Image1DLayeredLod<char> image4(allocator, 1024, 10, 5);
	static_assert(image4.format() == ns::Format::Char);
	assert(image4.getLevel(0).width() == 1024);
	assert(image4.getLevel(1).width() == 512);
	assert(image4.getLevel(2).width() == 256);
	assert(image4.getLevel(3).width() == 128);
	assert(image4.getLevel(4).width() == 64);
	assert(image4.allocator() == allocator);
	assert(image4.handle() != nullptr);
	assert(image4.numLayers() == 10);
	assert(image4.numLevels() == 5);
	assert(image4.width() == 1024);

	//=========================    2D    =========================

	ns::Image2D<int> image5(allocator, 128, 128);
	static_assert(image5.format() == ns::Format::Int);
	assert(image5.allocator() == allocator);
	assert(image5.data() != nullptr);
	assert(image5.height() == 128);
	assert(image5.width() == 128);

	ns::Image2DLayered<float> image6(allocator, 1024, 256, 5);
	static_assert(image6.format() == ns::Format::Float);
	assert(image6.allocator() == allocator);
	assert(image6.data() != nullptr);
	assert(image6.numLayers() == 5);
	assert(image6.width() == 1024);
	assert(image6.height() == 256);

	ns::Image2DLod<short> image7(allocator, 1024, 128, 3);
	static_assert(image7.format() == ns::Format::Short);
	assert(image7.getLevel(0).width() == 1024);
	assert(image7.getLevel(1).width() == 512);
	assert(image7.getLevel(2).width() == 256);
	assert(image7.getLevel(0).height() == 128);
	assert(image7.getLevel(1).height() == 64);
	assert(image7.getLevel(2).height() == 32);
	assert(image7.allocator() == allocator);
	assert(image7.handle() != nullptr);
	assert(image7.numLevels() == 3);
	assert(image7.width() == 1024);
	assert(image7.height() == 128);

	ns::Image2DLayeredLod<char> image8(allocator, 1024, 64, 10, 5);
	static_assert(image8.format() == ns::Format::Char);
	assert(image8.getLevel(0).width() == 1024);
	assert(image8.getLevel(1).width() == 512);
	assert(image8.getLevel(2).width() == 256);
	assert(image8.getLevel(3).width() == 128);
	assert(image8.getLevel(4).width() == 64);
	assert(image8.getLevel(0).height() == 64);
	assert(image8.getLevel(1).height() == 32);
	assert(image8.getLevel(2).height() == 16);
	assert(image8.getLevel(3).height() == 8);
	assert(image8.getLevel(4).height() == 4);
	assert(image8.allocator() == allocator);
	assert(image8.handle() != nullptr);
	assert(image8.numLayers() == 10);
	assert(image8.numLevels() == 5);
	assert(image8.width() == 1024);
	assert(image8.height() == 64);

	//=========================    3D    =========================

	ns::Image3D<int> image9(allocator, 128, 128, 128);
	static_assert(image9.format() == ns::Format::Int);
	assert(image9.allocator() == allocator);
	assert(image9.data() != nullptr);
	assert(image9.height() == 128);
	assert(image9.width() == 128);
	assert(image9.depth() == 128);

	ns::Image3DLod<short> image10(allocator, 1024, 128, 256, 3);
	static_assert(image10.format() == ns::Format::Short);
	assert(image10.getLevel(0).width() == 1024);
	assert(image10.getLevel(1).width() == 512);
	assert(image10.getLevel(2).width() == 256);
	assert(image10.getLevel(0).height() == 128);
	assert(image10.getLevel(1).height() == 64);
	assert(image10.getLevel(2).height() == 32);
	assert(image10.getLevel(0).depth() == 256);
	assert(image10.getLevel(1).depth() == 128);
	assert(image10.getLevel(2).depth() == 64);
	assert(image10.allocator() == allocator);
	assert(image10.handle() != nullptr);
	assert(image10.numLevels() == 3);
	assert(image10.height() == 128);
	assert(image10.width() == 1024);
	assert(image10.depth() == 256);

	//=========================    Cube    =========================

	ns::ImageCube<int> image11(allocator, 128);
	static_assert(image11.format() == ns::Format::Int);
	assert(image11.allocator() == allocator);
	assert(image11.data() != nullptr);
	assert(image11.width() == 128);

	ns::ImageCubeLayered<float> image12(allocator, 1024, 5);
	assert(image12.format() == ns::Format::Float);
	assert(image12.allocator() == allocator);
	assert(image12.data() != nullptr);
	assert(image12.numLayers() == 5);
	assert(image12.width() == 1024);

	ns::ImageCubeLod<short> image13(allocator, 1024, 3);
	static_assert(image13.format() == ns::Format::Short);
	assert(image13.getLevel(0).width() == 1024);
	assert(image13.getLevel(1).width() == 512);
	assert(image13.getLevel(2).width() == 256);
	assert(image13.allocator() == allocator);
	assert(image13.handle() != nullptr);
	assert(image13.numLevels() == 3);
	assert(image13.width() == 1024);

	ns::ImageCubeLayeredLod<char> image14(allocator, 1024, 64, 2);
	static_assert(image14.format() == ns::Format::Char);
	assert(image14.getLevel(0).width() == 1024);
	assert(image14.getLevel(1).width() == 512);
	assert(image14.allocator() == allocator);
	assert(image14.handle() != nullptr);
	assert(image14.numLayers() == 64);
	assert(image14.numLevels() == 2);
	assert(image14.width() == 1024);
}