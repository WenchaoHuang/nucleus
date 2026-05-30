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
#include <nucleus/device.h>
#include <nucleus/context.h>
#include <nucleus/image_1d.h>
#include <nucleus/image_2d.h>
#include <nucleus/image_3d.h>
#include <nucleus/image_cube.h>

/*********************************************************************************
********************************    image_test    ********************************
*********************************************************************************/

TEST(ImageTest, Image1D)
{
	auto device = ns::Context::getInstance()->device(0);
	auto allocator = device->defaultAllocator();

	ns::Image1D<int> image1(allocator, 128);
	static_assert(image1.format() == ns::Format::Int);
	EXPECT_FALSE(image1.isSurfaceLoadStoreSupported());
	EXPECT_EQ(image1.allocator(), allocator);
	EXPECT_NE(image1.data(), nullptr);
	EXPECT_EQ(image1.width(), 128u);

	ns::Image1DLayered<float> image2(allocator, 1024, 5, true);
	static_assert(image2.format() == ns::Format::Float);
	EXPECT_TRUE(image2.isSurfaceLoadStoreSupported());
	EXPECT_EQ(image2.allocator(), allocator);
	EXPECT_NE(image2.data(), nullptr);
	EXPECT_EQ(image2.numLayers(), 5u);
	EXPECT_EQ(image2.width(), 1024u);

	ns::Image1DLod<short> image3(allocator, 1024, 3);
	static_assert(image3.format() == ns::Format::Short);
	EXPECT_EQ(image3.getLevel(0).width(), 1024u);
	EXPECT_EQ(image3.getLevel(1).width(), 512u);
	EXPECT_EQ(image3.getLevel(2).width(), 256u);
	EXPECT_EQ(image3.allocator(), allocator);
	EXPECT_NE(image3.handle(), nullptr);
	EXPECT_EQ(image3.numLevels(), 3u);
	EXPECT_EQ(image3.width(), 1024u);

	ns::Image1DLayeredLod<char> image4(allocator, 1024, 10, 5);
	static_assert(image4.format() == ns::Format::Char);
	EXPECT_EQ(image4.getLevel(0).width(), 1024u);
	EXPECT_EQ(image4.getLevel(1).width(), 512u);
	EXPECT_EQ(image4.getLevel(2).width(), 256u);
	EXPECT_EQ(image4.getLevel(3).width(), 128u);
	EXPECT_EQ(image4.getLevel(4).width(), 64u);
	EXPECT_EQ(image4.allocator(), allocator);
	EXPECT_NE(image4.handle(), nullptr);
	EXPECT_EQ(image4.numLayers(), 10u);
	EXPECT_EQ(image4.numLevels(), 5u);
	EXPECT_EQ(image4.width(), 1024u);
}

TEST(ImageTest, Image2D)
{
	auto device = ns::Context::getInstance()->device(0);
	auto allocator = device->defaultAllocator();

	ns::Image2D<int> image5(allocator, 128, 128);
	static_assert(image5.format() == ns::Format::Int);
	EXPECT_FALSE(image5.isSurfaceLoadStoreSupported());
	EXPECT_EQ(image5.allocator(), allocator);
	EXPECT_NE(image5.data(), nullptr);
	EXPECT_EQ(image5.height(), 128u);
	EXPECT_EQ(image5.width(), 128u);

	ns::Image2DLayered<float> image6(allocator, 1024, 256, 5, true);
	static_assert(image6.format() == ns::Format::Float);
	EXPECT_TRUE(image6.isSurfaceLoadStoreSupported());
	EXPECT_EQ(image6.allocator(), allocator);
	EXPECT_NE(image6.data(), nullptr);
	EXPECT_EQ(image6.numLayers(), 5u);
	EXPECT_EQ(image6.width(), 1024u);
	EXPECT_EQ(image6.height(), 256u);

	ns::Image2DLod<short> image7(allocator, 1024, 128, 3);
	static_assert(image7.format() == ns::Format::Short);
	EXPECT_EQ(image7.getLevel(0).width(), 1024u);
	EXPECT_EQ(image7.getLevel(1).width(), 512u);
	EXPECT_EQ(image7.getLevel(2).width(), 256u);
	EXPECT_EQ(image7.getLevel(0).height(), 128u);
	EXPECT_EQ(image7.getLevel(1).height(), 64u);
	EXPECT_EQ(image7.getLevel(2).height(), 32u);
	EXPECT_EQ(image7.allocator(), allocator);
	EXPECT_NE(image7.handle(), nullptr);
	EXPECT_EQ(image7.numLevels(), 3u);
	EXPECT_EQ(image7.width(), 1024u);
	EXPECT_EQ(image7.height(), 128u);

	ns::Image2DLayeredLod<char> image8(allocator, 1024, 64, 10, 5);
	static_assert(image8.format() == ns::Format::Char);
	EXPECT_EQ(image8.getLevel(0).width(), 1024u);
	EXPECT_EQ(image8.getLevel(1).width(), 512u);
	EXPECT_EQ(image8.getLevel(2).width(), 256u);
	EXPECT_EQ(image8.getLevel(3).width(), 128u);
	EXPECT_EQ(image8.getLevel(4).width(), 64u);
	EXPECT_EQ(image8.getLevel(0).height(), 64u);
	EXPECT_EQ(image8.getLevel(1).height(), 32u);
	EXPECT_EQ(image8.getLevel(2).height(), 16u);
	EXPECT_EQ(image8.getLevel(3).height(), 8u);
	EXPECT_EQ(image8.getLevel(4).height(), 4u);
	EXPECT_EQ(image8.allocator(), allocator);
	EXPECT_NE(image8.handle(), nullptr);
	EXPECT_EQ(image8.numLayers(), 10u);
	EXPECT_EQ(image8.numLevels(), 5u);
	EXPECT_EQ(image8.width(), 1024u);
	EXPECT_EQ(image8.height(), 64u);
}

TEST(ImageTest, Image3D)
{
	auto device = ns::Context::getInstance()->device(0);
	auto allocator = device->defaultAllocator();

	ns::Image3D<int> image9(allocator, 128, 128, 128);
	static_assert(image9.format() == ns::Format::Int);
	EXPECT_FALSE(image9.isSurfaceLoadStoreSupported());
	EXPECT_EQ(image9.allocator(), allocator);
	EXPECT_NE(image9.data(), nullptr);
	EXPECT_EQ(image9.height(), 128u);
	EXPECT_EQ(image9.width(), 128u);
	EXPECT_EQ(image9.depth(), 128u);

	ns::Image3DLod<short> image10(allocator, 1024, 128, 256, 3);
	static_assert(image10.format() == ns::Format::Short);
	EXPECT_EQ(image10.getLevel(0).width(), 1024u);
	EXPECT_EQ(image10.getLevel(1).width(), 512u);
	EXPECT_EQ(image10.getLevel(2).width(), 256u);
	EXPECT_EQ(image10.getLevel(0).height(), 128u);
	EXPECT_EQ(image10.getLevel(1).height(), 64u);
	EXPECT_EQ(image10.getLevel(2).height(), 32u);
	EXPECT_EQ(image10.getLevel(0).depth(), 256u);
	EXPECT_EQ(image10.getLevel(1).depth(), 128u);
	EXPECT_EQ(image10.getLevel(2).depth(), 64u);
	EXPECT_EQ(image10.allocator(), allocator);
	EXPECT_NE(image10.handle(), nullptr);
	EXPECT_EQ(image10.numLevels(), 3u);
	EXPECT_EQ(image10.height(), 128u);
	EXPECT_EQ(image10.width(), 1024u);
	EXPECT_EQ(image10.depth(), 256u);
}

TEST(ImageTest, ImageCube)
{
	auto device = ns::Context::getInstance()->device(0);
	auto allocator = device->defaultAllocator();

	ns::ImageCube<int> image11(allocator, 128);
	static_assert(image11.format() == ns::Format::Int);
	EXPECT_FALSE(image11.isSurfaceLoadStoreSupported());
	EXPECT_EQ(image11.allocator(), allocator);
	EXPECT_NE(image11.data(), nullptr);
	EXPECT_EQ(image11.width(), 128u);

	ns::ImageCubeLayered<float> image12(allocator, 1024, 5, true);
	static_assert(image12.format() == ns::Format::Float);
	EXPECT_TRUE(image12.isSurfaceLoadStoreSupported());
	EXPECT_EQ(image12.allocator(), allocator);
	EXPECT_NE(image12.data(), nullptr);
	EXPECT_EQ(image12.numLayers(), 5u);
	EXPECT_EQ(image12.width(), 1024u);

	ns::ImageCubeLod<short> image13(allocator, 1024, 3);
	static_assert(image13.format() == ns::Format::Short);
	EXPECT_EQ(image13.getLevel(0).width(), 1024u);
	EXPECT_EQ(image13.getLevel(1).width(), 512u);
	EXPECT_EQ(image13.getLevel(2).width(), 256u);
	EXPECT_EQ(image13.allocator(), allocator);
	EXPECT_NE(image13.handle(), nullptr);
	EXPECT_EQ(image13.numLevels(), 3u);
	EXPECT_EQ(image13.width(), 1024u);

	ns::ImageCubeLayeredLod<char> image14(allocator, 1024, 64, 2);
	static_assert(image14.format() == ns::Format::Char);
	EXPECT_EQ(image14.getLevel(0).width(), 1024u);
	EXPECT_EQ(image14.getLevel(1).width(), 512u);
	EXPECT_EQ(image14.allocator(), allocator);
	EXPECT_NE(image14.handle(), nullptr);
	EXPECT_EQ(image14.numLayers(), 64u);
	EXPECT_EQ(image14.numLevels(), 2u);
	EXPECT_EQ(image14.width(), 1024u);
}