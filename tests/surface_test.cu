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
#include <nucleus/surface.h>
#include <nucleus/device_surface.h>
#include <device_launch_parameters.h>

/*********************************************************************************
******************************    test_dev_surf    *******************************
*********************************************************************************/

__global__ void test_device_surface(dev::Surf1D<int> surface0, dev::Surf1D<const int> surface1,
									dev::Surf2D<short> surface2, dev::Surf2D<const short> surface3,
									dev::Surf3D<float> surface4, dev::Surf3D<const float> surface5,
									dev::SurfCube<ns::float2> surface6, dev::SurfCube<const ns::float2> surface7,
									dev::Surf1DLayered<float> surface8, dev::Surf1DLayered<const float> surface9,
									dev::Surf2DLayered<float> surface10, dev::Surf2DLayered<const float> surface11,
									dev::SurfCubeLayered<int> surface12, dev::SurfCubeLayered<const int> surface13)
{
	//==	1D	  ==
	surface0.empty();
	surface0.width();
	surface0.read(10);
	surface0.write(10, 50);
	surface0.handle();

	surface1.empty();
	surface1.width();
	surface1.read(10);
//	surface1.write(10, 50);
	surface1.handle();

	//==	2D	  ==
	surface2.empty();
	surface2.width();
	surface2.height();
	surface2.read(10, 10);
	surface2.write(10, 50, 20);
	surface2.handle();

	surface3.empty();
	surface3.width();
	surface3.height();
	surface3.read(10, 10);
//	surface3.write(10, 50, 20);
	surface3.handle();

	//==	3D	  ==
	surface4.empty();
	surface4.width();
	surface4.depth();
	surface4.height();
	surface4.read(10, 10, 10);
	surface4.write(10.0f, 50, 20, 30);
	surface4.handle();

	surface5.empty();
	surface5.width();
	surface5.depth();
	surface5.height();
	surface5.read(10, 10, 10);
//	surface5.write(10.0f, 50, 20, 30);
	surface5.handle();

	//==	Cube	==
	surface6.empty();
	surface6.width();
	surface6.read(10, 10, 5);
	surface6.write({ 1.0f, 2.0f }, 50, 20, 2);
	surface6.handle();

	surface7.empty();
	surface7.width();
	surface7.read(10, 10, 5);
//	surface7.write({ 10.0f, 2.0f }, 50, 20, 30);
	surface7.handle();

	//==	1D Layered	  ==
	surface8.empty();
	surface8.width();
	surface8.numLayers();
	surface8.read(10, 2);
	surface8.write(10.0f, 50, 0);
	surface8.handle();

	surface9.empty();
	surface9.width();
	surface9.numLayers();
	surface9.read(10, 2);
//	surface9.write(10.0f, 50, 0);
	surface9.handle();

	//==	2D Layered	  ==
	surface10.empty();
	surface10.width();
	surface10.height();
	surface10.numLayers();
	surface10.read(10, 2, 0);
	surface10.write(10.0f, 50, 2, 0);
	surface10.handle();

	surface11.empty();
	surface11.width();
	surface11.height();
	surface11.numLayers();
	surface11.read(10, 2, 0);
//	surface11.write(10.0f, 50, 2, 0);
	surface11.handle();

	//==	Cube Layered	==
	surface12.empty();
	surface12.width();
	surface12.read(10, 10, 5, 0);
	surface12.write(3, 50, 20, 2, 0);
	surface12.handle();

	surface13.empty();
	surface13.width();
	surface13.read(10, 10, 5, 0);
//	surface13.write(3, 50, 20, 2, 0);
	surface13.handle();
}


void surface_test()
{
	auto device = ns::Runtime::device(0);
	auto allocator = device->defaultAllocator();

	ns::Surface1D<int>	surface0(ns::SharedImage1D<int>(allocator, 128, true));
	surface0.empty();
	surface0.image();
	surface0.accessor();
	surface0.handle();

	ns::Surface2D<short>	surface1(ns::SharedImage2D<short>(allocator, 128, 128, true));
	surface1.empty();
	surface1.image();
	surface1.accessor();
	surface1.handle();

	ns::Surface3D<float>	surface2(ns::SharedImage3D<float>(allocator, 128, 128, 128, true));
	surface2.empty();
	surface2.image();
	surface2.accessor();
	surface2.handle();

	ns::SurfaceCube<ns::float2>	surface3(ns::SharedImageCube<ns::float2>(allocator, 128, true));
	surface3.empty();
	surface3.image();
	surface3.accessor();
	surface3.handle();

	ns::Surface1DLayered<float>	surface4(ns::SharedImage1DLayered<float>(allocator, 128, 8, true));
	surface4.empty();
	surface4.image();
	surface4.accessor();
	surface4.handle();
	
	ns::Surface2DLayered<float>	surface5(ns::SharedImage2DLayered<float>(allocator, 128, 128, 8, true));
	surface5.empty();
	surface5.image();
	surface5.accessor();
	surface5.handle();

	ns::SurfaceCubeLayered<int>	surface6(ns::SharedImageCubeLayered<int>(allocator, 128, 8, true));
	surface6.empty();
	surface6.image();
	surface6.accessor();
	surface6.handle();

	test_device_surface << <1, 1 >> > (surface0, surface0,
									   surface1, surface1,
									   surface2, surface2,
									   surface3, surface3,
									   surface4, surface4,
									   surface5, surface5,
									   surface6, surface6);

	device->sync();
}