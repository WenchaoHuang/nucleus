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

#include "../example_window.h"

#include <nucleus/device.h>
#include <nucleus/stream.h>
#include <nucleus/runtime.h>
#include <nucleus/array_1d.h>
#include <nucleus/launch_utils.cuh>

/*********************************************************************************
********************************    multi_gpu    *********************************
*********************************************************************************/

struct __align__(8) Complex
{
	float x, y;
};


__device__ Complex complex_sqr(Complex z)
{
	Complex res;
	res.x = (z.x * z.x) - (z.y * z.y);
	res.y = 2.0f * z.x * z.y;
	return res;
}


__global__ void paint_kernel(dev::Ptr<ColorRGB> pixels, int width, float invN, float t, int num, int deviceId)
{
	CUDA_for(tid, num);

	int i = tid % width;
	int j = tid / width;

	Complex c = { -0.8f, cosf(t) * 0.2f };

	float real = (i * invN - 1.0f) * 2.0f;
	float imag = (j * invN - 0.5f) * 2.0f;

	Complex z = { real, imag };
	int iterations = 0;

	while (iterations < 50 && (z.x * z.x + z.y * z.y) < 400.0f)
	{
		Complex z_sq = complex_sqr(z);
		z.x = z_sq.x + c.x;
		z.y = z_sq.y + c.y;
		iterations++;
	}

	float value = 1.0f - iterations * 0.02f;

	unsigned char color255 = value * 255;

	if (deviceId != 0)
	{
		color255 = 255 - value * 255;
	}

	pixels[tid] = ColorRGB{ color255, color255, color255 };
}


int main()
{
	const int n = 512;
	const size_t width = n * 2;
	const size_t height = n;
	const size_t num_pixels = width * height;

	ExampleWindow window("Julia Set", width, height);
	window.show();

	/////////////////////////////////////////////////////////////////

	std::vector<ColorRGB>	h_pixels(width * height);

	if (ns::Runtime::getDevices().size() < 2)
	{
		printf("This example requires at least two CUDA-enabled devices to run properly.\n");

		system("pause");

		return -1;
	}

	//	GUP 0 runtime
	auto device0 = ns::Runtime::device(0);
	auto allocator0 = device0->defaultAllocator();
	auto & stream0 = device0->defaultStream();
	ns::Array<ColorRGB>		d_pixels0(allocator0, width * height);

	// GPU 1 runtime
	auto device1 = ns::Runtime::device(1);
	auto allocator1 = device1->defaultAllocator();
	auto & stream1 = device1->defaultStream();
	ns::Array<ColorRGB>		d_pixels1(allocator1, width * height);

	/////////////////////////////////////////////////////////////////

	int frames = 0;

	using namespace std::chrono_literals;
	auto t0 = std::chrono::steady_clock::now();

	for (int i = 0; i < 1000000; i++)
	{
		if (window.closed())	break;

		window.processEvents();

		////////////////////////////////////////////////////////////

		float t = i * 0.03f;
		constexpr int blockSize = 256;

		//	Although half the computations would suffice here, we keep the full calculation for clearer multi-GPU demonstration purposes.
		stream0.launch(paint_kernel, ns::ceil_div(num_pixels, blockSize), blockSize)(d_pixels0, width, 1.0f / height, t, num_pixels, 0);
		stream1.launch(paint_kernel, ns::ceil_div(num_pixels, blockSize), blockSize)(d_pixels1, width, 1.0f / height, t, num_pixels, 1);

		size_t offset = d_pixels0.size() / 2;
		stream0.memcpy(h_pixels.data(), d_pixels0.data(), d_pixels0.size() / 2);
		stream1.memcpy(h_pixels.data() + offset, d_pixels1.data() + offset, d_pixels0.size() / 2);

		stream0.sync();
		stream1.sync();

		window.updateImage(h_pixels);

		frames++;

		auto t1 = std::chrono::steady_clock::now();

		if (t1 - t0 > 0.5s)
		{
			int fps = frames * 1e9f / (t1 - t0).count();

			std::string text = "Julia Set (" + std::to_string(fps) + " FPS)";

			window.setTitle(text);

			frames = 0;

			t0 = t1;
		}
	};

	return 0;
}