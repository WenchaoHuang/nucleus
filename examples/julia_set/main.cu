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
********************************    julia_set    *********************************
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


__global__ void paint_kernel(dev::Ptr<ColorRGB> pixels, int width, float invN, float t, int num)
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

	auto device = ns::Runtime::device(0);
	auto allocator = device->defaultAllocator();
	auto & stream = device->defaultStream();
	
	std::vector<ColorRGB>	h_pixels(width * height);
	ns::Array<ColorRGB>		d_pixels(allocator, width * height);

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

		stream.launch(paint_kernel, ns::ceil_div(num_pixels, blockSize), blockSize)(d_pixels, width, 1.0f / height, t, num_pixels);
		stream.memcpy(h_pixels.data(), d_pixels.data(), d_pixels.size());
		stream.sync();

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