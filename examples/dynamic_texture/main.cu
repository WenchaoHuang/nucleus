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

#define NOMINMAX

#include "../example_window.h"

#include <nucleus/array.h>
#include <nucleus/device.h>
#include <nucleus/stream.h>
#include <nucleus/runtime.h>
#include <nucleus/surface.h>
#include <nucleus/texture.h>
#include <nucleus/array_2d.h>

#include <cuda_runtime_api.h>
#include <vector_functions.hpp>
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


__global__ void paint_kernel(dev::Surf2D<unsigned char> pixels, int width, float invN, float t, int num)
{
	//	Pixel coordinate
	auto px = blockDim.x * blockIdx.x + threadIdx.x;
	auto py = blockDim.y * blockIdx.y + threadIdx.y;

	Complex c = { -0.8f, cosf(t) * 0.2f };

	float real = (px * invN - 1.0f) * 2.0f;
	float imag = (py * invN - 0.5f) * 2.0f;

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

	pixels.write(color255, px, py);
}


__device__ __inline__ float3 operator+(float3 a, float b) { return make_float3(a.x + b, a.y + b, a.z + b); }
__device__ __inline__ float3 operator*(float3 a, float b) { return make_float3(a.x * b, a.y * b, a.z * b); }
__device__ __inline__ float3 operator/(float3 a, float b) { return make_float3(a.x / b, a.y / b, a.z / b); }
__device__ __inline__ float3 operator+(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ __inline__ float3 operator-(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }


namespace math
{
	static constexpr float pi = 3.141592654f;
	__device__ __inline__ float min(float a, float b) { return a < b ? a : b; }
	__device__ __inline__ float max(float a, float b) { return a > b ? a : b; }
	__device__ __inline__ float radians(float degree) { return degree * pi / 180.0f; }
	__device__ __inline__ float clamp(float a, float _min, float _max) { return min(max(_min, a), _max); }
	__device__ __inline__ float dot(float3 a, float3 b) { return (a.x * b.x) + (a.y * b.y) + (a.z * b.z); }
	__device__ __inline__ float3 pow(float3 a, float b) { return make_float3(powf(a.x, b), powf(a.y, b), powf(a.z, b)); }
	__device__ __inline__ float3 cross(float3 a, float3 b) { return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
	__device__ __inline__ float3 normalize(float3 a) { float sqr_len = dot(a, a); return sqr_len > 0.0f ? a / sqrtf(sqr_len) : make_float3(0.0f, 0.0f, 0.0f); }
	__device__ __inline__ float3 clamp(float3 a, float _min, float _max) { return make_float3(clamp(a.x, _min, _max), clamp(a.y, _min, _max), clamp(a.z, _min, _max)); }
}


__global__ void ray_tracing_kernel(dev::Ptr2<ColorRGB> output, dev::Tex2D<float> texture, float t)
{
	//	Pixel coordinate
	auto px = blockDim.x * blockIdx.x + threadIdx.x;
	auto py = blockDim.y * blockIdx.y + threadIdx.y;

	//	Bounds check
	if (px >= output.width())			return;
	if (py >= output.height())			return;

	//	Normalized device coordinate: map to [-1, 1]
	float nx = 2.0f * px / output.width() - 1.0f;
	float ny = 2.0f * py / output.height() - 1.0f;

	//	Configure ray
	float3 eye_pos = make_float3(0.0f, 0.0f, 2.5f);
	float3 lookAt = make_float3(0.0f, 0.0f, 0.0f);
	float3 axis_z = math::normalize(eye_pos - lookAt);
	float3 axis_x = math::normalize(math::cross(make_float3(0, 1, 0), axis_z));
	float3 axis_y = math::normalize(math::cross(axis_z, axis_x));

	float aspect = float(output.width()) / output.height();
	float tan_half_fov = tanf(math::radians(45.0f)) / 2.0f;
	float u = nx * tan_half_fov * aspect;
	float v = ny * tan_half_fov;

	//	ray direction
	float3 ray_dir = math::normalize(axis_x * u - axis_y * v - axis_z);

	//	ray-sphere hit
	float radius = 1.0f;
	float3 center = make_float3(0.0f, 0.0f, 0.0f);
	float3 eye_to_center = center - eye_pos;
	float ray_dot = math::dot(eye_to_center, ray_dir);
	float ray_sphere_dist = sqrt(math::dot(eye_to_center, eye_to_center) - ray_dot * ray_dot);

	//	Background color
	float3 color = make_float3(0.1f, 0.2f, 0.1f);

	//	Hit or miss?
	if (ray_sphere_dist < radius)
	{
		//	Solve: || (eye_pos + ray_dir * ray_time) - center || == radius
		float a = math::dot(ray_dir, ray_dir);
		float b = -2 * math::dot(ray_dir, eye_to_center);
		float c = math::dot(eye_to_center, eye_to_center) - (radius * radius);
		float ray_time = -(b + sqrt(b * b - 4 * a * c)) / (2 * a);
		float3 hit_pt = eye_pos + ray_dir * ray_time;

		//	Shading
		float3 N = math::normalize(hit_pt);
		float3 V = math::normalize(eye_pos - hit_pt);
		float3 L = V;
		float3 H = math::normalize(V + L);		// Half-Way
		float3 base_color = make_float3(1, 1, 1);
		float diffuse = abs(math::dot(N, H));
		float ambient = 0.1f;

		float theta = acosf(hit_pt.y);
		float r = sqrt(hit_pt.x * hit_pt.x + hit_pt.z * hit_pt.z);
		float phi = atan2f(hit_pt.x / r, hit_pt.z / r) + t;

		float2 tex_coord;
		tex_coord.x = 3.0f * phi / math::pi;
		tex_coord.y = 3.0f * theta / math::pi;

		base_color = base_color * texture.fetch(tex_coord.x, tex_coord.y);
		color = base_color * diffuse + ambient;
	}

	color = math::clamp(color * 255.0f, 0.0f, 255.0f);

	ColorRGB color_rgb;
	color_rgb.R = static_cast<unsigned char>(color.x);
	color_rgb.G = static_cast<unsigned char>(color.y);
	color_rgb.B = static_cast<unsigned char>(color.z);
	output[py][px] = color_rgb;
}


int main()
{
	const int n = 512;
	const size_t width = n * 2;
	const size_t height = n;
	auto num_pixels = width * height;

	ExampleWindow window("Dynamic Texture", width, height);
	window.show();

	/////////////////////////////////////////////////////////////////

	auto device = ns::Runtime::device(0);
	auto allocator = device->default_allocator();
	auto & stream = device->default_stream();
	
	auto image = ns::Image2D<unsigned char>(allocator, width, height);

	std::vector<ColorRGB>			h_pixels(width * height);
	ns::Array2D<ColorRGB>			d_pixels(allocator, width, height);
	ns::Surface2D<unsigned char>	surface(image);
	ns::Texture2D<float>			texture(image);

	/////////////////////////////////////////////////////////////////

	int frames = 0;

	using namespace std::chrono_literals;
	auto t0 = std::chrono::steady_clock::now();

	for (int i = 0; i < 1000000; i++)
	{
		if (window.closed())	break;

		window.processEvents();

		////////////////////////////////////////////////////////////

		float t = i * 0.005f;
		constexpr int block_size = 16;

		stream.launch(paint_kernel, { ns::ceil_div(width, block_size), ns::ceil_div(height, block_size) }, { block_size, block_size })(surface, width, 1.0f / height, t, num_pixels);
		stream.launch(ray_tracing_kernel, { ns::ceil_div(width, block_size), ns::ceil_div(height, block_size) }, { block_size, block_size })(d_pixels, texture, t * 0.2f);
		stream.memcpy(h_pixels.data(), d_pixels.data(), d_pixels.size());
		stream.sync();

		window.updateImage(h_pixels);

		frames++;

		auto t1 = std::chrono::steady_clock::now();

		if (t1 - t0 > 0.5s)
		{
			int fps = frames * 1e9f / (t1 - t0).count();

			std::string text = "Dynamic Texture (" + std::to_string(fps) + " FPS)";

			window.setTitle(text);

			frames = 0;

			t0 = t1;
		}
	};

	return 0;
}
