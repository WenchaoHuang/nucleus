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

#include <cuda_runtime.h>
#include <nucleus/array.h>
#include <nucleus/array_2d.h>
#include <nucleus/array_3d.h>
#include <nucleus/stream.h>
#include <nucleus/device.h>
#include <nucleus/runtime.h>
#include <nucleus/device_span_3d.h>
#include <nucleus/launch_utils.cuh>

/*********************************************************************************
*****************************    test_dev_span_nd    *****************************
*********************************************************************************/

__constant__ dev::Span2D<int, 3> d_const_span2d;
__constant__ dev::Span3D<int, 3, ns::dynamic_extent, 4> d_const_span3d;

namespace
{
	constexpr size_t dyn = ns::dynamic_extent;

	using Fixed2D = dev::Span2D<int, 3, 2>;
	using Fixed3D = dev::Span3D<int, 3, 2, 4>;

	static_assert(sizeof(Fixed2D) == sizeof(int *));
	static_assert(sizeof(Fixed3D) == sizeof(int *));
	static_assert(sizeof(dev::Span2D<int, 3, 3>) == sizeof(int *));
	static_assert(sizeof(dev::Span3D<int, 3, 3, 3>) == sizeof(int *));
	static_assert(sizeof(dev::Span2D<int, 3>) == sizeof(int *) + sizeof(size_t));
	static_assert(sizeof(dev::Span2D<int>) == sizeof(int *) + 2 * sizeof(size_t));
	static_assert(sizeof(dev::Span3D<int, 3, dyn, 4>) == sizeof(int *) + sizeof(size_t));
	static_assert(sizeof(dev::Span3D<int>) == sizeof(int *) + 3 * sizeof(size_t));
	static_assert(std::is_trivially_copyable_v<Fixed2D>);
	static_assert(std::is_trivially_copyable_v<Fixed3D>);
	static_assert(std::is_trivially_copyable_v<dev::Span2D<int>>);
	static_assert(std::is_trivially_copyable_v<dev::Span3D<int>>);
	static_assert(!std::is_default_constructible_v<Fixed2D>);
	static_assert(!std::is_default_constructible_v<Fixed3D>);
	static_assert(std::is_default_constructible_v<dev::Span2D<int, 3>>);
	static_assert(std::is_default_constructible_v<dev::Span3D<int, 3, 0, 4>>);

	static_assert(std::is_convertible_v<Fixed2D, dev::Span2D<int>>);
	static_assert(std::is_convertible_v<Fixed2D, dev::Span2D<const int>>);
	static_assert(std::is_constructible_v<Fixed2D, dev::Span2D<int>>);
	static_assert(!std::is_convertible_v<dev::Span2D<int>, Fixed2D>);
	static_assert(!std::is_constructible_v<Fixed2D, dev::Span2D<int, 4, 2>>);
	static_assert(!std::is_constructible_v<Fixed2D, dev::Span2D<const int>>);
	static_assert(!std::is_constructible_v<dev::Span2D<int>, const int *, size_t, size_t>);
	static_assert(!std::is_constructible_v<dev::Span2D<int>, int *, size_t>);
	static_assert(!std::is_constructible_v<dev::Span2D<int>, int *, float, float>);

	static_assert(std::is_convertible_v<Fixed3D, dev::Span3D<int>>);
	static_assert(std::is_convertible_v<Fixed3D, dev::Span3D<const int>>);
	static_assert(std::is_constructible_v<Fixed3D, dev::Span3D<int>>);
	static_assert(!std::is_convertible_v<dev::Span3D<int>, Fixed3D>);
	static_assert(!std::is_constructible_v<Fixed3D, dev::Span3D<int, 3, 2, 5>>);
	static_assert(!std::is_constructible_v<Fixed3D, dev::Span3D<const int>>);
	static_assert(!std::is_constructible_v<dev::Span3D<int>, const int *, size_t, size_t, size_t>);
	static_assert(!std::is_constructible_v<dev::Span3D<int>, int *, size_t, size_t>);

	static_assert(std::is_same_v<decltype(std::declval<Fixed2D &>()[0]), dev::Span<int, 3>>);
	static_assert(std::is_same_v<decltype(std::declval<const Fixed2D &>()[0]), dev::Span<const int, 3>>);
	static_assert(std::is_same_v<decltype(std::declval<Fixed2D &>()(0, 0)), int &>);
	static_assert(std::is_same_v<decltype(std::declval<const Fixed2D &>()(0, 0)), const int &>);
	static_assert(std::is_same_v<decltype(std::declval<Fixed3D &>()[0]), Fixed2D>);
	static_assert(std::is_same_v<decltype(std::declval<const Fixed3D &>()[0]), dev::Span2D<const int, 3, 2>>);
	static_assert(std::is_same_v<decltype(std::declval<Fixed3D &>()(0, 0, 0)), int &>);
	static_assert(std::is_same_v<decltype(std::declval<const Fixed3D &>()(0, 0, 0)), const int &>);

	constexpr dev::Span2D<int, 3>		  empty2d;
	constexpr dev::Span3D<int, 3, dyn, 4> empty3d;
	static_assert(empty2d.empty() && empty2d.width() == 3 && empty2d.height() == 0);
	static_assert(empty3d.empty() && empty3d.width() == 3 && empty3d.height() == 0 && empty3d.depth() == 4);
	static_assert(empty2d.size() == 0 && empty3d.size_bytes() == 0);

	// Conversion and slicing remain usable in constant expressions.
	constexpr int							  const_values[] = { 1, 2, 3, 4, 5, 6 };
	constexpr dev::Span2D<const int, 3, 2>	  const2d(const_values);
	constexpr dev::Span3D<const int, 3, 2, 1> const3d(const_values);
	constexpr dev::Span2D<const int>		  dynamic2d = const2d;
	constexpr dev::Span3D<const int>		  dynamic3d = const3d;
	static_assert(const2d(2, 1) == 6 && const2d[1][2] == 6);
	static_assert(const3d(2, 1, 0) == 6 && const3d[0][1][2] == 6);
	static_assert(dynamic2d.size() == 6 && dynamic3d.depth() == 1);

	// Shape arithmetic must not overflow a 32-bit intermediate.
	constexpr dev::Span3D<const int> large(nullptr, size_t(65536), size_t(65536), size_t(2));
	static_assert(large.size() == (size_t(1) << 33));

	template<typename Type, size_t W, size_t H>
	NS_CUDA_CALLABLE bool deduce_span2d(dev::Span2D<const Type, W, H> view)
	{
		return view.size() == 6;
	}

	template<typename Type, size_t W, size_t H, size_t D>
	NS_CUDA_CALLABLE bool deduce_span3d(dev::Span3D<const Type, W, H, D> view)
	{
		return view.size() == 24;
	}

	template<size_t W, size_t H> NS_CUDA_CALLABLE dev::Span2D<int, W, H> make_span2d(int * data)
	{
		if constexpr (W == dyn && H == dyn)
			return dev::Span2D<int, W, H>(data, 3, 2);
		else if constexpr (W == dyn)
			return dev::Span2D<int, W, H>(data, 3);
		else if constexpr (H == dyn)
			return dev::Span2D<int, W, H>(data, 2);
		else
			return dev::Span2D<int, W, H>(data);
	}

	template<size_t W, size_t H, size_t D> NS_CUDA_CALLABLE dev::Span3D<int, W, H, D> make_span3d(int * data)
	{
		if constexpr (W == dyn && H == dyn && D == dyn)
			return dev::Span3D<int, W, H, D>(data, 3, 2, 4);
		else if constexpr (W == dyn && H == dyn)
			return dev::Span3D<int, W, H, D>(data, 3, 2);
		else if constexpr (W == dyn && D == dyn)
			return dev::Span3D<int, W, H, D>(data, 3, 4);
		else if constexpr (H == dyn && D == dyn)
			return dev::Span3D<int, W, H, D>(data, 2, 4);
		else if constexpr (W == dyn)
			return dev::Span3D<int, W, H, D>(data, 3);
		else if constexpr (H == dyn)
			return dev::Span3D<int, W, H, D>(data, 2);
		else if constexpr (D == dyn)
			return dev::Span3D<int, W, H, D>(data, 4);
		else
			return dev::Span3D<int, W, H, D>(data);
	}

	template<size_t W, size_t H> NS_CUDA_CALLABLE bool test_span2d_shape()
	{
		int							 data[6]{};
		auto						 view = make_span2d<W, H>(data);
		dev::Span2D<int, W, H>		 full(data, 3, 2);
		dev::Span2D<int>			 dynamic = view;
		dev::Span2D<int, 3, 2>		 fixed(dynamic);
		dev::Span2D<const int>		 readonly = view;
		dev::Span2D<const int, 3, 2> const_fixed(dynamic);
		const auto					 const_view = view;

		if (view.width() != 3 || view.height() != 2 || view.size() != 6 || view.empty() || view.data() != data)
			return false;
		if (view.size_bytes() != sizeof(data) || view.pitch() != 3 * sizeof(int) || !deduce_span2d(view))
			return false;

		for (size_t y = 0; y < 2; ++y)
			for (size_t x = 0; x < 3; ++x)
				view(x, y) = int(10 * y + x);

		for (size_t i = 0; i < 6; ++i)
		{
			const size_t x = i % 3, y = i / 3;
			if (data[i] != int(10 * y + x) || &view[y][x] != data + i || &const_view[y][x] != data + i)
				return false;
			if (readonly(x, y) != data[i] || const_fixed(x, y) != data[i] || fixed(x, y) != full(x, y))
				return false;
		}
		view[1][2] = 42;
		return data[5] == 42;
	}

	template<size_t W, size_t H, size_t D> NS_CUDA_CALLABLE bool test_span3d_shape()
	{
		int								data[24]{};
		auto							view = make_span3d<W, H, D>(data);
		dev::Span3D<int, W, H, D>		full(data, 3, 2, 4);
		dev::Span3D<int>				dynamic = view;
		dev::Span3D<int, 3, 2, 4>		fixed(dynamic);
		dev::Span3D<const int>			readonly = view;
		dev::Span3D<const int, 3, 2, 4> const_fixed(dynamic);
		const auto						const_view = view;

		if (view.width() != 3 || view.height() != 2 || view.depth() != 4 || view.size() != 24 || view.empty() || view.data() != data)
			return false;
		if (view.size_bytes() != sizeof(data) || view.pitch() != 3 * sizeof(int) || !deduce_span3d(view))
			return false;

		for (size_t z = 0; z < 4; ++z)
			for (size_t y = 0; y < 2; ++y)
				for (size_t x = 0; x < 3; ++x)
					view(x, y, z) = int(100 * z + 10 * y + x);

		for (size_t i = 0; i < 24; ++i)
		{
			const size_t x = i % 3, y = (i / 3) % 2, z = i / 6;
			if (data[i] != int(100 * z + 10 * y + x) || &view[z][y][x] != data + i || &const_view[z][y][x] != data + i)
				return false;
			if (readonly(x, y, z) != data[i] || const_fixed(x, y, z) != data[i] || fixed(x, y, z) != full(x, y, z))
				return false;
		}
		view[3][1][2] = 42;
		return data[23] == 42;
	}

	NS_CUDA_CALLABLE void test_span_shapes(dev::Span<int, 12> results)
	{
		dev::Span2D<int, 3>			default2d;
		dev::Span3D<int, 3, dyn, 4> default3d;
		dev::Span2D<int, 0, 2>		zero2d;
		dev::Span3D<int, 3, 0, 4>	zero3d;
		if (!default2d.empty() || default2d.size() != 0 || !default3d.empty() || default3d.size() != 0 ||
			zero2d[1].data() != nullptr || zero3d[3].data() != nullptr)
			return;

		results[0]	= test_span2d_shape<dyn, dyn>();
		results[1]	= test_span2d_shape<3, dyn>();
		results[2]	= test_span2d_shape<dyn, 2>();
		results[3]	= test_span2d_shape<3, 2>();
		results[4]	= test_span3d_shape<dyn, dyn, dyn>();
		results[5]	= test_span3d_shape<3, dyn, dyn>();
		results[6]	= test_span3d_shape<dyn, 2, dyn>();
		results[7]	= test_span3d_shape<dyn, dyn, 4>();
		results[8]	= test_span3d_shape<3, 2, dyn>();
		results[9]	= test_span3d_shape<3, dyn, 4>();
		results[10] = test_span3d_shape<dyn, 2, 4>();
		results[11] = test_span3d_shape<3, 2, 4>();
	}

	__global__ void test_span_shapes_kernel(dev::Span<int, 12> results)
	{
		test_span_shapes(results);
	}

	__global__ void test_span_arrays_kernel(dev::Span2D<int, 3> image, dev::Span3D<int, dyn, 2> volume)
	{
		for (size_t y = 0; y < image.height(); ++y)
			for (size_t x = 0; x < image.width(); ++x)
				image[y][x] = int(10 * y + x);

		for (size_t z = 0; z < volume.depth(); ++z)
			for (size_t y = 0; y < volume.height(); ++y)
				for (size_t x = 0; x < volume.width(); ++x)
					volume[z][y][x] = int(100 * z + 10 * y + x);
	}
}

void test_dev_span_nd()
{
	int results[12]{};
	test_span_shapes(dev::Span<int, 12>(results, 12));
	for (int result : results)
		assert(result == 1);

	auto			 device	   = ns::Runtime::device(0);
	auto			 allocator = device->defaultAllocator();
	auto &			 stream	   = device->defaultStream();
	ns::Array<int>	 device_results(allocator, 12);
	ns::Array2D<int> image(allocator, 3, 2);
	ns::Array3D<int> volume(allocator, 3, 2, 4);
	int				 image_data[6]{};
	int				 volume_data[24]{};

	stream.fill(device_results.data(), -1, 12);
	stream.launch(test_span_shapes_kernel, 1, 1)(device_results.span<12>());
	stream.launch(test_span_arrays_kernel, 1, 1)(image.span<3>(), volume.span<dyn, 2>());
	stream.memcpy(results, device_results.data(), 12);
	stream.memcpy(image_data, image.data(), 6);
	stream.memcpy(volume_data, volume.data(), 24);
	stream.sync();

	const auto error = cudaGetLastError();
	assert(error == cudaSuccess);
	for (int result : results)
		assert(result == 1);
	for (size_t i = 0; i < 6; ++i)
		assert(image_data[i] == int(10 * (i / 3) + i % 3));
	for (size_t i = 0; i < 24; ++i)
		assert(volume_data[i] == int(100 * (i / 6) + 10 * ((i / 3) % 2) + i % 3));
}
