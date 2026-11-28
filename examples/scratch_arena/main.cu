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

#include <cstdio>
#include <vector>
#include <cstdlib>

#include <nucleus/array.h>
#include <nucleus/device.h>
#include <nucleus/stream.h>
#include <nucleus/runtime.h>
#include <nucleus/scratch_arena.h>
#include <nucleus/launch_utils.cuh>

/*********************************************************************************
******************************    scratch_arena    *******************************
*********************************************************************************/

__global__ void initialize(dev::Span<float> values, dev::Span<uint32_t> indices, size_t count)
{
	CUDA_for(i, count);

	values[i] = static_cast<float>(i);
	indices[i] = static_cast<uint32_t>(count - i - 1);
}


__global__ void reorder(dev::Span<float> outputs, dev::Span<const float> values, dev::Span<const uint32_t> indices, size_t count)
{
	CUDA_for(i, count);

	outputs[i] = values[indices[i]];
}


struct TemporaryOperation
{
	void execute(ns::Stream & stream, ns::ScratchArena & arena, dev::Span<float> outputs, size_t count)
	{
		NS_ASSERT(outputs.size() >= count);

		size_t required_capacity = 0;
		required_capacity = ns::aligned_end_offset<float>(required_capacity, count);
		required_capacity = ns::aligned_end_offset<uint32_t>(required_capacity, count);

		arena.reserve(required_capacity);
		arena.reuse();

		auto values = arena.allocate<float>(count);
		auto indices = arena.allocate<uint32_t>(count);

		constexpr uint32_t block_size = 128;
		const auto grid_size = ns::ceil_div(count, block_size);

		stream.launch(initialize, grid_size, block_size)(values, indices, count);
		stream.launch(reorder, grid_size, block_size)(outputs, values, indices, count);
	}
};


int main()
{
	auto device = ns::Runtime::device(0);
	auto & stream = device->default_stream();
	auto allocator = device->default_allocator();

	constexpr size_t first_count = 1024;
	constexpr size_t second_count = 512;

	ns::Array<float> outputs(allocator, first_count);
	ns::ScratchArena arena(allocator);
	TemporaryOperation operation;

	operation.execute(stream, arena, outputs, first_count);
	stream.sync();

	const size_t cached_capacity = arena.capacity();

	operation.execute(stream, arena, outputs, second_count);

	std::vector<float> results(second_count);
	stream.memcpy(results.data(), outputs.data(), second_count).sync();

	for (size_t i = 0; i < results.size(); i++)
	{
		if (results[i] != static_cast<float>(second_count - i - 1))
			return EXIT_FAILURE;
	}

	printf("First call scratch arena capacity:  %zu bytes\n", cached_capacity);
	printf("Second call scratch arena capacity: %zu bytes\n", arena.capacity());
	printf("Cached allocation reused:           %s\n", arena.capacity() == cached_capacity ? "yes" : "no");

#ifdef _WIN32
	system("pause");
#endif

	return EXIT_SUCCESS;
}
