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

#include <nucleus/device.h>
#include <nucleus/stream.h>
#include <nucleus/runtime.h>
#include <nucleus/array_1d.h>
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

		size_t requiredCapacity = 0;
		requiredCapacity = ns::aligned_end_offset<float>(requiredCapacity, count);
		requiredCapacity = ns::aligned_end_offset<uint32_t>(requiredCapacity, count);

		arena.reserve(requiredCapacity);
		arena.reuse();

		auto values = arena.allocate<float>(count);
		auto indices = arena.allocate<uint32_t>(count);

		constexpr uint32_t blockSize = 128;
		const auto gridSize = ns::ceil_div(count, blockSize);

		stream.launch(initialize, gridSize, blockSize)(values, indices, count);
		stream.launch(reorder, gridSize, blockSize)(outputs, values, indices, count);
	}
};


int main()
{
	auto device = ns::Runtime::device(0);
	auto & stream = device->defaultStream();
	auto allocator = device->defaultAllocator();

	constexpr size_t firstCount = 1024;
	constexpr size_t secondCount = 512;

	ns::Array<float> outputs(allocator, firstCount);
	ns::ScratchArena arena(allocator);
	TemporaryOperation operation;

	operation.execute(stream, arena, outputs, firstCount);
	stream.sync();

	const size_t cachedCapacity = arena.capacity();

	operation.execute(stream, arena, outputs, secondCount);

	std::vector<float> results(secondCount);
	stream.memcpy(results.data(), outputs.data(), secondCount).sync();

	for (size_t i = 0; i < results.size(); i++)
	{
		if (results[i] != static_cast<float>(secondCount - i - 1))
			return EXIT_FAILURE;
	}

	printf("First call scratch arena capacity:  %zu bytes\n", cachedCapacity);
	printf("Second call scratch arena capacity: %zu bytes\n", arena.capacity());
	printf("Cached allocation reused:           %s\n", arena.capacity() == cachedCapacity ? "yes" : "no");

#ifdef _WIN32
	system("pause");
#endif

	return EXIT_SUCCESS;
}
