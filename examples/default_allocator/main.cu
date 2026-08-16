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
****************************    default_allocator    *****************************
*********************************************************************************/

__global__ void initialize(dev::Span<int> values, dev::Span<int> offsets, size_t count)
{
	CUDA_for(i, count);

	values[i] = static_cast<int>(i);
	offsets[i] = static_cast<int>(count - i);
}


__global__ void transform(dev::Span<int> outputs, dev::Span<const int> values, dev::Span<const int> offsets, size_t count)
{
	CUDA_for(i, count);

	outputs[i] = values[i] + offsets[i];
}


int main()
{
	constexpr size_t count = 1024;
	constexpr uint32_t blockSize = 128;
	const auto gridSize = ns::ceil_div(count, blockSize);

	// These objects allocate through ns::Runtime::defaultAllocator().
	ns::Array<int> values(count);
	ns::Array<int> outputs(count);
	ns::ScratchArena scratch(sizeof(int) * count);
	auto offsets = scratch.allocate<int>(count);

	auto device = ns::Runtime::device(0);
	auto & stream = device->defaultStream();
	stream.launch(initialize, gridSize, blockSize)(values, offsets, count);
	stream.launch(transform, gridSize, blockSize)(outputs, values, offsets, count);

	std::vector<int> results(count);
	stream.memcpy(results.data(), outputs.data(), count).sync();

	for (int result : results)
	{
		if (result != static_cast<int>(count))
			return EXIT_FAILURE;
	}

	printf("Processed %zu values using the runtime default allocator.\n", count);

#ifdef _WIN32
	system("pause");
#endif

	return EXIT_SUCCESS;
}
