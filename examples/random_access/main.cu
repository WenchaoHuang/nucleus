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

#include <random>
#include <string>
#include <vector>
#include <iomanip>
#include <sstream>

#include <nucleus/device.h>
#include <nucleus/stream.h>
#include <nucleus/runtime.h>
#include <nucleus/array_1d.h>
#include <nucleus/allocator.h>
#include <nucleus/scoped_timer.h>
#include <nucleus/launch_utils.cuh>

/*********************************************************************************
******************************    random_access    *******************************
*********************************************************************************/

template<typename Type> std::string to_string_aligned(Type value, int align)
{
	std::ostringstream oss;

	oss << std::setw(align) << std::right << value;

	return oss.str();
}


template<typename Type> __global__ void indexed_copy(dev::Ptr<Type> outputs, dev::Ptr<const Type> inputs, dev::Ptr<const int> indices, int count)
{
	CUDA_for(i, count);

	outputs[i] = inputs[indices[i]];
}


int main()
{
	auto device = ns::Runtime::device(0);
	auto allocator = device->defaultAllocator();
	auto & stream = device->defaultStream();

	printf("\n |       |        sequential       |          random         |");
	printf("\n |       |-------------------------+-------------------------|");
	printf("\n | count |   float3   |   float4   |   float3   |   float4   |");
	printf("\n |-------+-------------------------+-------------------------|\n");

	for (int test_count = 1024; test_count <= 1024 * 1024 * 32; test_count <<= 1)
	{
		if (test_count >= 1024 * 1024 * 1024)
			printf(" | %sG |", to_string_aligned(test_count / 1024 / 1024 / 1024, 4).c_str());
		else if (test_count >= 1024 * 1024)
			printf(" | %sM |", to_string_aligned(test_count / 1024 / 1024, 4).c_str());
		else if (test_count >= 1024)
			printf(" | %sK |", to_string_aligned(test_count / 1024, 4).c_str());
		else
			printf(" |  %s |", to_string_aligned(test_count, 4).c_str());

		size_t samples = 100;
		std::default_random_engine	e;
		std::uniform_real_distribution<float> d;

		std::vector<int>						indices(test_count);
		std::vector<std::pair<float, int>>		keyValues(test_count);

		ns::Array<int>							randomIndices(allocator, test_count);
		ns::Array<int>							sequentialIndices(allocator, test_count);
		ns::Array<float3>						input_float3(allocator, test_count);
		ns::Array<float3>						output_float3(allocator, test_count);
		ns::Array<float4>						input_float4(allocator, test_count);
		ns::Array<float4>						output_float4(allocator, test_count);

		for (int i = 0; i < keyValues.size(); i++)
		{
			keyValues[i].first = d(e);
			keyValues[i].second = i;
			indices[i] = i;
		}

		stream.memcpy(sequentialIndices.data(), indices.data(), test_count).sync();

		std::sort(keyValues.begin(), keyValues.end());

		for (int i = 0; i < keyValues.size(); i++)
		{
			indices[i] = keyValues[i].second;
		}

		stream.memcpy(randomIndices.data(), indices.data(), test_count).sync();

		constexpr int blockSize = 128;
		constexpr size_t ratio = 1000000000;
		{
			ns::ScopedTimer timer(stream, [=](std::chrono::nanoseconds ns) { printf("%s us |", to_string_aligned(size_t(ns.count() * 1e-3 * ratio) / (ratio * samples), 8).c_str()); });

			for (int i = 0; i < samples; i++)
			{
				stream.launch(indexed_copy<float3>, ns::ceil_div(test_count, blockSize), blockSize)(output_float3, input_float3, sequentialIndices, test_count);
			}
		}
		{
			ns::ScopedTimer timer(stream, [=](std::chrono::nanoseconds ns) { printf("%s us |", to_string_aligned(size_t(ns.count() * 1e-3 * ratio) / (ratio * samples), 8).c_str()); });

			for (int i = 0; i < samples; i++)
			{
				stream.launch(indexed_copy<float4>, ns::ceil_div(test_count, blockSize), blockSize)(output_float4, input_float4, sequentialIndices, test_count);
			}
		}
		{
			ns::ScopedTimer timer(stream, [=](std::chrono::nanoseconds ns) { printf("%s us |", to_string_aligned(size_t(ns.count() * 1e-3 * ratio) / (ratio * samples), 8).c_str()); });

			for (int i = 0; i < samples; i++)
			{
				stream.launch(indexed_copy<float3>, ns::ceil_div(test_count, blockSize), blockSize)(output_float3, input_float3, randomIndices, test_count);
			}
		}
		{
			ns::ScopedTimer timer(stream, [=](std::chrono::nanoseconds ns) { printf("%s us |", to_string_aligned(size_t(ns.count() * 1e-3 * ratio) / (ratio * samples), 8).c_str()); });

			for (int i = 0; i < samples; i++)
			{
				stream.launch(indexed_copy<float4>, ns::ceil_div(test_count, blockSize), blockSize)(output_float4, input_float4, randomIndices, test_count);
			}
		}

		printf("\n");
	}

	printf("\n");
	system("pause");

	return 0;
}