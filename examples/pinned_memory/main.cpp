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

/*********************************************************************************
******************************    pinned_memory    *******************************
*********************************************************************************/

template<typename Type> std::string to_string_aligned(Type value, int align)
{
	std::ostringstream oss;

	oss << std::setw(align) << std::right << value;

	return oss.str();
}

int main()
{
	auto device = ns::Runtime::device(0);
	auto hostAlloc = std::make_shared<ns::HostAllocator>();
	auto devAlloc = device->defaultAllocator();

	size_t samples = 10;
	size_t count = 1024 * 1024 * 128;

	ns::Stream			stream(device);
	ns::Array<int>		global(devAlloc, count);
	ns::Array<int>		pinned(hostAlloc, count);
	std::vector<int>	pageable(count);

	printf("\n | bytes | pageable <--> global | pinned <--> global | pinned <--> pageable |\n");
	printf(  " |-------|----------------------|--------------------|----------------------|\n");

	for (size_t test_count = 256; test_count <= count; test_count <<= 1)
	{
		size_t mem_size = test_count * sizeof(int);

		if (mem_size >= 1024 * 1024 * 1024)
			printf(" |%sGB |", to_string_aligned(mem_size / 1024 / 1024 / 1024, 4).c_str());
		else if (mem_size >= 1024 * 1024)
			printf(" |%sMB |", to_string_aligned(mem_size / 1024 / 1024, 4).c_str());
		else if (mem_size >= 1024)
			printf(" |%sKB |", to_string_aligned(mem_size / 1024, 4).c_str());
		else
			printf(" | %sB |", to_string_aligned(mem_size, 4).c_str());

		constexpr size_t ratio = 1000000000;

		{
			ns::ScopedTimer timer(stream, [=](std::chrono::nanoseconds ns) { printf("    %s us |", to_string_aligned(size_t(ns.count() * 1e-3 * ratio) / (2 * ratio * samples), 14).c_str()); });

			for (int i = 0; i < samples; i++)
			{
				stream.memcpy(pageable.data(), global.data(), test_count);
				stream.memcpy(global.data(), pageable.data(), test_count);
			}
		}
		{
			ns::ScopedTimer timer(stream, [=](std::chrono::nanoseconds ns) { printf("    %s us |", to_string_aligned(size_t(ns.count() * 1e-3 * ratio) / (2 * ratio * samples), 12).c_str()); });

			for (int i = 0; i < samples; i++)
			{
				stream.memcpy(pinned.data(), global.data(), test_count);
				stream.memcpy(global.data(), pinned.data(), test_count);
			}
		}
		{
			ns::ScopedTimer timer(stream, [=](std::chrono::nanoseconds ns) { printf("    %s us |\n", to_string_aligned(size_t(ns.count() * 1e-3 * ratio) / (2 * ratio * samples), 14).c_str()); });

			for (int i = 0; i < samples; i++)
			{
				stream.memcpy(pageable.data(), pinned.data(), test_count);
				stream.memcpy(pinned.data(), pageable.data(), test_count);
			}
		}
	}

	printf("\n");
	system("pause");

	return 0;
}