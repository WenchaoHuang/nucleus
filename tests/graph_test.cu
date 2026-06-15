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

#include <nucleus/graph.h>
#include <nucleus/device.h>
#include <nucleus/stream.h>
#include <nucleus/runtime.h>
#include <nucleus/launch_utils.cuh>

/*********************************************************************************
********************************    graph_test    ********************************
*********************************************************************************/

__global__ void Test(unsigned int num, unsigned int num2)
{
	CUDA_for(i, num);
}

__global__ void Test0()
{
	CUDA_for(i, 1);
}

void graph_test()
{
	auto device = ns::Runtime::device(0);
	auto stream = &device->defaultStream();

	std::vector<int>	input(100, 5);
	std::vector<int>	output(100);

	ns::Graph graph;
	graph.restart();
	auto d1 = graph.memcpy(output.data(), input.data(), input.size());
	auto d2 = graph.barrier(d1);
	auto d3 = graph.launch(Test, d2, 1, 128)(2, 10);
	auto d4 = graph.launch(Test, d2, 1, 128)(2, 20);
	auto d5 = graph.launch(Test, d3, 1, 128)(2, 30);
	auto d6 = graph.launch(Test0, { d2, d3 }, 1, 128)();
	auto d7 = graph.memcpy(output.data(), input.data(), 1, { d6, d5 });

	graph.execute(stream);
	graph.execute(stream);
	graph.execute(stream);

	device->sync();

	graph.restart();
	d1 = graph.memcpy(output.data(), input.data(), input.size());
	d2 = graph.barrier(d1);
	d3 = graph.launch(Test, d2, 1, 128)(2, 22);
	d4 = graph.launch(Test, d2, 1, 128)(2, 20);
	d5 = graph.launch(Test, d3, 1, 128)(2, 30);
	d6 = graph.launch(Test0, { d2, d3 }, 1, 128)();
	d7 = graph.memcpy(output.data(), input.data(), 1, d5);
	graph.execute(stream);

	device->sync();
}