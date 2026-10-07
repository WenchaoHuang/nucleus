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
#include <nucleus/device_mdspan.h>

/*********************************************************************************
*****************************    test_dev_mdspan    ******************************
*********************************************************************************/

__constant__ dev::Mdspan<int, ns::dynamic_extent, 3> d_const_mdspan;
__constant__ dev::Mdspan<const int, ns::dynamic_extent, 3> d_const_read_only_mdspan;

extern void test_dev_mdspan_host();


template<typename Type, size_t... Extents> NS_CUDA_CALLABLE const Type & deduce_const_mdspan_device(dev::Mdspan<const Type, Extents...> view)
{
	return view.flatten()[0];
}


NS_CUDA_CALLABLE bool test_dev_mdspan_func()
{
	int arr[24]{};
	dev::Mdspan<int, 2, 3, 4> fixed(arr);
	dev::Mdspan<int, ns::dynamic_extent, 3, ns::dynamic_extent> mixed(arr, 2, 4);
	dev::Mdspan<int, ns::dynamic_extent, ns::dynamic_extent, ns::dynamic_extent> dynamic = mixed;
	dev::Mdspan<const int, ns::dynamic_extent, 3, ns::dynamic_extent> const_view = fixed;
	const auto & const_fixed = fixed;
	const dev::Mdspan<const int, 2, 3, 4> & read_only_fixed = fixed;
	const dev::Mdspan<int, ns::dynamic_extent, 3> default_constructor;

	fixed(1, 2, 3) = 7;
	mixed(0, 0, 0) = 9;
	fixed.data()[1] = 11;
	fixed.flatten()[2] = 13;
	dev::Mdspan<int, 24> one_dimension(arr);
	const auto & const_one_dimension = one_dimension;
	one_dimension[3] = 17;
	dev::Mdspan<int, 2, 3, 4> from_dynamic = dynamic;
	dev::Mdspan<int, ns::dynamic_extent, 3, ns::dynamic_extent> all_extents(arr, 2, 3, 4);
	dev::Mdspan<int, ns::dynamic_extent, 3, ns::dynamic_extent> zero_extent(arr, 0, 4);
	const dev::Mdspan<const int, 0, 3> static_zero_extent;
	dev::Mdspan<int, ns::dynamic_extent, 3, ns::dynamic_extent> other(arr, 1, 4);
	other = mixed;

	static_assert(std::is_same_v<decltype(const_fixed(0, 0, 0)), const int&>);
	static_assert(std::is_same_v<decltype(const_fixed.data()), const int*>);
	static_assert(std::is_same_v<decltype(const_fixed.flatten()), dev::Span<const int, 24>>);
	static_assert(std::is_same_v<decltype(fixed.flatten()), dev::Span<int, 24>>);
	static_assert(std::is_same_v<decltype(const_one_dimension[0]), const int&>);

	return default_constructor.empty() && default_constructor.size() == 0
		&& fixed.size() == 24 && fixed.size_bytes() == 24 * sizeof(int)
		&& other.extent(0) == 2 && other.extent(1) == 3 && other.extent(2) == 4
		&& dynamic.stride(0) == 12 && dynamic.stride(1) == 4 && dynamic.stride(2) == 1
		&& const_view(1, 2, 3) == 7 && const_fixed(0, 0, 0) == 9
		&& read_only_fixed(1, 2, 3) == 7 && &deduce_const_mdspan_device(fixed) == arr
		&& &deduce_const_mdspan_device(dynamic) == arr
		&& const_fixed(0, 0, 2) == 13 && const_one_dimension[3] == 17
		&& from_dynamic(1, 2, 3) == 7 && all_extents(1, 2, 3) == 7
		&& dynamic.data() == arr && dynamic.flatten().size() == 24
		&& const_fixed.flatten().data() == arr && const_fixed.flatten()[23] == 7
		&& zero_extent.size() == 0 && !zero_extent.empty() && static_zero_extent.empty();
}


__global__ void test_mdspan_kernel(int * result)
{
	*result = test_dev_mdspan_func() && d_const_mdspan.empty() && d_const_read_only_mdspan.empty();
}


void test_dev_mdspan()
{
	test_dev_mdspan_host();
	NS_ASSERT(test_dev_mdspan_func());

	int result = 0;
	int * device_result = nullptr;
	cudaError_t error = cudaMalloc(&device_result, sizeof(int));
	NS_ASSERT(error == cudaSuccess);
	test_mdspan_kernel << <1, 1 >> > (device_result);
	error = cudaGetLastError();
	NS_ASSERT(error == cudaSuccess);
	error = cudaMemcpy(&result, device_result, sizeof(int), cudaMemcpyDeviceToHost);
	NS_ASSERT(error == cudaSuccess);
	NS_ASSERT(result == 1);
	error = cudaFree(device_result);
	NS_ASSERT(error == cudaSuccess);
}
