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

#include <span>
#include <cuda_runtime.h>
#include <nucleus/device_span.h>

/*********************************************************************************
******************************    test_dev_span    *******************************
*********************************************************************************/

// Device-side constant span for testing.
__constant__ ns::dev::Span<int> d_const_span;


template<typename Type> void deduce_span_type(std::span<const Type> values) {}
template<typename Type> NS_CUDA_CALLABLE void deduce_span_type(dev::Span<const Type> values) {}


NS_CUDA_CALLABLE void test_dev_span_func()
{
	int arr[3] = { 0, 1, 2 };
	const int const_arr[5] = { 0, 1, 2, 3, 4 };

	int * data = arr;
	const int * const_data = const_arr;

	/*****************************************************************************
	*****************************    Constructors    *****************************
	*****************************************************************************/

	// Default constructor
	constexpr dev::Span<int>				default_constructor_0;
	constexpr dev::Span<int, 0>				default_constructor_1;
	constexpr dev::Span<const int>			default_constructor_2;
	constexpr dev::Span<const int, 0>		default_constructor_3;

	// Default copy constructor
	constexpr dev::Span<int>				default_copy_constructor_0 = default_constructor_0;
	constexpr dev::Span<int, 0>				default_copy_constructor_1 = default_constructor_1;
	constexpr dev::Span<const int>			default_copy_constructor_2 = default_constructor_2;
	constexpr dev::Span<const int, 0>		default_copy_constructor_3 = default_constructor_3;

	// Copy from non-const Span
	constexpr dev::Span<const int>			copy_from_non_const_0 = default_constructor_0;
	constexpr dev::Span<const int, 0>		copy_from_non_const_1 = default_constructor_1;

	// Copy from fixed-extent Span
	constexpr dev::Span<int>				copy_from_fixed_extented_0 = default_constructor_1;
	constexpr dev::Span<const int>			copy_from_fixed_extented_1 = default_constructor_1;
	constexpr dev::Span<const int>			copy_from_fixed_extented_2 = default_constructor_3;

	// Construct form raw pointer and size
	dev::Span<int>							construct_form_pointer_size_0(data, 0);
	dev::Span<const int>					construct_form_pointer_size_1(data, 0);
	dev::Span<const int>					construct_form_pointer_size_2(const_data, 0);

	// Construct from array
	dev::Span<int>							construct_form_array_0(arr, 3);
	dev::Span<const int>					construct_form_array_1(arr, 3);
	dev::Span<const int>					construct_form_array_2(const_arr, 5);
	dev::Span<int, 3>						construct_form_array_3(arr);
	dev::Span<const int, 3>					construct_form_array_4(arr);
	dev::Span<const int, 5>					construct_form_array_5(const_arr);

	/*****************************************************************************
	******************************    Observers    *******************************
	*****************************************************************************/

	static_assert(copy_from_non_const_0.empty());
	static_assert(copy_from_non_const_0.size() == 0);
	static_assert(copy_from_non_const_0.size_bytes() == 0);
	static_assert(copy_from_non_const_0.data() == nullptr);

	NS_ASSERT(!construct_form_array_0.empty());
	NS_ASSERT(construct_form_array_0.size() == 3);
	NS_ASSERT(construct_form_array_0.data() == arr);
	NS_ASSERT(construct_form_array_0.size_bytes() == 3 * sizeof(int));

	/*****************************************************************************
	****************************    Element Access    ****************************
	*****************************************************************************/

	int index = 0;

	for (auto v : construct_form_array_5)
	{
		NS_ASSERT(v == index++);
	}
	NS_ASSERT(index == 5);
	NS_ASSERT(construct_form_array_5[0] == 0);
	NS_ASSERT(construct_form_array_5[1] == 1);
	NS_ASSERT(construct_form_array_5[2] == 2);
	NS_ASSERT(construct_form_array_5[3] == 3);
	NS_ASSERT(construct_form_array_5[4] == 4);
	NS_ASSERT(construct_form_array_5.back() == 4);
	NS_ASSERT(construct_form_array_5.front() == 0);

	/*****************************************************************************
	*******************************    Subspan    ********************************
	*****************************************************************************/

	auto subspan0 = construct_form_array_5.last(2);
	auto subspan1 = construct_form_array_5.first(3);
	auto subspan2 = construct_form_array_5.subspan(1, 2);

	static_assert(std::is_same_v<decltype(subspan0), dev::Span<const int>>);
	static_assert(std::is_same_v<decltype(subspan1), dev::Span<const int>>);
	static_assert(std::is_same_v<decltype(subspan2), dev::Span<const int>>);

	NS_ASSERT(subspan0.size() == 2);
	NS_ASSERT(subspan1.size() == 3);
	NS_ASSERT(subspan2.size() == 2);

	NS_ASSERT(subspan0[0] == construct_form_array_5[3]);
	NS_ASSERT(subspan0[1] == construct_form_array_5[4]);

	NS_ASSERT(subspan1[0] == construct_form_array_5[0]);
	NS_ASSERT(subspan1[1] == construct_form_array_5[1]);
	NS_ASSERT(subspan1[2] == construct_form_array_5[2]);

	NS_ASSERT(subspan2[0] == construct_form_array_5[1]);
	NS_ASSERT(subspan2[1] == construct_form_array_5[2]);

	/*****************************************************************************
	*********************    as_bytes / as_writable_bytes    *********************
	*****************************************************************************/

	auto const_bytes_span = ns::as_bytes(construct_form_array_0);
	NS_ASSERT(const_bytes_span.size_bytes() == construct_form_array_0.size_bytes());
	static_assert(std::is_same_v<decltype(const_bytes_span), dev::Span<const ns::byte>>);

	auto const_bytes_span_fixed_extent = ns::as_bytes(construct_form_array_3);
	static_assert(std::is_same_v<decltype(const_bytes_span_fixed_extent), dev::Span<const ns::byte, 12>>);

	auto bytes_span = ns::as_writable_bytes(construct_form_array_0);
	NS_ASSERT(bytes_span.size_bytes() == construct_form_array_0.size_bytes());
	static_assert(std::is_same_v<decltype(bytes_span), dev::Span<ns::byte>>);

	auto bytes_span_fixed_extent = ns::as_writable_bytes(construct_form_array_3);
	static_assert(std::is_same_v<decltype(bytes_span_fixed_extent), dev::Span<ns::byte, 12>>);

	/*****************************************************************************
	***************************    Deduce span type    ***************************
	*****************************************************************************/
	
//	deduce_span_type(std::span<int>());					//	error!!!
	deduce_span_type<int>(std::span<int>());			//	pass!

	deduce_span_type(construct_form_array_0);			//	pass!
	deduce_span_type<int>(construct_form_array_0);		//	pass!
}


__global__ void test_span_kernel()
{
	test_dev_span_func();
}


void test_dev_span()
{
	//	Test device-side span functionality in host code.
	test_dev_span_func();

	// Launch a kernel to test device-side span functionality.
	test_span_kernel << <1, 1 >> > ();
}
