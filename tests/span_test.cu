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

#include <algorithm>
#include <nucleus/span.h>
#include <nucleus/device.h>
#include <nucleus/stream.h>
#include <nucleus/runtime.h>
#include <nucleus/array_1d.h>
#include <nucleus/launch_utils.cuh>

/*********************************************************************************
********************************    span_test    *********************************
*********************************************************************************/

//!	Device-side kernel to verify ns::Span compiles and works on GPU.
__global__ void span_test_kernel(int * d_output)
{
	//!	Default construction in device code.
	ns::Span<int> s0;
	ns::Span<const int> s0c;

	//!	Fixed-extent default construction.
	ns::Span<int, 4> s1;

	//!	size() on device.
	if (s1.size() == 4)
		d_output[0] = 1;

	//!	empty() on device.
	if (s0.empty())
		d_output[1] = 2;

	//!	Pointer+size construction on device.
	int local[3] = { 7, 8, 9 };
	ns::Span<int> s2(local, 3);
	if (s2.size() == 3 && s2[1] == 8)
		d_output[2] = 3;
}


//!	Helper to verify read-only access through const span.
static int sum_const_span(ns::Span<const int> s)
{
	int total = 0;

	for (auto v : s)
		total += v;

	return total;
}


void span_test()
{
	/*****************************************************************************
	***************************    Type definitions    ***************************
	*****************************************************************************/

	//!	pointer
	static_assert(std::is_same_v<ns::Span<int>::pointer, int*>);
	static_assert(std::is_same_v<ns::Span<const int>::pointer, const int*>);

	//!	reference
	static_assert(std::is_same_v<ns::Span<int>::reference, int&>);
	static_assert(std::is_same_v<ns::Span<const int>::reference, const int&>);

	//!	element_type
	static_assert(std::is_same_v<ns::Span<int>::element_type, int>);
	static_assert(std::is_same_v<ns::Span<const int>::element_type, const int>);

	//!	value_type
	static_assert(std::is_same_v<ns::Span<int>::value_type, int>);
	static_assert(std::is_same_v<ns::Span<const int>::value_type, int>);

	//!	iterator
	static_assert(std::is_same_v<ns::Span<int>::iterator, int*>);
	static_assert(std::is_same_v<ns::Span<const int>::iterator, const int*>);

	/*****************************************************************************
	************************    Fixed-extent ns::Span    *************************
	*****************************************************************************/

	//!	size() for fixed extent.
	static_assert(ns::Span<int, 10>::size() == 10);

	//!	Default construction for fixed extent.
	{
		ns::Span<const int, 5> s;
		assert(s.size() == 5);
	}

	//!	Construct from C-array.
	{
		const int arr[4] = { 1, 2, 3, 4 };
		ns::Span<const int, 4> s(arr);
		assert(s.size() == 4);
		assert(s.data() == arr);
		assert(s[0] == 1);
		assert(s[3] == 4);
		assert(s.front() == 1);
		assert(s.back() == 4);
	}

	//!	Construct from `std::array`.
	{
		const std::array<int, 3> arr = { 10, 20, 30 };
		ns::Span<const int, 3> s(arr);
		assert(s[0] == 10);
		assert(s[1] == 20);
		assert(s[2] == 30);
	}

	/*****************************************************************************
	***********************    Dynamic-extent ns::Span    ************************
	*****************************************************************************/

	//!	Default construction.
	{
		ns::Span<int> s;
		assert(s.empty());
		assert(s.size() == 0);
		assert(s.size_bytes() == 0);
		assert(s.data() == nullptr);
	}

	ns::Span<const int> sc;
	assert(sc.empty());

	//!	Construct from pointer + size.
	{
		int data[6] = { 0, 1, 2, 3, 4, 5 };
		ns::Span<int> s(data, 6);
		assert(!s.empty());
		assert(s.size() == 6);
		assert(s.size_bytes() == 6 * sizeof(int));
		assert(s.data() == data);
	}

	//!	Construct from C-array (mutable).
	{
		int data[5] = { 0, 0, 0, 0, 0 };
		ns::Span<int> s(data);
		s[2] = 99;
		assert(data[2] == 99);
	}

	//!	Construct from `std::vector`.
	{
		std::vector<int> vec = { 7, 8, 9 };
		ns::Span<int> s(vec);
		assert(s.size() == 3);
		assert(s[0] == 7);
		assert(s[2] == 9);

		//!	const ns::Span from const `std::vector`.
		const std::vector<int> cvec = { 1, 2 };
		ns::Span<const int> scv(cvec);
		assert(scv.size() == 2);
	}

	//!	Construct from `std::array` (mutable).
	{
		std::array<int, 4> arr = { 0, 0, 0, 0 };
		ns::Span<int> s(arr);
		s[0] = 42;
		assert(arr[0] == 42);
	}

	//!	Construct from `std::initializer_list`.
	{
		ns::Span<const int> s = { 0, 1, 2, 3 };
		assert(s.size() == 4);
		assert(s[0] == 0);
		assert(s[3] == 3);
	}

	/*****************************************************************************
	****************************    Element access    ****************************
	*****************************************************************************/

	{
		int data[4] = { 10, 20, 30, 40 };
		ns::Span<int> s(data, 4);

		assert(s[0] == 10);
		assert(s[2] == 30);
		assert(s.front() == 10);
		assert(s.back() == 40);

		//!	Mutable access.
		s[1] = 25;
		assert(data[1] == 25);
		s.front() = 100;
		assert(data[0] == 100);
		s.back() = 400;
		assert(data[3] == 400);
	}

	//!	Const element access.
	{
		const int data[3] = { 1, 2, 3 };
		ns::Span<const int> s(data, 3);

		assert(s[0] == 1);
		assert(s.front() == 1);
		assert(s.back() == 3);
	}

	/*****************************************************************************
	******************************    Iterators    *******************************
	*****************************************************************************/

	//!	Forward iteration on mutable span.
	{
		int data[4] = { 1, 2, 3, 4 };
		ns::Span<int> s(data, 4);

		int sum = 0;
		for (auto it = s.begin(); it != s.end(); ++it)
			sum += *it;
		assert(sum == 10);

		//!	Range-for on mutable span with modification.
		sum = 0;
		for (auto & v : s)
		{
			sum += v;
			v *= 2;
		}
		assert(sum == 10);
		assert(data[0] == 2);
		assert(data[3] == 8);
	}

	//!	Forward iteration on const span.
	{
		const int data[4] = { 5, 6, 7, 8 };
		ns::Span<const int> s(data, 4);

		int sum = 0;
		for (auto it = s.begin(); it != s.end(); ++it)
			sum += *it;
		assert(sum == 26);

		//!	Range-for on const span.
		sum = 0;
		for (const auto & v : s) sum += v;
		assert(sum == 26);
	}

	//!	cbegin/cend.
	{
		int data[3] = { 1, 2, 3 };
		ns::Span<int> s(data, 3);
		int sum = 0;
		for (auto it = s.cbegin(); it != s.cend(); ++it)
			sum += *it;
		assert(sum == 6);
	}

	//!	STL algorithm compatibility.
	{
		int data[5] = { 3, 1, 4, 1, 5 };
		ns::Span<int> s(data, 5);

		auto minIt = std::min_element(s.begin(), s.end());
		assert(*minIt == 1);
	}

	/*****************************************************************************
	*************************    Implicit conversion    **************************
	*****************************************************************************/

	//!	ns::Span<T> implicitly converts to ns::Span<const T>.
	{
		int data[3] = { 1, 2, 3 };
		ns::Span<int> s(data, 3);
		assert(sum_const_span(s) == 6);
	}

	//!	ns::Span<T> can be assigned to ns::Span<const T>.
	{
		int data[2] = { 5, 10 };
		ns::Span<int> ms(data, 2);
		ns::Span<const int> cs = ms;
		assert(cs[0] == 5);
		assert(cs[1] == 10);
	}

	/*****************************************************************************
	******************************    Subviews    ********************************
	*****************************************************************************/

	//!	first() on mutable span.
	{
		int data[6] = { 0, 1, 2, 3, 4, 5 };
		ns::Span<int> s(data, 6);
		auto sub = s.first(3);
		static_assert(std::is_same_v<decltype(sub), ns::Span<int>>);

		assert(sub.size() == 3);
		assert(sub[0] == 0);
		assert(sub[2] == 2);

		//!	Modifying through subview modifies original.
		sub[0] = 99;
		assert(data[0] == 99);
	}

	//!	last() on const span.
	{
		const int data[5] = { 10, 20, 30, 40, 50 };
		ns::Span<const int> s(data, 5);
		auto sub = s.last(2);
		assert(sub.size() == 2);
		assert(sub[0] == 40);
		assert(sub[1] == 50);
	}

	//!	subspan() with offset only.
	{
		int data[5] = { 1, 2, 3, 4, 5 };
		ns::Span<int> s(data, 5);
		auto sub = s.subspan(2);
		assert(sub.size() == 3);
		assert(sub[0] == 3);
		assert(sub[2] == 5);
	}

	//!	subspan() with offset + count.
	{
		int data[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
		ns::Span<const int> s(data, 8);
		auto sub = s.subspan(3, 3);
		assert(sub.size() == 3);
		assert(sub[0] == 3);
		assert(sub[2] == 5);
	}

	//!	Chain subviews.
	{
		int data[10] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
		ns::Span<int> s(data, 10);
		auto sub = s.subspan(1, 8).first(4).last(2);
		assert(sub.size() == 2);
		assert(sub[0] == 3);
		assert(sub[1] == 4);
	}

	/*****************************************************************************
	******************************    Observers    *******************************
	*****************************************************************************/

	//!	empty().
	{
		ns::Span<int> s0;
		assert(s0.empty());

		int data[1] = { 0 };
		ns::Span<int> s1(data, 1);
		assert(!s1.empty());
	}

	//!	size_bytes().
	{
		int data[4] = { 0 };
		ns::Span<int> s(data, 4);
		assert(s.size_bytes() == 4 * sizeof(int));

		ns::Span<short> ss(nullptr, 0);
		assert(ss.size_bytes() == 0);
	}

	//!	data() returns non-null for non-empty span.
	{
		int data[1] = { 42 };
		ns::Span<int> s(data, 1);
		assert(s.data() != nullptr);
		assert(*s.data() == 42);
	}

	//!	as_bytes() on mutable span → const byte view.
	{
		int data[2] = { 0x01020304, 0x05060708 };
		ns::Span<int> s(data, 2);
		auto bytes = ns::as_bytes(s);
		static_assert(std::is_same_v<decltype(bytes), ns::Span<const ns::byte>>);

		assert(bytes.size() == 2 * sizeof(int));
		//!	Verify byte-level access (little-endian).
		assert(bytes[0] == 0x04);
		assert(bytes[sizeof(int)] == 0x08);
	}

	//!	as_bytes() on const span.
	{
		const int data[1] = { 0x0A0B0C0D };
		ns::Span<const int> s(data, 1);
		auto bytes = ns::as_bytes(s);
		assert(bytes.size() == sizeof(int));
		assert(bytes[0] == 0x0D);
	}

	//!	as_writable_bytes() allows modification through byte view.
	{
		int data[1] = { 0 };
		ns::Span<int> s(data, 1);
		auto bytes = ns::as_writable_bytes(s);
		static_assert(std::is_same_v<decltype(bytes), ns::Span<ns::byte>>);

		assert(bytes.size() == sizeof(int));
		bytes[0] = 0x42;
		assert(data[0] == 0x42);
	}

	//!	as_bytes() on fixed-extent span.
	{
		int data[3] = { 0, 0, 0 };
		ns::Span<int, 3> s(data);
		auto bytes = ns::as_bytes(s);
		assert(bytes.size() == 3 * sizeof(int));
	}

	//!	as_bytes() on empty span.
	{
		ns::Span<int> s;
		auto bytes = ns::as_bytes(s);
		assert(bytes.empty());
		assert(bytes.size() == 0);
	}

	/*****************************************************************************
	****************************    CUDA kernel    *******************************
	*****************************************************************************/

	auto device = ns::Runtime::device(0);
	auto alloc = device->defaultAllocator();
	auto & stream = device->defaultStream();

	//!	Launch device-side test kernel.
	ns::Array<int>		d_output(alloc, 3);
	std::vector<int>	h_output(3, 0);

	stream.fill(d_output.data(), 0, 3);
	stream.launch(span_test_kernel, 1, 1)(d_output);
	stream.memcpy(h_output.data(), d_output.data(), 3);

	assert(h_output[0] == 1);
	assert(h_output[1] == 2);
	assert(h_output[2] == 3);
}