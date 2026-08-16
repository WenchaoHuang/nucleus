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

#include <array>
#include <vector>
#include <nucleus/span.h>

/*********************************************************************************
********************************    test_span    *********************************
*********************************************************************************/

namespace
{
	int sum(ns::Span<const int> values)
	{
		int result = 0;

		for (int value : values)
			result += value;

		return result;
	}

	constexpr bool test_constexpr_span()
	{
		int values[4] = { 1, 2, 3, 4 };
		ns::Span<int, 4> mutableValues(values);
		ns::Span<const int, 4> constValues(values);

		mutableValues[1] = 20;

		return constValues.data() == values && constValues.size() == 4 && constValues[1] == 20;
	}
}

void test_span()
{
	/*****************************************************************************
	**************************    Type selection    ******************************
	*****************************************************************************/

	// Mutable spans must remain exact aliases of their standard counterparts.
	static_assert(std::is_same_v<ns::Span<int>, std::span<int>>);
	static_assert(std::is_same_v<ns::Span<int, 4>, std::span<int, 4>>);

	// Const spans use the adapter while retaining public std::span compatibility.
	static_assert(!std::is_same_v<ns::Span<const int>, std::span<const int>>);
	static_assert(std::is_base_of_v<std::span<const int>, ns::Span<const int>>);
	static_assert(std::is_base_of_v<std::span<const int, 4>, ns::Span<const int, 4>>);

	static_assert(std::is_default_constructible_v<ns::Span<int>>);
	static_assert(std::is_default_constructible_v<ns::Span<const int>>);
	static_assert(std::is_default_constructible_v<ns::Span<const int, 0>>);
	static_assert(!std::is_default_constructible_v<ns::Span<const int, 4>>);

	static_assert(std::is_constructible_v<ns::Span<const int>, const int*, size_t>);
	static_assert(std::is_constructible_v<ns::Span<const int>, std::span<const int>>);
	static_assert(std::is_constructible_v<ns::Span<const int>, std::initializer_list<int>>);
	static_assert(!std::is_constructible_v<ns::Span<int>, std::initializer_list<int>>);
	static_assert(test_constexpr_span());

	/*****************************************************************************
	***********************    Standard construction    *************************
	*****************************************************************************/

	int values[5] = { 1, 2, 3, 4, 5 };
	const int constValues[3] = { 6, 7, 8 };

	// Dynamic mutable spans preserve standard pointer-and-count behavior.
	ns::Span<int> mutableValues(values, 5);
	assert(mutableValues.data() == values);
	assert(mutableValues.size() == 5);

	mutableValues[2] = 30;
	assert(values[2] == 30);

	// The inherited converting constructor provides mutable-to-const conversion.
	ns::Span<const int> convertedValues = mutableValues;
	assert(convertedValues.data() == values);
	assert(convertedValues.size() == mutableValues.size());

	// Array constructors are inherited for both dynamic and fixed const spans.
	ns::Span<const int> dynamicConstValues(values);
	ns::Span<const int, 3> fixedConstValues(constValues);
	assert(dynamicConstValues.data() == values);
	assert(dynamicConstValues.size() == 5);
	assert(fixedConstValues.data() == constValues);
	assert(fixedConstValues.size() == 3);

	std::array<int, 4> arrayValues = { 10, 20, 30, 40 };
	std::vector<int> vectorValues = { 11, 22, 33 };

	ns::Span<int, 4> mutableArray(arrayValues);
	ns::Span<const int> constVector(vectorValues);
	assert(mutableArray.front() == 10);
	assert(mutableArray.back() == 40);
	assert(constVector.size() == vectorValues.size());
	assert(constVector[1] == 22);

	// The base bridge accepts an exact std::span object and standard subviews.
	std::span<const int> standardSpan(values, 5);
	ns::Span<const int> bridgedSpan(standardSpan);
	auto middle = bridgedSpan.subspan(1, 3);
	assert(bridgedSpan.data() == standardSpan.data());
	assert(middle.size() == 3);
	assert(middle.front() == 2);
	assert(middle.back() == 4);

	/*****************************************************************************
	********************    Initializer-list construction    ********************
	*****************************************************************************/

	// The backing array remains alive throughout the call expression.
	assert(sum({ 3, 4, 5 }) == 12);

	// Empty const spans support the same observer behavior as std::span.
	ns::Span<const int> empty;
	assert(empty.empty());
	assert(empty.data() == nullptr);
	assert(empty.size() == 0);
}
