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

#include <nucleus/span.h>

/*********************************************************************************
********************************    test_span    *********************************
*********************************************************************************/

void test_span()
{
	//!	Test that `ns::Span` is compatible with `std::span`
	static_assert(std::is_same_v<ns::Span<int>, std::span<int>>);
	static_assert(std::is_base_of_v<std::span<const int>, ns::Span<const int>>);

	//!	Test that `ns::Span` can be constructed from an initializer list
	ns::Span<const int> aa = { 1, 2, 3, 4 };

	//!	Test that `ns::Span` can be constructed from a `std::span`
	std::span<const int> bb = aa;

	//!	Test that `ns::Span` can be constructed from a `std::span`
	ns::Span<const int> cc = bb;
}
