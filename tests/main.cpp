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

/*********************************************************************************
***********************************    main    ***********************************
*********************************************************************************/

extern void test_span();
extern void test_event();
extern void test_array();
extern void test_image();
extern void test_graph();
extern void test_logger();
extern void test_device();
extern void test_buffer();
extern void test_stream();
extern void test_runtime();
extern void test_dev_ptr();
extern void test_surface();
extern void test_texture();
extern void test_allocator();
extern void test_buffer_slice();
extern void test_shared_handle();
extern void test_scratch_arena();

int main()
{
	test_runtime();
	test_device();
	test_event();
	test_graph();
	test_allocator();
	test_buffer();
	test_dev_ptr();
	test_array();
	test_image();
	test_stream();
	test_surface();
	test_texture();
	test_buffer_slice();
	test_shared_handle();
	test_scratch_arena();
	test_logger();
	test_span();

	return 0;
}
