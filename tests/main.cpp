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

#include <stdlib.h>

/*********************************************************************************
***********************************    main    ***********************************
*********************************************************************************/

extern void event_test();
extern void array_test();
extern void image_test();
extern void graph_test();
extern void logger_test();
extern void device_test();
extern void buffer_test();
extern void stream_test();
extern void context_test();
extern void dev_ptr_test();
extern void surface_test();
extern void texture_test();
extern void allocator_test();
extern void array_proxy_test();
extern void buffer_view_test();
extern void shared_handle_test();

int main()
{
	context_test();
	device_test();
	event_test();
	graph_test();
	allocator_test();
	buffer_test();
	dev_ptr_test();
	array_test();
	image_test();
	stream_test();
	surface_test();
	texture_test();
	buffer_view_test();
	array_proxy_test();
	shared_handle_test();
	logger_test();

	system("pause");

	return 0;
}