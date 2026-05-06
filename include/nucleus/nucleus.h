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
#pragma once

/*********************************************************************************
*********************************    Nucleus     *********************************
*********************************************************************************/

/**
 *	@brief	Master header for the nucleus library.
 *	@note	This file aggregates all public headers of the nucleus library,
 *			including core runtime components (event, graph, logger, stream, device, context, allocator),
 *			utilities (scoped_timer, host_types, array_proxy, vector_types),
 *			and data abstractions (buffer, array, image, surface, texture, sampler).
 *
 *	@note	It is NOT recommended to include this file directly in large or complex projects.
 *			Instead, selectively include only the headers you need to reduce compilation overhead
 *			and improve clarity. This file is intended as a convenience for simple use cases,
 *			prototyping, and quick experimentation.
 */

#include "event.h"
#include "graph.h"
#include "logger.h"
#include "stream.h"
#include "device.h"
#include "context.h"
#include "allocator.h"
#include "scoped_timer.h"

#include "buffer.h"
#include "buffer_view.h"

#include "array_1d.h"
#include "array_2d.h"
#include "array_3d.h"

#include "image_1d.h"
#include "image_2d.h"
#include "image_3d.h"
#include "image_cube.h"

#include "surface.h"
#include "texture.h"
#include "sampler.h"

#include "host_types.h"
#include "array_proxy.h"
#include "vector_types.h"
#include "vector_traits.h"

#ifdef __CUDACC__
	#include "launch_utils.cuh"
#endif