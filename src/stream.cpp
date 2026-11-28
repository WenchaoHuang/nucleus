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

#include "event.h"
#include "image.h"
#include "logger.h"
#include "device.h"
#include "stream.h"
#include <cuda_runtime.h>

NS_USING_NAMESPACE

/*********************************************************************************
**********************************    Stream    **********************************
*********************************************************************************/

Stream::Stream(Device * device, int priority) : device_(device), stream_(nullptr), force_sync_(false), priority_(priority)
{
	device->set_current();

	//auto priority_range = device->getStreamPriorityRange();

	//priority = Math::Clamp(priority, priority_range.least, priority_range.greatest);

	cudaError_t err = cudaStreamCreateWithPriority(&stream_, cudaStreamNonBlocking, -priority);

	if (err != cudaSuccess)
	{
		NS_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();

		throw err;
	}
}


Stream::Stream(Device * device, std::nullptr_t) : device_(device), stream_(nullptr), force_sync_(false), priority_(0)
{

}


Stream & Stream::record(EventRef event)
{
	this->acquire_device_context();

	cudaError_t err = cudaEventRecord(event.handle(), stream_);

	if (err != cudaSuccess)
	{
		NS_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();
	}

	return *this;
}


Stream & Stream::wait_for(EventRef event)
{
	this->acquire_device_context();

	cudaError_t err = cudaStreamWaitEvent(stream_, event.handle(), cudaEventWaitDefault);

	if (err != cudaSuccess)
	{
		NS_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();
	}

	return *this;
}


Stream & Stream::launch_graph(cudaGraphExec_t graph_exec)
{
	this->acquire_device_context();

	cudaError_t err = cudaGraphLaunch(graph_exec, stream_);

	if (err != cudaSuccess)
	{
		NS_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();
	}
	else if (force_sync_)
	{
		this->sync();
	}

	return *this;
}


Stream & Stream::launch_host_func(HostFunc<void> func, void * user_data)
{
	this->acquire_device_context();

	cudaError_t err = cudaLaunchHostFunc(stream_, func, user_data);

	if (err != cudaSuccess)
	{
		NS_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();
	}
	else if (force_sync_)
	{
		this->sync();
	}

	return *this;
}


Stream & Stream::memcpy_linear(void * dst, size_t dst_pitch, size_t dst_height, const void * src, size_t src_pitch, size_t src_height, size_t width, size_t height, size_t depth)
{
	this->acquire_device_context();

	cudaMemcpy3DParms mempcy_params = {};
	mempcy_params.dstPtr	= make_cudaPitchedPtr(dst, dst_pitch, 0, dst_height);
	mempcy_params.srcPtr = make_cudaPitchedPtr(const_cast<void*>(src), src_pitch, 0, src_height);
	mempcy_params.extent = make_cudaExtent(width, height, depth);
	mempcy_params.kind = cudaMemcpyDefault;

	cudaError_t err = cudaMemcpy3DAsync(&mempcy_params, stream_);

	if (err != cudaSuccess)
	{
		NS_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();
	}
	else if (force_sync_)
	{
		this->sync();
	}

	return *this;
}


Stream & Stream::memcpy_linear_image(void * dst, size_t dst_pitch, size_t dst_height, ImageAccessor<void> src_img, size_t width, size_t height, size_t depth)
{
	this->acquire_device_context();

	cudaMemcpy3DParms mempcy_params = {};
	mempcy_params.dstPtr = make_cudaPitchedPtr(dst, dst_pitch, 0, dst_height);
	mempcy_params.srcArray = src_img.handle;
	mempcy_params.srcPos = make_cudaPos(src_img.pos.x, src_img.pos.y, src_img.pos.z);
	mempcy_params.extent = make_cudaExtent(width, height, depth);
	mempcy_params.kind = cudaMemcpyDefault;

	cudaError_t err = cudaMemcpy3DAsync(&mempcy_params, stream_);

	if (err != cudaSuccess)
	{
		NS_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();
	}
	else if (force_sync_)
	{
		this->sync();
	}

	return *this;
}


Stream & Stream::memcpy_image_linear(ImageAccessor<void> dst_img, const void * src, size_t src_pitch, size_t src_height, size_t width, size_t height, size_t depth)
{
	this->acquire_device_context();

	cudaMemcpy3DParms mempcy_params = {};
	mempcy_params.dstArray = dst_img.handle;
	mempcy_params.dstPos = make_cudaPos(dst_img.pos.x, dst_img.pos.y, dst_img.pos.z);
	mempcy_params.srcPtr = make_cudaPitchedPtr(const_cast<void*>(src), src_pitch, 0, src_height);
	mempcy_params.extent = make_cudaExtent(width, height, depth);
	mempcy_params.kind = cudaMemcpyDefault;

	cudaError_t err = cudaMemcpy3DAsync(&mempcy_params, stream_);

	if (err != cudaSuccess)
	{
		NS_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();
	}
	else if (force_sync_)
	{
		this->sync();
	}

	return *this;
}


Stream & Stream::memcpy_image(ImageAccessor<void> dst_img, ImageAccessor<void> src_img, size_t width, size_t height, size_t depth)
{
	this->acquire_device_context();

	cudaMemcpy3DParms mempcy_params = {};
	mempcy_params.dstArray = dst_img.handle;
	mempcy_params.dstPos = make_cudaPos(dst_img.pos.x, dst_img.pos.y, dst_img.pos.z);
	mempcy_params.srcArray = src_img.handle;
	mempcy_params.srcPos = make_cudaPos(src_img.pos.x, src_img.pos.y, src_img.pos.z);
	mempcy_params.extent = make_cudaExtent(width, height, depth);
	mempcy_params.kind = cudaMemcpyDefault;

	cudaError_t err = cudaMemcpy3DAsync(&mempcy_params, stream_);

	if (err != cudaSuccess)
	{
		NS_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();
	}
	else if (force_sync_)
	{
		this->sync();
	}

	return *this;
}


Stream & Stream::memcpy_to_symbol_impl(void * symbol, size_t offset, const void * src, size_t count)
{
	this->acquire_device_context();

	cudaError_t err = cudaMemcpyToSymbolAsync(symbol, src, count, offset, cudaMemcpyHostToDevice, stream_);

	if (err != cudaSuccess)
	{
		NS_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();
	}
	else if (force_sync_)
	{
		this->sync();
	}

	return *this;
}


Stream & Stream::memcpy_from_symbol_impl(void * dst, const void * symbol, size_t offset, size_t count)
{
	this->acquire_device_context();

	cudaError_t err = cudaMemcpyFromSymbolAsync(dst, symbol, count, offset, cudaMemcpyDeviceToHost, stream_);

	if (err != cudaSuccess)
	{
		NS_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();
	}
	else if (force_sync_)
	{
		this->sync();
	}

	return *this;
}


Stream & Stream::memset(void * address, int value, size_t bytes)
{
	this->acquire_device_context();

	cudaError_t err = cudaMemsetAsync(address, value, bytes, stream_);

	if (err != cudaSuccess)
	{
		NS_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();
	}
	else if (force_sync_)
	{
		this->sync();
	}

	return *this;
}


void Stream::sync() const
{
	this->acquire_device_context();

	cudaError_t err = cudaStreamSynchronize(stream_);

	if (err != cudaSuccess)
	{
		NS_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();
	}
}


void Stream::set_force_sync(bool enable)
{
	if (force_sync_ != enable)
	{
		if (enable)
		{
			NS_WARNING_LOG("Force-Sync enabled. This method is intended for debugging only and should NOT be used in production code.");
		}
		else
		{
			NS_INFO_LOG("Force-Sync disabled. Stream will not be forced to synchronize.");
		}

		force_sync_ = enable;
	}
}


bool Stream::is_complete() const
{
	this->acquire_device_context();

	cudaError_t err = cudaStreamQuery(stream_);

	if ((err != cudaSuccess) && (err != cudaErrorNotReady))
	{
		NS_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();
	}

	return err == cudaSuccess;
}


void Stream::acquire_device_context() const
{
	if (stream_ == nullptr)
	{
		device_->set_current();
	}
}


Stream::~Stream() noexcept
{
	if (stream_ != nullptr)
	{
		cudaError_t err = cudaStreamDestroy(stream_);

		if (err != cudaSuccess)
		{
			NS_ERROR_LOG("%s.", cudaGetErrorString(err));

			cudaGetLastError();
		}
	}
}
