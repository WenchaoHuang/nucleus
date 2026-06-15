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

#include "fwd.h"

namespace NS_NAMESPACE
{
	/*****************************************************************************
	********************************    Device    ********************************
	*****************************************************************************/

	/**
	 *	@brief		Wrapper for CUDA device object.
	 */
	class Device
	{
		friend class Runtime;

	private:

		//!	@brief		Create device object.
		NS_API Device(int, const cudaDeviceProp&);

		//!	@brief		Destroy device object.
		NS_API ~Device() noexcept;

	public:

		/**
		 *	@brief		Trigger initialization of the CUDA context.
		 *	@retval		cudaSuccess - If device's context was initialized successfully.
		 */
		NS_API Error_t init() noexcept;


		/**
		 *	@brief		Return CUDA device ID.
		 */
		int id() const { return m_id; }


		/**
		 *	@brief		Wait for compute device to finish.
		 *	@note		Block until the device has completed all preceding requested tasks.
		 */
		NS_API void sync() const;


		/**
		 *	@brief		Set device to be used for GPU executions.
		 *	@note		Mainly for internal call of Stream::Handle().
		 *	@note		Set device as the current device for the calling host thread.
		 *	@note		This call may be made from any host thread, to any device, and at
		 *				any time.  This function will do no synchronization with the previous
		 *				or new device, and should be considered a very low overhead call.
		 *	@warning	Callling ::cudaSetDevice() in other place is not allowed!
		 */
		NS_API void setCurrent() const;


		/**
		 *	@brief		query the size of free device memory.
		 *	@return		The free amount of memory available for allocation by the device in bytes.
		 */
		NS_API size_t freeMemorySize() const;


		/**
		 *	@brief		Return reference of the default stream.
		 */
		Stream & defaultStream() { return *m_defaultStream.get(); }


		/**
		 *	@brief		Return the device properties.
		 *	@note		Requires CUDA Toolkit.
		 */
		const cudaDeviceProp * properties() const { return m_devProp.get(); }


		/**
		 *	@brief		Return shared pointer to the default allocator.
		 */
		std::shared_ptr<DeviceAllocator> defaultAllocator() { return m_defaultAlloc; }


		/**
		 *	@brief		Returns occupancy for a device function.
		 *	@param[in]	func - Kernel function for which occupancy is calculated
		 *	@param[in]	blockSize - Block size the kernel is intended to be launched with
		 *	@param[in]	dynamicSMemSize - Per-block dynamic shared memory usage intended, in bytes.
		 */
		int OccupancyMaxActiveBlocksPerMultiprocessor(const void * func, int blockSize, size_t dynamicSMemSize = 0);

	private:

		const int									m_id;
		const std::unique_ptr<cudaDeviceProp>		m_devProp;
		const std::shared_ptr<DeviceAllocator>		m_defaultAlloc;
		const std::shared_ptr<Stream>				m_defaultStream;
	};
}