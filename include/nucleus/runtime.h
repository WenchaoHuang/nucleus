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
#include "version.h"
#include <vector>

namespace NS_NAMESPACE
{
	/*****************************************************************************
	*******************************    Runtime    ********************************
	*****************************************************************************/

	/**
	 *	@brief		Wrapper for CUDA runtime object (singleton).
	 */
	class Runtime
	{
		NS_NONCOPYABLE(Runtime)

	private:

		//!	@brief		Create CUDA runtime wrapper.
		NS_API Runtime();

		//!	@brief		Destroy CUDA runtime wrapper.
		NS_API ~Runtime();

	public:

		/**
		 *	@brief		Return a raw pointer to the CUDA runtime wrapper (singleton).
		 *	@throw		ErrorCode - If CUDA runtime initialization fails.
		 */
		static Runtime * instance()
		{
			static Runtime s_instance;

			return &s_instance;
		}

	public:

		/**
		 *	@brief		Return the last error from a runtime call.
		 *	@note		Return the last error that has been produced by any of the runtime calls
		 *				in the same host thread and reset it to `cudaSuccess`.
		 */
		NS_API static ErrorCode get_last_error() noexcept;


		/**
		 *	@brief		Return a string containing the name of an error code in the enum.
		 *	@note		If the error code is not recognized, "unrecognized error code" is returned.
		 */
		NS_API static const char * get_error_name(ErrorCode error) noexcept;


		/**
		 *	@brief		Return the description string for an error code.
		 *	@note		If the error code is not recognized, "unrecognized error code" is returned.
		 */
		NS_API static const char * get_error_string(ErrorCode error) noexcept;

	public:

		/**
		 *	@brief		Return the version number of the current CUDA Runtime instance.
		 *	@throw		ErrorCode - If CUDA runtime initialization fails.
		 */
		static Version version() { return Runtime::instance()->runtime_version_; }


		/**
		 *	@brief		Return the latest version of CUDA supported by the driver.
		 *	@throw		ErrorCode - If CUDA runtime initialization fails.
		 */
		static Version driver_version() { return Runtime::instance()->driver_version_; }


		/**
		 *	@brief		Return pointer to physical device.
		 *	@throw		ErrorCode - If CUDA runtime initialization fails.
		 */
		static Device * device(size_t index) { return Runtime::instance()->cuda_devices_[index]; }


		/**
		 *	@brief		Return physical device array.
		 *	@throw		ErrorCode - If CUDA runtime initialization fails.
		 */
		static const std::vector<Device*> & devices() { return Runtime::instance()->cuda_devices_; }


		/**
		 *	@brief		Return constant reference of the default device allocator.
		 *	@note		The default device allocator is used for allocating device memory when no allocator is specified.
		 *	@note		The default device allocator is initialized to the default allocator of the first device.
		 *	@throw		ErrorCode - If CUDA runtime initialization fails.
		 */
		static const std::shared_ptr<DeviceAllocator> & default_allocator() { return Runtime::instance()->default_alloc_; }


		/**
		 *	@brief		Set the default device allocator.
		 *	@throw		ErrorCode - If CUDA runtime initialization fails.
		 */
		static void set_default_allocator(std::shared_ptr<DeviceAllocator> alloc) { Runtime::instance()->default_alloc_ = alloc; }

	private:

		Version									driver_version_;
		Version									runtime_version_;
		std::shared_ptr<DeviceAllocator>		default_alloc_;
		std::vector<Device*>					cuda_devices_;
	};
}
