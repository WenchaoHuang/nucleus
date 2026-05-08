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
#include "host_types.h"
#include <vector>

namespace NS_NAMESPACE
{
	/*****************************************************************************
	*******************************    Context    ********************************
	*****************************************************************************/

	/**
	 *	@brief		Wrapper for CUDA context object (singleton).
	 */
	class Context
	{
		NS_NONCOPYABLE(Context)

	private:

		//!	@brief		Create CUDA context wrapper.
		NS_API Context();

		//!	@brief		Destroy CUDA context wrapper.
		NS_API ~Context();

	public:

		/**
		 *	@brief		Return a raw pointer to the CUDA context wrapper (singleton).
		 */
		static Context * getInstance()
		{
			static Context s_instance;

			return &s_instance;
		}

	public:

		/**
		 *	@brief		Return the last error from a runtime call.
		 *	@note		Return the last error that has been produced by any of the runtime calls
		 *				in the same host thread and reset it to Error::eSuccess.
		 */
		NS_API static Error_t getLastError() noexcept;


		/**
		 *	@brief		Return a string containing the name of an error code in the enum.
		 *	@note		If the error code is not recognized, "unrecognized error code" is returned.
		 */
		NS_API static const char * getErrorName(Error_t eValue) noexcept;


		/**
		 *	@brief		Return the description string for an error code.
		 *	@note		If the error code is not recognized, "unrecognized error code" is returned.
		 */
		NS_API static const char * getErrorString(Error_t eValue) noexcept;

	public:

		/**
		 *	@brief		Return the latest version of CUDA supported by the driver.
		 */
		Version driverVersion() const { return m_driverVersion; }


		/**
		 *	@brief		Return the version number of the current CUDA Runtime instance.
		 */
		Version runtimeVersion() const { return m_runtimeVersion; }


		/**
		 *	@brief		Return pointer to physical device.
		 */
		Device * device(size_t index) const { return m_cudaDevices[index]; }


		/**
		 *	@brief		Return physical device array.
		 */
		const std::vector<Device*> & getDevices() const { return m_cudaDevices; }

	private:

		Version						m_driverVersion;
		Version						m_runtimeVersion;
		std::vector<Device*>		m_cudaDevices;
	};
}