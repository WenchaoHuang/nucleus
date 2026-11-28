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
#include "event.h"
#include "stream.h"
#include <future>

namespace NS_NAMESPACE
{
	/*****************************************************************************
	*****************************    ScopedTimer    ******************************
	*****************************************************************************/

	/**
	 *	@brief		RAII-style GPU timer that measures execution duration between stream events.
	 *	@details	Records start/end events on construction/destruction and makes the measured duration
	 *				available through a future. The destructor synchronizes the stream.
	 *	@note		Designed specifically for development/debugging purposes to profile CUDA stream operations.
	 *				Not recommended for production code due to synchronization overhead.
	 */
	class ScopedTimer
	{
		NS_NONCOPYABLE(ScopedTimer)

	public:

		/**
		 *	@brief		Constructs timer and records start event.
		 *	@param[in]	stream - The CUDA stream to time operations on.
		 *	@throw		ErrorCode - If event creation or recording fails.
		 */
		explicit ScopedTimer(Stream & stream) : stream_(stream), end_event_(stream.device()), start_event_(stream.device())
		{
			stream_.record(start_event_);
		}


		/**
		 *	@brief		Constructs timer and records start event.
		 *	@param[in]	stream - The CUDA stream to time operations on.
		 *	@throw		ErrorCode - If event creation or recording fails.
		 */
		explicit ScopedTimer(Stream * stream) : ScopedTimer(*stream) {}


		/**
		 *	@brief		Constructs timer and records start event.
		 *	@param[in]	stream - The CUDA stream to time operations on.
		 *	@note		The caller must keep the stream alive until the timer is destroyed.
		 *	@throw		ErrorCode - If event creation or recording fails.
		 */
		explicit ScopedTimer(std::shared_ptr<Stream> stream) : ScopedTimer(*stream) {}


		/**
		 *	@brief		Returns a future for the elapsed time in nanoseconds.
		 *	@note		Can be called once. Wait for the result only after the timer is destroyed.
		 *	@throw		std::future_error - If the future has already been retrieved.
		 */
		std::future<std::chrono::nanoseconds> get_future() { return promise_.get_future(); }


		/**
		 *	@brief		Records the end event and synchronizes the stream.
		 *	@details	The future receives the elapsed time or the exception from measurement.
		 */
		~ScopedTimer() noexcept
		{
			try
			{
				stream_.record(end_event_).sync();

				promise_.set_value(end_event_ - start_event_);
			}
			catch (...)
			{
				promise_.set_exception(std::current_exception());
			}
		}

	private:

		Stream &									stream_;
		TimedEvent									end_event_;
		TimedEvent									start_event_;
		std::promise<std::chrono::nanoseconds>		promise_;
	};
}
