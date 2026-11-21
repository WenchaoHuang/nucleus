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
#include <chrono>
#include <cstddef>

namespace NS_NAMESPACE
{
	/*****************************************************************************
	*********************************    EventRef    *****************************
	*****************************************************************************/

	/**
	 *	@brief		Non-owning wrapper for a CUDA event object.
	 *	@note		Pass this view by value and keep the event alive while using it.
	 */
	class NS_API EventRef
	{
		friend class Event;

	public:

		/**
		 *	@brief		Create an event reference from a native CUDA event handle.
		 *	@param[in]	hEvent - Native CUDA event handle.
		 */
		constexpr EventRef(cudaEvent_t hEvent) noexcept : m_hEvent(hEvent) {}

		constexpr EventRef(const EventRef &) noexcept = default;

		//! @brief A copy borrows the same event. Assignment could rebind the handle of an owning Event.
		EventRef & operator=(const EventRef &) = delete;

		/**
		 *	@brief		Reject a bare `nullptr`. Pass `cudaEvent_t{}` to create an empty reference explicitly.
		 */
		EventRef(std::nullptr_t) = delete;


		/**
		 *	@brief		Prevent the integer literal `0` from silently becoming a null event handle.
		 */
		EventRef(int) = delete;

	public:

		/**
		 *	@brief		Wait for this event to complete.
		 *	@throw		ErrorCode - If CUDA cannot synchronize the event.
		 */
		void sync() const;


		/**
		 *	@brief		Check whether all work captured by this event has completed.
		 *	@return		False while recorded work is pending, true once it has completed.
		 *	@throw		ErrorCode - If CUDA cannot query the event.
		 */
		bool is_done() const;


		/**
		 *	@brief		Retrieve the native `cudaEvent_t` handle.
		 *	@return		The native handle being held by the `EventRef` object.
		 *	@warning	Only for CUDA-based project use.
		 */
		constexpr cudaEvent_t get() const noexcept { return m_hEvent; }


		/**
		 *	@brief		Return `true` if the stored CUDA event handle is non-null.
		 */
		explicit constexpr operator bool() const noexcept { return m_hEvent != nullptr; }


		/**
		 *	@brief		Compares two `EventRef` for equality.
		 *	@return		`true` if both `EventRef` refer to the same `cudaEvent_t` object, `false` otherwise.
		 */
		friend constexpr bool operator==(EventRef, EventRef) noexcept = default;

	private:

		cudaEvent_t		m_hEvent;
	};

	/*****************************************************************************
	********************************    Event    *********************************
	*****************************************************************************/

	/**
	 *	@brief		RAII wrapper for CUDA event object.
	 */
	class NS_API Event : public EventRef
	{
		NS_NONCOPYABLE(Event)

	private:

		friend class TimedEvent;

		/**
		 *	@brief		Create CUDA event object.
		 *	@param[in]	device - Pointer to the device associated with.
		 *	@param[in]	isBlockingSync - Specify that event should use blocking synchronization.
		 *	@param[in]	isDisableTiming - Specify that the created event does not need to record timing data.
		 *	@details	`EventRef::sync()` blocks the calling thread when isBlockingSync is true and busy-waits otherwise.
		 *	@throw		ErrorCode - In case of failure.
		 *	@note		Designed for class `TimedEvent`.
		 */
		explicit Event(Device * device, bool isBlockingSync, bool isDisableTiming);

	public:

		/**
		 *	@brief		Create CUDA event object.
		 *	@param[in]	device - Pointer to the device associated with.
		 *	@param[in]	isBlockingSync - Specify that event should use blocking synchronization.
		 *	@details	`EventRef::sync()` blocks the calling thread when isBlockingSync is true and busy-waits otherwise.
		 *	@throw		ErrorCode - In case of failure.
		 */
		explicit Event(Device * device, bool isBlockingSync = false) : Event(device, isBlockingSync, true) {}


		/**
		 *	@brief		Destroy CUDA event object.
		 */
		~Event() noexcept;

	public:

		/**
		 *	@brief		Return pointer to the device associated with.
		 */
		Device * device() const { return m_device; }

	private:

		Device * const		m_device;

		const bool			m_isBlockingSync;
	};

	/*****************************************************************************
	******************************    TimedEvent    ******************************
	*****************************************************************************/

	/**
	 *	@brief		RAII wrapper for timed CUDA event object.
	 */
	class NS_API TimedEvent : public Event
	{

	public:

		/**
		 *	@brief		Create CUDA event object.
		 *	@param[in]	device - Pointer to the device associated with.
		 *	@param[in]	isBlockingSync - Specify that event should use blocking synchronization.
		 *	@details	`EventRef::sync()` blocks the calling thread when isBlockingSync is true and busy-waits otherwise.
		 *	@throw		ErrorCode - In case of failure.
		 */
		explicit TimedEvent(Device * device, bool isBlockingSync = false) : Event(device, isBlockingSync, false) {}
	};


	/**
	 *	@brief		Compute the elapsed time between two completed events on the same device.
	 *	@return		The elapsed time as `std::chrono::nanoseconds`, with CUDA event timing resolution.
	 *	@throw		ErrorCode - If CUDA cannot measure the interval, including when a recording is still pending.
	 */
	NS_API std::chrono::nanoseconds operator-(const TimedEvent & end, const TimedEvent & start);
}
