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

		NS_NONREBINDABLE_HANDLE_REF(EventRef)

	public:

		/**
		 *	@brief		Create an event reference from a native CUDA event handle.
		 *	@param[in]	hEvent - Native CUDA event handle.
		 */
		constexpr EventRef(cudaEvent_t event) noexcept : m_event(event) {}


		/**
		 *	@brief		Return `true` if the stored CUDA event handle is non-null.
		 */
		explicit constexpr operator bool() const noexcept { return m_event != nullptr; }


		/**
		 *	@brief		Compares two `EventRef` for equality.
		 *	@return		`true` if both `EventRef` refer to the same `cudaEvent_t` object, `false` otherwise.
		 */
		friend constexpr bool operator==(EventRef, EventRef) noexcept = default;


		/**
		 *	@brief		Retrieve the native `cudaEvent_t` handle.
		 *	@return		The native handle being held by the `EventRef` object.
		 *	@warning	Only for CUDA-based project use.
		 */
		constexpr cudaEvent_t handle() const noexcept { return m_event; }


		/**
		 *	@brief		Check whether all work captured by this event has completed.
		 *	@return		False while recorded work is pending, true once it has completed.
		 *	@throw		ErrorCode - If CUDA cannot query the event.
		 */
		bool is_done() const;


		/**
		 *	@brief		Wait for this event to complete.
		 *	@throw		ErrorCode - If CUDA cannot synchronize the event.
		 */
		void sync() const;

	private:

		cudaEvent_t		m_event;
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
		 *	@param[in]	flags - Flags to specify the event behavior.
		 *	@details	`EventRef::sync()` blocks the calling thread when `is_blocking_sync` is true and busy-waits otherwise.
		 *	@throw		ErrorCode - In case of failure.
		 *	@note		Designed for class `TimedEvent`.
		 */
		explicit Event(Device * device, int flags);

	public:

		/**
		 *	@brief		Create CUDA event object.
		 *	@param[in]	device - Pointer to the device associated with.
		 *	@param[in]	blocking_sync - Specify that event should use blocking synchronization.
		 *	@details	`EventRef::sync()` blocks the calling thread when blocking_sync is true and busy-waits otherwise.
		 *	@throw		ErrorCode - In case of failure.
		 */
		explicit Event(Device * device, bool blocking_sync = false);


		/**
		 *	@brief		Move constructor.
		 */
		Event(Event && other) noexcept : EventRef(other.release()), m_device(std::exchange(other.m_device, nullptr)) {}


		/**
		 *	@brief		Destroy CUDA event object.
		 */
		~Event() noexcept;

	public:

		/**
		 *	@brief		Return pointer to the device associated with.
		 */
		Device * device() const { return m_device; }


		/**
		 *	@brief		Releases ownership of the native CUDA event handle.
		 *	@return		The native handle, or a null handle if this object is empty.
		 *	@note		The caller is responsible for destroying the returned handle.
		 */
		cudaEvent_t release() noexcept { return std::exchange(m_event, cudaEvent_t{}); }


		/**
		 *	@brief		Move assignment operator.
		 */
		Event & operator=(Event && other) noexcept
		{
			if (this != &other)
			{
				this->~Event();

				m_event = std::exchange(other.m_event, cudaEvent_t{});

				m_device = std::exchange(other.m_device, nullptr);
			}

			return *this;
		}

	private:

		Device *		m_device;
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
		 *	@param[in]	blocking_sync - Specify that event should use blocking synchronization.
		 *	@details	`EventRef::sync()` blocks the calling thread when blocking_sync is true and busy-waits otherwise.
		 *	@throw		ErrorCode - In case of failure.
		 */
		explicit TimedEvent(Device * device, bool blocking_sync = false);
	};


	/**
	 *	@brief		Compute the elapsed time between two completed events on the same device.
	 *	@return		The elapsed time as `std::chrono::nanoseconds`, with CUDA event timing resolution.
	 *	@throw		ErrorCode - If CUDA cannot measure the interval, including when a recording is still pending.
	 */
	NS_API std::chrono::nanoseconds operator-(const TimedEvent & end, const TimedEvent & start);
}
