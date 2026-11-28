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
	******************************    Allocator    *******************************
	*****************************************************************************/

	/**
	 *	@brief		An abstract interface to an unbounded set of classes
	 *				encapsulating memory resources accessible to the device.
	 */
	class NS_API NS_NOVTABLE Allocator
	{
		NS_NONCOPYABLE(Allocator)

	public:

		//!	@brief		Construct Allocator.
		Allocator() noexcept = default;

		//!	@brief		Destruct Allocator.
		virtual ~Allocator() noexcept {}

	public:

		/**
		 *	@brief		Allocate storage with a size of at least \p bytes
		 *	@param[in]	bytes - Requested allocation size in bytes.
		 *	@return		Pointer to newly allocated memory.
		 *	@throw		ErrorCode - In case of failure.
		 */
		NS_NODISCARD void * allocate_memory(size_t bytes) { return this->do_allocate_memory(bytes); }


		/**
		 *	@brief		Deallocates memory previously allocated by `Allocator::allocate_memory()`.
		 *	@param[in]	ptr - A pointer to the memory to be deallocated. This must be a pointer obtained from a prior
		 *				call to `Allocator::allocate_memory()`, and the memory it points to must not yet have been deallocated.
		 */
		void deallocate_memory(void * ptr) { this->do_deallocate_memory(ptr); }

	private:

		//!	@brief		Allocates memory.
		virtual void * do_allocate_memory(size_t bytes) = 0;

		//!	@brief		Deallocates memory.
		virtual void do_deallocate_memory(void * ptr) = 0;
	};

	/*****************************************************************************
	****************************    HostAllocator    *****************************
	*****************************************************************************/

	/**
	 *	@brief		Page-locked or pinned memory allocator.
	 *	@note		Since the memory can be accessed directly by the device,
	 *				it can be read or written with much higher bandwidth than pageable memory.
	 *	@note		Page-locked memory is best used sparingly to allocate staging areas for data exchange between host and device.
	 */
	class NS_API HostAllocator : public Allocator
	{

	protected:

		/**
		 *	@brief		Allocate page-locked memory on the host.
		 *	@note		Allocate host memory that is page-locked and accessible to the device.
		 *	@param[in]	bytes - Requested allocation size in bytes.
		 *	@return		Pointer to newly allocated memory.
		 *	@details	Memories allocated by CUDA API are always aligned to 512B.
		 *	@details	If \p bytes is 0, nullptr will be returned.
		 *	@throw		ErrorCode - In case of failure.
		 */
		virtual void * do_allocate_memory(size_t bytes) override;


		/**
		 *	@brief		Deallocate page-locked memory.
		 *	@param[in]	ptr - Memory address returned by HostAllocator::do_allocate.
		 *	@details	If \p ptr is nullptr, no operation is performed.
		 */
		virtual void do_deallocate_memory(void * ptr) override;
	};

	/*****************************************************************************
	***************************    DeviceAllocator    ****************************
	*****************************************************************************/

	/**
	 *	@brief		Device memory allocator.
	 */
	class NS_API DeviceAllocator : public Allocator
	{

	public:

		//!	@brief		Constructs device memory allocator.
		explicit DeviceAllocator(Device * device);

		//!	@brief		Return pointer to the device associated with.
		Device * device() const { return device_; }

	public:

		/**
		 *	@brief		Allocate texture memory on the device.
		 *	@param[in]	format - Texel format of the buffer.
		 *	@param[in]	width - width of the buffer.
		 *	@param[in]	height - height of the buffer.
		 *	@param[in]	depth - depth of the buffer.
		 *	@param[in]	flags - Flags for buffer creation (internal use).
		 *	@throw		ErrorCode - In case of failure.
		 */
		virtual cudaArray_t allocate_texture_memory(Format format, size_t width, size_t height, size_t depth, int flags = 0);


		/**
		 *	@brief		Free texture memory on the device.
		 *	@param[in]	array - Returned by a previous call to DeviceAllocator::allocate_texture_memory().
		 *	@details	If \p array is nullptr, no operation is performed.
		 */
		virtual void deallocate_texture_memory(cudaArray_t array);

	public:

		/**
		 *	@brief		Allocate mipmapped texture memory on the device.
		 *	@param[in]	format - Texel format of the buffer.
		 *	@param[in]	width - width of the buffer.
		 *	@param[in]	height - height of the buffer.
		 *	@param[in]	depth - depth of the buffer.
		 *	@param[in]	num_levels - Number of mipmap levels to allocated.
		 *	@param[in]	flags - Flags for buffer creation (interanl use).
		 *	@throw		ErrorCode - In case of failure.
		 */
		virtual cudaMipmappedArray_t allocate_mipmap_texture_memory(Format format, size_t width, size_t height, size_t depth, unsigned int num_levels, int flags = 0);


		/**
		 *	@brief		Free mipmapped texture memory on the device.
		 *	@param[in]	mipmaped_array - Returned by a previous call to DeviceAllocator::allocate_mipmap_texture_memory().
		 *	@details	If \p mipmaped_array is nullptr, no operation is performed.
		 */
		virtual void deallocate_mipmap_texture_memory(cudaMipmappedArray_t mipmaped_array);

	protected:

		/**
		 *	@brief		Allocate linear memory on device.
		 *	@param[in]	bytes - Requested allocation size in bytes.
		 *	@return		Pointer to the newly allocated memory.
		 *	@details	Memories allocated by CUDA API are always aligned to 512B.
		 *	@details	If \p bytes is 0, nullptr will be returned.
		 *	@throw		ErrorCode - In case of failure.
		 */
		virtual void * do_allocate_memory(size_t bytes) override;


		/**
		 *	@brief		Free memory on device.
		 *	@param[in]	ptr - Memory address returned by `DeviceAllocator::allocate_memory()`.
		 *	@details	If \p ptr is nullptr, no operation is performed.
		 */
		virtual void do_deallocate_memory(void * ptr) override;

	private:

		Device * const		device_;
	};
}
