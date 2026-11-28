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

namespace NS_NAMESPACE
{
	template<typename Type> using HostFunc = void(*)(Type*);

	/*****************************************************************************
	********************************    Stream    ********************************
	*****************************************************************************/

	/**
	 *	@brief		Wrapper for CUDA stream object.
	 */
	class Stream
	{
		NS_NONCOPYABLE(Stream)

	private:

		friend class Device;

		/**
		 *	@brief		Create a default Stream.
		 *	@param[in]	device - Device associated with.
		 *	@note		Invoked for class Device only.
		 */
		NS_API explicit Stream(Device * device, std::nullptr_t);

	public:

		/**
		 *	@brief		Create a non-blocking CUDA Stream.
		 *	@param[in]	device - Device associated with.
		 *	@param[in]	priority - Stream priority.
		 */
		NS_API explicit Stream(Device * device, int priority = 0);


		/**
		 *	@brief		Destroy CUDA stream object.
		 */
		NS_API ~Stream() noexcept;

	public:

		/**
		 *	@brief		Wait for stream tasks to complete.
		 *	@note		Wait until this stream has completed all operations.
		 */
		NS_API void sync() const;


		/**
		 *	@brief		Query an asynchronous stream for completion status.
		 *	@retval		True - If all operations in this stream have completed.
		 */
		NS_API bool is_complete() const;


		/**
		 *	@brief		Force synchronization of the CUDA stream for debugging purposes.
		 *	@param[in]	enable - If true, every subsequent launch and memcpy operation will automatically synchronize
		 *				(blocking until completion) until this option is disabled. If false, do not force synchronization.
		 *	@note		This interface is intended for debugging and troubleshooting during development only.
		 *	@warning	It should NOT be used in production code.
		 */
		NS_API void set_force_sync(bool enable);


		/**
		 *	@brief		Return pointer to the device associated with.
		 */
		Device * device() { return device_; }


		/**
		 *	@brief		Return stream priority.
		 */
		int priority() const { return priority_; }


		/**
		 *	@brief		Return CUDA stream type of this object.
		 *	@warning	Only for CUDA-based project.
		 */
		cudaStream_t handle() noexcept { return stream_; }

	public:

		/**
		 *	@brief		Record an event.
		 *	@param[in]	event - Valid event to record.
		 *	@note		Call such as `Event::is_done()` or `Stream::wait_for()` will then examine or wait for completion of the work that was captured.
		 *	@note		Can be called multiple times on the same event and will overwrite the previously captured state.
		 *	@warning	Event and stream must be on the same device.
		 */
		NS_API Stream & record(EventRef event);


		/**
		 *	@brief		Make a compute stream wait on an event.
		 *	@param[in]	event - Valid event to wait on.
		 * 	@retval		Stream - Reference to this stream (enables method chaining).
		 *	@note		Make all future work submitted to this stream wait for all work captured in event.
		 *	@note		Event may be from a different device than this stream.
		 */
		NS_API Stream & wait_for(EventRef event);

	public:

		/**
		 *	@brief		Launches an executable graph in a stream
		 *	@param[in]	graph_exec - Executable graph to launch
		 *	@retval		Stream - Reference to this stream (enables method chaining).
		 *	@note		Each launch is ordered behind both any previous work in this stream and any previous launches of graphExec.
		 *	@example	stream.launch_graph(graph)
		 *	@example	stream.launch_graph(graph).sync()
		 *	@example	stream.launch_graph(graph).launch_graph(graph)
		 */
		NS_API Stream & launch_graph(cudaGraphExec_t graph_exec);


		/**
		 *	@brief		Enqueue a host function call in a stream.
		 *	@param[in]	func - The function to call once preceding stream operations are complete.
		 *	@param[in]	user_data - User-specified data to be passed to the function.
		 *	@retval		Stream - Reference to this stream (enables method chaining).
		 *	@note		Host function will be called from the thread named nvcuda64.dll.
		 *	@note		The function will be called after currently enqueued work and will block work added after it.
		 *	@warning	The host function must not perform any synchronization that may depend on outstanding CUDA work not mandated to run earlier.
		 *	@warning	The host function must not make any CUDA API calls.
		 */
		template<typename Type> Stream & launch_host_func(HostFunc<Type> func, Type * user_data)
		{
			return this->launch_host_func(reinterpret_cast<HostFunc<void>>(func), user_data);
		}
		NS_API Stream & launch_host_func(HostFunc<void> func, void * user_data);


		/**
		 *	@brief		Prepares to launch a CUDA kernel with specified parameters and dependencies.
		 *	@param[in]	func - Device function symbol.
		 *	@param[in]	grid_dim - Grid dimensions.
		 *	@param[in]	block_dim - Block dimensions.
		 *	@param[in]	shared_mem - Number of bytes for shared memory.
		 *	@example	stream.launch(KernelAdd, grid_dim, block_dim, shared_mem)(A, B, C, count);
		 *	@note		The returned lambda is a temporary object that should be used immediately to configure and launch the kernel.
		 *				It encapsulates all necessary information for the kernel launch, including the kernel function, its arguments.
		 *	@warning	Only available in *.cu files (implemented in launch_utils.cuh).
		 */
		template<typename... Args> NS_NODISCARD auto launch(KernelFunc<Args...> func, const dim3 & grid_dim, const dim3 & block_dim, size_t shared_mem = 0);
		
	public:

		/**
		 *	@brief		Copies data between 3D objects.
		 *	@param[in]	dst - Destination memory address.
		 *	@param[in]	dst_pitch - Pitch of destination memory.
		 *	@param[in]	dst_height - Height of destination memory.
		 *	@param[in]	src - Source memory address.
		 *	@param[in]	src_pitch - Pitch of source memory.
		 *	@param[in]	src_height - Height of source memory.
		 *	@param[in]	width - Width of matrix transfer (columns).
		 *	@param[in]	height - Height of matrix transfer (rows).
		 *	@param[in]	depth - Depth of matrix transfer (layers).
		 *	@param[in]	extext - Extent of matrix transfer.
		 *	@retval		Stream - Reference to this stream (enables method chaining).
		 *	@warning	The memory areas may not overlap. \p width must not exceed either \p dst_pitch or \p src_pitch.
		 */
		template<typename Type> Stream & memcpy_3d(Type * dst, size_t dst_pitch, size_t dst_height, const Type * src, size_t src_pitch, size_t src_height, size_t width, size_t height, size_t depth)
		{
			if constexpr (!std::is_same_v<Type, void>)
				return this->memcpy_linear(dst, dst_pitch, dst_height, src, src_pitch, src_height, width * sizeof(Type), height, depth);
			else
				return this->memcpy_linear(dst, dst_pitch, dst_height, src, src_pitch, src_height, width, height, depth);
		}


		/**
		 *	@brief		Copies data between linear memory and image.
		 *	@param[in]	dst - Destination memory address.
		 *	@param[in]	dst_pitch - Pitch of destination memory.
		 *	@param[in]	dst_height - Height of destination memory.
		 *	@param[in]	src_img - Accessor to the source image.
		 *	@param[in]	width - Width of matrix transfer (columns).
		 *	@param[in]	height - Height of matrix transfer (rows).
		 *	@param[in]	depth - Depth of matrix transfer (layers).
		 *	@retval		Stream - Reference to this stream (enables method chaining).
		 */
		template<typename Type> Stream & memcpy_3d(Type * dst, size_t dst_pitch, size_t dst_height, ImageAccessor<Type> src_img, size_t width, size_t height, size_t depth)
		{
			return this->memcpy_linear_image(dst, dst_pitch, dst_height, src_img, width, height, depth);
		}


		/**
		 *	@brief		Copies data between image and linear memory.
		 *	@param[in]	dst_img - Accessor to the destination image.
		 *	@param[in]	src - Source memory address.
		 *	@param[in]	src_pitch - Pitch of source memory.
		 *	@param[in]	src_height - Height of source memory.
		 *	@param[in]	width - Width of matrix transfer (columns).
		 *	@param[in]	height - Height of matrix transfer (rows).
		 *	@param[in]	depth - Depth of matrix transfer (layers).
		 *	@retval		Stream - Reference to this stream (enables method chaining).
		 */
		template<typename Type> Stream & memcpy_3d(ImageAccessor<Type> dst_img, const Type * src, size_t src_pitch, size_t src_height, size_t width, size_t height, size_t depth)
		{
			return this->memcpy_image_linear(dst_img, src, src_pitch, src_height, width, height, depth);
		}


		/**
		 *	@brief		Copies data between images.
		 *	@param[in]	dst_img - Accessor to the destination image.
		 *	@param[in]	src_img - Accessor to the source image.
		 *	@param[in]	width - Width of matrix transfer (columns in bytes).
		 *	@param[in]	height - Height of matrix transfer (rows).
		 *	@param[in]	depth - Depth of matrix transfer (layers).
		 *	@retval		Stream - Reference to this stream (enables method chaining).
		 *	@warning	The memory areas may not overlap.
		 */
		template<typename Type> Stream & memcpy_3d(ImageAccessor<Type> dst_img, ImageAccessor<Type> src_img, size_t width, size_t height, size_t depth)
		{
			return this->memcpy_image(dst_img, src_img, width, height, depth);
		}

	public:

		/**
		 *	@brief		Copies data between 2D objects.
		 *	@param[in]	dst - Destination memory address.
		 *	@param[in]	dst_pitch - Pitch of destination memory.
		 *	@param[in]	src - Source memory address.
		 *	@param[in]	src_pitch - Pitch of source memory.
		 *	@param[in]	width - Width of matrix transfer (columns).
		 *	@param[in]	height - Height of matrix transfer (rows).
		 *	@param[in]	extent - Extent of matrix transfer.
		 *	@retval		Stream - Reference to this stream (enables method chaining).
		 *	@warning	The memory areas may not overlap. \p width must not exceed either \p dst_pitch or \p src_pitch.
		 */
		template<typename Type> Stream & memcpy_2d(Type * dst, size_t dst_pitch, const Type * src, size_t src_pitch, size_t width, size_t height)
		{
			if constexpr (!std::is_same_v<Type, void>)
				return this->memcpy_linear(dst, dst_pitch, 0, src, src_pitch, 0, width * sizeof(Type), height, 1);
			else
				return this->memcpy_linear(dst, dst_pitch, 0, src, src_pitch, 0, width, height, 1);
		}


		/**
		 *	@brief		Copies data between linear memory and image.
		 *	@param[in]	dst - Destination memory address.
		 *	@param[in]	dst_pitch - Pitch of destination memory.
		 *	@param[in]	src_img - Accessor to the source image.
		 *	@param[in]	width - Width of matrix transfer (columns).
		 *	@param[in]	height - Height of matrix transfer (rows).
		 *	@retval		Stream - Reference to this stream (enables method chaining).
		 */
		template<typename Type> Stream & memcpy_2d(Type * dst, size_t dst_pitch, ImageAccessor<Type> src_img, size_t width, size_t height)
		{
			return this->memcpy_linear_image(dst, dst_pitch, 0, src_img, width, height, 1);
		}


		/**
		 *	@brief		Copies data between image and linear memory.
		 *	@param[in]	dst_img - Accessor to the destination image.
		 *	@param[in]	src - Source memory address.
		 *	@param[in]	src_pitch - Pitch of source memory.
		 *	@param[in]	width - Width of matrix transfer (columns).
		 *	@param[in]	height - Height of matrix transfer (rows).
		 *	@retval		Stream - Reference to this stream (enables method chaining).
		 */
		template<typename Type> Stream & memcpy_2d(ImageAccessor<Type> dst_img, const Type * src, size_t src_pitch, size_t width, size_t height)
		{
			return this->memcpy_image_linear(dst_img, src, src_pitch, 0, width, height, 1);
		}


		/**
		 *	@brief		Copies data between images.
		 *	@param[in]	dst_img - Accessor to the destination image.
		 *	@param[in]	src_img - Accessor to the source image.
		 *	@param[in]	src_pitch - Offset of source image data.
		 *	@param[in]	width - Width of matrix transfer (columns in bytes).
		 *	@param[in]	height - Height of matrix transfer (rows).
		 *	@retval		Stream - Reference to this stream (enables method chaining).
		 *	@warning	The memory areas may not overlap.
		 */
		template<typename Type> Stream & memcpy_2d(ImageAccessor<Type> dst_img, ImageAccessor<Type> src_img, size_t width, size_t height)
		{
			return this->memcpy_image(dst_img, src_img, width, height, 1);
		}

	public:

		/**
		 *	@brief		Copies \p count bytes from the memory area pointed to by \p src to the memory area pointed to by \p dst.
		 *	@param[in]	dst - Destination memory address.
		 *	@param[in]	src - Source memory address.
		 *	@param[in]	count - If Type is void: bytes to copy. Otherwise, number of elements to copy.
		 * 	@retval		Stream - Reference to this stream (enables method chaining).
		 *	@note		Copying memory on different devices is also available.
		 */
		template<typename Type> Stream & memcpy(Type * dst, const Type * src, size_t count)
		{
			if constexpr (!std::is_same_v<Type, void>)
				return this->memcpy_linear(dst, 0, 0, src, 0, 0, count * sizeof(Type), 1, 1);
			else
				return this->memcpy_linear(dst, 0, 0, src, 0, 0, count, 1, 1);
		}


		/**
		 *	@brief		Copies data between linear memory and image.
		 *	@param[in]	dst - Destination memory address.
		 *	@param[in]	src_img - Accessor to the source image.
		 *	@param[in]	srcPosX - x offset of image data.
		 *	@param[in]	srcPosY - y offset of image data.
		 *	@param[in]	srcPosZ - z offset of image data.
		 *	@param[in]	count - number of elements to copy.
		 *	@retval		Stream - Reference to this stream (enables method chaining).
		 */
		template<typename Type> Stream & memcpy(Type * dst, ImageAccessor<Type> src_img, size_t count)
		{
			return this->memcpy_linear_image(dst, 0, 0, src_img, count, 1, 1);
		}


		/**
		 *	@brief		Copies data between image and linear memory.
		 *	@param[in]	dst_img - Accessor to the destination image.
		 *	@param[in]	dstPosX - x offset of destination image data.
		 *	@param[in]	dstPosY - y offset of destination image data.
		 *	@param[in]	dstPosZ - z offset of destination image data.
		 *	@param[in]	src - Source memory address.
		 *	@param[in]	count - number of elements to copy.
		 *	@retval		Stream - Reference to this stream (enables method chaining).
		 */
		template<typename Type> Stream & memcpy(ImageAccessor<Type> dst_img, const Type * src, size_t count)
		{
			return this->memcpy_image_linear(dst_img, src, 0, 0, count, 1, 1);
		}


		/**
		 *	@brief		Copies data between images.
		 *	@param[in]	dst_img - Accessor to the destination image.
		 *	@param[in]	src_img - Accessor to the source image.
		 *	@param[in]	count - number of elements to copy.
		 *	@retval		Stream - Reference to this stream (enables method chaining).
		 *	@warning	The memory areas may not overlap.
		 */
		template<typename Type> Stream & memcpy(ImageAccessor<Type> dst_img, ImageAccessor<Type> src_img, size_t count)
		{
			return this->memcpy_image(dst_img, src_img, count, 1, 1);
		}

	public:

		/**
		 *	@brief		Copies data to the given symbol on the device.
		 *	@param[in]	symbol - Device symbol address.
		 *	@param[in]	offset - Offset from start of symbol (in bytes when Type = void).
		 *	@param[in]	src - Source memory address.
		 *	@param[in]	count - Element count to copy (in bytes when Type = void).
		 *	@retval		Stream - Reference to this stream (enables method chaining).
		 */
		template<typename Type> Stream & memcpy_to_symbol(Type * symbol, size_t offset, const Type * src, size_t count)
		{
			if constexpr (!std::is_same_v<Type, void>)
				return this->memcpy_to_symbol_impl(symbol, offset * sizeof(Type), src, count * sizeof(Type));
			else
				return this->memcpy_to_symbol_impl(symbol, offset, src, count);
		}
		template<typename Type> Stream & memcpy_to_symbol(Type * symbol, const Type * src, size_t count)
		{
			if constexpr (!std::is_same_v<Type, void>)
				return this->memcpy_to_symbol_impl(symbol, 0, src, count * sizeof(Type));
			else
				return this->memcpy_to_symbol_impl(symbol, 0, src, count);
		}


		/**
		 *	@brief		Copies data from the given symbol on the device
		 *	@param[in]	src - Destination memory address.
		 *	@param[in]	symbol - Device symbol address.
		 *	@param[in]	offset - Offset from start of symbol (in bytes when Type = void).
		 *	@param[in]	count - Element count to copy (in bytes when Type = void).
		 *	@retval		Stream - Reference to this stream (enables method chaining).
		 */
		template<typename Type> Stream & memcpy_from_symbol(Type * dst, const Type * symbol, size_t offset, size_t count)
		{
			if constexpr (!std::is_same_v<Type, void>)
				return this->memcpy_from_symbol_impl(dst, symbol, offset * sizeof(Type), count * sizeof(Type));
			else
				return this->memcpy_from_symbol_impl(dst, symbol, offset, count);
		}
		template<typename Type> Stream & memcpy_from_symbol(Type * dst, const Type * symbol, size_t count)
		{
			if constexpr (!std::is_same_v<Type, void>)
				return this->memcpy_from_symbol_impl(dst, symbol, 0, count * sizeof(Type));
			else
				return this->memcpy_from_symbol_impl(dst, symbol, 0, count);
		}

	public:

		/**
		 *	@brief		Initialize or set device memory to a value.
		 *	@param[in]	values - Pointer to the device memory.
		 *	@param[in]	value - Value to set for.
		 *	@param[in]	count - Count of values to set.
		 *	@param[in]	block_size - CUDA thread block size (default = 256, which is near-optimal for most modern GPUs). 
		 *	@retval		Stream - Reference to this stream (enables method chaining).
		 *	@warning	Only available in *.cu files (CUDA compilation required).
		 */
		template<typename Type> Stream & fill(Type * values, Type value, size_t count, int block_size = 256);


		/**
		 *	@brief		Fills device memory with a constant byte value.
		 *	@param[in]	address - Pointer to device memory.
		 *	@param[in]	value - Value to fill with (only the least significant byte is used).
		 *	@param[in]	bytes - Size in bytes to set.
		 *	@retval		Stream - Reference to this stream (enables method chaining).
		 */
		NS_API Stream & memset(void * address, int value, size_t bytes);
		
	private:

		/**
		 *	@brief		Copies data between linear memory.
		 *	@param[in]	dst - Destination memory address.
		 *	@param[in]	dst_pitch - Pitch of destination memory.
		 *	@param[in]	dst_height - Height of destination memory.
		 *	@param[in]	src - Source memory address.
		 *	@param[in]	src_pitch - Pitch of source memory.
		 *	@param[in]	src_height - Height of source memory.
		 *	@param[in]	width - Width of matrix transfer (columns in bytes).
		 *	@param[in]	height - Height of matrix transfer (rows).
		 *	@param[in]	depth - Depth of matrix transfer (layers).
		 *	@retval		Stream - Reference to this stream (enables method chaining).
		 *	@warning	The memory areas may not overlap. \p width must not exceed either \p dst_pitch or \p src_pitch.
		 */
		NS_API Stream & memcpy_linear(void * dst, size_t dst_pitch, size_t dst_height, const void * src, size_t src_pitch, size_t src_height, size_t width, size_t height, size_t depth);


		/**
		 *	@brief		Copies data between linear memory and image.
		 *	@param[in]	dst - Destination memory address.
		 *	@param[in]	dst_pitch - Pitch of destination memory.
		 *	@param[in]	dst_height - Height of destination memory.
		 *	@param[in]	src_img - Accessor to the source image.
		 *	@param[in]	width - Width of matrix transfer (columns).
		 *	@param[in]	height - Height of matrix transfer (rows).
		 *	@param[in]	depth - Depth of matrix transfer (layers).
		 *	@retval		Stream - Reference to this stream (enables method chaining).
		 */
		NS_API Stream & memcpy_linear_image(void * dst, size_t dst_pitch, size_t dst_height, ImageAccessor<void> src_img, size_t width, size_t height, size_t depth);


		/**
		 *	@brief		Copies data between image and linear memory.
		 *	@param[in]	dst_img - Accessor to the destination image.
		 *	@param[in]	src - Source memory address.
		 *	@param[in]	src_pitch - Pitch of source memory.
		 *	@param[in]	src_height - Height of source memory.
		 *	@param[in]	width - Width of matrix transfer (columns).
		 *	@param[in]	height - Height of matrix transfer (rows).
		 *	@param[in]	depth - Depth of matrix transfer (layers).
		 *	@retval		Stream - Reference to this stream (enables method chaining).
		 */
		NS_API Stream & memcpy_image_linear(ImageAccessor<void> dst_img, const void * src, size_t src_pitch, size_t src_height, size_t width, size_t height, size_t depth);


		/**
		 *	@brief		Copies data between images.
		 *	@param[in]	dst_img - Accessor to the destination image.
		 *	@param[in]	src_img - Accessor to the source image.
		 *	@param[in]	src_pitch - Offset of source image data.
		 *	@retval		Stream - Reference to this stream (enables method chaining).
		 *	@warning	The memory areas may not overlap.
		 */
		NS_API Stream & memcpy_image(ImageAccessor<void> dst_img, ImageAccessor<void> src_img, size_t width, size_t height, size_t depth);


		/**
		 *	@brief		Copies data from the given symbol on the device
		 *	@param[in]	src - Destination memory address.
		 *	@param[in]	symbol - Device symbol address.
		 *	@param[in]	offset - Offset from start of symbol in bytes.
		 *	@param[in]	count - Element count to copy in bytes.
		 *	@retval		Stream - Reference to this stream (enables method chaining).
		 */
		NS_API Stream & memcpy_from_symbol_impl(void * dst, const void * symbol, size_t offset, size_t count);


		/**
		 *	@brief		Copies data to the given symbol on the device.
		 *	@param[in]	symbol - Device symbol address.
		 *	@param[in]	offset - Offset from start of symbol in bytes.
		 *	@param[in]	src - Source memory address.
		 *	@param[in]	count - Element count to copy in bytes.
		 *	@retval		Stream - Reference to this stream (enables method chaining).
		 */
		NS_API Stream & memcpy_to_symbol_impl(void * symbol, size_t offset, const void * src, size_t count);


		/**
		 *	@brief		Launches a device function
		 *	@param[in]	func - Device function symbol.
		 *	@param[in]	grid_dim - Grid dimensions.
		 *	@param[in]	block_dim - Block dimensions.
		 *	@param[in]	shared_mem_bytes - Number of bytes for shared memory.
		 *	@param[in]	args - Pointers to kernel arguments.
		 *	@retval		Stream - Reference to this stream (enables method chaining).
		 *	@warning	Only available in *.cu files.
		 */
		Stream & launch_kernel_impl(const void * func, const dim3 & grid_dim, const dim3 & block_dim, size_t shared_mem, void ** args);


		/**
		 *	@brief		Acquires the device context for this stream.
		 */
		NS_API void acquire_device_context() const;

	private:

		Device * const				device_;

		cudaStream_t  				stream_;

		bool						force_sync_;

		int							priority_;
	};
}
