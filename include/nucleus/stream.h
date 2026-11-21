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
		NS_API bool isComplete() const;


		/**
		 *	@brief		Force synchronization of the CUDA stream for debugging purposes.
		 *	@param[in]	enable - If true, every subsequent launch and memcpy operation will automatically synchronize
		 *				(blocking until completion) until this option is disabled. If false, do not force synchronization.
		 *	@note		This interface is intended for debugging and troubleshooting during development only.
		 *	@warning	It should NOT be used in production code.
		 */
		NS_API void setForceSync(bool enable);


		/**
		 *	@brief		Return pointer to the device associated with.
		 */
		Device * device() { return m_device; }


		/**
		 *	@brief		Return stream priority.
		 */
		int priority() const { return m_priority; }


		/**
		 *	@brief		Return CUDA stream type of this object.
		 *	@warning	Only for CUDA-based project.
		 */
		cudaStream_t handle() noexcept { return m_hStream; }

	public:

		/**
		 *	@brief		Record an event.
		 *	@param[in]	event - Valid event to record.
		 *	@note		Call such as `Event::is_done()` or `Stream::waitFor()` will then examine or wait for completion of the work that was captured.
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
		NS_API Stream & waitFor(EventRef event);

	public:

		/**
		 *	@brief		Launches an executable graph in a stream
		 *	@param[in]	hGraphExec - Executable graph to launch
		 *	@retval		Stream - Reference to this stream (enables method chaining).
		 *	@note		Each launch is ordered behind both any previous work in this stream and any previous launches of graphExec.
		 *	@example	stream.launchGraph(hGraph)
		 *	@example	stream.launchGraph(hGraph).sync()
		 *	@example	stream.launchGraph(hGraph).launchGraph(hGraph)
		 */
		NS_API Stream & launchGraph(cudaGraphExec_t hGraphExec);


		/**
		 *	@brief		Enqueue a host function call in a stream.
		 *	@param[in]	func - The function to call once preceding stream operations are complete.
		 *	@param[in]	userData - User-specified data to be passed to the function.
		 *	@retval		Stream - Reference to this stream (enables method chaining).
		 *	@note		Host function will be called from the thread named nvcuda64.dll.
		 *	@note		The function will be called after currently enqueued work and will block work added after it.
		 *	@warning	The host function must not perform any synchronization that may depend on outstanding CUDA work not mandated to run earlier.
		 *	@warning	The host function must not make any CUDA API calls.
		 */
		template<typename Type> Stream & launchHostFunc(HostFunc<Type> func, Type * userData)
		{
			return this->launchHostFunc(reinterpret_cast<HostFunc<void>>(func), userData);
		}
		NS_API Stream & launchHostFunc(HostFunc<void> func, void * userData);


		/**
		 *	@brief		Prepares to launch a CUDA kernel with specified parameters and dependencies.
		 *	@param[in]	func - Device function symbol.
		 *	@param[in]	gridDim - Grid dimensions.
		 *	@param[in]	blockDim - Block dimensions.
		 *	@param[in]	sharedMem - Number of bytes for shared memory.
		 *	@example	stream.launch(KernelAdd, gridDim, blockDim, sharedMem)(A, B, C, count);
		 *	@note		The returned lambda is a temporary object that should be used immediately to configure and launch the kernel.
		 *				It encapsulates all necessary information for the kernel launch, including the kernel function, its arguments.
		 *	@warning	Only available in *.cu files (implemented in launch_utils.cuh).
		 */
		template<typename... Args> NS_NODISCARD auto launch(KernelFunc<Args...> func, const dim3 & gridDim, const dim3 & blockDim, size_t sharedMem = 0);
		
	public:

		/**
		 *	@brief		Copies data between 3D objects.
		 *	@param[in]	dst - Destination memory address.
		 *	@param[in]	dstPitch - Pitch of destination memory.
		 *	@param[in]	dstHeight - Height of destination memory.
		 *	@param[in]	src - Source memory address.
		 *	@param[in]	srcPitch - Pitch of source memory.
		 *	@param[in]	srcHeight - Height of source memory.
		 *	@param[in]	width - Width of matrix transfer (columns).
		 *	@param[in]	height - Height of matrix transfer (rows).
		 *	@param[in]	depth - Depth of matrix transfer (layers).
		 *	@param[in]	extext - Extent of matrix transfer.
		 *	@retval		Stream - Reference to this stream (enables method chaining).
		 *	@warning	The memory areas may not overlap. \p width must not exceed either \p dstPitch or \p srcPitch.
		 */
		template<typename Type> Stream & memcpy3D(Type * dst, size_t dstPitch, size_t dstHeight, const Type * src, size_t srcPitch, size_t srcHeight, size_t width, size_t height, size_t depth)
		{
			if constexpr (!std::is_same_v<Type, void>)
				return this->memcpyLinear(dst, dstPitch, dstHeight, src, srcPitch, srcHeight, width * sizeof(Type), height, depth);
			else
				return this->memcpyLinear(dst, dstPitch, dstHeight, src, srcPitch, srcHeight, width, height, depth);
		}


		/**
		 *	@brief		Copies data between linear memory and image.
		 *	@param[in]	dst - Destination memory address.
		 *	@param[in]	dstPitch - Pitch of destination memory.
		 *	@param[in]	dstHeight - Height of destination memory.
		 *	@param[in]	srcImg - Accessor to the source image.
		 *	@param[in]	width - Width of matrix transfer (columns).
		 *	@param[in]	height - Height of matrix transfer (rows).
		 *	@param[in]	depth - Depth of matrix transfer (layers).
		 *	@retval		Stream - Reference to this stream (enables method chaining).
		 */
		template<typename Type> Stream & memcpy3D(Type * dst, size_t dstPitch, size_t dstHeight, ImageAccessor<Type> srcImg, size_t width, size_t height, size_t depth)
		{
			return this->memcpyLinearImage(dst, dstPitch, dstHeight, srcImg, width, height, depth);
		}


		/**
		 *	@brief		Copies data between image and linear memory.
		 *	@param[in]	dstImg - Accessor to the destination image.
		 *	@param[in]	src - Source memory address.
		 *	@param[in]	srcPitch - Pitch of source memory.
		 *	@param[in]	srcHeight - Height of source memory.
		 *	@param[in]	width - Width of matrix transfer (columns).
		 *	@param[in]	height - Height of matrix transfer (rows).
		 *	@param[in]	depth - Depth of matrix transfer (layers).
		 *	@retval		Stream - Reference to this stream (enables method chaining).
		 */
		template<typename Type> Stream & memcpy3D(ImageAccessor<Type> dstImg, const Type * src, size_t srcPitch, size_t srcHeight, size_t width, size_t height, size_t depth)
		{
			return this->memcpyImageLinear(dstImg, src, srcPitch, srcHeight, width, height, depth);
		}


		/**
		 *	@brief		Copies data between images.
		 *	@param[in]	dstImg - Accessor to the destination image.
		 *	@param[in]	srcImg - Accessor to the source image.
		 *	@param[in]	width - Width of matrix transfer (columns in bytes).
		 *	@param[in]	height - Height of matrix transfer (rows).
		 *	@param[in]	depth - Depth of matrix transfer (layers).
		 *	@retval		Stream - Reference to this stream (enables method chaining).
		 *	@warning	The memory areas may not overlap.
		 */
		template<typename Type> Stream & memcpy3D(ImageAccessor<Type> dstImg, ImageAccessor<Type> srcImg, size_t width, size_t height, size_t depth)
		{
			return this->memcpyImage(dstImg, srcImg, width, height, depth);
		}

	public:

		/**
		 *	@brief		Copies data between 2D objects.
		 *	@param[in]	dst - Destination memory address.
		 *	@param[in]	dstPitch - Pitch of destination memory.
		 *	@param[in]	src - Source memory address.
		 *	@param[in]	srcPitch - Pitch of source memory.
		 *	@param[in]	width - Width of matrix transfer (columns).
		 *	@param[in]	height - Height of matrix transfer (rows).
		 *	@param[in]	extent - Extent of matrix transfer.
		 *	@retval		Stream - Reference to this stream (enables method chaining).
		 *	@warning	The memory areas may not overlap. \p width must not exceed either \p dstPitch or \p srcPitch.
		 */
		template<typename Type> Stream & memcpy2D(Type * dst, size_t dstPitch, const Type * src, size_t srcPitch, size_t width, size_t height)
		{
			if constexpr (!std::is_same_v<Type, void>)
				return this->memcpyLinear(dst, dstPitch, 0, src, srcPitch, 0, width * sizeof(Type), height, 1);
			else
				return this->memcpyLinear(dst, dstPitch, 0, src, srcPitch, 0, width, height, 1);
		}


		/**
		 *	@brief		Copies data between linear memory and image.
		 *	@param[in]	dst - Destination memory address.
		 *	@param[in]	dstPitch - Pitch of destination memory.
		 *	@param[in]	srcImg - Accessor to the source image.
		 *	@param[in]	width - Width of matrix transfer (columns).
		 *	@param[in]	height - Height of matrix transfer (rows).
		 *	@retval		Stream - Reference to this stream (enables method chaining).
		 */
		template<typename Type> Stream & memcpy2D(Type * dst, size_t dstPitch, ImageAccessor<Type> srcImg, size_t width, size_t height)
		{
			return this->memcpyLinearImage(dst, dstPitch, 0, srcImg, width, height, 1);
		}


		/**
		 *	@brief		Copies data between image and linear memory.
		 *	@param[in]	dstImg - Accessor to the destination image.
		 *	@param[in]	src - Source memory address.
		 *	@param[in]	srcPitch - Pitch of source memory.
		 *	@param[in]	width - Width of matrix transfer (columns).
		 *	@param[in]	height - Height of matrix transfer (rows).
		 *	@retval		Stream - Reference to this stream (enables method chaining).
		 */
		template<typename Type> Stream & memcpy2D(ImageAccessor<Type> dstImg, const Type * src, size_t srcPitch, size_t width, size_t height)
		{
			return this->memcpyImageLinear(dstImg, src, srcPitch, 0, width, height, 1);
		}


		/**
		 *	@brief		Copies data between images.
		 *	@param[in]	dstImg - Accessor to the destination image.
		 *	@param[in]	srcImg - Accessor to the source image.
		 *	@param[in]	srcPitch - Offset of source image data.
		 *	@param[in]	width - Width of matrix transfer (columns in bytes).
		 *	@param[in]	height - Height of matrix transfer (rows).
		 *	@retval		Stream - Reference to this stream (enables method chaining).
		 *	@warning	The memory areas may not overlap.
		 */
		template<typename Type> Stream & memcpy2D(ImageAccessor<Type> dstImg, ImageAccessor<Type> srcImg, size_t width, size_t height)
		{
			return this->memcpyImage(dstImg, srcImg, width, height, 1);
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
				return this->memcpyLinear(dst, 0, 0, src, 0, 0, count * sizeof(Type), 1, 1);
			else
				return this->memcpyLinear(dst, 0, 0, src, 0, 0, count, 1, 1);
		}


		/**
		 *	@brief		Copies data between linear memory and image.
		 *	@param[in]	dst - Destination memory address.
		 *	@param[in]	srcImg - Accessor to the source image.
		 *	@param[in]	srcPosX - x offset of image data.
		 *	@param[in]	srcPosY - y offset of image data.
		 *	@param[in]	srcPosZ - z offset of image data.
		 *	@param[in]	count - number of elements to copy.
		 *	@retval		Stream - Reference to this stream (enables method chaining).
		 */
		template<typename Type> Stream & memcpy(Type * dst, ImageAccessor<Type> srcImg, size_t count)
		{
			return this->memcpyLinearImage(dst, 0, 0, srcImg, count, 1, 1);
		}


		/**
		 *	@brief		Copies data between image and linear memory.
		 *	@param[in]	dstImg - Accessor to the destination image.
		 *	@param[in]	dstPosX - x offset of destination image data.
		 *	@param[in]	dstPosY - y offset of destination image data.
		 *	@param[in]	dstPosZ - z offset of destination image data.
		 *	@param[in]	src - Source memory address.
		 *	@param[in]	count - number of elements to copy.
		 *	@retval		Stream - Reference to this stream (enables method chaining).
		 */
		template<typename Type> Stream & memcpy(ImageAccessor<Type> dstImg, const Type * src, size_t count)
		{
			return this->memcpyImageLinear(dstImg, src, 0, 0, count, 1, 1);
		}


		/**
		 *	@brief		Copies data between images.
		 *	@param[in]	dstImg - Accessor to the destination image.
		 *	@param[in]	srcImg - Accessor to the source image.
		 *	@param[in]	count - number of elements to copy.
		 *	@retval		Stream - Reference to this stream (enables method chaining).
		 *	@warning	The memory areas may not overlap.
		 */
		template<typename Type> Stream & memcpy(ImageAccessor<Type> dstImg, ImageAccessor<Type> srcImg, size_t count)
		{
			return this->memcpyImage(dstImg, srcImg, count, 1, 1);
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
		template<typename Type> Stream & memcpyToSymbol(Type * symbol, size_t offset, const Type * src, size_t count)
		{
			if constexpr (!std::is_same_v<Type, void>)
				return this->memcpyToSymbolImpl(symbol, offset * sizeof(Type), src, count * sizeof(Type));
			else
				return this->memcpyToSymbolImpl(symbol, offset, src, count);
		}
		template<typename Type> Stream & memcpyToSymbol(Type * symbol, const Type * src, size_t count)
		{
			if constexpr (!std::is_same_v<Type, void>)
				return this->memcpyToSymbolImpl(symbol, 0, src, count * sizeof(Type));
			else
				return this->memcpyToSymbolImpl(symbol, 0, src, count);
		}


		/**
		 *	@brief		Copies data from the given symbol on the device
		 *	@param[in]	src - Destination memory address.
		 *	@param[in]	symbol - Device symbol address.
		 *	@param[in]	offset - Offset from start of symbol (in bytes when Type = void).
		 *	@param[in]	count - Element count to copy (in bytes when Type = void).
		 *	@retval		Stream - Reference to this stream (enables method chaining).
		 */
		template<typename Type> Stream & memcpyFromSymbol(Type * dst, const Type * symbol, size_t offset, size_t count)
		{
			if constexpr (!std::is_same_v<Type, void>)
				return this->memcpyFromSymbolImpl(dst, symbol, offset * sizeof(Type), count * sizeof(Type));
			else
				return this->memcpyFromSymbolImpl(dst, symbol, offset, count);
		}
		template<typename Type> Stream & memcpyFromSymbol(Type * dst, const Type * symbol, size_t count)
		{
			if constexpr (!std::is_same_v<Type, void>)
				return this->memcpyFromSymbolImpl(dst, symbol, 0, count * sizeof(Type));
			else
				return this->memcpyFromSymbolImpl(dst, symbol, 0, count);
		}

	public:

		/**
		 *	@brief		Initialize or set device memory to a value.
		 *	@param[in]	pValues - Pointer to the device memory.
		 *	@param[in]	value - Value to set for.
		 *	@param[in]	count - Count of values to set.
		 *	@param[in]	blockSize - CUDA thread block size (default = 256, which is near-optimal for most modern GPUs). 
		 *	@retval		Stream - Reference to this stream (enables method chaining).
		 *	@warning	Only available in *.cu files (CUDA compilation required).
		 */
		template<typename Type> Stream & fill(Type * pValues, Type value, size_t count, int blockSize = 256);


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
		 *	@param[in]	dstPitch - Pitch of destination memory.
		 *	@param[in]	dstHeight - Height of destination memory.
		 *	@param[in]	src - Source memory address.
		 *	@param[in]	srcPitch - Pitch of source memory.
		 *	@param[in]	srcHeight - Height of source memory.
		 *	@param[in]	width - Width of matrix transfer (columns in bytes).
		 *	@param[in]	height - Height of matrix transfer (rows).
		 *	@param[in]	depth - Depth of matrix transfer (layers).
		 *	@retval		Stream - Reference to this stream (enables method chaining).
		 *	@warning	The memory areas may not overlap. \p width must not exceed either \p dstPitch or \p srcPitch.
		 */
		NS_API Stream & memcpyLinear(void * dst, size_t dstPitch, size_t dstHeight, const void * src, size_t srcPitch, size_t srcHeight, size_t width, size_t height, size_t depth);


		/**
		 *	@brief		Copies data between linear memory and image.
		 *	@param[in]	dst - Destination memory address.
		 *	@param[in]	dstPitch - Pitch of destination memory.
		 *	@param[in]	dstHeight - Height of destination memory.
		 *	@param[in]	srcImg - Accessor to the source image.
		 *	@param[in]	width - Width of matrix transfer (columns).
		 *	@param[in]	height - Height of matrix transfer (rows).
		 *	@param[in]	depth - Depth of matrix transfer (layers).
		 *	@retval		Stream - Reference to this stream (enables method chaining).
		 */
		NS_API Stream & memcpyLinearImage(void * dst, size_t dstPitch, size_t dstHeight, ImageAccessor<void> srcImg, size_t width, size_t height, size_t depth);


		/**
		 *	@brief		Copies data between image and linear memory.
		 *	@param[in]	dstImg - Accessor to the destination image.
		 *	@param[in]	src - Source memory address.
		 *	@param[in]	srcPitch - Pitch of source memory.
		 *	@param[in]	srcHeight - Height of source memory.
		 *	@param[in]	width - Width of matrix transfer (columns).
		 *	@param[in]	height - Height of matrix transfer (rows).
		 *	@param[in]	depth - Depth of matrix transfer (layers).
		 *	@retval		Stream - Reference to this stream (enables method chaining).
		 */
		NS_API Stream & memcpyImageLinear(ImageAccessor<void> dstImg, const void * src, size_t srcPitch, size_t srcHeight, size_t width, size_t height, size_t depth);


		/**
		 *	@brief		Copies data between images.
		 *	@param[in]	dstImg - Accessor to the destination image.
		 *	@param[in]	srcImg - Accessor to the source image.
		 *	@param[in]	srcPitch - Offset of source image data.
		 *	@retval		Stream - Reference to this stream (enables method chaining).
		 *	@warning	The memory areas may not overlap.
		 */
		NS_API Stream & memcpyImage(ImageAccessor<void> dstImg, ImageAccessor<void> srcImg, size_t width, size_t height, size_t depth);


		/**
		 *	@brief		Copies data from the given symbol on the device
		 *	@param[in]	src - Destination memory address.
		 *	@param[in]	symbol - Device symbol address.
		 *	@param[in]	offset - Offset from start of symbol in bytes.
		 *	@param[in]	count - Element count to copy in bytes.
		 *	@retval		Stream - Reference to this stream (enables method chaining).
		 */
		NS_API Stream & memcpyFromSymbolImpl(void * dst, const void * symbol, size_t offset, size_t count);


		/**
		 *	@brief		Copies data to the given symbol on the device.
		 *	@param[in]	symbol - Device symbol address.
		 *	@param[in]	offset - Offset from start of symbol in bytes.
		 *	@param[in]	src - Source memory address.
		 *	@param[in]	count - Element count to copy in bytes.
		 *	@retval		Stream - Reference to this stream (enables method chaining).
		 */
		NS_API Stream & memcpyToSymbolImpl(void * symbol, size_t offset, const void * src, size_t count);


		/**
		 *	@brief		Launches a device function
		 *	@param[in]	func - Device function symbol.
		 *	@param[in]	gridDim - Grid dimensions.
		 *	@param[in]	blockDim - Block dimensions.
		 *	@param[in]	sharedMemBytes - Number of bytes for shared memory.
		 *	@param[in]	args - Pointers to kernel arguments.
		 *	@retval		Stream - Reference to this stream (enables method chaining).
		 *	@warning	Only available in *.cu files.
		 */
		Stream & launchKernelImpl(const void * func, const dim3 & gridDim, const dim3 & blockDim, size_t sharedMem, void ** args);


		/**
		 *	@brief		Acquires the device context for this stream.
		 */
		NS_API void acquireDeviceContext() const;

	private:

		Device * const				m_device;

		cudaStream_t  				m_hStream;

		bool						m_forceSync;

		int							m_priority;
	};
}
