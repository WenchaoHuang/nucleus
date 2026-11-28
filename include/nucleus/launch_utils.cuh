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

#include "graph.h"
#include "stream.h"
#include "device.h"
#include "logger.h"
#include "utility.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstring>

#ifndef __CUDACC__
	// This is a no-op for actual compilation,
	// only for suppressing MSVC IntelliSense errors.
	#define __CUDACC__
	#error This file should be inclued in *.cu files only!
#endif

namespace NS_NAMESPACE
{
	/*****************************************************************************
	********************************    utils    *********************************
	*****************************************************************************/

	/**
	 *	@brief		Returns the globally unique thread ID in a 1D grid-block layout.
	 */
	__device__ __forceinline__ unsigned int tid() { return blockDim.x * blockIdx.x + threadIdx.x; }


	/**
	 *	@brief		Macro that automatically bounds-checks thread ID.
	 *	@details	1. Calculating the global thread ID (i)
	 *				2. Early-exiting if thread ID is out of bounds
	 * 
	 *	@note		- UPPER_CASE (NS_BOUNDS_CHECK): For codebases enforcing strict conventions
	 *				- lower_case (ns_bounds_check): For ergonomic coding and rapid prototyping
	 *				(Can you guess which one I prefer?)
	 */
	#define NS_BOUNDS_CHECK(i, num_threads)		const auto i = ns::tid();	if (i >= (num_threads)) return;
	#define ns_bounds_check(i, num_threads)		NS_BOUNDS_CHECK(i, num_threads)
	#define CUDA_for(i, num_threads)			NS_BOUNDS_CHECK(i, num_threads)

	/*****************************************************************************
	********************************    Device    ********************************
	*****************************************************************************/

	/**
	 *	@brief		Returns occupancy for a device function.
	 *	@param[in]	func - Kernel function for which occupancy is calculated
	 *	@param[in]	block_size - Block size the kernel is intended to be launched with
	 *	@param[in]	dynamic_s_mem_size - Per-block dynamic shared memory usage intended, in bytes.
	 */
	inline int Device::occupancy_max_active_blocks_per_multiprocessor(const void * func, int block_size, size_t dynamics_mem_size)
	{
		int num_blocks = 0;

		cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, func, block_size, dynamics_mem_size);

		if (err != cudaSuccess)
		{
			NS_ERROR_LOG("%s.", cudaGetErrorString(err));

			cudaGetLastError();
		}

		return num_blocks;
	}

	/*****************************************************************************
	********************************    kernel    ********************************
	*****************************************************************************/

	namespace kernel
	{
		template<typename Type> __global__ void memset(Type * values, Type value, size_t count)
		{
			CUDA_for(i, count);					values[i] = value;
		}
	}

	/*****************************************************************************
	********************************    Stream    ********************************
	*****************************************************************************/

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
	inline Stream & Stream::launch_kernel_impl(const void * func, const dim3 & grid_dim, const dim3 & block_dim, size_t shared_mem, void ** args)
	{
		if (grid_dim.x * grid_dim.y * grid_dim.z * block_dim.x * block_dim.y * block_dim.z != 0)
		{
			this->acquire_device_context();

			cudaError_t err = cudaLaunchKernel(func, grid_dim, block_dim, args, shared_mem, stream_);

			if (err != cudaSuccess)
			{
				NS_ERROR_LOG("%s.", cudaGetErrorString(err));

				cudaGetLastError();

				throw err;
			}
			else if (force_sync_)
			{
				this->sync();
			}
		}

		return *this;
	}


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
	template<typename... Args> auto Stream::launch(KernelFunc<Args...> func, const dim3 & grid_dim, const dim3 & block_dim, size_t shared_mem)
	{
		return [=, this](Args... args) -> Stream& { void * params[] = { &args... }; return this->launch_kernel_impl(reinterpret_cast<const void*>(func), grid_dim, block_dim, shared_mem, params); };
	}

	//	Specialization for parameterless kernels
	template<> inline auto Stream::launch(KernelFunc<> func, const dim3 & grid_dim, const dim3 & block_dim, size_t shared_mem)
	{
		return [=, this]() -> Stream& { return this->launch_kernel_impl(reinterpret_cast<const void*>(func), grid_dim, block_dim, shared_mem, nullptr); };
	}


	/**
	 *	@brief		Initialize or set device memory to a value.
	 *	@param[in]	values - Pointer to the device memory.
	 *	@param[in]	value - Value to set for.
	 *	@param[in]	count - Count of values to set.
	 *	@param[in]	block_size - CUDA thread block size (default = 256, which is near-optimal for most modern GPUs).
	 *	@retval		Stream - Reference to this stream (enables method chaining).
	 *	@warning	Only available in *.cu files (CUDA compilation required).
	 */
	template<typename Type> Stream & Stream::fill(Type * values, Type value, size_t count, int block_size)
	{
		return this->launch(kernel::memset<Type>, ns::ceil_div(count, block_size), block_size)(values, value, count);
	}

	/*****************************************************************************
	********************************    Graph    *********************************
	*****************************************************************************/

	template<typename Type> ExecDep Graph::memset(Type * values, Type value, size_t count, Span<const ExecDep> dependencies)
	{
		constexpr unsigned int [[maybe_unused]] optimal_block_size_RTX_3080_Ti = 512;
		constexpr unsigned int [[maybe_unused]] optimal_block_size_RTX_2070_SUPER = 256;

		constexpr int block_size = optimal_block_size_RTX_3080_Ti;

		return this->launch(kernel::memset<Type>, dependencies, ns::ceil_div(count, block_size), block_size)(values, value, count);
	}

	template<typename... Args> ExecDep Graph::launch_kernel(KernelFunc<Args...> func, Span<const ExecDep> dependencies, const dim3 & grid_dim, const dim3 & block_dim, unsigned int shared_mem, Args... args)
	{
		if (immediate_launch_stream_ != nullptr)	//	in immediate launch mode
		{
			immediate_launch_stream_->launch(func, grid_dim, block_dim, shared_mem)(args...);

			return ExecDep{ id_, -1 };
		}
		else if (grid_dim.x * grid_dim.y * grid_dim.z * block_dim.x * block_dim.y * block_dim.z == 0)
		{
			return this->barrier(dependencies);
		}
		else /////////////////////////////////////////////////////////////////////////////////////////////////////////////
		{
			const uint64_t dep_hash = this->cache_dependencies(dependencies);

			constexpr size_t param_bytes = sizeof(grid_dim) + sizeof(block_dim) + sizeof(shared_mem) + (sizeof(args) + ...);

			char param_cache[param_bytes];												size_t param_offset = 0;
			((std::memcpy(param_cache + param_offset, &grid_dim, sizeof(grid_dim)), param_offset += sizeof(grid_dim)));
			((std::memcpy(param_cache + param_offset, &block_dim, sizeof(block_dim)), param_offset += sizeof(block_dim)));
			((std::memcpy(param_cache + param_offset, &shared_mem, sizeof(shared_mem)), param_offset += sizeof(shared_mem)));
			((std::memcpy(param_cache + param_offset, &args, sizeof(args)), param_offset += sizeof(args)), ...);

			if (indicator_ < nodes_.size())	//	in validating state
			{
				if ((nodes_[indicator_].func != reinterpret_cast<void*>(func)) || (nodes_[indicator_].dep_hash != dep_hash))	//	dependencies changes
				{
					nodes_.resize(indicator_);
				}
				else if (std::memcmp(param_binaries_.data() + param_offset_, param_cache, param_bytes) != 0)	//	parameters changes
				{
					void * params[] = { ((void*)&args)... };

					cudaKernelNodeParams			launch_params = {};
					launch_params.func				= reinterpret_cast<void*>(func);
					launch_params.extra				= nullptr;
					launch_params.kernelParams		= params;
					launch_params.sharedMemBytes		= shared_mem;
					launch_params.blockDim			= block_dim;
					launch_params.gridDim			= grid_dim;

					cudaError_t err = cudaGraphKernelNodeSetParams(nodes_[indicator_].graph_node, &launch_params);

					NS_ERROR_LOG_IF(err != cudaSuccess, "%s.", cudaGetErrorString(cudaGetLastError()));

					std::memcpy(param_binaries_.data() + param_offset_, param_cache, param_bytes);

					is_param_chg_ = true;
				}
			}

			if (indicator_ >= nodes_.size())	//	topology changed
			{
				auto create_func = [=](cudaGraph_t graph, const cudaGraphNode_t * dependencies, size_t num_dependencies) -> cudaGraphNode_t
				{
					void * params[] = { ((void*)&args)... };	//	where magic happen, all parameters will captured in the lambda!

					cudaGraphNode_t					graph_node = nullptr;
					cudaKernelNodeParams			launch_params = {};
					launch_params.func				= reinterpret_cast<void*>(func);
					launch_params.extra				= nullptr;
					launch_params.kernelParams		= params;
					launch_params.sharedMemBytes		= shared_mem;
					launch_params.blockDim			= block_dim;
					launch_params.gridDim			= grid_dim;

					cudaError_t err = cudaGraphAddKernelNode(&graph_node, graph, dependencies, num_dependencies, &launch_params);

					NS_ERROR_LOG_IF(err != cudaSuccess, "%s.", cudaGetErrorString(cudaGetLastError()));

					return graph_node;
				};

				param_binaries_.resize(param_binaries_.size() + param_bytes);

				std::memcpy(param_binaries_.data() + param_offset_, param_cache, param_bytes);

				nodes_.emplace_back(reinterpret_cast<void*>(func), dep_hash, param_bytes, dep_indices_cache_, create_func);

				is_topo_chg_ = true;
			}

			param_offset_ += param_bytes;

			return ExecDep{ id_, indicator_++ };
		}
	}

	template<> inline ExecDep Graph::launch_kernel<>(KernelFunc<> func, Span<const ExecDep> dependencies, const dim3 & grid_dim, const dim3 & block_dim, unsigned int shared_mem)
	{
		if (immediate_launch_stream_ != nullptr)	//	in immediate launch mode
		{
			immediate_launch_stream_->launch(func, grid_dim, block_dim, shared_mem)();

			return ExecDep{ id_, -1 };
		}
		else if (grid_dim.x * grid_dim.y * grid_dim.z * block_dim.x * block_dim.y * block_dim.z == 0)
		{
			return this->barrier(dependencies);
		}
		else /////////////////////////////////////////////////////////////////////////////////////////////////////////////
		{
			const uint64_t dep_hash = this->cache_dependencies(dependencies);

			constexpr size_t param_bytes = sizeof(grid_dim) + sizeof(block_dim) + sizeof(shared_mem);

			char param_cache[param_bytes];												size_t param_offset = 0;
			((std::memcpy(param_cache + param_offset, &grid_dim, sizeof(grid_dim)), param_offset += sizeof(grid_dim)));
			((std::memcpy(param_cache + param_offset, &block_dim, sizeof(block_dim)), param_offset += sizeof(block_dim)));
			((std::memcpy(param_cache + param_offset, &shared_mem, sizeof(shared_mem)), param_offset += sizeof(shared_mem)));

			if (indicator_ < nodes_.size())	//	in validating state
			{
				if ((nodes_[indicator_].func != reinterpret_cast<void*>(func)) || (nodes_[indicator_].dep_hash != dep_hash))	//	dependencies changes
				{
					nodes_.resize(indicator_);
				}
				else if (std::memcmp(param_binaries_.data() + param_offset_, param_cache, param_bytes) != 0)	//	parameters changes
				{
					cudaKernelNodeParams			launch_params = {};
					launch_params.func				= reinterpret_cast<void*>(func);
					launch_params.extra				= nullptr;
					launch_params.kernelParams		= nullptr;
					launch_params.sharedMemBytes		= shared_mem;
					launch_params.blockDim			= block_dim;
					launch_params.gridDim			= grid_dim;

					cudaError_t err = cudaGraphKernelNodeSetParams(nodes_[indicator_].graph_node, &launch_params);

					NS_ERROR_LOG_IF(err != cudaSuccess, "%s.", cudaGetErrorString(cudaGetLastError()));

					std::memcpy(param_binaries_.data() + param_offset_, param_cache, param_bytes);

					is_param_chg_ = true;
				}
			}

			if (indicator_ >= nodes_.size())	//	topology changed
			{
				auto create_func = [=](cudaGraph_t graph, const cudaGraphNode_t * dependencies, size_t num_dependencies) -> cudaGraphNode_t
				{
					cudaGraphNode_t					graph_node = nullptr;
					cudaKernelNodeParams			launch_params = {};
					launch_params.func				= reinterpret_cast<void*>(func);
					launch_params.extra				= nullptr;
					launch_params.kernelParams		= nullptr;
					launch_params.sharedMemBytes		= shared_mem;
					launch_params.blockDim			= block_dim;
					launch_params.gridDim			= grid_dim;

					cudaError_t err = cudaGraphAddKernelNode(&graph_node, graph, dependencies, num_dependencies, &launch_params);

					NS_ERROR_LOG_IF(err != cudaSuccess, "%s.", cudaGetErrorString(cudaGetLastError()));

					return graph_node;
				};

				param_binaries_.resize(param_binaries_.size() + param_bytes);

				std::memcpy(param_binaries_.data() + param_offset_, param_cache, param_bytes);

				nodes_.emplace_back(reinterpret_cast<void*>(func), dep_hash, param_bytes, dep_indices_cache_, create_func);

				is_topo_chg_ = true;
			}

			param_offset_ += param_bytes;

			return ExecDep{ id_, indicator_++ };
		}
	}

	template<typename... Args> NS_NODISCARD auto Graph::launch(KernelFunc<Args...> func, Span<const ExecDep> dependencies, const dim3 & grid_dim, const dim3 & block_dim, unsigned int shared_mem)
	{
		return [=, this](Args... args) -> ExecDep { return this->launch_kernel(func, dependencies, grid_dim, block_dim, shared_mem, args...); };
	}

	template<typename... Args> NS_NODISCARD auto Graph::launch(KernelFunc<Args...> func, const dim3 & grid_dim, const dim3 & block_dim, unsigned int shared_mem)
	{
		return [=, this](Args... args) -> ExecDep { return this->launch_kernel(func, nullptr, grid_dim, block_dim, shared_mem, args...); };
	}
}
