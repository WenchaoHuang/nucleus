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
#include "span.h"
#include <unordered_set>
#include <functional>

namespace NS_NAMESPACE
{
	struct ExecDep { uint64_t ID = 0; int Index = -1; };

	/*****************************************************************************
	********************************    Graph    *********************************
	*****************************************************************************/

	/**
	 *	@brief		Manages graph-based execution of CUDA kernels, enabling optimized dependency handling and execution ordering.
	 *	@note		The Graph class is designed to facilitate the efficient execution of CUDA kernels by constructing
	 *				and managing a graph of execution nodes. It provides mechanisms for launching kernels, synchronizing
	 *				operations, and managing memory transfers within the CUDA graph framework.
	 *	@warning	Only for CUDA-based project.
	 */
	class Graph
	{
		NS_NONCOPYABLE(Graph)

	public:

		//!	@brief		Create graph object.
		NS_API Graph();

		//!	@brief		Destroy graph object.
		NS_API ~Graph();

	public:

		/**
		 *	@brief		Restarts the graph for a new sequence of operations, optionally switching to immediate launch mode.
		 *	@param[in]	pImmediateLaunchStream - Optional stream to associate with the graph upon restart. If provided (non-null),
		 *              the graph switches to immediate launch mode, where operations are executed immediately on the
		 *              specified stream. If null, the graph maintains its current execution mode without switching
		 *              to immediate launch. Providing a non-null stream will cause subsequent calls to execute
		 *              to skip their execution sequence, assuming immediate execution has already been handled.
		 *	@note		This method resets the graph to a clean state, making it ready for new operations. It is useful
		 *              for efficiently reusing the graph object with different streams or modes without needing to
		 *              reconstruct or reinitialize the graph. The transition to immediate launch mode provides instant
		 *              operation execution on the provided stream, beneficial in scenarios where minimal latency is critical.
		 */
		NS_API void restart(Stream * pImmediateLaunchStream = nullptr);


		/**
		 *	@brief		Updates the graph's configuration if necessary and executes it using the provided stream.
		 *	@param[in]	pStream - The stream on which the graph operations will be executed. This stream manages
		 *              the execution context and ensures that the graph operations are performed in the correct sequence
		 *              and with proper synchronization.
		 *	@note		This method handles any changes in the graph's topology or parameter configurations before
		 *              executing. It ensures that all nodes and dependencies are properly aligned and optimized
		 *              for execution. If no changes are required, it directly proceeds to execute the graph. If the graph
		 *              is already in immediate launch mode due to a non-null stream provided in restart, this method will
		 *              skip its operations to avoid redundant execution.
		 *	@note		Ideal for scenarios where periodic updates to the graph's configuration are necessary before execution,
		 *              such as updating data inputs or operation parameters.
		 */
		NS_API void execute(Stream * pStream);


		/**
		 *	@brief		Creates a synchronization barrier that waits on the completion of specified dependencies.
		 *	@param[in]	dependencies - An array of dependencies that must complete before proceeding.
		 *	@return		ExecDep representing the dependency handle for the launched kernel.
		 */
		NS_API ExecDep barrier(Span<const ExecDep> dependencies = {});


		/**
		 *	@brief		Sets memory to a specified value across a range, with dependency tracking.
		 *	@param[in]	pValues - Pointer to the memory location to set.
		 *	@param[in]	value - Value to set at the memory location.
		 *	@param[in]	count - Number of elements to set.
		 *	@param[in]	dependencies - Dependencies that must be resolved before this operation can start.
		 *	@return		ExecDep representing the dependency handle for the launched kernel.
		 */
		template<typename Type> ExecDep memset(Type * pValues, Type value, size_t count, Span<const ExecDep> dependencies = {});


		/**
		 *	@brief		Copies memory from source to destination, with dependency tracking.
		 *	@param[in]	dst - Destination memory address.
		 *	@param[in]	src - Source memory address.
		 *	@param[in]	count - Number of elements to copy.
		 *	@param[in]	dependencies - Dependencies that must be resolved before this operation can start.
		 *	@return		ExecDep representing the dependency handle for the launched kernel.
		 */
		template<typename Type> ExecDep memcpy(Type * dst, const Type * src, size_t count, Span<const ExecDep> dependencies = {})
		{
			return this->memcpy_void(dst, src, sizeof(Type) * count, dependencies);
		}


		/**
		 *	@brief		Prepares to launch a CUDA kernel with specified parameters and dependencies.
		 *	@param[in]	func - Device function symbol.
		 *	@param[in]	dependencies - Dependencies that must be resolved before this operation can start.
		 *	@param[in]	gDim - Grid dimensions.
		 *	@param[in]	bDim - Block dimensions.
		 *	@param[in]	sharedMemBytes - Number of bytes for shared memory.
		 *	@param[in]	args - Kernel launch arguments (by value).
		 *	@note		The returned ConfigFunctor is a temporary object that should be used immediately to configure and launch the kernel.
		 *				It encapsulates all necessary information for the kernel launch, including the kernel function, its arguments,
		 *				and any specified dependencies.
		 *	@example	grpah.Launch(KernelAdd, { d1, d2 }, gridDim, blockDim, sharedMem)(pResult, pA, pB, num);
		 *	@warning	Only available in *.cu files.
		 */
	private:
		template<typename... Args> ExecDep launchKernel(KernelFunc<Args...> func, Span<const ExecDep> dependencies, const dim3 & gridDim, const dim3 & blockDim, unsigned int sharedMem, Args... args);
	public:
		template<typename... Args> NS_NODISCARD auto launch(KernelFunc<Args...> func, Span<const ExecDep> dependencies, const dim3 & gridDim, const dim3 & blockDim, unsigned int sharedMem = 0);
		template<typename... Args> NS_NODISCARD auto launch(KernelFunc<Args...> func, const dim3 & gridDim, const dim3 & blockDim, unsigned int sharedMem = 0);

	private:

		NS_API uint64_t cacheDependencies(Span<const ExecDep> dependencies);
		
		NS_API ExecDep memcpy_void(void * dst, const void * src, size_t bytes, Span<const ExecDep> dependencies);

	private:

		struct NodeInfo
		{
			using CreateFunc = std::function<cudaGraphNode_t(cudaGraph_t hGraph, const cudaGraphNode_t * pDependencies, size_t numDependencies)>;

			explicit NodeInfo(void * _func, uint64_t _depHash, uint64_t _paramBytes, const std::unordered_set<int> & _dep, const CreateFunc & _createFunc)
				: func(_func), depHash(_depHash), paramBytes(_paramBytes), hGraphNode(nullptr), depIndices(_dep), createFunc(_createFunc)
			{
			}

			NodeInfo() : func(nullptr), depHash(0), paramBytes(0), createFunc(nullptr), hGraphNode(nullptr) {}

			void * const						func;
			const uint64_t						depHash;
			const uint64_t						paramBytes;
			const CreateFunc					createFunc;
			const std::unordered_set<int>		depIndices;
			cudaGraphNode_t						hGraphNode;
		};

	private:

		const uint64_t						m_ID;
		std::vector<NodeInfo>				m_nodes;
		std::unordered_set<int>				m_depIndicesCache;
		std::vector<cudaGraphNode_t>		m_depHandlesCache;
		std::vector<char>					m_paramBinaries;
		cudaGraphExec_t						m_hGraphExec;
		cudaGraph_t							m_hGraph;
		Stream *							m_pImmediateLaunchStream;
		size_t								m_paramOffset;
		bool								m_isParamChg;
		bool								m_isTopoChg;
		int									m_indicator;
	};
}