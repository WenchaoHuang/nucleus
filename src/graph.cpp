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

#include "graph.h"
#include "logger.h"
#include "stream.h"

#include <cstring>
#include <cuda_runtime_api.h>

NS_USING_NAMESPACE

static void IsEmptyNode() {}
static void IsMemcpyNode() {}

/*********************************************************************************
**********************************    Graph    ***********************************
*********************************************************************************/

Graph::Graph() : id_(std::hash<void*>{}(this)), graph_(nullptr), graph_exec_(nullptr), indicator_(0), param_offset_(0), is_param_chg_(false), is_topo_chg_(false), immediate_launch_stream_(nullptr)
{
	dep_indices_cache_.reserve(32);
	dep_handles_cache_.reserve(32);
}


ExecDep Graph::barrier(Span<const ExecDep> dependencies)
{
	if (immediate_launch_stream_ != nullptr)	//	in immediate launch mode
	{
		return ExecDep{ id_, -1 };
	}
	else //////////////////////////////////////////////////////////////////////////////////////
	{
		const uint64_t dep_hash = this->cache_dependencies(dependencies);

		if (indicator_ < nodes_.size())	//	in validating state
		{
			if ((nodes_[indicator_].func != reinterpret_cast<void*>(IsEmptyNode)) || (nodes_[indicator_].dep_hash != dep_hash))	//	dependencies changes
			{
				nodes_.resize(indicator_);
			}
		}

		if (indicator_ >= nodes_.size())	//	validation failed
		{
			auto create_func = [](cudaGraph_t graph, const cudaGraphNode_t * dependencies, size_t num_dependencies) -> cudaGraphNode_t
			{
				cudaGraphNode_t graph_node = nullptr;

				cudaError_t err = cudaGraphAddEmptyNode(&graph_node, graph, dependencies, num_dependencies);

				NS_ERROR_LOG_IF(err != cudaSuccess, "%s.", cudaGetErrorString(cudaGetLastError()));

				return graph_node;
			};

			nodes_.emplace_back(reinterpret_cast<void*>(IsEmptyNode), dep_hash, 0, dep_indices_cache_, create_func);

			is_topo_chg_ = true;
		}

		return ExecDep{ id_, indicator_++ };
	}
}


ExecDep Graph::memcpy_void(void * dst, const void * src, size_t bytes, Span<const ExecDep> dependencies)
{
	if (immediate_launch_stream_ != nullptr)	//	in immediate launch mode
	{
		immediate_launch_stream_->memcpy(dst, src, bytes);

		return ExecDep{ id_, -1 };
	}
	else ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	{
		const uint64_t dep_hash = this->cache_dependencies(dependencies);

		constexpr size_t param_bytes = sizeof(dst) + sizeof(src) + sizeof(bytes);

		const size_t param_cache[3] = { reinterpret_cast<size_t>(dst), reinterpret_cast<size_t>(src), bytes };

		if (indicator_ < nodes_.size())	//	in validating state
		{
			if ((nodes_[indicator_].func != reinterpret_cast<void*>(IsMemcpyNode)) || (nodes_[indicator_].dep_hash != dep_hash))	//	dependencies changes
			{
				nodes_.resize(indicator_);
			}
			else if (std::memcmp(param_binaries_.data() + param_offset_, param_cache, param_bytes) != 0)	//	parameters changes
			{
				cudaError_t err = cudaGraphMemcpyNodeSetParams1D(nodes_[indicator_].graph_node, dst, src, bytes, cudaMemcpyDefault);

				NS_ERROR_LOG_IF(err != cudaSuccess, "%s.", cudaGetErrorString(cudaGetLastError()));

				std::memcpy(param_binaries_.data() + param_offset_, param_cache, param_bytes);

				is_param_chg_ = true;
			}
		}

		if (indicator_ >= nodes_.size())	//	topology changed
		{
			auto create_func = [=](cudaGraph_t graph, const cudaGraphNode_t * dependencies, size_t num_dependencies) -> cudaGraphNode_t
			{
				cudaGraphNode_t graph_node = nullptr;

				cudaError_t err = cudaGraphAddMemcpyNode1D(&graph_node, graph, dependencies, num_dependencies, dst, src, bytes, cudaMemcpyDefault);

				NS_ERROR_LOG_IF(err != cudaSuccess, "%s.", cudaGetErrorString(cudaGetLastError()));

				return graph_node;
			};

			param_binaries_.resize(param_binaries_.size() + param_bytes);

			std::memcpy(param_binaries_.data() + param_offset_, param_cache, param_bytes);

			nodes_.emplace_back(reinterpret_cast<void*>(IsMemcpyNode), dep_hash, param_bytes, dep_indices_cache_, create_func);

			is_topo_chg_ = true;
		}

		param_offset_ += param_bytes;

		return ExecDep{ id_, indicator_++ };
	}
}


uint64_t Graph::cache_dependencies(Span<const ExecDep> dependencies)
{
	dep_indices_cache_.clear();

	for (size_t i = 0; i < dependencies.size(); i++)
	{
		if (dependencies[i].Index == -1)	continue;	//	skip ExecDep(x, -1) for convenience

		NS_ASSERT_LOG_IF(dependencies[i].ID != id_, "Invalid dependency!");
		NS_ASSERT_LOG_IF(dependencies[i].Index >= indicator_, "Invalid dependency!");

		if ((dependencies[i].ID == id_) && (dependencies[i].Index < indicator_))	//	dependency index should always smaller than current index
		{
			dep_indices_cache_.emplace(dependencies[i].Index);
		}
	}

	constexpr auto HashCombine = [](uint64_t hash_0, uint64_t hash_1)
	{
		constexpr uint64_t hash_magic = 0x9E3779B97f4A7C55;

		return hash_0 ^ (hash_1 + hash_magic + (hash_0 << 6) + (hash_0 >> 2));
	};

	uint64_t dep_hash = id_;

	for (auto dep_index : dep_indices_cache_)
	{
		dep_hash = HashCombine(dep_hash, dep_index);
	}

	return dep_hash;
}


void Graph::execute(Stream * stream)
{
	if (immediate_launch_stream_ != nullptr)	return;

	if (is_topo_chg_ || (indicator_ != nodes_.size()))
	{
		if (graph_ != nullptr)
		{
			cudaError_t err = cudaGraphDestroy(graph_);

			NS_ERROR_LOG_IF(err != cudaSuccess, "%s.", cudaGetErrorString(err));

			graph_ = nullptr;
		}
		if (graph_ == nullptr)
		{
			cudaError_t err = cudaGraphCreate(&graph_, 0);

			NS_ERROR_LOG_IF(err != cudaSuccess, "%s.", cudaGetErrorString(err));
		}

		//////////////////////////////////////////////////////////////////////////////////

		nodes_.resize(indicator_);

		for (size_t i = 0; i < nodes_.size(); i++)
		{
			dep_handles_cache_.clear();

			for (auto dep_index : nodes_[i].dep_indices)
			{
				dep_handles_cache_.emplace_back(nodes_[dep_index].graph_node);
			}

			nodes_[i].graph_node = nodes_[i].create_func(graph_, dep_handles_cache_.data(), dep_handles_cache_.size());
		}

		is_topo_chg_ = true;
	}

	//////////////////////////////////////////////////////////////////////////////////////

	if (is_topo_chg_ || is_param_chg_)
	{
		if (graph_exec_ != nullptr)
		{
			cudaError_t err = cudaGraphExecDestroy(graph_exec_);

			NS_ERROR_LOG_IF(err != cudaSuccess, "%s.", cudaGetErrorString(err));

			graph_exec_ = nullptr;
		}
		if (graph_exec_ == nullptr)
		{
			cudaError_t err = cudaGraphInstantiate(&graph_exec_, graph_);

			NS_ERROR_LOG_IF(err != cudaSuccess, "%s.", cudaGetErrorString(err));

			NS_DEBUG_LOG_IF(err == cudaSuccess, "Graph instantiate succeeded with %lld nodes.", nodes_.size());
		}

		is_param_chg_ = false;

		is_topo_chg_ = false;
	}

	//////////////////////////////////////////////////////////////////////////////////////

	if (graph_exec_ != nullptr)
	{
		stream->launch_graph(graph_exec_);
	}
}


void Graph::restart(Stream * immediate_launch_stream)
{
	immediate_launch_stream_ = immediate_launch_stream;

	if (graph_exec_ == nullptr)
	{
		nodes_.clear();
	}

	is_param_chg_ = false;

	param_offset_ = 0;

	indicator_ = 0;
}


Graph::~Graph()
{
	if (graph_exec_ != nullptr)
	{
		cudaError_t err = cudaGraphExecDestroy(graph_exec_);

		NS_ERROR_LOG_IF(err != cudaSuccess, "%s.", cudaGetErrorString(err));
	}
	if (graph_ != nullptr)
	{
		cudaError_t err = cudaGraphDestroy(graph_);

		NS_ERROR_LOG_IF(err != cudaSuccess, "%s.", cudaGetErrorString(err));
	}
}
