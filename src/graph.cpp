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

Graph::Graph() : m_ID(std::hash<void*>{}(this)), m_hGraph(nullptr), m_hGraphExec(nullptr), m_indicator(0), m_paramOffset(0), m_isParamChg(false), m_isTopoChg(false), m_pImmediateLaunchStream(nullptr)
{
	m_depIndicesCache.reserve(32);
	m_depHandlesCache.reserve(32);
}


ExecDep Graph::barrier(Span<const ExecDep> dependencies)
{
	if (m_pImmediateLaunchStream != nullptr)	//	in immediate launch mode
	{
		return ExecDep{ m_ID, -1 };
	}
	else //////////////////////////////////////////////////////////////////////////////////////
	{
		const uint64_t depHash = this->cacheDependencies(dependencies);

		if (m_indicator < m_nodes.size())	//	in validating state
		{
			if ((m_nodes[m_indicator].func != reinterpret_cast<void*>(IsEmptyNode)) || (m_nodes[m_indicator].depHash != depHash))	//	dependencies changes
			{
				m_nodes.resize(m_indicator);
			}
		}

		if (m_indicator >= m_nodes.size())	//	validation failed
		{
			auto createFunc = [](cudaGraph_t hGraph, const cudaGraphNode_t * pDependencies, size_t numDependencies) -> cudaGraphNode_t
			{
				cudaGraphNode_t hGraphNode = nullptr;

				cudaError_t err = cudaGraphAddEmptyNode(&hGraphNode, hGraph, pDependencies, numDependencies);

				NS_ERROR_LOG_IF(err != cudaSuccess, "%s.", cudaGetErrorString(cudaGetLastError()));

				return hGraphNode;
			};

			m_nodes.emplace_back(reinterpret_cast<void*>(IsEmptyNode), depHash, 0, m_depIndicesCache, createFunc);

			m_isTopoChg = true;
		}

		return ExecDep{ m_ID, m_indicator++ };
	}
}


ExecDep Graph::memcpy_void(void * dst, const void * src, size_t bytes, Span<const ExecDep> dependencies)
{
	if (m_pImmediateLaunchStream != nullptr)	//	in immediate launch mode
	{
		m_pImmediateLaunchStream->memcpy(dst, src, bytes);

		return ExecDep{ m_ID, -1 };
	}
	else ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	{
		const uint64_t depHash = this->cacheDependencies(dependencies);

		constexpr size_t paramBytes = sizeof(dst) + sizeof(src) + sizeof(bytes);

		const size_t paramCache[3] = { reinterpret_cast<size_t>(dst), reinterpret_cast<size_t>(src), bytes };

		if (m_indicator < m_nodes.size())	//	in validating state
		{
			if ((m_nodes[m_indicator].func != reinterpret_cast<void*>(IsMemcpyNode)) || (m_nodes[m_indicator].depHash != depHash))	//	dependencies changes
			{
				m_nodes.resize(m_indicator);
			}
			else if (std::memcmp(m_paramBinaries.data() + m_paramOffset, paramCache, paramBytes) != 0)	//	parameters changes
			{
				cudaError_t err = cudaGraphMemcpyNodeSetParams1D(m_nodes[m_indicator].hGraphNode, dst, src, bytes, cudaMemcpyDefault);

				NS_ERROR_LOG_IF(err != cudaSuccess, "%s.", cudaGetErrorString(cudaGetLastError()));

				std::memcpy(m_paramBinaries.data() + m_paramOffset, paramCache, paramBytes);

				m_isParamChg = true;
			}
		}

		if (m_indicator >= m_nodes.size())	//	topology changed
		{
			auto createFunc = [=](cudaGraph_t hGraph, const cudaGraphNode_t * pDependencies, size_t numDependencies) -> cudaGraphNode_t
			{
				cudaGraphNode_t hGraphNode = nullptr;

				cudaError_t err = cudaGraphAddMemcpyNode1D(&hGraphNode, hGraph, pDependencies, numDependencies, dst, src, bytes, cudaMemcpyDefault);

				NS_ERROR_LOG_IF(err != cudaSuccess, "%s.", cudaGetErrorString(cudaGetLastError()));

				return hGraphNode;
			};

			m_paramBinaries.resize(m_paramBinaries.size() + paramBytes);

			std::memcpy(m_paramBinaries.data() + m_paramOffset, paramCache, paramBytes);

			m_nodes.emplace_back(reinterpret_cast<void*>(IsMemcpyNode), depHash, paramBytes, m_depIndicesCache, createFunc);

			m_isTopoChg = true;
		}

		m_paramOffset += paramBytes;

		return ExecDep{ m_ID, m_indicator++ };
	}
}


uint64_t Graph::cacheDependencies(Span<const ExecDep> dependencies)
{
	m_depIndicesCache.clear();

	for (size_t i = 0; i < dependencies.size(); i++)
	{
		if (dependencies[i].Index == -1)	continue;	//	skip ExecDep(x, -1) for convenience

		NS_ASSERT_LOG_IF(dependencies[i].ID != m_ID, "Invalid dependency!");
		NS_ASSERT_LOG_IF(dependencies[i].Index >= m_indicator, "Invalid dependency!");

		if ((dependencies[i].ID == m_ID) && (dependencies[i].Index < m_indicator))	//	dependency index should always smaller than current index
		{
			m_depIndicesCache.emplace(dependencies[i].Index);
		}
	}

	constexpr auto HashCombine = [](uint64_t hash_0, uint64_t hash_1)
	{
		constexpr uint64_t hash_magic = 0x9E3779B97f4A7C55;

		return hash_0 ^ (hash_1 + hash_magic + (hash_0 << 6) + (hash_0 >> 2));
	};

	uint64_t depHash = m_ID;

	for (auto depIndex : m_depIndicesCache)
	{
		depHash = HashCombine(depHash, depIndex);
	}

	return depHash;
}


void Graph::execute(Stream * pStream)
{
	if (m_pImmediateLaunchStream != nullptr)	return;

	if (m_isTopoChg || (m_indicator != m_nodes.size()))
	{
		if (m_hGraph != nullptr)
		{
			cudaError_t err = cudaGraphDestroy(m_hGraph);

			NS_ERROR_LOG_IF(err != cudaSuccess, "%s.", cudaGetErrorString(err));

			m_hGraph = nullptr;
		}
		if (m_hGraph == nullptr)
		{
			cudaError_t err = cudaGraphCreate(&m_hGraph, 0);

			NS_ERROR_LOG_IF(err != cudaSuccess, "%s.", cudaGetErrorString(err));
		}

		//////////////////////////////////////////////////////////////////////////////////

		m_nodes.resize(m_indicator);

		for (size_t i = 0; i < m_nodes.size(); i++)
		{
			m_depHandlesCache.clear();

			for (auto depIndex : m_nodes[i].depIndices)
			{
				m_depHandlesCache.emplace_back(m_nodes[depIndex].hGraphNode);
			}

			m_nodes[i].hGraphNode = m_nodes[i].createFunc(m_hGraph, m_depHandlesCache.data(), m_depHandlesCache.size());
		}

		m_isTopoChg = true;
	}

	//////////////////////////////////////////////////////////////////////////////////////

	if (m_isTopoChg || m_isParamChg)
	{
		if (m_hGraphExec != nullptr)
		{
			cudaError_t err = cudaGraphExecDestroy(m_hGraphExec);

			NS_ERROR_LOG_IF(err != cudaSuccess, "%s.", cudaGetErrorString(err));

			m_hGraphExec = nullptr;
		}
		if (m_hGraphExec == nullptr)
		{
			cudaError_t err = cudaGraphInstantiate(&m_hGraphExec, m_hGraph);

			NS_ERROR_LOG_IF(err != cudaSuccess, "%s.", cudaGetErrorString(err));

			NS_DEBUG_LOG_IF(err == cudaSuccess, "Graph instantiate succeeded with %lld nodes.", m_nodes.size());
		}

		m_isParamChg = false;

		m_isTopoChg = false;
	}

	//////////////////////////////////////////////////////////////////////////////////////

	if (m_hGraphExec != nullptr)
	{
		pStream->launchGraph(m_hGraphExec);
	}
}


void Graph::restart(Stream * pImmediateLaunchStream)
{
	m_pImmediateLaunchStream = pImmediateLaunchStream;

	if (m_hGraphExec == nullptr)
	{
		m_nodes.clear();
	}

	m_isParamChg = false;

	m_paramOffset = 0;

	m_indicator = 0;
}


Graph::~Graph()
{
	if (m_hGraphExec != nullptr)
	{
		cudaError_t err = cudaGraphExecDestroy(m_hGraphExec);

		NS_ERROR_LOG_IF(err != cudaSuccess, "%s.", cudaGetErrorString(err));
	}
	if (m_hGraph != nullptr)
	{
		cudaError_t err = cudaGraphDestroy(m_hGraph);

		NS_ERROR_LOG_IF(err != cudaSuccess, "%s.", cudaGetErrorString(err));
	}
}