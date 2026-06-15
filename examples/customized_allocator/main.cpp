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

#include <map>
#include <nucleus/runtime.h>
#include <nucleus/array_1d.h>
#include <nucleus/allocator.h>

/*********************************************************************************
***************************    customized_allocator    ***************************
*********************************************************************************/

class MyAllocator : public ns::DeviceAllocator
{

public:

	MyAllocator(ns::Device * device) : ns::DeviceAllocator(device) {}

private:

	virtual void * doAllocateMemory(size_t bytes) override
	{
		m_totalMem += bytes;

		printf(" Allocate global memory: %lld, total allocated: %lld.\n", bytes, m_totalMem);

		auto ptr = ns::DeviceAllocator::doAllocateMemory(bytes);

		m_ptrSizeTable[ptr] = bytes;

		return ptr;
	}

	virtual void doDeallocateMemory(void * ptr) override
	{
		if (m_ptrSizeTable.find(ptr) != m_ptrSizeTable.end())
		{
			m_totalMem -= m_ptrSizeTable[ptr];

			ns::DeviceAllocator::doDeallocateMemory(ptr);

			printf(" Deallocate global memory, total allocated: %lld.\n", m_totalMem);

			m_ptrSizeTable.erase(ptr);
		}
	}

private:

	size_t						m_totalMem = 0;
	std::map<void*, size_t>		m_ptrSizeTable;
};


int main()
{
	auto device = ns::Runtime::device(0);
	auto devAlloc = std::make_shared<MyAllocator>(device);

	printf("\n");

	ns::Array<int>		myData(devAlloc, 100);
	myData.resize(200);
	myData.resize(300);
	myData.resize(400);
	myData.clear();

	printf("\n");
	system("pause");

	return 0;
}