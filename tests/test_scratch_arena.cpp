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

#include <new>
#include <cstdint>
#include <type_traits>
#include <nucleus/scratch_arena.h>

/*********************************************************************************
****************************    test_scratch_arena    ****************************
*********************************************************************************/

class ScratchArenaAllocator : public ns::Allocator
{

public:

	size_t allocations = 0;
	size_t deallocations = 0;

private:

	void * do_allocate_memory(size_t bytes) override
	{
		allocations++;

		return ::operator new(bytes);
	}

	void do_deallocate_memory(void * ptr) override
	{
		deallocations++;

		::operator delete(ptr);
	}
};


void test_scratch_arena()
{
	static_assert(!std::is_copy_assignable_v<ns::ScratchArena>);
	static_assert(!std::is_copy_constructible_v<ns::ScratchArena>);

	auto allocator = std::make_shared<ScratchArenaAllocator>();
	ns::ScratchArena arena(allocator);

	assert(arena.empty());
	assert(arena.used() == 0);
	assert(arena.capacity() == 0);
	assert(arena.remaining() == 0);
	assert(arena.allocator() == allocator);

	size_t required_capacity = 0;
	required_capacity = ns::aligned_end_offset<int>(required_capacity, 1);
	required_capacity = ns::aligned_end_offset<double>(required_capacity, 12);
	required_capacity = ns::aligned_end_offset<ns::byte>(required_capacity, 24);

	arena.reserve(required_capacity);
	arena.reuse();

	assert(!arena.empty());
	assert(arena.used() == 0);
	assert(arena.capacity() == required_capacity);
	assert(arena.remaining() == required_capacity);

	auto marker = arena.allocate<int>(1);
	auto matrix = arena.allocate2D<double>(3, 4);
	auto volume = arena.allocate3D<ns::byte>(2, 3, 4);

	const auto cached_address = reinterpret_cast<std::uintptr_t>(marker.data());

	assert(marker.size() == 1);
	assert(matrix.width() == 3);
	assert(matrix.height() == 4);
	assert(matrix.size() == 12);
	assert(volume.width() == 2);
	assert(volume.height() == 3);
	assert(volume.depth() == 4);
	assert(volume.size() == 24);
	assert(reinterpret_cast<std::uintptr_t>(matrix.data()) % alignof(int) == 0);
	assert(reinterpret_cast<std::uintptr_t>(volume.data()) % alignof(double) == 0);
	assert(arena.used() == required_capacity);
	assert(arena.remaining() == 0);

	arena.reuse();

	assert(arena.used() == 0);
	assert(arena.remaining() == required_capacity);
	assert(allocator->allocations == 1);
	assert(allocator->deallocations == 0);

	constexpr size_t smaller_capacity = ns::aligned_end_offset<int>(0, 4);

	arena.reserve(smaller_capacity);
	arena.reuse();

	auto values = arena.allocate<int>(4);

	assert(reinterpret_cast<std::uintptr_t>(values.data()) == cached_address);
	assert(arena.capacity() == required_capacity);
	assert(arena.used() == smaller_capacity);
	assert(allocator->deallocations == 0);
	assert(allocator->allocations == 1);

	arena.reserve(required_capacity * 2);

	assert(arena.used() == 0);
	assert(arena.capacity() == required_capacity * 2);
	assert(allocator->deallocations == 1);
	assert(allocator->allocations == 2);

	arena.reserve(required_capacity);

	assert(allocator->allocations == 2);
	assert(allocator->deallocations == 1);

	arena.clear();

	assert(arena.empty());
	assert(arena.used() == 0);
	assert(arena.capacity() == 0);
	assert(arena.remaining() == 0);
	assert(allocator->deallocations == 2);
	assert(allocator->allocations == 2);
}
