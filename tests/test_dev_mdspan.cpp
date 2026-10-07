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

#include <nucleus/device_mdspan.h>

/*********************************************************************************
***************************    test_dev_mdspan_host    ***************************
*********************************************************************************/

struct MdspanIndex
{
	constexpr operator size_t() const noexcept { return 1; }
};


template<typename View> concept MdspanHasSubscript = requires(const View & view) { view[0]; };
template<typename View> concept MdspanHasWrongRankCall = requires(const View & view) { view(0); };


template<typename Type, size_t... Extents> constexpr const Type & deduce_const_mdspan(dev::Mdspan<const Type, Extents...> view)
{
	return view.flatten()[0];
}


template<typename Type, size_t... Extents> constexpr const Type & deduce_const_mdspan_reference(const dev::Mdspan<const Type, Extents...> & view)
{
	return view.flatten()[0];
}


void test_dev_mdspan_host()
{
	using Fixed = dev::Mdspan<int, 2, 3>;
	using Mixed = dev::Mdspan<int, ns::dynamic_extent, 3>;
	using Dynamic = dev::Mdspan<int, ns::dynamic_extent, ns::dynamic_extent>;
	using Zero = dev::Mdspan<int, 2, 0>;
	using ConstFixed = dev::Mdspan<const int, 2, 3>;
	using ConstMixed = dev::Mdspan<const int, ns::dynamic_extent, 3>;
	using OneDimension = dev::Mdspan<int, 6>;

	static_assert(sizeof(Fixed) == sizeof(int*));
	static_assert(sizeof(Mixed) == sizeof(int*) + sizeof(size_t));
	static_assert(sizeof(Dynamic) == sizeof(int*) + 2 * sizeof(size_t));
	static_assert(sizeof(Fixed) == sizeof(ConstFixed));
	static_assert(sizeof(Mixed) == sizeof(ConstMixed));
	static_assert(std::is_trivially_copyable_v<Fixed>);
	static_assert(std::is_trivially_copyable_v<Dynamic>);
	static_assert(std::is_trivially_copyable_v<ConstFixed>);
	static_assert(std::is_base_of_v<ConstFixed, Fixed>);
	static_assert(std::is_base_of_v<ConstMixed, Mixed>);
	static_assert(std::is_convertible_v<Fixed*, ConstFixed*>);
	static_assert(!std::is_default_constructible_v<Fixed>);
	static_assert(std::is_default_constructible_v<Zero>);
	static_assert(std::is_default_constructible_v<Dynamic>);
	static_assert(!std::is_constructible_v<Fixed, const int*>);
	static_assert(!std::is_constructible_v<Fixed, double*, int, int>);
	static_assert(!std::is_constructible_v<Fixed, Zero>);
	static_assert(!std::is_constructible_v<Mixed, OneDimension>);
	static_assert(!std::is_constructible_v<Dynamic, int*, int>);
	static_assert(std::is_constructible_v<Dynamic, int*, MdspanIndex, MdspanIndex>);
	static_assert(std::is_convertible_v<Fixed, Mixed>);
	static_assert(std::is_convertible_v<Mixed, Fixed>);
	static_assert(std::is_convertible_v<Fixed, ConstFixed>);
	static_assert(!std::is_constructible_v<Fixed, ConstFixed>);
	static_assert(!std::is_assignable_v<Fixed &, ConstFixed>);
	static_assert(!MdspanHasSubscript<Fixed>);
	static_assert(!MdspanHasWrongRankCall<Fixed>);
	static_assert(MdspanHasSubscript<OneDimension>);
	static_assert(Fixed::rank() == 2);
	static_assert(Fixed::rank_dynamic() == 0);
	static_assert(Mixed::rank_dynamic() == 1);
	static_assert(Fixed::static_extent(0) == 2);
	static_assert(Fixed::static_extent(1) == 3);
	static_assert(Mixed::static_extent(0) == ns::dynamic_extent);
	static_assert(Fixed::size() == 6);
	static_assert(Fixed::size_bytes() == 6 * sizeof(int));
	static_assert(std::is_same_v<Fixed::element_type, int>);
	static_assert(std::is_same_v<Fixed::value_type, int>);
	static_assert(std::is_same_v<Fixed::reference, int&>);
	static_assert(std::is_same_v<Fixed::pointer, int*>);
	static_assert(std::is_same_v<ConstFixed::element_type, const int>);
	static_assert(std::is_same_v<ConstFixed::reference, const int&>);
	static_assert(std::is_same_v<ConstFixed::pointer, const int*>);

	constexpr Mixed default_constructor;
	constexpr Mixed default_copy_constructor = default_constructor;
	constexpr ConstMixed copy_from_non_const = default_constructor;
	constexpr Zero zero_extent;
	constexpr Dynamic dynamic_default;
	static_assert(default_copy_constructor.empty());
	static_assert(default_copy_constructor.data() == nullptr);
	static_assert(default_copy_constructor.extent(0) == 0);
	static_assert(default_copy_constructor.extent(1) == 3);
	static_assert(copy_from_non_const.size() == 0);
	static_assert(copy_from_non_const.size_bytes() == 0);
	static_assert(zero_extent.size() == 0);
	static_assert(zero_extent.empty());
	static_assert(dynamic_default.extent(0) == 0 && dynamic_default.extent(1) == 0);

	constexpr ConstFixed null_fixed(nullptr);
	constexpr ConstMixed null_mixed = null_fixed;
	static_assert(null_fixed.size() == 6);
	static_assert(null_fixed.empty());
	static_assert(null_mixed.extent(0) == 2);
	static_assert(null_mixed.size() == 6);
	static_assert(null_mixed.empty());

	int arr[6] = { 0, 1, 2, 3, 4, 5 };
	Fixed fixed(arr);
	Fixed fixed_all_extents(arr, 2, 3);
	Mixed mixed(arr, 2);
	Mixed mixed_all_extents(arr, 2, 3);
	Mixed mixed_from_fixed = fixed;
	Dynamic dynamic(arr, 2, 3);
	Dynamic dynamic_from_mixed = mixed;
	ConstFixed const_from_fixed = fixed;
	const Fixed & const_fixed = fixed;
	Fixed fixed_from_dynamic = dynamic;
	ConstMixed const_mixed = mixed;
	ConstFixed const_from_dynamic = const_mixed;

	NS_ASSERT(!fixed.empty());
	NS_ASSERT(fixed.data() == arr);
	NS_ASSERT(fixed_all_extents(1, 2) == 5);
	NS_ASSERT(mixed.size() == 6);
	NS_ASSERT(mixed.size_bytes() == 6 * sizeof(int));
	NS_ASSERT(mixed_all_extents.size() == 6);
	NS_ASSERT(mixed_from_fixed.extent(0) == 2);
	NS_ASSERT(dynamic_from_mixed.extent(1) == 3);
	NS_ASSERT(dynamic.size() == 6);
	NS_ASSERT(dynamic.stride(0) == 3);
	NS_ASSERT(dynamic.stride(1) == 1);
	NS_ASSERT(const_from_fixed(1, 2) == 5);
	NS_ASSERT(&fixed(1, 2) == arr + 5);
	NS_ASSERT(fixed_from_dynamic(1, 2) == 5);
	NS_ASSERT(const_from_dynamic(1, 2) == 5);
	static_assert(std::is_same_v<decltype(fixed(1, 2)), int&>);
	static_assert(std::is_same_v<decltype(const_fixed(1, 2)), const int&>);
	static_assert(std::is_same_v<decltype(const_from_fixed(1, 2)), const int&>);
	static_assert(std::is_same_v<decltype(const_fixed.data()), const int*>);
	static_assert(std::is_same_v<decltype(fixed.flatten()), dev::Span<int, 6>>);
	static_assert(std::is_same_v<decltype(const_fixed.flatten()), dev::Span<const int, 6>>);
	static_assert(std::is_same_v<decltype(mixed.flatten()), dev::Span<int>>);
	static_assert(std::is_same_v<decltype(const_mixed.flatten()), dev::Span<const int>>);

	fixed(1, 2) = 42;
	NS_ASSERT(const_fixed(1, 2) == 42);
	auto flat = fixed.flatten();
	NS_ASSERT(flat.data() == arr);
	NS_ASSERT(flat.size() == 6);
	flat[4] = 24;
	NS_ASSERT(mixed(MdspanIndex{}, MdspanIndex{}) == 24);
	NS_ASSERT(const_fixed.flatten()[4] == 24);
	NS_ASSERT(mixed.flatten().size_bytes() == mixed.size_bytes());

	OneDimension one_dimension(arr);
	const auto & const_one_dimension = one_dimension;
	one_dimension[0] = 12;
	NS_ASSERT(const_one_dimension[0] == 12);
	NS_ASSERT(one_dimension[5] == 42);
	static_assert(std::is_same_v<decltype(const_one_dimension[0]), const int&>);
	static_assert(std::is_same_v<decltype(deduce_const_mdspan(fixed)), const int&>);
	NS_ASSERT(&deduce_const_mdspan(fixed) == arr);
	NS_ASSERT(&deduce_const_mdspan_reference(const_fixed) == arr);
	NS_ASSERT(&deduce_const_mdspan(mixed) == arr);

	Dynamic non_null_zero_extent(arr, 0, 3);
	NS_ASSERT(non_null_zero_extent.size() == 0);
	NS_ASSERT(!non_null_zero_extent.empty());
	NS_ASSERT(non_null_zero_extent.flatten().size() == 0);
	NS_ASSERT(non_null_zero_extent.flatten().data() == arr);

	int other_arr[3]{};
	Mixed other(other_arr, 1);
	other = mixed;
	NS_ASSERT(other.data() == arr);
	NS_ASSERT(other.extent(0) == 2);
	NS_ASSERT(other.size() == 6);
}
