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

#include <memory>
#include "macros.h"

/*********************************************************************************
***************************    Forward Declarations    ***************************
*********************************************************************************/

#ifndef __CUDACC__
	struct dim3;
	struct cudaDeviceProp;
	typedef struct cudaArray *					cudaArray_t;
	typedef struct CUevent_st *					cudaEvent_t;
	typedef struct CUgraph_st *					cudaGraph_t;
	typedef struct CUstream_st *				cudaStream_t;
	typedef struct CUgraphNode_st *				cudaGraphNode_t;
	typedef struct CUgraphExec_st *				cudaGraphExec_t;
	typedef struct cudaMipmappedArray *			cudaMipmappedArray_t;
	typedef unsigned long long					cudaTextureObject_t;
	typedef unsigned long long					cudaSurfaceObject_t;
#endif	//	__CUDACC__

namespace NS_NAMESPACE
{
	class Event;
	class Graph;
	class Buffer;
	class Stream;
	class Device;
	class Context;
	class Allocator;
	class TimedEvent;
	class ScopedTimer;
	class HostAllocator;
	class DeviceAllocator;
	using Error_t = int;

	struct Version;
	struct Sampler;
	enum class Format;
	enum class Result;
	enum class FilterMode;
	enum class AddressMode;

	//	For device objects.
	namespace dev
	{
		template<typename Type> struct Ptr;
		template<typename Type> struct Ptr2;
		template<typename Type> struct Ptr3;

		template<typename Type> struct Surf1D;
		template<typename Type> struct Surf2D;
		template<typename Type> struct Surf3D;
		template<typename Type> struct SurfCube;
		template<typename Type> struct Surf1DLayered;
		template<typename Type> struct Surf2DLayered;
		template<typename Type> struct SurfCubeLayered;

		template<typename Type> struct Tex1D;
		template<typename Type> struct Tex2D;
		template<typename Type> struct Tex3D;
		template<typename Type> struct TexCube;
		template<typename Type> struct Tex1DLod;
		template<typename Type> struct Tex2DLod;
		template<typename Type> struct Tex3DLod;
		template<typename Type> struct TexCubeLod;
		template<typename Type> struct Tex1DLayered;
		template<typename Type> struct Tex2DLayered;
		template<typename Type> struct TexCubeLayered;
		template<typename Type> struct Tex1DLayeredLod;
		template<typename Type> struct Tex2DLayeredLod;
		template<typename Type> struct TexCubeLayeredLod;
	}

	template<typename Type> class Array;
	template<typename Type> class Array2D;
	template<typename Type> class Array3D;

	template<typename Type> class BufferView;
	template<typename Type> class BufferView2D;
	template<typename Type> class BufferView3D;

	template<typename Type> class Image1D;
	template<typename Type> class Image2D;
	template<typename Type> class Image3D;
	template<typename Type> class ImageCube;
	template<typename Type> class Image1DLayered;
	template<typename Type> class Image2DLayered;
	template<typename Type> class ImageCubeLayered;
	template<typename Type> class Image1DLod;
	template<typename Type> class Image2DLod;
	template<typename Type> class Image3DLod;
	template<typename Type> class ImageCubeLod;
	template<typename Type> class Image1DLayeredLod;
	template<typename Type> class Image2DLayeredLod;
	template<typename Type> class ImageCubeLayeredLod;

	template<typename Type> class Surface1D;
	template<typename Type> class Surface2D;
	template<typename Type> class Surface3D;
	template<typename Type> class SurfaceCube;
	template<typename Type> class Surface1DLayered;
	template<typename Type> class Surface2DLayered;
	template<typename Type> class SurfaceCubeLayered;

	template<template<typename> class Image, template<typename> class devTex, typename Type> class Texture;

	template<typename Type> using Texture1D					= Texture<Image1D, dev::Tex1D, Type>;
	template<typename Type> using Texture2D					= Texture<Image2D, dev::Tex2D, Type>;
	template<typename Type> using Texture3D					= Texture<Image3D, dev::Tex3D, Type>;
	template<typename Type> using TextureCube				= Texture<ImageCube, dev::TexCube, Type>;
	template<typename Type> using Texture1DLod				= Texture<Image1DLod, dev::Tex1DLod, Type>;
	template<typename Type> using Texture2DLod				= Texture<Image2DLod, dev::Tex2DLod, Type>;
	template<typename Type> using Texture3DLod				= Texture<Image3DLod, dev::Tex3DLod, Type>;
	template<typename Type> using TextureCubeLod			= Texture<ImageCubeLod, dev::TexCubeLod, Type>;
	template<typename Type> using Texture1DLayered			= Texture<Image1DLayered, dev::Tex1DLayered, Type>;
	template<typename Type> using Texture2DLayered			= Texture<Image2DLayered, dev::Tex2DLayered, Type>;
	template<typename Type> using TextureCubeLayered		= Texture<ImageCubeLayered, dev::TexCubeLayered, Type>;
	template<typename Type> using Texture1DLayeredLod		= Texture<Image1DLayeredLod, dev::Tex1DLayeredLod, Type>;
	template<typename Type> using Texture2DLayeredLod		= Texture<Image2DLayeredLod, dev::Tex2DLayeredLod, Type>;
	template<typename Type> using TextureCubeLayeredLod		= Texture<ImageCubeLayeredLod, dev::TexCubeLayeredLod, Type>;

	template<typename Type> struct ImageAccessor;
	template<typename... Args> using KernelFunc = void(*)(Args...);

	using BufferPtr			= std::shared_ptr<Buffer>;
	using AllocPtr			= std::shared_ptr<Allocator>;
	using HostAllocPtr		= std::shared_ptr<HostAllocator>;
	using DevAllocPtr		= std::shared_ptr<DeviceAllocator>;

	//!	Trait to check if two types are binary compatible in terms of size and alignment.
	template<typename Type1, typename Type2> struct BinaryCompatible
	{
		static constexpr bool value = (sizeof(Type1) == sizeof(Type2)) && (alignof(Type1) == alignof(Type2));
	};

	//!	Utility functions to reinterpret buffer views as another compatible element type.
	template<typename DstType, typename SrcType> BufferView<DstType> view_cast(BufferView<SrcType> view);
	template<typename DstType, typename SrcType> BufferView2D<DstType> view_cast(BufferView2D<SrcType> view);
	template<typename DstType, typename SrcType> BufferView3D<DstType> view_cast(BufferView3D<SrcType> view);

	template<typename DstType, typename SrcType> BufferView<const DstType> view_cast(BufferView<const SrcType> view);
	template<typename DstType, typename SrcType> BufferView2D<const DstType> view_cast(BufferView2D<const SrcType> view);
	template<typename DstType, typename SrcType> BufferView3D<const DstType> view_cast(BufferView3D<const SrcType> view);

	//! Utility functions to reinterpret device pointers as another compatible element type.
	template<typename DstType, typename SrcType> NS_CUDA_CALLABLE dev::Ptr<const DstType> ptr_cast(dev::Ptr<const SrcType> ptr);
	template<typename DstType, typename SrcType> NS_CUDA_CALLABLE dev::Ptr2<const DstType> ptr_cast(dev::Ptr2<const SrcType> ptr);
	template<typename DstType, typename SrcType> NS_CUDA_CALLABLE dev::Ptr3<const DstType> ptr_cast(dev::Ptr3<const SrcType> ptr);

	template<typename DstType, typename SrcType> NS_CUDA_CALLABLE dev::Ptr<DstType> ptr_cast(dev::Ptr<SrcType> ptr);
	template<typename DstType, typename SrcType> NS_CUDA_CALLABLE dev::Ptr2<DstType> ptr_cast(dev::Ptr2<SrcType> ptr);
	template<typename DstType, typename SrcType> NS_CUDA_CALLABLE dev::Ptr3<DstType> ptr_cast(dev::Ptr3<SrcType> ptr);

	/*****************************************************************************
	*****************************    SharedHandle    *****************************
	*****************************************************************************/

	/**
	 *	@brief		A thin wrapper around `std::shared_ptr<T>` that simplifies construction,
	 *				supports automatic shared ownership, and allows promotion from `std::unique_ptr<T>`.
	 *	@example	SharedHandle<Type> sharedHandle = std::make_unique<Type>(...);
	 *	@example	SharedHandle<Type> sharedHandle = std::make_shared<Type>(...);
	 *	@example	SharedHandle<Type> sharedHandle = std::move(uniqueObject);
	 *	@example	SharedHandle<Type> sharedHandle = std::move(sharedObject);
	 *	@example	SharedHandle<Type> sharedHandle = sharedObject;
	 * 	@example	SharedHandle<Type> sharedHandle = nullptr;
	 *	@example	SharedHandle<Type> sharedHandle(...);
	 */
	template<typename Type> struct SharedHandle : public std::shared_ptr<Type>
	{
		//!	@brief	Empty constructor.
		SharedHandle() = default;

		//!	@brief	Empty constructor.
		SharedHandle(std::nullptr_t) : std::shared_ptr<Type>(nullptr) {}

		//!	@brief	Constructs from an existing shared ownership reference.
		SharedHandle(const std::shared_ptr<Type> & object) : std::shared_ptr<Type>(object) {}

		//!	@brief	Constructs by moving from another `std::shared_ptr<T>`.
		SharedHandle(std::shared_ptr<Type> && object) : std::shared_ptr<Type>(std::move(object)) {}

		//!	@brief	Promotes a `std::unique_ptr<T>` to a shared object.
		SharedHandle(std::unique_ptr<Type> && object) : std::shared_ptr<Type>(std::move(object)) {}

		//!	@brief	Constructs a new shared object using forwarded arguments.
		//! @note	Here apply constructible constraint will cause compilation failure when `Type` is forward-declared.
		template<typename... Args> explicit SharedHandle(const Args &... args) : std::shared_ptr<Type>(std::make_shared<Type>(args...)) {}
	};

	/*****************************************************************************
	******************************    Type alias    ******************************
	*****************************************************************************/

	//!	Type aliases for commonly shared image objects.
	using SharedEvent											= SharedHandle<Event>;
	using SharedStream											= SharedHandle<Stream>;
	using SharedBuffer											= SharedHandle<Buffer>;
	using SharedAlloctor										= SharedHandle<Allocator>;
	using SharedHostAlloctor									= SharedHandle<HostAllocator>;
	using SharedDeviceAlloctor									= SharedHandle<DeviceAllocator>;

	template<typename Type> using SharedImage1D					= SharedHandle<Image1D<Type>>;
	template<typename Type> using SharedImage2D					= SharedHandle<Image2D<Type>>;
	template<typename Type> using SharedImage3D					= SharedHandle<Image3D<Type>>;
	template<typename Type> using SharedImageCube				= SharedHandle<ImageCube<Type>>;
	template<typename Type> using SharedImage1DLayered			= SharedHandle<Image1DLayered<Type>>;
	template<typename Type> using SharedImage2DLayered			= SharedHandle<Image2DLayered<Type>>;
	template<typename Type> using SharedImageCubeLayered		= SharedHandle<ImageCubeLayered<Type>>;
	template<typename Type> using SharedImage1DLod				= SharedHandle<Image1DLod<Type>>;
	template<typename Type> using SharedImage2DLod				= SharedHandle<Image2DLod<Type>>;
	template<typename Type> using SharedImage3DLod				= SharedHandle<Image3DLod<Type>>;
	template<typename Type> using SharedImageCubeLod			= SharedHandle<ImageCubeLod<Type>>;
	template<typename Type> using SharedImage1DLayeredLod		= SharedHandle<Image1DLayeredLod<Type>>;
	template<typename Type> using SharedImage2DLayeredLod		= SharedHandle<Image2DLayeredLod<Type>>;
	template<typename Type> using SharedImageCubeLayeredLod		= SharedHandle<ImageCubeLayeredLod<Type>>;

	template<typename Type> using SharedTexture1D				= SharedHandle<Texture1D<Type>>;
	template<typename Type> using SharedTexture2D				= SharedHandle<Texture2D<Type>>;
	template<typename Type> using SharedTexture3D				= SharedHandle<Texture3D<Type>>;
	template<typename Type> using SharedTextureCube				= SharedHandle<TextureCube<Type>>;
	template<typename Type> using SharedTexture1DLod			= SharedHandle<Texture1DLod<Type>>;
	template<typename Type> using SharedTexture2DLod			= SharedHandle<Texture2DLod<Type>>;
	template<typename Type> using SharedTexture3DLod			= SharedHandle<Texture3DLod<Type>>;
	template<typename Type> using SharedTextureCubeLod			= SharedHandle<TextureCubeLod<Type>>;
	template<typename Type> using SharedTexture1DLayered		= SharedHandle<Texture1DLayered<Type>>;
	template<typename Type> using SharedTexture2DLayered		= SharedHandle<Texture2DLayered<Type>>;
	template<typename Type> using SharedTextureCubeLayered		= SharedHandle<TextureCubeLayered<Type>>;
	template<typename Type> using SharedTexture1DLayeredLod		= SharedHandle<Texture1DLayeredLod<Type>>;
	template<typename Type> using SharedTexture2DLayeredLod		= SharedHandle<Texture2DLayeredLod<Type>>;
	template<typename Type> using SharedTextureCubeLayeredLod	= SharedHandle<TextureCubeLayeredLod<Type>>;
}