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
	class Runtime;
	class Allocator;
	class TimedEvent;
	class ScopedTimer;
	class ScratchArena;
	class HostAllocator;
	class DeviceAllocator;
	using Error_t = int;

	struct Extent;
	struct Version;
	struct Sampler;
	enum class Format;
	enum class Result;
	enum class FilterMode;
	enum class AddressMode;

	//!	shortcuts for unsigned char and unsigned char pointer types (same as `std::byte`).
	using byte = unsigned char;

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

	template<typename Type> class BufferSlice;
	template<typename Type> class BufferSlice2D;
	template<typename Type> class BufferSlice3D;

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

	using AllocPtr			= std::shared_ptr<Allocator>;
	using HostAllocPtr		= std::shared_ptr<HostAllocator>;
	using DevAllocPtr		= std::shared_ptr<DeviceAllocator>;

	//!	Trait to check if two types are binary compatible in terms of size and alignment.
	template<typename Type1, typename Type2> struct BinaryCompatible
	{
		static constexpr bool value = (sizeof(Type1) == sizeof(Type2)) && (alignof(Type1) == alignof(Type2));
	};
}
