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

#include "format.h"

#ifdef __CUDACC__
	#include <surface_indirect_functions.h>
#endif

namespace NS_NAMESPACE::dev
{
	/*****************************************************************************
	*******************************    Surface    ********************************
	*****************************************************************************/

	/**
	 *	@brief		Base class for all CUDA surface objects.
	 *	@details	Provides empy constructor for compatibility with CUDA constant memory.
	 *	@details	This struct encapsulates a CUDA surface object handle and provides
	 *				common interface methods for derived surface types. It allows querying
	 *				the underlying CUDA surface handle, checking if the surface is valid or empty,
	 *				and supports implicit boolean conversion for validity checks.
	 */
	struct Surface
	{
		//	Default constructor.
		NS_CUDA_CALLABLE Surface() {}

		//	Constructor with nullptr.
		NS_CUDA_CALLABLE Surface(std::nullptr_t) : m_hSurface(0) {}

		//	Constructor with cudaSurfaceObject_t.
		NS_CUDA_CALLABLE explicit Surface(cudaSurfaceObject_t hSurface) : m_hSurface(hSurface) {}

		//	Return CUDA type of this object for compatibility.
		NS_CUDA_CALLABLE cudaSurfaceObject_t handle() const { return m_hSurface; }

		//	Bool conversion operator.
		NS_CUDA_CALLABLE operator bool() const { return m_hSurface != 0; }

		//	Check if the surface is empty.
		NS_CUDA_CALLABLE bool empty() const { return m_hSurface == 0; }

	protected:

		cudaSurfaceObject_t		m_hSurface;
	};

	/*****************************************************************************
	**************************    Surf1D<const Type>    **************************
	*****************************************************************************/

	/**
	 *	@brief		Represents a 1D CUDA surface object for device access.
	 *	@tparam		Type - The data type stored in the surface.
	 *	@details	This struct provides an interface for 1D CUDA surface objects,
	 *				enabling device-side read and write operations (read-only for const Type).
	 */
	template<typename Type> struct Surf1D<const Type> : public Surface
	{
		//	Default constructor.
		NS_CUDA_CALLABLE Surf1D() {}

		//	Constructor with nullptr.
		NS_CUDA_CALLABLE Surf1D(std::nullptr_t) : Surface(nullptr), m_width(0) {}

		//	Constructor with cudaSurfaceObject_t.
		NS_CUDA_CALLABLE explicit Surf1D(cudaSurfaceObject_t hSurface, uint32_t width) : Surface(hSurface), m_width(width) {}

		//	Return width of the buffer.
		NS_CUDA_CALLABLE uint32_t width() const { return m_width; }

		//	Read method for CUDA surface object.
	#ifndef __CUDACC__
		__device__ Type read(int x, int boundaryMode = 2) const;
	#else
		__device__ Type read(int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) const
		{
			InternalValueType<Type> value;

			surf1Dread<decltype(value)>(&value, Surface::m_hSurface, sizeof(Type) * x, boundaryMode);

			return reinterpret_cast<Type&>(value);
		}
	#endif

	protected:

		uint32_t		m_width;
	};

	/*****************************************************************************
	*****************************    Surf1D<Type>    *****************************
	*****************************************************************************/

	/**
	 *	@brief		Represents a 1D CUDA surface object for device access.
	 *	@tparam		Type - The data type stored in the surface.
	 *	@details	This struct provides an interface for 1D CUDA surface objects,
	 *				enabling device-side read and write operations (read-only for const Type).
	 */
	template<typename Type> struct Surf1D : public Surf1D<const Type>
	{
		//	Default constructor.
		NS_CUDA_CALLABLE Surf1D() {}

		//	Constructor with nullptr.
		NS_CUDA_CALLABLE Surf1D(std::nullptr_t) : Surf1D<const Type>(nullptr) {}

		//	Constructor with cudaSurfaceObject_t.
		NS_CUDA_CALLABLE explicit Surf1D(cudaSurfaceObject_t hSurface, uint32_t width) : Surf1D<const Type>(hSurface, width) {}
		
		//	Write method for CUDA surface object.
	#ifndef __CUDACC__
		__device__ void write(Type value, int x, int boundaryMode = 2) const;
	#else
		__device__ void write(Type value, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) const
		{
			surf1Dwrite<InternalValueType<Type>>(reinterpret_cast<InternalValueType<Type>&>(value), Surface::m_hSurface, sizeof(Type) * x, boundaryMode);
		}
	#endif
	};

	/*****************************************************************************
	**************************    Surf2D<const Type>    **************************
	*****************************************************************************/

	/**
	 *	@brief		Represents a 2D CUDA surface object for device access.
	 *	@tparam		Type - The data type stored in the surface.
	 *	@details	This struct provides an interface for 2D CUDA surface objects,
	 *				enabling device-side read and write operations (read-only for const Type).
	 */
	template<typename Type> struct Surf2D<const Type> : public Surface
	{
		//	Default constructor.
		NS_CUDA_CALLABLE Surf2D() {}

		//	Constructor with nullptr.
		NS_CUDA_CALLABLE Surf2D(std::nullptr_t) : Surface(nullptr), m_width(0), m_height(0) {}

		//	Constructor with cudaSurfaceObject_t.
		NS_CUDA_CALLABLE explicit Surf2D(cudaSurfaceObject_t hSurface, uint32_t width, uint32_t height) : Surface(hSurface), m_width(width), m_height(height) {}

		//	Return height of the buffer.
		NS_CUDA_CALLABLE uint32_t height() const { return m_height; }

		//	Return width of the buffer.
		NS_CUDA_CALLABLE uint32_t width() const { return m_width; }

		//	Read method for CUDA surface object.
	#ifndef __CUDACC__
		__device__ Type read(int x, int y, int boundaryMode = 2) const;
	#else
		__device__ Type read(int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) const
		{
			InternalValueType<Type> value;

			surf2Dread<decltype(value)>(&value, Surface::m_hSurface, sizeof(Type) * x, y, boundaryMode);

			return reinterpret_cast<Type&>(value);
		}
	#endif

	protected:

		uint32_t		m_width;
		uint32_t		m_height;
	};

	/*****************************************************************************
	*****************************    Surf2D<Type>    *****************************
	*****************************************************************************/

	/**
	 *	@brief		Represents a 2D CUDA surface object for device access.
	 *	@tparam		Type - The data type stored in the surface.
	 *	@details	This struct provides an interface for 2D CUDA surface objects,
	 *				enabling device-side read and write operations (read-only for const Type).
	 */
	template<typename Type> struct Surf2D : public Surf2D<const Type>
	{
		//	Default constructor.
		NS_CUDA_CALLABLE Surf2D() {}

		//	Constructor with nullptr.
		NS_CUDA_CALLABLE Surf2D(std::nullptr_t) : Surf2D<const Type>(nullptr) {}

		//	Constructor with cudaSurfaceObject_t.
		NS_CUDA_CALLABLE explicit Surf2D(cudaSurfaceObject_t hSurface, uint32_t width, uint32_t height) : Surf2D<const Type>(hSurface, width, height) {}

		//	Write method for CUDA surface object.
	#ifndef __CUDACC__
		__device__ void write(Type value, int x, int y, int boundaryMode = 2) const;
	#else
		__device__ void write(Type value, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) const
		{
			surf2Dwrite<InternalValueType<Type>>(reinterpret_cast<InternalValueType<Type>&>(value), Surface::m_hSurface, sizeof(Type) * x, y, boundaryMode);
		}
	#endif
	};

	/*****************************************************************************
	**************************    Surf3D<const Type>    **************************
	*****************************************************************************/

	/**
	 *	@brief		Represents a 3D CUDA surface object for device access.
	 *	@tparam		Type - The data type stored in the surface.
	 *	@details	This struct provides an interface for 3D CUDA surface objects,
	 *				enabling device-side read and write operations (read-only for const Type).
	 */
	template<typename Type> struct Surf3D<const Type> : public Surface
	{
		//	Default constructor.
		NS_CUDA_CALLABLE Surf3D() {}

		//	Constructor with nullptr.
		NS_CUDA_CALLABLE Surf3D(std::nullptr_t) : Surface(nullptr), m_width(0), m_height(0), m_depth(0) {}

		//	Constructor with cudaSurfaceObject_t.
		NS_CUDA_CALLABLE explicit Surf3D(cudaSurfaceObject_t hSurface, uint32_t width, uint32_t height, uint32_t depth) : Surface(hSurface), m_width(width), m_height(height), m_depth(depth) {}

		//	Return height of the buffer.
		NS_CUDA_CALLABLE uint32_t height() const { return m_height; }

		//	Return depth of the buffer.
		NS_CUDA_CALLABLE uint32_t depth() const { return m_depth; }

		//	Return width of the buffer.
		NS_CUDA_CALLABLE uint32_t width() const { return m_width; }

		//	Read method for CUDA surface object.
	#ifndef __CUDACC__
		__device__ Type read(int x, int y, int z, int boundaryMode = 2) const;
	#else
		__device__ Type read(int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) const
		{
			InternalValueType<Type> value;

			surf3Dread<decltype(value)>(&value, Surface::m_hSurface, sizeof(Type) * x, y, z, boundaryMode);

			return reinterpret_cast<Type&>(value);
		}
	#endif

	protected:

		uint32_t		m_width;
		uint32_t		m_height;
		uint32_t		m_depth;
	};

	/*****************************************************************************
	*****************************    Surf3D<Type>    *****************************
	*****************************************************************************/

	/**
	 *	@brief		Represents a 3D CUDA surface object for device access.
	 *	@tparam		Type - The data type stored in the surface.
	 *	@details	This struct provides an interface for 3D CUDA surface objects,
	 *				enabling device-side read and write operations (read-only for const Type).
	 */
	template<typename Type> struct Surf3D : public Surf3D<const Type>
	{
		//	Default constructor.
		NS_CUDA_CALLABLE Surf3D() {}

		//	Constructor with nullptr.
		NS_CUDA_CALLABLE Surf3D(std::nullptr_t) : Surf3D<const Type>(nullptr) {}

		//	Constructor with cudaSurfaceObject_t.
		NS_CUDA_CALLABLE explicit Surf3D(cudaSurfaceObject_t hSurface, uint32_t width, uint32_t height, uint32_t depth) : Surf3D<const Type>(hSurface, width, height, depth) {}

		//	Write method for CUDA surface object.
	#ifndef __CUDACC__
		__device__ void write(Type value, int x, int y, int z, int boundaryMode = 2) const;
	#else
		__device__ void write(Type value, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) const
		{
			surf3Dwrite<InternalValueType<Type>>(reinterpret_cast<InternalValueType<Type>&>(value), Surface::m_hSurface, sizeof(Type) * x, y, z, boundaryMode);
		}
	#endif
	};

	/*****************************************************************************
	**********************    Surf1DLayered<const Type>    ***********************
	*****************************************************************************/

	/**
	 *	@brief		Represents a 1D layered CUDA surface object for device access.
	 *	@tparam		Type - The data type stored in the surface.
	 *	@details	This struct provides an interface for 1D layered CUDA surface objects,
	 *				enabling device-side read and write operations (read-only for const Type).
	 */
	template<typename Type> struct Surf1DLayered<const Type> : public Surface
	{
		//	Default constructor.
		NS_CUDA_CALLABLE Surf1DLayered() {}

		//	Constructor with nullptr.
		NS_CUDA_CALLABLE Surf1DLayered(std::nullptr_t) : Surface(nullptr), m_width(0), m_numLayers(0) {}

		//	Constructor with cudaSurfaceObject_t.
		NS_CUDA_CALLABLE explicit Surf1DLayered(cudaSurfaceObject_t hSurface, uint32_t width, uint32_t numLayers) : Surface(hSurface), m_width(width), m_numLayers(numLayers) {}

		//	Return the number of layers.
		NS_CUDA_CALLABLE uint32_t numLayers() const { return m_numLayers; }

		//	Return width of the buffer.
		NS_CUDA_CALLABLE uint32_t width() const { return m_width; }

		//	Read method for CUDA surface object.
	#ifndef __CUDACC__
		__device__ Type read(int x, int layer, int boundaryMode = 2) const;
	#else
		__device__ Type read(int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) const
		{
			InternalValueType<Type> value;

			surf1DLayeredread<decltype(value)>(&value, Surface::m_hSurface, sizeof(Type) * x, layer, boundaryMode);

			return reinterpret_cast<Type&>(value);
		}
	#endif

	protected:

		uint32_t		m_width;
		uint32_t		m_numLayers;
	};

	/*****************************************************************************
	*************************    Surf1DLayered<Type>    **************************
	*****************************************************************************/

	/**
	 *	@brief		Represents a 1D layered CUDA surface object for device access.
	 *	@tparam		Type - The data type stored in the surface.
	 *	@details	This struct provides an interface for 1D layered CUDA surface objects,
	 *				enabling device-side read and write operations (read-only for const Type).
	 */
	template<typename Type> struct Surf1DLayered : public Surf1DLayered<const Type>
	{
		//	Default constructor.
		NS_CUDA_CALLABLE Surf1DLayered() {}

		//	Constructor with nullptr.
		NS_CUDA_CALLABLE Surf1DLayered(std::nullptr_t) : Surf1DLayered<const Type>(nullptr) {}

		//	Constructor with cudaSurfaceObject_t.
		NS_CUDA_CALLABLE explicit Surf1DLayered(cudaSurfaceObject_t hSurface, uint32_t width, uint32_t numLayers) : Surf1DLayered<const Type>(hSurface, width, numLayers) {}

		//	Write method for CUDA surface object.
	#ifndef __CUDACC__
		__device__ void write(Type value, int x, int layer, int boundaryMode = 2) const;
	#else
		__device__ void write(Type value, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) const
		{
			surf1DLayeredwrite<InternalValueType<Type>>(reinterpret_cast<InternalValueType<Type>&>(value), Surface::m_hSurface, sizeof(Type) * x, layer, boundaryMode);
		}
	#endif
	};

	/*****************************************************************************
	**********************    Surf2DLayered<const Type>    ***********************
	*****************************************************************************/

	/**
	 *	@brief		Represents a 2D layered CUDA surface object for device access.
	 *	@tparam		Type - The data type stored in the surface.
	 *	@details	This struct provides an interface for 2D layered CUDA surface objects,
	 *				enabling device-side read and write operations (read-only for const Type).
	 */
	template<typename Type> struct Surf2DLayered<const Type> : public Surface
	{
		//	Default constructor.
		NS_CUDA_CALLABLE Surf2DLayered() {}

		//	Constructor with nullptr.
		NS_CUDA_CALLABLE Surf2DLayered(std::nullptr_t) : Surface(nullptr), m_width(0), m_height(0), m_numLayers(0) {}

		//	Constructor with cudaSurfaceObject_t.
		NS_CUDA_CALLABLE explicit Surf2DLayered(cudaSurfaceObject_t hSurface, uint32_t width, uint32_t height, uint32_t numLayers) : Surface(hSurface), m_width(width), m_height(height), m_numLayers(numLayers) {}

		//	Return the number of layers.
		NS_CUDA_CALLABLE uint32_t numLayers() const { return m_numLayers; }

		//	Return height of the buffer.
		NS_CUDA_CALLABLE uint32_t height() const { return m_height; }

		//	Return width of the buffer.
		NS_CUDA_CALLABLE uint32_t width() const { return m_width; }

		//	Read method for CUDA surface object.
	#ifndef __CUDACC__
		__device__ Type read(int x, int y, int layer, int boundaryMode = 2) const;
	#else
		__device__ Type read(int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) const
		{
			InternalValueType<Type> value;

			surf2DLayeredread<decltype(value)>(&value, Surface::m_hSurface, sizeof(Type) * x, y, layer, boundaryMode);

			return reinterpret_cast<Type&>(value);
		}
	#endif

	protected:

		uint32_t		m_width;
		uint32_t		m_height;
		uint32_t		m_numLayers;
	};

	/*****************************************************************************
	*************************    Surf2DLayered<Type>    **************************
	*****************************************************************************/

	/**
	 *	@brief		Represents a 2D layered CUDA surface object for device access.
	 *	@tparam		Type - The data type stored in the surface.
	 *	@details	This struct provides an interface for 2D layered CUDA surface objects,
	 *				enabling device-side read and write operations (read-only for const Type).
	 */
	template<typename Type> struct Surf2DLayered : public Surf2DLayered<const Type>
	{
		//	Default constructor.
		NS_CUDA_CALLABLE Surf2DLayered() {}

		//	Constructor with nullptr.
		NS_CUDA_CALLABLE Surf2DLayered(std::nullptr_t) : Surf2DLayered<const Type>(nullptr) {}

		//	Constructor with cudaSurfaceObject_t.
		NS_CUDA_CALLABLE explicit Surf2DLayered(cudaSurfaceObject_t hSurface, uint32_t width, uint32_t height, uint32_t numLayers) : Surf2DLayered<const Type>(hSurface, width, height, numLayers) {}

		//	Write method for CUDA surface object.
	#ifndef __CUDACC__
		__device__ void write(Type value, int x, int y, int layer, int boundaryMode = 2) const;
	#else
		__device__ void write(Type value, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) const
		{
			surf2DLayeredwrite<InternalValueType<Type>>(reinterpret_cast<InternalValueType<Type>&>(value), Surface::m_hSurface, sizeof(Type) * x, y, layer, boundaryMode);
		}
	#endif
	};

	/*****************************************************************************
	*************************    SurfCube<const Type>    *************************
	*****************************************************************************/
	
	/**
	 *	@brief		Represents a cube-type CUDA surface object for device access.
	 *	@tparam		Type - The data type stored in the surface.
	 *	@details	This struct provides an interface for cube-type CUDA surface objects,
	 *				enabling device-side read and write operations (read-only for const Type).
	 */
	template<typename Type> struct SurfCube<const Type> : public Surface
	{
		//	Default constructor.
		NS_CUDA_CALLABLE SurfCube() {}

		//	Constructor with nullptr.
		NS_CUDA_CALLABLE SurfCube(std::nullptr_t) : Surface(nullptr), m_width(0) {}

		//	Constructor with cudaSurfaceObject_t.
		NS_CUDA_CALLABLE explicit SurfCube(cudaSurfaceObject_t hSurface, uint32_t width) : Surface(hSurface), m_width(width) {}

		//	Return width of the buffer.
		NS_CUDA_CALLABLE uint32_t width() const { return m_width; }

		//	Read method for CUDA surface object.
	#ifndef __CUDACC__
		__device__ Type read(int x, int y, int face, int boundaryMode = 2) const;
	#else
		__device__ Type read(int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) const
		{
			InternalValueType<Type> value;

			surfCubemapread<decltype(value)>(&value, Surface::m_hSurface, sizeof(Type) * x, y, face, boundaryMode);

			return reinterpret_cast<Type&>(value);
		}
	#endif

	protected:

		uint32_t		m_width;
	};

	/*****************************************************************************
	****************************    SurfCube<Type>    ****************************
	*****************************************************************************/

	/**
	 *	@brief		Represents a cube-type CUDA surface object for device access.
	 *	@tparam		Type - The data type stored in the surface.
	 *	@details	This struct provides an interface for cube-type CUDA surface objects,
	 *				enabling device-side read and write operations (read-only for const Type).
	 */
	template<typename Type> struct SurfCube : public SurfCube<const Type>
	{
		//	Default constructor.
		NS_CUDA_CALLABLE SurfCube() {}

		//	Constructor with nullptr.
		NS_CUDA_CALLABLE SurfCube(std::nullptr_t) : SurfCube<const Type>(nullptr) {}

		//	Constructor with cudaSurfaceObject_t.
		NS_CUDA_CALLABLE explicit SurfCube(cudaSurfaceObject_t hSurface, uint32_t width) : SurfCube<const Type>(hSurface, width) {}

		//	Write method for CUDA surface object.
	#ifndef __CUDACC__
		__device__ void write(Type value, int x, int y, int face, int boundaryMode = 2) const;
	#else
		__device__ void write(Type value, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) const
		{
			surfCubemapwrite<InternalValueType<Type>>(reinterpret_cast<InternalValueType<Type>&>(value), Surface::m_hSurface, sizeof(Type) * x, y, face, boundaryMode);
		}
	#endif
	};

	/*****************************************************************************
	*********************    SurfCubeLayered<const Type>    **********************
	*****************************************************************************/

	/**
	 *	@brief		Represents a cube-type layered CUDA surface object for device access.
	 *	@tparam		Type - The data type stored in the surface.
	 *	@details	This struct provides an interface for cube-type layered CUDA surface objects,
	 *				enabling device-side read and write operations (read-only for const Type).
	 */
	template<typename Type> struct SurfCubeLayered<const Type> : public Surface
	{
		//	Default constructor.
		NS_CUDA_CALLABLE SurfCubeLayered() {}

		//	Constructor with nullptr.
		NS_CUDA_CALLABLE SurfCubeLayered(std::nullptr_t) : Surface(nullptr), m_width(0), m_numLayers(0) {}

		//	Constructor with cudaSurfaceObject_t.
		NS_CUDA_CALLABLE explicit SurfCubeLayered(cudaSurfaceObject_t hSurface, uint32_t width, uint32_t numLayers) : Surface(hSurface), m_width(width), m_numLayers(numLayers) {}

		//	Return the number of layers.
		NS_CUDA_CALLABLE uint32_t numLayers() const { return m_numLayers; }

		//	Return width of the buffer.
		NS_CUDA_CALLABLE uint32_t width() const { return m_width; }

		//	Read method for CUDA surface object.
	#ifndef __CUDACC__
		__device__ Type read(int x, int y, int face, int layer, int boundaryMode = 2) const;
	#else
		__device__ Type read(int x, int y, int face, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) const
		{
			InternalValueType<Type> value;

			surfCubemapLayeredread<decltype(value)>(&value, Surface::m_hSurface, sizeof(Type) * x, y, 6 * layer + face, boundaryMode);

			return reinterpret_cast<Type&>(value);
		}
	#endif

	protected:

		uint32_t		m_width;
		uint32_t		m_numLayers;
	};

	/*****************************************************************************
	************************    SurfCubeLayered<Type>    *************************
	*****************************************************************************/

	/**
	 *	@brief		Represents a cube-type layered CUDA surface object for device access.
	 *	@tparam		Type - The data type stored in the surface.
	 *	@details	This struct provides an interface for cube-type layered CUDA surface objects,
	 *				enabling device-side read and write operations (read-only for const Type).
	 */
	template<typename Type> struct SurfCubeLayered : public SurfCubeLayered<const Type>
	{
		//	Default constructor.
		NS_CUDA_CALLABLE SurfCubeLayered() {}

		//	Constructor with nullptr.
		NS_CUDA_CALLABLE SurfCubeLayered(std::nullptr_t) : SurfCubeLayered<const Type>(nullptr) {}

		//	Constructor with cudaSurfaceObject_t.
		NS_CUDA_CALLABLE explicit SurfCubeLayered(cudaSurfaceObject_t hSurface, uint32_t width, uint32_t numLayers) : SurfCubeLayered<const Type>(hSurface, width, numLayers) {}

		//	Write method for CUDA surface object.
	#ifndef __CUDACC__
		__device__ void write(Type value, int x, int y, int face, int layer, int boundaryMode = 2) const;
	#else
		__device__ void write(Type value, int x, int y, int face, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) const
		{
			surfCubemapLayeredwrite<InternalValueType<Type>>(reinterpret_cast<InternalValueType<Type>&>(value), Surface::m_hSurface, sizeof(Type) * x, y, 6 * layer + face, boundaryMode);
		}
	#endif
	};
}