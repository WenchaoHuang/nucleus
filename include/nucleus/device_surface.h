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
		//!	@brief		Default constructor.
		NS_CUDA_CALLABLE Surface() {}

		//!	@brief		Constructor with nullptr.
		NS_CUDA_CALLABLE Surface(std::nullptr_t) : surface_(0) {}

		//!	@brief		Constructor with cudaSurfaceObject_t.
		NS_CUDA_CALLABLE explicit Surface(cudaSurfaceObject_t surface) : surface_(surface) {}

		//!	@brief		Return CUDA type of this object for compatibility.
		NS_CUDA_CALLABLE cudaSurfaceObject_t handle() const { return surface_; }

		//!	@brief		Bool conversion operator.
		NS_CUDA_CALLABLE operator bool() const { return surface_ != 0; }

		//!	@brief		Check if the surface is empty.
		NS_CUDA_CALLABLE bool empty() const { return surface_ == 0; }

	protected:

		cudaSurfaceObject_t		surface_;
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
		//!	@brief		Default constructor.
		NS_CUDA_CALLABLE Surf1D() {}

		//!	@brief		Constructor with nullptr.
		NS_CUDA_CALLABLE Surf1D(std::nullptr_t) : Surface(nullptr), width_(0) {}

		//!	@brief		Constructor with cudaSurfaceObject_t.
		NS_CUDA_CALLABLE explicit Surf1D(cudaSurfaceObject_t surface, uint32_t width) : Surface(surface), width_(width) {}

		//!	@brief		Return width of the buffer.
		NS_CUDA_CALLABLE uint32_t width() const { return width_; }

		//!	@brief		Read method for CUDA surface object.
	#ifndef __CUDACC__
		__device__ Type read(int x, int boundary_mode = 2) const;
	#else
		__device__ Type read(int x, cudaSurfaceBoundaryMode boundary_mode = cudaBoundaryModeTrap) const
		{
			CudaTexelType_t<Type> value;

			surf1Dread<decltype(value)>(&value, Surface::surface_, sizeof(Type) * x, boundary_mode);

			return reinterpret_cast<Type&>(value);
		}
	#endif

	protected:

		uint32_t		width_;
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
		//!	@brief		Default constructor.
		NS_CUDA_CALLABLE Surf1D() {}

		//!	@brief		Constructor with nullptr.
		NS_CUDA_CALLABLE Surf1D(std::nullptr_t) : Surf1D<const Type>(nullptr) {}

		//!	@brief		Constructor with cudaSurfaceObject_t.
		NS_CUDA_CALLABLE explicit Surf1D(cudaSurfaceObject_t surface, uint32_t width) : Surf1D<const Type>(surface, width) {}
		
		//!	@brief		Write method for CUDA surface object.
	#ifndef __CUDACC__
		__device__ void write(Type value, int x, int boundary_mode = 2) const;
	#else
		__device__ void write(Type value, int x, cudaSurfaceBoundaryMode boundary_mode = cudaBoundaryModeTrap) const
		{
			surf1Dwrite<CudaTexelType_t<Type>>(reinterpret_cast<CudaTexelType_t<Type>&>(value), Surface::surface_, sizeof(Type) * x, boundary_mode);
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
		//!	@brief		Default constructor.
		NS_CUDA_CALLABLE Surf2D() {}

		//!	@brief		Constructor with nullptr.
		NS_CUDA_CALLABLE Surf2D(std::nullptr_t) : Surface(nullptr), width_(0), height_(0) {}

		//!	@brief		Constructor with cudaSurfaceObject_t.
		NS_CUDA_CALLABLE explicit Surf2D(cudaSurfaceObject_t surface, uint32_t width, uint32_t height) : Surface(surface), width_(width), height_(height) {}

		//!	@brief		Return height of the buffer.
		NS_CUDA_CALLABLE uint32_t height() const { return height_; }

		//!	@brief		Return width of the buffer.
		NS_CUDA_CALLABLE uint32_t width() const { return width_; }

		//!	@brief		Read method for CUDA surface object.
	#ifndef __CUDACC__
		__device__ Type read(int x, int y, int boundary_mode = 2) const;
	#else
		__device__ Type read(int x, int y, cudaSurfaceBoundaryMode boundary_mode = cudaBoundaryModeTrap) const
		{
			CudaTexelType_t<Type> value;

			surf2Dread<decltype(value)>(&value, Surface::surface_, sizeof(Type) * x, y, boundary_mode);

			return reinterpret_cast<Type&>(value);
		}
	#endif

	protected:

		uint32_t		width_;
		uint32_t		height_;
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
		//!	@brief		Default constructor.
		NS_CUDA_CALLABLE Surf2D() {}

		//!	@brief		Constructor with nullptr.
		NS_CUDA_CALLABLE Surf2D(std::nullptr_t) : Surf2D<const Type>(nullptr) {}

		//!	@brief		Constructor with cudaSurfaceObject_t.
		NS_CUDA_CALLABLE explicit Surf2D(cudaSurfaceObject_t surface, uint32_t width, uint32_t height) : Surf2D<const Type>(surface, width, height) {}

		//!	@brief		Write method for CUDA surface object.
	#ifndef __CUDACC__
		__device__ void write(Type value, int x, int y, int boundary_mode = 2) const;
	#else
		__device__ void write(Type value, int x, int y, cudaSurfaceBoundaryMode boundary_mode = cudaBoundaryModeTrap) const
		{
			surf2Dwrite<CudaTexelType_t<Type>>(reinterpret_cast<CudaTexelType_t<Type>&>(value), Surface::surface_, sizeof(Type) * x, y, boundary_mode);
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
		//!	@brief		Default constructor.
		NS_CUDA_CALLABLE Surf3D() {}

		//!	@brief		Constructor with nullptr.
		NS_CUDA_CALLABLE Surf3D(std::nullptr_t) : Surface(nullptr), width_(0), height_(0), depth_(0) {}

		//!	@brief		Constructor with cudaSurfaceObject_t.
		NS_CUDA_CALLABLE explicit Surf3D(cudaSurfaceObject_t surface, uint32_t width, uint32_t height, uint32_t depth) : Surface(surface), width_(width), height_(height), depth_(depth) {}

		//!	@brief		Return height of the buffer.
		NS_CUDA_CALLABLE uint32_t height() const { return height_; }

		//!	@brief		Return depth of the buffer.
		NS_CUDA_CALLABLE uint32_t depth() const { return depth_; }

		//!	@brief		Return width of the buffer.
		NS_CUDA_CALLABLE uint32_t width() const { return width_; }

		//!	@brief		Read method for CUDA surface object.
	#ifndef __CUDACC__
		__device__ Type read(int x, int y, int z, int boundary_mode = 2) const;
	#else
		__device__ Type read(int x, int y, int z, cudaSurfaceBoundaryMode boundary_mode = cudaBoundaryModeTrap) const
		{
			CudaTexelType_t<Type> value;

			surf3Dread<decltype(value)>(&value, Surface::surface_, sizeof(Type) * x, y, z, boundary_mode);

			return reinterpret_cast<Type&>(value);
		}
	#endif

	protected:

		uint32_t		width_;
		uint32_t		height_;
		uint32_t		depth_;
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
		//!	@brief		Default constructor.
		NS_CUDA_CALLABLE Surf3D() {}

		//!	@brief		Constructor with nullptr.
		NS_CUDA_CALLABLE Surf3D(std::nullptr_t) : Surf3D<const Type>(nullptr) {}

		//!	@brief		Constructor with cudaSurfaceObject_t.
		NS_CUDA_CALLABLE explicit Surf3D(cudaSurfaceObject_t surface, uint32_t width, uint32_t height, uint32_t depth) : Surf3D<const Type>(surface, width, height, depth) {}

		//!	@brief		Write method for CUDA surface object.
	#ifndef __CUDACC__
		__device__ void write(Type value, int x, int y, int z, int boundary_mode = 2) const;
	#else
		__device__ void write(Type value, int x, int y, int z, cudaSurfaceBoundaryMode boundary_mode = cudaBoundaryModeTrap) const
		{
			surf3Dwrite<CudaTexelType_t<Type>>(reinterpret_cast<CudaTexelType_t<Type>&>(value), Surface::surface_, sizeof(Type) * x, y, z, boundary_mode);
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
		//!	@brief		Default constructor.
		NS_CUDA_CALLABLE Surf1DLayered() {}

		//!	@brief		Constructor with nullptr.
		NS_CUDA_CALLABLE Surf1DLayered(std::nullptr_t) : Surface(nullptr), width_(0), num_layers_(0) {}

		//!	@brief		Constructor with cudaSurfaceObject_t.
		NS_CUDA_CALLABLE explicit Surf1DLayered(cudaSurfaceObject_t surface, uint32_t width, uint32_t num_layers) : Surface(surface), width_(width), num_layers_(num_layers) {}

		//!	@brief		Return the number of layers.
		NS_CUDA_CALLABLE uint32_t num_layers() const { return num_layers_; }

		//!	@brief		Return width of the buffer.
		NS_CUDA_CALLABLE uint32_t width() const { return width_; }

		//!	@brief		Read method for CUDA surface object.
	#ifndef __CUDACC__
		__device__ Type read(int x, int layer, int boundary_mode = 2) const;
	#else
		__device__ Type read(int x, int layer, cudaSurfaceBoundaryMode boundary_mode = cudaBoundaryModeTrap) const
		{
			CudaTexelType_t<Type> value;

			surf1DLayeredread<decltype(value)>(&value, Surface::surface_, sizeof(Type) * x, layer, boundary_mode);

			return reinterpret_cast<Type&>(value);
		}
	#endif

	protected:

		uint32_t		width_;
		uint32_t		num_layers_;
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
		//!	@brief		Default constructor.
		NS_CUDA_CALLABLE Surf1DLayered() {}

		//!	@brief		Constructor with nullptr.
		NS_CUDA_CALLABLE Surf1DLayered(std::nullptr_t) : Surf1DLayered<const Type>(nullptr) {}

		//!	@brief		Constructor with cudaSurfaceObject_t.
		NS_CUDA_CALLABLE explicit Surf1DLayered(cudaSurfaceObject_t surface, uint32_t width, uint32_t num_layers) : Surf1DLayered<const Type>(surface, width, num_layers) {}

		//!	@brief		Write method for CUDA surface object.
	#ifndef __CUDACC__
		__device__ void write(Type value, int x, int layer, int boundary_mode = 2) const;
	#else
		__device__ void write(Type value, int x, int layer, cudaSurfaceBoundaryMode boundary_mode = cudaBoundaryModeTrap) const
		{
			surf1DLayeredwrite<CudaTexelType_t<Type>>(reinterpret_cast<CudaTexelType_t<Type>&>(value), Surface::surface_, sizeof(Type) * x, layer, boundary_mode);
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
		//!	@brief		Default constructor.
		NS_CUDA_CALLABLE Surf2DLayered() {}

		//!	@brief		Constructor with nullptr.
		NS_CUDA_CALLABLE Surf2DLayered(std::nullptr_t) : Surface(nullptr), width_(0), height_(0), num_layers_(0) {}

		//!	@brief		Constructor with cudaSurfaceObject_t.
		NS_CUDA_CALLABLE explicit Surf2DLayered(cudaSurfaceObject_t surface, uint32_t width, uint32_t height, uint32_t num_layers) : Surface(surface), width_(width), height_(height), num_layers_(num_layers) {}

		//!	@brief		Return the number of layers.
		NS_CUDA_CALLABLE uint32_t num_layers() const { return num_layers_; }

		//!	@brief		Return height of the buffer.
		NS_CUDA_CALLABLE uint32_t height() const { return height_; }

		//!	@brief		Return width of the buffer.
		NS_CUDA_CALLABLE uint32_t width() const { return width_; }

		//!	@brief		Read method for CUDA surface object.
	#ifndef __CUDACC__
		__device__ Type read(int x, int y, int layer, int boundary_mode = 2) const;
	#else
		__device__ Type read(int x, int y, int layer, cudaSurfaceBoundaryMode boundary_mode = cudaBoundaryModeTrap) const
		{
			CudaTexelType_t<Type> value;

			surf2DLayeredread<decltype(value)>(&value, Surface::surface_, sizeof(Type) * x, y, layer, boundary_mode);

			return reinterpret_cast<Type&>(value);
		}
	#endif

	protected:

		uint32_t		width_;
		uint32_t		height_;
		uint32_t		num_layers_;
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
		//!	@brief		Default constructor.
		NS_CUDA_CALLABLE Surf2DLayered() {}

		//!	@brief		Constructor with nullptr.
		NS_CUDA_CALLABLE Surf2DLayered(std::nullptr_t) : Surf2DLayered<const Type>(nullptr) {}

		//!	@brief		Constructor with cudaSurfaceObject_t.
		NS_CUDA_CALLABLE explicit Surf2DLayered(cudaSurfaceObject_t surface, uint32_t width, uint32_t height, uint32_t num_layers) : Surf2DLayered<const Type>(surface, width, height, num_layers) {}

		//!	@brief		Write method for CUDA surface object.
	#ifndef __CUDACC__
		__device__ void write(Type value, int x, int y, int layer, int boundary_mode = 2) const;
	#else
		__device__ void write(Type value, int x, int y, int layer, cudaSurfaceBoundaryMode boundary_mode = cudaBoundaryModeTrap) const
		{
			surf2DLayeredwrite<CudaTexelType_t<Type>>(reinterpret_cast<CudaTexelType_t<Type>&>(value), Surface::surface_, sizeof(Type) * x, y, layer, boundary_mode);
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
		//!	@brief		Default constructor.
		NS_CUDA_CALLABLE SurfCube() {}

		//!	@brief		Constructor with nullptr.
		NS_CUDA_CALLABLE SurfCube(std::nullptr_t) : Surface(nullptr), width_(0) {}

		//!	@brief		Constructor with cudaSurfaceObject_t.
		NS_CUDA_CALLABLE explicit SurfCube(cudaSurfaceObject_t surface, uint32_t width) : Surface(surface), width_(width) {}

		//!	@brief		Return width of the buffer.
		NS_CUDA_CALLABLE uint32_t width() const { return width_; }

		//!	@brief		Read method for CUDA surface object.
	#ifndef __CUDACC__
		__device__ Type read(int x, int y, int face, int boundary_mode = 2) const;
	#else
		__device__ Type read(int x, int y, int face, cudaSurfaceBoundaryMode boundary_mode = cudaBoundaryModeTrap) const
		{
			CudaTexelType_t<Type> value;

			surfCubemapread<decltype(value)>(&value, Surface::surface_, sizeof(Type) * x, y, face, boundary_mode);

			return reinterpret_cast<Type&>(value);
		}
	#endif

	protected:

		uint32_t		width_;
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
		//!	@brief		Default constructor.
		NS_CUDA_CALLABLE SurfCube() {}

		//!	@brief		Constructor with nullptr.
		NS_CUDA_CALLABLE SurfCube(std::nullptr_t) : SurfCube<const Type>(nullptr) {}

		//!	@brief		Constructor with cudaSurfaceObject_t.
		NS_CUDA_CALLABLE explicit SurfCube(cudaSurfaceObject_t surface, uint32_t width) : SurfCube<const Type>(surface, width) {}

		//!	@brief		Write method for CUDA surface object.
	#ifndef __CUDACC__
		__device__ void write(Type value, int x, int y, int face, int boundary_mode = 2) const;
	#else
		__device__ void write(Type value, int x, int y, int face, cudaSurfaceBoundaryMode boundary_mode = cudaBoundaryModeTrap) const
		{
			surfCubemapwrite<CudaTexelType_t<Type>>(reinterpret_cast<CudaTexelType_t<Type>&>(value), Surface::surface_, sizeof(Type) * x, y, face, boundary_mode);
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
		//!	@brief		Default constructor.
		NS_CUDA_CALLABLE SurfCubeLayered() {}

		//!	@brief		Constructor with nullptr.
		NS_CUDA_CALLABLE SurfCubeLayered(std::nullptr_t) : Surface(nullptr), width_(0), num_layers_(0) {}

		//!	@brief		Constructor with cudaSurfaceObject_t.
		NS_CUDA_CALLABLE explicit SurfCubeLayered(cudaSurfaceObject_t surface, uint32_t width, uint32_t num_layers) : Surface(surface), width_(width), num_layers_(num_layers) {}

		//!	@brief		Return the number of layers.
		NS_CUDA_CALLABLE uint32_t num_layers() const { return num_layers_; }

		//!	@brief		Return width of the buffer.
		NS_CUDA_CALLABLE uint32_t width() const { return width_; }

		//!	@brief		Read method for CUDA surface object.
	#ifndef __CUDACC__
		__device__ Type read(int x, int y, int face, int layer, int boundary_mode = 2) const;
	#else
		__device__ Type read(int x, int y, int face, int layer, cudaSurfaceBoundaryMode boundary_mode = cudaBoundaryModeTrap) const
		{
			CudaTexelType_t<Type> value;

			surfCubemapLayeredread<decltype(value)>(&value, Surface::surface_, sizeof(Type) * x, y, 6 * layer + face, boundary_mode);

			return reinterpret_cast<Type&>(value);
		}
	#endif

	protected:

		uint32_t		width_;
		uint32_t		num_layers_;
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
		//!	@brief		Default constructor.
		NS_CUDA_CALLABLE SurfCubeLayered() {}

		//!	@brief		Constructor with nullptr.
		NS_CUDA_CALLABLE SurfCubeLayered(std::nullptr_t) : SurfCubeLayered<const Type>(nullptr) {}

		//!	@brief		Constructor with cudaSurfaceObject_t.
		NS_CUDA_CALLABLE explicit SurfCubeLayered(cudaSurfaceObject_t surface, uint32_t width, uint32_t num_layers) : SurfCubeLayered<const Type>(surface, width, num_layers) {}

		//!	@brief		Write method for CUDA surface object.
	#ifndef __CUDACC__
		__device__ void write(Type value, int x, int y, int face, int layer, int boundary_mode = 2) const;
	#else
		__device__ void write(Type value, int x, int y, int face, int layer, cudaSurfaceBoundaryMode boundary_mode = cudaBoundaryModeTrap) const
		{
			surfCubemapLayeredwrite<CudaTexelType_t<Type>>(reinterpret_cast<CudaTexelType_t<Type>&>(value), Surface::surface_, sizeof(Type) * x, y, 6 * layer + face, boundary_mode);
		}
	#endif
	};
}
