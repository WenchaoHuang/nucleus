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

#include "image_1d.h"
#include "image_2d.h"
#include "image_3d.h"
#include "image_cube.h"
#include "device_surface.h"

namespace NS_NAMESPACE
{
	/*****************************************************************************
	*******************************    Surface    ********************************
	*****************************************************************************/

	//!	@brief		Base class to manage CUDA surface resources.
	class Surface
	{
		NS_NONCOPYABLE(Surface)

	public:

		//!	@brief		Default constructor
		NS_API Surface();

		//!	@brief		Destructor
		NS_API ~Surface();

	public:

		//!	@brief		Unbinds the current surface resource.
		NS_API void unbind();

		//!	@brief		Checks if the surface is empty.
		bool empty() const { return m_hSurface == 0; }

		//!	@brief		Return CUDA type of this object.
		cudaSurfaceObject_t handle() const { return m_hSurface; }

	protected:

		/**
		 *	@brief		Binds a texture memory as the surface resource.
		 *	@param[in]	image - Shared pointer to the memory object.
		 *	@throw		cudaError_t - In case of failure.
		 */
		NS_API void bindImage(std::shared_ptr<Image> image);

	protected:

		cudaSurfaceObject_t			m_hSurface;

		std::shared_ptr<Image>		m_image;
	};

	/*****************************************************************************
	******************************    Surface1D    *******************************
	*****************************************************************************/

	//!	@brief		Representing a CUDA 1D surface object.
	template<typename Type> class Surface1D : public Surface
	{

	public:

		//!	@brief		Default constructor.
		Surface1D() {}

		//!	@brief		Constructs a 1D surface and binds a texture memory object.
		explicit Surface1D(const Image1D<Type> & image) { this->bind(image); }

	public:

		//!	@brief		Returns the binded texture memory.
		Image1D<Type> image() const { return *std::static_pointer_cast<Image1D<Type>>(m_image); }

		//!	@brief		Binds a 1D texture memory object to the surface.
		void bind(const Image1D<Type> & image) { this->bindImage(std::make_shared<Image1D<Type>>(image)); }

		//!	@brief		Converts to a device surface object for kernal access.
		operator dev::Surf1D<Type>() { return m_image ? dev::Surf1D<Type>(m_hSurface, m_image->width()) : nullptr; }

		//!	@brief		Converts to a device surface object for kernal access (constant version).
		operator dev::Surf1D<const Type>() const { return *this; }

		//!	@brief		Returns read-only device accessor explicitly.
		dev::Surf1D<const Type> accessor() const { return *this; }

		//!	@brief		Returns device accessor explicitly.
		dev::Surf1D<Type> accessor() { return *this; }
	};

	/*****************************************************************************
	******************************    Surface2D    *******************************
	*****************************************************************************/

	//!	@brief		Representing a CUDA 2D surface object.
	template<typename Type> class Surface2D : public Surface
	{

	public:

		//!	@brief		Default constructor.
		Surface2D() {}

		//!	@brief		Constructs a 2D surface and binds a texture memory object.
		explicit Surface2D(const Image2D<Type> & image) { this->bind(image); }

	public:

		//!	@brief		Returns the binded texture memory.
		Image2D<Type> image() const { return *std::static_pointer_cast<Image2D<Type>>(m_image); }

		//!	@brief		Binds a 2D texture memory object to the surface.
		void bind(const Image2D<Type> & image) { this->bindImage(std::make_shared<Image2D<Type>>(image)); }

		//!	@brief		Converts to a device surface object for kernal access.
		operator dev::Surf2D<Type>() { return m_image ? dev::Surf2D<Type>(m_hSurface, m_image->width(), this->image().height()) : nullptr; }

		//!	@brief		Converts to a device surface object for kernal access (constant version).
		operator dev::Surf2D<const Type>() const { return *this; }

		//!	@brief		Returns read-only device accessor explicitly.
		dev::Surf2D<const Type> accessor() const { return *this; }

		//!	@brief		Returns device accessor explicitly.
		dev::Surf2D<Type> accessor() { return *this; }
	};

	/*****************************************************************************
	******************************    Surface3D    *******************************
	*****************************************************************************/

	//!	@brief		Representing a CUDA 3D surface object.
	template<typename Type> class Surface3D : public Surface
	{

	public:

		//!	@brief		Default constructor.
		Surface3D() {}

		//!	@brief		Constructs a 3D surface and binds a texture memory object.
		explicit Surface3D(const Image3D<Type> & image) { this->bind(image); }

	public:

		//!	@brief		Returns the binded texture memory.
		Image3D<Type> image() const { return *std::static_pointer_cast<Image3D<Type>>(m_image); }

		//!	@brief		Binds a 3D texture memory object to the surface.
		void bind(const Image3D<Type> & image) { this->bindImage(std::make_shared<Image3D<Type>>(image)); }

		//!	@brief		Converts to a device surface object for kernal access.
		operator dev::Surf3D<Type>() { return m_image ? dev::Surf3D<Type>(m_hSurface, m_image->width(), this->image().height(), this->image().depth()) : nullptr; }

		//!	@brief		Converts to a device surface object for kernal access (constant version).
		operator dev::Surf3D<const Type>() const { return *this; }

		//!	@brief		Returns read-only device accessor explicitly.
		dev::Surf3D<const Type> accessor() const { return *this; }

		//!	@brief		Returns device accessor explicitly.
		dev::Surf3D<Type> accessor() { return *this; }
	};

	/*****************************************************************************
	***************************    Surface1DLayered    ***************************
	*****************************************************************************/

	//!	@brief		Representing a CUDA 1D layered surface object.
	template<typename Type> class Surface1DLayered : public Surface
	{

	public:

		//!	@brief		Default constructor.
		Surface1DLayered() {}

		//!	@brief		Constructs a 1D layered surface and binds a texture memory object.
		explicit Surface1DLayered(const Image1DLayered<Type> & image) { this->bind(image); }

	public:

		//!	@brief		Returns the binded texture memory.
		Image1DLayered<Type> image() const { return *std::static_pointer_cast<Image1DLayered<Type>>(m_image); }

		//!	@brief		Binds a 1D layered texture memory object to the surface.
		void bind(const Image1DLayered<Type> & image) { this->bindImage(std::make_shared<Image1DLayered<Type>>(image)); }

		//!	@brief		Converts to a device surface object for kernal access.
		operator dev::Surf1DLayered<Type>() { return m_image ? dev::Surf1DLayered<Type>(m_hSurface, m_image->width(), this->image().numLayers()) : nullptr; }

		//!	@brief		Converts to a device surface object for kernal access (constant version).
		operator dev::Surf1DLayered<const Type>() const { return *this; }

		//!	@brief		Returns read-only device accessor explicitly.
		dev::Surf1DLayered<const Type> accessor() const { return *this; }

		//!	@brief		Returns device accessor explicitly.
		dev::Surf1DLayered<Type> accessor() { return *this; }
	};

	/*****************************************************************************
	***************************    Surface2DLayered    ***************************
	*****************************************************************************/

	//!	@brief		Representing a CUDA 2D layered surface object.
	template<typename Type> class Surface2DLayered : public Surface
	{

	public:

		//!	@brief		Default constructor.
		Surface2DLayered() {}

		//!	@brief		Constructs a 2D layered surface and binds a texture memory object.
		explicit Surface2DLayered(const Image2DLayered<Type> & image) { this->bind(image); }

	public:

		//!	@brief		Returns the binded texture memory.
		Image2DLayered<Type> image() const { return *std::static_pointer_cast<Image2DLayered<Type>>(m_image); }

		//!	@brief		Binds a 2D layered texture memory object to the surface.
		void bind(const Image2DLayered<Type> & image) { this->bindImage(std::make_shared<Image2DLayered<Type>>(image)); }

		//!	@brief		Converts to a device surface object for kernal access.
		operator dev::Surf2DLayered<Type>() { return m_image ? dev::Surf2DLayered<Type>(m_hSurface, m_image->width(), this->image().height(), this->image().numLayers()) : nullptr; }

		//!	@brief		Converts to a device surface object for kernal access (constant version).
		operator dev::Surf2DLayered<const Type>() const { return *this; }

		//!	@brief		Returns read-only device accessor explicitly.
		dev::Surf2DLayered<const Type> accessor() const { return *this; }

		//!	@brief		Returns device accessor explicitly.
		dev::Surf2DLayered<Type> accessor() { return *this; }
	};

	/*****************************************************************************
	*****************************    SurfaceCube    ******************************
	*****************************************************************************/

	//!	@brief		Representing a CUDA cubemap surface object.
	template<typename Type> class SurfaceCube : public Surface
	{

	public:

		//!	@brief		Default constructor.
		SurfaceCube() {}

		//!	@brief		Constructs a cubemap surface and binds a texture memory object.
		explicit SurfaceCube(const ImageCube<Type> & image) { this->bind(image); }

	public:

		//!	@brief		Returns shared pointer to the binded texture memory.
		ImageCube<Type> image() const { return *std::static_pointer_cast<ImageCube<Type>>(m_image); }

		//!	@brief		Binds a cubemap texture memory object to the surface.
		void bind(const ImageCube<Type> & image) { this->bindImage(std::make_shared<ImageCube<Type>>(image)); }

		//!	@brief		Converts to a device surface object for kernal access.
		operator dev::SurfCube<Type>() { return m_image ? dev::SurfCube<Type>(m_hSurface, m_image->width()) : nullptr; }

		//!	@brief		Converts to a device surface object for kernal access (constant version).
		operator dev::SurfCube<const Type>() const { return *this; }

		//!	@brief		Returns read-only device accessor explicitly.
		dev::SurfCube<const Type> accessor() const { return *this; }

		//!	@brief		Returns device accessor explicitly.
		dev::SurfCube<Type> accessor() { return *this; }
	};

	/*****************************************************************************
	**************************    SurfaceCubeLayered    **************************
	*****************************************************************************/

	//!	@brief		Representing a CUDA layered cubemap surface object.
	template<typename Type> class SurfaceCubeLayered : public Surface
	{

	public:

		//!	@brief		Default constructor.
		SurfaceCubeLayered() {}

		//!	@brief		Constructs a layered cubemap surface and binds a texture memory object.
		explicit SurfaceCubeLayered(const ImageCubeLayered<Type> & image) { this->bind(image); }

	public:

		//!	@brief		Returns shared pointer to the binded texture memory.
		ImageCubeLayered<Type> image() const { return *std::static_pointer_cast<ImageCubeLayered<Type>>(m_image); }

		//!	@brief		Binds a layered cubemap texture memory object to the surface.
		void bind(const ImageCubeLayered<Type> & image) { this->bindImage(std::make_shared<ImageCubeLayered<Type>>(image)); }

		//!	@brief		Converts to a device surface object for kernal access.
		operator dev::SurfCubeLayered<Type>() { return m_image ? dev::SurfCubeLayered<Type>(m_hSurface, m_image->width(), this->image().numLayers()) : nullptr; }

		//!	@brief		Converts to a device surface object for kernal access (constant version).
		operator dev::SurfCubeLayered<const Type>() const { return *this; }

		//!	@brief		Returns read-only device accessor explicitly.
		dev::SurfCubeLayered<const Type> accessor() const { return *this; }

		//!	@brief		Returns device accessor explicitly.
		dev::SurfCubeLayered<Type> accessor() { return *this; }
	};
}
