# Nucleus: C++ Bindings for CUDA Resources

[![License](https://img.shields.io/github/license/WenchaoHuang/Nucleus)](LICENSE)
[![Build (Linux)](https://github.com/WenchaoHuang/Nucleus/actions/workflows/build-linux.yml/badge.svg)](https://github.com/WenchaoHuang/Nucleus/actions/workflows/build-linux.yml)
[![Build (Windows)](https://github.com/WenchaoHuang/Nucleus/actions/workflows/build-windows.yml/badge.svg)](https://github.com/WenchaoHuang/Nucleus/actions/workflows/build-windows.yml)
[![GitHub commit activity](https://img.shields.io/github/commit-activity/y/WenchaoHuang/Nucleus/main)](https://github.com/WenchaoHuang/Nucleus/commits/main)

> [!WARNING]
> **This project is under active development.** The API is unstable; features may be added or removed and breaking changes may occur without notice as the design is refined. It is currently not recommended for production use.

## Overview

Nucleus is an open-source C++ library that provides modern, type-safe abstractions over the CUDA Runtime API. It is designed to streamline heterogeneous computing by wrapping common CUDA resources — devices, streams, events, memory allocations, arrays, textures, surfaces, and CUDA graphs — in RAII-friendly C++ objects.

Key design goals:

- **CUDA Header Isolation** — CUDA headers are contained within Nucleus; host-side translation units that include Nucleus headers do not need to include CUDA headers directly, reducing toolchain conflicts.
- **Compile-Time Type Safety** — Typed device pointers (`dev::Ptr<T>`, `dev::Ptr2<T>`, `dev::Ptr3<T>`) and a compile-time `ptr_cast<>` utility catch type mismatches at compile time rather than at runtime.
- **Runtime Memory Checking** — An optional bounds-checking and null-pointer-detection mechanism (enabled via the `NS_CUDA_MEMORY_CHECK` compile flag) helps catch memory errors during development.
- **RAII Resource Management** — All CUDA objects (`Stream`, `Event`, `Buffer`, `Array`, `Texture`, `Surface`, `Graph`, …) follow RAII and are non-copyable, preventing resource leaks.
- **Fluent Stream API** — Stream operations return `Stream&`, enabling readable method chaining and easy synchronization.

For OptiX ray-tracing functionality, see the companion [Photon](https://github.com/WenchaoHuang/Photon) library.

## Features

- Device enumeration and properties via `ns::Runtime` and `ns::Device`
- CUDA stream management with a fluent API (`stream.launch(…)(args…)`, `stream.memcpy(…)`, `stream.sync()`)
- Typed 1-D, 2-D, and 3-D device arrays (`ns::Array<T>`, `ns::Array2D<T>`, `ns::Array3D<T>`)
- Pluggable allocator interface (`ns::Allocator`, `ns::DeviceAllocator`, `ns::HostAllocator`)
- CUDA events and timing (`ns::Event`, `ns::TimedEvent`, `ns::ScopedTimer`)
- CUDA graph support with automatic topology caching and parameter update (`ns::Graph`)
- Texture and surface objects with full dimensionality support (1-D/2-D/3-D, cubemap, layered, mipmapped)
- Kernel-launch helpers: `ns::ceil_div`, `ns::align_up`, `CUDA_for` / `NS_BOUNDS_CHECK` bounds-check macros
- Logger (`ns::Logger`) for structured diagnostic messages

## Prerequisites

| Requirement | Minimum version |
|---|---|
| CUDA Toolkit | 11.0 |
| CMake | 3.18 |
| C++ compiler | C++17 (C++20 recommended) |

Nucleus is currently developed and tested on Linux and Windows (MSVC). Other platforms are not officially supported.

## Installation

### Option 1 — Build and install as a standalone library

```bash
git clone https://github.com/WenchaoHuang/Nucleus.git
cd Nucleus
cmake -B build -DNUCLEUS_BUILD_SHARED_LIB=ON
cmake --build build
```

### Option 2 — Embed as a Git submodule

```bash
# From your project root:
git submodule add https://github.com/WenchaoHuang/Nucleus.git third_party/nucleus
```

Then in your project's `CMakeLists.txt`:

```cmake
add_subdirectory(third_party/nucleus)
target_link_libraries(my_app PRIVATE nucleus)
```

### CMake Options

| Option | Default | Description |
|---|---|---|
| `NUCLEUS_BUILD_SHARED_LIB` | `ON` | Build as a shared library; set to `OFF` for a static library |
| `NUCLEUS_BUILD_TESTS` | `OFF` | Build the test suite (`tests/`) |
| `NUCLEUS_BUILD_EXAMPLES` | `OFF` | Build the bundled examples (`examples/`) |
| `NUCLEUS_CPP_STANDARD` | `20` | C++ standard to use (`17`, `20`, `23`, or `26`) |

Example — build in static mode with tests and examples:

```bash
cmake -B build \
  -DNUCLEUS_BUILD_SHARED_LIB=OFF \
  -DNUCLEUS_BUILD_TESTS=ON \
  -DNUCLEUS_BUILD_EXAMPLES=ON \
  -DNUCLEUS_CPP_STANDARD=20
cmake --build build
```

## Quick Start

The include-all convenience header pulls in every Nucleus component:

```cpp
#include <nucleus/nucleus.h>
```

Or include individual headers as needed (see [Project Structure](#project-structure) below).

> **Note:** Headers that define CUDA device code (e.g. `launch_utils.cuh`) must be included only from `.cu` files compiled by `nvcc`.

### Vector Addition Example

```cu
// vector_add.cu  —  must be compiled with nvcc
#include <nucleus/runtime.h>
#include <nucleus/array_1d.h>
#include <nucleus/launch_utils.cuh>  // .cuh — only in .cu files

// Kernel: element-wise addition
// CUDA_for(i, count) expands to:
//   const auto i = ns::tid(); if (i >= count) return;
__global__ void add_kernel(dev::Ptr<int> out,
                           dev::Ptr<const int> x,
                           dev::Ptr<const int> y,
                           unsigned int count)
{
    CUDA_for(i, count);
    out[i] = x[i] + y[i];
}

int main()
{
    const int count = 1 << 20;

    // Obtain device, allocator, and default stream from the runtime singleton.
    auto * device    = ns::Runtime::device(0);
    auto   allocator = device->defaultAllocator();
    auto & stream    = device->defaultStream();

    // Allocate typed device arrays (RAII — freed automatically).
    ns::Array<int> A(allocator, count);
    ns::Array<int> B(allocator, count);
    ns::Array<int> C(allocator, count);

    // Initialize device memory (type-safe typed fill).
    stream.fill(A.data(), 1, A.size());
    stream.fill(B.data(), 2, B.size());

    // Launch the kernel — grid size is computed with ceil_div.
    constexpr int blockSize = 256;
    stream.launch(add_kernel, ns::ceil_div(count, blockSize), blockSize)(C, A, B, count);

    // Copy result back to host and synchronize.
    std::vector<int> host_result(count);
    stream.memcpy(host_result.data(), C.data(), C.size());
    stream.sync();

    return 0;
}
```

### Using a Custom Allocator

```cpp
#include <nucleus/allocator.h>

class MyPoolAllocator : public ns::DeviceAllocator
{
public:
    explicit MyPoolAllocator(ns::Device * device) : ns::DeviceAllocator(device) {}

protected:
    void * doAllocateMemory(size_t bytes) override { /* pool logic */ }
    void   doDeallocateMemory(void * ptr)  override { /* pool logic */ }
};

auto myAlloc = std::make_shared<MyPoolAllocator>(device);
ns::Array<float> buf(myAlloc, 1024);
```

### Creating a Custom Stream

```cpp
// Non-blocking stream with default priority
ns::Stream stream(device);

// Force synchronization after every operation (debug mode)
stream.setForceSync(true);
```

## Core Concepts

### `ns::Runtime`

A process-wide singleton that initializes the CUDA runtime and enumerates all available devices.

```cpp
std::cout << ns::Runtime::getDevices().size() << " GPU(s) found\n";
auto * device = ns::Runtime::device(0);
```

### `ns::Device`

Wraps a physical GPU. Provides access to device properties, a default allocator, and a default stream.

```cpp
device->freeMemorySize();           // bytes of free VRAM
device->properties()->name;         // device name string (cudaDeviceProp)
device->defaultAllocator();         // shared_ptr<DeviceAllocator>
device->defaultStream();            // Stream&
device->sync();                     // wait for all work on this device
```

### `ns::Stream`

All asynchronous GPU work (kernel launches, memory copies) is submitted to a stream. Operations on a stream execute in order; streams may run concurrently with each other.

```cpp
// Kernel launch — returns Stream& for chaining
stream.launch(my_kernel, gridDim, blockDim)(arg1, arg2, …);

// Typed device-to-host copy
stream.memcpy(dst_host, src_device, count);

// Wait for all pending operations
stream.sync();

// Non-blocking completion query
bool done = stream.isComplete();
```

### `ns::Array<T>` / `ns::Array2D<T>` / `ns::Array3D<T>`

RAII wrappers for typed device memory. They inherit from `dev::Ptr<T>` and can be passed directly to kernels.

```cpp
ns::Array<float>   a1(allocator, width);
ns::Array2D<float> a2(allocator, width, height);
ns::Array3D<float> a3(allocator, width, height, depth);

a1.size();            // element count
a2.width(); a2.height();
a1.data();            // raw device pointer (Type*)
a1.ptr();             // dev::Ptr<float> — typed device pointer
```

### `dev::Ptr<T>` — Typed Device Pointer

A lightweight, typed wrapper for a device pointer that carries array bounds. It is both host- and device-callable and supports optional runtime bounds checking.

```cpp
// Host side: pass as kernel argument
stream.launch(my_kernel, grid, block)(array.ptr(), count);

// Device side (in .cu): element access with optional bounds check
__global__ void my_kernel(dev::Ptr<float> data, unsigned int n)
{
    CUDA_for(i, n);   // auto bounds-check; returns if i >= n
    data[i] *= 2.0f;
}

// Type-safe reinterpret cast between binary-compatible types
auto as_bytes = ns::ptr_cast<unsigned char>(array.ptr());
```

### Kernel Launch Helpers (`.cuh`, CUDA only)

Include `<nucleus/launch_utils.cuh>` in `.cu` files to access:

| Symbol | Description |
|---|---|
| `ns::ceil_div(x, y)` | Ceiling integer division — use for computing grid dimensions |
| `ns::align_up(x, alignment)` | Round up to the nearest multiple of `alignment` |
| `ns::tid()` | Global 1-D thread index (`blockDim.x * blockIdx.x + threadIdx.x`) |
| `CUDA_for(i, n)` | Declare `i = tid()` and early-return if `i >= n` |
| `NS_BOUNDS_CHECK(i, n)` | Same as `CUDA_for` (uppercase alias) |

### `ns::Event` and `ns::Graph`

```cpp
// Event-based synchronization and timing
ns::Event event(device);
stream.record(event);
event.sync();

// CUDA graph — automatically caches topology and updates parameters
ns::Graph graph(device);
auto dep = graph.launch(my_kernel, {}, grid, block)(args…);
graph.execute(stream);
```

### Images, Textures, and Surfaces

Nucleus separates texture memory (the backing storage) from the sampling/access objects:

- **Image** — allocates CUDA array memory on the device (analogous to a GPU texture buffer). Created once and shared across texture/surface objects.
- **Texture** — a read-only, hardware-sampled view of an Image. Passed to kernels as a `dev::Tex*<T>` handle.
- **Surface** — a read-write view of an Image (requires `bSurfaceLoadStore = true` at image creation). Passed to kernels as a `dev::Surf*<T>` handle.

The table below lists all available Image, Texture, and Surface type combinations:

| Dimensionality | Image (host) | Texture (host → device) | Surface (host → device) |
|---|---|---|---|
| 1-D | `Image1D<T>` | `Texture1D<T>` → `dev::Tex1D<T>` | `Surface1D<T>` → `dev::Surf1D<T>` |
| 2-D | `Image2D<T>` | `Texture2D<T>` → `dev::Tex2D<T>` | `Surface2D<T>` → `dev::Surf2D<T>` |
| 3-D | `Image3D<T>` | `Texture3D<T>` → `dev::Tex3D<T>` | `Surface3D<T>` → `dev::Surf3D<T>` |
| Cubemap | `ImageCube<T>` | `TextureCube<T>` → `dev::TexCube<T>` | `SurfaceCube<T>` → `dev::SurfCube<T>` |
| 1-D layered | `Image1DLayered<T>` | `Texture1DLayered<T>` → `dev::Tex1DLayered<T>` | `Surface1DLayered<T>` → `dev::Surf1DLayered<T>` |
| 2-D layered | `Image2DLayered<T>` | `Texture2DLayered<T>` → `dev::Tex2DLayered<T>` | `Surface2DLayered<T>` → `dev::Surf2DLayered<T>` |
| Cubemap layered | `ImageCubeLayered<T>` | `TextureCubeLayered<T>` → `dev::TexCubeLayered<T>` | `SurfaceCubeLayered<T>` → `dev::SurfCubeLayered<T>` |
| 1-D mipmapped | `Image1DLod<T>` | `Texture1DLod<T>` → `dev::Tex1DLod<T>` | — |
| 2-D mipmapped | `Image2DLod<T>` | `Texture2DLod<T>` → `dev::Tex2DLod<T>` | — |
| 3-D mipmapped | `Image3DLod<T>` | `Texture3DLod<T>` → `dev::Tex3DLod<T>` | — |
| Cubemap mipmapped | `ImageCubeLod<T>` | `TextureCubeLod<T>` → `dev::TexCubeLod<T>` | — |
| 1-D layered mipmapped | `Image1DLayeredLod<T>` | `Texture1DLayeredLod<T>` → `dev::Tex1DLayeredLod<T>` | — |
| 2-D layered mipmapped | `Image2DLayeredLod<T>` | `Texture2DLayeredLod<T>` → `dev::Tex2DLayeredLod<T>` | — |
| Cubemap layered mipmapped | `ImageCubeLayeredLod<T>` | `TextureCubeLayeredLod<T>` → `dev::TexCubeLayeredLod<T>` | — |

**Typical usage pattern:**

```cpp
#include <nucleus/image_2d.h>
#include <nucleus/texture.h>
#include <nucleus/sampler.h>

// 1. Allocate image memory (CUDA array on the device).
auto image = std::make_shared<ns::Image2D<float4>>(device->defaultAllocator(), width, height);

// 2. Upload data via stream.memcpy (host → image).
stream.memcpy(image->data(), host_pixels.data(), width, height);

// 3. Create a texture bound to the image.
ns::Sampler sampler;                           // default: linear filter, clamp
ns::Texture2D<float4> tex(image, sampler);

// 4. Pass the device-side handle to a kernel.
stream.launch(my_kernel, grid, block)(tex.handle(), width, height);
```

```cu
// Device side (in .cu):
__global__ void my_kernel(dev::Tex2D<float4> tex, int width, int height)
{
    CUDA_for(tid, width * height);
    float u = (tid % width + 0.5f) / width;
    float v = (tid / width + 0.5f) / height;
    float4 color = tex.fetch(u, v);   // hardware-interpolated sample
    // ...
}
```

## Project Structure

```
Nucleus/
├── include/nucleus/        # Public headers (add to your include path)
│   ├── nucleus.h           # Convenience header — includes everything
│   ├── runtime.h           # ns::Runtime  (device enumeration, singleton)
│   ├── device.h            # ns::Device   (GPU handle, properties, allocator)
│   ├── stream.h            # ns::Stream   (async work, kernel launch, memcpy)
│   ├── event.h             # ns::Event, ns::TimedEvent
│   ├── graph.h             # ns::Graph    (CUDA graph with auto-caching)
│   ├── allocator.h         # ns::Allocator, DeviceAllocator, HostAllocator
│   ├── buffer.h            # ns::Buffer   (raw RAII memory block)
│   ├── buffer_view.h       # ns::BufferView/BufferView2D/BufferView3D (non-owning views)
│   ├── array_1d.h          # ns::Array<T>
│   ├── array_2d.h          # ns::Array2D<T>
│   ├── array_3d.h          # ns::Array3D<T>
│   ├── array_proxy.h       # ns::ArrayProxy<T> (span-like argument helper)
│   ├── device_pointer.h    # dev::Ptr<T>, dev::Ptr2<T>, dev::Ptr3<T>
│   ├── device_texture.h    # dev::Tex1D/2D/3D/Cube<T> (device-side sampling)
│   ├── device_surface.h    # dev::Surf1D/2D/3D/Cube<T> (device-side r/w)
│   ├── texture.h           # ns::Texture1D/2D/3D/Cube<T> (host wrapper)
│   ├── surface.h           # ns::Surface1D/2D/3D/Cube<T> (host wrapper)
│   ├── image_*.h           # ns::Image1D/2D/3D/Cube/Layered/Lod<T>
│   ├── format.h            # ns::Format  (texel format enum)
│   ├── sampler.h           # ns::Sampler (texture sampler configuration)
│   ├── logger.h            # ns::Logger
│   ├── scoped_timer.h      # ns::ScopedTimer
│   ├── launch_utils.cuh    # Kernel helpers — include in .cu files only
│   ├── fwd.h               # Forward declarations for all public types
│   └── macros.h            # Compiler/platform macros (NS_API, NS_CUDA_CALLABLE, …)
├── src/                    # Library implementation (.cpp / .cu)
├── examples/               # Standalone example programs
│   ├── customized_allocator/   Custom allocator implementation
│   ├── customized_pixel_type/  Struct-based pixel types in kernels
│   ├── dynamic_texture/        Dynamic texture creation and binding
│   ├── julia_set/              GPU fractal rendering
│   ├── multi_gpu/              Multi-GPU workload distribution
│   ├── pinned_memory/          Page-locked (pinned) host memory
│   └── random_access/          Texture-based random access
├── tests/                  # Unit and integration tests
├── cmake/                  # CMake helper scripts and export headers
└── CMakeLists.txt
```

## Building and Running Tests

Enable the test suite by passing `-DNUCLEUS_BUILD_TESTS=ON` at configure time:

```bash
cmake -B build -DNUCLEUS_BUILD_TESTS=ON
cmake --build build
cd build && ctest --output-on-failure
```

Test source files are in `tests/`. CPU-only tests use `.cpp` and CUDA tests use `.cu`.

## Building and Running Examples

```bash
cmake -B build -DNUCLEUS_BUILD_EXAMPLES=ON
cmake --build build
# Run a cross-platform example:
./build/examples/customized_allocator/customized_allocator
```

> **Note:** Some examples (`julia_set`, `multi_gpu`, `dynamic_texture`) open a graphical window and are currently supported on **Windows only**.

## License

Nucleus is distributed under the terms of the [MIT License](LICENSE).
