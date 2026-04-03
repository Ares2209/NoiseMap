# RayTracer/CMakeLists.txt

cmake_minimum_required(VERSION 3.14)
project(RayTracer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "CUDA architectures")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
endif()

# OptiX
set(OptiX_DIR "/usr/local/optix" CACHE PATH "Path to OptiX installation")
list(APPEND CMAKE_PREFIX_PATH ${OptiX_DIR})
list(APPEND CMAKE_MODULE_PATH "${OptiX_DIR}/SDK/CMake/")
include_directories("${OptiX_DIR}/include")
find_package(OptiX REQUIRED)

find_package(spdlog REQUIRED)
find_package(CGAL CONFIG REQUIRED)

# === Source files for host-side library ===
set(RAYTRACER_SRC
    ray_tracer.cpp
    acoustic_model.cpp
)

# === Create static library ===
add_library(RayTracer STATIC ${RAYTRACER_SRC})
target_include_directories(RayTracer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} "${OptiX_DIR}/include")

target_link_libraries(RayTracer
    PUBLIC
        CUDA::cudart
        CGAL::CGAL
        spdlog::spdlog
        cuda
)

# PTX Compilation from optix_ray.cu
set(OPTIX_PTX "${CMAKE_BINARY_DIR}/RayTracer.optixir")
set(OPTIX_COMPILE_FLAGS -optix-ir -arch=sm_75 -I "${OptiX_DIR}/include")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    list(APPEND OPTIX_COMPILE_FLAGS -G)
endif()

add_custom_command(
    OUTPUT ${OPTIX_PTX}
    COMMAND ${CMAKE_CUDA_COMPILER} ${OPTIX_COMPILE_FLAGS} ${CMAKE_CURRENT_SOURCE_DIR}/optix_ray.cu -o ${OPTIX_PTX}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/optix_ray.cu
    COMMENT "Compiling OptiX PTX: optix_ray.cu -> RayTracer.optixir"
)

add_custom_target(OptiXPTX DEPENDS ${OPTIX_PTX})
add_dependencies(RayTracer OptiXPTX)
