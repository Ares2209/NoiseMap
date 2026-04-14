cmake_minimum_required(VERSION 3.14)
project(NoiseMap LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "CUDA architectures")

# OptiX
set(OptiX_DIR "/home/ldena/Nvidia/NVIDIA-OptiX-SDK-9.0.0-linux64-x86_64" CACHE PATH "Path to OptiX installation")
include_directories("${OptiX_DIR}/include")

find_package(spdlog REQUIRED)
find_package(CGAL CONFIG REQUIRED)

set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/RayTracer")

# PTX Compilation
set(OPTIX_PTX "${CMAKE_BINARY_DIR}/RayTracer.optixir")
add_custom_command(
    OUTPUT ${OPTIX_PTX}
    COMMAND ${CMAKE_CUDA_COMPILER}
            -optix-ir -arch=sm_75
            -I "${OptiX_DIR}/include"
            ${SRC_DIR}/optix_ray.cu
            -o ${OPTIX_PTX}
    DEPENDS ${SRC_DIR}/optix_ray.cu
    COMMENT "Compiling OptiX IR"
)
add_custom_target(OptiXPTX DEPENDS ${OPTIX_PTX})

# RayTracer library
add_library(RayTracer STATIC
    ${SRC_DIR}/ray_tracer.cpp
    ${SRC_DIR}/acoustic_model.cpp
)
target_include_directories(RayTracer PUBLIC ${SRC_DIR} "${OptiX_DIR}/include")
target_link_libraries(RayTracer PUBLIC CUDA::cudart CGAL::CGAL spdlog::spdlog cuda)
add_dependencies(RayTracer OptiXPTX)

# Main executable
add_executable(NoiseMap
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scene.cpp
)
target_include_directories(NoiseMap PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${SRC_DIR}
    "${OptiX_DIR}/include"
)
target_link_libraries(NoiseMap PRIVATE RayTracer)