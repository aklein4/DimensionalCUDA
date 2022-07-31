cmake_minimum_required(VERSION 3.8.1)
message(STATUS "CMake version: ${CMAKE_VERSION}")

enable_language("CUDA")
if(NOT DEFINED ${CMAKE_CUDA_ARCHITECTURES})
    set(CMAKE_CUDA_ARCHITECTURES 52 61 75)
endif()
message(STATUS "CUDA architectures set to ${CMAKE_CUDA_ARCHITECTURES}")

project(DimensionalCUDA)

file(GLOB cpu_source_files "${CMAKE_SOURCE_DIR}/*.cpp")
file(GLOB gpu_source_files "${CMAKE_SOURCE_DIR}/*.cu")

add_library(DimensionalCUDALib ${cpu_source_files} ${gpu_source_files})

add_executable(DimensionalCUDAMain main.cu)
target_link_libraries(DimensionalCUDAMain DimensionalCUDALib)

add_definitions(-DUSE_CUDA)