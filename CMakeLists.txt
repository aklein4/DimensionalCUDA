cmake_minimum_required(VERSION 3.8.1)
message(STATUS "CMake version: ${CMAKE_VERSION}")

enable_language("CUDA")

file(GLOB cpu_source_files "${CMAKE_SOURCE_DIR}/*.cpp")
file(GLOB gpu_source_files "${CMAKE_SOURCE_DIR}/*.cu")

add_library(DimensionalCUDA ${cpu_source_files} ${gpu_source_files})
set_property(TARGET DimensionalCUDA PROPERTY CUDA_ARCHITECTURES "86")

add_definitions(-DUSE_CUDA)