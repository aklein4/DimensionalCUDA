
# cmake boiler plate
cmake_minimum_required(VERSION 3.8.1)
message(STATUS "CMake version: ${CMAKE_VERSION}")

# cuda setup
enable_language("CUDA")
if(NOT DEFINED ${CMAKE_CUDA_ARCHITECTURES})
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()
message(STATUS "CUDA architectures set to ${CMAKE_CUDA_ARCHITECTURES}")

# I am speed
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -O3 -use_fast_math -Xptxas -v")

# initialize the project
project(DimensionalCUDA LANGUAGES CXX CUDA)

# create library
file(GLOB cpu_source_files "${CMAKE_CURRENT_LIST_DIR}/*.cpp")
file(GLOB gpu_source_files "${CMAKE_CURRENT_LIST_DIR}/*.cu")
add_library(DimensionalCUDALib ${cpu_source_files} ${gpu_source_files})

# create and link exe
add_executable(GPU_test "${CMAKE_CURRENT_LIST_DIR}/main.cu")
target_link_libraries(GPU_test DimensionalCUDALib)
