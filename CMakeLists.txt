cmake_minimum_required(VERSION 3.30)
set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_C_COMPILER "/usr/bin/gcc")
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(cudaLearning CUDA CXX C)
set(CMAKE_CUDA_STANDARD 23)
set(CMAKE_CXX_STANDARD 23)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")
endif ()

add_executable(${PROJECT_NAME}_1 main.cpp)
# target_sources(${PROJECT_NAME} PRIVATE test.cu)
set_target_properties(${PROJECT_NAME}_1 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(${PROJECT_NAME}_1
    kernels_1
)
include_directories(${PROJECT_NAME}_1
        kernels/test_1
)

add_executable(${PROJECT_NAME}_2 main.cpp)
set_target_properties(${PROJECT_NAME}_2 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(${PROJECT_NAME}_2
        kernels_2
)
include_directories(${PROJECT_NAME}_2
        kernels/test_2
)

add_executable(${PROJECT_NAME}_3 main.cpp)
set_target_properties(${PROJECT_NAME}_3 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(${PROJECT_NAME}_3
        kernels_3
)
include_directories(${PROJECT_NAME}_3
        kernels/test_3
)

add_subdirectory(kernels)
