enable_language(CUDA)
add_library(kernels_1
        test_1/test.cu
        test_1/test.cuh
)
include_directories(kernels_1
    test_1
)
set_target_properties(kernels_1 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_library(kernels_2
        test_2/test.cu
        test_2/test.cuh
)
include_directories(kernels_2
        test_2
)
set_target_properties(kernels_2 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_library(kernels_3
        test_3/test.cu
        test_3/test.cuh
)
include_directories(kernels_3
        test_3
)
set_target_properties(kernels_3 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)