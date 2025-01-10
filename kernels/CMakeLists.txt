enable_language(CUDA)
add_library(kernels_1
        test_1/test.cu
        test_2/test.cuh
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