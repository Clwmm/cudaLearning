#include "test.cuh"

#include <cstdint>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>

constexpr uint16_t ARRAY_SIZE = 128;
constexpr uint32_t ARRAY_SIZE_IN_BYTES = sizeof(uint16_t) * ARRAY_SIZE;

constexpr uint8_t num_blocks = 2;
constexpr uint8_t num_threads = 64;

inline uint16_t cpu_block[ARRAY_SIZE];
inline uint16_t cpu_thread[ARRAY_SIZE];
inline uint16_t cpu_warp[ARRAY_SIZE];
inline uint16_t cpu_calc_thread[ARRAY_SIZE];

__global__ void what_is_my_id(uint16_t * const block,
                              uint16_t * const thread,
                              uint16_t * const warp,
                              uint16_t * const calc_thread)
{
    const uint16_t thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    block[thread_idx] = blockIdx.x;
    thread[thread_idx] = threadIdx.x;
    warp[thread_idx] = threadIdx.x / warpSize;
    calc_thread[thread_idx] = thread_idx;
}

void test_gpu()
{
    uint16_t * gpu_block;
    uint16_t * gpu_thread;
    uint16_t * gpu_warp;
    uint16_t * gpu_calc_thread;

    cudaMalloc(reinterpret_cast<void **>(&gpu_block), ARRAY_SIZE_IN_BYTES);
    cudaMalloc(reinterpret_cast<void **>(&gpu_thread), ARRAY_SIZE_IN_BYTES);
    cudaMalloc(reinterpret_cast<void **>(&gpu_warp), ARRAY_SIZE_IN_BYTES);
    cudaMalloc(reinterpret_cast<void **>(&gpu_calc_thread), ARRAY_SIZE_IN_BYTES);

    what_is_my_id<<<num_blocks, num_threads>>>(gpu_block, gpu_thread, gpu_warp, gpu_calc_thread);

    cudaMemcpy(cpu_block, gpu_block, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_thread, gpu_thread, ARRAY_SIZE_IN_BYTES,cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_warp, gpu_warp, ARRAY_SIZE_IN_BYTES,cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_calc_thread, gpu_calc_thread, ARRAY_SIZE_IN_BYTES,cudaMemcpyDeviceToHost);

    cudaFree(gpu_block);
    cudaFree(gpu_thread);
    cudaFree(gpu_warp);
    cudaFree(gpu_calc_thread);

    std::cout << "test_1 | test_gpu()" << std::endl;

    for (uint16_t i = 0; i < ARRAY_SIZE; i++)
    {
        std::cout << "Calculated Thread: " << cpu_calc_thread[i] << " - Block: " << cpu_block[i] << " - Warp: " << cpu_warp[i] << " - Thread: " << cpu_thread[i] << std::endl;
    }
}