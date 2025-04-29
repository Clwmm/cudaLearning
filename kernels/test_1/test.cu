#include "test.cuh"

#include <cstdint>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

typedef uint32_t type;

constexpr uint16_t num_blocks = 2;
constexpr uint16_t num_threads = 1024;

constexpr uint32_t ARRAY_SIZE = num_blocks * num_threads;
constexpr uint32_t ARRAY_SIZE_IN_BYTES = sizeof(uint16_t) * ARRAY_SIZE;

__global__ void what_is_my_id(uint16_t *const block, uint16_t *const thread, uint16_t *const warp,
                              uint32_t *const calc_thread) {
    const uint32_t thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    block[thread_idx] = blockIdx.x;
    thread[thread_idx] = threadIdx.x;
    warp[thread_idx] = threadIdx.x / warpSize;
    calc_thread[thread_idx] = thread_idx;
}

void test_gpu() {
    auto *cpu_block = new uint16_t[ARRAY_SIZE];
    auto *cpu_thread = new uint16_t[ARRAY_SIZE];
    auto *cpu_warp = new uint16_t[ARRAY_SIZE];
    auto *cpu_calc_thread = new uint32_t[ARRAY_SIZE];

    uint16_t *gpu_block;
    uint16_t *gpu_thread;
    uint16_t *gpu_warp;
    uint32_t *gpu_calc_thread;

    cudaMalloc(reinterpret_cast<void **>(&gpu_block), ARRAY_SIZE_IN_BYTES);
    cudaMalloc(reinterpret_cast<void **>(&gpu_thread), ARRAY_SIZE_IN_BYTES);
    cudaMalloc(reinterpret_cast<void **>(&gpu_warp), ARRAY_SIZE_IN_BYTES);
    cudaMalloc(reinterpret_cast<void **>(&gpu_calc_thread), sizeof(uint32_t) * ARRAY_SIZE);

    what_is_my_id<<<num_blocks, num_threads>>>(gpu_block, gpu_thread, gpu_warp, gpu_calc_thread);

    cudaMemcpy(cpu_block, gpu_block, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_thread, gpu_thread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_warp, gpu_warp, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_calc_thread, gpu_calc_thread, sizeof(uint32_t) * ARRAY_SIZE, cudaMemcpyDeviceToHost);

    cudaFree(gpu_block);
    cudaFree(gpu_thread);
    cudaFree(gpu_warp);
    cudaFree(gpu_calc_thread);

    std::cout << "test_1 | test_gpu()" << std::endl;

    for (uint32_t i = 0; i < ARRAY_SIZE; i++) {
        std::cout << "Calculated Thread: " << cpu_calc_thread[i] << " - Block: " << cpu_block[i]
                  << " - Warp: " << cpu_warp[i] << " - Thread: " << cpu_thread[i] << std::endl;
    }
    // std::cout << "Calculated Thread: " << cpu_calc_thread[ARRAY_SIZE - 1] << " - Block: " << cpu_block[ARRAY_SIZE - 1]
    //               << " - Warp: " << cpu_warp[ARRAY_SIZE - 1] << " - Thread: " << cpu_thread[ARRAY_SIZE -1] << std::endl;

    delete[] cpu_block;
    delete[] cpu_thread;
    delete[] cpu_warp;
    delete[] cpu_calc_thread;

    std::cout << "ARRAY_SIZE: " << ARRAY_SIZE << std::endl;
    std::cout << "ARRAY_SIZE_IN_BYTES: " << ARRAY_SIZE_IN_BYTES << std::endl;
}
