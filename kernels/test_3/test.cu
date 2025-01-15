#include "test.cuh"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

using type = int;
constexpr size_t N = 256;
constexpr size_t M = 16384;

constexpr uint16_t num_threads = 128;
constexpr uint16_t num_blocks = M / num_threads;

type cpu_bin[N];
type cpu_data[M];

__global__ void histogram256Kernel_01(const type const *d_hist_data, type *const d_bin_data) {
    // const uint32_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    // const uint32_t idy = (blockIdx.y * blockDim.y) + threadIdx.y;
    // const uint32_t tid = idx + idy * blockDim.x * gridDim.x;
    const uint16_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    const type value = d_hist_data[tid];
    atomicAdd((&d_bin_data[value]), 1);
}

__global__ void histogram256Kernel_02(const type const *d_hist_data, type *const d_bin_data) {
    // const uint32_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    // const uint32_t idy = (blockIdx.y * blockDim.y) + threadIdx.y;
    // const uint32_t tid = idx + idy * blockDim.x * gridDim.x;
    const uint16_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    const type value = d_hist_data[tid];
    atomicAdd((&d_bin_data[value]), 1);
}

size_t getRandomNumber(int min, int max) {
    // Ensure the range is valid
    if (min > max) {
        std::swap(min, max);
    }

    // Create a random device and seed it
    static std::random_device rd; // Non-deterministic random device
    static std::mt19937 generator(rd()); // Mersenne Twister engine

    // Define the range for the distribution
    std::uniform_int_distribution<int> distribution(min, max);

    // Generate and return the random number
    return distribution(generator);
}

void test_gpu() {
    type *gpu_bin;
    type *gpu_data;

    cudaMallocManaged(&gpu_bin, N * sizeof(type));
    cudaMallocManaged(&gpu_data, M * sizeof(type));

    void (*kernels[])(const type *d_hist_data, type *const d_bin_data) = {
        histogram256Kernel_01,
        histogram256Kernel_02
    };
    size_t num_kernels = sizeof(kernels) / sizeof(kernels[0]);
    std::cout << "Kernel size: " << num_kernels << std::endl;


    for (int i = 0; i < num_kernels; i++) {
        std::cout << "Kernel: " << i << std::endl;

        // Test 1
        for (size_t j = 0; j < M; j++) {
            cpu_data[j] = j % N;
        }
        cudaMemcpy(gpu_data, cpu_data, M * sizeof(type), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_bin, cpu_bin, N * sizeof(type), cudaMemcpyHostToDevice);
        auto start = std::chrono::high_resolution_clock::now();
        kernels[i]<<<num_blocks, num_threads>>>(gpu_data, gpu_bin);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed = end - start;
        std::cout << "\tTest 1:\n";
        std::cout << "\t\tTime: " << elapsed.count() << " s" << std::endl;

        // Test 2
        for (size_t j = 0; j < M; j++) {
            cpu_data[j] = 0;
        }
        cudaMemcpy(gpu_data, cpu_data, M * sizeof(type), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_bin, cpu_bin, N * sizeof(type), cudaMemcpyHostToDevice);
        start = std::chrono::high_resolution_clock::now();
        kernels[i]<<<num_blocks, num_threads>>>(gpu_data, gpu_bin);
        end = std::chrono::high_resolution_clock::now();
        cudaMemcpy(cpu_bin, gpu_bin, N * sizeof(type), cudaMemcpyDeviceToHost);

        for (size_t j = 0; j < N; j++) {
            std::cout << j << ". " << cpu_bin[j] << std::endl;
        }

        elapsed = end - start;
        std::cout << "\tTest 2:\n";
        std::cout << "\t\tTime: " << elapsed.count() << " s" << std::endl;

        // Test 3
        for (size_t j = 0; j < M; j++) {
            cpu_data[j] = static_cast<type>(getRandomNumber(0, N - 1));
        }
        cudaMemcpy(gpu_data, cpu_data, M * sizeof(type), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_bin, cpu_bin, N * sizeof(type), cudaMemcpyHostToDevice);
        start = std::chrono::high_resolution_clock::now();
        kernels[i]<<<num_blocks, num_threads>>>(gpu_data, gpu_bin);
        end = std::chrono::high_resolution_clock::now();

        elapsed = end - start;
        std::cout << "\tTest 3:\n";
        std::cout << "\t\tTime: " << elapsed.count() << " s" << std::endl;
    }

    // std::cout << "num_blocks: " << static_cast<uint16_t>(num_blocks) << std::endl;
    // std::cout << "num_threads: " << static_cast<uint16_t>(num_threads) << std::endl;
    //
    // cudaMemcpy(gpu_data, cpu_data, M * sizeof(type), cudaMemcpyHostToDevice);
    // kernels[0]<<<num_blocks, num_threads>>>(gpu_data, gpu_bin);
    // cudaMemcpy(cpu_bin, gpu_bin, N * sizeof(type), cudaMemcpyDeviceToHost);
    //
}
