#include "test.cuh"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

dim3 threads_rect(32, 4);
dim3 blocks_rect(1, 4);

dim3 threads_square(16, 8);
dim3 blocks_square(2, 2);

using type = uint32_t;
constexpr uint16_t ARRAY_SIZE_X = 32;
constexpr uint16_t ARRAY_SIZE_Y = 16;
constexpr uint32_t ARRAY_SIZE_IN_BYTES = ARRAY_SIZE_X * ARRAY_SIZE_Y * sizeof(type);

constexpr uint32_t LOOPS = 1024;

type cpu_block_x[ARRAY_SIZE_Y][ARRAY_SIZE_X];
type cpu_block_y[ARRAY_SIZE_Y][ARRAY_SIZE_X];
type cpu_thread[ARRAY_SIZE_Y][ARRAY_SIZE_X];
type cpu_warp[ARRAY_SIZE_Y][ARRAY_SIZE_X];
type cpu_calc_thread[ARRAY_SIZE_Y][ARRAY_SIZE_X];
type cpu_xthread[ARRAY_SIZE_Y][ARRAY_SIZE_X];
type cpu_ythread[ARRAY_SIZE_Y][ARRAY_SIZE_X];
type cpu_grid_dimx[ARRAY_SIZE_Y][ARRAY_SIZE_X];
type cpu_block_dimx[ARRAY_SIZE_Y][ARRAY_SIZE_X];
type cpu_grid_dimy[ARRAY_SIZE_Y][ARRAY_SIZE_X];
type cpu_block_dimy[ARRAY_SIZE_Y][ARRAY_SIZE_X];

__global__ void what_is_my_id_2d_A(type *const block_x, type *const block_y, type *const thread,
                                   type *const calc_thread, type *const x_thread, type *const y_thread,
                                   type *const grid_dimx, type *const block_dimx, type *const grid_dimy,
                                   type *const block_dimy) {
    const type idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    const type idy = (blockIdx.y * blockDim.y) + threadIdx.y;
    const type thread_idx = ((gridDim.x * blockDim.x) * idy) + idx;

    block_x[thread_idx] = blockIdx.x;
    block_y[thread_idx] = blockIdx.y;
    thread[thread_idx] = threadIdx.x;
    calc_thread[thread_idx] = thread_idx;
    x_thread[thread_idx] = idx;
    y_thread[thread_idx] = idy;
    grid_dimx[thread_idx] = gridDim.x;
    block_dimx[thread_idx] = blockDim.x;
    grid_dimy[thread_idx] = gridDim.y;
    block_dimy[thread_idx] = blockDim.y;
}

void test_gpu() {
    type *gpu_block_x;
    type *gpu_block_y;
    type *gpu_thread;
    type *gpu_calc_thread;
    type *gpu_xthread;
    type *gpu_ythread;
    type *gpu_grid_dimx;
    type *gpu_block_dimx;
    type *gpu_grid_dimy;
    type *gpu_block_dimy;

    cudaMalloc(reinterpret_cast<void **>(&gpu_block_x), ARRAY_SIZE_IN_BYTES);
    cudaMalloc(reinterpret_cast<void **>(&gpu_block_y), ARRAY_SIZE_IN_BYTES);
    cudaMalloc(reinterpret_cast<void **>(&gpu_thread), ARRAY_SIZE_IN_BYTES);
    cudaMalloc(reinterpret_cast<void **>(&gpu_calc_thread), ARRAY_SIZE_IN_BYTES);
    cudaMalloc(reinterpret_cast<void **>(&gpu_xthread), ARRAY_SIZE_IN_BYTES);
    cudaMalloc(reinterpret_cast<void **>(&gpu_ythread), ARRAY_SIZE_IN_BYTES);
    cudaMalloc(reinterpret_cast<void **>(&gpu_grid_dimx), ARRAY_SIZE_IN_BYTES);
    cudaMalloc(reinterpret_cast<void **>(&gpu_block_dimx), ARRAY_SIZE_IN_BYTES);
    cudaMalloc(reinterpret_cast<void **>(&gpu_grid_dimy), ARRAY_SIZE_IN_BYTES);
    cudaMalloc(reinterpret_cast<void **>(&gpu_block_dimy), ARRAY_SIZE_IN_BYTES);


    for (uint8_t kernel = 0; kernel < 2; ++kernel) {

        auto start = std::chrono::high_resolution_clock::now();
        switch (kernel) {
            case 0:
                for (uint32_t i = 0; i < LOOPS; ++i) {
                    what_is_my_id_2d_A<<<blocks_rect, threads_rect>>>(
                            gpu_block_x, gpu_block_y, gpu_thread, gpu_calc_thread, gpu_xthread, gpu_ythread,
                            gpu_grid_dimx, gpu_block_dimx, gpu_grid_dimy, gpu_block_dimy);
                }
                break;
            case 1:
                for (uint32_t i = 0; i < LOOPS; ++i) {
                    what_is_my_id_2d_A<<<blocks_square, threads_square>>>(
                            gpu_block_x, gpu_block_y, gpu_thread, gpu_calc_thread, gpu_xthread, gpu_ythread,
                            gpu_grid_dimx, gpu_block_dimx, gpu_grid_dimy, gpu_block_dimy);
                }
                break;
            default:
                exit(EXIT_FAILURE);
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;


        cudaMemcpy(cpu_block_x, gpu_block_x, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_block_y, gpu_block_y, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_thread, gpu_thread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_calc_thread, gpu_calc_thread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_xthread, gpu_xthread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_ythread, gpu_ythread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_grid_dimx, gpu_grid_dimx, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_block_dimx, gpu_block_dimx, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_grid_dimy, gpu_grid_dimy, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_block_dimy, gpu_block_dimy, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);

        std::cout << "\nKernel: " << static_cast<uint16_t>(kernel) << std::endl;
        std::cout << "Time: " << elapsed.count() << " s" << std::endl;
        // for (uint16_t y = 0; y < ARRAY_SIZE_Y; ++y) {
        //     for (uint16_t x = 0; x < ARRAY_SIZE_X; ++x) {
        //         std::cout << "CT: " << cpu_calc_thread[y][x] << " BKX: " << cpu_block_x[y][x] << " BKY: " <<
        //         cpu_block_y[y][x] << " TID: " << cpu_thread[y][x] << " YTID: " << cpu_ythread[y][x] << " XTID: " <<
        //         cpu_xthread[y][x] << " GDX: " << cpu_grid_dimx[y][x] << " BDX: " << cpu_block_x[y][x] << " GDY: " <<
        //         cpu_grid_dimy[y][x] << " BDY: " << cpu_block_y[y][x] << std::endl;
        //     }
        // }

        std::cout << std::endl;
    }


    cudaFree(gpu_block_x);
    cudaFree(gpu_block_y);
    cudaFree(gpu_thread);
    cudaFree(gpu_calc_thread);
    cudaFree(gpu_xthread);
    cudaFree(gpu_ythread);
    cudaFree(gpu_grid_dimx);
    cudaFree(gpu_block_dimx);
    cudaFree(gpu_grid_dimy);
    cudaFree(gpu_block_dimy);
}
