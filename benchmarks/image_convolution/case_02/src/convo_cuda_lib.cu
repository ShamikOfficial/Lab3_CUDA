#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <cuda_runtime.h>

// ============================================================
// CUDA kernels (same as the other codes before)
// ============================================================

__global__ void k_edge_n3(const uint8_t *image, uint8_t *out, int W, int H) {
    const int32_t K[9] = { 0, -1,  0, -1,  4, -1,  0, -1,  0 };
    int r = 1;
    int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= W || y >= H) return;
    int32_t sum = 0;
    for (int ky = 0; ky < 3; ky++) {
        for (int kx = 0; kx < 3; kx++) {
            int ix = x + (kx - r);
            int iy = y + (ky - r);
            uint8_t p = 0;
            if (ix >= 0 && ix < W && iy >= 0 && iy < H) p = image[iy * W + ix];
            sum += (int32_t)p * K[ky * 3 + kx];
        }
    }
    if (sum < 0) sum = -sum;
    if (sum < 0) sum = 0;
    if (sum > 255) sum = 255;
    out[y * W + x] = (uint8_t)sum;
}

__global__ void k_sharpen_n3(const uint8_t *image, uint8_t *out, int W, int H) {
    const int32_t K[9] = { -1, -1, -1, -1,  9, -1, -1, -1, -1 };
    int r = 1;
    int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= W || y >= H) return;
    int32_t sum = 0;
    for (int ky = 0; ky < 3; ky++) {
        for (int kx = 0; kx < 3; kx++) {
            int ix = x + (kx - r);
            int iy = y + (ky - r);
            uint8_t p = 0;
            if (ix >= 0 && ix < W && iy >= 0 && iy < H) p = image[iy * W + ix];
            sum += (int32_t)p * K[ky * 3 + kx];
        }
    }
    if (sum < 0) sum = 0;
    if (sum > 255) sum = 255;
    out[y * W + x] = (uint8_t)sum;
}

__global__ void k_blur_n3(const uint8_t *image, uint8_t *out, int W, int H) {
    const int32_t K[9] = { 1, 2, 1, 2, 4, 2, 1, 2, 1 };
    int r = 1;
    int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= W || y >= H) return;
    int32_t sum = 0;
    for (int ky = 0; ky < 3; ky++) {
        for (int kx = 0; kx < 3; kx++) {
            int ix = x + (kx - r);
            int iy = y + (ky - r);
            uint8_t p = 0;
            if (ix >= 0 && ix < W && iy >= 0 && iy < H) p = image[iy * W + ix];
            sum += (int32_t)p * K[ky * 3 + kx];
        }
    }
    sum /= 16;
    if (sum < 0) sum = 0;
    if (sum > 255) sum = 255;
    out[y * W + x] = (uint8_t)sum;
}

// ============================================================
// Functions for Python interface
// ============================================================

extern "C" void gpu_convolve_u8(uint8_t *h_image, uint8_t *h_out, int W, int H, const char *mode) {
    size_t n = (size_t)W * (size_t)H;
    uint8_t *d_img = NULL;
    uint8_t *d_out = NULL;

    cudaMalloc((void**)&d_img, n);
    cudaMalloc((void**)&d_out, n);

    cudaMemcpy(d_img, h_image, n, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);

    if (strcmp(mode, "edge_n3") == 0) k_edge_n3<<<grid, block>>>(d_img, d_out, W, H);
    else if (strcmp(mode, "sharpen_n3") == 0) k_sharpen_n3<<<grid, block>>>(d_img, d_out, W, H);
    else if (strcmp(mode, "blur_n3") == 0) k_blur_n3<<<grid, block>>>(d_img, d_out, W, H);

    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, n, cudaMemcpyDeviceToHost);

    cudaFree(d_img);
    cudaFree(d_out);
}
