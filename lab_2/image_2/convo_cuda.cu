#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <ctype.h>
#include <string.h>

#include <cuda_runtime.h>

#define DEFAULT_M 512
static void skip_ws_and_comments(FILE *f) {
    int c;
    while ((c = fgetc(f)) != EOF) {
        if (isspace(c)) continue;
        if (c == '#') { while ((c = fgetc(f)) != EOF && c != '\n') {} continue; }
        ungetc(c, f);
        return;
    }
}

// Read binary, maxval=255.
static uint8_t* read_pgm_p5(const char *path, int *W, int *H) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror("fopen"); return NULL; }

    char magic[3] = {0};
    if (fread(magic, 1, 2, f) != 2 || magic[0] != 'P' || magic[1] != '5') {
        fprintf(stderr, "Not a binary PGM (P5): %s\n", path);
        fclose(f);
        return NULL;
    }

    skip_ws_and_comments(f);
    if (fscanf(f, "%d", W) != 1) { fclose(f); return NULL; }

    skip_ws_and_comments(f);
    if (fscanf(f, "%d", H) != 1) { fclose(f); return NULL; }

    skip_ws_and_comments(f);
    int maxval = 0;
    if (fscanf(f, "%d", &maxval) != 1 || maxval != 255) {
        fprintf(stderr, "PGM maxval must be 255 (got %d)\n", maxval);
        fclose(f);
        return NULL;
    }
    fgetc(f);
    size_t n = (size_t)(*W) * (size_t)(*H);
    uint8_t *img = (uint8_t*)malloc(n);
    if (!img) { perror("malloc"); fclose(f); return NULL; }

    if (fread(img, 1, n, f) != n) {
        fprintf(stderr, "Failed reading pixel data\n");
        free(img);
        fclose(f);
        return NULL;
    }

    fclose(f);
    return img;
}

// Writing the binary
static int write_pgm_p5(const char *path, const uint8_t *img, int W, int H) {
    FILE *f = fopen(path, "wb");
    if (!f) { perror("fopen"); return 0; }

    fprintf(f, "P5\n%d %d\n255\n", W, H);

    size_t n = (size_t)W * (size_t)H;
    if (fwrite(img, 1, n, f) != n) { fclose(f); return 0; }

    fclose(f);
    return 1;
}

// ============================================================
// edge kernels (Same code as the previous code block)
// ============================================================

__global__ void k_edge_n3(const uint8_t *image, uint8_t *out, int W, int H) {
    const int32_t K[9] = {
         0, -1,  0,
        -1,  4, -1,
         0, -1,  0
    };
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
            if (ix >= 0 && ix < W && iy >= 0 && iy < H) {
                p = image[iy * W + ix];
            }

            sum += (int32_t)p * K[ky * 3 + kx];
        }
    }

    if (sum < 0) sum = -sum;
    if (sum < 0) sum = 0;
    if (sum > 255) sum = 255;
    out[y * W + x] = (uint8_t)sum;
}

__global__ void k_edge_n5(const uint8_t *image, uint8_t *out, int W, int H) {
    const int32_t K[25] = {
        -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1,
        -1, -1, 24, -1, -1,
        -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1
    };
    int r = 2;

    int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= W || y >= H) return;

    int32_t sum = 0;

    for (int ky = 0; ky < 5; ky++) {
        for (int kx = 0; kx < 5; kx++) {
            int ix = x + (kx - r);
            int iy = y + (ky - r);

            uint8_t p = 0;
            if (ix >= 0 && ix < W && iy >= 0 && iy < H) {
                p = image[iy * W + ix];
            }

            sum += (int32_t)p * K[ky * 5 + kx];
        }
    }

    if (sum < 0) sum = -sum;
    if (sum < 0) sum = 0;
    if (sum > 255) sum = 255;
    out[y * W + x] = (uint8_t)sum;
}

__global__ void k_edge_n7(const uint8_t *image, uint8_t *out, int W, int H) {
    const int32_t K[49] = {
        -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, 48, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1
    };
    int r = 3;

    int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= W || y >= H) return;

    int32_t sum = 0;

    for (int ky = 0; ky < 7; ky++) {
        for (int kx = 0; kx < 7; kx++) {
            int ix = x + (kx - r);
            int iy = y + (ky - r);

            uint8_t p = 0;
            if (ix >= 0 && ix < W && iy >= 0 && iy < H) {
                p = image[iy * W + ix];
            }

            sum += (int32_t)p * K[ky * 7 + kx];
        }
    }

    if (sum < 0) sum = -sum;
    if (sum < 0) sum = 0;
    if (sum > 255) sum = 255;
    out[y * W + x] = (uint8_t)sum;
}

__global__ void k_sharpen_n3(const uint8_t *image, uint8_t *out, int W, int H) {
    const int32_t K[9] = {
        -1, -1, -1,
        -1,  9, -1,
        -1, -1, -1
    };
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
            if (ix >= 0 && ix < W && iy >= 0 && iy < H) {
                p = image[iy * W + ix];
            }

            sum += (int32_t)p * K[ky * 3 + kx];
        }
    }

    if (sum < 0) sum = 0;
    if (sum > 255) sum = 255;
    out[y * W + x] = (uint8_t)sum;
}

__global__ void k_sharpen_n5(const uint8_t *image, uint8_t *out, int W, int H) {
    const int32_t K[25] = {
        -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1,
        -1, -1, 25, -1, -1,
        -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1
    };
    int r = 2;

    int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= W || y >= H) return;

    int32_t sum = 0;

    for (int ky = 0; ky < 5; ky++) {
        for (int kx = 0; kx < 5; kx++) {
            int ix = x + (kx - r);
            int iy = y + (ky - r);

            uint8_t p = 0;
            if (ix >= 0 && ix < W && iy >= 0 && iy < H) {
                p = image[iy * W + ix];
            }

            sum += (int32_t)p * K[ky * 5 + kx];
        }
    }

    if (sum < 0) sum = 0;
    if (sum > 255) sum = 255;
    out[y * W + x] = (uint8_t)sum;
}

__global__ void k_sharpen_n7(const uint8_t *image, uint8_t *out, int W, int H) {
    const int32_t K[49] = {
        -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, 49, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1
    };
    int r = 3;

    int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= W || y >= H) return;

    int32_t sum = 0;

    for (int ky = 0; ky < 7; ky++) {
        for (int kx = 0; kx < 7; kx++) {
            int ix = x + (kx - r);
            int iy = y + (ky - r);

            uint8_t p = 0;
            if (ix >= 0 && ix < W && iy >= 0 && iy < H) {
                p = image[iy * W + ix];
            }

            sum += (int32_t)p * K[ky * 7 + kx];
        }
    }

    if (sum < 0) sum = 0;
    if (sum > 255) sum = 255;
    out[y * W + x] = (uint8_t)sum;
}

__global__ void k_blur_n3(const uint8_t *image, uint8_t *out, int W, int H) {
    const int32_t K[9] = {
        1, 2, 1,
        2, 4, 2,
        1, 2, 1
    };
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
            if (ix >= 0 && ix < W && iy >= 0 && iy < H) {
                p = image[iy * W + ix];
            }

            sum += (int32_t)p * K[ky * 3 + kx];
        }
    }

    sum /= 16;
    if (sum < 0) sum = 0;
    if (sum > 255) sum = 255;
    out[y * W + x] = (uint8_t)sum;
}

__global__ void k_blur_n5(const uint8_t *image, uint8_t *out, int W, int H) {
    const int32_t K[25] = {
         1,  4,  6,  4,  1,
         4, 16, 24, 16,  4,
         6, 24, 36, 24,  6,
         4, 16, 24, 16,  4,
         1,  4,  6,  4,  1
    };
    int r = 2;

    int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= W || y >= H) return;

    int32_t sum = 0;

    for (int ky = 0; ky < 5; ky++) {
        for (int kx = 0; kx < 5; kx++) {
            int ix = x + (kx - r);
            int iy = y + (ky - r);

            uint8_t p = 0;
            if (ix >= 0 && ix < W && iy >= 0 && iy < H) {
                p = image[iy * W + ix];
            }

            sum += (int32_t)p * K[ky * 5 + kx];
        }
    }

    sum /= 256;
    if (sum < 0) sum = 0;
    if (sum > 255) sum = 255;
    out[y * W + x] = (uint8_t)sum;
}

__global__ void k_blur_n7(const uint8_t *image, uint8_t *out, int W, int H) {
    const int32_t K[49] = {
          1,   6,  15,  20,  15,   6,   1,
          6,  36,  90, 120,  90,  36,   6,
         15,  90, 225, 300, 225,  90,  15,
         20, 120, 300, 400, 300, 120,  20,
         15,  90, 225, 300, 225,  90,  15,
          6,  36,  90, 120,  90,  36,   6,
          1,   6,  15,  20,  15,   6,   1
    };
    int r = 3;

    int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= W || y >= H) return;

    int32_t sum = 0;

    for (int ky = 0; ky < 7; ky++) {
        for (int kx = 0; kx < 7; kx++) {
            int ix = x + (kx - r);
            int iy = y + (ky - r);

            uint8_t p = 0;
            if (ix >= 0 && ix < W && iy >= 0 && iy < H) {
                p = image[iy * W + ix];
            }

            sum += (int32_t)p * K[ky * 7 + kx];
        }
    }

    sum /= 4096;
    if (sum < 0) sum = 0;
    if (sum > 255) sum = 255;
    out[y * W + x] = (uint8_t)sum;
}


// ===============================================================
// main-CUDA only!!
// ================================================================
int main(int argc, char **argv) {
    if (argc != 4 && argc != 5) {
        fprintf(stderr, "Usage: %s input.pgm output.pgm <mode> [M]\n", argv[0]);
        fprintf(stderr, "Modes:\n");
        fprintf(stderr, "  edge_n3 edge_n5 edge_n7\n");
        fprintf(stderr, "  sharpen_n3 sharpen_n5 sharpen_n7\n");
        fprintf(stderr, "  blur_n3 blur_n5 blur_n7\n");
        fprintf(stderr, "Example: %s input.pgm out.pgm edge_n5 256\n", argv[0]);
        return 1;
    }

    const char *mode = argv[3];

    int M = DEFAULT_M;
    if (argc == 5) {
        M = atoi(argv[4]);
        if (M <= 0) {
            fprintf(stderr, "Error: M must be positive (got %s)\n", argv[4]);
            return 1;
        }
    }

    int W0 = 0, H0 = 0;
    uint8_t *img0 = read_pgm_p5(argv[1], &W0, &H0);
    if (!img0) return 1;

    int max_square = (W0 < H0) ? W0 : H0;
    if (M > max_square) {
        fprintf(stderr, "Warning: requested M=%d but input is %dx%d; using M=%d\n",
                M, W0, H0, max_square);
        M = max_square;
    }

    size_t n = (size_t)M * (size_t)M;
    uint8_t *img = (uint8_t*)malloc(n);
    uint8_t *out = (uint8_t*)malloc(n);
    if (!img || !out) {
        perror("malloc");
        free(img0);
        free(img);
        free(out);
        return 1;
    }

    for (int y = 0; y < M; y++) {
        memcpy(&img[y * M], &img0[y * W0], (size_t)M);
    }
    free(img0);
    uint8_t *d_img = NULL;
    uint8_t *d_out = NULL;

    cudaError_t err = cudaMalloc((void**)&d_img, n);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc d_img failed: %s\n", cudaGetErrorString(err)); return 1; }

    err = cudaMalloc((void**)&d_out, n);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc d_out failed: %s\n", cudaGetErrorString(err)); return 1; }

    err = cudaMemcpy(d_img, img, n, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMemcpy H2D failed: %s\n", cudaGetErrorString(err)); return 1; }

    dim3 block(16, 16);
    dim3 grid((M + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    // Capturing the timing for CUDA.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    if (strcmp(mode, "edge_n3") == 0) k_edge_n3<<<grid, block>>>(d_img, d_out, M, M);
    else if (strcmp(mode, "edge_n5") == 0) k_edge_n5<<<grid, block>>>(d_img, d_out, M, M);
    else if (strcmp(mode, "edge_n7") == 0) k_edge_n7<<<grid, block>>>(d_img, d_out, M, M);

    else if (strcmp(mode, "sharpen_n3") == 0) k_sharpen_n3<<<grid, block>>>(d_img, d_out, M, M);
    else if (strcmp(mode, "sharpen_n5") == 0) k_sharpen_n5<<<grid, block>>>(d_img, d_out, M, M);
    else if (strcmp(mode, "sharpen_n7") == 0) k_sharpen_n7<<<grid, block>>>(d_img, d_out, M, M);

    else if (strcmp(mode, "blur_n3") == 0) k_blur_n3<<<grid, block>>>(d_img, d_out, M, M);
    else if (strcmp(mode, "blur_n5") == 0) k_blur_n5<<<grid, block>>>(d_img, d_out, M, M);
    else if (strcmp(mode, "blur_n7") == 0) k_blur_n7<<<grid, block>>>(d_img, d_out, M, M);

    else {
        fprintf(stderr, "Unknown mode '%s'\n", mode);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_img); cudaFree(d_out);
        free(img); free(out);
        return 1;
    }

    cudaEventRecord(stop, 0);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_img); cudaFree(d_out);
        free(img); free(out);
        return 1;
    }

    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    int N = (strstr(mode, "_n3") ? 3 : (strstr(mode, "_n5") ? 5 : 7));
    printf("CUDA kernel time (events): mode=%s, N=%d, M=%d -> %.3f ms\n", mode, N, M, ms);

    err = cudaMemcpy(out, d_out, n, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy D2H failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_img); cudaFree(d_out);
        free(img); free(out);
        return 1;
    }

    // Writing the final output:
    if (!write_pgm_p5(argv[2], out, M, M)) {
        fprintf(stderr, "Failed writing output\n");
    }

    cudaFree(d_img);
    cudaFree(d_out);
    free(img);
    free(out);
    return 0;
}

