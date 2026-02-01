#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <ctype.h>
#include <string.h>
#include <time.h>

#define DEFAULT_M 512

static inline uint8_t clamp_to_u8(int32_t v) {
    if (v < 0) return 0;
    if (v > 255) return 255;
    return (uint8_t)v;
}

static void skip_ws_and_comments(FILE *f) {
    int c;
    while ((c = fgetc(f)) != EOF) {
        if (isspace(c)) continue;
        if (c == '#') { while ((c = fgetc(f)) != EOF && c != '\n') {} continue; }
        ungetc(c, f);
        return;
    }
}

// Zero-padding pixel read for boundary
static uint8_t get_pixel_zero_pad(const uint8_t *img, int W, int H, int x, int y) {
    if (x < 0 || x >= W || y < 0 || y >= H) return 0;
    return img[y * W + x];
}

// Readindg the binary, maxval=255
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
// edge kernels
// ============================================================

// EDGE N=3:
static void edge_n3(const uint8_t *image, uint8_t *out, int W, int H) {
    const int32_t K[9] = {
         0, -1,  0,
        -1,  4, -1,
         0, -1,  0
    };
    int r = 1;

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int32_t sum = 0;

            for (int ky = 0; ky < 3; ky++) {
                for (int kx = 0; kx < 3; kx++) {
                    int ix = x + (kx - r);
                    int iy = y + (ky - r);
                    uint8_t p = get_pixel_zero_pad(image, W, H, ix, iy);
                    sum += (int32_t)p * K[ky * 3 + kx];
                }
            }

            if (sum < 0) sum = -sum;
            out[y * W + x] = clamp_to_u8(sum);
        }
    }
}

// EDGE N=5: (similar logic with N=3, high center and negative corners)
static void edge_n5(const uint8_t *image, uint8_t *out, int W, int H) {
    const int32_t K[25] = {
        -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1,
        -1, -1, 24, -1, -1,
        -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1
    };
    int r = 2;

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int32_t sum = 0;

            for (int ky = 0; ky < 5; ky++) {
                for (int kx = 0; kx < 5; kx++) {
                    int ix = x + (kx - r);
                    int iy = y + (ky - r);
                    uint8_t p = get_pixel_zero_pad(image, W, H, ix, iy);
                    sum += (int32_t)p * K[ky * 5 + kx];
                }
            }

            if (sum < 0) sum = -sum;
            out[y * W + x] = clamp_to_u8(sum);
        }
    }
}

// EDGE N=7: (same thing as other edge detections, big center and -1 edges/corners)
static void edge_n7(const uint8_t *image, uint8_t *out, int W, int H) {
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

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int32_t sum = 0;

            for (int ky = 0; ky < 7; ky++) {
                for (int kx = 0; kx < 7; kx++) {
                    int ix = x + (kx - r);
                    int iy = y + (ky - r);
                    uint8_t p = get_pixel_zero_pad(image, W, H, ix, iy);
                    sum += (int32_t)p * K[ky * 7 + kx];
                }
            }

            if (sum < 0) sum = -sum;
            out[y * W + x] = clamp_to_u8(sum);
        }
    }
}

// ============================================================
// Sharpen kernels
// ============================================================

static void sharpen_n3(const uint8_t *image, uint8_t *out, int W, int H) {
    const int32_t K[9] = {
        -1, -1, -1,
        -1,  9, -1,
        -1, -1, -1
    };
    int r = 1;

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int32_t sum = 0;

            for (int ky = 0; ky < 3; ky++) {
                for (int kx = 0; kx < 3; kx++) {
                    int ix = x + (kx - r);
                    int iy = y + (ky - r);
                    uint8_t p = get_pixel_zero_pad(image, W, H, ix, iy);
                    sum += (int32_t)p * K[ky * 3 + kx];
                }
            }

            out[y * W + x] = clamp_to_u8(sum);
        }
    }
}

static void sharpen_n5(const uint8_t *image, uint8_t *out, int W, int H) {
    const int32_t K[25] = {
        -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1,
        -1, -1, 25, -1, -1,
        -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1
    };
    int r = 2;

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int32_t sum = 0;

            for (int ky = 0; ky < 5; ky++) {
                for (int kx = 0; kx < 5; kx++) {
                    int ix = x + (kx - r);
                    int iy = y + (ky - r);
                    uint8_t p = get_pixel_zero_pad(image, W, H, ix, iy);
                    sum += (int32_t)p * K[ky * 5 + kx];
                }
            }

            out[y * W + x] = clamp_to_u8(sum);
        }
    }
}

static void sharpen_n7(const uint8_t *image, uint8_t *out, int W, int H) {
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

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int32_t sum = 0;

            for (int ky = 0; ky < 7; ky++) {
                for (int kx = 0; kx < 7; kx++) {
                    int ix = x + (kx - r);
                    int iy = y + (ky - r);
                    uint8_t p = get_pixel_zero_pad(image, W, H, ix, iy);
                    sum += (int32_t)p * K[ky * 7 + kx];
                }
            }

            out[y * W + x] = clamp_to_u8(sum);
        }
    }
}

// ============================================================
// Gaussian blur kernelss
// ============================================================

static void blur_n3(const uint8_t *image, uint8_t *out, int W, int H) {
    const int32_t K[9] = {
        1, 2, 1,
        2, 4, 2,
        1, 2, 1
    };
    int r = 1;

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int32_t sum = 0;

            for (int ky = 0; ky < 3; ky++) {
                for (int kx = 0; kx < 3; kx++) {
                    int ix = x + (kx - r);
                    int iy = y + (ky - r);
                    uint8_t p = get_pixel_zero_pad(image, W, H, ix, iy);
                    sum += (int32_t)p * K[ky * 3 + kx];
                }
            }

            sum /= 16; // normalization with dividing by 16 (from wikipedia page)
            out[y * W + x] = clamp_to_u8(sum);
        }
    }
}

static void blur_n5(const uint8_t *image, uint8_t *out, int W, int H) {
    const int32_t K[25] = {
         1,  4,  6,  4,  1,
         4, 16, 24, 16,  4,
         6, 24, 36, 24,  6,
         4, 16, 24, 16,  4,
         1,  4,  6,  4,  1
    };
    int r = 2;

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int32_t sum = 0;

            for (int ky = 0; ky < 5; ky++) {
                for (int kx = 0; kx < 5; kx++) {
                    int ix = x + (kx - r);
                    int iy = y + (ky - r);
                    uint8_t p = get_pixel_zero_pad(image, W, H, ix, iy);
                    sum += (int32_t)p * K[ky * 5 + kx];
                }
            }

            sum /= 256; // normalization
            out[y * W + x] = clamp_to_u8(sum);
        }
    }
}

static void blur_n7(const uint8_t *image, uint8_t *out, int W, int H) {
    // Got the values from internet. Sum is equal to 4096.
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

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int32_t sum = 0;

            for (int ky = 0; ky < 7; ky++) {
                for (int kx = 0; kx < 7; kx++) {
                    int ix = x + (kx - r);
                    int iy = y + (ky - r);
                    uint8_t p = get_pixel_zero_pad(image, W, H, ix, iy);
                    sum += (int32_t)p * K[ky * 7 + kx];
                }
            }

            sum /= 4096; // normalization
            out[y * W + x] = clamp_to_u8(sum);
        }
    }
}

// ============================================================
// main
// ============================================================

int main(int argc, char **argv) {

    const char *mode = argv[3];

    int M = DEFAULT_M;
    if (argc == 5) {
        M = atoi(argv[4]);
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

    // Copying the top-left MxM into the buffer
    uint8_t *img = (uint8_t*)malloc(n);
    if (!img) { perror("malloc"); free(img0); return 1; }

    uint8_t *out = (uint8_t*)malloc(n);
    if (!out) { perror("malloc"); free(img0); free(img); return 1; }

    for (int y = 0; y < M; y++) {
        memcpy(&img[y * M], &img0[y * W0], (size_t)M);
    }

    clock_t start = clock();

    if (strcmp(mode, "edge_n3") == 0) edge_n3(img, out, M, M);
    else if (strcmp(mode, "edge_n5") == 0) edge_n5(img, out, M, M);
    else if (strcmp(mode, "edge_n7") == 0) edge_n7(img, out, M, M);

    else if (strcmp(mode, "sharpen_n3") == 0) sharpen_n3(img, out, M, M);
    else if (strcmp(mode, "sharpen_n5") == 0) sharpen_n5(img, out, M, M);
    else if (strcmp(mode, "sharpen_n7") == 0) sharpen_n7(img, out, M, M);

    else if (strcmp(mode, "blur_n3") == 0) blur_n3(img, out, M, M);
    else if (strcmp(mode, "blur_n5") == 0) blur_n5(img, out, M, M);
    else if (strcmp(mode, "blur_n7") == 0) blur_n7(img, out, M, M);

    else {
        fprintf(stderr, "Unknown mode '%s'\n", mode);
        free(img0); free(img); free(out);
        return 1;
    }

    clock_t end = clock();
    double ms = 1000.0 * (double)(end - start) / (double)CLOCKS_PER_SEC;

    int N = (strstr(mode, "_n3") ? 3 : (strstr(mode, "_n5") ? 5 : 7));
    printf("Compute time (convolution only): mode=%s, N=%d, M=%d -> %.3f ms\n", mode, N, M, ms);

    if (!write_pgm_p5(argv[2], out, M, M)) {
        fprintf(stderr, "Failed writing output\n");
        free(img0); free(img); free(out);
        return 1;
    }

    free(img0);
    free(img);
    free(out);
    return 0;
}
