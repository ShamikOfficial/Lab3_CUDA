# Custom Python Library using CUDA

## Overview

This laboratory implements matrix multiplication and image convolution operations using CPU, CUDA, and optimized GPU implementations. The project demonstrates performance comparisons between different approaches and provides Python bindings through shared libraries.

## Prerequisites

This lab was developed and tested on **Google Colab** with **Tesla T4 GPU**.

### Google Colab (Recommended)
- **Google Colab** with GPU runtime (Tesla T4)
- CUDA Toolkit pre-installed
- GCC compiler available
- Python 3 with NumPy

## Setup Instructions

### Google Colab Setup

1. Upload the project files to Google Colab
2. Enable GPU runtime: Runtime → Change runtime type → Hardware accelerator: GPU
3. Verify GPU availability:
   ```python
   !nvidia-smi
   ```
4. Verify CUDA compiler:
   ```python
   !nvcc --version
   ```
5. Open the notebook (`Lab3_CUDA.ipynb` for Part 1 or `Lab3_Part2.ipynb` for Part 2)

**Note:** Tesla T4 GPU has Compute Capability 7.5, which matches the compilation flags used (`-arch=sm_75`).

## Project Structure

```
lab_3/
├── part_1/                      # Matrix Multiplication Implementations
│   └── Lab3_CUDA.ipynb          # Main Jupyter notebook with all Part 1 code
└── part_2/                      # Image Convolution Implementations
    ├── image_1/
    │   ├── Lab3_Part2.ipynb     # Main notebook for image_1
    │   ├── input.pgm            # Input image
    │   ├── convo.c              # CPU convolution implementation
    │   ├── convo_cuda.cu        # CUDA convolution implementation
    │   ├── convo_cuda_lib.cu    # CUDA shared library for Python
    │   ├── libconvo.so          # Compiled shared library
    │   └── out_*.pgm            # Generated output images
    ├── image_2/
    │   ├── Lab3_Part2.ipynb     # Main notebook for image_2
    │   ├── input.pgm            # Input image
    │   └── ...                  # Similar structure
    └── image_3/
        ├── Lab3_Part2.ipynb     # Main notebook for image_3
        ├── input.pgm            # Input image
        └── ...                  # Similar structure
```

---

## Part 1: Matrix Multiplication

### Step 1.1: CPU Implementation

**File:** `part_1/matrix_cpu.c`

Standard triple-loop matrix multiplication on CPU.

**Compilation:**
```bash
gcc matrix_cpu.c -o matrix_cpu -O2
```

**Execution:**
```bash
./matrix_cpu 512
./matrix_cpu 1024
./matrix_cpu 2048
```

### Step 2.1: Naïve CUDA Implementation

**File:** `part_1/matrix_naive_gpu.cu`

Each thread computes one element of the output matrix using global memory.

**Compilation:**
```bash
nvcc -O2 -arch=sm_75 matrix_naive_gpu.cu -o matrix_naive_gpu
```

**Execution:**
```bash
./matrix_naive_gpu 512
./matrix_naive_gpu 1024
./matrix_naive_gpu 2048
```

### Step 4.1: Optimized CUDA Implementation (Tiled)

**File:** `part_1/matrix_optimizing_gpu.cu`

Uses shared memory tiling (TILE_WIDTH=16) to reduce global memory accesses.

**Compilation:**
```bash
nvcc -O2 -arch=sm_75 matrix_optimizing_gpu.cu -o matrix_optimizing_gpu
```

**Execution:**
```bash
./matrix_optimizing_gpu 512
./matrix_optimizing_gpu 1024
./matrix_optimizing_gpu 2048
```

### Step 6: cuBLAS Implementation

**File:** `part_1/matrix_cublas_gpu.cu`

Uses NVIDIA's cuBLAS library for optimized matrix multiplication.

**Compilation:**
```bash
nvcc -O2 -arch=sm_75 matrix_cublas_gpu.cu -o matrix_cublas_gpu -lcublas
```

**Execution:**
```bash
./matrix_cublas_gpu 512
./matrix_cublas_gpu 1024
./matrix_cublas_gpu 2048
```

### Step 7: Shared Library for Python

**File:** `part_1/matrix_lib.cu`

Compiles optimized CUDA kernel into a shared library callable from Python.

**Compilation:**
```bash
nvcc -Xcompiler -fPIC -shared matrix_lib.cu -o libmatrix.so
```

---

## Part 2: Image Convolution

Each image folder (`image_1`, `image_2`, `image_3`) contains a complete implementation with:
- **Main Notebook:** `Lab3_Part2.ipynb` - Contains all code, compilation, execution, and performance analysis
- **Input Image:** `input.pgm` - Binary PGM format image for processing
- **Source Files:** `convo.c`, `convo_cuda.cu`, `convo_cuda_lib.cu`
- **Output Images:** Generated `.pgm` files with results from different filters and sizes

### Running Part 2

Open the Jupyter notebook in any image folder:
```bash
cd lab_3/part_2/image_1
jupyter notebook Lab3_Part2.ipynb
```

The notebook contains:
- Code generation for CPU and CUDA implementations
- Compilation instructions
- Performance measurements
- Image processing with different filters (edge detection, sharpening, blur)
- Comparison between CPU and CUDA implementations

### Step 1: CPU Convolution Implementation

**File:** `part_2/image_X/convo.c`

Implements convolution operations for image processing with three filter types:
- **Edge Detection** (n3, n5, n7)
- **Sharpening** (n3, n5, n7)
- **Gaussian Blur** (n3, n5, n7)

**Compilation:**
```bash
gcc convo.c -o convo -O2
```

**Execution:**
```bash
./convo input.pgm output.pgm <mode> [M]
```

**Modes:**
- `edge_n3`, `edge_n5`, `edge_n7`
- `sharpen_n3`, `sharpen_n5`, `sharpen_n7`
- `blur_n3`, `blur_n5`, `blur_n7`

**Example:**
```bash
./convo input.pgm out_edge.pgm edge_n5 512
```

### Step 2: CUDA Convolution Implementation

**File:** `part_2/image_X/convo_cuda.cu`

GPU-accelerated convolution using CUDA kernels. Each thread processes one output pixel.

**Compilation:**
```bash
nvcc -O2 -arch=sm_75 convo_cuda.cu -o convo_cuda
```

**Execution:**
```bash
./convo_cuda input.pgm output.pgm <mode> [M]
```

**Example:**
```bash
cd lab_3/part_2/image_1
./convo_cuda input.pgm out_blur.pgm blur_n3 1024
```

### Step 3: Shared Library for Python

**File:** `part_2/image_X/convo_cuda_lib.cu`

CUDA convolution functions compiled as a shared library for Python integration.

**Compilation:**
```bash
nvcc -Xcompiler -fPIC -shared convo_cuda_lib.cu -o libconvo.so
```

**Note:** The main workflow for Part 2 is through the `Lab3_Part2.ipynb` notebook, which automates compilation, execution, and performance analysis.

---

## Performance Measurement

### Matrix Multiplication

Measure execution times for different matrix sizes (N = 512, 1024, 2048, etc.) and compare:
- CPU (C)
- Naïve CUDA
- Optimized CUDA (Tiled)
- cuBLAS

**Expected Results:**
- CPU performance degrades quadratically with matrix size
- GPU implementations show better scaling
- cuBLAS provides optimal performance for large matrices
- Tiling optimization improves performance over naïve CUDA for medium to large matrices

### Image Convolution

Measure execution times for different image sizes (M = 256, 512, 1024) and filter sizes (N = 3, 5, 7).

**Performance Comparison:**
- CPU convolution time increases with image size and filter size
- CUDA acceleration provides significant speedup for larger images
- Overhead of GPU memory transfer becomes negligible for larger problem sizes

