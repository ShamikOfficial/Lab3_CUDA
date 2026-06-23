# CUDA-Accelerated Python Library

A CUDA-focused project for benchmarking matrix multiplication and image convolution workflows across CPU and GPU implementations, with notebook-driven analysis and reproducible outputs.

## Project Goals

- Benchmark CPU vs CUDA performance for core numerical workloads.
- Provide executable C/CUDA implementations for image convolution.
- Preserve reproducible benchmarking artifacts and visual outputs.
- Document methodology and results in a final project report.

## Repository Layout

```text
CUDA-Accelerated-Python-Library/
├── native_image_convolution_sources/
│   ├── image_convolution_cpu.c
│   ├── image_convolution_cuda.cu
│   └── image_convolution_cuda_python_bridge.cu
├── benchmarks/
│   └── image_convolution/
│       ├── case_01/
│       │   ├── src/
│       │   ├── notebooks/
│       │   ├── data/input/
│       │   ├── results/final/
│       │   ├── results/intermediate/
│       │   └── artifacts/
│       ├── case_02/
│       └── case_03/
├── notebooks/
│   └── matrix_multiplication/
│       └── matrix_multiplication_benchmark.ipynb
├── docs/
│   └── report/
│       ├── CUDA_Project_Report_Final.pdf
│       └── CUDA_Project_Report_Final.docx
├── README.md
└── LICENSE
```

## Workload Modules

### Matrix Multiplication

- Notebook: `notebooks/matrix_multiplication/matrix_multiplication_benchmark.ipynb`
- Covers CPU, naive CUDA, optimized CUDA, and cuBLAS comparisons.
- Includes timing, plotting, and performance analysis.

### Image Convolution

Each benchmark case (`case_01`, `case_02`, `case_03`) is organized as:

- `native_image_convolution_sources/` (repository root) - canonical source files used by notebooks:
  - `image_convolution_cpu.c`
  - `image_convolution_cuda.cu`
  - `image_convolution_cuda_python_bridge.cu`
- `src/` - local case source snapshots
- `notebooks/` - `image_convolution_benchmark.ipynb`
- `data/input/` - `input.pgm`, `image.jpg`
- `results/final/` - final output images (`out_*`)
- `results/intermediate/` - temporary benchmark outputs (`tmp_*`)
- `artifacts/bin` and `artifacts/lib` - compiled executables and shared libraries

## Build and Run

From any convolution case directory, for example `benchmarks/image_convolution/case_01`:

### CPU Build

```bash
gcc ../../../native_image_convolution_sources/image_convolution_cpu.c -O2 -o artifacts/bin/image_convolution_cpu
```

### CUDA Build

```bash
nvcc -O2 -arch=sm_75 ../../../native_image_convolution_sources/image_convolution_cuda.cu -o artifacts/bin/image_convolution_cuda
```

### Python-Interop Shared Library

```bash
nvcc -Xcompiler -fPIC -shared ../../../native_image_convolution_sources/image_convolution_cuda_python_bridge.cu -o artifacts/lib/libimage_convolution.so
```

### Sample Execution

```bash
./artifacts/bin/image_convolution_cpu data/input/input.pgm results/final/out_edge_n5_512.pgm edge_n5 512
./artifacts/bin/image_convolution_cuda data/input/input.pgm results/final/out_edge_n5_cuda_512.pgm edge_n5 512
```

## Environment

- NVIDIA GPU with CUDA support
- CUDA toolkit (`nvcc`)
- GCC
- Python 3 + Jupyter for notebook workflows

The benchmark configuration in this project is aligned with Tesla T4-compatible compilation (`sm_75`), as used in the report workflow.

## Report

Project report files are available in `docs/report/`:

- `CUDA_Project_Report_Final.pdf`
- `CUDA_Project_Report_Final.docx`

## License

This project is licensed under the MIT License. See `LICENSE` for details.
