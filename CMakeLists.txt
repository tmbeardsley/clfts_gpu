cmake_minimum_required(VERSION 3.18)
project(CLFTS-GPU LANGUAGES CXX CUDA)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Specify the GPU architectures to compile for
set(CMAKE_CUDA_ARCHITECTURES all)

# Find required packages
find_package(CUDAToolkit REQUIRED)

# Add the source file for the CUDA code
add_executable(clfts-gpu ./src/clfts_gpu_thrust.cu)

# Link required libraries (cuFFT, cuRAND)
target_link_libraries(clfts-gpu
    CUDA::cufft       # CUDA FFT library
    CUDA::curand      # CUDA random number generation library
)