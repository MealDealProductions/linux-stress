#!/bin/bash

# Variable to keep track of the GPU stress test process ID
gpu_stress_pid=""



# Function to run CPU stress test
function run_cpu_stress_test {
  local cpu_cores=$(nproc)
  local cpu_stress_level="$1"
  echo "Running CPU stress test on $cpu_cores cores with stress level: $cpu_stress_level%..."
  stress-ng --cpu $cpu_cores --cpu-load $cpu_stress_level
}

# Function to run RAM stress test
function run_ram_stress_test {
  local total_ram_mb=$(free -m | awk 'FNR == 2 {print $2}')
  local ram_stress_level="$1"
  local ram_stress_bytes=$(($total_ram_mb * $ram_stress_level / 100))
  echo "Running RAM stress test with all available RAM ($total_ram_mb MB) and stress level: $ram_stress_level%..."
  stress-ng --vm 1 --vm-bytes ${ram_stress_bytes}M
}

# Function to run GPU stress test (using CUDA or OpenGL)
function run_gpu_stress_test {
  local gpu_stress_duration=0  # Set to 0 for continuous stress test
  local gpu_stress_level="$1"
  local use_opengl="$2"

  echo "Running GPU stress test..."
  if [ "$use_opengl" = true ]; then
    echo "Using OpenGL for GPU stress test..."
    glmark2 --run-forever --benchmark 0 &
    gpu_stress_pid=$!
  else
    echo "Using CUDA for GPU stress test..."

    # Check if CUDA is installed and the GPU is available
    if command -v nvcc &>/dev/null; then
      # CUDA code for the stress test
      cuda_code=$(cat <<EOF
#include <cuda_runtime.h>
#include <iostream>

__global__ void matrixMultiply(int N, float* A, float* B, float* C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * N) {
        int row = idx / N;
        int col = idx % N;
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[idx] = sum;
    }
}

int main(int argc, char** argv) {
    int N = 1024; // Matrix size: N x N

    float* A;
    float* B;
    float* C;

    size_t matrixSize = N * N * sizeof(float);
    cudaMalloc((void**)&A, matrixSize);
    cudaMalloc((void**)&B, matrixSize);
    cudaMalloc((void**)&C, matrixSize);

    // Initialize matrices A and B with random data
    cudaMemcpy(A, new float[N * N], matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(B, new float[N * N], matrixSize, cudaMemcpyHostToDevice);

    // Launch kernel to stress the GPU
    dim3 threadsPerBlock(256);
    dim3 numBlocks((N * N + threadsPerBlock.x - 1) / threadsPerBlock.x);

    while (true) {
        matrixMultiply<<<numBlocks, threadsPerBlock>>>(N, A, B, C);
    }

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}
EOF
)
      echo "$cuda_code" >gpu_stress_test.cu
      nvcc gpu_stress_test.cu -o gpu_stress_test
      ./gpu_stress_test &
      gpu_stress_pid=$!
      rm gpu_stress_test gpu_stress_test.cu
    else
      echo "CUDA is not installed on your system, or the GPU is not available."
      echo "Please install CUDA and ensure that the GPU is properly set up."
      return 1
    fi
  fi
}

# Function to stop GPU stress test
function stop_gpu_stress_test {
  if [ -n "$gpu_stress_pid" ]; then
    echo "Stopping GPU stress test..."
    kill $gpu_stress_pid
    wait $gpu_stress_pid 2>/dev/null
    gpu_stress_pid=""
  fi
}

# Function to stop stress tests and clean up
function stop_stress_tests {
  echo "Stopping stress tests..."
  pkill stress-ng  # Stop CPU and RAM stress tests
  stop_gpu_stress_test  # Stop GPU stress test
  pkill glmark2   # Stop OpenGL-based GPU stress test
}

# Trap to stop stress tests on script termination
trap stop_stress_tests SIGINT SIGTERM


# Show menu
while true; do
  echo -e "\nSelect the stress test to run:"
  echo "1. CPU Stress Test"
  echo "2. RAM Stress Test"
  echo "3. GPU Stress Test (CUDA)"
  echo "4. GPU Stress Test (OpenGL)"
  echo "5. Run All Tests (CPU, RAM, CUDA GPU)"
  echo "6. Run All Tests (CPU, RAM, OpenGL GPU)"
  echo "7. Stop All Tests and Exit"

  read -p "Enter your choice (1/2/3/4/5/6/7): " choice

  case $choice in
    1)
      read -p "Enter CPU stress level (0-100): " cpu_stress_level
      run_cpu_stress_test $cpu_stress_level &
      ;;

    2)
      read -p "Enter RAM stress level (0-100): " ram_stress_level
      run_ram_stress_test $ram_stress_level &
      ;;
    3)
      read -p "Enter GPU stress level (0-100): " gpu_stress_level
      run_gpu_stress_test $gpu_stress_level false &
      ;;
    4)
      read -p "Enter GPU stress level (0-100): " gpu_stress_level
      run_gpu_stress_test $gpu_stress_level true &
      ;;
    5)
      read -p "Enter CPU stress level (0-100): " cpu_stress_level
      read -p "Enter RAM stress level (0-100): " ram_stress_level
      read -p "Enter GPU stress level (0-100): " gpu_stress_level

      run_cpu_stress_test $cpu_stress_level &  # Run CPU stress test in the background
      run_ram_stress_test $ram_stress_level &  # Run RAM stress test in the background
      run_gpu_stress_test $gpu_stress_level false &  # Run CUDA GPU stress test in the background
      ;;
    6)
      read -p "Enter CPU stress level (0-100): " cpu_stress_level
      read -p "Enter RAM stress level (0-100): " ram_stress_level
      read -p "Enter GPU stress level (0-100): " gpu_stress_level

      run_cpu_stress_test $cpu_stress_level &  # Run CPU stress test in the background
      run_ram_stress_test $ram_stress_level &  # Run RAM stress test in the background
      run_gpu_stress_test $gpu_stress_level true &  # Run OpenGL GPU stress test in the background
      ;;
    7)
      stop_stress_tests
      echo "Exiting stress test script."
      exit 0
      ;;
    *)
      echo "Invalid choice. Please select a valid option."
      ;;
  esac
done
