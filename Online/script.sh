#!/bin/bash

# Variable to keep track of the GPU stress test process ID
gpu_stress_pid=""

# Function to install required dependencies
function install_dependencies {
  echo "Installing required dependencies..."
  sudo apt-get update
  sudo apt-get install stress-ng glmark2 -y
  # Add installation commands for CUDA if necessary
}

# Function to display a colored message
function colored_message {
  local color_code=$1
  local message=$2
  echo -e "${color_code}${message}${Color_Off}"
}

# Colors for formatting
Color_Off='\033[0m'
Green='\033[0;32m'
Blue='\033[0;34m'
Cyan='\033[0;36m'
Yellow='\033[1;33m'

# Function to display the main menu
function display_menu {
  clear
  echo -e "${Cyan}┌──────────────────────────────────┐"
  echo -e "│          Stress Test Menu        │"
  echo -e "└──────────────────────────────────┘${Color_Off}"
  echo -e "${Green}1. ${Color_Off}CPU Stress Test"
  echo -e "${Green}2. ${Color_Off}RAM Stress Test"
  echo -e "${Green}3. ${Color_Off}GPU Stress Test (CUDA)"
  echo -e "${Green}4. ${Color_Off}GPU Stress Test (OpenGL)"
  echo -e "${Green}5. ${Color_Off}Run All Tests (CPU, RAM, CUDA GPU)"
  echo -e "${Green}6. ${Color_Off}Run All Tests (CPU, RAM, OpenGL GPU)"
  echo -e "${Yellow}7. ${Color_Off}Stop All Tests and Exit"
}

# Function to get user input with color
function get_colored_input {
  local prompt="$1"
  local color_code="$2"
  local input
  echo -ne "${color_code}${prompt}${Color_Off}"
  read input
  echo "$input"
}


# ... (Rest of the script remains unchanged)

# Function to run CPU stress test
function run_cpu_stress_test {
  local cpu_cores=$(nproc)
  local cpu_stress_level="$1"
  colored_message $Blue "Running CPU stress test on $cpu_cores cores with stress level: $cpu_stress_level%..."
  stress-ng --cpu $cpu_cores --cpu-load $cpu_stress_level
}

# Function to run RAM stress test
function run_ram_stress_test {
  local total_ram_mb=$(free -m | awk 'FNR == 2 {print $2}')
  local ram_stress_level="$1"
  local ram_stress_bytes=$(($total_ram_mb * $ram_stress_level / 100))
  colored_message $Blue "Running RAM stress test with all available RAM ($total_ram_mb MB) and stress level: $ram_stress_level%..."
  stress-ng --vm 1 --vm-bytes ${ram_stress_bytes}M
}

# Function to run GPU stress test (using CUDA or OpenGL)
function run_gpu_stress_test {
  local gpu_stress_duration=0  # Set to 0 for continuous stress test
  local gpu_stress_level="$1"
  local use_opengl="$2"

  colored_message $Blue "Running GPU stress test..."
  if [ "$use_opengl" = true ]; then
    colored_message $Green "Using OpenGL for GPU stress test..."
    glmark2 --run-forever --benchmark 0 &
    gpu_stress_pid=$!
  else
    colored_message $Green "Using CUDA for GPU stress test..."
    
    # Add CUDA GPU stress test code here
    
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
    colored_message $Blue "Stopping GPU stress test..."
    kill $gpu_stress_pid
    wait $gpu_stress_pid 2>/dev/null
    gpu_stress_pid=""
  fi
}

# Function to stop stress tests and clean up
function stop_stress_tests {
  colored_message $Blue "Stopping stress tests..."
  pkill stress-ng  # Stop CPU and RAM stress tests
  stop_gpu_stress_test  # Stop GPU stress test
  pkill glmark2   # Stop OpenGL-based GPU stress test
}

# Trap to stop stress tests on script termination
trap stop_stress_tests SIGINT SIGTERM

# Install dependencies
install_dependencies

# Main loop
while true; do
  display_menu
  read -p "Enter your choice (1/2/3/4/5/6/7): " choice


  case $choice in
    1)
      read -p "Enter CPU stress level (0-100): " cpu_stress_level
      run_cpu_stress_test $cpu_stress_level
      ;;

    2)
      read -p "Enter RAM stress level (0-100): " ram_stress_level
      run_ram_stress_test $ram_stress_level
      ;;

    3)
      read -p "Enter GPU stress level (0-100): " gpu_stress_level
      run_gpu_stress_test $gpu_stress_level false
      ;;

    4)
      read -p "Enter GPU stress level (0-100): " gpu_stress_level
      run_gpu_stress_test $gpu_stress_level true
      ;;

    5)
      read -p "Enter CPU stress level (0-100): " cpu_stress_level
      read -p "Enter RAM stress level (0-100): " ram_stress_level
      read -p "Enter GPU stress level (0-100): " gpu_stress_level

      run_cpu_stress_test $cpu_stress_level &  
      run_ram_stress_test $ram_stress_level &  
      run_gpu_stress_test $gpu_stress_level false &  
      ;;

    6)
      read -p "Enter CPU stress level (0-100): " cpu_stress_level
      read -p "Enter RAM stress level (0-100): " ram_stress_level
      read -p "Enter GPU stress level (0-100): " gpu_stress_level

      run_cpu_stress_test $cpu_stress_level &  
      run_ram_stress_test $ram_stress_level &  
      run_gpu_stress_test $gpu_stress_level true &  
      ;;

    7)
      stop_stress_tests
      colored_message $Green "Exiting stress test script."
      exit 0
      ;;

    *)
      colored_message $Green "Invalid choice. Please select a valid option."
      ;;
  esac
done

