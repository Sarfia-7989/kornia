#!/bin/bash
# SmolVLM Platform Evaluation Tool
# Runs benchmarks on both desktop and NVIDIA Jetson platforms

set -e

# Configuration
OUTPUT_DIR="benchmark_results"
TEST_IMAGES="test_images"
MODEL_DIR="models"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="${OUTPUT_DIR}/benchmark_${TIMESTAMP}.json"
LOG_FILE="${OUTPUT_DIR}/benchmark_${TIMESTAMP}.log"

# Create necessary directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${TEST_IMAGES}"

# Check for test image
if [ ! -f "test_image.jpg" ]; then
    echo "Error: test_image.jpg not found. Please add a test image to the project root."
    exit 1
fi

# Copy test image to test images directory
cp test_image.jpg "${TEST_IMAGES}/"

# Detect platform
detect_platform() {
    if [ -f "/etc/nv_tegra_release" ]; then
        echo "nvidia_jetson"
    elif [ -f "/proc/device-tree/model" ] && grep -q "Raspberry Pi" /proc/device-tree/model; then
        echo "raspberry_pi"
    else
        echo "desktop"
    fi
}

PLATFORM=$(detect_platform)
echo "Detected platform: ${PLATFORM}"

# Get system information
get_system_info() {
    echo "===== System Information ====="
    echo "Platform: ${PLATFORM}"
    echo "Date: $(date)"
    echo "Kernel: $(uname -r)"
    echo "Architecture: $(uname -m)"
    
    # CPU information
    echo "CPU Information:"
    if [ -f "/proc/cpuinfo" ]; then
        CPU_MODEL=$(grep "model name" /proc/cpuinfo | head -n 1 | cut -d ":" -f 2 | sed 's/^[ \t]*//')
        CPU_CORES=$(grep -c "processor" /proc/cpuinfo)
        echo "  Model: ${CPU_MODEL}"
        echo "  Cores: ${CPU_CORES}"
    else
        echo "  CPU info not available"
    fi
    
    # Memory information
    echo "Memory Information:"
    if [ -f "/proc/meminfo" ]; then
        MEM_TOTAL=$(grep "MemTotal" /proc/meminfo | awk '{print $2 / 1024 / 1024}')
        echo "  Total Memory: ${MEM_TOTAL} GB"
    else
        echo "  Memory info not available"
    fi
    
    # GPU information for NVIDIA platforms
    if [ "${PLATFORM}" = "nvidia_jetson" ]; then
        echo "GPU Information:"
        if command -v nvidia-smi &> /dev/null; then
            nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
        else
            echo "  NVIDIA GPU detected but nvidia-smi not available"
        fi
    fi
    
    # Rust and Cargo version
    echo "Rust Information:"
    if command -v rustc &> /dev/null; then
        echo "  Rust: $(rustc --version)"
    else
        echo "  Rust not found"
    fi
    
    if command -v cargo &> /dev/null; then
        echo "  Cargo: $(cargo --version)"
    else
        echo "  Cargo not found"
    fi
    
    # Python version
    echo "Python Information:"
    if command -v python3 &> /dev/null; then
        echo "  Python: $(python3 --version)"
    else
        echo "  Python not found"
    fi
}

# Run a benchmark and measure time
run_benchmark() {
    local name=$1
    local cmd=$2
    
    echo "===== Running Benchmark: ${name} ====="
    echo "${cmd}"
    
    # Measure time
    start_time=$(date +%s.%N)
    eval "${cmd}"
    end_time=$(date +%s.%N)
    
    # Calculate elapsed time
    elapsed=$(echo "${end_time} - ${start_time}" | bc)
    echo "Elapsed time: ${elapsed} seconds"
    echo "===== Benchmark Complete: ${name} ====="
    echo ""
    
    # Return elapsed time
    echo "${elapsed}"
}

# Run Python benchmarks
run_python_benchmarks() {
    echo "===== Running Python Benchmarks ====="
    
    # Run standard benchmarks
    python3 benchmark.py -i test_image.jpg -b python -s small medium -t objects description -r 3 -o "${OUTPUT_DIR}/python_standard.json"
    
    # Run with Hugging Face API if token is available
    if [ -n "${HF_TOKEN}" ]; then
        python3 benchmark.py -i test_image.jpg -b python -s small medium -t objects description -r 3 --use-hf -o "${OUTPUT_DIR}/python_hf.json"
    else
        echo "Skipping Hugging Face API benchmarks (HF_TOKEN not set)"
    fi
}

# Run Rust benchmarks
run_rust_benchmarks() {
    echo "===== Running Rust Benchmarks ====="
    
    # Ensure we're building in release mode
    CARGO_FLAGS="--release"
    
    # Add platform-specific flags
    if [ "${PLATFORM}" = "nvidia_jetson" ]; then
        # For Jetson, we want to enable CUDA support if available
        if command -v nvcc &> /dev/null; then
            echo "CUDA detected, enabling CUDA support for Candle"
            CARGO_FLAGS="${CARGO_FLAGS} --features=kornia-models/candle-cuda kornia-models/onnx"
        else
            CARGO_FLAGS="${CARGO_FLAGS} --features=kornia-models/candle kornia-models/onnx"
        fi
    else
        # Standard desktop build
        CARGO_FLAGS="${CARGO_FLAGS} --features=kornia-models/candle kornia-models/onnx"
    fi
    
    # Build the comparison example
    echo "Building Rust SmolVLM with flags: ${CARGO_FLAGS}"
    cargo build --example smolvlm_compare ${CARGO_FLAGS}
    
    # Run benchmarks for each backend and model size
    for backend in candle onnx; do
        for size in small medium; do
            # Skip if model directory doesn't exist
            if [ ! -d "${MODEL_DIR}/${backend}/${size^}" ]; then
                echo "Skipping ${backend}/${size} (model directory not found)"
                continue
            fi
            
            # Run the benchmark
            MODEL_PATH="${MODEL_DIR}/${backend}/${size^}"
            OUTPUT_JSON="${OUTPUT_DIR}/rust_${backend}_${size}.json"
            
            echo "Running benchmark: ${backend} backend, ${size} model"
            cargo run --example smolvlm_compare ${CARGO_FLAGS} -- \
                --image test_image.jpg \
                --prompt "What objects are in this image?" \
                --model-path "${MODEL_PATH}" \
                --model-size "${size}" \
                --backend "${backend}" \
                --benchmark \
                --runs 3 \
                --warmup 1 \
                --output "${OUTPUT_JSON}"
        done
    done
}

# Generate a summary report
generate_summary() {
    echo "===== Generating Summary Report ====="
    
    # Collect all JSON files
    JSON_FILES=$(find "${OUTPUT_DIR}" -name "*.json")
    
    # Create summary file
    SUMMARY_FILE="${OUTPUT_DIR}/summary_${TIMESTAMP}.txt"
    
    {
        echo "SmolVLM Benchmark Summary"
        echo "=========================="
        echo "Platform: ${PLATFORM}"
        echo "Date: $(date)"
        echo ""
        
        # Include system information
        get_system_info
        
        echo ""
        echo "Benchmark Results"
        echo "================="
        
        # Process each JSON file and extract key metrics
        for json_file in ${JSON_FILES}; do
            backend=$(basename "${json_file}" .json | cut -d "_" -f 2)
            model_size=$(basename "${json_file}" .json | cut -d "_" -f 3)
            
            echo "Backend: ${backend}, Model Size: ${model_size}"
            
            # Extract metrics if jq is available
            if command -v jq &> /dev/null; then
                if [ -f "${json_file}" ]; then
                    # This is a simplified example - adjust based on your actual JSON structure
                    avg_time=$(jq '.results[0].avg_duration // "N/A"' "${json_file}" 2>/dev/null || echo "N/A")
                    success_rate=$(jq '.results[0].success_rate // "N/A"' "${json_file}" 2>/dev/null || echo "N/A")
                    
                    echo "  Average Time: ${avg_time} seconds"
                    echo "  Success Rate: ${success_rate}%"
                else
                    echo "  File not found or empty"
                fi
            else
                echo "  (jq not available, can't parse JSON metrics)"
            fi
            
            echo ""
        done
        
        echo ""
        echo "Conclusion"
        echo "=========="
        echo "For detailed results, see the individual JSON files in ${OUTPUT_DIR}"
    } > "${SUMMARY_FILE}"
    
    echo "Summary saved to: ${SUMMARY_FILE}"
    
    # Display the summary
    cat "${SUMMARY_FILE}"
}

# Main function
main() {
    # Start logging
    exec > >(tee -a "${LOG_FILE}") 2>&1
    
    echo "===== SmolVLM Platform Evaluation ====="
    echo "Date: $(date)"
    echo "Platform: ${PLATFORM}"
    echo "Output directory: ${OUTPUT_DIR}"
    echo "Log file: ${LOG_FILE}"
    echo ""
    
    # Get system information
    get_system_info
    
    # Run benchmarks
    run_python_benchmarks
    run_rust_benchmarks
    
    # Generate summary
    generate_summary
    
    echo ""
    echo "===== Evaluation Complete ====="
    echo "Results saved to: ${OUTPUT_DIR}"
}

# Run the main function
main "$@"
