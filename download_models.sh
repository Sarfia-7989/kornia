#!/bin/bash
# SmolVLM Model Downloader
# Downloads model weights for both Candle and ONNX backends

set -e

# Configuration
MODEL_BASE_DIR="models"
HF_REPO_SMALL="Kornia/SmolVLM-Small"
HF_REPO_MEDIUM="Kornia/SmolVLM-Medium"
HF_REPO_LARGE="Kornia/SmolVLM-Large"

# Parse command line arguments
DOWNLOAD_SMALL=false
DOWNLOAD_MEDIUM=false
DOWNLOAD_LARGE=false
DOWNLOAD_CANDLE=false
DOWNLOAD_ONNX=false

# If no specific arguments, download all
if [ $# -eq 0 ]; then
  DOWNLOAD_SMALL=true
  DOWNLOAD_MEDIUM=true
  DOWNLOAD_LARGE=true
  DOWNLOAD_CANDLE=true
  DOWNLOAD_ONNX=true
fi

# Parse arguments
for arg in "$@"; do
  case $arg in
    --small)
      DOWNLOAD_SMALL=true
      ;;
    --medium)
      DOWNLOAD_MEDIUM=true
      ;;
    --large)
      DOWNLOAD_LARGE=true
      ;;
    --candle)
      DOWNLOAD_CANDLE=true
      ;;
    --onnx)
      DOWNLOAD_ONNX=true
      ;;
    --help)
      echo "SmolVLM Model Downloader"
      echo ""
      echo "Usage: ./download_models.sh [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --small     Download small model weights"
      echo "  --medium    Download medium model weights"
      echo "  --large     Download large model weights"
      echo "  --candle    Download Candle backend weights"
      echo "  --onnx      Download ONNX backend weights"
      echo "  --help      Show this help message"
      echo ""
      echo "If no options are specified, all models and backends will be downloaded."
      exit 0
      ;;
    *)
      echo "Unknown option: $arg"
      echo "Use --help to see available options."
      exit 1
      ;;
  esac
done

# Create model directories
mkdir -p "${MODEL_BASE_DIR}/candle"
mkdir -p "${MODEL_BASE_DIR}/onnx"

# Check if huggingface-cli is available
if ! command -v huggingface-cli &> /dev/null; then
    echo "huggingface-cli is not installed. Please install it using:"
    echo "pip install huggingface_hub"
    echo ""
    echo "Alternatively, you can manually download the models from:"
    echo "https://huggingface.co/${HF_REPO_SMALL}"
    echo "https://huggingface.co/${HF_REPO_MEDIUM}"
    echo "https://huggingface.co/${HF_REPO_LARGE}"
    exit 1
fi

# Check for Hugging Face token
if [ -z "${HF_TOKEN}" ]; then
    echo "Warning: HF_TOKEN environment variable not set."
    echo "You may encounter rate limits or be unable to access private models."
    echo "Set the HF_TOKEN environment variable with your Hugging Face API token."
    echo ""
    HUGGINGFACE_AUTH=""
else
    HUGGINGFACE_AUTH="--token ${HF_TOKEN}"
    echo "Using Hugging Face token for authentication."
fi

# Function to download a specific model for a specific backend
download_model() {
    local size=$1
    local backend=$2
    local repo_name=$3
    local target_dir="${MODEL_BASE_DIR}/${backend}/${size}"
    
    echo "===== Downloading ${size} model for ${backend} backend ====="
    
    # Create target directory
    mkdir -p "${target_dir}"
    
    # Determine files to download based on backend
    if [ "${backend}" = "candle" ]; then
        # For Candle, download SafeTensors or GGML files
        echo "Downloading Candle model files from ${repo_name}..."
        huggingface-cli download ${HUGGINGFACE_AUTH} ${repo_name} \
            --local-dir "${target_dir}" \
            --include "*.safetensors" "*.ggml" "tokenizer.json" "config.json" \
            --quiet
    elif [ "${backend}" = "onnx" ]; then
        # For ONNX, download ONNX files
        echo "Downloading ONNX model files from ${repo_name}..."
        huggingface-cli download ${HUGGINGFACE_AUTH} ${repo_name} \
            --local-dir "${target_dir}" \
            --include "*.onnx" "tokenizer.json" "config.json" \
            --quiet
    fi
    
    echo "Download complete: ${target_dir}"
    echo ""
}

# Download models based on command line arguments
if [ "${DOWNLOAD_SMALL}" = true ]; then
    if [ "${DOWNLOAD_CANDLE}" = true ]; then
        download_model "Small" "candle" "${HF_REPO_SMALL}"
    fi
    if [ "${DOWNLOAD_ONNX}" = true ]; then
        download_model "Small" "onnx" "${HF_REPO_SMALL}"
    fi
fi

if [ "${DOWNLOAD_MEDIUM}" = true ]; then
    if [ "${DOWNLOAD_CANDLE}" = true ]; then
        download_model "Medium" "candle" "${HF_REPO_MEDIUM}"
    fi
    if [ "${DOWNLOAD_ONNX}" = true ]; then
        download_model "Medium" "onnx" "${HF_REPO_MEDIUM}"
    fi
fi

if [ "${DOWNLOAD_LARGE}" = true ]; then
    if [ "${DOWNLOAD_CANDLE}" = true ]; then
        download_model "Large" "candle" "${HF_REPO_LARGE}"
    fi
    if [ "${DOWNLOAD_ONNX}" = true ]; then
        download_model "Large" "onnx" "${HF_REPO_LARGE}"
    fi
fi

echo "===== All requested models have been downloaded ====="
echo "Model directory: ${MODEL_BASE_DIR}"
echo ""
echo "To use these models, provide the path to the model directory:"
echo "For Candle: ${MODEL_BASE_DIR}/candle/Small"
echo "For ONNX: ${MODEL_BASE_DIR}/onnx/Small"