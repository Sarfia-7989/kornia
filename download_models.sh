#!/bin/bash
# Script to download SmolVLM models for both ONNX and Candle backends

set -e

# Create models directory structure if it doesn't exist
mkdir -p models/smolvlm/{small,medium,large}/{onnx,candle}

echo "==== SmolVLM Model Downloader ===="
echo "This script will download model weights for SmolVLM."
echo

# Model URLs - these would be updated with actual URLs when available
MODELS=(
  # SmolVLM Small
  "small:onnx:processor:https://huggingface.co/kornia/SmolVLM-small/resolve/main/processor.onnx"
  "small:onnx:tokenizer:https://huggingface.co/kornia/SmolVLM-small/resolve/main/tokenizer.onnx"
  "small:onnx:vision:https://huggingface.co/kornia/SmolVLM-small/resolve/main/vision_encoder.onnx"
  "small:onnx:llm:https://huggingface.co/kornia/SmolVLM-small/resolve/main/llm.onnx"
  "small:candle:processor:https://huggingface.co/kornia/SmolVLM-small/resolve/main/processor.safetensors"
  "small:candle:tokenizer:https://huggingface.co/kornia/SmolVLM-small/resolve/main/tokenizer.safetensors"
  "small:candle:vision:https://huggingface.co/kornia/SmolVLM-small/resolve/main/vision_encoder.safetensors"
  "small:candle:llm:https://huggingface.co/kornia/SmolVLM-small/resolve/main/llm.safetensors"
  
  # SmolVLM Medium
  "medium:onnx:processor:https://huggingface.co/kornia/SmolVLM-medium/resolve/main/processor.onnx"
  "medium:onnx:tokenizer:https://huggingface.co/kornia/SmolVLM-medium/resolve/main/tokenizer.onnx"
  "medium:onnx:vision:https://huggingface.co/kornia/SmolVLM-medium/resolve/main/vision_encoder.onnx"
  "medium:onnx:llm:https://huggingface.co/kornia/SmolVLM-medium/resolve/main/llm.onnx"
  "medium:candle:processor:https://huggingface.co/kornia/SmolVLM-medium/resolve/main/processor.safetensors"
  "medium:candle:tokenizer:https://huggingface.co/kornia/SmolVLM-medium/resolve/main/tokenizer.safetensors"
  "medium:candle:vision:https://huggingface.co/kornia/SmolVLM-medium/resolve/main/vision_encoder.safetensors"
  "medium:candle:llm:https://huggingface.co/kornia/SmolVLM-medium/resolve/main/llm.safetensors"
  
  # SmolVLM Large
  "large:onnx:processor:https://huggingface.co/kornia/SmolVLM-large/resolve/main/processor.onnx"
  "large:onnx:tokenizer:https://huggingface.co/kornia/SmolVLM-large/resolve/main/tokenizer.onnx"
  "large:onnx:vision:https://huggingface.co/kornia/SmolVLM-large/resolve/main/vision_encoder.onnx"
  "large:onnx:llm:https://huggingface.co/kornia/SmolVLM-large/resolve/main/llm.onnx"
  "large:candle:processor:https://huggingface.co/kornia/SmolVLM-large/resolve/main/processor.safetensors"
  "large:candle:tokenizer:https://huggingface.co/kornia/SmolVLM-large/resolve/main/tokenizer.safetensors"
  "large:candle:vision:https://huggingface.co/kornia/SmolVLM-large/resolve/main/vision_encoder.safetensors"
  "large:candle:llm:https://huggingface.co/kornia/SmolVLM-large/resolve/main/llm.safetensors"
)

# Check for size argument
if [ "$#" -gt 0 ]; then
  SIZE="$1"
  echo "Will download only '$SIZE' model size."
else
  SIZE="all"
  echo "Will download models for all sizes: small, medium, large."
fi

# Check if HF token is available for authenticated downloads
if [ -n "$HF_TOKEN" ]; then
  AUTH_HEADER="--header 'Authorization: Bearer $HF_TOKEN'"
  echo "Using Hugging Face authentication token."
else
  AUTH_HEADER=""
  echo "No Hugging Face authentication token found. Trying unauthenticated download."
fi

# Count total number of models to download
if [ "$SIZE" = "all" ]; then
  TOTAL=${#MODELS[@]}
else
  TOTAL=0
  for model in "${MODELS[@]}"; do
    model_size=$(echo $model | cut -d':' -f1)
    if [ "$model_size" = "$SIZE" ]; then
      TOTAL=$((TOTAL + 1))
    fi
  done
fi

echo "Will download $TOTAL model files."
echo

# Download the models
COUNT=0
for model in "${MODELS[@]}"; do
  # Parse model info
  model_size=$(echo $model | cut -d':' -f1)
  backend=$(echo $model | cut -d':' -f2)
  component=$(echo $model | cut -d':' -f3)
  url=$(echo $model | cut -d':' -f4-)
  
  # Skip if not the requested size
  if [ "$SIZE" != "all" ] && [ "$model_size" != "$SIZE" ]; then
    continue
  fi
  
  COUNT=$((COUNT + 1))
  
  # Create destination directory and filename
  dest_dir="models/smolvlm/$model_size/$backend"
  filename="$component.$(echo $url | grep -o '[^.]*$')"
  dest_file="$dest_dir/$filename"
  
  echo "[$COUNT/$TOTAL] Downloading $model_size model ($backend backend): $component"
  echo "  URL: $url"
  echo "  Destination: $dest_file"
  
  # Create directory if it doesn't exist
  mkdir -p "$dest_dir"
  
  # Download the file
  if [ -n "$AUTH_HEADER" ]; then
    curl -L -o "$dest_file" $AUTH_HEADER "$url"
  else
    curl -L -o "$dest_file" "$url"
  fi
  
  echo "  Download complete!"
  echo
done

echo "==== Download Complete ===="
echo "Downloaded $COUNT model files."
echo
echo "Models are stored in the 'models/smolvlm/' directory organized by size and backend:"
echo "- models/smolvlm/small/{onnx,candle}/"
echo "- models/smolvlm/medium/{onnx,candle}/"
echo "- models/smolvlm/large/{onnx,candle}/"
echo
echo "You can now use these models with the SmolVLM demo applications."
