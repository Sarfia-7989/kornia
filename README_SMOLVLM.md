# SmolVLM for Kornia-rs

This is a Rust implementation of SmolVLM (Small Vision Language Models) for the Kornia library. SmolVLM provides lightweight and efficient vision-language capabilities, particularly suitable for embedded systems and resource-constrained environments.

## Overview

SmolVLM integrates vision-language capabilities into Kornia, allowing for tasks such as:
- Image captioning
- Visual question answering
- Scene understanding
- Object detection description

The implementation supports multiple backends and model sizes:

**Backends:**
- Candle: Native Rust machine learning framework for efficient on-device inference
- ONNX Runtime: Cross-platform inferencing with ORT-Pyke integration

**Model Sizes:**
- Small (~300MB): For resource-constrained environments
- Medium (~500MB): Balanced between size and capability
- Large (~1GB): For best performance when resources allow

## Features

- 🚀 **Multiple backend support**: Choose between Candle (pure Rust) or ONNX Runtime
- 📏 **Size options**: Three model sizes to fit different resource constraints
- 📊 **Comprehensive benchmarking**: Tools to evaluate performance across backends and configurations
- 🔧 **Easy integration**: Simple API for kornia-rs applications
- 🧪 **Test utilities**: Example applications and testing framework
- 🔄 **Platform compatibility**: Works on desktop and NVIDIA Jetson platforms

## Installation

### Prerequisites

- Rust 1.76 or newer
- For ONNX backend: ONNX Runtime libraries
- For Candle backend: No additional dependencies

### Building

To build with both backends:

```bash
cargo build --features="kornia-models/candle kornia-models/onnx"
```

For only Candle backend:

```bash
cargo build --features="kornia-models/candle"
```

For only ONNX backend:

```bash
cargo build --features="kornia-models/onnx"
```

### Downloading Models

Use the provided script to download model weights:

```bash
# Download all models for all backends
./download_models.sh

# Download only small model for Candle backend
./download_models.sh --small --candle

# Download only medium model for ONNX backend
./download_models.sh --medium --onnx
```

## Usage

### Rust API

```rust
use kornia_models::smolvlm::common::{ModelSize, SmolVLMConfig};
use kornia_models::smolvlm::processor::ImageProcessor;
use kornia_models::smolvlm::candle::CandleBackend;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure SmolVLM
    let config = SmolVLMConfig::new(ModelSize::Small);
    
    // Initialize image processor
    let processor = ImageProcessor::new(&config)?;
    
    // Process an image
    let image = processor.process_image_from_path("path/to/image.jpg")?;
    
    // Initialize backend with model path
    let mut backend = CandleBackend::new("models/candle/Small", &config)?;
    
    // Generate a caption
    let caption = backend.generate_caption_for_image(&image, "What objects are in this image?")?;
    
    println!("Caption: {}", caption);
    
    Ok(())
}
```

### Example Applications

The repository includes several example applications:

#### Basic SmolVLM Demo

```bash
cargo run --example smolvlm_demo --features="kornia-models/candle" -- --image test_image.jpg --prompt "What objects are in this image?"
```

#### Backend Comparison

```bash
cargo run --example smolvlm_compare --features="kornia-models/candle kornia-models/onnx" -- --image test_image.jpg --prompt "What objects are in this image?" --backends candle onnx
```

#### Benchmarking Tool

```bash
cargo run --example smolvlm_compare --features="kornia-models/candle kornia-models/onnx" -- --image test_image.jpg --prompt "What objects are in this image?" --backends candle onnx --benchmark --runs 5
```

### Python Integration

The repository also includes Python scripts for demonstration and benchmarking:

#### Demo

```bash
cd kornia-rs
python smolvlm_demo.py -i test_image.jpg -p "What objects are in this image?"
```

With Hugging Face API integration:

```bash
export HF_TOKEN="your_hugging_face_token"
cd kornia-rs
python smolvlm_demo.py -i test_image.jpg -p "What objects are in this image?" --use-hf
```

#### Benchmarking

```bash
cd kornia-rs
python benchmark.py -i test_image.jpg -b python candle onnx -s small medium -t objects scene -r 3
```

## Benchmarking and Evaluation

### Performance Comparison

The implementation includes tools to benchmark and compare the performance of different backends and model sizes. Here are some key metrics to look for:

1. **Inference Time**: How long it takes to process an image and generate a response
2. **Memory Usage**: Peak memory consumption during inference
3. **Model Loading Time**: Time required to load the model into memory
4. **Accuracy**: Quality of generated captions or answers

### NVIDIA Jetson Compatibility

The implementation is designed to work well on NVIDIA Jetson platforms. When running on Jetson:

1. Ensure CUDA libraries are properly installed
2. For optimal performance with ONNX, TensorRT acceleration is recommended
3. For Candle backend, ensure CUDA support is enabled during build

## Architecture

The SmolVLM implementation consists of several key components:

1. **Common Interfaces**: Provides model configuration, error handling, and shared types
2. **Image Processor**: Handles image loading, preprocessing, and normalization
3. **Tokenizer**: Manages text tokenization for prompts and outputs
4. **Backend Implementations**:
   - Candle: Pure Rust implementation
   - ONNX: Integration with ONNX Runtime
5. **Benchmarking Tools**: Utilities for performance testing and comparison

## Contributing

Contributions to improve SmolVLM in kornia-rs are welcome! Areas for improvement include:

- Performance optimizations
- Additional backend implementations
- Enhanced model capabilities
- Better integration with other Kornia components
- Expanded testing on different platforms

## License

This project is licensed under Apache License 2.0 - see the LICENSE file for details.
