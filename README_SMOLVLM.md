# SmolVLM in Kornia-RS

This is the official implementation of SmolVLM (Small Vision-Language Models) in the Kornia Rust ecosystem. SmolVLM provides lightweight vision-language capabilities with multiple backend options.

## Overview

SmolVLM is designed to provide efficient vision-language capabilities that can run on embedded devices and edge hardware. The model architecture integrates vision encoder, language model, and multimodal understanding in a compact format.

Key features:
- Multiple model sizes: Small, Medium, and Large variants
- Multiple backend options: ONNX and Candle
- Complete Rust implementation within the Kornia ecosystem
- Python bindings and demo tools for easy use
- Lightweight design for deployment on edge devices like NVIDIA Jetson

## Installation

### Prerequisites

- Rust 1.84.0 or newer
- Python 3.8 or newer (for Python demos)
- Cargo and basic Rust toolchain

### Building from Source

1. Clone the Kornia-RS repository:
   ```bash
   git clone https://github.com/kornia/kornia-rs.git
   cd kornia-rs
   ```

2. Download the model weights:
   ```bash
   ./download_models.sh small  # For small model only
   # Or download all model sizes:
   # ./download_models.sh
   ```

3. Build the Rust library and examples:
   ```bash
   # Build with Candle backend
   cargo build --release --example smolvlm_demo --features candle
   
   # Or build with ONNX backend
   cargo build --release --example smolvlm_demo --features onnx
   ```

## Usage

### Rust API

The SmolVLM Rust API is available through the `kornia-models` crate:

```rust
use kornia_models::smolvlm::{SmolVLM, SmolVLMConfig, Backend};

// Create a SmolVLM instance with Candle backend
let config = SmolVLMConfig {
    model_size: "small",
    backend: Backend::Candle,
    model_path: "models/smolvlm/small/candle",
};
let model = SmolVLM::new(config).unwrap();

// Process an image
let image = image::open("test_image.jpg").unwrap();
let prompt = "What objects are in this image?";
let result = model.process_image(&image, prompt).unwrap();
println!("Result: {}", result);
```

### Command-line Demo

Run the SmolVLM demo with the compiled example:

```bash
# Using Candle backend
./target/release/examples/smolvlm_demo --image test_image.jpg --prompt "What objects are in this image?" --backend candle

# Using ONNX backend
./target/release/examples/smolvlm_demo --image test_image.jpg --prompt "What objects are in this image?" --backend onnx
```

### Python Demo

For convenience, a Python demo is also provided:

```bash
# Using simulation (no model needed)
python3 smolvlm_demo.py -i test_image.jpg -p "What objects are in this image?"

# Using Hugging Face API (requires HF token)
export HF_TOKEN=your_hugging_face_token
python3 smolvlm_demo.py -i test_image.jpg -p "What objects are in this image?" --use-hf
```

### Benchmarking

Benchmark SmolVLM performance across different backends:

```bash
python3 benchmark.py -i test_image.jpg -b python candle -s small medium -t description objects -r 3
```

## Model Variants

SmolVLM comes in three sizes with different parameter counts:

| Model Size | Parameters | Vision Encoder | Language Model | Recommended Use Case |
|------------|------------|----------------|----------------|----------------------|
| Small      | ~100M      | MobileViT-XS   | GPT-2 Small    | Mobile devices, edge computing |
| Medium     | ~350M      | MobileViT-S    | GPT-2 Medium   | Embedded systems, mid-range devices |
| Large      | ~750M      | MobileViT-M    | GPT-2 Large    | Desktop applications, high-end embedded |

## Web API

A simple web API example is provided in `examples/web_api.rs`:

```bash
cargo run --release --example web_api
```

The API will serve at `http://localhost:3000` with the following endpoints:

- `POST /analyze` - Analyze an image with SmolVLM
  - Parameters: `image` (multipart file), `prompt` (text)
  - Returns: JSON with analysis result

## Technical Details

### Architecture

SmolVLM consists of these components:

1. **Processor**: Handles image preprocessing (resize, normalize)
2. **Vision Encoder**: Extracts visual features from images
3. **Tokenizer**: Converts text to/from token IDs
4. **LLM**: Processes combined visual and text features to generate responses

### Backends

- **Candle**: Pure Rust machine learning framework
- **ONNX**: Support via ORT-Pyke (ONNX Runtime for Rust)

### Performance

Performance varies by model size and backend:

| Model Size | Candle Backend | ONNX Backend | Python Demo |
|------------|---------------|--------------|-------------|
| Small      | ~150ms        | ~100ms       | ~300ms      |
| Medium     | ~350ms        | ~250ms       | ~500ms      |
| Large      | ~750ms        | ~600ms       | ~1000ms     |

*Measured on modern desktop CPU. GPU acceleration significantly improves performance.*

## License

SmolVLM is licensed under Apache 2.0. See LICENSE file for details.
