use std::path::Path;
use std::time::Instant;

use clap::Parser;
use kornia_models::smolvlm::{SmolVLM, SmolVLMConfig, SmolVLMBackend};

/// SmolVLM Demo CLI
#[derive(Parser, Debug)]
#[clap(author, version, about)]
struct Args {
    /// Path to the image file
    #[clap(short, long, required = true)]
    image: String,

    /// Text prompt for the model
    #[clap(short, long, required = true)]
    prompt: String,

    /// Path to the model directory or file
    #[clap(short, long, default_value = "models/smolvlm")]
    model_path: String,

    /// Model size to use
    #[clap(short = 's', long, default_value = "small")]
    model_size: String,

    /// Backend to use (candle or onnx)
    #[clap(short, long, default_value = "candle")]
    backend: String,
}

/// Simple command-line interface for SmolVLM
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line arguments
    let args = Args::parse();

    println!("======================================");
    println!("SmolVLM Demo");
    println!("======================================");
    println!("Image: {}", args.image);
    println!("Prompt: {}", args.prompt);
    println!("Model: {} ({})", args.model_size, args.backend);
    println!("--------------------------------------");

    // Check if image exists
    if !Path::new(&args.image).exists() {
        return Err(format!("Image file not found: {}", args.image).into());
    }

    // Configure the model based on size
    let config = match args.model_size.to_lowercase().as_str() {
        "small" => SmolVLMConfig::small(),
        "medium" => SmolVLMConfig::medium(),
        "large" => SmolVLMConfig::large(),
        _ => return Err(format!("Invalid model size: {}", args.model_size).into()),
    };

    // Create the model with the specified backend
    let backend = match args.backend.to_lowercase().as_str() {
        "candle" => SmolVLMBackend::Candle,
        "onnx" => SmolVLMBackend::Onnx,
        _ => return Err(format!("Invalid backend: {}", args.backend).into()),
    };

    println!("Loading model...");
    let start = Instant::now();
    
    // Load the model (implementation deferred to user code as it depends on the actual backend)
    #[cfg(all(feature = "candle", feature = "onnx"))]
    let model = match backend {
        SmolVLMBackend::Candle => SmolVLM::with_candle(&args.model_path, config)?,
        SmolVLMBackend::Onnx => SmolVLM::with_onnx(&args.model_path, config)?,
    };
    
    #[cfg(all(feature = "candle", not(feature = "onnx")))]
    let model = SmolVLM::with_candle(&args.model_path, config)?;
    
    #[cfg(all(not(feature = "candle"), feature = "onnx"))]
    let model = SmolVLM::with_onnx(&args.model_path, config)?;
    
    #[cfg(all(not(feature = "candle"), not(feature = "onnx")))]
    let _ = model; // To avoid unused variable warning
    #[cfg(all(not(feature = "candle"), not(feature = "onnx")))]
    return Err("No backend features enabled. Enable either 'candle' or 'onnx' feature.".into());
    
    println!("Model loaded in {:?}", start.elapsed());

    // Process the image and generate the response
    println!("Processing image: {}", args.image);
    let start = Instant::now();
    
    #[cfg(any(feature = "candle", feature = "onnx"))]
    let response = model.generate(&args.image, &args.prompt)?;
    
    #[cfg(all(not(feature = "candle"), not(feature = "onnx")))]
    let response = "No backend features enabled.".to_string();
    
    println!("Processing completed in {:?}", start.elapsed());

    // Print the response
    println!("\n======================================");
    println!("RESPONSE:");
    println!("======================================");
    println!("{}", response);
    println!("======================================");

    Ok(())
}