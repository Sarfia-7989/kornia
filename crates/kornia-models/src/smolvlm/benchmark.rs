//! Benchmarking utilities for SmolVLM

use std::path::Path;
use std::time::{Duration, Instant};

use super::common::{SmolVLMBackend, SmolVLMError, SmolVLMVariant};
use super::{load_backend, SmolVLMModel};

/// Benchmark result for a single SmolVLM operation
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// The backend used
    pub backend: SmolVLMModel,
    /// The model variant
    pub variant: SmolVLMVariant,
    /// Whether CPU was used (true) or GPU (false)
    pub use_cpu: bool,
    /// Time taken to load the model
    pub load_time: Duration,
    /// Time taken to process the image
    pub process_time: Duration,
    /// Time taken to generate text
    pub generate_time: Duration,
    /// The generated text
    pub output: String,
}

impl BenchmarkResult {
    /// Format the benchmark result as a string
    pub fn to_string(&self) -> String {
        format!(
            "Backend: {:?}, Variant: {:?}, Device: {}\n\
             Load time: {:?}\n\
             Process time: {:?}\n\
             Generate time: {:?}\n\
             Total time: {:?}\n\
             Output: {}",
            self.backend,
            self.variant,
            if self.use_cpu { "CPU" } else { "GPU" },
            self.load_time,
            self.process_time,
            self.generate_time,
            self.load_time + self.process_time + self.generate_time,
            self.output
        )
    }
}

/// Run a comprehensive benchmark of SmolVLM models
///
/// # Arguments
///
/// * `backends` - List of backends to benchmark
/// * `variants` - List of model variants to benchmark
/// * `devices` - List of devices to benchmark (true for CPU, false for GPU)
/// * `model_path` - Path to model directory
/// * `image_path` - Path to test image
/// * `prompt` - Text prompt for the model
///
/// # Returns
///
/// Vector of benchmark results
pub fn run_benchmarks(
    backends: &[SmolVLMModel],
    variants: &[SmolVLMVariant],
    devices: &[bool],
    model_path: &Path,
    image_path: &Path,
    prompt: &str,
) -> Vec<Result<BenchmarkResult, SmolVLMError>> {
    let mut results = Vec::new();
    
    for &backend in backends {
        for &variant in variants {
            for &use_cpu in devices {
                // Skip GPU benchmarks if CUDA/GPU is not available
                if !use_cpu && !gpu_available() {
                    continue;
                }
                
                // Run benchmark
                let result = benchmark_model(
                    backend,
                    variant,
                    use_cpu,
                    model_path,
                    image_path,
                    prompt,
                );
                
                results.push(result);
            }
        }
    }
    
    results
}

/// Check if a GPU is available for inference
fn gpu_available() -> bool {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    return false;
    
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        std::process::Command::new("nvidia-smi")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    }
}

/// Benchmark a single model configuration
///
/// # Arguments
///
/// * `backend` - The backend to use
/// * `variant` - The model variant
/// * `use_cpu` - Whether to use CPU (true) or GPU (false)
/// * `model_path` - Path to model directory
/// * `image_path` - Path to test image
/// * `prompt` - Text prompt for the model
///
/// # Returns
///
/// Benchmark results
fn benchmark_model(
    backend: SmolVLMModel,
    variant: SmolVLMVariant,
    use_cpu: bool,
    model_path: &Path,
    image_path: &Path,
    prompt: &str,
) -> Result<BenchmarkResult, SmolVLMError> {
    // Load the backend
    let start = Instant::now();
    let mut model = load_backend(backend, variant, use_cpu, model_path)?;
    let load_time = start.elapsed();
    
    // Process the image
    let start = Instant::now();
    let image_tensor = model.process_image(image_path)?;
    let process_time = start.elapsed();
    
    // Generate text
    let start = Instant::now();
    let output = model.generate(&*image_tensor, prompt)?;
    let generate_time = start.elapsed();
    
    Ok(BenchmarkResult {
        backend,
        variant,
        use_cpu,
        load_time,
        process_time,
        generate_time,
        output,
    })
}

/// Print benchmark results in a tabular format
///
/// # Arguments
///
/// * `results` - Vector of benchmark results
pub fn print_benchmark_table(results: &[Result<BenchmarkResult, SmolVLMError>]) {
    // Print header
    println!("{:<10} {:<8} {:<5} {:<15} {:<15} {:<15} {:<15}", 
             "Backend", "Variant", "Device", "Load Time", "Process Time", "Generate Time", "Total Time");
    println!("{:-<90}", "");
    
    for result in results {
        match result {
            Ok(r) => {
                println!("{:<10?} {:<8?} {:<5} {:<15?} {:<15?} {:<15?} {:<15?}",
                         r.backend,
                         r.variant,
                         if r.use_cpu { "CPU" } else { "GPU" },
                         r.load_time,
                         r.process_time,
                         r.generate_time,
                         r.load_time + r.process_time + r.generate_time);
            }
            Err(e) => {
                println!("Error: {}", e);
            }
        }
    }
}

/// Get average FPS for image processing
///
/// # Arguments
///
/// * `result` - Benchmark result
///
/// # Returns
///
/// Frames per second for image processing
pub fn get_fps(result: &BenchmarkResult) -> f64 {
    // Combine load and process time for a more realistic FPS estimate
    let seconds = result.process_time.as_secs_f64();
    if seconds > 0.0 {
        1.0 / seconds
    } else {
        0.0
    }
}

/// Compare two backends on the same input
///
/// # Arguments
///
/// * `results` - Vector of benchmark results
///
/// # Returns
///
/// Comparison as a formatted string
pub fn compare_backends(
    results: &[Result<BenchmarkResult, SmolVLMError>],
) -> String {
    // Group results by variant and device
    let mut comparisons = String::new();
    
    for variant in [SmolVLMVariant::Tiny, SmolVLMVariant::Small, SmolVLMVariant::Medium] {
        for use_cpu in [true, false] {
            // Find results for this variant and device
            let filtered: Vec<_> = results.iter()
                .filter_map(|r| {
                    match r {
                        Ok(result) if result.variant == variant && result.use_cpu == use_cpu => Some(result),
                        _ => None,
                    }
                })
                .collect();
            
            if filtered.len() > 1 {
                comparisons.push_str(&format!(
                    "\n=== {variant:?} on {} ===\n",
                    if use_cpu { "CPU" } else { "GPU" }
                ));
                
                for result in &filtered {
                    comparisons.push_str(&format!(
                        "{:?} - Process: {:?}, Generate: {:?}, Total: {:?}, FPS: {:.2}\n",
                        result.backend,
                        result.process_time,
                        result.generate_time,
                        result.process_time + result.generate_time,
                        get_fps(result)
                    ));
                }
                
                // Compare outputs
                if filtered.len() >= 2 {
                    comparisons.push_str("\nOutputs:\n");
                    for result in &filtered {
                        comparisons.push_str(&format!("{:?}: {}\n", result.backend, result.output));
                    }
                }
            }
        }
    }
    
    comparisons
}