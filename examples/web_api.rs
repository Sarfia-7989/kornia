use std::net::SocketAddr;
use std::sync::Arc;

use axum::{
    extract::{Multipart, State},
    http::StatusCode,
    response::{Html, IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tokio::time::Instant;
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::services::ServeDir;
use tower_http::trace::TraceLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use kornia_models::smolvlm::{SmolVLM, SmolVLMConfig};

/// Application state
struct AppState {
    model: Mutex<Option<SmolVLM>>,
    model_path: String,
    model_size: String,
}

/// Response format for image analysis
#[derive(Serialize)]
struct AnalysisResponse {
    result: String,
    processing_time_ms: u64,
}

/// Request format for image analysis
#[derive(Deserialize)]
struct AnalysisRequest {
    prompt: String,
}

/// Error response
#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

/// Main function to start the web server
#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Get port from environment or use default
    let port = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse::<u16>().ok())
        .unwrap_or(5000);

    // Get model path and size from environment or use defaults
    let model_path = std::env::var("MODEL_PATH").unwrap_or_else(|_| "models/smolvlm".into());
    let model_size = std::env::var("MODEL_SIZE").unwrap_or_else(|_| "small".into());

    // Create application state
    let app_state = Arc::new(AppState {
        model: Mutex::new(None),
        model_path,
        model_size,
    });

    // Build the router
    let app = Router::new()
        .route("/", get(index_handler))
        .route("/api/analyze", post(analyze_handler))
        .route("/api/info", get(info_handler))
        .route("/api/load", post(load_model_handler))
        .nest_service("/static", ServeDir::new("static"))
        .layer(TraceLayer::new_for_http())
        .layer(RequestBodyLimitLayer::new(10 * 1024 * 1024)) // 10MB limit
        .with_state(app_state);

    // Bind to address and start server
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    tracing::info!("Starting server on {}", addr);
    
    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
}

/// Handler for the index page
async fn index_handler() -> impl IntoResponse {
    Html(
        r#"
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>SmolVLM Demo</title>
            <style>
                body {
                    font-family: system-ui, -apple-system, sans-serif;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                    line-height: 1.6;
                }
                h1 {
                    color: #333;
                    border-bottom: 1px solid #eee;
                    padding-bottom: 10px;
                }
                form {
                    background: #f9f9f9;
                    padding: 20px;
                    border-radius: 8px;
                    margin-bottom: 20px;
                }
                label {
                    display: block;
                    margin-bottom: 8px;
                    font-weight: bold;
                }
                input[type="file"], input[type="text"] {
                    width: 100%;
                    padding: 8px;
                    margin-bottom: 16px;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                }
                button {
                    background: #4CAF50;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 4px;
                    cursor: pointer;
                }
                button:hover {
                    background: #45a049;
                }
                #result {
                    background: #f0f0f0;
                    padding: 20px;
                    border-radius: 8px;
                    white-space: pre-wrap;
                    min-height: 100px;
                    border: 1px solid #ddd;
                }
                #loading {
                    display: none;
                    text-align: center;
                    margin: 20px 0;
                }
                #preview {
                    max-width: 100%;
                    max-height: 300px;
                    margin-bottom: 16px;
                    display: none;
                }
                #error {
                    color: red;
                    display: none;
                    margin-bottom: 16px;
                }
                #info {
                    background: #e9f7fe;
                    padding: 10px;
                    border-radius: 4px;
                    margin-bottom: 20px;
                }
                .action-buttons {
                    display: flex;
                    gap: 10px;
                    margin-bottom: 16px;
                }
            </style>
        </head>
        <body>
            <h1>SmolVLM Demo</h1>
            
            <div id="info">
                <p>Model status: <span id="model-status">Not loaded</span></p>
                <p>Model size: <span id="model-size">Unknown</span></p>
            </div>
            
            <div class="action-buttons">
                <button type="button" id="load-button">Load Model</button>
            </div>
            
            <form id="upload-form">
                <label for="image-upload">Select Image:</label>
                <input type="file" id="image-upload" accept="image/*" required>
                <img id="preview" src="#" alt="Preview">
                
                <label for="prompt">Prompt:</label>
                <input type="text" id="prompt" required placeholder="Describe what you see in this image">
                
                <button type="submit">Analyze Image</button>
            </form>
            
            <div id="error"></div>
            
            <div id="loading">
                <p>Processing image, please wait...</p>
            </div>
            
            <h2>Result</h2>
            <pre id="result">No analysis yet...</pre>
            
            <script>
                document.addEventListener('DOMContentLoaded', function() {
                    const uploadForm = document.getElementById('upload-form');
                    const imageUpload = document.getElementById('image-upload');
                    const promptInput = document.getElementById('prompt');
                    const preview = document.getElementById('preview');
                    const result = document.getElementById('result');
                    const loading = document.getElementById('loading');
                    const errorDiv = document.getElementById('error');
                    const loadButton = document.getElementById('load-button');
                    const modelStatus = document.getElementById('model-status');
                    const modelSize = document.getElementById('model-size');
                    
                    // Get model info on page load
                    fetchModelInfo();
                    
                    // Load model button
                    loadButton.addEventListener('click', async function() {
                        loading.style.display = 'block';
                        loadButton.disabled = true;
                        loadButton.textContent = 'Loading...';
                        
                        try {
                            const response = await fetch('/api/load', {
                                method: 'POST'
                            });
                            
                            if (!response.ok) {
                                const data = await response.json();
                                throw new Error(data.error || 'Failed to load model');
                            }
                            
                            fetchModelInfo();
                        } catch (error) {
                            errorDiv.textContent = `Error: ${error.message}`;
                            errorDiv.style.display = 'block';
                        } finally {
                            loading.style.display = 'none';
                            loadButton.disabled = false;
                            loadButton.textContent = 'Load Model';
                        }
                    });
                    
                    // Image preview
                    imageUpload.addEventListener('change', function() {
                        if (this.files && this.files[0]) {
                            const reader = new FileReader();
                            reader.onload = function(e) {
                                preview.src = e.target.result;
                                preview.style.display = 'block';
                            };
                            reader.readAsDataURL(this.files[0]);
                        }
                    });
                    
                    // Form submission
                    uploadForm.addEventListener('submit', async function(e) {
                        e.preventDefault();
                        
                        if (!imageUpload.files || imageUpload.files.length === 0) {
                            errorDiv.textContent = 'Please select an image file';
                            errorDiv.style.display = 'block';
                            return;
                        }
                        
                        const prompt = promptInput.value.trim();
                        if (!prompt) {
                            errorDiv.textContent = 'Please enter a prompt';
                            errorDiv.style.display = 'block';
                            return;
                        }
                        
                        // Hide error if shown
                        errorDiv.style.display = 'none';
                        
                        // Show loading indicator
                        loading.style.display = 'block';
                        
                        // Create form data
                        const formData = new FormData();
                        formData.append('image', imageUpload.files[0]);
                        formData.append('prompt', prompt);
                        
                        try {
                            const response = await fetch('/api/analyze', {
                                method: 'POST',
                                body: formData
                            });
                            
                            if (!response.ok) {
                                const data = await response.json();
                                throw new Error(data.error || 'Failed to analyze image');
                            }
                            
                            const data = await response.json();
                            result.textContent = data.result + '\n\nProcessing time: ' + 
                                                (data.processing_time_ms / 1000).toFixed(2) + ' seconds';
                        } catch (error) {
                            errorDiv.textContent = `Error: ${error.message}`;
                            errorDiv.style.display = 'block';
                            result.textContent = 'Analysis failed';
                        } finally {
                            loading.style.display = 'none';
                        }
                    });
                    
                    // Fetch model info
                    async function fetchModelInfo() {
                        try {
                            const response = await fetch('/api/info');
                            
                            if (!response.ok) {
                                throw new Error('Failed to fetch model info');
                            }
                            
                            const data = await response.json();
                            modelStatus.textContent = data.loaded ? 'Loaded' : 'Not loaded';
                            modelSize.textContent = data.model_size;
                        } catch (error) {
                            console.error('Error fetching model info:', error);
                        }
                    }
                });
            </script>
        </body>
        </html>
        "#
    )
}

/// Handler for analyzing images
async fn analyze_handler(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Result<Json<AnalysisResponse>, AppError> {
    // Extract prompt and image from multipart form
    let mut image_data = None;
    let mut prompt = None;

    while let Some(field) = multipart.next_field().await.map_err(|e| {
        AppError::new(
            StatusCode::BAD_REQUEST,
            format!("Failed to process multipart form: {}", e),
        )
    })? {
        let name = field.name().unwrap_or_default().to_string();

        if name == "image" {
            let data = field.bytes().await.map_err(|e| {
                AppError::new(
                    StatusCode::BAD_REQUEST,
                    format!("Failed to read image data: {}", e),
                )
            })?;
            image_data = Some(data.to_vec());
        } else if name == "prompt" {
            let text = field.text().await.map_err(|e| {
                AppError::new(
                    StatusCode::BAD_REQUEST,
                    format!("Failed to read prompt text: {}", e),
                )
            })?;
            prompt = Some(text);
        }
    }

    let image_data = image_data.ok_or_else(|| {
        AppError::new(StatusCode::BAD_REQUEST, "Missing image data".to_string())
    })?;

    let prompt = prompt.ok_or_else(|| {
        AppError::new(StatusCode::BAD_REQUEST, "Missing prompt".to_string())
    })?;

    // Make sure the model is loaded
    let model_guard = state.model.lock().await;
    if model_guard.is_none() {
        return Err(AppError::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "Model not loaded. Please load the model first.".to_string(),
        ));
    }

    // Simulate model inference
    let start = Instant::now();
    
    // TODO: Replace with actual model implementation
    // Here we'd usually call something like:
    // let result = model_guard.as_ref().unwrap().generate_from_bytes(&image_data, &prompt)?;
    
    // Instead, we'll use a simulated response
    let result = format!(
        "This is a simulated SmolVLM response for the given image and prompt: '{}'.\n\n\
        The model would analyze the image and generate a detailed description based on the prompt.\n\n\
        This is just a placeholder until the full model implementation is complete.",
        prompt
    );
    
    let processing_time = start.elapsed().as_millis() as u64;

    // Return the result
    Ok(Json(AnalysisResponse {
        result,
        processing_time_ms: processing_time,
    }))
}

/// Handler for getting model info
async fn info_handler(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let model_guard = state.model.lock().await;
    
    Json(serde_json::json!({
        "loaded": model_guard.is_some(),
        "model_size": state.model_size,
        "model_path": state.model_path
    }))
}

/// Handler for loading the model
async fn load_model_handler(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>, AppError> {
    let mut model_guard = state.model.lock().await;
    
    // Don't reload if already loaded
    if model_guard.is_some() {
        return Ok(Json(serde_json::json!({
            "status": "Model already loaded",
            "model_size": state.model_size
        })));
    }
    
    // Configure the model based on size
    let config = match state.model_size.as_str() {
        "small" => SmolVLMConfig::small(),
        "medium" => SmolVLMConfig::medium(),
        "large" => SmolVLMConfig::large(),
        _ => {
            return Err(AppError::new(
                StatusCode::BAD_REQUEST,
                format!("Invalid model size: {}", state.model_size),
            ));
        }
    };
    
    // TODO: Load the actual model
    // Instead of calling the real implementation, we'll simulate model loading:
    // *model_guard = Some(SmolVLM::with_candle(&state.model_path, config)?);
    
    // For demo, we just wait a bit to simulate loading
    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
    
    *model_guard = Some(SmolVLM::dummy(config));
    
    Ok(Json(serde_json::json!({
        "status": "Model loaded successfully",
        "model_size": state.model_size
    })))
}

/// Custom error type for API errors
struct AppError {
    code: StatusCode,
    message: String,
}

impl AppError {
    fn new(code: StatusCode, message: String) -> Self {
        Self { code, message }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let body = Json(ErrorResponse {
            error: self.message.clone(),
        });

        (self.code, body).into_response()
    }
}

/// Dummy implementation for SmolVLM (to be implemented in the actual module)
impl SmolVLM {
    fn dummy(config: SmolVLMConfig) -> Self {
        // Placeholder implementation - replace with actual implementation in module
        todo!("Implement SmolVLM in the kornia_models::smolvlm module")
    }
}