import os
import numpy as np
from PIL import Image
import io
import base64
import logging
import time
import psutil
from typing import Dict, Any
from model import ONNXModel, ImagePreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize model at module level
def initialize_model():
    """Initialize model and preprocessor"""
    try:
        model_path = "mtailor_classifier.onnx"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found")
        
        model = ONNXModel(model_path)
        preprocessor = ImagePreprocessor()
        logger.info("Model loaded successfully")
        return model, preprocessor
        
    except Exception as e:
        logger.error(f"Model initialization error: {str(e)}")
        raise e

# Initialize at module import
model, preprocessor = initialize_model()

def predict(item):
    """
    Main prediction function called by Cerebrium
    
    Args:
        item: Dictionary containing either:
            - 'image_base64': Base64 encoded image
            - 'image_url': URL to image
            - 'test_mode': Boolean to run health checks
    
    Returns:
        Dictionary with prediction results
    """
    global model, preprocessor
    
    try:
        start_time = time.time()
        
        # Check if model is initialized
        if model is None or preprocessor is None:
            return {
                "error": "Model not initialized",
                "status": "error"
            }
        
        # Health check mode
        if item.get('test_mode', False):
            return run_health_check()
        
        # Process image input
        if 'image_base64' in item:
            image = decode_base64_image(item['image_base64'])
        elif 'image_url' in item:
            image = load_image_from_url(item['image_url'])
        else:
            return {
                "error": "No image provided. Include 'image_base64' or 'image_url'",
                "status": "error"
            }
        
        # Make prediction
        prediction = model.predict_from_pil(image)
        inference_time = time.time() - start_time
        
        # Get prediction probabilities for top 5 classes
        input_data = preprocessor.preprocess_numpy(image)
        raw_output = model.predict(input_data)
        probabilities = softmax(raw_output[0])
        top5_indices = np.argsort(probabilities)[-5:][::-1]
        top5_probs = probabilities[top5_indices]
        
        return {
            "predicted_class": int(prediction),
            "confidence": float(probabilities[prediction]),
            "top5_predictions": [
                {"class": int(idx), "confidence": float(prob)} 
                for idx, prob in zip(top5_indices, top5_probs)
            ],
            "inference_time": inference_time,
            "status": "success",
            "model_info": {
                "input_shape": list(input_data.shape),
                "output_shape": list(raw_output.shape)
            }
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return {
            "error": str(e),
            "status": "error",
            "inference_time": time.time() - start_time if 'start_time' in locals() else 0
        }

def decode_base64_image(base64_string: str) -> Image.Image:
    """Decode base64 string to PIL Image"""
    try:
        # Remove data URL prefix if present
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        return image
    except Exception as e:
        raise ValueError(f"Failed to decode base64 image: {str(e)}")

def load_image_from_url(url: str) -> Image.Image:
    """Load image from URL"""
    try:
        import requests
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert('RGB')
        return image
    except Exception as e:
        raise ValueError(f"Failed to load image from URL: {str(e)}")

def softmax(x: np.ndarray) -> np.ndarray:
    """Apply softmax to get probabilities"""
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def run_health_check() -> Dict[str, Any]:
    """Run comprehensive health check"""
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "checks": {}
    }
    
    try:
        # Model availability check
        if model is None:
            health_status["checks"]["model_loaded"] = {"status": "failed", "message": "Model not loaded"}
            health_status["status"] = "unhealthy"
        else:
            health_status["checks"]["model_loaded"] = {"status": "passed"}
        
        # Memory usage check
        memory = psutil.virtual_memory()
        health_status["checks"]["memory"] = {
            "status": "passed" if memory.percent < 90 else "warning",
            "usage_percent": memory.percent,
            "available_gb": round(memory.available / (1024**3), 2)
        }
        
        # Model inference test
        dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        start_time = time.time()
        test_output = model.predict(dummy_input)
        inference_time = time.time() - start_time
        
        health_status["checks"]["inference_test"] = {
            "status": "passed",
            "inference_time": inference_time,
            "output_shape": list(test_output.shape)
        }
        
        # Disk space check
        disk = psutil.disk_usage('/')
        health_status["checks"]["disk_space"] = {
            "status": "passed" if disk.percent < 90 else "warning",
            "usage_percent": disk.percent,
            "free_gb": round(disk.free / (1024**3), 2)
        }
        
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["checks"]["error"] = {"status": "failed", "message": str(e)}
    
    return health_status