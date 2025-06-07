# Image Classification Model Deployment on Cerebrium

This project demonstrates the deployment of a PyTorch-based image classification model on Cerebrium's serverless GPU platform using Docker containerization and ONNX optimization.

## Project Overview

The solution converts a PyTorch ImageNet classification model to ONNX format and deploys it on Cerebrium for production-ready inference with sub-3-second response times.

## Architecture & Process

### 1. Model Conversion Pipeline
- **PyTorch to ONNX**: Converted the original PyTorch model to ONNX format for optimized inference
- **Weight Integration**: Downloaded and integrated pre-trained ImageNet weights
- **Preprocessing Optimization**: Implemented efficient image preprocessing pipeline

### 2. Deployment Strategy
- **Docker-based Deployment**: Used custom Dockerfile for Cerebrium deployment
- **Serverless Architecture**: Leveraged Cerebrium's CPU infrastructure for scalable inference
- **Function-based API**: Implemented Cerebrium's native predict() function interface

### 3. Testing Framework
- **Local Testing**: Comprehensive test suite for model functionality
- **Server Testing**: Remote API testing for deployed model
- **Health Monitoring**: Endpoint monitoring and performance validation

## Project Structure

```
├── cerebrium-instance/          # Cerebrium deployment configuration
│   ├── cerebrium.toml          # Cerebrium configuration
│   ├── Dockerfile              # Custom Docker image
│   ├── main.py                 # API endpoint handler
│   ├── model.py                # Model inference logic
│   └── mtailor_classifier.onnx # Converted ONNX model
├── convert_to_onnx.py          # PyTorch to ONNX conversion script
├── test.py                     # Local model testing suite
├── test_server.py              # Remote API testing script
├── testing.py                  # Personal testing utilities
├── pytorch_model_weights.pth   # Pre-trained model weights
├── n01440764_tench.jpeg        # Test image (class 0)
├── n01667114_mud_turtle.JPEG   # Test image (class 35)
└── requirements.txt            # Python dependencies
```

## Implementation Process

### Phase 1: Model Preparation
1. **Weight Download**: Retrieved pre-trained ImageNet weights from provided Dropbox link
2. **Model Conversion**: Implemented `convert_to_onnx.py` to convert PyTorch model to ONNX format
3. **Preprocessing Pipeline**: Extracted and optimized image preprocessing steps from original PyTorch implementation

### Phase 2: Cerebrium Integration
1. **Docker Configuration**: Created custom Dockerfile following Cerebrium's requirements
2. **Function Development**: Implemented Cerebrium's native predict() function in `main.py`
3. **Model Loading**: Implemented efficient ONNX model loading and inference in `model.py`
4. **Configuration**: Set up `cerebrium.toml` for deployment parameters

### Phase 3: Testing & Validation
1. **Local Testing**: Developed comprehensive test suite (`test.py`) for model accuracy and preprocessing
2. **API Testing**: Created `test_server.py` for remote endpoint validation
3. **Performance Monitoring**: Implemented health checks and response time monitoring
4. **Multiple Deployments**: Tested 3 different API versions for optimization

### Phase 4: Deployment & Monitoring
1. **Cerebrium Deployment**: Successfully deployed using `cerebrium deploy` command
2. **API Validation**: Verified all endpoints respond within 2-3 second requirement
3. **Production Testing**: Validated classification accuracy on provided test images

## API Architecture

### Cerebrium Function Interface
The deployment uses Cerebrium's native function-based approach:
- **Main Function**: `predict(item)` in `main.py` handles all requests
- **Input Formats**: Supports `image_base64`, `image_url`, and `test_mode` parameters
- **Health Monitoring**: Built-in health check functionality with system metrics
- **Error Handling**: Comprehensive exception handling with detailed error responses

### Model Components
- **ONNXModel**: ONNX runtime wrapper for efficient inference
- **ImagePreprocessor**: Handles image preprocessing pipeline (resize, normalize, format conversion)
- **Health Checker**: System monitoring including memory, disk space, and model status

## Technical Specifications

- **Input**: 224x224 RGB images
- **Output**: 1000-class probability distribution (ImageNet classes)
- **Preprocessing**: RGB conversion, resize, normalization with ImageNet statistics
- **Performance**: <3 second inference time
- **Platform**: Cerebrium serverless GPU infrastructure

## Usage Instructions

### Local Development
```bash
# Convert PyTorch model to ONNX
python convert_to_onnx.py

### Deployment
```bash
cd cerebrium-instance
cerebrium deploy
```

### API Testing & Usage

The deployed model provides three main endpoints accessible via `test_server.py`:

#### Health Check
```bash
python test_server.py --health
```
- Validates model loading, memory usage, disk space, and inference capability
- Returns system status and performance metrics

#### Local Image Prediction
```bash
python test_server.py --local-image "path/to/image.jpg"
```
- Processes local image files
- Returns predicted class, confidence score, and top-5 predictions

#### Remote Image Prediction  
```bash
python test_server.py --remote-image "https://example.com/image.jpg"
```
- Fetches and processes images from URLs
- Supports any publicly accessible image URL

#### API Configuration
- **Endpoint**: `https://api.cortex.cerebrium.ai/v4/p-0aee9e27/mtailor-classifier`
- **Authentication**: Bearer token included in `test_server.py`
- **Response Format**: JSON with run_id, predictions, confidence scores, and timing metrics

## Testing Results

The solution successfully:
- Converts PyTorch model to ONNX format
- Deploys on Cerebrium with custom Docker configuration
- Achieves sub-3-second inference times
- Correctly classifies test images (tench: class 0, mud turtle: class 35)
- Provides comprehensive API testing and monitoring capabilities