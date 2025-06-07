import unittest
import torch
import numpy as np
from PIL import Image
import os
import tempfile
from model import Classifier, BasicBlock, ImagePreprocessor, ONNXModel
from convert_to_onnx import convert_pytorch_to_onnx

class TestMLModelDeployment(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests"""
        # Create a dummy PyTorch model for testing
        cls.pytorch_model = Classifier(BasicBlock, [2, 2, 2, 2])
        cls.pytorch_model.eval()
        
        # Create temporary files
        cls.temp_dir = tempfile.mkdtemp()
        cls.pytorch_weights_path = os.path.join(cls.temp_dir, "test_model.pth")
        cls.onnx_model_path = os.path.join(cls.temp_dir, "test_model.onnx")
        
        # Save dummy weights
        torch.save(cls.pytorch_model.state_dict(), cls.pytorch_weights_path)
        
        # Convert to ONNX
        convert_pytorch_to_onnx(
            pytorch_model_path=cls.pytorch_weights_path,
            onnx_model_path=cls.onnx_model_path
        )
        
        # Create test image
        cls.test_image = Image.new('RGB', (256, 256), color='red')
        cls.test_image_path = os.path.join(cls.temp_dir, "test_image.jpg")
        cls.test_image.save(cls.test_image_path)
    
    def test_pytorch_model_loading(self):
        """Test PyTorch model loading and basic functionality"""
        model = Classifier(BasicBlock, [2, 2, 2, 2])
        model.load_state_dict(torch.load(self.pytorch_weights_path, map_location='cpu'))
        model.eval()
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        
        self.assertEqual(output.shape, (1, 1000))
        self.assertIsInstance(output, torch.Tensor)
    
    def test_image_preprocessing(self):
        """Test image preprocessing functionality"""
        preprocessor = ImagePreprocessor()
        
        # Test PIL preprocessing
        tensor_output = preprocessor.preprocess_pil(self.test_image)
        self.assertEqual(tensor_output.shape, (3, 224, 224))
        self.assertIsInstance(tensor_output, torch.Tensor)
        
        # Test numpy preprocessing
        numpy_output = preprocessor.preprocess_numpy(self.test_image)
        self.assertEqual(numpy_output.shape, (1, 3, 224, 224))
        self.assertIsInstance(numpy_output, np.ndarray)
        
        # Test preprocessing from path
        tensor_from_path = preprocessor.preprocess_from_path(self.test_image_path)
        numpy_from_path = preprocessor.preprocess_from_path(self.test_image_path, for_onnx=True)
        
        self.assertEqual(tensor_from_path.shape, (3, 224, 224))
        self.assertEqual(numpy_from_path.shape, (1, 3, 224, 224))
    
    def test_onnx_model_loading(self):
        """Test ONNX model loading and initialization"""
        onnx_model = ONNXModel(self.onnx_model_path)
        
        self.assertIsNotNone(onnx_model.session)
        self.assertEqual(onnx_model.input_name, "input")
        self.assertEqual(onnx_model.output_name, "output")
    
    def test_onnx_model_inference(self):
        """Test ONNX model inference"""
        onnx_model = ONNXModel(self.onnx_model_path)
        
        # Test with numpy array
        dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        output = onnx_model.predict(dummy_input)
        
        self.assertEqual(output.shape, (1, 1000))
        self.assertIsInstance(output, np.ndarray)
        
        # Test prediction from image path
        prediction = onnx_model.predict_from_image(self.test_image_path)
        self.assertIsInstance(prediction, (int, np.integer))
        self.assertGreaterEqual(prediction, 0)
        self.assertLess(prediction, 1000)
        
        # Test prediction from PIL image
        prediction_pil = onnx_model.predict_from_pil(self.test_image)
        self.assertIsInstance(prediction_pil, (int, np.integer))
        self.assertGreaterEqual(prediction_pil, 0)
        self.assertLess(prediction_pil, 1000)
    
    def test_pytorch_onnx_consistency(self):
        """Test consistency between PyTorch and ONNX models"""
        # Load PyTorch model
        pytorch_model = Classifier(BasicBlock, [2, 2, 2, 2])
        pytorch_model.load_state_dict(torch.load(self.pytorch_weights_path, map_location='cpu'))
        pytorch_model.eval()
        
        # Load ONNX model
        onnx_model = ONNXModel(self.onnx_model_path)
        
        # Prepare same input
        preprocessor = ImagePreprocessor()
        torch_input = preprocessor.preprocess_pil(self.test_image).unsqueeze(0)
        numpy_input = preprocessor.preprocess_numpy(self.test_image)
        
        # Get predictions
        with torch.no_grad():
            pytorch_output = pytorch_model(torch_input)
            pytorch_prediction = torch.argmax(pytorch_output).item()
        
        onnx_output = onnx_model.predict(numpy_input)
        onnx_prediction = np.argmax(onnx_output)
        
        # Check consistency
        self.assertEqual(pytorch_prediction, onnx_prediction)
        
        # Check output similarity (allowing for small numerical differences)
        np.testing.assert_allclose(
            pytorch_output.numpy(), 
            onnx_output, 
            rtol=1e-5, 
            atol=1e-5
        )
    
    def test_model_conversion(self):
        """Test PyTorch to ONNX conversion process"""
        temp_onnx_path = os.path.join(self.temp_dir, "conversion_test.onnx")
        
        # Test conversion
        result_path = convert_pytorch_to_onnx(
            pytorch_model_path=self.pytorch_weights_path,
            onnx_model_path=temp_onnx_path
        )
        
        self.assertEqual(result_path, temp_onnx_path)
        self.assertTrue(os.path.exists(temp_onnx_path))
        
        # Test that converted model works
        onnx_model = ONNXModel(temp_onnx_path)
        dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        output = onnx_model.predict(dummy_input)
        
        self.assertEqual(output.shape, (1, 1000))
    
    def test_error_handling(self):
        """Test error handling for various edge cases"""
        # Test with non-existent model path
        with self.assertRaises(Exception):
            ONNXModel("non_existent_model.onnx")
        
        # Test with invalid image path
        onnx_model = ONNXModel(self.onnx_model_path)
        with self.assertRaises(Exception):
            onnx_model.predict_from_image("non_existent_image.jpg")
        
        # Test with wrong input shape
        onnx_model = ONNXModel(self.onnx_model_path)
        wrong_input = np.random.randn(1, 3, 100, 100).astype(np.float32)
        with self.assertRaises(Exception):
            onnx_model.predict(wrong_input)
    
    def test_batch_processing(self):
        """Test batch processing capabilities"""
        onnx_model = ONNXModel(self.onnx_model_path)
        
        # Test with batch size > 1
        batch_input = np.random.randn(3, 3, 224, 224).astype(np.float32)
        batch_output = onnx_model.predict(batch_input)
        
        self.assertEqual(batch_output.shape, (3, 1000))
        
        # Test predictions for each item in batch
        predictions = np.argmax(batch_output, axis=1)
        self.assertEqual(len(predictions), 3)
        for pred in predictions:
            self.assertGreaterEqual(pred, 0)
            self.assertLess(pred, 1000)
    
    def test_model_info(self):
        """Test model information retrieval"""
        onnx_model = ONNXModel(self.onnx_model_path)
        
        # This should not raise an exception
        try:
            onnx_model.get_model_info()
        except Exception as e:
            self.fail(f"get_model_info() raised an exception: {e}")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(cls.temp_dir)

def run_performance_test():
    """Run performance benchmarks"""
    print("\n" + "="*50)
    print("PERFORMANCE BENCHMARKS")
    print("="*50)
    
    # Create temporary model for testing
    temp_dir = tempfile.mkdtemp()
    pytorch_model = Classifier(BasicBlock, [2, 2, 2, 2])
    pytorch_weights_path = os.path.join(temp_dir, "perf_test_model.pth")
    onnx_model_path = os.path.join(temp_dir, "perf_test_model.onnx")
    
    torch.save(pytorch_model.state_dict(), pytorch_weights_path)
    convert_pytorch_to_onnx(pytorch_weights_path, onnx_model_path)
    
    # Load models
    pytorch_model.load_state_dict(torch.load(pytorch_weights_path, map_location='cpu'))
    pytorch_model.eval()
    onnx_model = ONNXModel(onnx_model_path)
    
    # Prepare test data
    test_input_torch = torch.randn(10, 3, 224, 224)
    test_input_numpy = test_input_torch.numpy()
    
    # PyTorch inference time
    import time
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = pytorch_model(test_input_torch)
    pytorch_time = time.time() - start_time
    
    # ONNX inference time
    start_time = time.time()
    for _ in range(100):
        _ = onnx_model.predict(test_input_numpy)
    onnx_time = time.time() - start_time
    
    print(f"PyTorch inference time (100 batches): {pytorch_time:.4f}s")
    print(f"ONNX inference time (100 batches): {onnx_time:.4f}s")
    print(f"ONNX speedup: {pytorch_time/onnx_time:.2f}x")
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)

if __name__ == "__main__":
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run performance tests
    run_performance_test()