import numpy as np
from PIL import Image
import onnxruntime as ort

class ImagePreprocessor:
    """
    Image preprocessing class using NumPy and Pillow for consistent preprocessing,
    equivalent to the original PyTorch-based version.
    """

    def __init__(self, input_size: tuple = (224, 224)):
        """
        Initializes the preprocessor with a target input size.
        """
        self.input_size = input_size
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def _apply_transforms(self, img: Image.Image) -> np.ndarray:
        """
        Applies the full sequence of transformations to a PIL Image.
        """
        # 1. Resize the image
        img = img.resize(self.input_size, Image.Resampling.BILINEAR)

        # 2. Center Crop the image
        width, height = img.size
        crop_width, crop_height = self.input_size
        left = (width - crop_width) / 2
        top = (height - crop_height) / 2
        right = (width + crop_width) / 2
        bottom = (height + crop_height) / 2
        img = img.crop((left, top, right, bottom))

        # 3. Convert to "tensor" (numpy array) and normalize
        img_np = np.array(img, dtype=np.float32)
        img_np /= 255.0

        # Transpose from (H, W, C) to (C, H, W)
        img_np = np.transpose(img_np, (2, 0, 1))

        # Normalize the image
        normalized_img = (img_np - self.mean[:, np.newaxis, np.newaxis]) / self.std[:, np.newaxis, np.newaxis]

        return normalized_img.astype(np.float32)

    def preprocess_pil(self, img: Image.Image) -> np.ndarray:
        """
        Preprocess PIL Image. Returns a numpy array of shape (C, H, W).
        This is the equivalent of the original method that returned a PyTorch tensor.
        """
        return self._apply_transforms(img)

    def preprocess_numpy(self, img: Image.Image) -> np.ndarray:
        """
        Preprocess PIL Image for an ONNX-like model.
        Returns a numpy array with an added batch dimension, shape (1, C, H, W).
        """
        processed_arr = self._apply_transforms(img)
        # Add a batch dimension
        return np.expand_dims(processed_arr, axis=0)

    def preprocess_from_path(self, img_path: str, for_onnx: bool = False):
        """
        Load and preprocess image from a file path.
        """
        img = Image.open(img_path).convert('RGB')
        if for_onnx:
            return self.preprocess_numpy(img)
        else:
            return self.preprocess_pil(img)

class ONNXModel:
    """ONNX model wrapper for inference"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.preprocessor = ImagePreprocessor()
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference on preprocessed input data"""
        outputs = self.session.run([self.output_name], {self.input_name: input_data})
        return outputs[0]
    
    def predict_from_image(self, img_path: str) -> int:
        """Predict class from image file path"""
        input_data = self.preprocessor.preprocess_from_path(img_path, for_onnx=True)
        outputs = self.predict(input_data)
        return np.argmax(outputs)
    
    def predict_from_pil(self, img: Image.Image) -> int:
        """Predict class from PIL Image"""
        input_data = self.preprocessor.preprocess_numpy(img)
        outputs = self.predict(input_data)
        return np.argmax(outputs)
    
    def get_model_info(self):
        """Get model input/output information"""
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        
        print("Model Information:")
        print(f"Input: {inputs[0].name}, Shape: {inputs[0].shape}, Type: {inputs[0].type}")
        print(f"Output: {outputs[0].name}, Shape: {outputs[0].shape}, Type: {outputs[0].type}")