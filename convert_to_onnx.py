import torch
from model import Classifier, BasicBlock

def convert_pytorch_to_onnx(
    pytorch_model_path: str = "./pytorch_model_weights.pth",
    onnx_model_path: str = "mtailor_classifier.onnx",
    input_shape: tuple = (1, 3, 224, 224),
    opset_version: int = 12
):
    """
    Convert PyTorch model to ONNX format
    
    Args:
        pytorch_model_path: Path to PyTorch model weights
        onnx_model_path: Output path for ONNX model
        input_shape: Input tensor shape (batch_size, channels, height, width)
        opset_version: ONNX opset version
    """
    # Initialize model
    model = Classifier(BasicBlock, [2, 2, 2, 2])
    
    # Load weights
    model.load_state_dict(torch.load(pytorch_model_path, map_location='cpu'))
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=opset_version,
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model successfully exported to {onnx_model_path}")
    return onnx_model_path

if __name__ == "__main__":
    convert_pytorch_to_onnx()