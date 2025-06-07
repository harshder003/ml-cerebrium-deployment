import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms

# Preprocess your image (must match your training/preprocessing!)
def preprocess_image(img_path):
    img = Image.open(img_path)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = preprocess(img)
    return img_tensor.unsqueeze(0).numpy()  # shape: (1, 3, 224, 224)

# Load model and run inference
ort_session = ort.InferenceSession("mtailor_classifier.onnx")
input_image = preprocess_image("n01440764_tench.jpeg")

# Get input and output names
input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name

# Run inference
outputs = ort_session.run([output_name], {input_name: input_image})
prediction = np.argmax(outputs[0])
print("Predicted class:", prediction)