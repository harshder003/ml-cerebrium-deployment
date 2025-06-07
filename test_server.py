import argparse
import requests
import json
import os
import base64
from io import BytesIO
from PIL import Image

def encode_image_to_base64(image_path):
    """Encode a local image file or a remote image URL to base64 string."""
    if image_path.startswith('http://') or image_path.startswith('https://'):
        # Download image from URL
        response = requests.get(image_path)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert('RGB')
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    else:
        # Local file
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        with open(image_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

def main():
    parser = argparse.ArgumentParser(description="Test Cerebrium model endpoint with an image.")
    parser.add_argument('--api-url', type=str, required=True, help='Cerebrium endpoint URL (should end with /run)')
    parser.add_argument('--api-key', type=str, required=True, help='Bearer token for Cerebrium')
    parser.add_argument('--image', type=str, required=True, help='Path to local image or image URL')
    parser.add_argument('--test-mode', action='store_true', help='Run health check instead of prediction')
    args = parser.parse_args()

    headers = {
        'Authorization': f'Bearer {args.api_key}',
        'Content-Type': 'application/json'
    }

    if args.test_mode:
        payload = json.dumps({"test_mode": True})
    else:
        # Use image_base64 as the key, as your deployed function expects
        try:
            image_base64 = encode_image_to_base64(args.image)
        except Exception as e:
            print(f"Exception: {e}\n\nPredicted Class ID: Error")
            return
        payload = json.dumps({"image_base64": image_base64})

    try:
        response = requests.post(args.api_url, headers=headers, data=payload)
        print("Status Code:", response.status_code)
        print("Response:", response.text)
    except Exception as e:
        print(f"Exception during request: {e}\n\nPredicted Class ID: Error")

if __name__ == "__main__":
    main()