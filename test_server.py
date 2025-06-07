import argparse
import requests
import json
import os
import base64
from io import BytesIO
from PIL import Image

API_KEY = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJwcm9qZWN0SWQiOiJwLTBhZWU5ZTI3IiwiaWF0IjoxNzQ5MzA5NDk3LCJleHAiOjIwNjQ4ODU0OTd9.se-fDvTihZo2p4TInAWtJYZ5WpFWJQPK4RRjJZ0wnhXEJqoRhueKHpS2rvNLoY4bHZyrauez4UbEU_J5WvmQw6X6lwPK4RAgB6VBGN2ZcQadKVu60YLMEy3JfJgE5dQd6VQJQGGTv1TxVBQEfkg-C9VXlbH5ind1eW5xniZewyJfaHy2Znn0Fh8r4quaNmHOQcYY6I8RB1uGVuASEQI1OTbu-ihR40J5iUASVbvtKf0yVh7c18Ws77fStWDFVx8AstqNqRL93uHlJp1PJLnx0Z7jf_xqimKB7EgL48XscLZ-OeRVRdcp2eI1W80LlWDZ-ZCd8FYDwOXs6sNllq4tkQ"
BASE_URL = "https://api.cortex.cerebrium.ai/v4/p-0aee9e27/mtailor-classifier"

def health_check():
    url = f"{BASE_URL}/run_health_check"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.post(url, headers=headers)
    print("Health Check")
    print("Status Code:", response.status_code)
    print("Response:", response.text)

def predict_image(image_base64):
    url = f"{BASE_URL}/predict"
    payload = json.dumps({'item': {"image_base64": image_base64}})
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.post(url, headers=headers, data=payload)
    print("Image Prediction")
    print("Status Code:", response.status_code)
    print("Response:", response.text)

def encode_local_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def encode_remote_image(image_url):
    response_img = requests.get(image_url)
    response_img.raise_for_status()
    image = Image.open(BytesIO(response_img.content)).convert("RGB")
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Cerebrium API (health check, local image, remote image)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--health', action='store_true', help='Run health check')
    group.add_argument('--local-image', type=str, help='Path to local image file')
    group.add_argument('--remote-image', type=str, help='URL to remote image file (RAW URL)')
    args = parser.parse_args()

    if args.health:
        health_check()
    elif args.local_image:
        image_base64 = encode_local_image(args.local_image)
        predict_image(image_base64)
    elif args.remote_image:
        image_base64 = encode_remote_image(args.remote_image)
        predict_image(image_base64)