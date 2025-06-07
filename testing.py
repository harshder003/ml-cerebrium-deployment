import requests
import json
import base64
import os

api_url = "https://api.cortex.cerebrium.ai/v4/p-0aee9e27/mtailor-classifier/init"
api_key = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJwcm9qZWN0SWQiOiJwLTBhZWU5ZTI3IiwiaWF0IjoxNzQ5MzA5NDk3LCJleHAiOjIwNjQ4ODU0OTd9.se-fDvTihZo2p4TInAWtJYZ5WpFWJQPK4RRjJZ0wnhXEJqoRhueKHpS2rvNLoY4bHZyrauez4UbEU_J5WvmQw6X6lwPK4RAgB6VBGN2ZcQadKVu60YLMEy3JfJgE5dQd6VQJQGGTv1TxVBQEfkg-C9VXlbH5ind1eW5xniZewyJfaHy2Znn0Fh8r4quaNmHOQcYY6I8RB1uGVuASEQI1OTbu-ihR40J5iUASVbvtKf0yVh7c18Ws77fStWDFVx8AstqNqRL93uHlJp1PJLnx0Z7jf_xqimKB7EgL48XscLZ-OeRVRdcp2eI1W80LlWDZ-ZCd8FYDwOXs6sNllq4tkQ"
image_path = r"D:\ml-cerebrium-deployment\n01440764_tench.jpeg"

if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image file not found: {image_path}")

with open(image_path, "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")

payload = json.dumps({"image_base64": image_base64})

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

response = requests.post(api_url, headers=headers, data=payload)
print("Status Code:", response.status_code)
print("Response:", response.text)