#!/bin/bash

# Cerebrium deployment script
set -e

echo "Deploying to Cerebrium..."

# Check if required files exist
required_files=("main.py" "model.py" "mtailor_classifier.onnx" "cerebrium.toml")
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "Error: Required file $file not found"
        exit 1
    fi
done

# Install Cerebrium CLI if not present
if ! command -v cerebrium &> /dev/null; then
    echo "Installing Cerebrium CLI..."
    pip install cerebrium
fi

# Login to Cerebrium (if not already logged in)
echo "Checking Cerebrium authentication..."
if ! cerebrium whoami &> /dev/null; then
    echo "Please login to Cerebrium:"
    cerebrium login
fi

# Deploy the model
echo "Deploying model..."
cerebrium deploy

echo "Deployment complete!"
echo ""
echo "Next steps:"
echo "1. Note your API endpoint URL from the deployment output"
echo "2. Get your API key from Cerebrium dashboard"
echo "3. Test your deployment:"
echo "   python test_server.py --api-url YOUR_API_URL --api-key YOUR_API_KEY --image test_image.jpg"
echo "4. Run comprehensive tests:"
echo "   python test_server.py --api-url YOUR_API_URL --api-key YOUR_API_KEY --image test_image.jpg --comprehensive"