#!/bin/bash

# Hugging Face Backend Deployment Script
# Usage: ./deploy-to-hf.sh

set -e

echo "🚀 Deploying Backend to Hugging Face Inference Endpoints"
echo "=========================================================="

# Configuration
REPO_NAME="humanoid-robotics-backend"
ENDPOINT_NAME="humanoid-robotics-api"
HF_USERNAME=""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Helper functions
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    print_error "huggingface-cli not found. Installing..."
    pip install huggingface_hub
fi

# Check if logged in
print_info "Checking Hugging Face login..."
if ! huggingface-cli whoami &> /dev/null; then
    print_info "Logging in to Hugging Face..."
    huggingface-cli login
    print_success "Logged in successfully!"
else
    print_success "Already logged in"
fi

# Get username
HF_USERNAME=$(huggingface-cli whoami)
print_info "Logged in as: $HF_USERNAME"

# Navigate to backend directory
cd "$(dirname "$0")"
BACKEND_DIR=$(pwd)

print_info "Backend directory: $BACKEND_DIR"

# Create repository
print_info "Creating repository: $REPO_NAME"
huggingface-cli repo create $REPO_NAME --type model 2>/dev/null || print_info "Repository already exists"

# Upload files
print_info "Uploading backend files..."
cd "$BACKEND_DIR"

# Create .hfignore file
cat > .hfignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/
.venv
.venv/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log

# Test files
test_*.py
*_test.py

# Development
.env
.env.local
*.bak
EOF

print_info "Uploading to Hugging Face..."
huggingface-cli upload \
    "$HF_USERNAME/$REPO_NAME" \
    . \
    . \
    --include "agent/**/*" \
    --include "api/**/*" \
    --include "retrieval/**/*" \
    --include "vector_store/**/*" \
    --include "simple_server.py" \
    --include "requirements-hf.txt" \
    --include "Dockerfile-hf" \
    --include ".env.example" \
    --include "README.md"

print_success "Files uploaded successfully!"

# Create Inference Endpoint
print_info ""
print_info "Creating Inference Endpoint..."
print_info "Please follow these steps:"
echo ""
echo "1. Go to: https://huggingface.co/inference-endpoints"
echo "2. Click 'Create Endpoint'"
echo "3. Select repository: $HF_USERNAME/$REPO_NAME"
echo "4. Choose configuration:"
echo "   - Vendor: AWS"
echo "   - Region: us-east-1"
echo "   - Instance Type: CPU"
echo "   - Instance Size: x2"
echo "5. Click 'Create Endpoint'"
echo ""

# Open in browser
if command -v start &> /dev/null; then
    # Windows
    start https://huggingface.co/inference-endpoints
elif command -v xdg-open &> /dev/null; then
    # Linux
    xdg-open https://huggingface.co/inference-endpoints
elif command -v open &> /dev/null; then
    # macOS
    open https://huggingface.co/inference-endpoints
fi

print_info ""
print_info "After endpoint is created:"
echo "1. Go to Settings → Environment Variables"
echo "2. Add HF_TOKEN with your Hugging Face token"
echo "3. Add other required variables from .env.example"
echo ""

# Get endpoint URL
print_info "Your endpoint URL will be:"
echo "https://$HF_USERNAME-$REPO_NAME.hf.space"
echo ""

print_success "Deployment script completed!"
print_info "Next steps:"
echo "1. Create Inference Endpoint (follow steps above)"
echo "2. Set environment variables in dashboard"
echo "3. Test endpoint: curl https://YOUR_ENDPOINT.hf.space/health"
echo ""

# Create frontend config file
print_info "Creating frontend configuration..."
cat > frontend-config.js << EOF
// Frontend Configuration for Hugging Face
// Copy this to: humanoid-robotics-textbook/static/js/config.js

window.HF_BACKEND_URL = 'https://$HF_USERNAME-$REPO_NAME.hf.space';

console.log('Backend URL:', window.HF_BACKEND_URL);
EOF

print_success "Frontend config created: frontend-config.js"
print_info "Copy this file to your frontend!"

echo ""
echo "=========================================================="
print_success "🎉 Deployment Ready!"
echo "=========================================================="
