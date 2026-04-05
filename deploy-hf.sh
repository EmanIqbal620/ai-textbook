#!/bin/bash

# Hugging Face Deployment Script
# Run this to deploy your Humanoid Robotics Textbook to Hugging Face Spaces

set -e

echo "🚀 Hugging Face Deployment Script"
echo "=================================="

# Configuration
HF_SPACE_ID="${HF_SPACE_ID:-humanoid-robotics/humanoid-robotics-textbook}"
BUILD_DIR="humanoid-robotics-textbook/build"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if Hugging Face CLI is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo -e "${RED}❌ Hugging Face CLI not found${NC}"
    echo "Installing..."
    pip install huggingface_hub
fi

# Check if logged in
if ! huggingface-cli whoami &> /dev/null; then
    echo -e "${BLUE}🔐 Logging in to Hugging Face...${NC}"
    huggingface-cli login
fi

# Navigate to project root
cd "$(dirname "$0")"

# Build the site
echo -e "${BLUE}🔨 Building site...${NC}"
cd humanoid-robotics-textbook
npm install
npm run build
cd ..

# Check if build was successful
if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${RED}❌ Build failed! Build directory not found.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Build successful!${NC}"

# Deploy to Hugging Face
echo -e "${BLUE}📤 Deploying to Hugging Face Spaces...${NC}"
echo "Space ID: $HF_SPACE_ID"

huggingface-cli upload \
    --repo-type space \
    "$HF_SPACE_ID" \
    "$BUILD_DIR" \
    . \
    --token "${HF_TOKEN:-}"

echo -e "${GREEN}✅ Deployment complete!${NC}"
echo ""
echo "🌐 Your site is now live at:"
echo "   https://huggingface.co/spaces/$HF_SPACE_ID"
echo ""
echo "📊 Monitor deployment:"
echo "   https://huggingface.co/spaces/$HF_SPACE_ID/tree/main"
