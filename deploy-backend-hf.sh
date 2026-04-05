#!/bin/bash

# Backend MCP Deployment to Hugging Face
# Run this to deploy the AI chatbot backend

set -e

echo "🚀 Backend MCP Deployment to Hugging Face"
echo "=========================================="

# Configuration
BACKEND_DIR="backend"
HF_REPO="${HF_REPO:-humanoid-robotics/humanoid-robotics-backend}"
ENDPOINT_NAME="humanoid-robotics-mcp"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check if Hugging Face CLI is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo -e "${RED}❌ Hugging Face CLI not found${NC}"
    echo "Installing..."
    pip install huggingface_hub inference-endpoints
fi

# Check if logged in
if ! huggingface-cli whoami &> /dev/null; then
    echo -e "${BLUE}🔐 Logging in to Hugging Face...${NC}"
    huggingface-cli login
fi

# Navigate to backend directory
cd "$(dirname "$0")/$BACKEND_DIR"

# Create requirements file for HF
echo -e "${BLUE}📦 Creating requirements...${NC}"
cat > requirements-hf.txt << EOF
fastapi==0.109.0
uvicorn==0.27.0
pydantic==2.5.3
huggingface_hub==0.20.3
sentence-transformers==2.2.2
EOF

# Create Dockerfile for HF Inference
echo -e "${BLUE}🐳 Creating Dockerfile...${NC}"
cat > Dockerfile << 'EOF'
FROM python:3.10-slim

WORKDIR /app

COPY requirements-hf.txt .
RUN pip install --no-cache-dir -r requirements-hf.txt

COPY mcp_server.py .
COPY mcp-config.json .

EXPOSE 8000

CMD ["python", "mcp_server.py"]
EOF

# Create MCP config
echo -e "${BLUE}⚙️ Creating MCP config...${NC}"
cat > mcp-config.json << 'EOF'
{
  "version": "1.0",
  "model": {
    "name": "mistralai/Mistral-7B-Instruct-v0.2",
    "max_tokens": 500,
    "temperature": 0.7,
    "top_p": 0.95
  },
  "system_prompt": "You are an expert AI tutor for Humanoid Robotics.",
  "endpoints": {
    "chat": "/api/mcp/chat",
    "health": "/api/mcp/health",
    "embed": "/api/mcp/embed"
  }
}
EOF

# Upload to Hugging Face
echo -e "${BLUE}📤 Uploading to Hugging Face...${NC}"
huggingface-cli upload \
    --repo-type model \
    "$HF_REPO" \
    . \
    . \
    --token "${HF_TOKEN:-}"

echo -e "${GREEN}✅ Code uploaded!${NC}"

# Create Inference Endpoint
echo -e "${BLUE}🔧 Creating Inference Endpoint...${NC}"

ENDPOINT_URL=$(huggingface-cli inference-endpoint create \
    --name "$ENDPOINT_NAME" \
    --repository "$HF_REPO" \
    --vendor aws \
    --region us-east-1 \
    --instance-type cpu \
    --instance-size x2 \
    --framework pytorch \
    --token "${HF_TOKEN:-}" 2>&1 | grep -oP 'https://[^\s]+' | head -1)

if [ -z "$ENDPOINT_URL" ]; then
    echo -e "${YELLOW}⚠️ Endpoint creation may have failed. Check manually.${NC}"
    echo "Visit: https://huggingface.co/inference-endpoints"
else
    echo -e "${GREEN}✅ Endpoint created!${NC}"
    echo ""
    echo "🌐 Backend URL: $ENDPOINT_URL"
    echo ""
    echo "📝 Update frontend configuration:"
    echo "   Edit: static/js/chat-widget.js"
    echo "   Set: API_BASE_URL = '$ENDPOINT_URL'"
fi

# Test endpoint
echo ""
echo -e "${BLUE}🧪 Testing endpoint...${NC}"
sleep 30  # Wait for endpoint to initialize

HEALTH_URL="$ENDPOINT_URL/api/mcp/health"
echo "Testing: $HEALTH_URL"

curl -s "$HEALTH_URL" || echo -e "${YELLOW}⚠️ Endpoint still initializing. Try again in 2 minutes.${NC}"

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}✅ DEPLOYMENT COMPLETE!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "📊 Monitor your endpoint:"
echo "   https://huggingface.co/inference-endpoints"
echo ""
echo "📝 API Documentation:"
echo "   $ENDPOINT_URL/docs"
echo ""
echo "🔧 Update frontend:"
echo "   1. Edit: static/js/chat-widget.js"
echo "   2. Set: API_BASE_URL = '$ENDPOINT_URL'"
echo "   3. Rebuild: npm run build"
echo ""
