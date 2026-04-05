# 🤖 Hugging Face MCP Backend Deployment

## Overview

Deploy the AI Chatbot backend to Hugging Face Inference Endpoints with MCP (Model Context Protocol) integration.

---

## 📋 Prerequisites

1. **Hugging Face Account**: https://huggingface.co
2. **Hugging Face Pro Account** (for Inference Endpoints): $9/month
3. **Git LFS**: For model files
4. **Python 3.10+**: For backend

---

## 🚀 Quick Deploy

### Step 1: Install Hugging Face Dependencies

```bash
# Install Hugging Face Hub
pip install huggingface_hub inference-endpoints

# Login to Hugging Face
huggingface-cli login

# Enter your token (get from: https://huggingface.co/settings/tokens)
```

### Step 2: Create Inference Endpoint

```bash
# Create endpoint via CLI
huggingface-cli inference-endpoint create \
    --name humanoid-robotics-chatbot \
    --repository mistralai/Mistral-7B-Instruct-v0.2 \
    --vendor aws \
    --region us-east-1 \
    --instance-type cpu \
    --instance-size x2 \
    --framework pytorch
```

### Step 3: Deploy Backend Code

```bash
cd /mnt/d/Humanoid-Robotics-AI-textbook/backend

# Create requirements.txt for HF
cat > requirements-hf.txt << EOF
fastapi==0.109.0
uvicorn==0.27.0
pydantic==2.5.3
sentence-transformers==2.2.2
qdrant-client==1.7.0
huggingface_hub==0.20.3
EOF

# Deploy to Hugging Face
huggingface-cli upload \
    --repo-type model \
    YOUR_USERNAME/humanoid-robotics-backend \
    . \
    .
```

---

## 🔧 MCP (Model Context Protocol) Configuration

### What is MCP?

MCP allows your application to communicate with AI models using a standardized protocol.

### Install MCP

```bash
pip install mcp huggingface_hub
```

### Configure MCP

Create `mcp-config.json`:

```json
{
  "version": "1.0",
  "huggingface": {
    "endpoint_url": "https://YOUR_ENDPOINT.hf.space",
    "token": "hf_your_token",
    "models": {
      "chat": {
        "model": "mistralai/Mistral-7B-Instruct-v0.2",
        "max_tokens": 500,
        "temperature": 0.7,
        "top_p": 0.95
      },
      "embeddings": {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "dimension": 384
      }
    },
    "vector_store": {
      "provider": "qdrant",
      "url": "https://YOUR_QDRANT.hf.space",
      "collection": "humanoid_robotics"
    }
  },
  "chatbot": {
    "system_prompt": "You are an expert AI tutor for Humanoid Robotics. Help students learn ROS2, simulation, AI, and VLA systems.",
    "context_window": 4096,
    "stream": true
  }
}
```

### MCP Server Code

Create `backend/mcp_server.py`:

```python
#!/usr/bin/env python3
"""
MCP Server for Hugging Face Inference
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huggingface_hub import InferenceClient
import json
import os

app = FastAPI(title="Humanoid Robotics MCP Server")

# Load MCP config
with open('mcp-config.json', 'r') as f:
    MCP_CONFIG = json.load(f)

# Initialize HF client
hf_client = InferenceClient(token=os.getenv('HF_TOKEN'))

class ChatRequest(BaseModel):
    message: str
    conversation_history: list = []

class ChatResponse(BaseModel):
    response: str
    sources: list
    confidence: float

@app.post("/api/mcp/chat")
async def mcp_chat(request: ChatRequest) -> ChatResponse:
    """MCP-compliant chat endpoint"""
    
    # Build prompt with context
    system_prompt = MCP_CONFIG['chatbot']['system_prompt']
    
    # Format conversation for Mistral
    messages = [
        {"role": "system", "content": system_prompt},
        *request.conversation_history,
        {"role": "user", "content": request.message}
    ]
    
    # Call Hugging Face Inference API
    response = hf_client.chat_completion(
        model=MCP_CONFIG['huggingface']['models']['chat']['model'],
        messages=messages,
        max_tokens=MCP_CONFIG['huggingface']['models']['chat']['max_tokens'],
        temperature=MCP_CONFIG['huggingface']['models']['chat']['temperature']
    )
    
    return ChatResponse(
        response=response.choices[0].message.content,
        sources=[],
        confidence=0.95
    )

@app.get("/api/mcp/health")
async def health_check():
    """MCP health check"""
    return {"status": "healthy", "mcp_version": "1.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 📦 Backend Deployment Options

### Option 1: Hugging Face Inference Endpoints (Recommended)

**Pros:**
- ✅ Managed infrastructure
- ✅ Auto-scaling
- ✅ Built-in monitoring
- ✅ Pay-per-use

**Cons:**
- 💰 $0.06/hour (CPU) or $0.39/hour (GPU)

**Deploy:**

```bash
# Create endpoint
huggingface-cli inference-endpoint create \
    --name humanoid-robotics-backend \
    --repository YOUR_USERNAME/humanoid-robotics-backend \
    --vendor aws \
    --region us-east-1 \
    --instance-type cpu \
    --instance-size x2

# Get endpoint URL
ENDPOINT_URL=$(huggingface-cli inference-endpoint list | grep humanoid-robotics-backend | awk '{print $1}')

echo "Backend deployed at: $ENDPOINT_URL"
```

### Option 2: Hugging Face Spaces (Backend + Frontend)

**Pros:**
- ✅ Free for static sites
- ✅ Easy deployment
- ✅ Integrated with GitHub

**Cons:**
- ⚠️ Limited compute for backend

**Deploy:**

```bash
# Create Space
huggingface-cli repo create \
    --type space \
    --space_sdk static \
    YOUR_USERNAME/humanoid-robotics-textbook

# Deploy
huggingface-cli upload \
    --repo-type space \
    YOUR_USERNAME/humanoid-robotics-textbook \
    ./build \
    .
```

### Option 3: Hugging Face AutoTrain

For custom model fine-tuning:

```bash
pip install autotrain-advanced

autotrain llm \
    --train \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --data-path ./training_data \
    --text-column text \
    --lr 2e-5 \
    --batch_size 4 \
    --epochs 3 \
    --trainer sft
```

---

## 🔗 Connect Frontend to HF Backend

### Update Chat Widget

Edit `humanoid-robotics-textbook/static/js/chat-widget.js`:

```javascript
// Replace this line:
const API_BASE_URL = 'http://localhost:8000';

// With your Hugging Face Endpoint:
const API_BASE_URL = 'https://YOUR_ENDPOINT.hf.space';
```

### Update MCP Config

```javascript
const MCP_CONFIG = {
  endpoint: 'https://YOUR_ENDPOINT.hf.space/api/mcp',
  model: 'mistralai/Mistral-7B-Instruct-v0.2',
  streaming: true
};
```

---

## 📊 Deployment Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Hugging Face Platform                                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────┐      ┌──────────────────┐        │
│  │  Frontend       │      │  Backend         │        │
│  │  (Static Space) │─────▶│  (Inference      │        │
│  │  http://localhost:3000 │   Endpoint)      │        │
│  └─────────────────┘      │  http://YOUR_ENDPOINT     │
│                           └──────────────────┘        │
│                                                         │
│  ┌─────────────────┐      ┌──────────────────┐        │
│  │  Vector Store   │      │  MCP Server      │        │
│  │  (Qdrant)       │◀────▶│  (Orchestrator)  │        │
│  └─────────────────┘      └──────────────────┘        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 🔐 Security Configuration

### Environment Variables

Create `.env`:

```env
# Hugging Face
HF_TOKEN=hf_your_token_here
HF_ENDPOINT_URL=https://YOUR_ENDPOINT.hf.space

# MCP Configuration
MCP_VERSION=1.0
MCP_SYSTEM_PROMPT=You are an expert AI tutor for Humanoid Robotics

# Rate Limiting
RATE_LIMIT_PER_MINUTE=60
MAX_TOKENS=500

# CORS (for frontend)
CORS_ORIGINS=https://YOUR_USERNAME-humanoid-robotics-textbook.hf.space
```

### CORS Configuration

Update `backend/server.py`:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://YOUR_USERNAME-humanoid-robotics-textbook.hf.space",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 📈 Monitoring & Scaling

### Monitor Endpoint

```bash
# Check endpoint status
huggingface-cli inference-endpoint list

# View logs
huggingface-cli inference-endpoint logs \
    --name humanoid-robotics-backend
```

### Auto-Scaling

```yaml
# scaling-config.yaml
autoscaling:
  min_instances: 1
  max_instances: 5
  scale_on:
    - metric: cpu_utilization
      threshold: 70
    - metric: request_latency
      threshold: 500ms
```

---

## 🧪 Testing

### Test MCP Endpoint

```bash
# Health check
curl https://YOUR_ENDPOINT.hf.space/api/mcp/health

# Chat test
curl -X POST https://YOUR_ENDPOINT.hf.space/api/mcp/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer hf_your_token" \
  -d '{
    "message": "What is ROS2?",
    "conversation_history": []
  }'
```

### Test Frontend Integration

```bash
# Start frontend
cd humanoid-robotics-textbook
npm run start

# Open: http://localhost:3000
# Click chat widget and test
```

---

## 💰 Cost Estimate

| Component | Tier | Cost/Month |
|-----------|------|------------|
| **Frontend Space** | Static | FREE |
| **Backend Endpoint** | CPU x2 | ~$43/month |
| **Backend Endpoint** | GPU | ~$280/month |
| **Vector Store** | Qdrant Cloud | ~$25/month |
| **Total (CPU)** | | ~$68/month |

---

## 🚀 One-Click Deploy Script

Create `deploy-backend-hf.sh`:

```bash
#!/bin/bash

echo "🚀 Deploying Backend to Hugging Face..."

# Configuration
HF_TOKEN="${HF_TOKEN:-}"
ENDPOINT_NAME="humanoid-robotics-backend"

# Check login
if ! huggingface-cli whoami &> /dev/null; then
    echo "🔐 Logging in to Hugging Face..."
    huggingface-cli login
fi

# Create endpoint
echo "📦 Creating Inference Endpoint..."
huggingface-cli inference-endpoint create \
    --name $ENDPOINT_NAME \
    --repository mistralai/Mistral-7B-Instruct-v0.2 \
    --vendor aws \
    --region us-east-1 \
    --instance-type cpu \
    --instance-size x2

# Get endpoint URL
ENDPOINT_URL=$(huggingface-cli inference-endpoint list | grep $ENDPOINT_NAME | awk '{print $1}')

echo "✅ Backend deployed at: $ENDPOINT_URL"
echo ""
echo "📝 Update frontend configuration:"
echo "   Edit: static/js/chat-widget.js"
echo "   Set: API_BASE_URL = '$ENDPOINT_URL'"
```

---

## 📚 Resources

- **Hugging Face Docs**: https://huggingface.co/docs/inference-endpoints
- **MCP Specification**: https://modelcontextprotocol.io
- **Pricing**: https://huggingface.co/pricing
- **Support**: https://huggingface.co/support

---

## ✅ Deployment Checklist

- [ ] Hugging Face account created
- [ ] Token generated
- [ ] Inference Endpoint created
- [ ] Backend code uploaded
- [ ] MCP config set
- [ ] Frontend API URL updated
- [ ] CORS configured
- [ ] Health check passing
- [ ] Chat test successful
- [ ] Monitoring enabled

---

**Ready to deploy backend to Hugging Face with MCP!** 🚀
