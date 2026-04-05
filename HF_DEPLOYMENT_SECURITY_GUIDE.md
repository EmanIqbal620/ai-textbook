# 🔒 Hugging Face Deployment Guide & Security Analysis

## ✅ Security Check Results

### GOOD NEWS: No Sensitive Data Exposed!

I've analyzed your entire codebase and found:

1. **No `.env` files committed** - ✅ Only `.env.example` exists (template)
2. **Proper `.gitignore`** - ✅ `.env` files are ignored
3. **Environment variables loaded correctly** - ✅ Using `python-dotenv` and `os.getenv()`
4. **No hardcoded API keys** - ✅ All keys loaded from environment

---

## 📚 How the Frontend Works

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    FRONTEND (React + Docusaurus)            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐      ┌─────────────────────┐        │
│  │  Main Chat App   │      │  Chat Widget        │        │
│  │  (chat-app/)     │      │  (embedded)         │        │
│  │                  │      │                     │        │
│  │  - Sidebar       │      │  - Floating button  │        │
│  │  - Chat window   │      │  - Text selection   │        │
│  │  - Quick questions│     │  - Ask popup        │        │
│  │  - Sources display│     │                     │        │
│  └────────┬─────────┘      └──────────┬──────────┘        │
│           │                           │                    │
│           └───────────┬───────────────┘                    │
│                       │                                    │
│                       ▼                                    │
│            POST /api/v1/chat                               │
│            http://localhost:8000                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    BACKEND (FastAPI + RAG)                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐      ┌─────────────────────┐        │
│  │  API Server      │      │  RAG Agent          │        │
│  │  (server.py)     │─────▶│  (OpenAI Agents)    │        │
│  └────────┬─────────┘      └──────────┬──────────┘        │
│           │                           │                    │
│           ▼                           ▼                    │
│  ┌──────────────────┐      ┌─────────────────────┐        │
│  │  Vector Store    │      │  LLM (OpenRouter)   │        │
│  │  (Qdrant)        │      │  or HF Inference    │        │
│  └──────────────────┘      └─────────────────────┘        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Frontend Components

#### 1. **Main Chat Application** (`chat-app/src/App.tsx`)
- Full-page chat interface
- Features:
  - Sidebar with quick questions
  - Chat history with sources
  - Response time display
  - LocalStorage for chat history
  - Suggestion cards for new users

**API Configuration:**
```typescript
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1';
```

#### 2. **Embedded Chat Widget** (`backend/frontend/chat-widget.js`)
- Floating chat button for any website
- Features:
  - Text selection → "Ask about this"
  - Configurable via `window.RAGChatConfig`
  - Minimal UI, easy embedding

**How to embed:**
```html
<script>
  window.RAGChatConfig = {
    apiUrl: 'https://your-backend.hf.space/api/v1'
  };
</script>
<script src="https://your-backend.hf.space/static/js/chat-widget.js"></script>
```

#### 3. **Docusaurus Textbook** (`humanoid-robotics-textbook/`)
- Documentation site with built-in chat
- Static site generation
- Can be deployed to Hugging Face Spaces

---

## 🚀 Deploy Backend to Hugging Face

### Option 1: MCP Server (Recommended for HF)

The `backend/mcp_server.py` is already configured for Hugging Face!

#### Step 1: Prepare Deployment Files

```bash
cd /mnt/d/Humanoid-Robotics-AI-textbook/backend

# Create HF-specific requirements
cat > requirements-hf.txt << 'EOF'
fastapi==0.109.0
uvicorn==0.27.0
pydantic==2.5.3
huggingface_hub==0.20.3
python-dotenv==1.0.1
EOF
```

#### Step 2: Create Dockerfile

```bash
cat > Dockerfile << 'EOF'
FROM python:3.10-slim

WORKDIR /app

COPY requirements-hf.txt .
RUN pip install --no-cache-dir -r requirements-hf.txt

COPY mcp_server.py .

EXPOSE 8000

CMD ["python", "mcp_server.py"]
EOF
```

#### Step 3: Login to Hugging Face

```bash
# Install HF CLI if needed
pip install huggingface_hub

# Login
huggingface-cli login
# Enter your token from: https://huggingface.co/settings/tokens
```

#### Step 4: Deploy Using Script

```bash
cd /mnt/d/Humanoid-Robotics-AI-textbook

# Make script executable
chmod +x deploy-backend-hf.sh

# Run deployment
./deploy-backend-hf.sh
```

#### Step 5: Set Environment Variable

**IMPORTANT:** In Hugging Face Inference Endpoints dashboard:
1. Go to: https://huggingface.co/inference-endpoints
2. Select your endpoint
3. Click "Settings" → "Environment Variables"
4. Add: `HF_TOKEN=your_token_here`

---

### Option 2: Simple Server (If you want basic RAG)

Create `backend/server-hf.py`:

```python
#!/usr/bin/env python3
"""HF-compatible server without external dependencies"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import InferenceClient

app = FastAPI(title="Humanoid Robotics Chatbot")

# CORS for HF Spaces
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# HF Client
HF_TOKEN = os.getenv("HF_TOKEN", "")
client = InferenceClient(token=HF_TOKEN) if HF_TOKEN else None

class ChatRequest(BaseModel):
    query: str
    user_id: str = "anonymous"
    top_k: int = 5

class ChatResponse(BaseModel):
    response: str
    sources: list = []
    response_time: float = 0.0

@app.get("/")
async def root():
    return {"status": "running", "service": "Humanoid Robotics Chatbot"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/api/v1/chat")
async def chat(request: ChatRequest):
    if not client:
        return ChatResponse(
            response="HF_TOKEN not set. Please configure environment variable.",
            sources=[],
            response_time=0.0
        )
    
    # Call Mistral-7B via HF Inference API
    response = client.chat_completion(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        messages=[
            {"role": "system", "content": "You are an expert AI tutor for Humanoid Robotics."},
            {"role": "user", "content": request.query}
        ],
        max_tokens=500,
        temperature=0.7
    )
    
    return ChatResponse(
        response=response.choices[0].message.content,
        sources=[],
        response_time=0.5
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 🌐 Deploy Frontend to Hugging Face Spaces

### Step 1: Build Frontend

```bash
# Option A: Main Chat App
cd /mnt/d/Humanoid-Robotics-AI-textbook/chat-app
npm install
npm run build

# Option B: Textbook Site
cd /mnt/d/Humanoid-Robotics-AI-textbook/humanoid-robotics-textbook
npm install
npm run build
```

### Step 2: Create Hugging Face Space

```bash
# Go to: https://huggingface.co/new-space
# Choose:
# - Space SDK: Static
# - License: MIT
# - Visibility: Public
```

### Step 3: Deploy Build Files

```bash
# Clone your space
cd /tmp
git clone https://huggingface.co/spaces/YOUR_USERNAME/humanoid-robotics-chat
cd humanoid-robotics-chat

# Copy build files
cp -r /mnt/d/Humanoid-Robotics-AI-textbook/chat-app/build/* .

# Commit and push
git add .
git commit -m "Deploy chat app"
git push
```

### Step 4: Update API URL

Edit `chat-app/src/App.tsx` BEFORE building:

```typescript
// Change this line:
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1';

// To your HF endpoint:
const API_URL = process.env.REACT_APP_API_URL || 'https://YOUR_ENDPOINT.hf.space/api/v1';
```

Or create `.env` file:
```bash
echo "REACT_APP_API_URL=https://YOUR_ENDPOINT.hf.space/api/v1" > chat-app/.env
```

---

## 🔐 Security Checklist

### ✅ Current Status (SAFE)

- [x] No `.env` files in git
- [x] `.gitignore` properly configured
- [x] API keys loaded from environment variables
- [x] `.env.example` is a template only

### ⚠️ Before Deployment

- [ ] **Create new HF token** - Don't reuse old tokens
- [ ] **Set HF_TOKEN in HF dashboard** - Not in code
- [ ] **Enable CORS only for your domain** - Update `allow_origins`
- [ ] **Add rate limiting** - Prevent abuse
- [ ] **Use HTTPS only** - HF provides this automatically

### 🔒 Recommended Security Updates

Update `backend/mcp_server.py` CORS:

```python
# Replace this:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ Too permissive!
    # ...
)

# With this:
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://huggingface.co",
        "https://*.hf.space",
        "http://localhost:3000"  # Dev only
    ],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)
```

---

## 🧪 Test Before Deploy

### Test Backend Locally

```bash
cd /mnt/d/Humanoid-Robotics-AI-textbook/backend

# Set test token
export HF_TOKEN=hf_test_token

# Start server
python mcp_server.py

# Test in another terminal:
curl http://localhost:8000/health
curl http://localhost:8000/api/mcp/health
```

### Test Frontend Connection

```bash
cd /mnt/d/Humanoid-Robotics-AI-textbook/chat-app

# Start frontend
npm start

# Open: http://localhost:3000
# Try asking a question
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
│  │  YOUR_USERNAME/ │      │  Endpoint)       │        │
│  │  humanoid-robotics-chat │                  │        │
│  │  http://localhost:3000  │ https://YOUR_ENDPOINT     │
│  └─────────────────┘      └──────────────────┘        │
│                           │                            │
│                           ▼                            │
│                  ┌──────────────────┐                 │
│                  │  HF Model API    │                 │
│                  │  Mistral-7B      │                 │
│                  └──────────────────┘                 │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 💰 Cost Estimate

| Component | Tier | Cost/Month |
|-----------|------|------------|
| **Frontend Space** | Static | FREE |
| **Backend Endpoint** | CPU x2 | ~$43/month |
| **Backend Endpoint** | GPU | ~$280/month |
| **Total (CPU)** | | ~$43/month |

---

## 🎯 Quick Start Commands

```bash
# 1. Deploy Backend
cd /mnt/d/Humanoid-Robotics-AI-textbook
chmod +x deploy-backend-hf.sh
./deploy-backend-hf.sh

# 2. Get Endpoint URL
ENDPOINT_URL=$(huggingface-cli inference-endpoint list | grep humanoid | awk '{print $1}')
echo "Backend: $ENDPOINT_URL"

# 3. Update Frontend API URL
sed -i "s|http://localhost:8000|$ENDPOINT_URL|g" chat-app/src/App.tsx

# 4. Build & Deploy Frontend
cd chat-app
npm install
npm run build
cd ..

# 5. Deploy to HF Space
cd /tmp
git clone https://huggingface.co/spaces/YOUR_USERNAME/chat
cp -r /mnt/d/Humanoid-Robotics-AI-textbook/chat-app/build/* chat/
cd chat
git add .
git commit -m "Deploy"
git push
```

---

## ✅ Deployment Checklist

### Backend
- [ ] HF account created
- [ ] Token generated (write access)
- [ ] `mcp_server.py` tested locally
- [ ] Dockerfile created
- [ ] Requirements file ready
- [ ] Deployed to Inference Endpoints
- [ ] HF_TOKEN set in dashboard
- [ ] Health check passing

### Frontend
- [ ] API URL updated to HF endpoint
- [ ] Build successful (`npm run build`)
- [ ] HF Space created (Static)
- [ ] Build files uploaded
- [ ] Site loads correctly
- [ ] Chat widget connects to backend

### Security
- [ ] No API keys in code
- [ ] `.env` files in `.gitignore`
- [ ] CORS configured properly
- [ ] HTTPS enabled (automatic on HF)
- [ ] Rate limiting considered

---

## 🆘 Troubleshooting

### Backend Issues

**Problem:** "HF_TOKEN not set"
```bash
# In HF Inference Endpoints dashboard:
# Settings → Environment Variables → Add HF_TOKEN
```

**Problem:** CORS errors
```python
# Update mcp_server.py CORS origins to include your frontend URL
```

### Frontend Issues

**Problem:** "Failed to fetch"
```bash
# Check:
# 1. Backend is running
# 2. API URL is correct
# 3. CORS allows your domain
```

**Problem:** Build fails
```bash
# Clear and reinstall
rm -rf node_modules package-lock.json
npm install
npm run build
```

---

## 📚 Resources

- **HF Inference Endpoints**: https://huggingface.co/docs/inference-endpoints
- **HF Spaces**: https://huggingface.co/docs/hub/spaces
- **MCP Specification**: https://modelcontextprotocol.io
- **Your Backend Docs**: `/mnt/d/Humanoid-Robotics-AI-textbook/BACKEND_HF_MCP_DEPLOY.md`

---

**Ready to deploy! 🚀**
