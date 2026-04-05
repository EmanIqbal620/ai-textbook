# 🚀 Hugging Face Deployment Guide

## Quick Summary

You need to deploy **TWO separate services** to Hugging Face:

1. **Frontend** (Textbook + Chat UI) → Hugging Face Spaces (Static)
2. **Backend** (Chatbot API) → Hugging Face Inference Endpoints

---

## 📦 Files & Folders You Need

### 1. **For Frontend Deployment** (Textbook)

```
📁 humanoid-robotics-textbook/
├── docs/                    # All textbook content
├── src/
│   ├── css/
│   │   └── custom.final.css # Your custom styles
│   └── pages/
│       └── index.js         # Homepage with module cards
├── static/
│   └── js/
│       └── chat-widget.js   # Floating chat widget
├── docusaurus.config.ts     # Site configuration
├── sidebars.ts              # Navigation
├── package.json             # Dependencies
└── build/                   # AFTER running npm run build
```

**What to upload:**
- ✅ Entire `humanoid-robotics-textbook/` folder
- ✅ Build output to HF Spaces

---

### 2. **For Backend Deployment** (Chatbot API)

```
📁 backend/
├── agent/
│   └── rag_agent.py         # RAG logic with caching
├── api/
│   ├── chat.py              # Chat endpoint
│   └── models.py            # Pydantic models
├── retrieval/
│   └── retriever.py         # Vector search
├── vector_store/
│   └── qdrant_store.py      # Qdrant connection
├── simple_server.py         # ✅ USE THIS (simpler)
├── requirements.txt         # Python dependencies
├── .env.example             # Environment template
└── Dockerfile               # For container deployment
```

**What to upload:**
- ✅ `backend/agent/` folder
- ✅ `backend/api/` folder  
- ✅ `backend/retrieval/` folder
- ✅ `backend/vector_store/` folder
- ✅ `backend/simple_server.py` (or mcp_server.py)
- ✅ `backend/requirements.txt`
- ✅ `backend/Dockerfile`

---

## 🎯 Deployment Steps

### **STEP 1: Prepare Backend**

Create `backend/.env` file:
```env
# Get token from: https://huggingface.co/settings/tokens
HF_TOKEN=hf_your_token_here

# Optional: Qdrant for vector search
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_key

# Model configuration
OPENROUTER_API_KEY=your_key
OPENROUTER_MODEL=meta-llama/llama-3-8b-instruct
```

Create `backend/Dockerfile` (optimized for HF):
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy requirements first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY agent/ ./agent/
COPY api/ ./api/
COPY retrieval/ ./retrieval/
COPY vector_store/ ./vector_store/
COPY simple_server.py .

# Expose port
EXPOSE 8000

# Start server
CMD ["python", "simple_server.py"]
```

---

### **STEP 2: Deploy Backend to Inference Endpoints**

#### Option A: Using Docker (Recommended)

```bash
# 1. Login to Hugging Face
pip install huggingface_hub
huggingface-cli login

# 2. Create model repository
huggingface-cli repo create humanoid-robotics-backend --type model

# 3. Upload backend code
cd backend
huggingface-cli upload \
  YOUR_USERNAME/humanoid-robotics-backend \
  . \
  .

# 4. Create Inference Endpoint
# Go to: https://huggingface.co/inference-endpoints
# Click "Create Endpoint"
# Select your repository
# Choose: AWS, us-east-1, CPU x2
```

#### Option B: Direct Deployment Script

Create `deploy-backend.sh`:
```bash
#!/bin/bash

echo "🚀 Deploying Backend to Hugging Face..."

# Configuration
REPO_NAME="humanoid-robotics-backend"
ENDPOINT_NAME="humanoid-robotics-api"

# Login
huggingface-cli login

# Create repository
huggingface-cli repo create $REPO_NAME --type model

# Upload files
cd backend
huggingface-cli upload \
  YOUR_USERNAME/$REPO_NAME \
  . \
  .

# Create endpoint
huggingface-cli inference-endpoint create \
  --name $ENDPOINT_NAME \
  --repository YOUR_USERNAME/$REPO_NAME \
  --vendor aws \
  --region us-east-1 \
  --instance-type cpu \
  --instance-size x2

echo "✅ Backend deployed!"
echo "Get URL from: https://huggingface.co/inference-endpoints"
```

Run it:
```bash
chmod +x deploy-backend.sh
./deploy-backend.sh
```

---

### **STEP 3: Deploy Frontend to Spaces**

#### 3.1 Build Frontend

```bash
cd humanoid-robotics-textbook

# Install dependencies
npm install

# Build for production
npm run build

# Check build output
ls -la build/
```

#### 3.2 Create Hugging Face Space

1. Go to: https://huggingface.co/new-space
2. Fill in:
   - **Space name:** `humanoid-robotics-textbook`
   - **Space SDK:** `Static`
   - **License:** `MIT`
   - **Visibility:** `Public`
3. Click "Create Space"

#### 3.3 Upload Build Files

```bash
# Clone your space
cd /tmp
git clone https://huggingface.co/spaces/YOUR_USERNAME/humanoid-robotics-textbook
cd humanoid-robotics-textbook

# Copy build files
cp -r /path/to/project/humanoid-robotics-textbook/build/* .

# Add README if needed
cat > README.md << 'EOF'
# Humanoid Robotics Textbook

Interactive textbook for learning humanoid robotics.

## Features
- 6 comprehensive modules
- AI chatbot assistant
- Interactive examples

## Tech Stack
- Docusaurus
- React
- FastAPI Backend
EOF

# Commit and push
git add .
git commit -m "Deploy humanoid robotics textbook"
git push
```

---

### **STEP 4: Connect Frontend to Backend**

Update `humanoid-robotics-textbook/static/js/chat-widget.js`:

```javascript
// Find this line (around line 10):
const API_BASE_URL = 'http://localhost:8000';

// Replace with your HF endpoint:
const API_BASE_URL = 'https://YOUR_ENDPOINT.hf.space';

// Or use environment variable:
const API_BASE_URL = window.HF_BACKEND_URL || 'https://YOUR_ENDPOINT.hf.space';
```

Update `humanoid-robotics-textbook/docusaurus.config.ts`:

```typescript
// Add this to scripts section:
scripts: [
  {
    src: '/js/chat-widget.js',
    async: true,
    defer: true,
  },
  // Add backend URL as global variable
  {
    src: 'https://YOUR_ENDPOINT.hf.space/static/js/config.js',
    async: true,
  },
],
```

---

## 📋 Complete File Checklist

### ✅ Backend Files (Upload to Model Repo)

```
backend/
├── agent/
│   ├── __init__.py
│   └── rag_agent.py              # ✅ Required
├── api/
│   ├── __init__.py
│   ├── chat.py                   # ✅ Required
│   └── models.py                 # ✅ Required
├── retrieval/
│   ├── __init__.py
│   └── retriever.py              # ✅ Required
├── vector_store/
│   ├── __init__.py
│   └── qdrant_store.py           # ✅ Required (or use HF embeddings)
├── simple_server.py              # ✅ Required (main entry)
├── requirements.txt              # ✅ Required
├── Dockerfile                    # ✅ Required
├── .env.example                  # ℹ️ Template only
└── README.md                     # ℹ️ Documentation
```

### ✅ Frontend Files (Upload to Space)

```
humanoid-robotics-textbook/
├── build/                        # ✅ Upload contents only
│   ├── index.html
│   ├── static/
│   │   ├── css/
│   │   ├── js/
│   │   └── media/
│   └── assets/
├── README.md                     # ✅ Required for Space
└── .gitignore                    # ✅ Recommended
```

**DO NOT upload:**
- ❌ `node_modules/`
- ❌ `.docusaurus/`
- ❌ `src/` (already built)
- ❌ `.env` files

---

## 🔧 Environment Variables

### Backend (Set in Inference Endpoints Dashboard)

Go to: https://huggingface.co/inference-endpoints → Your Endpoint → Settings

Add these environment variables:

| Variable | Value | Required? |
|----------|-------|-----------|
| `HF_TOKEN` | `hf_xxx` | ✅ Yes |
| `QDRANT_URL` | `https://xxx.hf.space` | ⚠️ Optional |
| `QDRANT_API_KEY` | `your_key` | ⚠️ Optional |
| `OPENROUTER_API_KEY` | `your_key` | ⚠️ For OpenRouter |

### Frontend (Hardcode in JS or use HF Secrets)

Edit `chat-widget.js`:
```javascript
const API_BASE_URL = 'https://YOUR_USERNAME-humanoid-robotics-backend.hf.space';
```

---

## 🧪 Testing After Deploy

### Test Backend

```bash
# Get your endpoint URL from:
# https://huggingface.co/inference-endpoints

ENDPOINT_URL="https://YOUR_ENDPOINT.hf.space"

# Test health
curl $ENDPOINT_URL/health

# Test chat
curl -X POST $ENDPOINT_URL/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "hi"}'
```

### Test Frontend

```bash
# Your space URL will be:
# https://YOUR_USERNAME-humanoid-robotics-textbook.hf.space

# Open in browser and:
# 1. Check homepage loads
# 2. Click chat widget
# 3. Ask a question
# 4. Verify backend responds
```

---

## 💰 Cost Breakdown

| Service | Tier | Monthly Cost |
|---------|------|--------------|
| **Frontend Space** | Static | FREE |
| **Backend Endpoint** | CPU x2 | ~$43/month |
| **Backend Endpoint** | GPU | ~$280/month |
| **Qdrant Cloud** (optional) | Standard | ~$25/month |
| **Total (CPU)** | | **~$68/month** |

---

## ⚡ Quick Deploy Commands

```bash
# 1. Prepare backend
cd backend
cp .env.example .env
# Edit .env with your tokens

# 2. Deploy backend
huggingface-cli login
huggingface-cli repo create humanoid-robotics-backend --type model
huggingface-cli upload YOUR_USERNAME/humanoid-robotics-backend . .

# 3. Build frontend
cd ../humanoid-robotics-textbook
npm install
npm run build

# 4. Deploy frontend
cd /tmp
git clone https://huggingface.co/spaces/YOUR_USERNAME/humanoid-robotics-textbook
cd humanoid-robotics-textbook
cp -r /path/to/build/* .
git add .
git commit -m "Deploy"
git push

# 5. Update API URL in frontend
# Edit: static/js/chat-widget.js
# Set: API_BASE_URL = 'https://YOUR_ENDPOINT.hf.space'
```

---

## 🆘 Troubleshooting

### Backend Issues

**Problem:** "Module not found"
```bash
# Ensure all dependencies in requirements.txt
# Rebuild Docker image
```

**Problem:** "HF_TOKEN not set"
```bash
# Set in Inference Endpoints dashboard:
# Settings → Environment Variables → Add HF_TOKEN
```

### Frontend Issues

**Problem:** "Failed to connect to backend"
```javascript
// Check chat-widget.js has correct URL:
const API_BASE_URL = 'https://YOUR_ENDPOINT.hf.space';

// Test in browser console:
fetch('https://YOUR_ENDPOINT.hf.space/health')
  .then(r => r.json())
  .then(console.log)
```

**Problem:** "Build failed"
```bash
# Clear and rebuild
rm -rf node_modules build
npm install
npm run build
```

---

## 📚 Resources

- **HF Inference Endpoints:** https://huggingface.co/docs/inference-endpoints
- **HF Spaces:** https://huggingface.co/docs/hub/spaces
- **Pricing:** https://huggingface.co/pricing
- **Your existing guide:** `BACKEND_HF_MCP_DEPLOY.md`

---

## ✅ Final Checklist

### Before Deploy
- [ ] HF account created
- [ ] HF token generated (write access)
- [ ] Backend tested locally
- [ ] Frontend builds successfully
- [ ] All API keys ready

### Backend Deploy
- [ ] Repository created
- [ ] All files uploaded
- [ ] Dockerfile working
- [ ] Inference Endpoint created
- [ ] HF_TOKEN set in dashboard
- [ ] Health check passing

### Frontend Deploy
- [ ] Space created (Static)
- [ ] Build files uploaded
- [ ] API URL updated
- [ ] Site loads correctly
- [ ] Chat connects to backend

### Security
- [ ] No .env files committed
- [ ] API keys in HF dashboard only
- [ ] CORS configured
- [ ] HTTPS enabled (automatic)

---

**Ready to deploy! 🚀**
