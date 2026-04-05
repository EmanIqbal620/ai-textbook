# 📦 Hugging Face Deployment - File Checklist

## 🎯 What You Need to Deploy

You're deploying **2 separate services**:

### 1️⃣ Frontend (Textbook + Chat UI)
**Platform:** Hugging Face Spaces (Static)  
**URL:** `https://YOUR_USERNAME-humanoid-robotics-textbook.hf.space`

### 2️⃣ Backend (Chatbot API)  
**Platform:** Hugging Face Inference Endpoints  
**URL:** `https://YOUR_USERNAME-humanoid-robotics-backend.hf.space`

---

## 📁 Files to Take

### ✅ For BACKEND Deployment

**Location:** `D:\Humanoid-Robotics-AI-textbook\backend\`

```
📁 backend/
│
├── 📁 agent/                    # ✅ REQUIRED - RAG agent logic
│   ├── __init__.py
│   └── rag_agent.py             # Main chatbot with caching
│
├── 📁 api/                      # ✅ REQUIRED - API endpoints
│   ├── __init__.py
│   ├── chat.py                  # Chat endpoint
│   └── models.py                # Data models
│
├── 📁 retrieval/                # ✅ REQUIRED - Vector search
│   ├── __init__.py
│   └── retriever.py             # Search function
│
├── 📁 vector_store/             # ✅ REQUIRED - Qdrant connection
│   ├── __init__.py
│   └── qdrant_store.py
│
├── 📄 simple_server.py          # ✅ REQUIRED - Main server file
├── 📄 requirements-hf.txt       # ✅ REQUIRED - Python dependencies
├── 📄 Dockerfile-hf             # ✅ REQUIRED - For container deployment
├── 📄 .env.example              # ✅ REQUIRED - Environment template
└── 📄 README.md                 # ℹ️ Optional - Documentation
```

**❌ DO NOT Upload:**
- `venv/` or `.venv/` (virtual environment)
- `__pycache__/` (Python cache)
- `.env` (contains secrets!)
- `test_*.py` (test files)
- `data/` (too large)

---

### ✅ For FRONTEND Deployment

**Location:** `D:\Humanoid-Robotics-AI-textbook\humanoid-robotics-textbook\`

**First, build the frontend:**
```bash
cd humanoid-robotics-textbook
npm install
npm run build
```

**Then upload the `build/` folder contents:**

```
📁 build/                        # ✅ Upload EVERYTHING inside
│
├── 📄 index.html                # ✅ Main HTML file
│
├── 📁 static/                   # ✅ Static assets
│   ├── 📁 css/
│   │   └── *.css               # All stylesheets
│   ├── 📁 js/
│   │   └── *.js                # All JavaScript (including chat-widget.js)
│   └── 📁 media/
│       └── *.svg, *.png        # Images and icons
│
└── 📁 assets/                   # ✅ Additional assets
    └── *.js, *.css
```

**❌ DO NOT Upload:**
- `node_modules/` (too large)
- `src/` (already compiled)
- `.docusaurus/` (build cache)
- `docs/` (already in build)

---

## 🚀 Quick Deploy Commands

### Step 1: Login to Hugging Face

```bash
pip install huggingface_hub
huggingface-cli login
```

### Step 2: Deploy Backend

```bash
cd D:\Humanoid-Robotics-AI-textbook\backend

# Upload to HF
huggingface-cli upload \
  YOUR_USERNAME/humanoid-robotics-backend \
  . \
  . \
  --include "agent/**/*" \
  --include "api/**/*" \
  --include "retrieval/**/*" \
  --include "vector_store/**/*" \
  --include "simple_server.py" \
  --include "requirements-hf.txt" \
  --include "Dockerfile-hf"
```

### Step 3: Deploy Frontend

```bash
# Build first
cd D:\Humanoid-Robotics-AI-textbook\humanoid-robotics-textbook
npm run build

# Clone your space
cd /tmp
git clone https://huggingface.co/spaces/YOUR_USERNAME/humanoid-robotics-textbook
cd humanoid-robotics-textbook

# Copy build files
cp -r D:/Humanoid-Robotics-AI-textbook/humanoid-robotics-textbook/build/* .

# Push to HF
git add .
git commit -m "Deploy textbook"
git push
```

---

## 🔧 Environment Variables

### Backend (Set in HF Dashboard)

Go to: **Inference Endpoints → Your Endpoint → Settings → Environment Variables**

Add these:

| Variable | Where to Get | Required? |
|----------|--------------|-----------|
| `HF_TOKEN` | https://huggingface.co/settings/tokens | ✅ YES |
| `QDRANT_URL` | Your Qdrant Cloud dashboard | ⚠️ Optional |
| `QDRANT_API_KEY` | Your Qdrant Cloud dashboard | ⚠️ Optional |
| `OPENROUTER_API_KEY` | https://openrouter.ai/keys | ⚠️ For OpenRouter |

### Frontend (Hardcode in JS)

Edit `humanoid-robotics-textbook/static/js/chat-widget.js`:

```javascript
// Line ~10, change this:
const API_BASE_URL = 'http://localhost:8000';

// To this:
const API_BASE_URL = 'https://YOUR_USERNAME-humanoid-robotics-backend.hf.space';
```

---

## ✅ Deployment Checklist

### Before You Start
- [ ] Hugging Face account created
- [ ] Hugging Face token generated (write access)
- [ ] Backend works locally (`python simple_server.py`)
- [ ] Frontend builds successfully (`npm run build`)

### Backend Deployment
- [ ] All required folders uploaded (`agent/`, `api/`, `retrieval/`, `vector_store/`)
- [ ] `simple_server.py` uploaded
- [ ] `requirements-hf.txt` uploaded
- [ ] `Dockerfile-hf` uploaded
- [ ] Inference Endpoint created (CPU x2, us-east-1)
- [ ] `HF_TOKEN` set in dashboard
- [ ] Health check passes: `curl https://YOUR_ENDPOINT.hf.space/health`

### Frontend Deployment
- [ ] Frontend built (`npm run build`)
- [ ] Hugging Face Space created (Static)
- [ ] Build files uploaded (contents of `build/` folder)
- [ ] API URL updated in `chat-widget.js`
- [ ] Site loads: `https://YOUR_USERNAME-humanoid-robotics-textbook.hf.space`
- [ ] Chat widget appears

### Testing
- [ ] Backend health check: `GET /health` returns `{"status": "healthy"}`
- [ ] Backend chat endpoint: `POST /api/v1/chat` returns answer
- [ ] Frontend loads without errors
- [ ] Chat widget connects to backend
- [ ] Asking questions works

---

## 💰 Cost Estimate

| Service | Tier | Monthly Cost |
|---------|------|--------------|
| Frontend Space | Static | **FREE** |
| Backend Endpoint | CPU x2 | **~$43/month** |
| Backend Endpoint | GPU | ~$280/month |
| **Total (CPU)** | | **~$43/month** |

---

## 🆘 Common Issues

### Backend: "Module not found"
```bash
# Ensure requirements-hf.txt has all dependencies
# Rebuild Docker image
```

### Backend: "HF_TOKEN not set"
```bash
# Set in Inference Endpoints dashboard:
# Settings → Environment Variables → Add HF_TOKEN
```

### Frontend: "Failed to connect"
```javascript
// Check chat-widget.js has correct URL:
const API_BASE_URL = 'https://YOUR_USERNAME-humanoid-robotics-backend.hf.space';

// Test in browser console:
fetch('https://YOUR_ENDPOINT.hf.space/health')
  .then(r => r.json())
  .then(console.log)
```

### Frontend: Build fails
```bash
# Clear and rebuild
cd humanoid-robotics-textbook
rm -rf node_modules build
npm install
npm run build
```

---

## 📚 Reference Files

Created for you in project root:

- `HF_DEPLOYMENT_GUIDE.md` - Complete deployment guide
- `deploy-to-hf.sh` - Automated deployment script
- `backend/requirements-hf.txt` - HF-specific dependencies
- `backend/Dockerfile-hf` - Optimized Dockerfile

---

## 🎯 Summary

**Minimum files needed:**

### Backend (5 items):
1. `agent/` folder
2. `api/` folder
3. `retrieval/` folder
4. `vector_store/` folder
5. `simple_server.py`
6. `requirements-hf.txt`
7. `Dockerfile-hf`

### Frontend (1 item):
1. Contents of `build/` folder

**That's it!** 🚀

---

**Questions? Check:** `HF_DEPLOYMENT_GUIDE.md` for detailed instructions.
