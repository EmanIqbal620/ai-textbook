# 🚀 Vercel Full-Stack Deployment Guide

## Architecture

```
┌─────────────────────────────────────────┐
│           Vercel Deployment             │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │  Frontend (Docusaurus)            │  │
│  │  Static Site (CDN)                │  │
│  │  your-app.vercel.app              │  │
│  └───────────────────────────────────┘  │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │  Backend (Python Serverless)      │  │
│  │  /api/chat                        │  │
│  │  /api/health                      │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

**Both frontend and backend deployed on Vercel!** ✨

---

## 📋 Prerequisites

1. ✅ Vercel account (free tier works, but check limits)
2. ✅ Node.js 18+ installed
3. ✅ Python 3.11+ installed
4. ✅ Environment variables ready (see below)

---

## 🔑 Required Environment Variables

You'll need to set these in Vercel Dashboard during deployment:

| Variable | Purpose | Example |
|----------|---------|---------|
| `OPENROUTER_API_KEY` | LLM API key | `sk-or-v1-...` |
| `COHERE_API_KEY` | Embeddings API | `your-cohere-key` |
| `QDRANT_URL` | Vector database URL | `https://your-cluster.qdrant.io` |
| `QDRANT_API_KEY` | Vector database auth | `your-qdrant-key` |
| `QDRANT_COLLECTION_NAME` | Collection name | `humanoid_ai_book` |
| `NEON_DATABASE_URL` | PostgreSQL (optional) | `postgresql://...` |

---

## 🛠️ Step-by-Step Deployment

### Step 1: Prepare the Chat Widget

```bash
# Copy the Vercel chat config (from project root)
cp VERCEL_chat-config.js humanoid-robotics-textbook/static/js/chat-config.js
```

This configures the frontend to call `/api/chat` on the same domain (Vercel).

### Step 2: Test Locally (Optional but Recommended)

#### Test Frontend Build:
```bash
cd humanoid-robotics-textbook
npm install
npm run build
```

#### Test Backend Locally:
```bash
cd backend
pip install -r requirements.txt
uvicorn server:app --reload
```

Visit `http://localhost:8000/health` to verify backend works.

### Step 3: Deploy to Vercel

#### Option A: Vercel CLI (Recommended)

```bash
# Install Vercel CLI
npm i -g vercel

# Login
vercel login

# Deploy from project root
vercel

# Follow prompts:
# - Set up and deploy? Y
# - Which scope? (select your account)
# - Link to existing project? N
# - Project name: humanoid-robotics-textbook
# - Directory: . (root)
# - Override settings? N
```

#### Option B: Vercel Dashboard

1. Go to [vercel.com](https://vercel.com)
2. Click **"Add New Project"**
3. Import your GitHub repository
4. Configure build settings:

   **Frontend Settings:**
   - **Framework Preset**: Docusaurus 3
   - **Root Directory**: `humanoid-robotics-textbook`
   - **Build Command**: `docusaurus build`
   - **Output Directory**: `build`

   **Backend Settings:**
   - The `api/` folder will be auto-detected as Python serverless functions
   - No additional config needed

5. Add environment variables (see Required Environment Variables above)
6. Click **"Deploy"**

### Step 4: Add Environment Variables

In Vercel Dashboard:
1. Go to your project → **Settings** → **Environment Variables**
2. Add all required variables (see table above)
3. Select environments: ✅ Production ✅ Preview ✅ Development
4. Click **Save**

### Step 5: Redeploy with Environment Variables

```bash
vercel --prod
```

Or trigger redeploy from Vercel Dashboard.

### Step 6: Test Deployment

Visit your Vercel URL and test:

1. ✅ Site loads: `https://your-app.vercel.app`
2. ✅ Health check: `https://your-app.vercel.app/api/health`
3. ✅ Chat widget works (open chat and ask a question)
4. ✅ Check browser console for errors

---

## 📁 Files Created for Vercel

| File | Purpose |
|------|---------|
| `vercel.json` | Main Vercel configuration (builds + routes) |
| `api/chat.py` | Serverless function for chat endpoint |
| `api/health.py` | Serverless function for health check |
| `api/requirements.txt` | Python dependencies for serverless functions |
| `api/.vcspython` | Python runtime configuration |
| `.vercelignore` | Files to exclude from Vercel deployment |
| `VERCEL_chat-config.js` | Chat widget config for Vercel |
| `VERCEL_DEPLOYMENT.md` | This guide |

**Your original files are NOT modified:**
- ✅ `backend/server.py` - unchanged
- ✅ `backend/requirements.txt` - unchanged
- ✅ All existing deployment files - unchanged

---

## ⚠️ Important Vercel Limits

### Free Tier Limits:
- **Serverless Function Size**: 50 MB (compressed)
- **Function Timeout**: 10 seconds (free) / 30 seconds (pro)
- **Memory**: 1024 MB per function
- **Bandwidth**: 100 GB/month

### Potential Issues:

**1. Function Timeout**
- RAG queries might take >10 seconds
- **Solution**: Upgrade to Vercel Pro ($20/month) for 30s timeout

**2. Cold Starts**
- Serverless functions have cold start latency
- **Solution**: Enable Vercel Analytics to monitor performance

**3. Package Size**
- Python dependencies must fit in 50MB
- **Solution**: The `requirements.txt` is optimized for Vercel

---

## 🔧 Troubleshooting

### Build Fails

**Problem**: `npm run build` fails

**Solution**:
```bash
cd humanoid-robotics-textbook
npm install
npm run build  # Test locally first
```

### API Returns 500 Error

**Problem**: Chat endpoint fails

**Solution**:
1. Check environment variables are set in Vercel Dashboard
2. Check Vercel function logs: Dashboard → Functions → Logs
3. Verify API keys are valid

### Function Timeout

**Problem**: Request takes too long

**Solution**:
1. Check Vercel logs for timeout errors
2. Upgrade to Vercel Pro for 30s timeout
3. Optimize RAG agent response time

### CORS Errors

**Problem**: Browser blocks API requests

**Solution**: The `vercel.json` routes should handle this. If issues persist, add CORS headers to `api/chat.py`.

---

## 📊 Monitoring

### Vercel Dashboard:
- **Analytics**: Traffic and performance
- **Functions**: Serverless function logs
- **Deployments**: Build and deployment history

### API Endpoints:
- Health: `https://your-app.vercel.app/api/health`
- Chat: `https://your-app.vercel.app/api/chat` (POST)

---

## 🔄 Redeploying

### Automatic (GitHub connected):
Push to main branch → Vercel auto-deploys

### Manual:
```bash
vercel --prod
```

### With Environment Variable Changes:
1. Update variables in Vercel Dashboard
2. Trigger redeploy: `vercel --prod`

---

## 💰 Cost Estimate

**Free Tier** (if within limits):
- ✅ Frontend: Free (static site)
- ⚠️ Backend: May hit 10s timeout limit

**Pro Tier** ($20/month):
- ✅ Frontend: Included
- ✅ Backend: 30s timeout, better performance

---

## 🎯 Custom Domain

1. Vercel Dashboard → Your Project → **Settings** → **Domains**
2. Add your domain
3. Update DNS records as instructed

---

## 📞 Support

- Vercel Python Functions: https://vercel.com/docs/functions/serverless-functions/runtimes/python
- Vercel Configuration: https://vercel.com/docs/project-configuration
- Docusaurus Deployment: https://docusaurus.io/docs/deployment

---

## 🚀 Quick Deploy Commands

```bash
# Login
vercel login

# Preview deployment
vercel

# Production deployment
vercel --prod

# Check deployment status
vercel ls
```
