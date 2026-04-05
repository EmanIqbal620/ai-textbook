# 🚀 Deploy Frontend to GitHub Pages

## Quick Summary

**Frontend:** GitHub Pages (FREE)  
**Backend:** Hugging Face Inference Endpoints (~$43/month)

---

## 📋 Prerequisites

1. **GitHub Account**
2. **Node.js 18+** installed
3. **Git** installed

---

## ⚙️ Step 1: Update Configuration

### Update `docusaurus.config.ts`

Edit these lines (around line 11-13):

```typescript
// Change FROM:
url: 'https://humanoid-robotics-textbook.vercel.app',
baseUrl: '/',

// Change TO (replace YOUR_USERNAME with your GitHub username):
url: 'https://YOUR_USERNAME.github.io',
baseUrl: '/humanoid-robotics-textbook/',
```

### Update `package.json`

Add these fields:

```json
{
  "name": "humanoid-robotics-textbook",
  "version": "1.0.0",
  "homepage": "https://YOUR_USERNAME.github.io/humanoid-robotics-textbook",
  "scripts": {
    "start": "docusaurus start",
    "build": "docusaurus build",
    "deploy": "gh-pages -d build"
  },
  "devDependencies": {
    "gh-pages": "^6.1.0"
  }
}
```

---

## 🔧 Step 2: Install Dependencies

```bash
cd D:\Humanoid-Robotics-AI-textbook\humanoid-robotics-textbook

# Install gh-pages package
npm install --save-dev gh-pages

# Or use the deploy script I created
cd ..
./setup-github-pages.sh
```

---

## 📦 Step 3: Build for GitHub Pages

```bash
cd humanoid-robotics-textbook

# Build the site
npm run build

# Check build output
ls -la build/
```

---

## 🚀 Step 4: Deploy to GitHub Pages

### Option A: Using gh-pages package (Recommended)

```bash
# From humanoid-robotics-textbook folder
npm run deploy
```

### Option B: Manual deployment

```bash
# Clone your GitHub repository
cd /tmp
git clone https://github.com/YOUR_USERNAME/YOUR_USERNAME.github.io.git
cd YOUR_USERNAME.github.io

# Copy build files
cp -r D:/Humanoid-Robotics-AI-textbook/humanoid-robotics-textbook/build/* .

# Commit and push
git add .
git commit -m "Deploy humanoid robotics textbook"
git push
```

### Option C: Deploy to repository gh-pages branch

```bash
cd humanoid-robotics-textbook

# Initialize git if not already
git init
git add .
git commit -m "Initial commit"

# Deploy using gh-pages
npm run deploy
```

---

## 🔗 Step 5: Update Chat Widget API URL

**IMPORTANT:** Update the backend API URL in the chat widget!

Edit `static/js/chat-widget.js` (around line 10):

```javascript
// Change FROM:
const API_BASE_URL = 'http://localhost:8000';

// Change TO your Hugging Face backend:
const API_BASE_URL = 'https://YOUR_USERNAME-humanoid-robotics-backend.hf.space';
```

Then rebuild and redeploy:

```bash
npm run build
npm run deploy
```

---

## 🌐 Your URLs Will Be

### Frontend (GitHub Pages):
```
https://YOUR_USERNAME.github.io/humanoid-robotics-textbook/
```

### Backend (Hugging Face):
```
https://YOUR_USERNAME-humanoid-robotics-backend.hf.space
```

---

## ✅ Complete Deployment Script

I created `deploy-to-github.sh` for you:

```bash
#!/bin/bash

# Make executable
chmod +x deploy-to-github.sh

# Run it
./deploy-to-github.sh
```

---

## 🧪 Test After Deploy

### 1. Check Frontend Loads

Open in browser:
```
https://YOUR_USERNAME.github.io/humanoid-robotics-textbook/
```

### 2. Test Chat Widget

1. Scroll to bottom-right
2. Click chat icon
3. Ask: "hi"
4. Should get instant response!

### 3. Check Browser Console

Press `F12` and check for errors. You should see:
```
Backend URL: https://YOUR_USERNAME-humanoid-robotics-backend.hf.space
```

---

## ⚠️ Common Issues

### Issue: 404 Error

**Problem:** Page not found

**Solution:**
```bash
# Ensure baseUrl is correct in docusaurus.config.ts
baseUrl: '/humanoid-robotics-textbook/'

# Rebuild
npm run build

# Redeploy
npm run deploy
```

### Issue: Chat Not Working

**Problem:** Backend URL incorrect

**Solution:**
```javascript
// Check chat-widget.js has correct URL:
const API_BASE_URL = 'https://YOUR_USERNAME-humanoid-robotics-backend.hf.space';

// NOT this:
const API_BASE_URL = 'http://localhost:8000';
```

### Issue: Assets Not Loading

**Problem:** Incorrect base URL

**Solution:**
```typescript
// In docusaurus.config.ts, ensure:
url: 'https://YOUR_USERNAME.github.io',
baseUrl: '/humanoid-robotics-textbook/',
```

---

## 📊 Deployment Architecture

```
┌─────────────────────────────────────────────────────────┐
│  GitHub Pages (FREE)                                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────┐                                   │
│  │  Frontend       │                                   │
│  │  (Static Site)  │                                   │
│  │                 │                                   │
│  │  - Textbook     │                                   │
│  │  - Module Cards │                                   │
│  │  - Chat Widget  │                                   │
│  └────────┬────────┘                                   │
│           │                                            │
│           │ POST /api/v1/chat                          │
│           ▼                                            │
└─────────────────────────────────────────────────────────┘
           │
           │ HTTPS
           ▼
┌─────────────────────────────────────────────────────────┐
│  Hugging Face Inference Endpoints (~$43/month)         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────┐                                   │
│  │  Backend API    │                                   │
│  │  (FastAPI)      │                                   │
│  │                 │                                   │
│  │  - RAG Agent    │                                   │
│  │  - Caching      │                                   │
│  │  - LLM          │                                   │
│  └─────────────────┘                                   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 💰 Cost Breakdown

| Service | Cost |
|---------|------|
| **GitHub Pages** | FREE |
| **Hugging Face Backend (CPU)** | ~$43/month |
| **Total** | ~$43/month |

---

## 🔄 Update Deployment

Whenever you make changes:

```bash
cd humanoid-robotics-textbook

# Make your changes...
# Edit files...

# Build and deploy
npm run build
npm run deploy
```

Changes take 1-2 minutes to appear on GitHub Pages.

---

## 📚 GitHub Pages Settings

### Enable GitHub Pages

1. Go to: https://github.com/YOUR_USERNAME/YOUR_USERNAME.github.io/settings/pages
2. Source: Deploy from a branch
3. Branch: `main` (or `master`)
4. Folder: `/ (root)`
5. Save

### Custom Domain (Optional)

If you want a custom domain:

1. Go to: Settings → Pages → Custom domain
2. Enter your domain: `textbook.yourdomain.com`
3. Update DNS records
4. Update `docusaurus.config.ts`:
   ```typescript
   url: 'https://textbook.yourdomain.com',
   baseUrl: '/',
   ```

---

## ✅ Deployment Checklist

### Before Deploy
- [ ] Updated `docusaurus.config.ts` with GitHub URL
- [ ] Updated `package.json` with gh-pages script
- [ ] Installed `gh-pages` package
- [ ] Updated `chat-widget.js` with backend URL
- [ ] Backend deployed to Hugging Face

### Deploy
- [ ] Built successfully: `npm run build`
- [ ] Deployed to gh-pages: `npm run deploy`
- [ ] GitHub Pages enabled in repository settings

### After Deploy
- [ ] Frontend loads: `https://YOUR_USERNAME.github.io/humanoid-robotics-textbook/`
- [ ] Chat widget appears
- [ ] Chat connects to backend
- [ ] Asking questions works
- [ ] No console errors

---

## 🎯 Quick Commands

```bash
# 1. Setup (one time)
cd humanoid-robotics-textbook
npm install --save-dev gh-pages

# 2. Configure
# Edit docusaurus.config.ts and chat-widget.js

# 3. Build
npm run build

# 4. Deploy
npm run deploy

# 5. Test
# Open: https://YOUR_USERNAME.github.io/humanoid-robotics-textbook/
```

---

## 📚 Resources

- **GitHub Pages Docs:** https://pages.github.com/
- **Docusaurus Deployment:** https://docusaurus.io/docs/deployment
- **Your Backend Guide:** `HF_DEPLOYMENT_GUIDE.md`

---

**Ready to deploy to GitHub Pages! 🚀**
