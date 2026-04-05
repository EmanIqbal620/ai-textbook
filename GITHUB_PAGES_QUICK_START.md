# 📦 GitHub Pages Deployment - Quick Checklist

## ✅ Files You Need

### From Project Root:

```
📁 humanoid-robotics-textbook/     # ✅ Take EVERYTHING
├── docs/                          # ✅ All textbook content
├── src/
│   ├── css/
│   │   └── custom.final.css       # ✅ Your styles
│   └── pages/
│       └── index.js               # ✅ Homepage
├── static/
│   └── js/
│       ├── chat-widget.js         # ✅ Chat widget
│       └── chat-config.js         # ✅ Config (I created this)
├── docusaurus.config.ts           # ✅ Configuration
├── sidebars.ts                    # ✅ Navigation
├── package.json                   # ✅ Dependencies
└── build/                         # ✅ AFTER npm run build
```

---

## 🚀 Deploy Steps

### 1️⃣ Setup (One Time)

```bash
cd D:\Humanoid-Robotics-AI-textbook\humanoid-robotics-textbook

# Install gh-pages
npm install --save-dev gh-pages

# Or run the setup script
cd ..
chmod +x setup-github-pages.sh
./setup-github-pages.sh
```

### 2️⃣ Configure URLs

**Edit `docusaurus.config.ts`** (lines 11-13):

```typescript
// Change this:
url: 'https://humanoid-robotics-textbook.vercel.app',
baseUrl: '/',

// To this (replace YOUR_USERNAME):
url: 'https://YOUR_USERNAME.github.io',
baseUrl: '/humanoid-robotics-textbook/',
```

**Edit `static/js/chat-config.js`**:

```javascript
// Change this:
const API_BASE_URL = 'http://localhost:8000';

// To your Hugging Face backend:
const API_BASE_URL = 'https://YOUR_USERNAME-humanoid-robotics-backend.hf.space';
```

### 3️⃣ Build

```bash
cd humanoid-robotics-textbook
npm run build
```

### 4️⃣ Deploy to GitHub Pages

```bash
# Deploy using gh-pages
npm run deploy
```

This will:
- Create/update `gh-pages` branch
- Upload build files
- Make site live in 1-2 minutes

### 5️⃣ Enable GitHub Pages

1. Go to: https://github.com/YOUR_USERNAME/YOUR_USERNAME.github.io/settings/pages
2. **Source:** Deploy from a branch
3. **Branch:** `gh-pages` (or `main` if using user org site)
4. **Folder:** `/ (root)`
5. Click **Save**

---

## 🌐 Your URLs

### Frontend (GitHub Pages - FREE):
```
https://YOUR_USERNAME.github.io/humanoid-robotics-textbook/
```

### Backend (Hugging Face - ~$43/month):
```
https://YOUR_USERNAME-humanoid-robotics-backend.hf.space
```

---

## 🧪 Test After Deploy

### 1. Check Frontend

Open in browser:
```
https://YOUR_USERNAME.github.io/humanoid-robotics-textbook/
```

✅ Should see homepage with module cards

### 2. Test Chat Widget

1. Scroll to bottom-right corner
2. Click robot icon
3. Type: "hi"
4. ✅ Should respond instantly

### 3. Check Console

Press `F12`, check for errors:
```javascript
// Should see:
Backend URL: https://YOUR_USERNAME-humanoid-robotics-backend.hf.space
```

---

## ⚠️ Common Issues

### Issue: Page shows 404

**Solution:**
```bash
# Check docusaurus.config.ts has correct baseUrl:
baseUrl: '/humanoid-robotics-textbook/'

# Rebuild and redeploy:
npm run build
npm run deploy
```

### Issue: Chat doesn't work

**Solution:**
```javascript
// Check chat-config.js has backend URL:
const API_BASE_URL = 'https://YOUR_USERNAME-humanoid-robotics-backend.hf.space';

// NOT localhost:
const API_BASE_URL = 'http://localhost:8000'; // ❌ Wrong!

// Rebuild and redeploy:
npm run build
npm run deploy
```

### Issue: Styles not loading

**Solution:**
```bash
# Ensure clean build:
rm -rf build
npm run build
npm run deploy
```

---

## 🔄 Update After Changes

Whenever you edit content:

```bash
# 1. Make your changes
# Edit docs/, src/, etc.

# 2. Build
npm run build

# 3. Deploy
npm run deploy

# 4. Wait 1-2 minutes for GitHub Pages to update
```

---

## 💰 Cost

| Service | Cost |
|---------|------|
| **GitHub Pages** | FREE |
| **Hugging Face Backend** | ~$43/month |
| **Total** | ~$43/month |

---

## 📋 Complete Checklist

### Before Deploy
- [ ] GitHub account created
- [ ] Node.js 18+ installed
- [ ] Backend deployed to Hugging Face
- [ ] Backend URL ready

### Configuration
- [ ] Updated `docusaurus.config.ts` with GitHub URL
- [ ] Updated `chat-config.js` with backend URL
- [ ] Installed `gh-pages` package

### Build & Deploy
- [ ] Built successfully: `npm run build`
- [ ] Deployed: `npm run deploy`
- [ ] GitHub Pages enabled in settings

### Testing
- [ ] Frontend loads correctly
- [ ] Chat widget appears
- [ ] Chat connects to backend
- [ ] No console errors
- [ ] All links work

---

## 🎯 Quick Commands

```bash
# Setup (one time)
cd humanoid-robotics-textbook
npm install --save-dev gh-pages

# Configure
# Edit docusaurus.config.ts and chat-config.js

# Build
npm run build

# Deploy
npm run deploy

# Test
# Open: https://YOUR_USERNAME.github.io/humanoid-robotics-textbook/
```

---

## 📚 Resources

- **Main Guide:** `GITHUB_PAGES_DEPLOY.md`
- **Setup Script:** `setup-github-pages.sh`
- **Backend Guide:** `HF_DEPLOYMENT_GUIDE.md`

---

**Ready to deploy to GitHub Pages! 🚀**
