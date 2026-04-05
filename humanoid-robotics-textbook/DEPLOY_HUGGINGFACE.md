# Hugging Face Space Deployment Guide

## 🚀 Deploy to Hugging Face Spaces

This textbook can be deployed as a static site on Hugging Face Spaces.

## Prerequisites

1. **Hugging Face Account**: Sign up at https://huggingface.co
2. **GitHub Account**: For version control
3. **Git LFS**: For large file handling

## Quick Deploy

### Option 1: Deploy via Hugging Face UI

1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Choose:
   - **Space SDK**: Static
   - **License**: MIT
   - **Visibility**: Public
4. Upload your files or connect GitHub repository

### Option 2: Deploy via Git

```bash
# Clone your Hugging Face Space
git clone https://huggingface.co/spaces/YOUR_USERNAME/humanoid-robotics-textbook
cd humanoid-robotics-textbook

# Copy textbook files
cp -r /mnt/d/Humanoid-Robotics-AI-textbook/humanoid-robotics-textbook/* .

# Commit and push
git add .
git commit -m "Initial deployment"
git push
```

## File Structure for Hugging Face

```
humanoid-robotics-textbook/
├── README.md              # Space description
├── index.html            # Main entry (built site)
├── static/               # Static assets
│   ├── js/
│   │   └── chat-widget.js
│   └── img/
├── docs/                 # Documentation
└── src/                  # Source files
```

## Build Before Deploy

```bash
cd /mnt/d/Humanoid-Robotics-AI-textbook/humanoid-robotics-textbook

# Install dependencies
npm install

# Build static site
npm run build

# Deploy the 'build' folder to Hugging Face
```

## Environment Variables

Create `.env` file in backend:

```env
# Backend API Configuration
API_URL=https://YOUR_SPACE_ID.hf.space/api
QDRANT_URL=https://your-qdrant-cluster.hf.space
HF_TOKEN=hf_your_huggingface_token

# Chatbot Configuration
MAX_TOKENS=500
TEMPERATURE=0.7
```

## Backend Deployment (Optional)

For the AI chatbot backend, deploy separately:

### Hugging Face Inference Endpoints

1. Go to https://huggingface.co/inference-endpoints
2. Create new endpoint
3. Configure:
   - **Repository**: Your backend code
   - **Instance**: CPU or GPU
   - **Framework**: PyTorch/FastAPI

### Update Frontend to Use HF Endpoint

Edit `static/js/chat-widget.js`:

```javascript
const API_BASE_URL = 'https://YOUR_ENDPOINT.hf.space/api';
```

## GitHub Integration

### Connect GitHub to Hugging Face

1. In your Space, go to "Settings"
2. Click "Connect GitHub repository"
3. Select your repository
4. Enable auto-deployment

### GitHub Actions for Auto-Deploy

Create `.github/workflows/deploy-hf.yml`:

```yaml
name: Deploy to Hugging Face

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '20'
      
      - name: Install dependencies
        run: npm install
        working-directory: ./humanoid-robotics-textbook
      
      - name: Build site
        run: npm run build
        working-directory: ./humanoid-robotics-textbook
      
      - name: Deploy to Hugging Face
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.HF_DEPLOY_TOKEN }}
          publish_dir: ./humanoid-robotics-textbook/build
```

## MCP (Model Context Protocol) Integration

For advanced AI features, integrate Hugging Face MCP:

### Install MCP

```bash
pip install mcp-huggingface
```

### Configure MCP

Create `mcp-config.json`:

```json
{
  "huggingface": {
    "token": "hf_your_token",
    "models": {
      "chat": "mistralai/Mistral-7B-Instruct-v0.2",
      "embeddings": "sentence-transformers/all-MiniLM-L6-v2"
    },
    "inference_endpoint": "https://YOUR_ENDPOINT.hf.space"
  }
}
```

## Testing Before Deploy

```bash
# Test locally
npm run start

# Test build
npm run build

# Test chatbot
curl http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello"}'
```

## Troubleshooting

### Chat Widget Not Working

1. Check API URL in `chat-widget.js`
2. Ensure CORS is enabled on backend
3. Verify Hugging Face token

### Build Fails

```bash
# Clear cache
rm -rf node_modules package-lock.json
npm install

# Rebuild
npm run build
```

## Support

- Hugging Face Docs: https://huggingface.co/docs/hub/spaces
- GitHub Issues: https://github.com/humanoid-robotics/humanoid-robotics-textbook/issues
