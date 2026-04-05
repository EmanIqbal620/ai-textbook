# 🚀 Quick Deploy to Hugging Face

## Step 1: Create Hugging Face Account

1. Go to https://huggingface.co
2. Click "Sign Up"
3. Complete registration

## Step 2: Create Access Token

1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Name: `deploy-token`
4. Type: **Write**
5. Copy the token (starts with `hf_`)

## Step 3: Create GitHub Repository

```bash
cd /mnt/d/Humanoid-Robotics-AI-textbook

# Initialize git
git init
git add .
git commit -m "Initial commit: Humanoid Robotics Textbook"

# Create repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/humanoid-robotics-textbook.git
git push -u origin main
```

## Step 4: Create Hugging Face Space

1. Go to https://huggingface.co/new-space
2. **Space ID**: `YOUR_USERNAME/humanoid-robotics-textbook`
3. **License**: MIT
4. **Visibility**: Public
5. Click "Create Space"

## Step 5: Deploy

### Option A: Using Deploy Script (Recommended)

```bash
# Set your token
export HF_TOKEN="hf_your_token_here"
export HF_SPACE_ID="YOUR_USERNAME/humanoid-robotics-textbook"

# Run deploy
./deploy-hf.sh
```

### Option B: Manual Deploy

```bash
# Install Hugging Face Hub
pip install huggingface_hub

# Login
huggingface-cli login
# Paste your token when prompted

# Build and deploy
cd humanoid-robotics-textbook
npm install
npm run build

# Upload to Hugging Face
huggingface-cli upload \
    --repo-type space \
    YOUR_USERNAME/humanoid-robotics-textbook \
    ./build \
    .
```

## Step 6: Configure GitHub Actions (Optional)

1. Go to your GitHub repo
2. Settings → Secrets and variables → Actions
3. Add repository secrets:
   - `HF_DEPLOY_TOKEN`: Your Hugging Face token
   - `HF_REPO_ID`: `YOUR_USERNAME/humanoid-robotics-textbook`

## Step 7: Connect Backend (AI Chatbot)

### Deploy Backend to Hugging Face Inference Endpoints

1. Go to https://huggingface.co/inference-endpoints
2. Click "Create endpoint"
3. Configure:
   - **Model**: `mistralai/Mistral-7B-Instruct-v0.2`
   - **Instance**: CPU (or GPU for better performance)
   - **Framework**: PyTorch

### Update Frontend Configuration

Edit `humanoid-robotics-textbook/static/js/chat-widget.js`:

```javascript
const API_BASE_URL = 'https://YOUR_ENDPOINT.hf.space/api';
```

## Step 8: Test Deployment

1. **Frontend**: Visit `https://huggingface.co/spaces/YOUR_USERNAME/humanoid-robotics-textbook`
2. **Chatbot**: Click the chat widget and test
3. **Navigation**: Test all module links

## Troubleshooting

### Build Fails

```bash
# Clear and reinstall
rm -rf node_modules package-lock.json
npm install
npm run build
```

### Deployment Fails

```bash
# Check token
huggingface-cli whoami

# Re-login if needed
huggingface-cli login
```

### Chatbot Not Working

1. Check backend is deployed
2. Update API URL in `chat-widget.js`
3. Check CORS settings

## Cost Estimate

- **Frontend (Static Space)**: FREE
- **Backend (Inference Endpoint)**: ~$0.06/hour (CPU) or ~$0.39/hour (GPU)

## Next Steps

1. ✅ Share your deployed site
2. ✅ Connect custom domain (Settings → Custom Domain)
3. ✅ Add analytics
4. ✅ Enable GitHub auto-deploy

## Support

- Hugging Face Docs: https://huggingface.co/docs
- GitHub Issues: https://github.com/humanoid-robotics/humanoid-robotics-textbook/issues
