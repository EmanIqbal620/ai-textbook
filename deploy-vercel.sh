#!/bin/bash
# Quick Vercel Deployment Script
# Run this from the project root

echo "🚀 Starting Vercel Full-Stack Deployment..."
echo ""

# Step 1: Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "❌ Vercel CLI not found"
    echo "📦 Installing Vercel CLI..."
    npm i -g vercel
    echo "✅ Vercel CLI installed"
fi

# Step 2: Copy Vercel chat config
echo "📝 Setting up chat widget configuration..."
cp VERCEL_chat-config.js humanoid-robotics-textbook/static/js/chat-config.js
echo "✅ Chat config updated"

# Step 3: Test frontend build
echo ""
echo "🔨 Testing frontend build..."
cd humanoid-robotics-textbook
npm install
npm run build
cd ..
echo "✅ Frontend build successful"

# Step 4: Deploy to Vercel
echo ""
echo "🚀 Deploying to Vercel..."
echo ""
vercel --prod

echo ""
echo "✅ Deployment complete!"
echo ""
echo "⚠️  IMPORTANT: Don't forget to set environment variables in Vercel Dashboard:"
echo "   - OPENROUTER_API_KEY"
echo "   - COHERE_API_KEY"
echo "   - QDRANT_URL"
echo "   - QDRANT_API_KEY"
echo "   - QDRANT_COLLECTION_NAME"
echo ""
echo "📖 See VERCEL_DEPLOYMENT.md for complete instructions"
