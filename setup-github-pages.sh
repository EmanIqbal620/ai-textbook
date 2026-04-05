#!/bin/bash

# GitHub Pages Setup Script for Humanoid Robotics Textbook
# Usage: ./setup-github-pages.sh

set -e

echo "🚀 Setting up GitHub Pages Deployment"
echo "======================================"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Navigate to textbook directory
cd "$(dirname "$0")"
TEXTBOOK_DIR=$(pwd)/humanoid-robotics-textbook

print_info "Textbook directory: $TEXTBOOK_DIR"

if [ ! -d "$TEXTBOOK_DIR" ]; then
    print_error "Directory not found: $TEXTBOOK_DIR"
    exit 1
fi

cd "$TEXTBOOK_DIR"

# Check if package.json exists
if [ ! -f "package.json" ]; then
    print_error "package.json not found!"
    exit 1
fi

# Install gh-pages
print_info "Installing gh-pages package..."
npm install --save-dev gh-pages
print_success "gh-pages installed!"

# Get GitHub username
print_info ""
echo "Enter your GitHub username:"
read -r GITHUB_USERNAME

if [ -z "$GITHUB_USERNAME" ]; then
    print_error "GitHub username is required!"
    exit 1
fi

print_info "GitHub username: $GITHUB_USERNAME"

# Update docusaurus.config.ts
print_info ""
print_info "Updating docusaurus.config.ts..."

# Backup original file
cp docusaurus.config.ts docusaurus.config.ts.backup

# Update configuration using sed (cross-platform)
if grep -q "humanoid-robotics-textbook.vercel.app" docusaurus.config.ts; then
    # Replace URLs
    sed -i.bak "s|https://humanoid-robotics-textbook.vercel.app|https://${GITHUB_USERNAME}.github.io|g" docusaurus.config.ts
    sed -i.bak "s|baseUrl: '/'|baseUrl: '/humanoid-robotics-textbook/'|g" docusaurus.config.ts
    rm docusaurus.config.ts.bak
    print_success "Configuration updated!"
else
    print_info "Configuration already customized or different format"
    print_info "Please manually update docusaurus.config.ts:"
    echo "  url: 'https://${GITHUB_USERNAME}.github.io',"
    echo "  baseUrl: '/humanoid-robotics-textbook/',"
fi

# Update chat widget
print_info ""
print_info "Updating chat widget configuration..."

if [ -f "static/js/chat-widget.js" ]; then
    # Backup original file
    cp static/js/chat-widget.js static/js/chat-widget.js.backup
    
    # Ask for backend URL
    print_info ""
    echo "Enter your Hugging Face backend URL (or press Enter for localhost):"
    read -r BACKEND_URL
    
    if [ -z "$BACKEND_URL" ]; then
        BACKEND_URL="http://localhost:8000"
    fi
    
    print_info "Backend URL: $BACKEND_URL"
    
    # Update API URL in chat widget
    if grep -q "http://localhost:8000" static/js/chat-widget.js; then
        sed -i.bak "s|http://localhost:8000|${BACKEND_URL}|g" static/js/chat-widget.js
        rm static/js/chat-widget.js.bak
        print_success "Chat widget updated!"
    else
        print_info "Chat widget already customized or different format"
    fi
else
    print_error "chat-widget.js not found!"
fi

# Add deploy script to package.json
print_info ""
print_info "Adding deploy script to package.json..."

if ! grep -q '"deploy":' package.json; then
    # Add deploy script using node
    node -e "
    const fs = require('fs');
    const pkg = JSON.parse(fs.readFileSync('package.json', 'utf8'));
    pkg.scripts.deploy = 'gh-pages -d build';
    pkg.homepage = 'https://${GITHUB_USERNAME}.github.io/humanoid-robotics-textbook';
    fs.writeFileSync('package.json', JSON.stringify(pkg, null, 2));
    "
    print_success "Deploy script added!"
else
    print_info "Deploy script already exists"
fi

# Build the site
print_info ""
print_info "Building site..."
npm run build
print_success "Build complete!"

# Instructions
print_info ""
echo "=========================================================="
print_success "Setup Complete!"
echo "=========================================================="
print_info ""
print_info "Next steps:"
echo ""
echo "1. Review changes:"
echo "   - docusaurus.config.ts"
echo "   - static/js/chat-widget.js"
echo "   - package.json"
echo ""
echo "2. Commit your changes:"
echo "   git add ."
echo "   git commit -m 'Setup for GitHub Pages'"
echo ""
echo "3. Deploy to GitHub Pages:"
echo "   npm run deploy"
echo ""
echo "4. Enable GitHub Pages:"
echo "   Go to: https://github.com/${GITHUB_USERNAME}/${GITHUB_USERNAME}.github.io/settings/pages"
echo "   Source: Deploy from a branch"
echo "   Branch: main"
echo "   Folder: / (root)"
echo ""
echo "5. Your site will be at:"
echo "   https://${GITHUB_USERNAME}.github.io/humanoid-robotics-textbook/"
echo ""
echo "=========================================================="

# Create README for deployment
cat > DEPLOY_README.md << EOF
# Deployment to GitHub Pages

## Quick Deploy

\`\`\`bash
# Build and deploy
npm run deploy
\`\`\`

## Your URLs

**Frontend:** https://${GITHUB_USERNAME}.github.io/humanoid-robotics-textbook/

**Backend:** ${BACKEND_URL}

## Configuration

### docusaurus.config.ts
- url: https://${GITHUB_USERNAME}.github.io
- baseUrl: /humanoid-robotics-textbook/

### chat-widget.js
- API_BASE_URL: ${BACKEND_URL}

## Update Deployment

Whenever you make changes:

\`\`\`bash
npm run build
npm run deploy
\`\`\`

Changes take 1-2 minutes to appear on GitHub Pages.
EOF

print_success "Created DEPLOY_README.md"

echo ""
print_info "Read DEPLOY_README.md for deployment instructions!"
