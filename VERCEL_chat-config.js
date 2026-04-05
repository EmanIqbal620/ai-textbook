/**
 * Chat Widget Configuration for Vercel Full-Stack Deployment
 * 
 * This file is for Vercel deployment ONLY
 * Your original chat-config.js remains unchanged
 * 
 * INSTRUCTIONS:
 * 1. Copy this file to: humanoid-robotics-textbook/static/js/chat-config.js
 * 2. Update the API_BASE_URL if needed (should work automatically)
 * 3. Build and deploy to Vercel
 */

// Backend API URL - Points to Vercel serverless functions
// For Vercel deployment, the API is at /api/chat on the same domain
const API_BASE_URL = window.location.origin;

// Export for use in chat-widget.js
if (typeof window !== 'undefined') {
  window.CHAT_WIDGET_CONFIG = {
    apiUrl: API_BASE_URL,
    title: 'AI Robotics Tutor',
    subtitle: 'Ask me anything about robotics!',
    theme: 'dark'
  };
}
