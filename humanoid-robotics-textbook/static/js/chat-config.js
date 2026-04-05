/**
 * Chat Widget Configuration
 * 
 * Update this file BEFORE building for deployment!
 */

// Backend API URL
// For local development: 'http://localhost:8000'
// For GitHub Pages + Hugging Face: 'https://YOUR_USERNAME-humanoid-robotics-backend.hf.space'

const API_BASE_URL = 'http://localhost:8000';

// Export for use in chat-widget.js
if (typeof window !== 'undefined') {
  window.CHAT_WIDGET_CONFIG = {
    apiUrl: API_BASE_URL,
    title: 'AI Robotics Tutor',
    subtitle: 'Ask me anything about robotics!',
    theme: 'dark'
  };
}
