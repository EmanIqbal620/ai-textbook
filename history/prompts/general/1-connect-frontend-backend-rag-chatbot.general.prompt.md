---
id: 1
title: "Connect Frontend with Backend RAG Chatbot"
stage: "general"
date: "2025-12-21"
surface: "agent"
model: "claude-sonnet-4-5-20250929"
feature: "none"
branch: "main"
user: "user"
command: "/sp.specify"
labels: ["frontend", "backend", "integration", "chatbot", "rag"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - "humanoid-robotics-textbook/src/components/ChatInterface/ChatInterface.tsx"
  - "humanoid-robotics-textbook/src/components/ChatInterface/ChatInterface.module.css"
  - "humanoid-robotics-textbook/src/components/ChatInterface/index.ts"
  - "humanoid-robotics-textbook/src/pages/index.tsx"
  - "humanoid-robotics-textbook/src/pages/index.module.css"
tests: []
---

# Connect Frontend with Backend RAG Chatbot

## Summary

This PHR documents the implementation of a connection between the frontend (Docusaurus site) and the FastAPI backend RAG chatbot. The implementation includes a new chat interface component that allows users to submit queries to the backend and display responses.

## Implementation Details

### Frontend Changes
- Created a new ChatInterface component with:
  - Message display area with user/bot differentiation
  - Input field with support for multi-line text
  - Loading indicators and error handling
  - Selected text integration for context-aware queries
  - Responsive design for different screen sizes

### Backend Integration
- Implemented API connection using fetch to `/api/v1/chat` endpoint
- Added support for environment variable configuration (REACT_APP_BACKEND_URL)
- Created proper request/response handling with error management
- Included support for selected text context in queries

### UI/UX Features
- Professional styling with CSS modules
- Smooth scrolling to latest messages
- Typing indicators during backend processing
- Timestamps for messages
- Error handling with user-friendly messages

### Testing
- Successfully tested local backend connection
- Verified API communication works correctly
- Confirmed proper error handling when backend is unavailable

## Technical Implementation

The chat interface component handles:
- User input validation and submission
- API communication with proper headers
- Response parsing and display
- Error handling with fallback messages
- Integration with page text selection for context

## Outcome

The frontend now successfully connects to the backend RAG chatbot, allowing users to ask questions about the humanoid robotics textbook and receive AI-generated responses based on the textbook content. The implementation follows best practices for React development and API integration.

## Files Modified

- `humanoid-robotics-textbook/src/components/ChatInterface/ChatInterface.tsx` - Main chat component
- `humanoid-robotics-textbook/src/components/ChatInterface/ChatInterface.module.css` - Component styling
- `humanoid-robotics-textbook/src/components/ChatInterface/index.ts` - Component export
- `humanoid-robotics-textbook/src/pages/index.tsx` - Homepage integration
- `humanoid-robotics-textbook/src/pages/index.module.css` - Homepage styling

## Evaluation

The implementation successfully meets all requirements:
- ✅ Queries reach backend successfully
- ✅ Backend responses displayed clearly under input
- ✅ Error handling with "Error, try again" message
- ✅ Works with local backend (tested)
- ✅ Ready for deployed backend when available